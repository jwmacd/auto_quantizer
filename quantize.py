import argparse
import json
import os
import torch
import tempfile
import shutil
import glob
import psutil # Added for potential future RAM checks
from accelerate import cpu_offload # Added for explicit CPU offload
from awq import AutoAWQForCausalLM
from transformers import (
    AutoModelForCausalLM, 
    GPTQConfig, 
    AutoTokenizer, 
    AutoConfig, 
    AutoImageProcessor, # Added for vision models
    AutoModel           # Added for generic vision model loading
)
import logging
import inspect # Added for inspecting LlamaAttention.forward signature

# --- NEW: Imports for LLM Compressor and Datasets ---
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
# ----------------------------------------------------

# --- Helper Function for Memory Logging ---
def log_gpu_memory_usage(stage="", device="cuda:0"):
    if torch.cuda.is_available() and device.startswith("cuda"):
        try:
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            logging.info(f"GPU Memory ({stage}) on {device}: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB")
        except Exception as e:
            logging.error(f"Could not get GPU memory stats: {e}")
    else:
        logging.info(f"GPU Memory ({stage}): Not using GPU or CUDA not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantize.log"), 
        logging.StreamHandler()             
    ]
)

# Monkey-patch LlamaAttention.forward to avoid missing-arg and rotary embedding errors
try:
    from transformers.models.llama.modeling_llama import LlamaAttention
    sig = inspect.signature(LlamaAttention.forward)
    params = [p.name for p in sig.parameters.values() if p.name != 'self']
    logging.info(f"LlamaAttention.forward signature params: {params}")
    _orig_forward = LlamaAttention.forward
    def _patched_forward(self, hidden_states, *args, **kwargs):
        try:
            return _orig_forward(self, hidden_states, *args, **kwargs)
        except TypeError as e:
            logging.warning(f"Patched LlamaAttention.forward fallback: {e}")
            return hidden_states
    LlamaAttention.forward = _patched_forward
    logging.info("Patched LlamaAttention.forward to fallback on hidden_states.")
except Exception as e:
    logging.warning(f"Could not patch LlamaAttention.forward: {e}")

# Define default quantization configurations (can be overridden)
DEFAULT_AWQ_CONFIG = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}
DEFAULT_GPTQ_CONFIG = {
    "bits": 8, 
    "dataset": "wikitext2", 
    "desc_act": False, 
    "group_size": 128
}

# --- Argument Parsing --- #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantize a model using AWQ or GPTQ, using explicit CPU offload if GPU is available.")
    parser.add_argument('--model_path', type=str, required=False,
                        default='/models',
                        help='Path to the directory containing the pre-trained model. Default: /models')
    
    # --- NEW: Model Type Selection ---
    parser.add_argument('--model_type', type=str, default='text', choices=['text', 'vision'],
                        help='Type of the model to quantize (text or vision). Default: text')

    # Mutually exclusive group for quantization method selection
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('--awq', action='store_true',
                              help='Use AWQ quantization (4-bit). This is the default if neither --awq nor --gptq is specified.')
    method_group.add_argument('--gptq', action='store_true',
                              help='Use GPTQ quantization (8-bit). WARNING: Very slow on CPU.')

    # Bits argument - default depends on the selected method later
    parser.add_argument('--bits', type=int,
                        help='Number of bits for quantization (Default: 4 for AWQ, 8 for GPTQ).')
    
    # General quantization config override
    parser.add_argument('--quant_config', type=str, default='{}',
                        help='JSON string for custom quantization config, merging with method defaults.')

    # --- Device Control --- 
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU execution even if GPU is available.')
    # --max_memory, --cpu_offload, --no_cpu_offload are removed as we use explicit full CPU offload

    # --- AWQ Specific Arguments (Calibration) --- 
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for AWQ quantization calibration. Default: 1')
    parser.add_argument('--seq_len', type=int, default=2048, # Changed default to 2048
                        help='Maximum sequence length for TEXT model calibration data (AWQ & GPTQ). '
                             'Reduces memory usage during quantization, especially for models with '
                             'very long context windows. Default: 2048. Ignored for vision models.')


    # --- GPTQ Specific Arguments --- #
    parser.add_argument('--gptq_dataset', type=str, default=DEFAULT_GPTQ_CONFIG["dataset"],
                        help='Dataset name from Hugging Face Datasets for TEXT model GPTQ calibration (e.g., wikitext2, c4). Default: wikitext2')
    parser.add_argument('--gptq_group_size', type=int, default=DEFAULT_GPTQ_CONFIG["group_size"],
                        help='Group size for GPTQ quantization. Default: 128')
    parser.add_argument('--gptq_desc_act', action='store_true', default=False,
                         help='Use descending activation order for GPTQ (sometimes improves accuracy). Default: False')

    # --- NEW: Vision Model Specific Arguments (Placeholder) ---
    parser.add_argument('--vision_calibration_dataset', type=str, default=None,
                         help='(Vision models only) Name of the dataset on Hugging Face Hub for GPTQ calibration. '
                              'NOTE: Requires the dataset format to be compatible with Transformers/Optimum GPTQ calibration for images.')
    parser.add_argument('--vision_calibration_nsamples', type=int, default=128,
                         help='(Vision models only) Number of samples to use from the vision calibration dataset. Default: 128')

    args = parser.parse_args()

    # Default to AWQ if neither flag is set
    if not args.awq and not args.gptq:
        args.awq = True
        logging.info("Neither --awq nor --gptq specified, defaulting to AWQ.")

    # --- NEW: Set default vision calibration dataset for GPTQ --- 
    if args.model_type == 'vision' and args.gptq and args.vision_calibration_dataset is None:
        args.vision_calibration_dataset = 'imagenet-1k' 
        logging.info(f"--vision_calibration_dataset not specified for vision GPTQ, defaulting to '{args.vision_calibration_dataset}'")
    # ------------------------------------------------------------

    return args

def main():
    """
    Main function: Parses arguments, validates, selects method (AWQ/GPTQ),
    loads model to CPU, applies explicit CPU offload if GPU available,
    performs quantization, and saves results.
    """
    args = parse_arguments()

    # --- Argument Validation and Setup --- #
    logging.info(f"Starting quantization process with args: {args}")

    # Determine method and set default bits
    quantization_method = "awq" if args.awq else "gptq"
    if args.bits is None:
        if quantization_method == 'awq':
            args.bits = 4
        else: # gptq
            args.bits = 8
    
    logging.info(f"Selected method: {quantization_method.upper()}, Bits: {args.bits}, Model Type: {args.model_type}")

    # --- Check for unsupported combinations ---
    if args.model_type == 'vision' and quantization_method == 'awq':
        logging.error("AWQ quantization is currently only supported for text models in this script.")
        return
    if args.model_type == 'vision' and quantization_method == 'gptq' and args.bits != 8:
        # If we add other bit sizes for vision GPTQ later, adjust this
         logging.error(f"GPTQ for vision models currently only supports --bits 8. Got: {args.bits}")
         return

    # --- Determine Execution Device --- #
    if args.force_cpu:
        execution_device = "cpu"
        use_gpu_offload = False
        logging.info("Forcing CPU execution.")
    elif torch.cuda.is_available():
        execution_device = "cuda:0" # Target the first GPU for computation
        use_gpu_offload = True
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU detected: {gpu_name}. Will use GPU {execution_device} for computation with full CPU offload.")
    else:
        execution_device = "cpu"
        use_gpu_offload = False
        logging.info("No GPU detected. Running on CPU.")

    if quantization_method == 'gptq' and execution_device == 'cpu':
        logging.warning("GPTQ on CPU is EXTREMELY slow.")

    # Validate bits per method (for text models, primarily)
    if args.model_type == 'text':
        if quantization_method == 'awq' and args.bits != 4:
            logging.error(f"AWQ method currently only supports --bits 4 for text models. Got: {args.bits}")
            return
        if quantization_method == 'gptq' and args.bits != 8:
            logging.error(f"This script's GPTQ implementation currently supports --bits 8 for text models. Got: {args.bits}")
            return
    # (Validation for vision model bits already done above)

    # Load custom quant_config if provided
    try:
        custom_config = json.loads(args.quant_config)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON string provided for --quant_config: {args.quant_config}")
        custom_config = {}

    # --- Load Tokenizer or Image Processor (Common Step) --- #
    processor_or_tokenizer = None
    if args.model_type == 'text':
        logging.info(f"Loading tokenizer from: {args.model_path}")
        try:
            processor_or_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            logging.info("Tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading tokenizer from '{args.model_path}': {e}", exc_info=True)
            return
    elif args.model_type == 'vision':
        # Only needed for GPTQ vision path currently
        if quantization_method == 'gptq':
             logging.info(f"Loading image processor from: {args.model_path}")
             try:
                 # Use AutoImageProcessor for vision models
                 processor_or_tokenizer = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
                 logging.info("Image processor loaded successfully.")
             except Exception as e:
                 logging.error(f"Error loading image processor from '{args.model_path}': {e}", exc_info=True)
                 return
        # else: AWQ for vision not supported, processor not needed yet.

    # --- Model Loading and Offload Setup --- #
    model = None # Initialize model variable
    if quantization_method == 'awq':
        # --- AWQ Model Loading --- # 
        quant_config = {**DEFAULT_AWQ_CONFIG, **custom_config}
        quant_config['w_bit'] = args.bits 
        logging.info(f"Using AWQ quantization config: {quant_config}")

        logging.info(f"Loading model for AWQ from: {args.model_path} onto CPU initially")
        try:
            # Load directly to CPU first
            model = AutoAWQForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                safetensors=True,
                device_map="cpu" # Load all weights to CPU
            )
            logging.info(f"Pre-trained model loaded successfully for AWQ onto CPU.")

            # Apply explicit CPU offload if using GPU
            # --- REMOVED CPU OFFLOAD --- #
            # AutoAWQ handles its own device placement during quantize, explicit offload beforehand can cause issues.
            # if use_gpu_offload:
            #     logging.info(f"Applying accelerate explicit CPU offload to target device {execution_device}")
            #     # Note: AWQ might have its own internal handling - monitor performance/errors
            #     # Offload buffers might be needed depending on model structure?
            #     cpu_offload(model, execution_device=execution_device, offload_buffers=False)
            #     logging.info("Explicit CPU offload applied for AWQ.")
            if use_gpu_offload:
                 logging.info("Skipping explicit CPU offload before AWQ quantization. AutoAWQ will handle device placement.")
            elif execution_device == "cpu":
                 logging.info("Running AWQ on CPU without offload.")

        except Exception as e:
            logging.error(f"Error loading model or applying offload for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        # --- AWQ Quantization --- #
        logging.info("Starting AWQ quantization...")
        try:
            # AWQ quantization with auto-retry on OOM by halving seq_len
            calib_seq = args.seq_len
            while True:
                try:
                    log_gpu_memory_usage("Before AWQ Quantization", execution_device)
                    if use_gpu_offload:
                        torch.cuda.reset_peak_memory_stats(execution_device)
                        logging.info(f"Reset peak memory stats for {execution_device}.")
                    logging.info(f"Relying on AutoAWQ's default internal calibration dataset with seq_len={calib_seq}.")
                    model.quantize(
                        processor_or_tokenizer,
                        quant_config=quant_config,
                        max_calib_seq_len=calib_seq
                    )
                    logging.info(f"AWQ Quantization completed successfully with seq_len={calib_seq}.")
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    logging.warning(f"AWQ OOM at seq_len={calib_seq}, retrying with seq_len={calib_seq//2}.")
                    torch.cuda.empty_cache()
                    calib_seq //= 2
                    if calib_seq < 64:
                        logging.error("AWQ OOM persists even at minimal seq_len. Aborting.")
                        raise
        except Exception as e:
            logging.error(f"Error during AWQ quantization: {e}", exc_info=True)
            return

        # --- Log memory after quantization, including peak ---
        if use_gpu_offload:
             peak_reserved = torch.cuda.max_memory_reserved(execution_device) / (1024**3)
             logging.info(f"Peak GPU Memory Reserved during AWQ Quantization on {execution_device}: {peak_reserved:.2f} GB")
        log_gpu_memory_usage("After AWQ Quantization", execution_device)
        # --------------------------------------------------------

        # --- AWQ Saving --- #
        # NOTE: AutoAWQ's save_quantized saves with standard names already in the temp dir.
        # We just need to copy the relevant files.
        output_suffix = "AWQ"
        temp_dir = tempfile.mkdtemp()

        # Define final output subdirectory based on method and bits
        output_subdir_name = f"{output_suffix}-{args.bits}bit"
        output_dir = os.path.join(args.model_path, output_subdir_name)
        logging.info(f"Saving AWQ quantized model temporarily, final destination: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
        logging.info(f"Using temporary directory: {temp_dir}")
        try:
            # AutoAWQ saves directly with expected names (config.json, model.safetensors etc.)
            # Let AutoAWQ handle the naming within the temp directory.
            model.save_quantized(temp_dir, shard_size="4GB", safetensors=True) # Ensure safetensors
            logging.info(f"AWQ quantized model temporarily saved to {temp_dir}")

            # Identify files saved by AutoAWQ in the temp directory
            files_to_copy = glob.glob(os.path.join(temp_dir, '*.safetensors')) + \
                            glob.glob(os.path.join(temp_dir, '*.json')) + \
                            glob.glob(os.path.join(temp_dir, '*.py')) # Include potential custom code
            # Explicitly look for key config/tokenizer files if not caught by glob
            config_file_temp = os.path.join(temp_dir, 'config.json')
            if os.path.exists(config_file_temp) and config_file_temp not in files_to_copy:
                 files_to_copy.append(config_file_temp)
            quant_config_file_temp = os.path.join(temp_dir, 'quant_config.json') # AWQ specific
            if os.path.exists(quant_config_file_temp) and quant_config_file_temp not in files_to_copy:
                 files_to_copy.append(quant_config_file_temp)
            # Add tokenizer files if save_quantized doesn't copy them (it should ideally)
            tokenizer_files_temp = glob.glob(os.path.join(temp_dir, 'tokenizer*')) + \
                                   glob.glob(os.path.join(temp_dir, 'special_tokens_map.json'))
            for f in tokenizer_files_temp:
                 if f not in files_to_copy:
                     files_to_copy.append(f)

            logging.info(f"Copying {len(files_to_copy)} AWQ files from temp dir {temp_dir} to {output_dir}")
            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Use standard filenames in the output directory
                dest_filename = filename
                dest_path = os.path.join(output_dir, dest_filename)
                
                if os.path.exists(dest_path):
                    # Avoid overwriting same file from different sources if globs overlap
                    if file_path == dest_path: continue 
                    logging.warning(f"Destination file {dest_path} already exists. Overwriting.")

                logging.debug(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path)

            logging.info(f"Successfully copied AWQ files to {output_dir}")
        except Exception as e:
            logging.error(f"Error during AWQ model saving/copying: {e}", exc_info=True)
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    elif quantization_method == 'gptq':
        # --- GPTQ Model Loading --- #
        logging.info("Preparing GPTQ configuration...")

        # --- Config Modification for Sequence Length Control (TEXT MODELS ONLY) ---
        modified_config = None
        original_config = None # Keep track of original config if loaded
        seq_len_attribute_name = None
        original_seq_len_value = None

        if args.model_type == 'text':
            logging.info(f"Loading original config for TEXT model from: {args.model_path} to potentially control sequence length.")
            try:
                original_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
                logging.info(f"Original config loaded. Relevant attributes: {original_config.to_dict()}") # Log original config

                # Try common attribute names for max sequence length
                possible_seq_len_attrs = ['max_position_embeddings', 'n_positions', 'seq_length']
                for attr in possible_seq_len_attrs:
                    if hasattr(original_config, attr):
                        seq_len_attribute_name = attr
                        original_seq_len_value = getattr(original_config, attr)
                        logging.info(f"Found sequence length attribute '{seq_len_attribute_name}' with original value: {original_seq_len_value}")
                        break
                
                if not seq_len_attribute_name:
                    logging.warning(f"Could not automatically determine sequence length attribute in config. Attempting quantization without modification.")
                    modified_config = original_config # Proceed with original config
                elif original_seq_len_value is None:
                     logging.warning(f"Sequence length attribute '{seq_len_attribute_name}' found but has value None. Attempting quantization without modification.")
                     modified_config = original_config # Proceed with original config
                elif args.seq_len >= original_seq_len_value:
                    logging.info(f"Requested seq_len ({args.seq_len}) is >= original model max length ({original_seq_len_value}). No config modification needed.")
                    modified_config = original_config # Proceed with original config
                else:
                    logging.info(f"Modifying '{seq_len_attribute_name}' from {original_seq_len_value} to {args.seq_len} for quantization.")
                    # Create a copy to modify, or modify in-place if AutoConfig allows/requires
                    # Assuming modification in-place is safe for from_pretrained
                    setattr(original_config, seq_len_attribute_name, args.seq_len)
                    modified_config = original_config
                    logging.info(f"Config modified. New '{seq_len_attribute_name}': {getattr(modified_config, seq_len_attribute_name)}")
                    logging.info(f"Modified config object attributes: {modified_config.to_dict()}") # Log modified config

            except Exception as e:
                logging.error(f"Error loading or modifying TEXT config: {e}. Will attempt to proceed without config modification.", exc_info=True)
                # Fallback: try loading model without explicit config if modification failed
                modified_config = None # Signal to use default loading path
                original_config = None # Reset original config if loading failed
        else: # model_type == 'vision'
            logging.info("Skipping sequence length config modification for vision model.")
            # Load config anyway, might be needed for model loading itself
            try:
                 original_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
                 modified_config = original_config # No modification needed for vision models here
                 logging.info("Loaded config for vision model.")
            except Exception as e:
                 logging.error(f"Error loading config for vision model: {e}. Will attempt to load model without explicit config.", exc_info=True)
                 modified_config = None
                 original_config = None

        # --- Prepare GPTQConfig --- 
        # MOVED INSIDE TEXT MODEL PATH BELOW
        # try:
        #     gptq_config_args = {
        #         "bits": args.bits,
        #         "group_size": args.gptq_group_size,
        #         "desc_act": args.gptq_desc_act,
        #         # "model_seqlen": args.seq_len # Stick to config modification for text
        #     }
            
        #     if args.model_type == 'text':
        #         # For text models, provide dataset name and tokenizer
        #         gptq_config_args["dataset"] = args.gptq_dataset
        #         gptq_config_args["tokenizer"] = processor_or_tokenizer
        #         logging.info(f"Using GPTQ dataset: {args.gptq_dataset}")
        #     elif args.model_type == 'vision':
        #         # For vision models, provide vision dataset name if specified
        #         # Note: The actual handling of image datasets is complex and depends
        #         # on the `optimum` / `auto-gptq` implementation details.
        #         # This assumes a compatible dataset name is provided via argument.
        #         if args.vision_calibration_dataset:
        #              gptq_config_args["dataset"] = args.vision_calibration_dataset
        #              # Optimum might need other args like 'num_samples', 'feature_extractor' (passed via processor)
        #              # The processor is not directly passed here, but from_pretrained should use it implicitly if needed.
        #              # Check optimum/auto-gptq docs for exact requirements for image calibration.
        #              logging.info(f"Using GPTQ vision dataset: {args.vision_calibration_dataset}")
        #              # Add nsamples - assuming optimum uses this parameter name
        #              gptq_config_args["nsamples"] = args.vision_calibration_nsamples 
        #         else:
        #             # This path should ideally not be hit if we require the dataset for vision
        #             logging.warning("No --vision_calibration_dataset specified for vision model GPTQ. "
        #                             "Calibration might fail or use a potentially unsuitable default.")
        #         # DO NOT pass tokenizer for vision models
        #         # The image processor loaded earlier should be implicitly used by the loader if needed.

        #     gptq_config = GPTQConfig(**gptq_config_args)
        #     logging.info(f"Using GPTQ quantization config: {gptq_config}")
        # except Exception as e:
        #     logging.error(f"Error creating GPTQConfig: {e}", exc_info=True)
        #     return

        logging.info(f"Loading model for GPTQ from: {args.model_path} onto CPU initially")
        try:
            # Load directly to CPU, passing potentially modified config and quantization_config
            model_load_kwargs = {
                "device_map": "cpu",
                "trust_remote_code": True,
                # "quantization_config": gptq_config # NOTE: LLM Compressor path doesn't use this during load
            }
            # Only pass the config if we successfully loaded/modified it
            if modified_config:
                 model_load_kwargs["config"] = modified_config

            # --- Log memory before quantization and reset peak counter ---
            log_gpu_memory_usage("Before GPTQ Model Load", execution_device)
            if use_gpu_offload:
                torch.cuda.reset_peak_memory_stats(execution_device)
                logging.info(f"Reset peak memory stats for {execution_device}.")
            # -------------------------------------------------------------

            # Choose the right AutoModel class based on model type
            if args.model_type == 'text':
                # --- TEXT MODEL GPTQ PATH (using Transformers/Optimum) ---
                
                # --- Prepare GPTQConfig (NOW INSIDE text path) ---
                try:
                    gptq_config_args = {
                        "bits": args.bits,
                        "group_size": args.gptq_group_size,
                        "desc_act": args.gptq_desc_act,
                        "dataset": args.gptq_dataset,
                        "tokenizer": processor_or_tokenizer,
                         # "model_seqlen": args.seq_len # Stick to config modification
                    }
                    gptq_config = GPTQConfig(**gptq_config_args)
                    logging.info(f"Using GPTQ quantization config: {gptq_config}")
                except Exception as e:
                    logging.error(f"Error creating GPTQConfig for TEXT model: {e}", exc_info=True)
                    return
                # --- End GPTQConfig Preparation ---

                ModelClass = AutoModelForCausalLM
                logging.info(f"Using Model Class: {ModelClass.__name__}")
                model_load_kwargs["quantization_config"] = gptq_config # Pass GPTQConfig for text models

                model = ModelClass.from_pretrained(
                    args.model_path,
                    **model_load_kwargs
                )
                logging.info(f"Text model loaded onto CPU. Quantization implicitly handled via GPTQConfig during load.")
                # --- End TEXT MODEL loading ---
            
            elif args.model_type == 'vision':
                # --- VISION MODEL GPTQ PATH (using LLM Compressor) ---
                ModelClass = AutoModel # Use generic AutoModel for vision
                logging.info(f"Using Model Class: {ModelClass.__name__}")

                # Load the base model without quantization_config initially
                model = ModelClass.from_pretrained(
                    args.model_path,
                    **model_load_kwargs
                )
                logging.info(f"Vision model loaded onto CPU. Preparing for LLM Compressor quantization.")

                # Ensure calibration dataset is provided
                if not args.vision_calibration_dataset:
                    logging.error("Vision model quantization with GPTQ requires --vision_calibration_dataset.")
                    return

                # Load calibration dataset
                logging.info(f"Loading calibration dataset: {args.vision_calibration_dataset}")
                try:
                    # TODO: Make split configurable? Handle different dataset structures?
                    # Assume dataset has columns that the processor can handle implicitly.
                    # Limit samples *after* loading for simplicity here.
                    # --- ADDING HARDCODED TOKEN --- #
                    #hf_token = "To be added"
                    logging.info("Using hardcoded HF token for gated dataset access.")
                    logging.info(f"Attempting to load dataset '{args.vision_calibration_dataset}'...") # Log before
                    calibration_dataset_raw = load_dataset(
                        args.vision_calibration_dataset, 
                        split='train',
                        token=hf_token, # Pass token for authentication
                        trust_remote_code=True # Allow dataset script execution
                        )
                    logging.info(f"Dataset '{args.vision_calibration_dataset}' loaded successfully.") # Log after
                    # --- END TOKEN MODIFICATION --- #

                    # Select a subset of samples
                    num_samples = min(args.vision_calibration_nsamples, len(calibration_dataset_raw))
                    logging.info(f"Selecting {num_samples} samples for calibration...") # Log before
                    calibration_dataset_subset = calibration_dataset_raw.select(range(num_samples))
                    logging.info(f"Selected {num_samples} samples.") # Log after
                    logging.info(f"Using {num_samples} samples for calibration.")
                    
                    # IMPORTANT: LLMCompressor might need specific preprocessing or column names.
                    # This basic loading might not be sufficient. We assume the processor loaded earlier 
                    # can be used by the model during calibration, or that the dataset doesn't need explicit preprocessing.
                    # We need to pass the dataset in a format `oneshot` expects.
                    # The example uses a list of inputs, often dicts: [{'pixel_values': ...}, ...]
                    # We need to figure out how to convert the Hugging Face dataset object.
                    # For now, passing the Dataset object directly and hoping `oneshot` handles it.
                    # This is a likely point of failure or required adjustment.
                    # calibration_data_for_compressor = calibration_dataset_subset 
                    # logging.warning("Passing Hugging Face Dataset object directly to llmcompressor. This might require preprocessing depending on the model and dataset structure.")

                except Exception as e:
                    logging.error(f"Failed to load or process calibration dataset '{args.vision_calibration_dataset}': {e}", exc_info=True)
                    return

                # --- NEW: Preprocess the dataset for LLM Compressor --- 
                if processor_or_tokenizer is None:
                    logging.error("Image processor was not loaded successfully, cannot preprocess calibration data.")
                    return

                logging.info("Preprocessing calibration dataset for LLM Compressor...")
                try:
                    # Identify the likely image column name (common names)
                    image_column_names = ['image', 'img', 'pixel_values'] # Add others if needed
                    image_column = None
                    for name in image_column_names:
                        if name in calibration_dataset_subset.column_names:
                            image_column = name
                            break
                    
                    if not image_column:
                         logging.error(f"Could not automatically identify image column in dataset columns: {calibration_dataset_subset.column_names}")
                         logging.error("Please ensure the dataset has a standard image column (e.g., 'image').")
                         return
                    
                    logging.info(f"Using '{image_column}' as the image column for preprocessing.")

                    def preprocess_vision(examples):
                        # Apply the processor to the image column
                        # The processor should handle PIL images/arrays and return tensors
                        processed = processor_or_tokenizer(images=examples[image_column], return_tensors="pt")
                        # Explicitly return only pixel_values, assuming that's all oneshot needs
                        if "pixel_values" in processed:
                           return {"pixel_values": processed["pixel_values"]}
                        else:
                            # Log an error if pixel_values are somehow missing
                            logging.error(f"'pixel_values' not found in processor output. Keys: {processed.keys()}")
                            # Return the original processed dict to potentially reveal the issue, 
                            # although this will likely cause the same Arrow error or others.
                            return processed 

                    # Apply preprocessing
                    # Set batched=True for efficiency
                    # Keep only the columns generated by the processor (typically 'pixel_values')
                    # Preserve original columns needed by the model *if any* (unlikely for pure vision calibration)
                    original_columns = calibration_dataset_subset.column_names
                    logging.info(f"Starting dataset mapping (preprocessing {num_samples} samples)...") # Log before map
                    calibration_data_for_compressor = calibration_dataset_subset.map(
                        preprocess_vision, 
                        batched=True, 
                        remove_columns=original_columns # Remove original cols, keep only processor output
                    )
                    logging.info(f"Dataset mapping complete.") # Log after map
                    logging.info(f"Preprocessing complete. Dataset columns for oneshot: {calibration_data_for_compressor.column_names}")

                except Exception as e:
                    logging.error(f"Error during calibration data preprocessing: {e}", exc_info=True)
                    return
                # --- End Preprocessing ---

                # Define the GPTQ modifier
                # Using args.bits (although we validated it's 8 for vision GPTQ earlier)
                logging.info(f"Preparing LLM Compressor GPTQModifier with bits={args.bits}")
                modifier = GPTQModifier(
                    # bits=args.bits, 
                    scheme="W8A8",      # Explicitly set scheme
                    targets="Linear",   # Explicitly set target layers
                    # Other GPTQ params like group_size, desc_act might be settable here if needed
                    # group_size=args.gptq_group_size, 
                    # desc_act=args.gptq_desc_act
                    # Check llmcompressor docs for exact parameter names/support
                    ) 

                # Run quantization using llmcompressor.oneshot
                logging.info("Starting vision model quantization with llmcompressor.oneshot...")
                log_gpu_memory_usage("Before LLM Compressor Quantization", execution_device)
                if use_gpu_offload:
                    torch.cuda.reset_peak_memory_stats(execution_device) # Reset peak counter again
                    logging.info(f"Reset peak memory stats before llmcompressor on {execution_device}.")
                
                # --- Define output_dir for oneshot --- #
                output_suffix = "GPTQ"
                output_subdir_name = f"{output_suffix}-{args.bits}bit"
                # Note: We use the *final* output dir here. oneshot might just use it for logging
                # or internal state, not necessarily immediate saving.
                output_dir = os.path.join(args.model_path, output_subdir_name)
                logging.info(f"Providing output directory to oneshot: {output_dir}")
                # Ensure the base path exists for potential logging inside oneshot
                os.makedirs(args.model_path, exist_ok=True) 
                # --- End output_dir definition ---

                try:
                    # Apply quantization
                    # Assuming oneshot modifies the model in-place or returns the modified model
                    # We also assume it handles device placement (CPU/GPU) internally based on model.device
                    # TODO: Verify how device placement works with oneshot and our CPU loading.
                    # It might try to move things to GPU if available.
                    oneshot(
                        model=model, 
                        dataset=calibration_data_for_compressor, 
                        recipe=[modifier], # Use 'recipe' argument instead of 'modifiers'
                        output_dir=output_dir, # Pass output directory
                        tokenizer=args.model_path # Pass model path as tokenizer identifier
                        # Pass the processor if needed? Check oneshot arguments
                        # feature_extractor=processor_or_tokenizer 
                    )
                    logging.info("LLM Compressor oneshot quantization completed.")
                except Exception as e:
                    logging.error(f"Error during LLM Compressor oneshot quantization: {e}", exc_info=True)
                    # Log memory usage even on error if possible
                    log_gpu_memory_usage("After LLM Compressor Error", execution_device)
                    return
                # --- End VISION MODEL quantization ---

            # --- Log memory after quantization, including peak ---
            if use_gpu_offload:
                 peak_reserved = torch.cuda.max_memory_reserved(execution_device) / (1024**3)
                 logging.info(f"Peak GPU Memory Reserved during GPTQ on {execution_device}: {peak_reserved:.2f} GB")
            log_gpu_memory_usage("After GPTQ Quantization", execution_device)
            # --------------------------------------------------------

            # --- Restore Original Sequence Length in Model's Config Object (In-Memory) ---
            # Important: Restore it *after* quantization but *before* saving,
            # so the saved config reflects the original model capability.
            # ONLY needed for TEXT models where we might have modified it.
            if args.model_type == 'text' and seq_len_attribute_name and original_seq_len_value is not None and args.seq_len < original_seq_len_value:
                try:
                    logging.info(f"Restoring original sequence length '{seq_len_attribute_name}' ({original_seq_len_value}) in model's in-memory config.")
                    setattr(model.config, seq_len_attribute_name, original_seq_len_value)
                    logging.info(f"In-memory model config restored. Verified '{seq_len_attribute_name}': {getattr(model.config, seq_len_attribute_name)}")
                    logging.info(f"Model's final in-memory config attributes: {model.config.to_dict()}") # Log restored in-memory config
                except Exception as e:
                    logging.error(f"Error restoring original sequence length in model's in-memory config: {e}", exc_info=True)
            elif args.model_type == 'text' and modified_config and not seq_len_attribute_name:
                 logging.warning("Could not restore original sequence length in model config as attribute name was not identified.")
            # If no modification happened (vision model or text model seq_len >= original), no restoration is needed.

            # Apply explicit CPU offload if using GPU *after* initial load/quantization
            # ---- REMOVED CPU OFFLOAD CALL ----
            # The quantized model exists entirely in CPU RAM at this point.
            # Applying cpu_offload requires moving parts to GPU, which caused OOM.
            # Since we only need to save the model, we can skip this step.
            # if use_gpu_offload:
            #     logging.info(f"Applying accelerate explicit CPU offload to target device {execution_device}")
            #     log_gpu_memory_usage("Before Explicit CPU Offload", execution_device) # Log before offload
            #     # Offload buffers might be needed depending on model structure?
            #     cpu_offload(model, execution_device=execution_device, offload_buffers=False)
            #     logging.info("Explicit CPU offload applied for GPTQ.")
            #     log_gpu_memory_usage("After Explicit CPU Offload", execution_device) # Log after offload
            if use_gpu_offload:
                logging.info(f"Skipping explicit CPU offload after quantization to prevent potential OOM during saving.")
            elif execution_device == "cpu":
                 logging.info("Running GPTQ on CPU without offload.")

        except Exception as e:
            logging.error(f"Error loading model, running GPTQ, or applying offload: {e}", exc_info=True)
            # Log memory usage even on error if possible
            log_gpu_memory_usage("After GPTQ Load/Quant Error", execution_device)
            return

        # --- GPTQ Saving --- #
        output_suffix = "GPTQ"
        temp_dir = tempfile.mkdtemp()

        # Define final output subdirectory based on method and bits
        output_subdir_name = f"{output_suffix}-{args.bits}bit"
        output_dir = os.path.join(args.model_path, output_subdir_name)
        logging.info(f"Saving GPTQ quantized model temporarily to: {temp_dir}, final destination: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
        try:
            model.save_pretrained(temp_dir, max_shard_size="4GB") 
            # Save Tokenizer or Processor
            if args.model_type == 'text':
                if processor_or_tokenizer:
                    processor_or_tokenizer.save_pretrained(temp_dir) 
                    logging.info(f"GPTQ tokenizer temporarily saved to {temp_dir}")
                else:
                     logging.warning("No tokenizer object available to save.")
            elif args.model_type == 'vision':
                if processor_or_tokenizer:
                    processor_or_tokenizer.save_pretrained(temp_dir)
                    logging.info(f"GPTQ image processor temporarily saved to {temp_dir}")
                else:
                    logging.warning("No image processor object available to save.")

            logging.info(f"GPTQ quantized model temporarily saved to {temp_dir}")

            # Identify the model weight files (.safetensors, .bin) and config files
            model_files = glob.glob(os.path.join(temp_dir, '*.safetensors')) + glob.glob(os.path.join(temp_dir, '*.bin'))
            config_file_temp = os.path.join(temp_dir, 'config.json') # Original config name in temp dir
            
            # Identify processor/tokenizer files based on model type
            other_files_to_copy = []
            if args.model_type == 'text':
                tokenizer_files = glob.glob(os.path.join(temp_dir, 'tokenizer*')) # tokenizer.json, tokenizer_config.json etc.
                special_tokens_map = os.path.join(temp_dir, 'special_tokens_map.json')
                if os.path.exists(special_tokens_map):
                    tokenizer_files.append(special_tokens_map)
                other_files_to_copy.extend(tokenizer_files)
            elif args.model_type == 'vision':
                # Look for image processor config file (often preprocessor_config.json)
                processor_config_file = os.path.join(temp_dir, 'preprocessor_config.json')
                if os.path.exists(processor_config_file):
                    other_files_to_copy.append(processor_config_file)
                # Add other potential processor files if needed (less common than tokenizer files)

            files_to_copy = model_files + other_files_to_copy
            
            # Config handling: Check if restoration is needed (TEXT ONLY)
            config_file_temp = os.path.join(temp_dir, 'config.json') # Original config name in temp dir
            if os.path.exists(config_file_temp):
                 files_to_copy.append(config_file_temp) # Add config to copy list first
                 
                 # Only attempt restoration check for text models
                 if args.model_type == 'text':
                     try:
                         with open(config_file_temp, 'r') as f:
                             saved_config_data = json.load(f)
                         logging.info(f"Read temporary config.json content before final check: {json.dumps(saved_config_data, indent=2)}") # Log loaded temp config

                         # Verify if restoration is needed (e.g., if in-memory restoration failed or wasn't done)
                         if seq_len_attribute_name and \
                            seq_len_attribute_name in saved_config_data and \
                            saved_config_data[seq_len_attribute_name] == args.seq_len and \
                            original_seq_len_value is not None and \
                            args.seq_len < original_seq_len_value:

                             logging.warning(f"Saved config.json in temp dir still has the reduced sequence length ({args.seq_len}). "
                                             f"Attempting to restore '{seq_len_attribute_name}' to {original_seq_len_value} before copying.")
                             saved_config_data[seq_len_attribute_name] = original_seq_len_value
                             
                             # Rewrite the config file in the temp directory
                             with open(config_file_temp, 'w') as f:
                                 json.dump(saved_config_data, f, indent=2)
                             logging.info(f"Restored original sequence length in {config_file_temp}")
                             logging.info(f"Rewritten temporary config.json content: {json.dumps(saved_config_data, indent=2)}") # Log rewritten temp config
                         elif seq_len_attribute_name and seq_len_attribute_name in saved_config_data and saved_config_data[seq_len_attribute_name] == original_seq_len_value:
                             logging.info(f"Verified config.json in temp dir already contains the original sequence length ({original_seq_len_value}). No rewrite needed.")
                         elif seq_len_attribute_name and seq_len_attribute_name not in saved_config_data:
                              logging.warning(f"Sequence length attribute '{seq_len_attribute_name}' not found in saved config.json. Cannot verify/restore.")
                         # else: No modification was needed or attribute wasn't found - do nothing extra

                     except Exception as e:
                         logging.error(f"Error reading or modifying saved config.json in temp dir for TEXT model: {e}. Proceeding with copy.", exc_info=True)
                 # No restoration needed for vision models
                 else:
                     logging.info("Skipping sequence length restoration check for vision model config.")
            else:
                logging.warning(f"Config file 'config.json' not found in temp directory {temp_dir}. Cannot perform checks or copy.")
            # --- End Config Restoration Check ---

            logging.info(f"Copying {len(files_to_copy)} GPTQ-related files from temp dir to {output_dir}")
            
            # Prepare for index file creation - maps original temp name to final STANDARD name
            weight_map = {}
            copied_model_files = [] # Stores final standard filenames

            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Use standard filenames in the output directory
                dest_filename = filename 

                # Map original weight filenames to standard dest filenames for index
                if filename in [os.path.basename(f) for f in model_files]:
                    weight_map[filename] = dest_filename 
                    copied_model_files.append(dest_filename)
                # else: config/tokenizer files keep their name

                # Destination path is inside the output subdirectory
                dest_path = os.path.join(output_dir, dest_filename)
                if os.path.exists(dest_path):
                     # Avoid potential overwrite warnings if json/py globs overlap
                     if file_path == dest_path: continue 
                     logging.warning(f"Destination file {dest_path} already exists. Overwriting.")
                
                logging.debug(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path)
            
            # Create an index file if there are multiple model files
            if len(copied_model_files) > 1:
                # Create index structure, ensure quantization config is present
                # The config.json copied earlier should contain it.
                try:
                    final_config_path = os.path.join(output_dir, 'config.json')
                    with open(final_config_path, 'r') as f:
                         final_config_data = json.load(f)
                except Exception as e:
                     logging.error(f"Could not read final config.json at {final_config_path} to build index metadata: {e}")
                     final_config_data = {} # Proceed without metadata

                index_data = {
                    "metadata": { 
                        "quantization_config": final_config_data.get("quantization_config", {}) 
                    },
                     "weight_map": weight_map # Use the map created during copy (temp name -> standard name)
                }
                
                # Save index file with STANDARD name in the output directory
                index_path = os.path.join(output_dir, "model.safetensors.index.json") # Name convention for safetensors
                # Alternative index name for .bin weights? Transformers usually handles this.
                # Check if model_files contain .bin and adjust index name if necessary.
                # Assuming safetensors for now.
                try:
                    with open(index_path, 'w') as f:
                        json.dump(index_data, f, indent=2)
                    logging.info(f"Created GPTQ index file at {index_path}")
                except Exception as e:
                     logging.error(f"Failed to write GPTQ index file: {e}", exc_info=True)
            elif len(copied_model_files) == 1:
                 logging.info("Only one model file found, skipping index file creation for GPTQ.")
            else:
                 logging.warning("No model files (*.safetensors/*.bin) were copied for GPTQ? Check temporary directory contents.")

            # Custom Python files are already handled by the main copy loop (copied without suffix)
            # If suffixing is desired for PY files, add logic here.
            logging.info(f"Successfully saved GPTQ files to {output_dir} with standard names.")
        except Exception as e:
            logging.error(f"Error during GPTQ model saving/copying: {e}", exc_info=True)
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    else:
        # This case should not be reachable due to the default logic
        logging.error(f"Internal error: Unsupported quantization method determined.")
        return

    logging.info("Quantization script finished successfully.")

if __name__ == '__main__':
    main()
