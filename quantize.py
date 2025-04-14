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
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer, AutoConfig
from datasets import load_dataset
import logging
# Attempt to import exllamav2 - allows script to run even if not built/installed
try:
    from exllamav2 import ExLlamaV2
    from exllamav2.config import ExLlamaV2Config
    from exllamav2.data import ExLlamaV2Dataset
    from exllamav2.quantize import ExLlamaV2Quantizer
    EXLLAMA_V2_AVAILABLE = True
except ImportError:
    logging.warning("exllamav2 library not found. EXL2 quantization will not be available.")
    EXLLAMA_V2_AVAILABLE = False

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
    
    # Mutually exclusive group for quantization method selection
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('--awq', action='store_true',
                              help='Use AWQ quantization (4-bit). This is the default if neither --awq nor --gptq is specified.')
    method_group.add_argument('--gptq', action='store_true',
                              help='Use GPTQ quantization (8-bit). WARNING: Very slow on CPU.')
    method_group.add_argument('--exl2', action='store_true',
                              help='Use EXL2 quantization (requires exllamav2 library). Specify target bits per weight with --exl2_bpw.')

    # Bits argument - default depends on the selected method later
    parser.add_argument('--bits', type=int,
                        help='Number of bits for quantization (AWQ/GPTQ only). Default: 4 for AWQ, 8 for GPTQ.')
    
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
    parser.add_argument('--seq_len', type=int, default=8192, # Reduced default for safety
                        help='Maximum sequence length to use for calibration data (AWQ & GPTQ). '
                             'Reduces memory usage during quantization, especially for models with '
                             'very long context windows. Default: 8192')


    # --- GPTQ Specific Arguments --- #
    parser.add_argument('--gptq_dataset', type=str, default=DEFAULT_GPTQ_CONFIG["dataset"],
                        help='Dataset name from Hugging Face Datasets for GPTQ calibration (e.g., wikitext2, c4). Default: wikitext2')
    parser.add_argument('--gptq_group_size', type=int, default=DEFAULT_GPTQ_CONFIG["group_size"],
                        help='Group size for GPTQ quantization. Default: 128')
    parser.add_argument('--gptq_desc_act', action='store_true', default=False,
                         help='Use descending activation order for GPTQ (sometimes improves accuracy). Default: False')

    # --- EXL2 Specific Arguments --- #
    exl2_group = parser.add_argument_group('EXL2 Specific Arguments')
    exl2_group.add_argument('--exl2_bpw', type=float, default=4.0,
                           help='Target bits per weight for EXL2 quantization. Default: 4.0')
    exl2_group.add_argument('--exl2_cal_dataset', type=str, default=None,
                           help='Path to calibration dataset file for EXL2 (e.g., calibration_data.parquet). Required for EXL2.')
    exl2_group.add_argument('--exl2_cal_rows', type=int, default=100,
                            help='Number of rows from dataset to use for calibration. Default: 100')
    exl2_group.add_argument('--exl2_measurement', type=str, default=None,
                           help='Path to load/save measurement JSON file for EXL2 calibration. If exists, loads; otherwise, saves after measurement.')
    exl2_group.add_argument('--exl2_head_bits', type=int, default=6,
                           help='Number of bits for head layers in EXL2. Default: 6')
    # Add other EXL2 options as needed, e.g., shard size, overlap, rope scale/alpha?

    args = parser.parse_args()

    # Default to AWQ if neither flag is set
    if not args.awq and not args.gptq and not args.exl2:
        args.awq = True
        logging.info("Neither --awq, --gptq, nor --exl2 specified, defaulting to AWQ.")

    # Validate EXL2 prerequisites if selected
    if args.exl2:
        if not EXLLAMA_V2_AVAILABLE:
            logging.error("EXL2 quantization selected, but the exllamav2 library is not installed or importable.")
            # Optionally exit here or let it fail later
            parser.error("exllamav2 library required for --exl2")
        if args.exl2_cal_dataset is None:
            parser.error("--exl2_cal_dataset is required when using --exl2")
        if not os.path.exists(args.exl2_cal_dataset):
             parser.error(f"EXL2 calibration dataset not found: {args.exl2_cal_dataset}")

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
    if args.awq:
        quantization_method = "awq"
    elif args.gptq:
        quantization_method = "gptq"
    elif args.exl2:
        quantization_method = "exl2"
    else: # Should be unreachable due to default logic
        quantization_method = "awq"

    # Set default bits only for AWQ/GPTQ
    if args.bits is None and quantization_method in ["awq", "gptq"]:
        if quantization_method == 'awq':
            args.bits = 4
        else: # gptq
            args.bits = 8
    
    # Log method and appropriate bits
    if quantization_method == "exl2":
        logging.info(f"Selected method: {quantization_method.upper()}, Target BPW: {args.exl2_bpw}")
    else:
        logging.info(f"Selected method: {quantization_method.upper()}, Bits: {args.bits}")

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
    if quantization_method == 'exl2' and execution_device == 'cpu':
        logging.error("EXL2 quantization requires a GPU.")
        return

    # Validate bits per method
    if quantization_method == 'awq' and args.bits != 4:
        logging.error(f"AWQ method currently only supports --bits 4. Got: {args.bits}")
        return
    if quantization_method == 'gptq' and args.bits != 8:
        logging.error(f"This script's GPTQ implementation currently supports --bits 8. Got: {args.bits}")
        return

    # Load custom quant_config if provided
    custom_config = {}
    if quantization_method != 'exl2': # EXL2 uses its own config params
        try:
            custom_config = json.loads(args.quant_config)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON string provided for --quant_config: {args.quant_config}")
            # Reset to empty dict if invalid, default logic will handle method defaults
            custom_config = {}

    # --- Load Tokenizer (Common Step) --- #
    logging.info(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizer from '{args.model_path}': {e}", exc_info=True)
        return

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
            if use_gpu_offload:
                logging.info(f"Applying accelerate explicit CPU offload to target device {execution_device}")
                # Note: AWQ might have its own internal handling - monitor performance/errors
                # Offload buffers might be needed depending on model structure?
                cpu_offload(model, execution_device=execution_device, offload_buffers=False)
                logging.info("Explicit CPU offload applied for AWQ.")
            elif execution_device == "cpu":
                 logging.info("Running AWQ on CPU without offload.")

        except Exception as e:
            logging.error(f"Error loading model or applying offload for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        # --- AWQ Quantization --- #
        logging.info("Starting AWQ quantization...")
        try:
            # Pass tokenizer and merged quant_config
            # Relies on offload hook to move layers to execution_device
            model.quantize(
                tokenizer,
                quant_config=quant_config
                # batch_size=args.batch_size, # Still relying on defaults / need research
                # max_seq_len=args.seq_len  # Still relying on defaults / need research
            )
            logging.info("AWQ Quantization completed successfully.")
        except Exception as e:
            logging.error(f"Error during AWQ quantization: {e}", exc_info=True)
            return

        # --- AWQ Saving --- #
        output_suffix = "AWQ"
        temp_dir = tempfile.mkdtemp()
        # Define final output subdirectory
        base_model_name = os.path.basename(args.model_path.rstrip('/'))
        output_subdir_name = f"{base_model_name}-{output_suffix}"
        output_dir = os.path.join(args.model_path, output_subdir_name)
        logging.info(f"Saving AWQ quantized model temporarily, final destination: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
        logging.info(f"Using temporary directory: {temp_dir}")
        try:
            # Use a potentially smaller shard size if memory is constrained? Default 4GB is usually fine.
            model.save_quantized(temp_dir, shard_size="4GB")
            logging.info(f"AWQ quantized model temporarily saved to {temp_dir}")

            # Files to copy: *.safetensors, quant_config.json, AND model.safetensors.index.json
            files_to_copy = glob.glob(os.path.join(temp_dir, '*.safetensors'))
            quant_config_file_temp = os.path.join(temp_dir, 'quant_config.json')
            if os.path.exists(quant_config_file_temp):
                 files_to_copy.append(quant_config_file_temp)
            else: 
                 logging.warning(f"Expected quant_config.json not found in temp dir: {quant_config_file_temp}")

            index_file_temp = os.path.join(temp_dir, 'model.safetensors.index.json')
            if os.path.exists(index_file_temp):
                files_to_copy.append(index_file_temp)
            else:
                # This might be expected if the model isn't sharded
                logging.info(f"Index file not found in temp dir (model might not be sharded): {index_file_temp}")

            logging.info(f"Copying {len(files_to_copy)} AWQ files from temp dir to {output_dir}")
            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Add -AWQ suffix to safetensors and config files
                dest_filename = filename # Default to original name
                if filename.endswith('.safetensors') and not filename.startswith('model'):
                    # Handle potential non-model safetensors if any? Unlikely.
                    dest_filename = f"{filename[:-len('.safetensors')]}-{output_suffix}.safetensors"
                elif filename.startswith('model') and filename.endswith('.safetensors'):
                    dest_filename = filename.replace(".safetensors", f"-{output_suffix}.safetensors")
                elif filename == 'quant_config.json':
                    dest_filename = f"quant_config-{output_suffix}.json"
                elif filename == 'model.safetensors.index.json':
                    dest_filename = f"model-{output_suffix}.safetensors.index.json"
                # else: Keep original filename if unexpected

                # Destination path is inside the output subdirectory
                dest_path = os.path.join(output_dir, dest_filename)
                if os.path.exists(dest_path):
                    logging.warning(f"Destination file {dest_path} already exists. Overwriting.")

                logging.debug(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path) 
            
            # Handle any custom Python files
            custom_code_files = glob.glob(os.path.join(temp_dir, '*.py'))
            if custom_code_files:
                logging.info(f"Found {len(custom_code_files)} custom Python files to copy")
                for py_file in custom_code_files:
                    py_filename = os.path.basename(py_file)
                    # Add suffix to python file and save in subdirectory
                    dest_filename = f"{py_filename[:-3]}-{output_suffix}.py"
                    dest_path = os.path.join(output_dir, dest_filename)
                    logging.info(f"Copying custom code file {py_filename} to {dest_path}")
                    shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully copied AWQ files to {output_dir}")
        except Exception as e:
            logging.error(f"Error during AWQ model saving/copying: {e}", exc_info=True)
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    elif quantization_method == 'gptq':
        # --- GPTQ Model Loading --- #
        logging.info("Preparing GPTQ configuration...")

        # --- Config Modification for Sequence Length Control ---
        logging.info(f"Loading original config from: {args.model_path} to control sequence length.")
        original_config = None
        original_seq_len_value = None
        seq_len_attribute_name = None
        modified_config = None
        try:
            original_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            logging.info(f"Original config loaded. Relevant attributes: {original_config.to_dict()}") # Log original config

            # Try common attribute names for max sequence length
            possible_seq_len_attrs = ['max_position_embeddings', 'n_positions', 'seq_length']
            for attr in possible_seq_len_attrs:
                if hasattr(original_config, attr):
                    seq_len_attribute_name = attr
                    original_seq_len_value = getattr(original_config, seq_len_attribute_name)
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
            logging.error(f"Error loading or modifying config: {e}. Will attempt to proceed without config modification.", exc_info=True)
            # Fallback: try loading model without explicit config if modification failed
            modified_config = None # Signal to use default loading path

        # --- Prepare GPTQConfig ---
        try:
            # Note: We pass the tokenizer here, but the dataset loading/processing happens
            # internally in transformers/optimum based on the dataset string name.
            # The modified config (with shorter seq_len) should influence this internal processing.
            gptq_config = GPTQConfig(
                bits=args.bits,
                dataset=args.gptq_dataset, # Dataset name string
                tokenizer=tokenizer,
                group_size=args.gptq_group_size,
                desc_act=args.gptq_desc_act,
                # model_seqlen=args.seq_len # This might be an alternative, but less standard? Stick to config modification.
            )
            logging.info(f"Using GPTQ quantization config: {gptq_config}")
        except Exception as e:
            logging.error(f"Error creating GPTQConfig: {e}", exc_info=True)
            return

        logging.info(f"Loading model for GPTQ from: {args.model_path} onto CPU initially")
        try:
            # Load directly to CPU, passing potentially modified config and quantization_config
            model_load_kwargs = {
                "device_map": "cpu",
                "trust_remote_code": True,
                "quantization_config": gptq_config
            }
            # Only pass the config if we successfully loaded/modified it
            if modified_config:
                 model_load_kwargs["config"] = modified_config

            # --- Log memory before quantization and reset peak counter ---
            log_gpu_memory_usage("Before GPTQ Model Load/Quantization", execution_device)
            if use_gpu_offload:
                torch.cuda.reset_peak_memory_stats(execution_device)
                logging.info(f"Reset peak memory stats for {execution_device}.")
            # -------------------------------------------------------------

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                **model_load_kwargs
            )
            logging.info(f"Model loaded onto CPU. Quantization implicitly handled via GPTQConfig during load.")

            # --- Log memory after quantization, including peak ---
            if use_gpu_offload:
                 peak_reserved = torch.cuda.max_memory_reserved(execution_device) / (1024**3)
                 logging.info(f"Peak GPU Memory Reserved during GPTQ Load/Quantization on {execution_device}: {peak_reserved:.2f} GB")
            log_gpu_memory_usage("After GPTQ Model Load/Quantization", execution_device)
            # --------------------------------------------------------

            # --- Restore Original Sequence Length in Model's Config Object (In-Memory) ---
            # Important: Restore it *after* quantization but *before* saving,
            # so the saved config reflects the original model capability.
            if seq_len_attribute_name and original_seq_len_value is not None and args.seq_len < original_seq_len_value:
                try:
                    logging.info(f"Restoring original sequence length '{seq_len_attribute_name}' ({original_seq_len_value}) in model's in-memory config.")
                    setattr(model.config, seq_len_attribute_name, original_seq_len_value)
                    logging.info(f"In-memory model config restored. Verified '{seq_len_attribute_name}': {getattr(model.config, seq_len_attribute_name)}")
                    logging.info(f"Model's final in-memory config attributes: {model.config.to_dict()}") # Log restored in-memory config
                except Exception as e:
                    logging.error(f"Error restoring original sequence length in model's in-memory config: {e}", exc_info=True)
            elif modified_config and not seq_len_attribute_name:
                 logging.warning("Could not restore original sequence length in model config as attribute name was not identified.")
            # If no modification happened, no restoration is needed.


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
            return

        # --- GPTQ Saving --- #
        output_suffix = "GPTQ"
        temp_dir = tempfile.mkdtemp()
        # Define final output subdirectory
        base_model_name = os.path.basename(args.model_path.rstrip('/'))
        output_subdir_name = f"{base_model_name}-{output_suffix}"
        output_dir = os.path.join(args.model_path, output_subdir_name)
        logging.info(f"Saving GPTQ quantized model temporarily to: {temp_dir}, final destination: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
        try:
            model.save_pretrained(temp_dir, max_shard_size="4GB") 
            tokenizer.save_pretrained(temp_dir) 
            logging.info(f"GPTQ quantized model and tokenizer temporarily saved to {temp_dir}")

            # Identify the model weight files (.safetensors, .bin) and config files
            model_files = glob.glob(os.path.join(temp_dir, '*.safetensors')) + glob.glob(os.path.join(temp_dir, '*.bin'))
            config_file_temp = os.path.join(temp_dir, 'config.json') # Original config name in temp dir
            tokenizer_files = glob.glob(os.path.join(temp_dir, 'tokenizer*')) # tokenizer.json, tokenizer_config.json etc.
            special_tokens_map = os.path.join(temp_dir, 'special_tokens_map.json')
            if os.path.exists(special_tokens_map):
                tokenizer_files.append(special_tokens_map)

            files_to_copy = model_files + tokenizer_files
            # Config handling is now different: we might need to restore original seq len
            config_file_temp = os.path.join(temp_dir, 'config.json') # Original config name in temp dir

            # --- Config Restoration Before Copying ---
            # We already restored it in the model's in-memory config before model.save_pretrained
            # So the config.json saved in temp_dir should *already* have the original length.
            # Double-check just in case.
            if os.path.exists(config_file_temp):
                 files_to_copy.append(config_file_temp)
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
                     logging.error(f"Error reading or modifying saved config.json in temp dir: {e}. Proceeding with copy.", exc_info=True)
            else:
                logging.warning(f"Config file 'config.json' not found in temp directory {temp_dir}. Cannot restore sequence length.")
            # --- End Config Restoration ---

            logging.info(f"Copying {len(files_to_copy)} GPTQ-related files from temp dir to {output_dir}")
            
            # Prepare for index file creation - maps original temp name to final suffixed name
            weight_map = {}
            copied_model_files = [] # Stores final suffixed filenames

            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                dest_filename = filename # Default name in the final directory

                # Add -GPTQ suffix to model weights and config
                # Determine the final suffixed filename
                is_model_weight = filename in [os.path.basename(f) for f in model_files]
                is_config = filename == 'config.json'
                is_tokenizer = filename in [os.path.basename(f) for f in tokenizer_files]

                if is_model_weight:
                    base, ext = os.path.splitext(filename)
                    dest_filename = f"{base}-{output_suffix}{ext}"
                    weight_map[filename] = dest_filename # Map temp name to final suffixed name
                    copied_model_files.append(dest_filename)
                elif is_config:
                    dest_filename = f'config-{output_suffix}.json'
                elif is_tokenizer:
                     dest_filename = filename # Keep tokenizer files as they are
                else:
                     logging.warning(f"Unexpected file found in GPTQ temp directory: {filename}. Copying as-is.")
                     # Keep original name if unexpected
                     dest_filename = filename 

                # Destination path is inside the output subdirectory
                dest_path = os.path.join(output_dir, dest_filename)
                if os.path.exists(dest_path):
                    logging.warning(f"Destination file {dest_path} already exists. Overwriting.")
                
                logging.debug(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path)
            
            # Create an index file if there are multiple model files
            if len(copied_model_files) > 1:
                # Create index structure
                index_data = {
                    "metadata": {
                         "quantization_config": { # Store GPTQ config used
                            "bits": args.bits,
                            "group_size": args.gptq_group_size,
                            "desc_act": args.gptq_desc_act,
                            "dataset": args.gptq_dataset
                        }
                    },
                     "weight_map": weight_map # Use the map created during copy
                }
                
                # Save index file with suffix in the output directory
                index_path = os.path.join(output_dir, f"model-{output_suffix}.safetensors.index.json")
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

            # Copy any custom code files if they exist
            custom_code_files = glob.glob(os.path.join(temp_dir, '*.py'))
            for py_file in custom_code_files:
                py_filename = os.path.basename(py_file)
                # Add suffix to python file and save in subdirectory
                dest_filename = f"{py_filename[:-3]}-{output_suffix}.py"
                dest_path = os.path.join(output_dir, dest_filename)
                logging.info(f"Copying custom code file {py_filename} to {dest_path}")
                shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully saved GPTQ files to {output_dir} with -{output_suffix} suffix")
        except Exception as e:
            logging.error(f"Error during GPTQ model saving/copying: {e}", exc_info=True)
            # Don't return here, proceed to finally
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    elif quantization_method == 'exl2':
        # --- EXL2 Quantization --- # 
        if not EXLLAMA_V2_AVAILABLE: # Double check, though parser should have caught it
             logging.error("Cannot perform EXL2 quantization, exllamav2 library not available.")
             return

        logging.info(f"Starting EXL2 quantization for model: {args.model_path}")
        logging.info(f"Target BPW: {args.exl2_bpw}, Head Bits: {args.exl2_head_bits}")
        logging.info(f"Calibration Dataset: {args.exl2_cal_dataset}, Rows: {args.exl2_cal_rows}")
        if args.exl2_measurement:
            logging.info(f"Measurement file path: {args.exl2_measurement}")

        # Define output directory for EXL2
        output_suffix = f"EXL2_{args.exl2_bpw:.2f}bpw" # More descriptive suffix
        base_model_name = os.path.basename(args.model_path.rstrip('/'))
        output_subdir_name = f"{base_model_name}-{output_suffix}"
        output_dir = os.path.join(args.model_path, output_subdir_name)
        logging.info(f"EXL2 output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. Create config
            config = ExLlamaV2Config()
            config.model_dir = args.model_path
            config.prepare()
            # Note: Exllama determines max_seq_len from model config, might need override if original is huge
            # If args.seq_len is provided and lower than config, consider using it?
            effective_cal_seq_len = config.max_seq_len
            if args.seq_len < config.max_seq_len:
                 logging.warning(f"Requested --seq_len {args.seq_len} is less than model's max {config.max_seq_len}. "
                                 f"Using {args.seq_len} for calibration dataset sequence length.")
                 effective_cal_seq_len = args.seq_len
            else:
                 logging.info(f"Using model's max sequence length ({config.max_seq_len}) for calibration dataset.")

            logging.info("EXL2 config prepared.")

            # 2. Prepare quantizer
            quantizer = ExLlamaV2Quantizer(config)
            quantizer.output_dir = output_dir

            # 3. Load/Run measurement
            measurement_path = args.exl2_measurement
            if measurement_path and os.path.exists(measurement_path):
                logging.info(f"Loading measurement data from: {measurement_path}")
                quantizer.load_measurement(measurement_path)
            else:
                logging.info("Performing measurement...")
                # Prepare dataset using the determined sequence length
                dataset = ExLlamaV2Dataset(config,
                                         args.exl2_cal_dataset,
                                         seq_len = effective_cal_seq_len, 
                                         rows = args.exl2_cal_rows)

                quantizer.measure(dataset)
                if measurement_path:
                    logging.info(f"Saving measurement data to: {measurement_path}")
                    quantizer.save_measurement(measurement_path)
                else:
                    logging.info("Measurement done. Not saving (no --exl2_measurement path specified).")
            
            # 4. Run quantization
            logging.info(f"Starting quantization job (Target BPW: {args.exl2_bpw})...")
            # Define quantization arguments based on parameters
            quant_args = {
                "target_bpw": args.exl2_bpw,
                "head_bits": args.exl2_head_bits,
                # Add other potential args like shard_size, overlap etc. if needed
            }
            quantizer.quantize(**quant_args)
            logging.info("EXL2 quantization finished successfully.")
            logging.info(f"Output saved to: {output_dir}")

            # 5. Copy tokenizer files (optional but recommended for usability)
            logging.info("Copying tokenizer files to EXL2 output directory...")
            try:
                 tokenizer_files = glob.glob(os.path.join(args.model_path, 'tokenizer*')) + \
                                   glob.glob(os.path.join(args.model_path, 'special_tokens_map.json')) + \
                                   glob.glob(os.path.join(args.model_path, 'vocab.*')) # Include vocab files if separate
                 for file_path in tokenizer_files:
                      filename = os.path.basename(file_path)
                      dest_path = os.path.join(output_dir, filename)
                      if not os.path.exists(dest_path): # Avoid overwriting if somehow present
                           logging.debug(f"Copying {filename} to {output_dir}")
                           shutil.copy2(file_path, dest_path)
                 logging.info("Tokenizer files copied.")
            except Exception as e:
                 logging.warning(f"Could not copy tokenizer files to EXL2 output directory: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"Error during EXL2 quantization: {e}", exc_info=True)
            # Optional: Clean up output dir on failure?
            # if os.path.exists(output_dir):
            #     logging.info(f"Cleaning up partially created EXL2 output dir: {output_dir}")
            #     shutil.rmtree(output_dir)
            return

    else:
        # This case should not be reachable due to the default logic
        logging.error(f"Internal error: Unsupported quantization method determined.")
        return

    logging.info("Quantization script finished successfully.")

if __name__ == '__main__':
    main()
