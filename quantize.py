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
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
from datasets import load_dataset
import logging

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
    parser.add_argument('--seq_len', type=int, default=8192, # Default from AWQ, user can reduce
                        help='Maximum sequence length for AWQ calibration data. Default: 8192')


    # --- GPTQ Specific Arguments --- #
    parser.add_argument('--gptq_dataset', type=str, default=DEFAULT_GPTQ_CONFIG["dataset"],
                        help='Dataset name from Hugging Face Datasets for GPTQ calibration (e.g., wikitext2, c4). Default: wikitext2')
    parser.add_argument('--gptq_group_size', type=int, default=DEFAULT_GPTQ_CONFIG["group_size"],
                        help='Group size for GPTQ quantization. Default: 128')
    parser.add_argument('--gptq_desc_act', action='store_true', default=False,
                         help='Use descending activation order for GPTQ (sometimes improves accuracy). Default: False')

    args = parser.parse_args()

    # Default to AWQ if neither flag is set
    if not args.awq and not args.gptq:
        args.awq = True
        logging.info("Neither --awq nor --gptq specified, defaulting to AWQ.")

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

    # Validate bits per method
    if quantization_method == 'awq' and args.bits != 4:
        logging.error(f"AWQ method currently only supports --bits 4. Got: {args.bits}")
        return
    if quantization_method == 'gptq' and args.bits != 8:
        logging.error(f"This script's GPTQ implementation currently supports --bits 8. Got: {args.bits}")
        return

    # Load custom quant_config if provided
    try:
        custom_config = json.loads(args.quant_config)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON string provided for --quant_config: {args.quant_config}")
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
        logging.info(f"Saving AWQ quantized model temporarily before moving to: {args.model_path}")
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

            logging.info(f"Copying {len(files_to_copy)} AWQ files from temp dir to {args.model_path}")
            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Add -AWQ suffix to safetensors and config files
                if filename.endswith('.safetensors') and not filename.startswith('model'):
                    # Handle potential non-model safetensors if any? Unlikely.
                    dest_filename = f"{filename[:-len('.safetensors')]}-AWQ.safetensors"
                elif filename.startswith('model') and filename.endswith('.safetensors'):
                    dest_filename = filename.replace(".safetensors", "-AWQ.safetensors")
                elif filename == 'quant_config.json':
                    dest_filename = "quant_config-AWQ.json"
                elif filename == 'model.safetensors.index.json':
                     dest_filename = "model-AWQ.safetensors.index.json"
                else:
                     logging.warning(f"Unexpected file found in AWQ temp directory: {filename}. Copying as-is.")
                     dest_filename = filename 

                dest_path = os.path.join(args.model_path, dest_filename)
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
                    dest_path = os.path.join(args.model_path, f"{py_filename[:-3]}-AWQ.py") # Add suffix
                    logging.info(f"Copying custom code file to {dest_path}")
                    shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully copied AWQ files to {args.model_path}")
        except Exception as e:
            logging.error(f"Error during AWQ model saving/copying: {e}", exc_info=True)
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    elif quantization_method == 'gptq':
        # --- GPTQ Model Loading --- #
        logging.info("Preparing GPTQ configuration...")
        try:
            gptq_config = GPTQConfig(
                bits=args.bits, 
                dataset=args.gptq_dataset, # Still uses string name - need manual load for seq_len
                tokenizer=tokenizer,
                group_size=args.gptq_group_size,
                desc_act=args.gptq_desc_act,
            )
            logging.info(f"Using GPTQ quantization config: {gptq_config}")
        except Exception as e:
            logging.error(f"Error creating GPTQConfig: {e}", exc_info=True)
            return

        logging.info(f"Loading model for GPTQ from: {args.model_path} onto CPU initially")
        try:
            # Load directly to CPU, passing quantization_config
            # Quantization happens *during* this load when using quantization_config
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="cpu", # Load weights to CPU
                trust_remote_code=True,
                quantization_config=gptq_config 
            )
            logging.info(f"Model loaded onto CPU. Quantization implicitly handled via GPTQConfig during load.")

            # Apply explicit CPU offload if using GPU *after* initial load/quantization
            if use_gpu_offload:
                logging.info(f"Applying accelerate explicit CPU offload to target device {execution_device}")
                # Offload buffers might be needed depending on model structure?
                cpu_offload(model, execution_device=execution_device, offload_buffers=False)
                logging.info("Explicit CPU offload applied for GPTQ.")
            elif execution_device == "cpu":
                 logging.info("Running GPTQ on CPU without offload.")

        except Exception as e:
            logging.error(f"Error loading model, running GPTQ, or applying offload: {e}", exc_info=True)
            return

        # --- GPTQ Saving --- #
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Saving GPTQ quantized model temporarily to: {temp_dir}")
        try:
            model.save_pretrained(temp_dir, max_shard_size="4GB") 
            tokenizer.save_pretrained(temp_dir) 
            logging.info(f"GPTQ quantized model and tokenizer temporarily saved to {temp_dir}")

            # Identify the model weight files (.safetensors, .bin) and config files
            model_files = glob.glob(os.path.join(temp_dir, '*.safetensors')) + glob.glob(os.path.join(temp_dir, '*.bin'))
            config_file = os.path.join(temp_dir, 'config.json') # Original config name in temp dir
            tokenizer_files = glob.glob(os.path.join(temp_dir, 'tokenizer*')) # tokenizer.json, tokenizer_config.json etc.
            special_tokens_map = os.path.join(temp_dir, 'special_tokens_map.json')
            if os.path.exists(special_tokens_map):
                tokenizer_files.append(special_tokens_map)

            files_to_copy = model_files + tokenizer_files
            if os.path.exists(config_file):
                files_to_copy.append(config_file)

            logging.info(f"Copying {len(files_to_copy)} GPTQ-related files from temp dir to {args.model_path}")
            
            # Prepare for index file creation
            weight_map = {}
            copied_model_files = []

            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                dest_filename = filename # Default

                # Add -GPTQ suffix to model weights and config
                if filename in [os.path.basename(f) for f in model_files]:
                    base, ext = os.path.splitext(filename)
                    dest_filename = f"{base}-GPTQ{ext}"
                    weight_map[filename] = dest_filename # Map original name to new suffixed name for index
                    copied_model_files.append(dest_filename)
                elif filename == 'config.json':
                    dest_filename = 'config-GPTQ.json'
                # Keep tokenizer files as they are (no suffix needed)
                elif filename in [os.path.basename(f) for f in tokenizer_files]:
                     dest_filename = filename 
                else:
                     logging.warning(f"Unexpected file found in GPTQ temp directory: {filename}. Copying as-is.")

                dest_path = os.path.join(args.model_path, dest_filename)
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
                
                index_path = os.path.join(args.model_path, "model-GPTQ.safetensors.index.json")
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
                dest_path = os.path.join(args.model_path, f"{py_filename[:-3]}-GPTQ.py") # Add suffix
                logging.info(f"Copying custom code file to {dest_path}")
                shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully saved GPTQ files to {args.model_path} with -GPTQ suffix")
        except Exception as e:
            logging.error(f"Error during GPTQ model saving/copying: {e}", exc_info=True)
            # Don't return here, proceed to finally
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            if use_gpu_offload and execution_device == "cuda" and os.path.exists("offload_gptq"):
                logging.info("Cleaning up CPU offload directory: offload_gptq")
                shutil.rmtree("offload_gptq")

    else:
        # This case should not be reachable due to the default logic
        logging.error(f"Internal error: Unsupported quantization method determined.")
        return

    logging.info("Quantization script finished successfully.")

if __name__ == '__main__':
    main()
