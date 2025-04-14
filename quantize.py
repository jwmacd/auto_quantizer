import argparse
import json
import os
import torch
import tempfile
import shutil
import glob
import psutil # Added for potential future RAM checks
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
    parser = argparse.ArgumentParser(description="Quantize a model using AWQ or GPTQ, automatically using GPU if available, or CPU.")
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

    # --- Device and Memory Management ---
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU execution even if GPU is available.')
    parser.add_argument('--max_memory', type=str, default="1GiB",
                        help='Maximum memory per GPU (e.g., "20GiB"). Used for device_map="auto". Default: 1GiB')
    parser.add_argument('--cpu_offload', action='store_true', default=None, # Default is determined later
                        help='Enable CPU offloading for GPU runs. Default: true if GPU, false if CPU.')
    parser.add_argument('--no_cpu_offload', action='store_true', default=False,
                        help='Disable CPU offloading (overrides --cpu_offload).')
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
    detects hardware (CPU/GPU), performs quantization, and saves results.
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

    # --- Determine Device and Configure Memory --- #
    if args.force_cpu:
        device = "cpu"
        device_map = "cpu"
        max_memory = None
        logging.info("Forcing CPU execution.")
    elif torch.cuda.is_available():
        device = "cuda"
        device_map = "auto" # Let accelerate handle distribution
        # Configure max_memory for device_map="auto"
        max_memory_dict = None
        gpu_limit_gb = None # Track the per-GPU limit

        if args.max_memory:
            try:
                # Simple parsing for now, assumes format like "20GiB"
                max_memory_dict = {i: args.max_memory for i in range(torch.cuda.device_count())}
                logging.info(f"Setting max_memory per GPU based on argument: {args.max_memory}")
                # Attempt to parse the numeric part for potential CPU limit calculation (optional)
                try: gpu_limit_gb = int(args.max_memory.lower().replace('gib', ''))
                except: pass # Ignore if parsing fails
            except Exception as e:
                logging.warning(f"Could not parse --max_memory '{args.max_memory}'. Using accelerate's default. Error: {e}")
                max_memory_dict = {}
        else:
            # Auto-detect GPU memory and set a limit
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                buffer_gb = 1 
                auto_max_memory_gb = int(total_mem_gb - buffer_gb)
                if auto_max_memory_gb > 0:
                    max_memory_dict = {i: f"{auto_max_memory_gb}GiB" for i in range(torch.cuda.device_count())}
                    logging.info(f"Auto-detected GPU memory. Setting max_memory per GPU: {auto_max_memory_gb}GiB")
                    gpu_limit_gb = auto_max_memory_gb
                else:
                    logging.warning("Could not auto-determine sufficient GPU memory. Using accelerate's default.")
                    max_memory_dict = {}
            except Exception as e:
                 logging.warning(f"Failed to auto-determine GPU memory: {e}. Using accelerate's default.")
                 max_memory_dict = {}

        # *** Add explicit CPU memory limit ***
        # Use a large fixed value since user has ample RAM and plans for larger models
        cpu_limit = "350GiB"
        max_memory_dict["cpu"] = cpu_limit
        logging.info(f"Setting explicit CPU memory limit for offloading: {cpu_limit}")

        max_memory = max_memory_dict # This dict now includes per-GPU and CPU limits

        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU detected: {gpu_name}")
        logging.info(f"Using device_map='{device_map}' with max_memory configuration: {max_memory}")
        if quantization_method == 'gptq':
            logging.warning("GPTQ is generally slower than AWQ, even on GPU.")
    else:
        device = "cpu"
        device_map = "cpu"
        max_memory = None # No max_memory dict needed for CPU-only
        logging.info("No GPU detected or --force_cpu used. Running on CPU.")
        if quantization_method == 'gptq':
            logging.warning("GPTQ on CPU is EXTREMELY slow.")

    # Configure CPU offloading (primarily for GPU runs)
    if args.no_cpu_offload:
        cpu_offload = False
        logging.info("CPU offload explicitly disabled by --no_cpu_offload.")
    elif args.cpu_offload is not None:
        cpu_offload = args.cpu_offload
        logging.info(f"CPU offload set to {cpu_offload} by argument.")
    else: # Default behavior
        cpu_offload = (device == "cuda") # Enable by default only if using GPU
        logging.info(f"CPU offload automatically set to {cpu_offload} (enabled for GPU, disabled for CPU).")

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

    # --- Method Specific Logic --- #
    # `device_map` and `max_memory` are now defined based on detection/args
    # `cpu_offload` is determined above

    if quantization_method == 'awq':
        # --- AWQ Quantization --- # 
        
        # Merge default AWQ config with custom config
        quant_config = {**DEFAULT_AWQ_CONFIG, **custom_config}
        quant_config['w_bit'] = args.bits # Ensure bits argument overrides config
        # Remove batch_size and seq_len from quant_config, they are passed directly to quantize
        # quant_config['calib_batch_size'] = args.batch_size
        # quant_config['calib_max_seq_len'] = args.seq_len
        logging.info(f"Using AWQ quantization config: {quant_config}")

        logging.info(f"Loading model for AWQ from: {args.model_path} using device_map='{device_map}'")
        load_kwargs = {
            "trust_remote_code": True,
            "safetensors": True,
            "device_map": device_map, 
        }
        # Pass the max_memory dict (which now includes "cpu") if using GPU
        if max_memory and device == "cuda":
            load_kwargs["max_memory"] = max_memory
        # Note: CPU offload is implicitly handled by device_map="auto" and max_memory settings
        # No explicit cpu_offload boolean needed for AutoAWQ load?
        # if cpu_offload and device == "cuda":
        #      pass # AutoAWQ handles offloading internally
        
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                args.model_path,
                **load_kwargs
            )
            logging.info(f"Pre-trained model loaded successfully for AWQ using device_map='{device_map}'.")
        except Exception as e:
            logging.error(f"Error loading model for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        logging.info("Starting AWQ quantization...")
        try:
            # Pass tokenizer and merged quant_config
            # Remove batch_size and max_seq_len, rely on autoawq defaults for now
            model.quantize(
                tokenizer,
                quant_config=quant_config
                # batch_size=args.batch_size, # Removed
                # max_seq_len=args.seq_len  # Removed
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
        # --- GPTQ Quantization --- #
        
        # Merge default GPTQ config with custom config
        gptq_base_config = {**DEFAULT_GPTQ_CONFIG, **custom_config} 
        # Note: gptq args like dataset, group_size, desc_act are separate command line args now

        logging.info("Preparing GPTQ configuration...")
        try:
            # Use args directly for GPTQConfig
            gptq_config = GPTQConfig(
                bits=args.bits, # Should be 8
                dataset=args.gptq_dataset,
                tokenizer=tokenizer,
                group_size=args.gptq_group_size,
                desc_act=args.gptq_desc_act,
                # Potentially add batch_size/seq_len here if GPTQConfig supports them directly?
                # Check Optimum documentation for GPTQConfig parameters if needed.
            )
            logging.info(f"Using GPTQ quantization config: {gptq_config}")
        except Exception as e:
            logging.error(f"Error creating GPTQConfig: {e}", exc_info=True)
            return

        logging.info(f"Loading model for GPTQ from: {args.model_path} using device_map='{device_map}'")
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "quantization_config": gptq_config 
        }
        # Pass the max_memory dict (which now includes "cpu") if using GPU
        if max_memory and device == "cuda":
            load_kwargs["max_memory"] = max_memory
        # Accelerate uses offload_folder when CPU offload is intense / disk might be involved
        # Setting the cpu limit in max_memory should prevent disk, but keep folder just in case
        if cpu_offload and device == "cuda":
             load_kwargs["offload_folder"] = "offload_gptq" 

        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                **load_kwargs
            )
            logging.info(f"Model loaded and quantization implicitly handled via GPTQConfig using device_map='{device_map}'.")
        except Exception as e:
            logging.error(f"Error loading model or initiating GPTQ quantization: {e}", exc_info=True)
            return

        # GPTQ quantization happens during load with quantization_config
        logging.info("GPTQ Quantization completed implicitly during model loading.")

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
            if cpu_offload and device == "cuda" and os.path.exists("offload_gptq"):
                logging.info("Cleaning up CPU offload directory: offload_gptq")
                shutil.rmtree("offload_gptq")

    else:
        # This case should not be reachable due to the default logic
        logging.error(f"Internal error: Unsupported quantization method determined.")
        return

    logging.info("Quantization script finished successfully.")

if __name__ == '__main__':
    main()
