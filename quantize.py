import argparse
import json
import os
import torch
import tempfile
import shutil
import glob
import logging
import time
import threading
import gc
import psutil
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
from datasets import load_dataset

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
    parser = argparse.ArgumentParser(description="Quantize a model using AWQ or GPTQ (CPU only).")
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
    
    # GPTQ specific arguments (only relevant if --gptq is used)
    parser.add_argument('--gptq_dataset', type=str, default=DEFAULT_GPTQ_CONFIG["dataset"],
                        help='Dataset name from Hugging Face Datasets for GPTQ calibration (e.g., wikitext2, c4). Default: wikitext2')
    parser.add_argument('--gptq_group_size', type=int, default=DEFAULT_GPTQ_CONFIG["group_size"],
                        help='Group size for GPTQ quantization. Default: 128')
    parser.add_argument('--gptq_desc_act', action='store_true',
                         help='Use descending activation order for GPTQ (sometimes improves accuracy). Default: False')

    args = parser.parse_args()

    # Default to AWQ if neither flag is set
    if not args.awq and not args.gptq:
        args.awq = True
        logging.info("Neither --awq nor --gptq specified, defaulting to AWQ.")

    return args

def log_memory_usage(tag=""):
    """Log current memory usage with an optional tag for identification"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert bytes to MB for more readable output
    rss_mb = memory_info.rss / (1024 * 1024)
    vms_mb = memory_info.vms / (1024 * 1024)
    
    logging.info(f"Memory usage {tag}: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB")
    
    # Log PyTorch memory stats if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
            logging.info(f"CUDA:{i} memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")

def main():
    """
    Main function: Parses arguments, validates, selects method (AWQ/GPTQ),
    performs quantization ON CPU, and saves results.
    """
    # Force CPU mode at the beginning of execution
    # These settings ensure no GPU will be used even if available
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide any CUDA devices
    torch.cuda.is_available = lambda: False  # Ensure CUDA is reported as unavailable
    
    # Log initial memory state
    log_memory_usage("at startup")
    
    args = parse_arguments()

    # --- Argument Validation and Setup --- #
    logging.info(f"Starting quantization process with args: {args}")
    logging.info("Forcing CPU-only mode for all operations")

    # Determine method and set default bits
    quantization_method = "awq" if args.awq else "gptq"
    if args.bits is None:
        if quantization_method == 'awq':
            args.bits = 4
        else: # gptq
            args.bits = 8
    
    logging.info(f"Selected method: {quantization_method.upper()}, Bits: {args.bits}")
    logging.info(f"Device: CPU (Forced for both methods in this script)")
    if quantization_method == 'gptq':
        logging.warning("GPTQ on CPU is EXTREMELY slow.")

    # Validate bits per method
    if quantization_method == 'awq' and args.bits != 4:
        logging.error(f"AWQ method currently only supports --bits 4. Got: {args.bits}")
        return
    if quantization_method == 'gptq' and args.bits != 8:
        # While GPTQ can technically do 4-bit, this script focuses on 8-bit for simplicity
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
    device = "cpu" # Force CPU for all operations
    
    # Force PyTorch to only use CPU
    torch.set_num_threads(torch.get_num_threads())  # Preserve thread count but ensure we're on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide any CUDA devices
    
    if quantization_method == 'awq':
        # --- AWQ Quantization (CPU Only) --- # 
        
        # Merge default AWQ config with custom config
        quant_config = {**DEFAULT_AWQ_CONFIG, **custom_config}
        quant_config['w_bit'] = args.bits # Ensure bits argument overrides config
        logging.info(f"Using AWQ quantization config: {quant_config}")

        logging.info(f"Loading model for AWQ from: {args.model_path} onto CPU")
        try:
            # Ensure no GPU usage
            torch.cuda.is_available = lambda: False
            
            logging.info("Loading model for AWQ quantization...")
            log_memory_usage("before model loading")
            
            model = AutoAWQForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                safetensors=True,
                device_map="cpu" # Explicitly load to CPU with string value
            )
            
            log_memory_usage("after model loading")
            logging.info("Pre-trained model loaded successfully for AWQ onto CPU.")
        except Exception as e:
            logging.error(f"Error loading model for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        logging.info("Starting AWQ quantization...")
        try:
            # Add progress tracking
            start_time = time.time()
            logging.info(f"AWQ quantization started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Set up a progress checker on a separate thread
            def log_progress():
                last_log_time = time.time()
                while True:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if current_time - last_log_time >= 300:  # Log every 5 minutes
                        last_log_time = current_time
                        logging.info(f"AWQ quantization still in progress... (Elapsed: {elapsed:.2f} seconds)")
                        log_memory_usage("during AWQ quantization")
                        # Force garbage collection to get accurate memory readings
                        gc.collect()
                    time.sleep(60)  # Check every minute
                    
            import threading
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            log_memory_usage("before AWQ quantization")
            model.quantize(
                tokenizer,
                quant_config=quant_config
            )
            
            elapsed = time.time() - start_time
            log_memory_usage("after AWQ quantization")
            logging.info(f"AWQ Quantization completed successfully in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        except Exception as e:
            logging.error(f"Error during AWQ quantization: {e}", exc_info=True)
            return

        # --- AWQ Saving --- #
        output_suffix = "AWQ"
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Saving AWQ quantized model temporarily before moving to: {args.model_path}")
        logging.info(f"Using temporary directory: {temp_dir}")
        try:
            model.save_quantized(temp_dir, shard_size="4GB") # Limit to 4GB for Hugging Face compatibility
            logging.info(f"AWQ quantized model temporarily saved to {temp_dir}")

            # Files to copy: *.safetensors, quant_config.json
            files_to_copy = glob.glob(os.path.join(temp_dir, '*.safetensors'))
            files_to_copy.append(os.path.join(temp_dir, 'quant_config.json'))
            
            # Handle index file specially
            index_file_temp = os.path.join(temp_dir, 'model.safetensors.index.json')
            original_index_file = os.path.join(args.model_path, 'model.safetensors.index.json')
            
            logging.info(f"Copying {len(files_to_copy)} AWQ files from temp dir to {args.model_path}")
            
            # First, process the model files to collect mapping for index
            model_file_mapping = {}
            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Add -AWQ suffix to safetensors and config files
                if filename.endswith('.safetensors') and not filename.endswith('-AWQ.safetensors'):
                    # Files in temp_dir don't have suffix yet, add it during copy
                    orig_name = filename
                    dest_filename = filename.replace(".safetensors", "-AWQ.safetensors")
                    model_file_mapping[orig_name] = dest_filename
                elif filename == 'quant_config.json':
                    dest_filename = "quant_config-AWQ.json"
                else:
                    # This case should ideally not be hit. Log a warning if it does.
                    logging.warning(f"Unexpected file found in AWQ temp directory: {filename}. Copying as-is.")
                    dest_filename = filename 

                dest_path = os.path.join(args.model_path, dest_filename)
                # Safety check - though unlikely to hit originals with this logic
                if os.path.exists(dest_path):
                    logging.warning(f"Destination file {dest_path} already exists. Overwriting.") 

                logging.debug(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path) # Use copy2 to preserve metadata
                
            # Create proper index file using mapping from original files if it exists
            if os.path.exists(original_index_file) and model_file_mapping:
                logging.info(f"Creating AWQ index file based on original index: {original_index_file}")
                try:
                    with open(original_index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    # Create new weight map with AWQ file names
                    if 'weight_map' in index_data:
                        new_weight_map = {}
                        for tensor_name, file_name in index_data['weight_map'].items():
                            # Map to the new AWQ file name if available
                            base_name = os.path.basename(file_name)
                            if base_name in model_file_mapping:
                                new_weight_map[tensor_name] = model_file_mapping[base_name]
                            else:
                                # Keep original mapping if not found in our processed files
                                new_weight_map[tensor_name] = file_name.replace(".safetensors", "-AWQ.safetensors")
                        
                        # Update with new weight map
                        index_data['weight_map'] = new_weight_map
                        
                        # Add AWQ metadata
                        if 'metadata' not in index_data:
                            index_data['metadata'] = {}
                        index_data['metadata']['awq_bits'] = args.bits
                        
                        # Save the new index file
                        awq_index_path = os.path.join(args.model_path, "model-AWQ.safetensors.index.json")
                        with open(awq_index_path, 'w') as f:
                            json.dump(index_data, f, indent=2)
                        logging.info(f"Created AWQ index file at {awq_index_path}")
                    else:
                        logging.warning("Original index file exists but doesn't contain a weight_map. Creating simplified index.")
                        # Fall back to simple index file creation
                        if os.path.exists(index_file_temp):
                            dest_path = os.path.join(args.model_path, "model-AWQ.safetensors.index.json")
                            shutil.copy2(index_file_temp, dest_path)
                            logging.info(f"Copied original AWQ index file to {dest_path}")
                except Exception as e:
                    logging.error(f"Error creating AWQ index file: {e}", exc_info=True)
                    # Fall back to simple index file copy if available
                    if os.path.exists(index_file_temp):
                        dest_path = os.path.join(args.model_path, "model-AWQ.safetensors.index.json")
                        shutil.copy2(index_file_temp, dest_path)
                        logging.info(f"Copied original AWQ index file to {dest_path} after index creation error")
            elif os.path.exists(index_file_temp):
                # Just copy the temp index file if no original index exists
                dest_path = os.path.join(args.model_path, "model-AWQ.safetensors.index.json")
                shutil.copy2(index_file_temp, dest_path)
                logging.info(f"Copied AWQ-generated index file to {dest_path}")
            
            # Handle any custom Python files
            custom_code_files = glob.glob(os.path.join(temp_dir, '*.py'))
            if custom_code_files:
                logging.info(f"Found {len(custom_code_files)} custom Python files to copy")
                for py_file in custom_code_files:
                    py_filename = os.path.basename(py_file)
                    dest_path = os.path.join(args.model_path, f"{py_filename[:-3]}-AWQ.py")
                    logging.info(f"Copying custom code file to {dest_path}")
                    shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully copied AWQ files to {args.model_path}")
        except Exception as e:
            logging.error(f"Error during AWQ model saving/copying: {e}", exc_info=True)
        finally:
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    elif quantization_method == 'gptq':
        # --- GPTQ Quantization (8-bit, CPU Only) --- #
        
        # Merge default GPTQ config with custom config
        gptq_base_config = {**DEFAULT_GPTQ_CONFIG, **custom_config}

        logging.info("Preparing GPTQ configuration...")
        try:
            gptq_config = GPTQConfig(
                bits=args.bits, # Should be 8
                dataset=args.gptq_dataset,
                tokenizer=tokenizer,
                group_size=args.gptq_group_size,
                desc_act=args.gptq_desc_act,
            )
            logging.info(f"Using GPTQ quantization config: {gptq_config}")
        except Exception as e:
            logging.error(f"Error creating GPTQConfig: {e}", exc_info=True)
            return

        logging.info(f"Loading model for GPTQ from: {args.model_path} onto CPU")
        # For GPTQ using transformers, we load the model first and then quantize
        # Device placement is handled by from_pretrained
        # Force CPU explicitly with string value
        device_map = "cpu" 
        logging.info(f"Using device '{device_map}' for CPU GPTQ quantization.")
        
        try:
            # Ensure no GPU usage for GPTQ
            torch.cuda.is_available = lambda: False
            
            # Add progress tracking for GPTQ
            start_time = time.time()
            logging.info(f"GPTQ quantization started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Set up a progress checker on a separate thread
            def log_progress():
                last_log_time = time.time()
                while True:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if current_time - last_log_time >= 300:  # Log every 5 minutes
                        last_log_time = current_time
                        logging.info(f"GPTQ quantization still in progress... (Elapsed: {elapsed:.2f} seconds)")
                        log_memory_usage("during GPTQ quantization")
                        # Force garbage collection to get accurate memory readings
                        gc.collect()
                    time.sleep(60)  # Check every minute
                    
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            # Force CPU operations
            log_memory_usage("before GPTQ model loading")
            with torch.device('cpu'):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    device_map=device_map, 
                    trust_remote_code=True,
                    quantization_config=gptq_config # Pass the config here
                )
            
            elapsed = time.time() - start_time
            log_memory_usage("after GPTQ quantization")
            logging.info(f"Model loaded and GPTQ quantization completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        except Exception as e:
            logging.error(f"Error loading model or initiating GPTQ quantization: {e}", exc_info=True)
            return

        logging.info("GPTQ Quantization completed implicitly during model loading.")

        # --- GPTQ Saving --- #
        # Create a temporary directory for saving first, then move files with suffix
        temp_dir = tempfile.mkdtemp()
        logging.info(f"Saving GPTQ quantized model temporarily to: {temp_dir}")
        try:
            # save_pretrained will save the model shards, config with quant info, tokenizer etc.
            model.save_pretrained(temp_dir, max_shard_size="4GB") # Limit to 4GB for Hugging Face compatibility
            tokenizer.save_pretrained(temp_dir) 
            logging.info(f"GPTQ quantized model and tokenizer temporarily saved to {temp_dir}")

            # Copy files from temp_dir to original directory with -GPTQ suffix
            # Identify the model weight files (.safetensors, .bin) and config files
            model_files = glob.glob(os.path.join(temp_dir, '*.safetensors')) + glob.glob(os.path.join(temp_dir, '*.bin'))
            config_file = os.path.join(temp_dir, 'config.json')
            
            # Copy model files with -GPTQ suffix
            for file_path in model_files:
                filename = os.path.basename(file_path)
                base, ext = os.path.splitext(filename)
                dest_filename = f"{base}-GPTQ{ext}"
                dest_path = os.path.join(args.model_path, dest_filename)
                logging.info(f"Copying {file_path} to {dest_path}")
                shutil.copy2(file_path, dest_path)
            
            # Copy and rename config file
            if os.path.exists(config_file):
                dest_path = os.path.join(args.model_path, 'config-GPTQ.json')
                logging.info(f"Copying config to {dest_path}")
                shutil.copy2(config_file, dest_path)
            
            # Handle index file creation for GPTQ
            original_index_file = os.path.join(args.model_path, 'model.safetensors.index.json')
            temp_index_file = os.path.join(temp_dir, 'model.safetensors.index.json')
            
            # Build mapping of original files to GPTQ files
            model_file_mapping = {}
            for file_path in model_files:
                filename = os.path.basename(file_path)
                base, ext = os.path.splitext(filename)
                dest_filename = f"{base}-GPTQ{ext}"
                model_file_mapping[filename] = dest_filename
            
            # If original index exists, use it as a template
            if os.path.exists(original_index_file) and model_files:
                logging.info(f"Creating GPTQ index file based on original index: {original_index_file}")
                try:
                    with open(original_index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    # Update the weight map with GPTQ file names
                    if 'weight_map' in index_data:
                        new_weight_map = {}
                        for tensor_name, file_name in index_data['weight_map'].items():
                            base_name = os.path.basename(file_name)
                            # Try to find a mapping for this file
                            matching_new_file = None
                            for orig_file in model_file_mapping:
                                # Look for partial matches to handle potential name differences
                                if orig_file in base_name or base_name in orig_file:
                                    matching_new_file = model_file_mapping[orig_file]
                                    break
                            
                            if matching_new_file:
                                new_weight_map[tensor_name] = matching_new_file
                            else:
                                # If no match, use pattern replacement
                                new_weight_map[tensor_name] = file_name.replace(".safetensors", "-GPTQ.safetensors")
                        
                        # Update the index data
                        index_data['weight_map'] = new_weight_map
                        
                        # Add GPTQ metadata
                        if 'metadata' not in index_data:
                            index_data['metadata'] = {}
                        index_data['metadata']['gptq_bits'] = args.bits
                        index_data['metadata']['gptq_group_size'] = args.gptq_group_size
                        
                        # Save the new index file
                        gptq_index_path = os.path.join(args.model_path, "model-GPTQ.safetensors.index.json")
                        with open(gptq_index_path, 'w') as f:
                            json.dump(index_data, f, indent=2)
                        logging.info(f"Created GPTQ index file at {gptq_index_path}")
                    else:
                        logging.warning("Original index exists but doesn't have a weight_map. Creating simplified index.")
                        # Fall back to temp file or create a simple mapping
                        _create_simple_gptq_index(temp_index_file, args, model_files, model_file_mapping)
                except Exception as e:
                    logging.error(f"Error creating GPTQ index file: {e}", exc_info=True)
                    # Fall back to simple creation
                    _create_simple_gptq_index(temp_index_file, args, model_files, model_file_mapping)
            # If temp index exists, try to use it
            elif os.path.exists(temp_index_file):
                logging.info(f"Using GPTQ-generated index file as base")
                try:
                    with open(temp_index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    # Add GPTQ metadata
                    if 'metadata' not in index_data:
                        index_data['metadata'] = {}
                    index_data['metadata']['gptq_bits'] = args.bits
                    index_data['metadata']['gptq_group_size'] = args.gptq_group_size
                    
                    # Update weight map if it exists
                    if 'weight_map' in index_data:
                        new_weight_map = {}
                        for tensor_name, file_name in index_data['weight_map'].items():
                            new_weight_map[tensor_name] = file_name.replace(".safetensors", "-GPTQ.safetensors")
                        index_data['weight_map'] = new_weight_map
                    
                    # Save the modified index file
                    gptq_index_path = os.path.join(args.model_path, "model-GPTQ.safetensors.index.json")
                    with open(gptq_index_path, 'w') as f:
                        json.dump(index_data, f, indent=2)
                    logging.info(f"Created GPTQ index file from temp index at {gptq_index_path}")
                except Exception as e:
                    logging.error(f"Error processing temp GPTQ index file: {e}", exc_info=True)
                    # Fall back to simple creation
                    _create_simple_gptq_index(temp_index_file, args, model_files, model_file_mapping)
            # Otherwise create a simple index from scratch if we have multiple files
            elif len(model_files) > 1:
                _create_simple_gptq_index(None, args, model_files, model_file_mapping)

# Helper function to create a simple GPTQ index
def _create_simple_gptq_index(temp_index_file, args, model_files, model_file_mapping):
    """Helper method to create a simplified GPTQ index file"""
    # Create a simple index for GPTQ
    index_data = {"metadata": {"gptq_bits": args.bits, "gptq_group_size": args.gptq_group_size}}
    weight_map = {}
    
    for orig_file, gptq_file in model_file_mapping.items():
        # Create a simple placeholder mapping with file names as tensor names
        # This is not ideal but better than nothing
        weight_map[f"tensor_{orig_file}"] = gptq_file
    
    index_data["weight_map"] = weight_map
    
    # Save the index file
    index_path = os.path.join(args.model_path, "model-GPTQ.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    logging.info(f"Created simplified GPTQ index file at {index_path}")

            # Copy any custom code files if they exist
            custom_code_files = glob.glob(os.path.join(temp_dir, '*.py'))
            for py_file in custom_code_files:
                py_filename = os.path.basename(py_file)
                dest_path = os.path.join(args.model_path, f"{py_filename[:-3]}-GPTQ.py")
                logging.info(f"Copying custom code file to {dest_path}")
                shutil.copy2(py_file, dest_path)

            logging.info(f"Successfully saved GPTQ files to {args.model_path} with -GPTQ suffix")
        except Exception as e:
            logging.error(f"Error during GPTQ model saving: {e}", exc_info=True)
            return
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
