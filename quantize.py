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
try:
    import psutil
except ImportError:
    psutil = None  # We'll handle missing psutil gracefully
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
    
    # GPU memory management options
    parser.add_argument('--max_memory', type=str, default=None,
                        help='Maximum memory to use for model loading, format: "24GiB" or {"cuda:0": "24GiB"}. Default: auto-detect.')
    parser.add_argument('--cpu_offload', action='store_true', default=None,
                        help='Enable CPU offloading for memory efficiency (enabled by default, use --no_cpu_offload to disable).')
    parser.add_argument('--no_cpu_offload', action='store_true',
                        help='Disable CPU offloading (overrides --cpu_offload)')
    parser.add_argument('--load_in_8bit', action='store_true', default=False,
                        help='Load model in 8-bit mode first to reduce memory usage (may slightly impact quality).')
    
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

def calculate_safe_memory_limits():
    """Automatically calculate safe memory limits for both GPU and CPU based on available hardware"""
    memory_dict = {}
    
    # Check for psutil availability
    if psutil is not None:
        try:
            # Get CPU memory information
            cpu_total_mem = psutil.virtual_memory().total / (1024**3)  # Total RAM in GB
            cpu_available_mem = psutil.virtual_memory().available / (1024**3)  # Available RAM in GB
            
            # Determine how much CPU memory to use (more aggressive on high-memory systems)
            if cpu_total_mem > 256:  # Very high memory systems (>256GB)
                # Use up to 80% of available memory or 70% of total, whichever is less
                cpu_usable_mem = min(cpu_available_mem * 0.8, cpu_total_mem * 0.7)
            elif cpu_total_mem > 64:  # High memory systems (64-256GB)
                # Use up to 75% of available memory or 65% of total, whichever is less
                cpu_usable_mem = min(cpu_available_mem * 0.75, cpu_total_mem * 0.65)
            elif cpu_total_mem > 16:  # Medium memory systems (16-64GB)
                # Use up to 70% of available memory or 60% of total, whichever is less
                cpu_usable_mem = min(cpu_available_mem * 0.7, cpu_total_mem * 0.6)
            else:  # Low memory systems (<16GB)
                # Use up to 60% of available memory or 50% of total, whichever is less
                cpu_usable_mem = min(cpu_available_mem * 0.6, cpu_total_mem * 0.5)
            
            # Format CPU memory string (round down to integer GB)
            cpu_mem_str = f"{int(cpu_usable_mem)}GiB"
            memory_dict["cpu"] = cpu_mem_str
            
            logging.info(f"Auto-configured CPU memory: {cpu_mem_str} (total RAM: {cpu_total_mem:.1f}GB, available: {cpu_available_mem:.1f}GB)")
        except Exception as e:
            # Fallback to a reasonable default if memory detection fails
            memory_dict["cpu"] = "64GiB"
            logging.warning(f"Error detecting CPU memory details: {e}. Using default 64GiB.")
    else:
        # psutil is not available, use a reasonable default
        memory_dict["cpu"] = "64GiB"
        logging.warning("The 'psutil' package is not installed. Using default CPU memory limit of 64GiB.")
    
    # Add GPU memory allocation if available - using ONLY the first GPU (cuda:0)
    if torch.cuda.is_available():
        try:
            # We'll only use the first GPU (cuda:0) to avoid multi-GPU complexity
            if torch.cuda.device_count() > 1:
                logging.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), but using only the first GPU for reliability")
            
            # Get total GPU memory for first GPU only
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            free_mem = torch.cuda.mem_get_info(0)[0] / (1024**3)  # Free memory in GB
            
            # Use a conservative percentage of total memory
            # Reserve more for larger GPUs, less for smaller ones
            if total_mem > 20:  # High-end GPUs (24GB+)
                # Reserve 20% of total memory or 4GB, whichever is greater
                reserve_percentage = 0.20
                minimum_reserve = 4
            elif total_mem > 10:  # Mid-range GPUs (12-16GB)
                # Reserve 15% of total memory or 2GB, whichever is greater
                reserve_percentage = 0.15
                minimum_reserve = 2
            else:  # Low-end GPUs (<10GB)
                # Reserve 10% of total memory or 1GB, whichever is greater
                reserve_percentage = 0.10
                minimum_reserve = 1
            
            # Calculate reserve amount
            reserve_amount = max(total_mem * reserve_percentage, minimum_reserve)
            
            # Calculate usable memory
            usable_mem = total_mem - reserve_amount
            
            # Format as string
            usable_mem_str = f"{int(usable_mem)}GiB"
            
            # Create memory dict with ONLY cuda:0 (single GPU)
            memory_dict = {
                "cuda:0": usable_mem_str,  # Only use first GPU
                "cpu": memory_dict.get("cpu", "64GiB")  # Keep CPU allocation
            }
            
            logging.info(f"Auto-configured GPU memory: {usable_mem_str} (total: {total_mem:.1f}GB, reserved: {reserve_amount:.1f}GB)")
        except Exception as e:
            logging.warning(f"Error calculating GPU memory: {e}. Using automatic device mapping.")
            return {"cuda:0": "auto", "cpu": memory_dict.get("cpu", "64GiB")}  # Fall back to auto but still limit to first GPU
    
    return memory_dict

def parse_max_memory(max_memory_str):
    """Parse max_memory argument into a usable format for device_map"""
    if not max_memory_str:
        # No explicit value provided, use automatic calculation
        return calculate_safe_memory_limits()
    
    try:
        # Check if it's a JSON string like '{"cuda:0": "24GiB"}'
        if max_memory_str.startswith('{'):
            return json.loads(max_memory_str)
        # Otherwise assume it's a simple string like "24GiB"
        else:
            # Start with auto-detected CPU memory values
            memory_dict = {}
            try:
                # Get automatic CPU memory value
                auto_limits = calculate_safe_memory_limits()
                if "cpu" in auto_limits:
                    memory_dict["cpu"] = auto_limits["cpu"]
                else:
                    memory_dict["cpu"] = "64GiB"  # Fallback
            except:
                memory_dict["cpu"] = "64GiB"  # Fallback if auto-detection fails
            
            # Apply GPU memory limit from provided value - ONLY to the first GPU
            if torch.cuda.is_available():
                # Only use the first GPU to avoid multi-GPU errors
                memory_dict["cuda:0"] = max_memory_str
                
                if torch.cuda.device_count() > 1:
                    logging.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), but using only the first GPU for reliability")
                
            return memory_dict
    except Exception as e:
        logging.warning(f"Error parsing max_memory argument: {e}. Using automatic memory allocation.")
        return calculate_safe_memory_limits()

def main():
    """
    Main function: Parses arguments, validates, selects method (AWQ/GPTQ),
    performs quantization, and saves results.
    """
    args = parse_arguments()
    
    # Process overriding args
    if args.no_cpu_offload:
        args.cpu_offload = False  # Disable CPU offloading if explicitly requested
    
    # Check if GPU is available and log the device that will be used
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logging.info(f"Using GPU for quantization (much faster)")
        
        # Log GPU memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU total memory: {total_mem:.2f} GB")
        
        # Set default optimizations for GPU
        if args.cpu_offload is None:
            args.cpu_offload = True  # Enable by default on GPU for better reliability
    else:
        logging.info("No GPU detected. Using CPU for quantization (will be slow)")

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
    if not gpu_available and quantization_method == 'gptq':
        logging.warning("GPTQ on CPU is EXTREMELY slow. Consider using a GPU if available.")

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
    # Use the device determined at the start (CPU or GPU)

    if quantization_method == 'awq':
        # --- AWQ Quantization --- # 
        
        # Merge default AWQ config with custom config
        quant_config = {**DEFAULT_AWQ_CONFIG, **custom_config}
        quant_config['w_bit'] = args.bits # Ensure bits argument overrides config
        logging.info(f"Using AWQ quantization config: {quant_config}")

        # Only use the first GPU to avoid multi-GPU complexity
        if gpu_available:
            # Dynamically create max_memory map for available GPUs
            num_gpus = torch.cuda.device_count()
            logging.info(f"Detected {num_gpus} CUDA devices available")
            
            # Create a simpler memory map with a single GPU
            # This is a critical change from the working version
            max_memory = {0: "23GiB"}  # Allocate fixed 23GB to first GPU only
            
            if torch.cuda.device_count() > 1:
                logging.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), but using only the first GPU for reliability")
            
            logging.info(f"Using GPU memory map: {max_memory}")
        else:
            max_memory = None
            logging.info("No GPU available, using CPU only")
        
        logging.info(f"Loading model for AWQ from: {args.model_path}")
        try:
            # Load model directly with simplified approach (from working version)
            model = AutoAWQForCausalLM.from_pretrained(
                args.model_path,
                max_memory=max_memory,  # Use the simplified memory map
                device_map="auto",      # Let HF Accelerate handle device mapping
                trust_remote_code=True,
                safetensors=True,       # Prefer safetensors loading
            )
            logging.info("Pre-trained model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        logging.info("Starting AWQ quantization...")
        try:
            # Simple direct quantization (from working version)
            model.quantize(
                tokenizer, 
                quant_config=quant_config
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
            model.save_quantized(temp_dir, shard_size="10GB") # Increase shard size maybe? CPU might handle larger files better
            logging.info(f"AWQ quantized model temporarily saved to {temp_dir}")

            # Files to copy: *.safetensors, quant_config.json, AND model.safetensors.index.json
            files_to_copy = glob.glob(os.path.join(temp_dir, '*.safetensors'))
            files_to_copy.append(os.path.join(temp_dir, 'quant_config.json'))
            index_file_temp = os.path.join(temp_dir, 'model.safetensors.index.json')
            if os.path.exists(index_file_temp):
                files_to_copy.append(index_file_temp)
            else:
                logging.warning(f"Expected index file not found in temp dir: {index_file_temp}")

            logging.info(f"Copying {len(files_to_copy)} AWQ files from temp dir to {args.model_path}")
            for file_path in files_to_copy:
                filename = os.path.basename(file_path)
                # Add -AWQ suffix to safetensors and config files
                if filename.endswith('.safetensors') and not filename.endswith('-AWQ.safetensors'):
                    # Files in temp_dir don't have suffix yet, add it during copy
                    dest_filename = filename.replace(".safetensors", "-AWQ.safetensors")
                elif filename == 'quant_config.json':
                    dest_filename = "quant_config-AWQ.json"
                # *** RENAME the index file on copy to preserve the original ***
                elif filename == 'model.safetensors.index.json':
                     dest_filename = "model-AWQ.safetensors.index.json" # New name for AWQ index
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

        # Set up device map based on args - similar to AWQ but for GPTQ
        device_map = "auto"  # Default to auto device map
        max_memory = parse_max_memory(args.max_memory)
        
        if gpu_available:
            if args.cpu_offload:
                logging.info("CPU offloading enabled for memory efficiency")
                device_map = "auto"
            elif max_memory:
                logging.info(f"Using custom memory limits: {max_memory}")
                device_map = max_memory
            else:
                # Try to auto-determine conservative memory usage
                # Reserve 2GB of GPU memory for quantization processes
                free_mem = torch.cuda.mem_get_info(0)[0] / (1024**3)
                reserve_mem = 2.0  # 2 GB reserve
                usable_mem = max(free_mem - reserve_mem, free_mem * 0.8)  # Use at most 80% of free memory
                
                logging.info(f"Free GPU memory: {free_mem:.2f} GB, using {usable_mem:.2f} GB for model loading")
                max_memory = {
                    "cuda:0": f"{usable_mem:.0f}GiB",
                    "cpu": "64GiB"
                }
                device_map = max_memory
        else:
            device_map = "cpu"
        
        logging.info(f"Loading model for GPTQ from: {args.model_path} with device_map: {device_map}")
        # For GPTQ using transformers, we load the model first and then quantize
        # Device placement is handled by from_pretrained
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map=device_map,
                offload_folder="offload", 
                offload_state_dict=args.cpu_offload,
                trust_remote_code=True,
                quantization_config=gptq_config # Pass the config here
            )
            logging.info("Model loaded and quantization process initiated via GPTQConfig.")
        except Exception as e:
            logging.error(f"Error loading model or initiating GPTQ quantization: {e}", exc_info=True)
            return

        logging.info("GPTQ Quantization completed implicitly during model loading.")

        # --- GPTQ Saving --- #
        # Transformers' save_pretrained handles GPTQ saving
        output_dir = os.path.join(args.model_path, f"GPTQ_{args.bits}bit_CPU") # Add CPU indication
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving GPTQ quantized model to subdirectory: {output_dir}")
        try:
            # save_pretrained will save the model shards, config with quant info, tokenizer etc.
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir) 
            logging.info(f"GPTQ quantized model and tokenizer saved successfully to {output_dir}")

            custom_code_files = glob.glob(os.path.join(args.model_path, '*.py'))
            if custom_code_files:
                logging.info(f"Copying custom code files to {output_dir}: {custom_code_files}")
                for py_file in custom_code_files:
                     shutil.copy2(py_file, output_dir)

        except Exception as e:
            logging.error(f"Error during GPTQ model saving: {e}", exc_info=True)
            return

    else:
        # This case should not be reachable due to the default logic
        logging.error(f"Internal error: Unsupported quantization method determined.")
        return

    logging.info("Quantization script finished successfully.")

if __name__ == '__main__':
    main()
