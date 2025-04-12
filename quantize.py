import argparse
import json
import os
import torch
import tempfile
import shutil
import glob
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

def main():
    """
    Main function: Parses arguments, validates, selects method (AWQ/GPTQ),
    performs quantization ON CPU, and saves results.
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

    if quantization_method == 'awq':
        # --- AWQ Quantization (CPU Only) --- # 
        
        # Merge default AWQ config with custom config
        quant_config = {**DEFAULT_AWQ_CONFIG, **custom_config}
        quant_config['w_bit'] = args.bits # Ensure bits argument overrides config
        logging.info(f"Using AWQ quantization config: {quant_config}")

        logging.info(f"Loading model for AWQ from: {args.model_path} onto CPU")
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                safetensors=True,
                device_map=device # Explicitly load to CPU
            )
            logging.info("Pre-trained model loaded successfully for AWQ onto CPU.")
        except Exception as e:
            logging.error(f"Error loading model for AWQ from '{args.model_path}': {e}", exc_info=True)
            return

        logging.info("Starting AWQ quantization...")
        try:
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

        logging.info(f"Loading model for GPTQ from: {args.model_path} onto CPU")
        # For GPTQ using transformers, we load the model first and then quantize
        # Device placement is handled by from_pretrained
        device_map = device # Force CPU
        logging.info(f"Using device '{device}' for CPU GPTQ quantization.")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map=device_map, 
                trust_remote_code=True,
                quantization_config=gptq_config # Pass the config here
            )
            logging.info("Model loaded onto CPU and quantization process initiated via GPTQConfig.")
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
