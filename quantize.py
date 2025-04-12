import argparse
import json
import os
import torch
import tempfile
import shutil
import glob
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantize.log"), # Log to a file
        logging.StreamHandler()             # Log to console
    ]
)

# Define a default quantization configuration
DEFAULT_QUANT_CONFIG = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}

DEFAULT_QUANT_CONFIG_STR = json.dumps(DEFAULT_QUANT_CONFIG)

# Define approximate max memory per GPU (leaving some headroom)
# Assumes 5 GPUs are passed through (e.g., 4x 4090, 1x 3090)
# Adjust indices and values if your system maps GPUs differently or has different limits
max_memory_map = {
    0: "21GiB",  # GPU 0 (RTX 4090)
    1: "21GiB",  # GPU 1 (RTX 4090)
    2: "21GiB",  # GPU 2 (RTX 3090)
    3: "21GiB",  # GPU 3 (RTX 4090)
    4: "21GiB"   # GPU 4 (RTX 4090)
}

def main():
    """
    Main function to parse arguments, load model/tokenizer, perform AWQ quantization,
    and save the quantized model files (*-AWQ.safetensors, quant_config-AWQ.json)
    alongside the original model files.
    """
    parser = argparse.ArgumentParser(description="Quantize a model using AutoAWQ and save results in the model directory.")
    parser.add_argument('--model_path', type=str, required=False,
                        default='/models',
                        help='Path to the directory containing the pre-trained model and tokenizer (e.g., HF model format). Output files will also be saved here.')
    parser.add_argument('--zero_point', type=bool, required=False,
                        default=True,
                        help='Whether to use zero point quantization.')
    parser.add_argument('--q_group_size', type=int, required=False,
                        default=128,
                        help='Quantization group size.')
    parser.add_argument('--w_bit', type=int, required=False,
                        default=4,
                        help='Weight bit width.')
    parser.add_argument('--version', type=str, required=False,
                        default="GEMM",
                        help='Quantization version.')
    # Correctly escape the inner JSON example within the help string for the JSON parser
    parser.add_argument('--quant_config', type=str, required=False,
                        default=DEFAULT_QUANT_CONFIG_STR,
                        help='JSON string with quantization configuration (e.g., \'{"w_bit": 4, "q_group_size": 128, "zero_point": true, "version": "GEMM"}\')')

    args = parser.parse_args()
    logging.info(f"Starting quantization process with args: {args}")

    # --- Input Validation ---
    if not os.path.isdir(args.model_path):
        logging.error(f"Model path '{args.model_path}' not found or is not a directory.")
        return
    # Check if any safetensors file exists - AWQ often works directly with HF format dirs
    # but safetensors files might be part of it.
    safetensor_files = glob.glob(os.path.join(args.model_path, '*.safetensors')) # Use glob for better matching
    if not safetensor_files:
         # This might be okay if it's a standard HF model directory without explicit .safetensors at the top level
         logging.info(f"No top-level '.safetensors' files found in '{args.model_path}'. Assuming standard HF directory structure.")
    else:
        logging.info(f"Found safetensors files in '{args.model_path}': {safetensor_files}")

    # --- Ensure Output Directory (model_path) is Writable ---
    # A simple check, though Docker volume permissions are key
    if not os.access(args.model_path, os.W_OK):
        logging.warning(f"The target directory '{args.model_path}' might not be writable from within the script's context. Ensure the Docker volume mount has Read/Write permissions.")

    # --- Parse Quantization Config ---
    try:
        # Load the JSON string directly
        quant_config = json.loads(args.quant_config)
        logging.info(f"Using quantization config: {quant_config}")
        # Validate required keys (example for AWQ)
        required_keys = ["w_bit", "q_group_size", "zero_point", "version"]
        if not all(key in quant_config for key in required_keys):
             logging.warning(f"Quant config might be missing standard AWQ keys (w_bit, q_group_size, zero_point, version). Provided: {quant_config.keys()}")

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing quant_config JSON: {e}")
        logging.error(f"Received: {args.quant_config}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during quant_config processing: {e}")
        return

    # --- Define Quantization Configuration Early --- 
    # Use wikitext2 for calibration instead of the default pile-val-backup
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }
    logging.info(f"Using quantization config: {quant_config}")

    # --- Device Selection --- # Force CPU
    device = "cpu" 
    logging.info(f"Forcing device: {device}")
    max_memory_map = None # Not used for CPU

    # --- Load Model and Tokenizer --- # Moved Tokenizer loading earlier
    logging.info(f"Loading tokenizer from: {args.model_path}")
    try:
        # Load tokenizer using the provided model path
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizer from '{args.model_path}': {e}", exc_info=True)
        return # Exit if tokenizer loading fails

    logging.info(f"Loading model from: {args.model_path}") # Separate model loading log
    try:
        # Load model for AWQ quantization using the provided model path
        # AutoAWQForCausalLM.from_pretrained handles loading safetensors if present
        # and mapping to device.
        model = AutoAWQForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            safetensors=True, # Prefer safetensors loading if available
        )
        logging.info("Pre-trained model loaded successfully.")

    except Exception as e:
        # Provide more context in error logging
        logging.error(f"Error loading model from '{args.model_path}': {e}", exc_info=True) # Log traceback for model only
        return # Exit if model loading fails

    # --- Quantization --- 
    logging.info(f"Starting quantization with config: {quant_config}")
    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config, # Pass the pre-defined config
        )
        logging.info("Quantization completed successfully.")
    except Exception as e:
        # Log the specific error during quantization
        logging.error(f"Error during quantization: {e}", exc_info=True) # Log traceback
        return

    # --- Save Quantized Model to Temp Dir, then Rename and Copy ---
    logging.info(f"Saving quantized model temporarily before moving to: {args.model_path}")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Using temporary directory: {temp_dir}")

            # Save quantized model and config to the temporary directory
            model.save_quantized(temp_dir, safetensors=True)
            logging.info(f"Quantized model temporarily saved to {temp_dir}")

            # Define the final destination directory
            target_dir = args.model_path

            # --- Rename and Copy .safetensors file(s) ---
            # Find all .safetensors files in the temp directory
            temp_safetensor_files = glob.glob(os.path.join(temp_dir, '*.safetensors'))
            if not temp_safetensor_files:
                logging.error(f"No '.safetensors' files found in temporary directory '{temp_dir}' after saving. Cannot proceed.")
                return

            for temp_sf_path in temp_safetensor_files:
                base_name, ext = os.path.splitext(os.path.basename(temp_sf_path))
                new_name = f"{base_name}-AWQ{ext}"
                dest_path = os.path.join(target_dir, new_name)
                logging.info(f"Copying '{os.path.basename(temp_sf_path)}' to '{dest_path}'")
                shutil.copy2(temp_sf_path, dest_path) # copy2 preserves metadata

            # --- Rename and Copy quant_config.json ---
            temp_config_path = os.path.join(temp_dir, "quant_config.json")
            if os.path.exists(temp_config_path):
                new_config_name = "quant_config-AWQ.json"
                dest_config_path = os.path.join(target_dir, new_config_name)
                logging.info(f"Copying 'quant_config.json' to '{dest_config_path}'")
                shutil.copy2(temp_config_path, dest_config_path)
            else:
                logging.warning(f"'quant_config.json' not found in temporary directory '{temp_dir}'. It might not be strictly required for loading but is usually expected.")

            # --- Skip saving tokenizer --- 
            # The original tokenizer files in args.model_path will be used.
            logging.info(f"Skipping tokenizer save. Original tokenizer in {args.model_path} should be used.")

        logging.info(f"Quantized model files (with -AWQ suffix) saved successfully to {target_dir}.")

    except Exception as e:
        logging.error(f"Error saving or moving quantized model files: {e}", exc_info=True) # Log traceback
        return

    logging.info("Quantization script finished successfully.")

if __name__ == "__main__":
    main()
