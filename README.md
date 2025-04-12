# LLM Quantizer Docker (CPU-Focused AWQ/GPTQ)

This project provides a Dockerized environment for quantizing large language models using either AWQ (Activation-aware Weight Quantization) or GPTQ (Generative Pre-trained Transformer Quantization) methods.

**IMPORTANT:** This tool is specifically designed and configured for **CPU-based quantization**. While the base Docker image includes CUDA libraries, the quantization script (`quantize.py`) explicitly forces execution on the CPU for both AWQ and GPTQ methods. The primary goal is to create smaller, potentially faster-inferencing models that are compatible with CPU execution environments.

## Features

*   Quantize models using AWQ (4-bit default) or GPTQ (8-bit default).
*   Runs entirely within a Docker container, simplifying dependency management.
*   **CPU execution is enforced** for compatibility and accessibility.
*   Simple command-line interface.
*   Handles model loading, quantization, and saving.
*   AWQ outputs preserve the original model files, adding suffixed files and a separate index.
*   GPTQ outputs are saved to a dedicated subdirectory.

## Prerequisites

*   Docker installed on your system.
*   Sufficient disk space for the Docker image and models.
*   A pre-trained model in Hugging Face format (containing `config.json`, tokenizer files, and model weights, preferably in `safetensors` format).

## Building the Docker Image

1.  Clone this repository:
    ```bash
    git clone https://github.com/jwmacd/quantizer_docker.git # Replace with actual URL if different
    cd quantizer_docker
    ```

2.  Build the Docker image:
    ```bash
    # Build locally with a simple tag
    docker build -t quantizer .
    
    # Optional: Tag for pushing to a registry (e.g., GHCR)
    # docker tag quantizer ghcr.io/your-username/your-repo:latest
    # docker push ghcr.io/your-username/your-repo:latest
    ```

## Running the Quantization Script (Usage)

The container uses an `ENTRYPOINT` which defaults to running `python /app/quantize.py`. You provide arguments *after* the image name in your `docker run` command, which are passed directly to the script.

You **must** mount the host directory containing your model into the container at the path `/models` using the `-v` flag.

### Command-Line Arguments for `quantize.py`

*   `--model_path`: Path *inside the container* to the model directory. **Default: `/models`** (You usually don't need to specify this if you mount your model to `/models`).
*   `--awq`: (Flag) Use AWQ quantization (4-bit). **This is the default behavior** if neither `--awq` nor `--gptq` is specified.
*   `--gptq`: (Flag) Use GPTQ quantization (8-bit). **Warning:** This is extremely slow on the CPU.
*   `--bits`: (Optional) Number of bits for quantization. Default: 4 for `awq`, 8 for `gptq`. Currently, only these defaults are supported by the script.
*   `--quant_config`: (Optional) JSON string for custom quantization config, merging with method defaults (e.g., `'{"q_group_size": 64}'`). Note the escaped quotes if running directly in bash.
*   `--gptq_dataset`: (Optional, for GPTQ only) Dataset name from Hugging Face Datasets for GPTQ calibration. Default: `wikitext2`.
*   `--gptq_group_size`: (Optional, for GPTQ only) Group size for GPTQ quantization. Default: 128.
*   `--gptq_desc_act`: (Optional, flag, for GPTQ only) Use descending activation order for GPTQ. Default: False.

### Example Docker Run Commands

Replace `/path/to/your/host/models/YourModelName` with the actual path to your model directory on the host machine.

#### 1. AWQ Quantization (4-bit, CPU - Easiest/Default)

Since `--model_path` defaults to `/models` and `--awq` is the default method, you only need to provide the volume mount:

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer
```

*   Output: New AWQ files (`*-AWQ.safetensors`, `quant_config-AWQ.json`, `model-AWQ.safetensors.index.json`) will be saved *directly* within the mounted `/path/to/your/host/models/YourModelName` directory, alongside the original files.

#### 2. GPTQ Quantization (8-bit, CPU Only - Very Slow)

Requires specifying the `--gptq` flag after the image name:

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer \
  --gptq
```

*   Optional GPTQ parameters:

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer \
  --gptq --gptq_dataset c4 --gptq_group_size 64 --gptq_desc_act
```

*   Output: A new subdirectory named `GPTQ_8bit_CPU` will be created *inside* the mounted `/path/to/your/host/models/YourModelName` directory. This subdirectory will contain the full GPTQ quantized model, tokenizer, and config files.

## Notes

*   Ensure the host model directory you mount is writable (`:rw`).
*   Quantization can be memory-intensive. Monitor resource usage.
*   GPTQ on CPU is extremely slow and primarily included for completeness or specific testing scenarios.
*   Logs are written to `quantize.log` inside the container's `/app` directory and also printed to the console.

## Unraid Specific Setup

When configuring this container via the Unraid GUI:

1.  **Repository:** Use the tag you built locally (e.g., `quantizer`) or the tag you pushed/pulled from a registry (e.g., `ghcr.io/jwmacd/quantizer_docker:latest`).
2.  **Extra Parameters / Post Arguments:** Add only the *arguments* for the script (e.g., leave blank for default AWQ, or add `--gptq` for GPTQ, or add `--awq --quant_config '{"q_group_size": 64}'` for custom AWQ). Do **not** include `python quantize.py` here, as the `ENTRYPOINT` handles that.
3.  **GPU:** GPU passthrough is **not required** as the script only uses the CPU.
4.  **Volume Mappings:** Map your host model directory (e.g., `/mnt/user/models/YourModelName`) to the container path `/models` with Read/Write access.
