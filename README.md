# Model Quantizer Docker Container

This repository contains the necessary files to build a Docker container capable of quantizing large language models using AWQ (Activation-aware Weight Quantization) or GPTQ methods **on the CPU**.

## Prerequisites

*   Docker installed on your system.
*   NVIDIA Container Toolkit installed (if using GPU for GPTQ).
*   Sufficient disk space for the Docker image and models.
*   A pre-trained model in Hugging Face format (containing configuration files like `config.json`, tokenizer files, and model weights, preferably in `safetensors` format).

## Building the Docker Image

1.  Clone this repository:

    ```bash
git clone <repository_url>
cd <repository_directory>
```

2.  Build the Docker image:

    ```bash
docker build -t quantizer .
```

## Running the Quantization Script

The `quantize.py` script inside the container performs the quantization. You need to mount the directory containing your model into the container (e.g., to `/models`) and potentially specify other parameters.

### Command-Line Arguments for `quantize.py`

*   `--model_path`: (Optional) Path *inside the container* to the model directory. Default: `/models`.
*   `--awq`: (Flag) Use AWQ quantization (4-bit). This is the default if neither `--awq` nor `--gptq` is specified.
*   `--gptq`: (Flag) Use GPTQ quantization (8-bit). **Warning:** This is extremely slow on the CPU.
*   `--bits`: (Optional) Number of bits for quantization. Default: 4 for `awq`, 8 for `gptq`. Currently, only these defaults are fully supported by the script's logic.
*   `--quant_config`: (Optional) JSON string for custom quantization config, merging with method defaults (e.g., `'{"q_group_size": 64}'`).
*   `--gptq_dataset`: (Optional, for GPTQ only) Dataset name from Hugging Face Datasets for GPTQ calibration. Default: `wikitext2`.
*   `--gptq_group_size`: (Optional, for GPTQ only) Group size for GPTQ quantization. Default: 128.
*   `--gptq_desc_act`: (Optional, flag, for GPTQ only) Use descending activation order for GPTQ. Default: False.

### Example Docker Run Commands

Replace `/path/to/your/host/models/YourModelName` with the actual path to your model directory on the host machine.

#### 1. AWQ Quantization (4-bit, CPU - Default)

This uses the default setting (`--awq` is implied if neither flag is present).

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer
```

Or explicitly with the flag:

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer \
  python quantize.py --model_path /models --awq
```

*   Output: Quantized files (`*-AWQ.safetensors`, `quant_config-AWQ.json`, modified `model.safetensors.index.json`) will be saved *directly* within the mounted `/path/to/your/host/models/YourModelName` directory.

#### 2. GPTQ Quantization (8-bit, CPU Only - Very Slow)

This requires specifying the `--gptq` flag.

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer \
  python quantize.py --model_path /models --gptq
```

*   Optional GPTQ parameters:

```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  quantizer \
  python quantize.py --model_path /models --gptq --gptq_dataset c4 --gptq_group_size 64 --gptq_desc_act
```

*   Output: A new subdirectory named `GPTQ_8bit_CPU` will be created *inside* the mounted `/path/to/your/host/models/YourModelName` directory. This subdirectory will contain the full GPTQ quantized model, tokenizer, and config files.

## Notes

*   Ensure the host model directory you mount (`/path/to/your/host/models/YourModelName`) is writable by the Docker container (`:rw`).
*   Quantization can be memory-intensive. Monitor resource usage.
*   Both AWQ and GPTQ quantization methods are **forced to run on the CPU** in this script. GPTQ on CPU is extremely slow.
*   Logs are written to `quantize.log` inside the container and also printed to the console.

## Unraid Specific Setup

When configuring this container via the Unraid GUI:

1.  **Repository:** `quantizer` (or the name you used during the build).
2.  **Extra Parameters:** Add `python quantize.py` followed by any desired flags (e.g., `python quantize.py --gptq` or `python quantize.py --awq --quant_config '{"q_group_size": 64}'`). If no flags like `--awq` or `--gptq` are added, it will default to AWQ.
3.  **GPU:** GPU passthrough is **not required** as the script only uses the CPU.
4.  **Volume Mappings:** Map your host model directory (e.g., `/mnt/user/models/YourModelName`) to the container path (e.g., `/models`) with Read/Write access.
