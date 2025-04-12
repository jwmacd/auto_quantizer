# Project Context

## Project Overview

This project provides a Dockerized environment for quantizing large language models using either the AWQ (Activation-aware Weight Quantization) method (4-bit) or the GPTQ method (8-bit). The primary goal is to create smaller, faster-inferencing models compatible with CPU execution, although GPTQ on CPU is notably slow.

## Architecture

- **Docker Container:** Encapsulates all dependencies (`transformers`, `autoawq`, `optimum[gptq]`, `torch`, `datasets`, etc.) and the quantization script.
- **Quantization Script (`quantize.py`):** Python script that handles model loading, quantization logic for both AWQ and GPTQ, and saving the quantized model. It uses command-line flags (`--awq` or `--gptq`) to select the method.
- **CPU Execution:** The script is configured to **force CPU usage** for both AWQ and GPTQ methods.
- **Input/Output:** Mounts a host directory containing the pre-trained model into the container (`/models`). Saves AWQ outputs directly in the model directory (with `-AWQ` suffix) and GPTQ outputs into a new subdirectory (`GPTQ_8bit_CPU`).

## Key Components

- **`Dockerfile`:** Defines the Docker image build process, including dependency installation.
- **`requirements.txt`:** Lists Python package dependencies.
- **`quantize.py`:** The core script. Parses arguments (like model path, `--awq`, `--gptq`, `--bits`, `--quant_config`, GPTQ-specific params), loads the model and tokenizer, applies the chosen quantization method (AWQ via `AutoAWQForCausalLM` or GPTQ via `AutoModelForCausalLM` with `GPTQConfig`), and saves the output.
- **`README.md`:** Provides usage instructions, build steps, and examples.
- **`project_context.md`:** (This file) Overview of the project.

## Conventions

- AWQ default: 4-bit, group size 128, zero point true.
- GPTQ default: 8-bit, wikitext2 dataset, group size 128.
- Method selection via mutually exclusive flags: `--awq` (default) or `--gptq`.
- Forced CPU execution for both methods.
- AWQ output files are saved in the model root directory:
  - Weight files (`.safetensors`) are renamed to `*-AWQ.safetensors`.
  - `quant_config.json` is renamed to `quant_config-AWQ.json`.
  - The generated index file is saved as `model-AWQ.safetensors.index.json` (preserving the original index file).
- GPTQ output files are saved in a dedicated `GPTQ_8bit_CPU` subdirectory.
- Logging to both console and `quantize.log` file within the container.

## Dependencies

- Python 3.x
- Docker
- PyTorch (with CUDA support if GPU is used)
- AutoAWQ (`autoawq`)
- Transformers (`transformers`)
- Safetensors (`safetensors`)
- Accelerate (`accelerate`)
- Optimum (`optimum`)
- Datasets (`datasets`)
