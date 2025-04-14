# Project Context

## Project Overview

This project provides a Dockerized environment for quantizing large language models using either the AWQ (Activation-aware Weight Quantization) method (4-bit) or the GPTQ method (8-bit). It aims to simplify the quantization workflow, particularly for users with capable hardware (significant CPU RAM and modern GPUs). It supports both CPU-only execution (via `--force_cpu`) and GPU-accelerated quantization using `accelerate`'s explicit CPU offload (`cpu_offload`) for efficient memory management on single-GPU setups. A key feature is its ability to handle GPTQ quantization for models with extremely long context lengths (e.g., 128k) by temporarily modifying the model configuration during calibration to use a shorter sequence length, thus avoiding common out-of-memory errors.

**Important Note:** Quantizing large language models (especially 60GB+ or using GPTQ) is inherently resource-intensive. This tool automates several complex steps but does not eliminate the need for substantial CPU RAM (potentially ~1.5x the original model size during peak quantization) and GPU VRAM (even with offloading). It is best suited for users who understand these requirements and possess hardware capable of meeting them.

## Architecture

- **Docker Container:** Encapsulates all dependencies (`transformers`, `autoawq`, `optimum[auto-gptq]`, `torch`, `accelerate`, `datasets`, etc.) and the quantization script.
- **Quantization Script (`quantize.py`):** Python script that handles model loading, quantization logic, and saving. Uses command-line flags (`--awq` or `--gptq`) for method selection. Implements explicit CPU offload via `accelerate.cpu_offload` when a GPU is detected (and not overridden by `--force_cpu`). For GPTQ, it includes logic to temporarily modify the model's sequence length configuration (`max_position_embeddings` or similar) based on the `--seq_len` argument to manage memory during calibration, restoring the original value before saving.
- **Execution:** Runs on GPU by default if available, leveraging CPU RAM for offloading model layers not currently in computation. Can be forced to run entirely on CPU using `--force_cpu`.
- **Input/Output:** Mounts a host directory containing the pre-trained model into the container (`/models`). Creates a new subdirectory within the mounted path named `<original_model_name>-AWQ` or `<original_model_name>-GPTQ`. Saves quantized outputs (weights, configs, index file, custom code) *inside this subdirectory*, with filenames also having the corresponding `-AWQ` or `-GPTQ` suffix.

## Key Components

- **`Dockerfile`:** Defines the Docker image build process, including dependency installation and setting `TORCH_CUDA_ARCH_LIST` for potentially faster CUDA kernels.
- **`requirements.txt`:** Lists Python package dependencies.
- **`quantize.py`:** The core script. Parses arguments (model path, method flags, bits, `--seq_len`, `--force_cpu`, etc.), loads model/tokenizer, determines execution device, applies explicit CPU offload if using GPU, performs AWQ or GPTQ (including the sequence length config modification/restoration for GPTQ), and saves outputs with suffixes.
- **`README.md`:** Provides usage instructions, build steps, examples, and troubleshooting tips, focusing on the sequence length workaround for large context models.
- **`project_context.md`:** (This file) Overview of the project architecture and approach.

## Conventions

- AWQ default: 4-bit, group size 128, zero point true.
- GPTQ default: 8-bit, wikitext2 dataset, group size 128.
- Method selection via mutually exclusive flags: `--awq` (default) or `--gptq`.
- Execution device determined automatically (GPU if available), override with `--force_cpu`.
- Explicit CPU offload used when running on GPU.
- Sequence length for calibration controlled by `--seq_len` (default 8192), crucial for GPTQ on large-context models.
- Output files (weights, relevant configs) for both AWQ and GPTQ are saved with suffixes (`-AWQ` or `-GPTQ`) in the original model directory.
- Output files for both AWQ and GPTQ are saved inside a new subdirectory `<original_model_name>-AWQ` or `<original_model_name>-GPTQ` within the original model path.
- Filenames within the output subdirectory also have the corresponding `-AWQ` or `-GPTQ` suffix applied (e.g., `model-AWQ.safetensors`).
- Logging to both console and `quantize.log` file within the container (path relative to where the script is run inside the container).

## Dependencies

- Python 3.x
- Docker
- PyTorch (with CUDA support if GPU is used)
- AutoAWQ (`autoawq`)
- Transformers (`transformers`)
- Optimum (`optimum[auto-gptq]`)
- Accelerate (`accelerate`)
- Datasets (`datasets`)
- Safetensors (`safetensors`)
- PSUtil (`psutil`)
