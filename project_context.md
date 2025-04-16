# Project Context

## Project Overview

This project provides a Dockerized environment for quantizing large language models using AWQ (typically 4-bit) or GPTQ (typically 8-bit) methods. It aims to simplify the quantization workflow by providing a unified command-line interface and encapsulating dependencies.

A core design principle is **memory efficiency** for single-GPU setups. This is achieved by:
*   Loading the base model initially to CPU RAM (`device_map="cpu"`).
*   Providing control over the calibration sequence length (`--seq_len`) to manage peak VRAM usage during quantization.
*   Logging peak VRAM usage for monitoring and tuning.

This tool is primarily intended for users who need to quantize custom models locally and possess hardware with sufficient CPU RAM and GPU VRAM for the chosen model and method.

## Architecture

- **Docker Container:** Encapsulates Python, PyTorch, CUDA (if applicable), quantization libraries (`autoawq`, `optimum[auto-gptq]`), `transformers`, `accelerate`, and other dependencies.
- **Quantization Script (`quantize.py`):** The main entry point. Parses arguments, determines the execution device (GPU preferred, CPU fallback), loads the model using `device_map="cpu"`, orchestrates the selected quantization process (AWQ for text, GPTQ via Optimum for text, or GPTQ via LLM Compressor for vision) applying the `--seq_len` parameter (for text models), monitors peak VRAM, and saves the results.
- **VRAM Monitoring (`utils/vram_monitor.py`):** A utility class run in a background thread to track peak GPU memory usage during quantization.
- **Execution:** Runs on an available NVIDIA GPU by default. Can be forced to run entirely on CPU using `--force_cpu`. The quantization libraries handle internal layer processing on the target device.
- **Input/Output:** Expects a host directory containing the pre-trained model mounted into the container at `/models`. Creates a new subdirectory named `METHOD-BITRATE` (e.g., `AWQ-4bit`, `GPTQ-8bit`) within the input model directory and saves the quantized model, tokenizer/processor files, configuration, and `quantization_report.log` there.
- **Status Note:** GPTQ quantization for vision models using `llmcompressor` is currently **blocked** by an internal library error (`AttributeError: module 'torch' has no attribute 'OutOfMemoryError'`).

## Key Components

- **`Dockerfile`:** Defines the Docker image, installing dependencies from `requirements.txt`.
- **`requirements.txt`:** Lists Python package dependencies.
- **`quantize.py`:** The core quantization script handling argument parsing, model loading (`device_map="cpu"`), AWQ/GPTQ execution logic (using `--seq_len`), VRAM monitoring invocation, and saving results to the `METHOD-BITRATE` directory.
- **`utils/vram_monitor.py`:** Contains the `VRAMMonitor` class for peak memory tracking.
- **`README.md`:** User-facing documentation covering setup, usage, examples, command-line arguments, memory considerations, and troubleshooting.
- **`devlog.md`:** Internal development log tracking history, decisions, status, and next steps.
- **`project_context.md`:** (This file) High-level static overview of the project architecture, components, and conventions.
- **AutoAWQ (`autoawq`)**
- **Transformers (`transformers`)**
- **Optimum (`optimum[auto-gptq]`)**
- **LLM Compressor (`llm-compressor[torch]`)**
- **Accelerate (`accelerate`)**
- **Datasets (`datasets`)**
- **Safetensors (`safetensors`)**

## Conventions

- AWQ default: 4-bit.
- GPTQ default: 8-bit.
- Default calibration sequence length (`--seq_len`): 2048.
- Method selection via mutually exclusive flags: `--awq` (default if neither specified) or `--gptq`.
- Execution device determined automatically (GPU preferred), override with `--force_cpu`.
- Base model loading uses `device_map="cpu"` strategy.
- Output is saved in a subdirectory named `METHOD-BITRATE` (e.g., `AWQ-4bit`, `GPTQ-8bit`) inside the input model directory.
- Peak VRAM usage is logged to `quantization_report.log` within the output subdirectory.

## Dependencies

- Python 3.x
- Docker
- PyTorch (with CUDA support if GPU is used)
- AutoAWQ (`autoawq`)
- Transformers (`transformers`)
- Optimum (`optimum[auto-gptq]`)
- LLM Compressor (`llm-compressor[torch]`)
- Accelerate (`accelerate`)
- Datasets (`