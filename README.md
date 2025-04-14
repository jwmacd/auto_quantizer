# LLM Auto Quantizer (CPU/GPU AWQ/GPTQ Quantization)

This project provides a Dockerized environment for **performing quantization of large language models** using either AWQ (4-bit) or GPTQ (8-bit) methods. It supports both CPU-only and GPU-accelerated quantization with explicit CPU offload for memory efficiency, making it suitable for various hardware configurations. It also includes a workaround for quantizing models with very large context lengths (like 128k) that might otherwise cause out-of-memory errors.

## Target Audience & Limitations

While aiming for ease of use, quantizing large language models remains a resource-intensive task. This tool is best suited for:

*   **Users with Capable Hardware:** Requires significant CPU RAM (peak usage potentially ~1.5x original model size during GPTQ) and a modern NVIDIA GPU (especially for reasonable GPTQ speed). Quantizing 60GB+ models challenges even high-end consumer/prosumer systems.
*   **Users Needing Specific Features:** Offers explicit CPU offload, handling for very long context models via sequence length control, and unified AWQ/GPTQ interface.
*   **Users Quantizing Locally:** Useful for those who need to quantize custom models or fine-tunes not available pre-quantized online.

**Limitations:**

*   **Does Not Eliminate Hardware Needs:** Automates the workflow but cannot overcome fundamental RAM/VRAM limitations. Users must ensure their hardware meets the demands of the chosen model and quantization method.
*   **Requires Some Understanding:** Users benefit from understanding concepts like sequence length impact, dataset choice, and resource trade-offs to use the tool effectively and troubleshoot.
*   **Single GPU Focus:** Designed for single-GPU setups with CPU offload; does not support multi-GPU distribution.

**For users without sufficient local hardware, using pre-quantized models from sources like Hugging Face or leveraging cloud computing platforms is often a more practical approach.**

## Overview

**Purpose:** This tool automates the process of performing AWQ (4-bit) or GPTQ (8-bit) model quantizations, producing standard quantized model formats compatible with common inference frameworks.

**Key Benefits (for users with adequate hardware):**
* **Flexible Execution:** Run quantization on CPU or utilize a GPU with explicit CPU offload.
* **Handles Large Context:** Successfully quantizes models with extremely long context windows (e.g., 128k) using a sequence length modification technique for GPTQ.
* **Simplified Setup:** Docker encapsulates all dependencies.
* **Multiple Methods:** Supports both AWQ (4-bit, faster) and GPTQ (8-bit, higher quality but slower on CPU).
* **Standard Output:** Creates quantized models compatible with standard inference tools.
* **Consistent Output Format:** Both methods preserve original files and add suffixed versions in the source directory.

## Technical Details

| Method | Default Bits | Speed (GPU) | Speed (CPU) | Output Location | Use Case |
|--------|-------------|-------------|-------------|-----------------|----------|
| AWQ    | 4-bit       | Fast        | Reasonable  | Original directory with `-AWQ` suffix | Smaller model files for faster inference |
| GPTQ   | 8-bit       | Moderate    | VERY Slow   | Original directory with `-GPTQ` suffix | Better quality but with slightly larger model files |

## Prerequisites

* Docker installed on your system
* NVIDIA GPU with appropriate drivers (highly recommended for GPTQ, beneficial for AWQ) OR sufficient CPU resources.
* Sufficient disk space for Docker image (~5-10GB depending on CUDA version) and models.
* A pre-trained model in Hugging Face format (must contain `config.json`, tokenizer files, and model weights preferably in `safetensors` format).
* Sufficient CPU RAM (especially when using CPU offload, >64GB recommended for large models).

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jwmacd/auto_quantizer.git
   cd auto_quantizer
   ```

2. **Build the Docker image**
   ```bash
   docker build -t auto_quantizer .
   ```

3. **Run AWQ quantization (default, 4-bit, GPU recommended)**
   ```bash
   # Using GPU (recommended)
   docker run --rm -it --gpus all \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer

   # Using CPU only (add --force_cpu)
   docker run --rm -it \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --force_cpu
   ```

4. **Run GPTQ quantization (8-bit, GPU HIGHLY recommended)**
   ```bash
   # Using GPU (HIGHLY recommended)
   docker run --rm -it --gpus all \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --gptq

   # Using CPU only (EXTREMELY SLOW - not recommended)
   docker run --rm -it \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --gptq --force_cpu
   ```

## Detailed Usage

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to model directory inside container | `/models` |
| `--awq` | Use AWQ quantization (4-bit) | Default if neither specified |
| `--gptq` | Use GPTQ quantization (8-bit) | - |
| `--bits` | Number of bits for quantization | 4 (AWQ), 8 (GPTQ) |
| `--quant_config` | JSON string for custom quantization config (merges with defaults) | `{}` |
| `--force_cpu` | Force CPU execution even if GPU is available | `false` |
| `--batch_size` | Batch size for quantization calibration | `1` |
| `--seq_len` | **Crucial for GPTQ on long-context models.** Max sequence length for calibration data. Reduces VRAM/RAM usage during quantization by temporarily shortening the model's context length view. | `8192` |
| `--gptq_dataset` | Dataset for GPTQ calibration (from Hugging Face Datasets) | `wikitext2` |
| `--gptq_group_size` | Group size for GPTQ quantization | 128 |
| `--gptq_desc_act` | Use descending activation order for GPTQ | `false` |

### AWQ Examples

**Basic AWQ (4-bit, GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer
```

**Custom AWQ configuration (GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --awq --quant_config '{"q_group_size": 64, "zero_point": false}'
```

### GPTQ Examples

**Basic GPTQ (8-bit, GPU HIGHLY recommended):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq
```

**GPTQ for a model with very long context (e.g., 128k) on GPU:**
```bash
# Use --seq_len to reduce memory peak during calibration
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/Qwen2-7B-Instruct-128k:/models:rw \
  auto_quantizer \
  --gptq --seq_len 8192
```

**Custom GPTQ configuration (GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq --gptq_dataset c4 --gptq_group_size 64 --gptq_desc_act
```

## Output Files

Both AWQ and GPTQ methods preserve all original model files in the source directory (`/models` inside the container). They create a new subdirectory within the source directory and save the quantized outputs there.

**Output Subdirectory:**
- A new directory named `<original_model_name>-AWQ` or `<original_model_name>-GPTQ` is created inside the source model path.
  - Example: If `model_path` is `/models/MyModel`, AWQ output goes into `/models/MyModel/MyModel-AWQ/`.

**Files inside the Output Subdirectory:**

### AWQ Output (`<model_path>/<model_name>-AWQ/`)
* `model-<...>-AWQ.safetensors`: Quantized model weight shards with suffix.
* `quant_config-AWQ.json`: Quantization configuration used, with suffix.
* `model-AWQ.safetensors.index.json`: Index for sharded models, with suffix.
* `<custom_code>-AWQ.py`: Any original `.py` files copied with suffix.
* Other necessary original files (like `config.json`, tokenizer files) are **not** copied here by default.

### GPTQ Output (`<model_path>/<model_name>-GPTQ/`)
* `model-<...>-GPTQ.safetensors` or `model-<...>-GPTQ.bin`: Quantized model weight shards with suffix.
* `config-GPTQ.json`: Model configuration (original seq len restored), with suffix.
* `model-GPTQ.safetensors.index.json`: Index for sharded models, with suffix.
* `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, etc.: Original tokenizer files are copied here *without* suffix.
* `<custom_code>-GPTQ.py`: Any original `.py` files copied with suffix.

## Resource Requirements

* **GPU VRAM:** Depends heavily on model size and `--seq_len` (for GPTQ). Explicit CPU offload helps, but peak computation still requires significant VRAM (e.g., ~18-20GB might be needed for a 7B model with `--seq_len 8192`).
* **CPU RAM:** Peak usage during quantization might be ~1.5x the original model size (e.g., ~90GB for a 60GB model). After quantization, needs enough RAM for the quantized model (~0.5x original size for 8-bit, ~0.25x for 4-bit) plus overhead. >64GB recommended for 7B+ models, potentially much more for larger models.
* **Disk:** Space for original model + quantized output subdirectory + Docker image.
* **Time:** AWQ is relatively fast on GPU. GPTQ is slower on GPU and *extremely* slow on CPU. Expect hours for large models using GPU, potentially days using CPU.

## Unraid Integration

When configuring this container via the Unraid GUI:

1. **Repository:** Use `auto_quantizer` or your custom tag.
2. **Extra Parameters:** Add script arguments (e.g., `--gptq --seq_len 4096`).
3. **GPU:** Enable GPU passthrough if desired (highly recommended for GPTQ).
4. **Volume Mappings:** Map `/mnt/user/models/YourModelName` to `/models` with Read/Write access.

## Troubleshooting

* **GPU Out-of-Memory (OOM) Errors during Quantization:**
  * **Primary Culprit (especially GPTQ):** The calibration sequence length. Models with very long context windows (e.g., 128k) cause huge memory spikes during calibration if the full length is used.
  * **Solution:** Use the `--seq_len` argument to specify a shorter sequence length for calibration (e.g., `--seq_len 8192` or `--seq_len 4096`). The script temporarily modifies the model config for quantization and restores it before saving, preserving the model's original context capability.
  * **Other factors:** While less common with explicit offload, extremely large models might still exceed single GPU VRAM even with a reduced `--seq_len`. Try reducing `--seq_len` further (e.g., 2048) or use a machine with more VRAM. Reducing `--batch_size` (already defaults to 1) usually has minimal impact here.
* **CPU Out-of-Memory Errors:** Ensure the system has sufficient free CPU RAM to hold the entire model when offloading. Increase system swap space if necessary.
* **Slow Quantization:** GPTQ on CPU is inherently very slow. Use a GPU if possible. AWQ is generally faster.
* **File Permission Errors:** Ensure the mounted volume (`/models`) has write permissions for the user running the Docker container.
* **Docker Errors:** Check Docker logs (`docker logs <container_id>`). Ensure the Docker image built correctly.
* **Errors Finding Sequence Length Attribute:** The script tries common names (`max_position_embeddings`, `n_positions`, `seq_length`). If the model uses a different attribute, the script will log a warning and proceed without modification (potentially causing OOM). Check the model's `config.json` and logs.

## Technical Implementation

This project uses:
* Python 3.x with PyTorch 2.x
* AutoAWQ (`autoawq`) for 4-bit quantization
* Transformers/Optimum (`transformers`, `optimum[auto-gptq]`) for GPTQ 8-bit quantization
* Accelerate (`accelerate`) for explicit CPU offload (`cpu_offload`)
* Datasets (`datasets`) for GPTQ calibration data
* PSUtil (`psutil`) for potential system monitoring (not currently used for logic)
* Docker for dependency management

The core functionality is in `quantize.py`. It parses arguments, loads the model to CPU, applies `accelerate.cpu_offload` if a GPU is used, performs AWQ or GPTQ (using config modification for sequence length if needed), restores the original config, saves results with appropriate suffixes, and cleans up temporary files.

## License

This project is open source. Please contribute by reporting issues or submitting pull requests.
