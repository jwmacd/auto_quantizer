# LLM Auto Quantizer (CPU/GPU AWQ/GPTQ Quantization)

This project provides a Dockerized environment for **performing quantization of large language models** using either AWQ (typically 4-bit) or GPTQ (typically 8-bit) methods. It supports both CPU-only and GPU-accelerated quantization, incorporating techniques to manage memory usage on single-GPU setups, especially for large models or those with very long context lengths.

## Target Audience & Limitations

While aiming for ease of use, quantizing large language models remains a resource-intensive task. This tool is best suited for:

*   **Users with Capable Hardware:** Requires significant CPU RAM (peak usage potentially ~1.5x original model size during GPTQ) and a modern NVIDIA GPU (especially for reasonable GPTQ speed). Quantizing 60GB+ models challenges even high-end consumer/prosumer systems.
*   **Users Needing Specific Features:** Offers:
    *   **Automated Memory Management:** Loads the base model to CPU RAM (`device_map="cpu"`) to prevent initial VRAM overflow.
    *   **Long Context Handling:** Enables quantization of models with very long context windows by controlling the calibration sequence length (`--seq_len`), significantly reducing peak VRAM usage.
    *   **VRAM Usage Logging:** Reports peak VRAM usage during the quantization step for monitoring.
    *   **Unified Interface:** Provides a consistent command-line interface for both AWQ and GPTQ methods.
*   **Users Quantizing Locally:** Useful for those who need to quantize custom models or fine-tunes not available pre-quantized online.

**Limitations:**

*   **Does Not Eliminate Hardware Needs:** Automates the workflow but cannot overcome fundamental RAM/VRAM limitations. Users must ensure their hardware meets the demands of the chosen model and quantization method.
*   **Requires Some Understanding:** Users benefit from understanding concepts like sequence length impact (`--seq_len`), dataset choice, and resource trade-offs to use the tool effectively and troubleshoot (e.g., needing to adjust `--seq_len` based on model size and VRAM).
*   **Single GPU Focus:** Designed for single-GPU setups; does not support multi-GPU distribution.
*   **Output Format:** Generates standard AWQ and GPTQ formats. Does not currently support other formats like EXL2 or GGUF.

**For users without sufficient local hardware, using pre-quantized models from sources like Hugging Face or leveraging cloud computing platforms is often a more practical approach.**

## Overview

**Purpose:** This tool automates the process of performing AWQ (e.g., 4-bit) or GPTQ (e.g., 8-bit) model quantizations, producing standard quantized model formats compatible with common inference frameworks.

**Key Benefits (for users with adequate hardware):**
* **Flexible Execution:** Run quantization on CPU or utilize a GPU.
* **Memory Management:** Automatically loads base models to CPU RAM (`device_map="cpu"`) to avoid initial VRAM OOM. The quantization libraries handle layer-by-layer processing on the GPU internally.
* **Handles Large Context Models:** Successfully quantizes models with extremely long context windows (e.g., 128k) by using the `--seq_len` argument to reduce peak VRAM during calibration. Lowering `--seq_len` significantly reduces memory pressure.
* **Simplified Setup:** Docker encapsulates all dependencies.
* **Multiple Methods:** Supports both AWQ (e.g., 4-bit, often faster/less VRAM) and GPTQ (e.g., 8-bit).
* **Standard Output:** Creates quantized models in standard formats loadable by tools like vLLM, Transformers, etc., organized into a `METHOD-BITRATE` subdirectory (e.g., `AWQ-4bit`, `GPTQ-8bit`).
* **Monitoring:** Logs peak VRAM usage during critical quantization steps (AWQ & GPTQ), aiding in tuning `--seq_len`.

## Technical Details

| Method | Default Bits | Speed (GPU) | Speed (CPU) | Output Location | Use Case |
|--------|-------------|-------------|-------------|-----------------|----------|
| AWQ    | 4-bit       | Fast        | Reasonable  | `<model_path>/AWQ-4bit/` | Smaller model size, faster inference, generally lower VRAM usage during quantization. |
| GPTQ   | 8-bit       | Moderate    | VERY Slow   | `<model_path>/GPTQ-8bit/` | Potentially better quality than 4-bit, more VRAM intensive during quantization (requires `--seq_len` tuning for large models/limited VRAM). |

## Prerequisites

* Docker installed on your system
* NVIDIA GPU with appropriate drivers (highly recommended for GPTQ, beneficial for AWQ) OR sufficient CPU resources.
* Sufficient disk space for Docker image (~5-10GB for `-devel` image) and models.
* A pre-trained model in Hugging Face format (must contain `config.json`, tokenizer files, and model weights preferably in `safetensors` format).
* Sufficient CPU RAM (peak usage can be ~1.5x original model size, especially for GPTQ; >64GB recommended for 7B+ models).

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jwmacd/auto_quantizer.git
   cd auto_quantizer
   ```

2. **Build the Docker image**
   ```bash
   docker build -t auto_quantizer .
   # OR use the build script (Windows):
   ./build_and_push.ps1 # Builds and pushes to ghcr.io/jwmacd/auto_quantizer:latest
   ```

3. **Run AWQ quantization (4-bit, GPU recommended)**
   ```bash
   # Using GPU (recommended)
   docker run --rm -it --gpus all \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --awq # Uses default --seq_len 2048

   # Using CPU only (add --force_cpu)
   docker run --rm -it \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --awq --force_cpu
   ```

4. **Run GPTQ quantization (8-bit, GPU HIGHLY recommended)**
   ```bash
   # Using GPU (HIGHLY recommended, --seq_len 2048 default)
   docker run --rm -it --gpus all \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --gptq 

   # Explicitly setting seq_len (e.g., if 2048 causes OOM)
   docker run --rm -it --gpus all \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --gptq --seq_len 4096 

   # Using CPU only (EXTREMELY SLOW)
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
| `--awq` | Use AWQ quantization (default 4-bit) | `True` if `--gptq` not specified |
| `--gptq` | Use GPTQ quantization (default 8-bit) | `False` |
| `--bits` | Number of bits for quantization (AWQ/GPTQ) | 4 (AWQ), 8 (GPTQ) |
| `--quant_config` | JSON string for custom AWQ/GPTQ config (merges with defaults) | `{}` |
| `--force_cpu` | Force CPU execution even if GPU is available | `false` |
| `--batch_size` | Batch size for AWQ quantization calibration | `1` |
| `--seq_len` | **Crucial for VRAM.** Max sequence length for calibration data (AWQ/GPTQ). Lower values (e.g., 2048, 1024, 512) significantly reduce peak VRAM, enabling quantization of larger models on limited hardware. Higher values may improve accuracy but require more VRAM. | `2048` |
| `--gptq_dataset` | Dataset for GPTQ calibration (from Hugging Face Datasets) | `wikitext2` |
| `--gptq_group_size` | Group size for GPTQ quantization | 128 |
| `--gptq_desc_act` | Use descending activation order for GPTQ | `false` |

### AWQ Examples

**Basic AWQ (4-bit, GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer --awq # Uses default --seq_len 2048
```

**Custom AWQ configuration (GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --awq --quant_config '{"q_group_size": 64, "zero_point": false}'
```

### GPTQ Examples

**Basic GPTQ (8-bit, GPU, default seq_len 2048):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq
```

**GPTQ with longer sequence length calibration (requires more VRAM):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq --seq_len 4096
```

**Custom GPTQ configuration (GPU):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq --gptq_dataset c4 --gptq_group_size 64 --gptq_desc_act
```

**GPTQ with reduced sequence length calibration (to save VRAM):**
```bash
docker run --rm -it --gpus all \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq --seq_len 1024 # Reduce seq_len if default (2048) causes OOM
```

## Output Files

Both AWQ and GPTQ methods preserve the original model files in the source directory (`/models/.../YourModelName` inside the container). They create a new subdirectory within the source directory and save the quantized outputs there using standard Hugging Face filenames.

**Output Subdirectory:**
- A new directory named using the pattern `METHOD-BITRATE` (e.g., `AWQ-4bit`, `GPTQ-8bit`, or `AWQ-custom_name` if a specific name is derived) is created inside the source model path (`args.model_path`).
  - Example: If `model_path` is `/models/MyModel`, default AWQ output goes into `/models/MyModel/AWQ-4bit/`.
  - Example: If `model_path` is `/models/MyModel`, default GPTQ output goes into `/models/MyModel/GPTQ-8bit/`.

**Files inside the Output Subdirectory (e.g., `GPTQ-8bit/`):**

*   `config.json`: Updated model configuration including `quantization_config` details.
*   `model.safetensors.index.json`: Index for sharded models (if applicable).
*   `model-*.safetensors`: Quantized model weight shards.
*   `tokenizer.json`, `tokenizer_config.json`, etc.: Copied tokenizer files.
*   `special_tokens_map.json`, `vocab.json`, etc.: Other necessary vocabulary/tokenizer files.
*   `quant_config.json`: (AWQ only) AWQ-specific quantization parameters.
*   `*.py`: Copied custom code files (if any) from the original model directory.
*   `quantization_report.log`: Contains logs, including peak VRAM usage during quantization.

## Resource Requirements

* **GPU VRAM:** **Highly dependent on `--seq_len`**. Peak usage is logged in `quantization_report.log`. Lowering `--seq_len` is the primary way to reduce VRAM needs. GPTQ (especially with larger `--seq_len`) is generally more demanding than AWQ. The base model is loaded to CPU (`device_map="cpu"`) first, minimizing initial VRAM spike.
* **CPU RAM:** Peak usage during quantization can be ~1.5x the original model size (e.g., ~90GB for a 60GB model), especially for GPTQ. Less RAM is needed after the quantization process completes.
* **Disk:** Space for original model + output subdirectory (`METHOD-BITRATE/`) + Docker image.
* **Time:** AWQ is relatively fast on GPU (minutes to hours). GPTQ is slower on GPU (expect hours for large models) and *extremely* slow on CPU (potentially days).

## Unraid Integration

When configuring this container via the Unraid GUI:

1. **Repository:** Use `auto_quantizer` or your custom tag (e.g., `ghcr.io/jwmacd/auto_quantizer:latest`).
2. **Extra Parameters:** Add script arguments (e.g., `--gptq --seq_len 2048` or just `--awq`).
3. **GPU:** Enable GPU passthrough if desired (highly recommended for speed).
4. **Volume Mappings:** Map your host model directory (e.g., `/mnt/user/models/YourModelName`) to `/models` inside the container with Read/Write access.

## Troubleshooting

* **GPU Out-of-Memory (OOM) Errors during Quantization:**
  * **Primary Cause:** Calibration sequence length (`--seq_len`) is too high for the model size, quantization method (GPTQ is more sensitive), and available VRAM.
  * **Solution:** **Reduce `--seq_len`**. Start with the default `2048`. If OOM persists, try significantly smaller values (e.g., 1024, 512, or even lower for very large models/limited VRAM). Check the peak VRAM reported in `quantization_report.log` from previous (even failed) runs to guide adjustments.
  * **Other:** Ensure no other processes are consuming significant VRAM.
* **CPU Out-of-Memory Errors:** Occurs if peak RAM usage (~1.5x model size, especially during GPTQ) exceeds available system RAM. Requires more system RAM or using a smaller model.
* **`ImportError: ... auto-gptq ... not found`:** Ensure `optimum[gptq]` is in `requirements.txt` and the Docker image built successfully. Check Docker build logs.
* **`NotImplementedError: Cannot copy out of meta tensor` (AWQ):** This can happen with complex model architectures or if memory management conflicts occur. Using `device_map="cpu"` for the initial load should mitigate this. Ensure library versions are compatible.
* **Slow Quantization:** GPTQ on CPU is extremely slow. AWQ is faster. Ensure GPU is being utilized if available (check startup logs).
* **File Permission Errors:** Ensure the mounted volume (`/models`) has write permissions for the container.
* **Docker Errors:** Check Docker build logs and container runtime logs.

## Technical Implementation

This project uses:
* Python 3.x with PyTorch 2.x
* AutoAWQ (`autoawq`) for 4-bit quantization
* Transformers/Optimum (`transformers`, `optimum[auto-gptq]`) for GPTQ 8-bit quantization
* Accelerate (`accelerate`) - Primarily for the `cpu_offload` import, though explicit call is not used during quantization.
* Datasets (`datasets`) for GPTQ calibration data loading (via library internals).
* PSUtil (`psutil`) for system monitoring (currently informational).
* Docker for dependency management

The core functionality is in `quantize.py`. Key steps:
1. Parse arguments.
2. Determine execution device (GPU preferred, fallback to CPU or use `--force_cpu`).
3. Load tokenizer.
4. **Load Base Model to CPU:** Uses `device_map="cpu"` during `AutoModelForCausalLM.from_pretrained` to load the full-precision model weights into system RAM first, preventing initial VRAM OOM errors, especially critical for large models.
5. **Perform Quantization:**
    *   **GPTQ:** Invoked implicitly by loading the model with a `GptqConfig` within the `quantization_config` argument. The `--seq_len` parameter is applied by temporarily modifying the model's `AutoConfig` *before* this loading step, influencing the memory usage during the internal calibration process performed by the `optimum` library.
    *   **AWQ:** Invoked via an explicit `model.quantize()` call on the CPU-loaded model. The calibration dataset (controlled by `--seq_len`) is passed directly to this method. The `autoawq` library handles the layer-by-layer quantization, moving necessary components to the GPU as needed.
6. **Log Peak VRAM:** Uses a background thread (`VRAMMonitor`) to track and record the maximum GPU memory allocated during the `model.quantize()` (AWQ) or the model loading/quantization process (GPTQ). Logs are saved to `quantization_report.log`.
7. **Restore Config (GPTQ):** After quantization, the temporary modification to the model's config regarding sequence length is reverted if necessary.
8. **Save Results:** Saves the quantized model (using standard Hugging Face `save_pretrained` conventions) and tokenizer files into the designated `METHOD-BITRATE` subdirectory (e.g., `AWQ-4bit/`). The VRAM log is also saved here.
9. Clean up temporary files and resources.

## License

This project is open source. Please contribute by reporting issues or submitting pull requests.
