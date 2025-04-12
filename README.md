# LLM Auto Quantizer (CPU-Based AWQ/GPTQ Quantization)

This project provides a Dockerized environment for **performing quantization of large language models** using either AWQ or GPTQ methods on **CPU-only** systems. This makes the quantization process accessible to users without dedicated GPUs.

## Overview

**Purpose:** This tool automatically performs model quantizations using the CPU, producing standard AWQ (4-bit) or GPTQ (8-bit) format models that can later be used for inference on either CPU or GPU with appropriate frameworks.

**Key Benefits:**
* **No GPU Required:** Run quantization operations entirely on CPU hardware
* **Simplified Setup:** Docker encapsulates all dependencies
* **Multiple Methods:** Supports both AWQ (4-bit, faster) and GPTQ (8-bit, higher quality but slower)
* **Standard Output:** Creates quantized models compatible with standard inference tools
* **Consistent Output Format:** Both methods preserve original files and add suffixed versions

## Technical Details

| Method | Default Bits | Speed on CPU | Output Location | Use Case |
|--------|-------------|--------------|-----------------|----------|
| AWQ    | 4-bit       | Reasonable   | Original directory with `-AWQ` suffix | Smaller model files for faster inference |
| GPTQ   | 8-bit       | Slow    | Original directory with `-GPTQ` suffix | Better quality but with slightly larger model files |

## Prerequisites

* Docker installed on your system
* Sufficient disk space for Docker image (~4GB) and models
* A pre-trained model in Hugging Face format (must contain `config.json`, tokenizer files, and model weights in `safetensors` format for best results)

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

3. **Run AWQ quantization (default, 4-bit)**
   ```bash
   docker run --rm -it \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer
   ```

4. **Run GPTQ quantization (8-bit, slower)**
   ```bash
   docker run --rm -it \
     -v /path/to/your/host/models/YourModelName:/models:rw \
     auto_quantizer \
     --gptq
   ```

## Detailed Usage

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to model directory inside container | `/models` |
| `--awq` | Use AWQ quantization | Default if neither specified |
| `--gptq` | Use GPTQ quantization | - |
| `--bits` | Number of bits for quantization | 4 (AWQ), 8 (GPTQ) |
| `--quant_config` | JSON string for custom quantization config | `{}` |
| `--gptq_dataset` | Dataset for GPTQ calibration | `wikitext2` |
| `--gptq_group_size` | Group size for GPTQ quantization | 128 |
| `--gptq_desc_act` | Use descending activation order for GPTQ | `false` |

### AWQ Examples

**Basic AWQ (4-bit):**
```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer
```

**Custom AWQ configuration:**
```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --awq --quant_config '{"q_group_size": 64, "zero_point": true}'
```

### GPTQ Examples

**Basic GPTQ (8-bit):**
```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq
```

**Custom GPTQ configuration:**
```bash
docker run --rm -it \
  -v /path/to/your/host/models/YourModelName:/models:rw \
  auto_quantizer \
  --gptq --gptq_dataset c4 --gptq_group_size 64 --gptq_desc_act
```

## Output Files

### AWQ Output
The AWQ quantization process preserves all original files and adds:
* `*-AWQ.safetensors` - Quantized model weights
* `quant_config-AWQ.json` - Quantization configuration
* `model-AWQ.safetensors.index.json` - Index for multi-file models
* Any custom `.py` files will be renamed with `-AWQ` suffix

### GPTQ Output
The GPTQ quantization also preserves all original files and adds:
* `*-GPTQ.safetensors` or `*-GPTQ.bin` - Quantized model weights
* `config-GPTQ.json` - Configuration with quantization parameters
* `model-GPTQ.safetensors.index.json` - Index for multi-file models
* Any custom `.py` files will be renamed with `-GPTQ` suffix

## Resource Requirements

* **Memory:** AWQ typically requires 1.5-2x the original model size temporarily
* **Disk:** Need space for original model + quantized output
* **Time:** Expect several hours for quantization on CPU depending on model size

## Unraid Integration

When configuring this container via the Unraid GUI:

1. **Repository:** Use `auto_quantizer` or your pushed tag
2. **Extra Parameters:** Add script arguments (e.g., `--gptq --gptq_dataset c4` for custom GPTQ)
3. **GPU:** Not required (CPU-only quantization)
4. **Volume Mappings:** Map `/mnt/user/models/YourModelName` to `/models` with Read/Write

## Troubleshooting

* **Memory Issues:** Reduce model size or increase system swap space
* **File Permission Errors:** Ensure mounted volume has proper write permissions
* **Slow Quantization:** GPTQ on CPU is extremely slow by design, consider AWQ
* **Docker Errors:** Check Docker logs with `docker logs <container_id>`

## Technical Implementation

This project uses:
* Python 3.x with PyTorch 2.x
* AutoAWQ for 4-bit quantization
* Transformers/Optimum for GPTQ 8-bit quantization
* Docker for dependency management and isolation

The core functionality is in `quantize.py`, which handles argument parsing, model loading, quantization, and saving the results.

## License

This project is open source. Please contribute by reporting issues or submitting pull requests.
