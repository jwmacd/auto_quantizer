# Project Context

## Project Overview

This project provides a **Dockerised**, memory‑efficient workflow for quantising Hugging Face text models to **4‑bit AWQ**. It intentionally drops all previous GPTQ and vision functionality to keep the stack lean and reliable.

Key design goals:
* **Single‑GPU friendly** – the base model is first loaded on **CPU** (`device_map="cpu"`) and layers are moved to GPU only when needed.
* **Automatic OOM recovery** – if the chosen calibration sequence length causes an out‑of‑memory error, the script halves the length and retries until it succeeds (or reaches the hard minimum of 64 tokens).
* **One‑command usage** – a single script, `awq_quantize.py`, handles the entire process.

## Architecture

- **Docker container** – ships with Python 3, PyTorch + CUDA, AutoAWQ, Transformers, Accelerate and Safetensors. No GPTQ / vision dependencies are included.
- **Quantisation script (`awq_quantize.py`)** – parses CLI arguments, selects the execution device (GPU preferred), loads the model on CPU, runs AWQ quantisation with automatic seq‑len back‑off, and saves the artefacts.
- **Execution** – runs on the first available NVIDIA GPU by default; use `--force_cpu` to disable GPU usage.
- **Input / Output** –
  * **Input**: a directory containing the full‑precision model (e.g. a path mounted into the container).
  * **Output**: a sub‑directory called `AWQ-4bit` (or a custom `--output_dir`) holding the quantised weights (`*.safetensors`), tokenizer files and `config.json`.

## Key Components

- `Dockerfile` – builds the lightweight image with the trimmed dependency set.
- `requirements.txt` – lists only the packages needed for AWQ.
- `awq_quantize.py` – the single entry point; implements automatic sequence‑length back‑off.
- `README.md` – user‑facing documentation.
- `devlog.md` – internal development log.

## Conventions

- Quantisation method: **AWQ 4‑bit** only.
- Default calibration sequence length: **2048** tokens.
- Automatic back‑off halving rule: 2048 → 1024 → 512 … down to 64 tokens.
- Execution device: GPU if available, otherwise CPU (or use `--force_cpu`).
- Output directory: `AWQ-4bit` inside the input model folder unless overridden.
- High-quality mode: use `--max_quality` to force CPU-only, apply advanced calibration settings (keep specific layers in FP16, descending activation order, zero-point, extended sequence length and dataset, double-scale search, outlier rescue), and save results in `AWQ-4bit-MAX` subdirectory.

## Dependencies

- Python 3.9+
- PyTorch (compatible with your CUDA toolkit)
- AutoAWQ (latest, installed from the official Git repo)
- Transformers
- Accelerate
- Safetensors
- Docker (optional, but recommended for consistent environments)

## Tooling

- `build_and_push.ps1` – Windows‑friendly helper to build the Docker image and push it to a registry (`docker login` required).

---

This condensed context reflects the refactored, AWQ‑only codebase and deployment environment.