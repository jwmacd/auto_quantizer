---
# AWQ 4‑Bit Model Quantizer – Container Edition

Quantise any Hugging Face **text** model to efficient 4‑bit AWQ by simply pointing this container at a folder that holds the original *safetensors* weights.  When the container starts it automatically performs the quantisation and exits when finished – no shell access or bash commands required by the user.

## High-Quality Mode

For scenarios where maximum model fidelity is paramount, you can enable *High-Quality Mode* by adding `--max_quality` to the container arguments.  This mode:
* Forces CPU-only quantization to avoid any GPU OOM issues.
* Keeps key layers in FP16 (`--keep-fp-layers 0,1,-1`): embedding layer, first transformer block, and the LM head.
* Enables descending activation order and zero-point adjustments (`--desc-act --zero-point`).
* Uses an extended calibration pass (`--seq_len 16384 --calib-size 20000 --calib-split train,valid --dataset wikitext2,c4,book,github,stack`).
* Performs a double-scale Hessian-based search (`--iter 40 --search-step 2`).
* Dumps salient outlier columns in INT8 alongside the main 4‑bit tensors (`--dump-salient ./outlier.json --salient-quota 0.004`).
* Writes the results into `/models/AWQ-4bit-MAX/` instead of the default `AWQ-4bit/`.

## CLI Flags

### General Options
- `--model_path` (required): Path to your model directory.
- `--seq_len` (default: 2048): Max sequence length for calibration (back-offs apply).
- `--output_dir` (default: `<model_path>/AWQ-4bit`): Directory to write quantised files.
- `--force_cpu`: Force entire quantisation on CPU.
- `--max_quality`: Enable high-quality mode (see above).

### Quality Knobs
- `--keep-fp-layers`: Comma-separated layer indices to keep in FP16 (e.g. `0,1,-1`).
- `--desc-act`: Use descending activation order for improved scale accuracy.
- `--zero-point`: Enable zero-point quantisation for shifted ranges.
- `--q-group-size`: Override the weight quantisation group size (default: 128).
- `--iter`: Number of iterations for Hessian-based scale search (default: 1).
- `--search-step`: Step multiplier for scale search (default: 1).
- `--dump-salient`: Path to write salient outlier indices in JSON.
- `--salient-quota`: Fraction of columns to treat as outliers (e.g. 0.004).

---

## How it works

1. **Volume mapping** – you map a host directory that contains your full‑precision model files (`config.json`, `model‑*.safetensors`, tokenizer files, …) to `/models` inside the container.
2. **Automatic execution** – the image entry‑point runs `awq_quantize.py --model_path /models`.  Nothing else is needed.
3. **OOM‑aware calibration** – the script starts with a 2048‑token calibration sequence length and automatically halves that value if a GPU out‑of‑memory error is encountered, down to a minimum of 64 tokens.
4. **Output** – upon success a new directory `/models/AWQ-4bit/` is created containing:
   • quantised weights (`*.safetensors`),
   • tokenizer files,
   • updated `config.json` and `quant_config.json`.
5. **Container exit** – once the files are written the container stops; you can now use the quantised model for inference with any AWQ‑aware runtime.

---

## Unraid deployment

In the Unraid template:

• **Repository** – point to the image tag you built or pulled (e.g. `ghcr.io/yourname/awq_quantizer:latest`).  
• **Additional fields** – leave *Console command* blank; the image's entry‑point handles execution.  
• **Volume** – map the host model directory (read/write) to `/models` in the container.  
• **GPU** – enable NVIDIA GPU passthrough for fastest quantisation.  If you have no GPU, omit GPU settings; the script will run on CPU (slower).  
• **Environment / extra args** – optionally set `SEQ_LEN` or `FORCE_CPU` as environment variables if your UI template supports them; otherwise adjust the container command line (e.g. `--seq_len 1024`).

The container log will show progress and report the final sequence length chosen (after any OOM back‑off) together with timing information.

---

## Resource guidelines

| Resource | Typical needs for a 7B model |
|----------|--------------------------------|
| **GPU VRAM** | ~3‑6 GB at seq‑len 2048 (halve seq‑len to roughly halve VRAM). |
| **CPU RAM** | At least the full‑precision model size (e.g. 16–20 GB for a 7B). |
| **Disk** | Original model + ~25 % for the 4‑bit output. |

Larger models (13B, 70B, etc.) scale these requirements proportionally.

---

## Internals (for the curious)

The `awq_quantize.py` script faithfully reproduces the AWQ workflow previously embedded in `quantize.py`:

* Loads tokenizer and model to **CPU first** using `device_map="cpu"` to avoid the initial VRAM spike.
* Runs `model.quantize()` with the GEMM kernel configuration (`w_bit=4`, `q_group_size=128`, `zero_point=True`).
* Enters a retry loop that halves the calibration sequence length on every CUDA OOM until it succeeds or the length would fall below 64 tokens.
* Uses `model.save_quantized()` to write safetensors, then copies tokenizer files and `config.json` into the output folder.

---

## License

Apache 2.0 – contributions welcome.
