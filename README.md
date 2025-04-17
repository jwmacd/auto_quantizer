---
# AWQ 4‑Bit Model Quantizer – Container Edition

Quantise any Hugging Face **text** model to efficient 4‑bit AWQ by simply pointing this container at a folder that holds the original *safetensors* weights.  When the container starts it automatically performs the quantisation and exits when finished – no shell access or bash commands required by the user.

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
