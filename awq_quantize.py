#!/usr/bin/env python
"""awq_quantize.py
A minimal utility for quantising Hugging Face causal‑language‑models to 4‑bit AWQ.

Key Features
------------
* Loads the base model **on CPU first** (`device_map="cpu"`) to avoid initial GPU OOM.
* Uses AutoAWQ's built‑in calibration routine – only the maximum sequence length
  can be tuned via ``--seq_len`` (default: 2048) to trade accuracy for memory.
* Supports single‑GPU or CPU‑only execution.  When a GPU is available the actual
  layer‑by‑layer quantisation will be executed there automatically by AutoAWQ.
* Persists the quantised weights (safetensors) *and* the tokenizer in an output
  directory called ``AWQ-4bit`` inside the original model folder (or a custom
  location via ``--output_dir``).

Usage
-----
```bash
python awq_quantize.py --model_path /path/to/model [--seq_len 1024] [--force_cpu]
```

Developed as a clean replacement for the previous, multi‑method ``quantize.py``
script – now focusing solely on the most common workflow: 4‑bit AWQ for text
models.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
import inspect
from transformers.models.llama.modeling_llama import LlamaAttention

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Compatibility patch for some model versions where AWQ invokes LlamaAttention
# without the newer `attention_mask` / `position_embeddings` arguments.
# ---------------------------------------------------------------------------
try:
    _orig_forward = LlamaAttention.forward

    def _patched_forward(self, hidden_states, *args, **kwargs):
        """Fallback wrapper that ignores missing optional args."""
        try:
            return _orig_forward(self, hidden_states, *args, **kwargs)
        except TypeError as e:
            if 'attention_mask' not in kwargs:
                kwargs['attention_mask'] = None
            if 'position_embeddings' not in kwargs:
                kwargs['position_embeddings'] = None
            return _orig_forward(self, hidden_states, *args, **kwargs)

    LlamaAttention.forward = _patched_forward
except Exception:
    pass

# ---------------------------------------------------------------------------
# Default configuration – tweakable via CLI if desired in the future
# ---------------------------------------------------------------------------
DEFAULT_SEQ_LEN = 2048
DEFAULT_AWQ_CONFIG = {
    "w_bit": 4,          # 4‑bit weights
    "q_group_size": 128, # group size used by most published models
    "zero_point": True,
    "version": "GEMM",  # GEMM kernels give the best performance right now
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI helper.</p>"""

    parser = argparse.ArgumentParser(
        description="Quantise a text model to 4‑bit AWQ (Activation‑aware Weight Quantisation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the *unquantised* model directory (as downloaded from the Hub).",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Maximum sequence length used during calibration.  Lower values reduce peak VRAM/RAM at the cost of slightly lower accuracy.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the quantised model.  Defaults to `<model_path>/AWQ-4bit`.",
    )

    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Run the entire pipeline on CPU even if a GPU is available.",
    )

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def ensure_output_dir(model_path: str, output_dir: str | None) -> Path:
    """Determine and create the directory that will hold the quantised artefacts."""

    if output_dir is None:
        output_path = Path(model_path) / "AWQ-4bit"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # Device selection
    # ---------------------------------------------------------------------
    if args.force_cpu or not torch.cuda.is_available():
        device = "cpu"
        logging.info("Running on CPU – this will be slow but memory efficient.")
    else:
        device = "cuda:0"
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)} – will offload layers on‑the‑fly.")

    # ---------------------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------------------
    logging.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ---------------------------------------------------------------------
    # Load model on CPU first (device_map="cpu") – avoids initial VRAM OOM
    # ---------------------------------------------------------------------
    logging.info("Loading model weights to CPU … this might take a while.")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        safetensors=True,
        device_map="cpu",
    )

    # ---------------------------------------------------------------------
    # Quantisation – AutoAWQ internally moves layers to <device> as needed
    # ---------------------------------------------------------------------
    logging.info("Starting AWQ quantisation …")
    quant_cfg = DEFAULT_AWQ_CONFIG.copy()

    # Attempt quantisation, automatically reducing seq_len on OOM
    calib_seq = args.seq_len
    while True:
        try:
            model.quantize(
                tokenizer,
                quant_config=quant_cfg,
                max_calib_seq_len=calib_seq,
            )
            logging.info(f"AWQ quantisation completed with seq_len={calib_seq}.")
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            # Detect generic CUDA OOMs as RuntimeError too
            is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()
            if not is_oom:
                raise  # Unknown error, bubble up

            logging.warning(
                f"OOM encountered at seq_len={calib_seq}.  Attempting again with seq_len={calib_seq // 2}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            calib_seq //= 2
            if calib_seq < 64:
                logging.error("Sequence length back‑off reached <64 and still OOM – aborting.")
                raise
            # Loop will retry with the new, smaller seq_len

    # ---------------------------------------------------------------------
    # Persist artefacts
    # ---------------------------------------------------------------------
    save_path = ensure_output_dir(args.model_path, args.output_dir)

    logging.info(f"Saving quantised model to {save_path} …")
    model.save_quantized(save_path, safetensors=True)
    # Ensure the tokenizer is kept in‑sync with the model directory
    tokenizer.save_pretrained(save_path)

    # Also copy the config to the new directory (AutoAWQ does this already
    # but we enforce it just in case).
    src_config = Path(args.model_path) / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, save_path / "config.json")

    logging.info("Done – your 4‑bit AWQ model is ready for use! 🎉")


if __name__ == "__main__":
    main() 