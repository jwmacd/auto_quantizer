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
import subprocess, sys
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

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

    parser.add_argument(
        '--max_quality', action='store_true',
        help='Enable max-quality AWQ workflow (CPU-only, advanced flags, AWQ-4bit-MAX)'
    )

    parser.add_argument('--keep-fp-layers', type=str, default=None, help='Comma-separated layers to keep in FP16 (e.g. 0,1,-1)')
    parser.add_argument('--desc_act', action='store_true', help='Use descending activation order')
    parser.add_argument('--zero_point', action='store_true', help='Enable zero-point quantization')
    parser.add_argument('--q_group_size', type=int, default=DEFAULT_AWQ_CONFIG['q_group_size'], help='Override group size for quantization')
    parser.add_argument('--iter', type=int, default=1, help='Number of iterations for scale search')
    parser.add_argument('--search_step', type=int, default=1, help='Multiplier for search step size')
    parser.add_argument('--dump-salient', type=str, default=None, help='Path to dump salient outlier indices (JSON)')
    parser.add_argument('--salient-quota', type=float, default=None, help='Target fraction of salient columns (e.g. 0.004)')

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
    logging.info(f"Parsed CLI args: {args}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    # Early exit if default output exists
    outdir = args.output_dir or os.path.join(args.model_path, 'AWQ-4bit')
    if os.path.exists(outdir) and not args.max_quality:
        logging.warning(f"Output directory {outdir} already exists; aborting to avoid overwrite.")
        sys.exit(0)

    # Load custom quant_config if provided
    try:
        custom_config = json.loads(args.quant_config)
    except json.JSONDecodeError:
        custom_config = {}

    # Merge AWQ defaults, custom JSON config, and CLI quality flags
    quality_kwargs = {}
    if args.keep_fp_layers:
        quality_kwargs['keep_fp_layers'] = [int(x) for x in args.keep_fp_layers.split(',')]
    quality_kwargs.update({
        'desc_act': args.desc_act,
        'zero_point': args.zero_point,
        'q_group_size': args.q_group_size,
        'iter': args.iter,
        'search_step': args.search_step,
    })
    if args.dump_salient:
        quality_kwargs['dump_salient'] = args.dump_salient
    if args.salient_quota:
        quality_kwargs['salient_quota'] = args.salient_quota
    quant_cfg = {**DEFAULT_AWQ_CONFIG, **custom_config, **quality_kwargs}

    # Max-quality workflow
    if args.max_quality:
        logging.info("Running MAX_QUALITY AWQ CLI on CPU...")
        # Temporarily disable GPUs for calibration
        orig_env = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        cli_cmd = [
            sys.executable, '-m', 'awq.entry', 'quantize',
            '--model_path', args.model_path,
            '--keep-fp-layers', args.keep_fp_layers or '0,1,-1',
            '--desc_act', '--zero_point',
            '--q_group_size', str(args.q_group_size),
            '--seq_len', '16384',
            '--calib-size', '20000',
            '--calib-split', 'train,valid',
            '--dataset', 'wikitext2,c4,book,github,stack',
            '--iter', '40',
            '--search_step', '2',
            '--dump-salient', './outlier.json',
            '--salient-quota', '0.004'
        ]
        logging.info(f"CLI command: {' '.join(cli_cmd)}")
        result = subprocess.run(cli_cmd)
        # Restore GPU visibility for packing
        if orig_env is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = orig_env
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        if result.returncode != 0:
            logging.error(f"Max-quality AWQ CLI failed with code {result.returncode}.")
            sys.exit(result.returncode)

        # Rename output folder (if no custom output_dir)
        if args.output_dir is None:
            default_out = Path(args.model_path) / 'AWQ-4bit'
            max_out = default_out.parent / 'AWQ-4bit-MAX'
            if default_out.exists():
                if max_out.exists():
                    logging.error(f"{max_out} already exists—aborting rename.")
                    sys.exit(1)
                shutil.move(str(default_out), str(max_out))
                logging.info(f"Max-quality output moved to {max_out}")
            else:
                logging.warning(f"Expected output {default_out} not found. Skipping rename.")
        else:
            logging.info(f"Custom output_dir supplied ({args.output_dir}); skipping MAX suffix rename.")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Device selection
    if args.force_cpu or not torch.cuda.is_available():
        device = "cpu"
        logging.info("Running on CPU – this will be slow but memory efficient.")
    else:
        device = "cuda:0"
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)} – will offload layers on‑the‑fly.")
    logging.info(f"Using compute device: {device}")

    # ---------------------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------------------
    logging.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        vocab_size = len(tokenizer.get_vocab())
        logging.info(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except Exception:
        logging.info("Tokenizer loaded.")

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
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model loaded to CPU. Total parameters: {num_params}")

    # ---------------------------------------------------------------------
    # Quantisation – AutoAWQ internally moves layers to <device> as needed
    # ---------------------------------------------------------------------
    logging.info("Starting AWQ quantisation …")
    logging.info(f"Quantization configuration: {quant_cfg}")

    # Attempt quantisation, automatically reducing seq_len on OOM
    calib_seq = args.seq_len
    while True:
        logging.info(f"Attempting AWQ quantization with max_calib_seq_len={calib_seq}")
        try:
            model.quantize(
                tokenizer,
                quant_config=quant_cfg,
                max_calib_seq_len=calib_seq,
            )
            logging.info(f"AWQ quantisation completed successfully with seq_len={calib_seq}.")
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

    logging.info(f"Persisting quantized artifacts to {save_path} …")
    model.save_quantized(str(save_path), safetensors=True)
    # Ensure the tokenizer is kept in‑sync with the model directory
    tokenizer.save_pretrained(save_path)

    # Also copy the config to the new directory (AutoAWQ does this already
    # but we enforce it just in case).
    src_config = Path(args.model_path) / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, save_path / "config.json")
    # Log final directory contents
    try:
        saved_files = [f.name for f in save_path.iterdir()]
        logging.info(f"Quantized model output files ({len(saved_files)}): {saved_files}")
    except Exception:
        logging.info("Unable to list saved files.")

    logging.info("Done – your 4‑bit AWQ model is ready for use! 🎉")


if __name__ == "__main__":
    main() 