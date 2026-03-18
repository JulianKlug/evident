#!/usr/bin/env python3
"""Export fine-tuned model to GGUF and create Ollama Modelfile.

Uses Unsloth's built-in GGUF export, then creates an Ollama model.

Usage:
    conda activate finetune
    python scripts/export_to_ollama.py --model-dir artifacts/finetune/model

    # Custom quantization:
    python scripts/export_to_ollama.py \
        --model-dir artifacts/finetune/model \
        --quantization q5_k_m \
        --ollama-name qwen3-14b-ft-q5

    # Skip Ollama import (just create GGUF):
    python scripts/export_to_ollama.py --model-dir artifacts/finetune/model --no-ollama
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF + Ollama")
    parser.add_argument("--model-dir", required=True,
                        help="Path to fine-tuned model directory (from finetune_qwen3.py)")
    parser.add_argument("--base-model", default="unsloth/Qwen3-14B-unsloth-bnb-4bit",
                        help="Base model used for fine-tuning")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for GGUF files (default: {model-dir}/gguf)")
    parser.add_argument("--quantization", default="q4_k_m",
                        help="GGUF quantization method (q4_k_m, q5_k_m, q8_0, f16)")
    parser.add_argument("--ollama-name", default="qwen3-14b-ft",
                        help="Name for the Ollama model")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Max sequence length (must match training)")
    parser.add_argument("--no-ollama", action="store_true",
                        help="Skip Ollama model creation (just export GGUF)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "gguf")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    print(f"Loading fine-tuned model from {args.model_dir}", flush=True)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # ── Export to GGUF ──
    print(f"\nExporting to GGUF ({args.quantization})...", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quantization,
    )

    print(f"GGUF export complete!", flush=True)

    # Find the GGUF file
    gguf_files = [f for f in os.listdir(args.output_dir) if f.endswith(".gguf")]
    if not gguf_files:
        print("ERROR: No GGUF file found in output directory", file=sys.stderr)
        sys.exit(1)

    gguf_path = os.path.join(args.output_dir, gguf_files[0])
    print(f"GGUF file: {gguf_path}")

    # ── Create Ollama Modelfile ──
    modelfile_path = os.path.join(args.output_dir, "Modelfile")

    # Check if Unsloth already created a Modelfile
    if os.path.isfile(modelfile_path):
        print(f"Modelfile already created by Unsloth: {modelfile_path}")
    else:
        # Create manually
        modelfile_content = (
            f"FROM {gguf_path}\n"
            f"\n"
            f"PARAMETER temperature 0\n"
            f"PARAMETER num_ctx {args.max_seq_length}\n"
            f"PARAMETER stop <|im_end|>\n"
            f"PARAMETER stop <|endoftext|>\n"
            f"\n"
            f'TEMPLATE """{{{{- range .Messages }}}}\n'
            f"<|im_start|>{{{{ .Role }}}}\n"
            f"{{{{ .Content }}}}<|im_end|>\n"
            f"{{{{- end }}}}\n"
            f"<|im_start|>assistant\n"
            f'"""\n'
        )
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        print(f"Modelfile created: {modelfile_path}")

    # ── Import into Ollama ──
    if args.no_ollama:
        print(f"\nSkipping Ollama import (--no-ollama).")
        print(f"To import manually:")
        print(f"  ollama create {args.ollama_name} -f {modelfile_path}")
        return

    print(f"\nImporting into Ollama as '{args.ollama_name}'...", flush=True)
    try:
        result = subprocess.run(
            ["ollama", "create", args.ollama_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print(f"Ollama model '{args.ollama_name}' created successfully!")
            print(f"\nVerify with: ollama list | grep {args.ollama_name}")
            print(f"Test with:   ollama run {args.ollama_name} 'Hello'")
        else:
            print(f"Ollama import failed (exit code {result.returncode}):")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            print(f"\nTry manually: ollama create {args.ollama_name} -f {modelfile_path}")
    except FileNotFoundError:
        print("Ollama CLI not found. Install from https://ollama.ai")
        print(f"Then run: ollama create {args.ollama_name} -f {modelfile_path}")
    except subprocess.TimeoutExpired:
        print("Ollama import timed out (>10 min). Try manually:")
        print(f"  ollama create {args.ollama_name} -f {modelfile_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
