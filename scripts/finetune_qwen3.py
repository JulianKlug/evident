#!/usr/bin/env python3
"""Fine-tune qwen3:14b with QLoRA using Unsloth + TRL SFTTrainer.

Requires separate conda env with Python 3.10+:
    conda create -n finetune python=3.10
    conda activate finetune
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install trl peft bitsandbytes datasets

Usage:
    conda activate finetune
    python scripts/finetune_qwen3.py \
        --train-data artifacts/finetune/train_data.jsonl \
        --output-dir artifacts/finetune/model

    # Custom hyperparameters:
    python scripts/finetune_qwen3.py \
        --train-data artifacts/finetune/train_data.jsonl \
        --output-dir artifacts/finetune/model \
        --epochs 5 --lr 1e-4 --lora-rank 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load_jsonl_dataset(path: str):
    """Load JSONL training data into a HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune qwen3:14b with QLoRA")
    parser.add_argument("--train-data", required=True,
                        help="Path to training JSONL (ChatML messages format)")
    parser.add_argument("--output-dir", default="artifacts/finetune/model",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--base-model", default="unsloth/Qwen3-14B-unsloth-bnb-4bit",
                        help="Base model (Unsloth 4-bit quantized)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (scaling)")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save hyperparameters for reproducibility
    hparams = vars(args)
    hparams_path = os.path.join(args.output_dir, "hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"Hyperparameters saved to {hparams_path}")

    # ── Load model with Unsloth ──
    print(f"\nLoading base model: {args.base_model}", flush=True)
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # ── Apply LoRA ──
    print(f"\nApplying LoRA (r={args.lora_rank}, alpha={args.lora_alpha})", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # memory-efficient
        random_state=args.seed,
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Load dataset ──
    print(f"\nLoading training data: {args.train_data}", flush=True)
    dataset = load_jsonl_dataset(args.train_data)
    print(f"Training examples: {len(dataset)}")

    # ── Format for ChatML ──
    # Unsloth's apply_chat_template handles ChatML formatting
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=["messages"])

    # Check token lengths
    sample_tokens = tokenizer(dataset[0]["text"], return_length=True)
    print(f"Sample token length: {sample_tokens['length'][0]}")

    # ── Training ──
    print(f"\nStarting training ({args.epochs} epochs, effective batch={args.batch_size * args.grad_accum})",
          flush=True)

    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Detect dtype from model — Ampere+ GPUs (RTX 3090, etc.) use bfloat16
    use_bf16 = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,  # Don't pack sequences (variable-length pages)
    )

    # Train
    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # Save final model
    print(f"\nSaving model to {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    metrics = {
        "total_steps": train_result.global_step,
        "final_loss": train_result.training_loss,
        "n_examples": len(dataset),
        "epochs": args.epochs,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nDone! Next step: export to GGUF/Ollama with scripts/export_to_ollama.py")


if __name__ == "__main__":
    main()
