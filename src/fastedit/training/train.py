"""Fine-tuning script for FastEdit model using bf16 LoRA via Unsloth.

Uses Qwen3.5-4B dense model (GatedDeltaNet hybrid attention). bf16 LoRA
instead of QLoRA — Unsloth recommends against 4-bit quantization for Qwen3.5
due to higher-than-normal quantization errors in GatedDeltaNet layers.

Cloud:  Unsloth bf16 LoRA on L40S 48GB RunPod (~$0.40/hr, ~10GB VRAM)
Local:  mlx-tune on M-series Mac (free, slower but viable for 4B)

Requires: transformers v5 (Unsloth installs automatically).

Usage:
    python -m fastedit.training.train
    python -m fastedit.training.train --train-path data/train.jsonl --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig


def load_dataset(path: str):
    """Load JSONL dataset into HuggingFace format."""
    from datasets import Dataset

    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)

    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Format a training example using the model's chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(config: TrainingConfig):
    """Run the full training loop."""
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    print(f"Loading base model: {config.base_model}")
    print(f"  Max seq length: {config.max_seq_length}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )

    print("Applying LoRA adapters to attention + expert FFN layers...")
    print(f"  Targets: {config.lora.target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
    )

    print(f"Loading training data from {config.train_path}")
    train_dataset = load_dataset(config.train_path)
    train_dataset = train_dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = None
    if Path(config.test_path).exists():
        print(f"Loading eval data from {config.test_path}")
        eval_dataset = load_dataset(config.test_path)
        eval_dataset = eval_dataset.map(
            lambda ex: format_chat(ex, tokenizer),
            remove_columns=eval_dataset.column_names,
        )

    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    print(f"Training: {len(train_dataset)} examples, {config.num_epochs} epochs")
    print(f"  Effective batch size: {effective_batch}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        report_to="wandb",
        run_name=config.wandb_run_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        # Packing DISABLED for Qwen3.5 — GatedDeltaNet linear attention layers
        # use recurrent state that bleeds across packed sequences, causing NaN
        # gradients (unslothai/unsloth#4160, axolotl#3453, transformers#44717).
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Also save merged model (full weights, no adapters)
    merged_dir = config.output_dir + "-merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train FastEdit model")
    parser.add_argument("--train-path", help="Override training data path")
    parser.add_argument("--test-path", help="Override test data path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    args = parser.parse_args()

    config = TrainingConfig()

    if args.train_path:
        config.train_path = args.train_path
    if args.test_path:
        config.test_path = args.test_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length

    train(config)


if __name__ == "__main__":
    main()
