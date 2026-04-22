"""Export trained FastEdit model to formats for local serving.

Supports:
- GGUF (for LM Studio, llama.cpp, Ollama, oMLX)
- MLX (for mlx-lm, oMLX, vllm-mlx — recommended on Mac)
- Merged 16-bit (for vLLM, SGLang)
- HuggingFace upload

Usage:
    python -m fastedit.training.export --model output/fastedit-4b --format gguf --quant Q4_K_M
    python -m fastedit.training.export --model output/fastedit-4b --format merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig

DEFAULT_MAX_SEQ = TrainingConfig.max_seq_length


def export_gguf(model_path: str, quant: str = "Q4_K_M", output_dir: str = "export"):
    """Export to GGUF format for LM Studio / Ollama / oMLX.

    Uses Unsloth's built-in GGUF export which handles
    the LoRA merge + quantization in one step.
    4B model at Q4_K_M is only ~2-3GB — runs anywhere.
    """
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=DEFAULT_MAX_SEQ,
        load_in_4bit=False,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to GGUF with {quant} quantization...")
    model.save_pretrained_gguf(
        str(out / f"fastedit-{quant.lower()}"),
        tokenizer,
        quantization_method=quant,
    )
    print(f"GGUF model saved to {out}")
    print("\nTo use in LM Studio:")
    print("  1. Open LM Studio")
    print("  2. Go to 'My Models' -> 'Import'")
    print(f"  3. Select {out / f'fastedit-{quant.lower()}'}")
    print("  4. Start the server (it'll be at http://localhost:1234/v1)")
    print("\nNote: 4B Q4_K_M GGUF is ~2-3 GB (~200-250 tok/s raw on M3 Max)")


def export_merged(model_path: str, output_dir: str = "export"):
    """Export merged 16-bit weights for vLLM / SGLang."""
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=DEFAULT_MAX_SEQ,
        load_in_4bit=False,
    )

    out = Path(output_dir) / "fastedit-merged"
    out.mkdir(parents=True, exist_ok=True)

    print("Merging LoRA weights and saving 16-bit model...")
    model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {out}")
    print("\nTo serve with vLLM:")
    print(f"  vllm serve {out} --max-model-len {DEFAULT_MAX_SEQ}")


def export_to_hub(model_path: str, repo_id: str, quant: str = "Q4_K_M"):
    """Push model to HuggingFace Hub."""
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=DEFAULT_MAX_SEQ,
        load_in_4bit=False,
    )

    print(f"Pushing to HuggingFace Hub: {repo_id}")
    model.push_to_hub_gguf(
        repo_id,
        tokenizer,
        quantization_method=quant,
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export FastEdit model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument(
        "--format",
        choices=["gguf", "merged", "hub"],
        default="gguf",
        help="Export format",
    )
    parser.add_argument("--quant", default="Q4_K_M", help="GGUF quantization (Q4_K_M, Q5_K_M, Q8_0)")
    parser.add_argument("--output", default="export", help="Output directory")
    parser.add_argument("--repo-id", help="HuggingFace repo ID (for hub export)")
    args = parser.parse_args()

    if args.format == "gguf":
        export_gguf(args.model, args.quant, args.output)
    elif args.format == "merged":
        export_merged(args.model, args.output)
    elif args.format == "hub":
        if not args.repo_id:
            parser.error("--repo-id required for hub export")
        export_to_hub(args.model, args.repo_id, args.quant)


if __name__ == "__main__":
    main()
