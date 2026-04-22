"""Main data generation orchestrator via OpenRouter.

Ties together repo collection, AST analysis, edit sampling,
prompt generation, OpenRouter API calls, and validation
into a complete pipeline for generating training data.

Default model: MiniMax M2.7 ($0.30/M in, $1.20/M out)
~12x cheaper than Claude Sonnet for 50K examples (~$270 vs ~$3,300).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import tiktoken
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from tqdm.asyncio import tqdm

from .ast_analyzer import FileStructure
from .edit_sampler import sample_edit
from .prompt_templates import build_data_gen_prompt
from .repo_collector import SourceFile, collect_and_analyze
from .validate import validate_example

load_dotenv()

_enc = tiktoken.get_encoding("cl100k_base")

# OpenRouter rate limits vary by model; M2.7 is generous
_rate_limiter = AsyncLimiter(50, 60)

# Supported models on OpenRouter (model_id -> friendly name)
MODELS = {
    "minimax/minimax-m2.7": "MiniMax M2.7 ($0.30/$1.20 per M tok)",
    "minimax/minimax-m2": "MiniMax M2 ($0.26/$1.02 per M tok)",
    "deepseek/deepseek-chat": "DeepSeek V3 ($0.32/$0.89 per M tok)",
    "google/gemini-2.5-flash-preview": "Gemini 2.5 Flash ($0.15/$0.60 per M tok)",
    "mistralai/devstral-medium": "Devstral Medium ($0.40/$2.00 per M tok)",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5 ($3.00/$15.00 per M tok)",
}

DEFAULT_MODEL = "minimax/minimax-m2.7"


def _make_client() -> AsyncOpenAI:
    """Create an OpenRouter-compatible AsyncOpenAI client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": "FastEdit Data Gen",
        },
    )


def _parse_response(response_text: str) -> tuple[str | None, str | None]:
    """Parse update_snippet and final_code from model response."""
    update_snippet = None
    final_code = None

    for start_tag, end_tag in [
        ("<update_snippet>", "</update_snippet>"),
        ("<update-snippet>", "</update-snippet>"),
    ]:
        start_idx = response_text.find(start_tag)
        end_idx = response_text.find(end_tag, start_idx + len(start_tag) if start_idx != -1 else 0)
        if start_idx != -1 and end_idx != -1:
            update_snippet = response_text[start_idx + len(start_tag):end_idx].strip()
            break

    for start_tag, end_tag in [
        ("<final_code>", "</final_code>"),
        ("<final-code>", "</final-code>"),
    ]:
        start_idx = response_text.find(start_tag)
        end_idx = response_text.find(end_tag, start_idx + len(start_tag) if start_idx != -1 else 0)
        if start_idx != -1 and end_idx != -1:
            final_code = response_text[start_idx + len(start_tag):end_idx].strip()
            break

    return update_snippet, final_code


async def generate_edit_pair(
    client: AsyncOpenAI,
    source_file: SourceFile,
    structure: FileStructure,
    target,
    model: str = DEFAULT_MODEL,
    rate_limiter: AsyncLimiter | None = None,
) -> dict | None:
    """Generate a single (original, snippet, merged) training example via OpenRouter."""
    prompt = build_data_gen_prompt(source_file.content, structure, target)
    limiter = rate_limiter or _rate_limiter

    try:
        async with limiter:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_tokens=source_file.token_count + 2000,
                messages=[{"role": "user", "content": prompt}],
            )
    except BadRequestError:
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None

    response_text = response.choices[0].message.content
    if not response_text:
        return None

    update_snippet, final_code = _parse_response(response_text)
    if not update_snippet or not final_code:
        return None

    validation = validate_example(
        original_code=source_file.content,
        update_snippet=update_snippet,
        merged_code=final_code,
        language=source_file.language,
    )

    # Track cost from OpenRouter usage header
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return {
        "original_code": source_file.content,
        "update_snippet": update_snippet,
        "final_code": final_code,
        "language": structure.language,
        "edit_type": target.edit_type.value,
        "file_path": str(source_file.path),
        "token_count": source_file.token_count,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "validation": {
            "is_valid": validation.is_valid,
            "original_parses": validation.original_parses,
            "merged_parses": validation.merged_parses,
            "is_nontrivial": validation.is_nontrivial,
            "structural_ok": validation.structural_ok,
            "reasons": validation.reasons,
        },
    }


async def generate_from_repo(
    repo_path: Path,
    output_path: Path,
    languages: set[str] | None = None,
    max_examples: int = 1000,
    examples_per_file: int = 1,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    concurrency: int = 20,
) -> dict:
    """Generate training examples from a single repository via OpenRouter.

    Args:
        repo_path: Path to the cloned repository.
        output_path: Path to write JSONL output.
        languages: Languages to include (None = all).
        max_examples: Maximum examples to generate.
        examples_per_file: How many edits to sample per file.
        model: OpenRouter model ID (default: minimax/minimax-m2.7).
        seed: Random seed for reproducibility.
        concurrency: Max concurrent API requests.
    """
    client = _make_client()

    model_name = MODELS.get(model, model)
    print(f"Model: {model_name}")
    print("Endpoint: OpenRouter (https://openrouter.ai/api/v1)")

    file_structures = list(collect_and_analyze(repo_path, languages))
    print(f"Found {len(file_structures)} parseable files in {repo_path}")

    if not file_structures:
        return {"total": 0, "valid": 0, "invalid": 0}

    # Sample edit targets
    tasks = []
    for i, (sf, struct) in enumerate(file_structures):
        if len(tasks) >= max_examples:
            break
        for j in range(examples_per_file):
            target = sample_edit(struct, seed=seed + i * 100 + j)
            tasks.append((sf, struct, target))

    tasks = tasks[:max_examples]
    print(f"Generating {len(tasks)} edit examples (concurrency={concurrency})...")

    # Semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)

    async def bounded_generate(sf, struct, target):
        async with sem:
            return await generate_edit_pair(client, sf, struct, target, model)

    async_tasks = [
        bounded_generate(sf, struct, target)
        for sf, struct, target in tasks
    ]

    results = []
    valid_count = 0
    invalid_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for coro in tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks), desc="Generating"):
            result = await coro
            if result is None:
                invalid_count += 1
                continue

            results.append(result)
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)

            if result["validation"]["is_valid"]:
                valid_count += 1
                f.write(json.dumps(result) + "\n")
            else:
                invalid_count += 1

    # Cost estimate
    model_pricing = {
        "minimax/minimax-m2.7": (0.30, 1.20),
        "minimax/minimax-m2": (0.26, 1.02),
        "deepseek/deepseek-chat": (0.32, 0.89),
        "google/gemini-2.5-flash-preview": (0.15, 0.60),
        "mistralai/devstral-medium": (0.40, 2.00),
        "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    }
    in_price, out_price = model_pricing.get(model, (0, 0))
    est_cost = (total_input_tokens * in_price / 1_000_000) + (total_output_tokens * out_price / 1_000_000)

    stats = {
        "total": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "model": model,
        "repo": str(repo_path),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": round(est_cost, 2),
        "languages": list({r["language"] for r in results}),
        "edit_types": list({r["edit_type"] for r in results}),
    }

    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone: {valid_count} valid, {invalid_count} invalid")
    print(f"Tokens: {total_input_tokens:,} in + {total_output_tokens:,} out")
    print(f"Estimated cost: ${est_cost:.2f}")
    print(f"Output: {output_path}")

    return stats


def list_models():
    """Print available models and their pricing."""
    print("\nAvailable models on OpenRouter:\n")
    print(f"  {'Model ID':<40} {'Pricing'}")
    print(f"  {'─' * 40} {'─' * 40}")
    for model_id, desc in MODELS.items():
        marker = " ← default" if model_id == DEFAULT_MODEL else ""
        print(f"  {model_id:<40} {desc}{marker}")
    print()
