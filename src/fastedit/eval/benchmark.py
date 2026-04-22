"""Benchmark runner for FastEdit evaluation.

Runs the full evaluation suite against a test set:
- Load test examples
- Run model inference on each
- Compute all metrics
- Generate per-language and per-edit-type breakdowns
- Output results as JSON + human-readable summary
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..inference.merge import FastEditEngine
from .metrics import (
    BenchmarkSummary,
    EvalResult,
    evaluate_single,
    parse_model_output,
    summarize_results,
)

console = Console()


def load_test_set(path: Path) -> list[dict]:
    """Load test examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)
    return examples


def _extract_fields(example: dict) -> tuple[str, str, str, str, str]:
    """Extract original_code, update_snippet, expected, language, edit_type from an example."""
    # Handle both raw format and training format
    if "messages" in example:
        # Training format — parse from messages
        user_msg = example["messages"][1]["content"]
        assistant_msg = example["messages"][2]["content"]

        # Extract original_code from <code>...</code>
        code_start = user_msg.find("<code>") + len("<code>")
        code_end = user_msg.find("</code>")
        original_code = user_msg[code_start:code_end]

        # Extract update_snippet from <update>...</update>
        update_start = user_msg.find("<update>") + len("<update>")
        update_end = user_msg.find("</update>")
        update_snippet = user_msg[update_start:update_end]

        # Extract expected from <updated-code>...</updated-code>
        expected = parse_model_output(assistant_msg)

        metadata = example.get("metadata", {})
        language = metadata.get("language", "unknown")
        edit_type = metadata.get("edit_type", "unknown")
    else:
        # Raw format
        original_code = example["original_code"]
        update_snippet = example["update_snippet"]
        expected = example["final_code"]
        language = example.get("language", "unknown")
        edit_type = example.get("edit_type", "unknown")

    return original_code, update_snippet, expected, language, edit_type


def run_benchmark(
    engine: FastEditEngine,
    test_path: Path,
    output_path: Path | None = None,
    limit: int | None = None,
) -> BenchmarkSummary:
    """Run the full benchmark suite.

    Args:
        engine: The FastEdit inference engine.
        test_path: Path to test JSONL file.
        output_path: Optional path to save detailed results.
        limit: Optional limit on number of examples.

    Returns:
        BenchmarkSummary with aggregated results.
    """
    examples = load_test_set(test_path)
    if limit:
        examples = examples[:limit]

    console.print(f"\n[bold]Running benchmark on {len(examples)} examples...[/bold]\n")

    results: list[tuple[EvalResult, str, str]] = []
    detailed: list[dict] = []
    total_latency = 0.0
    total_tokens = 0

    for i, example in enumerate(examples):
        original_code, update_snippet, expected, language, edit_type = _extract_fields(example)

        try:
            merge_result = engine.merge(original_code, update_snippet, language)
        except Exception as e:
            console.print(f"  [red]Error on example {i}: {e}[/red]")
            continue

        eval_result = evaluate_single(expected, merge_result.merged_code, language)
        results.append((eval_result, language, edit_type))

        total_latency += merge_result.latency_ms
        total_tokens += merge_result.tokens_generated

        status = "[green]PASS[/green]" if eval_result.exact_match else (
            "[yellow]AST OK[/yellow]" if eval_result.ast_match else "[red]FAIL[/red]"
        )
        console.print(
            f"  [{i+1}/{len(examples)}] {language:12s} {edit_type:25s} "
            f"{status}  diff={eval_result.diff_lines:3d} lines  "
            f"{merge_result.tokens_per_second:.0f} tok/s"
        )

        detailed.append({
            "index": i,
            "language": language,
            "edit_type": edit_type,
            "exact_match": eval_result.exact_match,
            "ast_match": eval_result.ast_match,
            "parse_valid": eval_result.parse_valid,
            "diff_lines": eval_result.diff_lines,
            "similarity": eval_result.similarity_ratio,
            "latency_ms": merge_result.latency_ms,
            "tokens_per_second": merge_result.tokens_per_second,
        })

    summary = summarize_results(results)

    # Print summary table
    _print_summary(summary, total_latency, total_tokens, len(results))

    # Save detailed results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "summary": asdict(summary),
                "detailed": detailed,
            }, f, indent=2, default=str)
        console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    return summary


def _print_summary(
    summary: BenchmarkSummary,
    total_latency: float,
    total_tokens: int,
    n: int,
) -> None:
    """Print a formatted summary table."""
    console.print("\n[bold]═══ Benchmark Results ═══[/bold]\n")

    # Overall metrics
    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total Examples", str(summary.total))
    table.add_row("Exact Match Rate", f"{summary.exact_match_rate:.1%}")
    table.add_row("AST Match Rate", f"{summary.ast_match_rate:.1%}")
    table.add_row("Parse Valid Rate", f"{summary.parse_valid_rate:.1%}")
    table.add_row("Avg Diff Lines", f"{summary.avg_diff_lines:.1f}")
    table.add_row("Avg Similarity", f"{summary.avg_similarity:.3f}")
    if n > 0 and total_latency > 0:
        table.add_row("Avg Latency", f"{total_latency / n:.0f} ms")
        table.add_row("Avg Throughput", f"{total_tokens / (total_latency / 1000):.0f} tok/s")
    console.print(table)

    # Per-language breakdown
    if summary.by_language:
        lang_table = Table(title="By Language")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Count")
        lang_table.add_column("Exact Match")
        lang_table.add_column("AST Match")
        lang_table.add_column("Parse Valid")
        for lang, stats in sorted(summary.by_language.items()):
            lang_table.add_row(
                lang,
                str(stats["count"]),
                f"{stats['exact_match_rate']:.1%}",
                f"{stats['ast_match_rate']:.1%}",
                f"{stats['parse_valid_rate']:.1%}",
            )
        console.print(lang_table)

    # Per-edit-type breakdown
    if summary.by_edit_type:
        type_table = Table(title="By Edit Type")
        type_table.add_column("Edit Type", style="cyan")
        type_table.add_column("Count")
        type_table.add_column("Exact Match")
        type_table.add_column("AST Match")
        for et, stats in sorted(summary.by_edit_type.items()):
            type_table.add_row(
                et,
                str(stats["count"]),
                f"{stats['exact_match_rate']:.1%}",
                f"{stats['ast_match_rate']:.1%}",
            )
        console.print(type_table)
