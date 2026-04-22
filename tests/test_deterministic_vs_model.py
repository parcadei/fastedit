"""Head-to-head: deterministic text_match vs model — production-accurate.

Simulates the EXACT production path for replace= edits:
1. Write original code to a temp file
2. get_ast_map() via tree-sitter (same as CLI/MCP)
3. Find which symbol changed by comparing original vs final ASTs
4. Extract the target symbol (same as chunked_merge.py L240-245)
5. Run deterministic_edit() on the scoped chunk
6. Splice back and compare to expected final code

This is the ground truth for how many benchmark edits the deterministic
path can handle vs how many need the 1.7B model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fastedit.inference.text_match import deterministic_edit
from fastedit.inference.ast_utils import get_ast_map

BENCHMARK_PATH = Path(__file__).parent.parent / "data" / "benchmark.jsonl"

# Map benchmark language names to file extensions
LANG_EXT = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "rust": ".rs",
    "go": ".go",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "ruby": ".rb",
    "swift": ".swift",
    "kotlin": ".kt",
    "csharp": ".cs",
    "php": ".php",
}


def _normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


def _load_benchmark() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"Benchmark file not found: {BENCHMARK_PATH}")
    examples = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


_EXAMPLES = None


def _get_examples():
    global _EXAMPLES
    if _EXAMPLES is None:
        _EXAMPLES = _load_benchmark()
    return _EXAMPLES


def _get_ast_for_code(code: str, language: str) -> list:
    """Write code to a temp file and get its AST map — same as production."""
    ext = LANG_EXT.get(language, ".txt")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    try:
        nodes = get_ast_map(tmp_path, len(code.splitlines()))
        return nodes or []
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _find_changed_symbol(original: str, final: str, language: str) -> str | None:
    """Find which symbol changed between original and final code.

    Compares AST maps and returns the name of the first symbol whose
    code differs between original and final.
    """
    orig_nodes = _get_ast_for_code(original, language)
    final_nodes = _get_ast_for_code(final, language)

    if not orig_nodes:
        return None

    orig_lines = original.splitlines()
    final_lines = final.splitlines()

    # Build name→node maps
    orig_map = {}
    for n in orig_nodes:
        key = f"{n.parent}.{n.name}" if n.parent else n.name
        orig_map[key] = n

    final_map = {}
    for n in final_nodes:
        key = f"{n.parent}.{n.name}" if n.parent else n.name
        final_map[key] = n

    # Find first symbol whose code differs
    for key, orig_node in orig_map.items():
        orig_code = "\n".join(orig_lines[orig_node.line_start - 1:orig_node.line_end])

        if key in final_map:
            final_node = final_map[key]
            final_code = "\n".join(final_lines[final_node.line_start - 1:final_node.line_end])
            if orig_code != final_code:
                return orig_node.name
        else:
            # Symbol was removed/renamed — this is a modification
            return orig_node.name

    # Check for new symbols (added in final but not in original)
    for key in final_map:
        if key not in orig_map:
            # New symbol added — for replace= test, find the nearest existing symbol
            # In production this would be an after= edit, not replace=
            return None

    return None


def _extract_symbol_code(code: str, symbol_name: str, language: str):
    """Extract a symbol's code from source — same as production chunked_merge.py L240-245.

    Returns (func_start_0idx, func_end_0idx, original_func_str) or None.
    """
    nodes = _get_ast_for_code(code, language)
    if not nodes:
        return None

    # Find the target node (simple name match, like _resolve_symbol)
    target = None
    for n in nodes:
        if n.name == symbol_name:
            target = n
            break

    if target is None:
        return None

    lines = code.splitlines(keepends=True)
    func_start = target.line_start - 1  # 0-indexed
    func_end = target.line_end           # 1-indexed inclusive → exclusive for slice
    original_func = "".join(lines[func_start:func_end])

    return func_start, func_end, original_func


class TestProductionPath:
    """Benchmark deterministic_edit using the exact production AST pipeline."""

    def test_production_accuracy(self):
        """Run the full production path: AST scope → deterministic_edit → splice."""
        examples = _get_examples()

        det_pass = 0
        det_fail = 0
        det_skip = 0  # deterministic returned None (needs model)
        no_symbol = 0  # couldn't identify target symbol
        by_edit_type: dict[str, dict[str, int]] = {}
        failures: list[dict] = []

        for i, ex in enumerate(examples):
            if ex["edit_type"] == "delete_function":
                continue

            edit_type = ex["edit_type"]
            language = ex["language"]

            if edit_type not in by_edit_type:
                by_edit_type[edit_type] = {"pass": 0, "fail": 0, "skip": 0}

            original = ex["original_code"]
            snippet = ex["update_snippet"]
            expected = ex["final_code"]

            # Step 1: Find which symbol changed (simulates knowing replace=X)
            symbol = _find_changed_symbol(original, expected, language)
            if symbol is None:
                no_symbol += 1
                det_skip += 1
                by_edit_type[edit_type]["skip"] += 1
                continue

            # Step 2: Extract symbol code via AST (same as production L240-245)
            extracted = _extract_symbol_code(original, symbol, language)
            if extracted is None:
                no_symbol += 1
                det_skip += 1
                by_edit_type[edit_type]["skip"] += 1
                continue

            func_start, func_end, original_func = extracted

            # Step 3: Run deterministic_edit on the scoped chunk
            edited = deterministic_edit(original_func, snippet)

            if edited is None:
                det_skip += 1
                by_edit_type[edit_type]["skip"] += 1
                continue

            # Step 4: Splice back (same as production L249-254)
            edited_lines = edited.splitlines(keepends=True)
            if edited_lines and not edited_lines[-1].endswith("\n"):
                edited_lines[-1] += "\n"
            original_lines = original.splitlines(keepends=True)
            result_lines = list(original_lines)
            result_lines[func_start:func_end] = edited_lines
            reconstructed = "".join(result_lines)

            # Step 5: Compare
            if _normalize(reconstructed) == _normalize(expected):
                det_pass += 1
                by_edit_type[edit_type]["pass"] += 1
            else:
                det_fail += 1
                by_edit_type[edit_type]["fail"] += 1
                failures.append({
                    "index": i,
                    "language": language,
                    "edit_type": edit_type,
                    "symbol": symbol,
                })

        total = det_pass + det_fail + det_skip
        tried = det_pass + det_fail

        print(f"\n{'='*60}")
        print("PRODUCTION PATH: AST scope → deterministic_edit → splice")
        print(f"{'='*60}")
        print(f"Total examples (excl delete): {total}")
        print(f"  Couldn't find target symbol: {no_symbol}")
        print(f"  Deterministic PASS:  {det_pass}", end="")
        if total:
            print(f" ({det_pass/total*100:.1f}%)")
        else:
            print()
        print(f"  Deterministic FAIL:  {det_fail}", end="")
        if total:
            print(f" ({det_fail/total*100:.1f}%)")
        else:
            print()
        print(f"  Needs model (skip):  {det_skip}", end="")
        if total:
            print(f" ({det_skip/total*100:.1f}%)")
        else:
            print()

        if tried > 0:
            print(f"\n  Accuracy when tried: {det_pass}/{tried} = {det_pass/tried*100:.1f}%")
        print(f"  Model accuracy (benchmark): 91.7%")

        print(f"\n--- By edit type ---")
        for et, counts in sorted(by_edit_type.items()):
            t = counts["pass"] + counts["fail"] + counts["skip"]
            tried_et = counts["pass"] + counts["fail"]
            acc = f"{counts['pass']/tried_et*100:.0f}%" if tried_et else "n/a"
            print(
                f"  {et:30s}  pass={counts['pass']:3d}  fail={counts['fail']:3d}  "
                f"skip={counts['skip']:3d}  acc={acc}"
            )

        if failures:
            print(f"\n--- Failures ({len(failures)}) ---")
            for f in failures[:20]:
                print(
                    f"  #{f['index']} {f['language']:12s} {f['edit_type']:25s} "
                    f"symbol={f['symbol']}"
                )

        print(f"\n--- Summary ---")
        if tried > 0:
            accuracy = det_pass / tried * 100
            print(f"Deterministic: {det_pass}/{tried} = {accuracy:.1f}% when it tries")
        print(f"Model (benchmark): 143/156 = 91.7%")
        print(f"Deterministic skips (model needed): {det_skip}/{total}")
        if det_pass > 0:
            print(f"Free edits (0 tokens): {det_pass}/{total} = {det_pass/total*100:.1f}%")
