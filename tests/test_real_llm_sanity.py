"""A1 sanity: one tiny REAL-LLM replace edit on a small Python file.

Marked ``llm`` — deselected from the default tier by pyproject ``addopts``;
run explicitly with ``uv run pytest tests/test_real_llm_sanity.py -m llm``.

There is NO fake engine anywhere in this test. The trained fastedit
mlx-8bit model must actually run for these assertions to mean anything,
which the test proves mechanically: the snippet is a genuine wrap_block
shape (marker nested deeper than the opener — the exact shape locked as
deterministic-declined by
``tests/test_add_guard_marker_position.py::TestGenuineWrapBlockStillFallsThrough``),
so the deterministic text-match editor declines, the direct-swap fast
path is skipped because the snippet carries preservation markers, and
``chunked_merge`` must therefore invoke ``engine.merge_auto`` on the real
model. The test asserts positive ``tokens_generated`` and positive
``latency_ms`` — a stub could satisfy neither honestly.

Assertions follow the plan's non-determinism policy (§0): pipeline gates
(parse-valid, nothing rejected), target-span conformance to the op spec
(body wrapped inside the ``with ledger_lock:`` block), untouched regions
byte-identical to the original, and file-level structural validity via a
fresh :func:`ast.parse` independent of fastedit's own gate. Measured
latency/tokens ride in every failure message via :func:`metrics_tag`.
"""

from __future__ import annotations

import ast

import pytest
from llm_fixtures import first_diff_tag, metrics_tag, run_real_edit

pytestmark = pytest.mark.llm

# Small real file (~30 lines): several symbols, so a replace= edit has a
# meaningful untouched context around the target span.
SOURCE = '''\
#!/usr/bin/env python3
"""Inventory service — tracks stock levels for the warehouse app."""

from collections import OrderedDict


class StockLedger:
    """Keeps per-SKU counts with insertion order preserved."""

    def __init__(self):
        self.entries = OrderedDict()

    def record(self, sku, count):
        self.entries[sku] = self.entries.get(sku, 0) + count


def critical_section(ledger):
    ledger.record("widget", 3)
    ledger.record("gadget", 7)
    return True


def summarize(ledger):
    total = sum(ledger.entries.values())
    return f"total={total}"
'''

# Marker-bearing wrap_block: the model must wrap the existing body inside
# `with ledger_lock:` and re-indent it. The `# ... existing code ...`
# marker makes this shape genuinely deterministic-declined (see module
# docstring), so this snippet is what forces the model path.
SNIPPET = """\
def critical_section(ledger):
    with ledger_lock:
        # ... existing code ...
"""


def test_real_model_runs_tiny_replace_edit(real_engine, tmp_path):
    """One tiny replace= edit through the REAL model, end to end."""
    target = tmp_path / "inventory.py"
    target.write_text(SOURCE, encoding="utf-8")

    run = run_real_edit(
        SOURCE,
        SNIPPET,
        file_path=str(target),
        language="python",
        engine=real_engine,
        replace="critical_section",
    )
    result = run.result
    m = metrics_tag(run)

    # ── 1. The REAL model ran ──────────────────────────────────────────
    # The deterministic editor declines this shape, so merge_auto must
    # have been invoked, with positive tokens and positive latency.
    assert run.merge_results, (
        "engine.merge_auto was never invoked — the edit took a zero-token "
        f"deterministic path, so no real LLM ran: {m}"
    )
    assert any(r.tokens_generated > 0 for r in run.merge_results), (
        f"model generated zero tokens — no real inference ran: {m}"
    )
    assert any(r.latency_ms > 0 for r in run.merge_results), (
        f"zero latency — no real inference ran: {m}"
    )

    # ── 2. Pipeline gates ──────────────────────────────────────────────
    assert result.parse_valid, (
        f"merged output rejected as parse-invalid: {m}\n{result.merged_code}"
    )
    assert result.chunks_rejected == 0, (
        f"chunk(s) rejected by the validator — kept original instead: {m}"
    )
    assert result.model_tokens > 0, f"pipeline accounted zero model tokens: {m}"

    merged = result.merged_code

    # ── 3. Target-span conformance to the op spec ──────────────────────
    # The body must be wrapped inside the with-block, every original body
    # line preserved (re-indented) inside it, nothing duplicated.
    assert "    with ledger_lock:" in merged, (
        f"wrap block missing from the merge: {m}\n{merged}"
    )
    for body_line in (
        'ledger.record("widget", 3)',
        'ledger.record("gadget", 7)',
        "return True",
    ):
        indented = "        " + body_line
        assert indented in merged, (
            f"original body line {body_line!r} not preserved under the "
            f"with-block: {m}\n{merged}"
        )
        assert merged.count(indented) == 1, (
            f"body line {body_line!r} duplicated by the merge: {m}\n{merged}"
        )

    # ── 4. Untouched regions byte-identical (policy (b)) ───────────────
    head, _, _ = SOURCE.partition("def critical_section")
    tail = SOURCE[SOURCE.index("def summarize"):]
    assert merged.startswith(head), (
        f"untouched file header changed: "
        f"{first_diff_tag(head, merged[: len(head)])} | {m}"
    )
    assert merged.endswith(tail), (
        f"untouched file tail changed: "
        f"{first_diff_tag(tail, merged[-len(tail):])} | {m}"
    )

    # ── 5. File-level structural assertions ────────────────────────────
    ast.parse(merged)  # full-file structural validity, independent of fastedit's gate
    assert merged.count("def critical_section") == 1, (
        f"target symbol duplicated or lost: {m}\n{merged}"
    )
    assert "def summarize" in merged, f"untouched symbol lost: {m}\n{merged}"
    assert "# ... existing code ..." not in merged, (
        f"preservation marker leaked into the file: {m}\n{merged}"
    )

    # The disk copy is untouched by design: chunked_merge returns the
    # merged text; writing is the caller's (cli.py's) job.
    assert target.read_text(encoding="utf-8") == SOURCE
