"""Step D2 — the 100MB REAL-LLM text/markdown stress suite (plan Phase D).

MISSION (plan Step D2): "enables 100MB txt/md" — a ~100MB structureless
``.txt`` and a ~100MB grammar-backed ``.md``, edited through the REAL
trained fastedit mlx-8bit model (``llm_fixtures.real_engine``), proven
byte-perfect. No fake anywhere on the LLM path (plan req. 1).

Cases (each marked ``llm`` and ``stress``, gated on ``FASTEDIT_RUN_STRESS=1``):

a. **100MB txt — deep-offset paragraph revision via ONE anchor window**
   (REAL model): the marker-free bracketed insertion anchored inside a
   paragraph at ~1/2 of a ~1.9M-line shift log. Asserts the D2 WINDOW path
   ran (``chunk_regions`` equals the locator's own anchor-derived window —
   never the whole file, never a whole-file-gate refusal), the REAL model
   ran (``model_tokens > 0``), the untouched head/tail are byte-identical
   (policy §0(b)), and the full-file golden is byte-exact against the
   independent line-splice arithmetic within bounded OUTER attempts
   (non-convergence is a product defect, reported with ``first_diff_tag``).

   MEASURED OP CHOICE: the snippet is the D1-measured MARKER-FREE shape.
   The marker-bearing paragraph-replace idiom was probed deterministically
   against the real model across window budgets 20-40, payload styles and
   corrective notes and echoes the preservation marker into the prose
   output on EVERY attempt (the D1 docstring's measured echo failure
   mode); the marker-free bracketed insertion converges. The whole-file
   trait gate's own predicate is re-run on the accepted merge so the gate's
   engagement is directly observable.

b. **100MB txt — undo full stack byte-exact** (the C2 (d) MCP pattern):
   two anchored appends at distant paragraphs through ``tools_edit``
   (real writes, real 100MB backups), then two ``tools_ast.fast_undo``
   calls restore the post-edit-1 state and the ORIGINAL bytes exactly, and
   the ledger is then empty (fail-loud).

c. **100MB .md — the AST path** (grammar-backed since B2): a corpus
   section (heading-anchored ``section`` symbol) replaced via
   ``replace=<heading>``. Asserts WHICH path ran — ``chunk_regions`` is the
   manifest section span tagged with the symbol name, never a
   ``<text anchor>`` window — and that the edit was DETERMINISTIC
   (``model_tokens == 0``: the direct-swap fast path), with the oracle's
   byte-exact golden, full-file relative parse validity, and the oracle's
   recorded inverse op regenerating the ORIGINAL bytes from the real
   pipeline output (plan req. 3).

The golden oracle is C1's INDEPENDENT line-splice arithmetic
(``tests/corpus.py`` + explicit splices on manifest-recorded spans); it
never imports fastedit. The markdown dialect was added to the corpus's
declarative per-language table for this suite (heading + stride-drawn
prose + one name-embedding fenced block per section — sections stay far
below the chunk locator's parent-snap cap).

Memory & runtime discipline (measured on this machine, M-series):
* corpus build (~100MB): ~1 s and a ~0.6 GB transient per source; each
  source is built ONCE per module run via module-scoped fixtures and
  released (cache pop + ``gc``) before the next build, so at most one
  100MB source is alive at a time.
* the txt edits are window-path: no AST parse anywhere; the heavy fixed
  costs are the 100MB splices/reads/writes and the assembly trait gate
  (one full-file trait pass per edit). The md case pays the AST path's
  parses (``get_ast_map_from_source`` + the relative parse gate's
  ``parse_diagnostics`` ×2 on the full file).
* per-case wall times print as ``[D2 stress] ...`` lines (run pytest with
  ``-s``); tmp files live under pytest's ``tmp_path``, edit backups in
  conftest's session-isolated ``FASTEDIT_BACKUP_DIR``.
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import Self

import corpus
import pytest
from corpus import RECIPE_TEXT
from llm_fixtures import (
    first_diff_tag,
    metrics_tag,
    require_stress_env,
    run_real_edit,
)

pytestmark = [pytest.mark.llm, pytest.mark.stress]

TARGET_BYTES = 100_000_000
"""The stress size: ~100MB per source (plan req. 2)."""

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b)).

Each outer attempt is a COMPLETE real edit; non-convergence after the
bound is reported as a product defect (plan §0), never absorbed as
flakiness."""

_TXT_SEED = "d2-txt-100mb"
_MD_SEED = "d2-md-100mb"

_PAYLOAD_LINES = 4
"""The anchored appends' declared payload size (a four-line addendum)."""


class _StopWatch:
    """Print one per-case wall-time line (the C2 runtime-discipline print)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        print(
            f"[D2 stress] {self.label}: {time.perf_counter() - self._t0:.1f}s",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Corpus fixtures — each 100MB source built ONCE per module run
# ---------------------------------------------------------------------------

_CASE_CACHE: dict[str, corpus.CorpusCase] = {}
"""Holds at most ONE 100MB source at a time (see the fixture teardowns)."""


def _paragraph_at(case: corpus.CorpusCase, fraction: float) -> corpus.SymbolSpan:
    """The manifest paragraph nearest ``fraction`` of the file (deterministic)."""
    manifest = case.source.manifest
    return manifest.symbols[int(manifest.symbol_count * fraction)]


def _append_payload(span: corpus.SymbolSpan) -> str:
    """The declared payload: four per-span-distinct addendum lines.

    Two measured constraints shape it (deterministic greedy probes):

    * every line embeds the span's own name, so two appends at DIFFERENT
      paragraphs declare fully disjoint payloads — a shared payload line
      would exist in the post-edit-1 file (inserted by edit 1) and the D2
      anchor matcher would then anchor edit 2's window on EDIT 1's
      insertion instead of the target paragraph;
    * the four lines carry distinct natural tails — a payload of
      near-identical lines is the C3 lossy-compression trap inside the
      snippet itself, and a rigid ``note N:`` sequence invites the model to
      continue the pattern with its own invented line (both measured as
      battery rejections).
    """
    tails = (
        "the duty supervisor signed the corrected figures for this entry",
        "Dana re-checked the pallet counts before the second shift started",
        "Luis filed the note with the regional desk this morning",
        "the overflow cage list moved onto the shared drive for good",
    )
    return "".join(
        f"{span.name} addendum: {tails[j]}.\n"
        for j in range(_PAYLOAD_LINES)
    )


def _bracketed_insert_snippet(
    source_lines: list[str], span: corpus.SymbolSpan, payload: str,
) -> str:
    """The measured converging marker-free bracketed insertion.

    The paragraph's last line (the position anchor ADJACENT to the
    insertion point), the declared payload, ONE declared blank seam line,
    then the next paragraph's entry line — the corpus's one file-wide
    unique line per paragraph, which is what the D2 anchor matcher anchors
    the window on. The blank is DECLARED because the measured model drops
    an undeclared blank at the payload's tail seam.
    """
    anchor_last = source_lines[span.end_line - 1].rstrip("\r\n")
    next_anchor = source_lines[span.end_line + 1].rstrip("\r\n")
    assert "shift log entry" in next_anchor, (
        "fixture invariant: the next paragraph's entry line follows the span"
    )
    return f"{anchor_last}\n{payload}\n{next_anchor}\n"


def _golden_insert_after(
    lines: list[str], span: corpus.SymbolSpan, payload: str,
) -> str:
    """Independent golden: the payload spliced in before the following
    blank separator (the declared blank maps onto the original's)."""
    return "".join(lines[: span.end_line] + [payload] + lines[span.end_line:])


def _expected_window(source: str, snippet: str) -> tuple[int, int]:
    """The single window the D2 locator must produce for this snippet.

    Computed through the locator's OWN window function — the test pins the
    real cut (anchors, cluster fit, content-edge trim), never a second
    copy of the arithmetic.
    """
    from fastedit.inference.chunk_locator import _text_anchor_windows

    windows = _text_anchor_windows(snippet, source.splitlines())
    assert len(windows) == 1, (
        f"fixture invariant: the snippet must yield exactly one window, "
        f"got {windows}"
    )
    return windows[0]


def _run_until_converged(check, real_engine, tmp_path, name, source, snippet):
    """Run the real edit with bounded outer attempts (plan §0 policy (b)).

    Each outer attempt is a COMPLETE real edit (fresh model calls, fresh
    validation, fresh 100MB write); non-convergence after the bound is a
    product defect per the plan, so the last failure is re-raised with
    metrics.
    """
    last_error: AssertionError | None = None
    last_metrics = ""
    path = tmp_path / f"{name}.txt"
    path.write_text(source, encoding="utf-8")
    for attempt in range(1, OUTER_ATTEMPTS + 1):
        run = run_real_edit(
            source,
            snippet,
            file_path=str(path),
            language=None,
            engine=real_engine,
        )
        try:
            check(run)
            return run
        except AssertionError as exc:
            last_error = exc
            last_metrics = (
                f"[outer attempt {attempt}/{OUTER_ATTEMPTS}] {metrics_tag(run)}"
            )
    raise AssertionError(
        f"edit did not converge within {OUTER_ATTEMPTS} outer attempts — "
        f"non-convergence is a product defect (plan §0): {last_error}\n"
        f"{last_metrics}"
    )


@pytest.fixture(scope="module")
def txt_case():
    """The 100MB text corpus (module-scoped, built once, released after)."""
    require_stress_env()  # belt-and-braces: skip before building anything
    if "txt" not in _CASE_CACHE:
        with _StopWatch("build 100MB text corpus"):
            _CASE_CACHE["txt"] = corpus.build_corpus_case(
                "text", TARGET_BYTES, seed=_TXT_SEED, recipe=RECIPE_TEXT,
            )
    yield _CASE_CACHE["txt"]
    _CASE_CACHE.pop("txt", None)
    gc.collect()


@pytest.fixture(scope="module")
def md_case():
    """The 100MB markdown corpus (module-scoped, built once, released)."""
    require_stress_env()
    if "md" not in _CASE_CACHE:
        with _StopWatch("build 100MB markdown corpus"):
            _CASE_CACHE["md"] = corpus.build_corpus_case(
                "markdown", TARGET_BYTES, seed=_MD_SEED,
                kinds=("replace_symbol_body",),
            )
    yield _CASE_CACHE["md"]
    _CASE_CACHE.pop("md", None)
    gc.collect()


# ---------------------------------------------------------------------------
# (a) 100MB txt — deep-offset paragraph revision via ONE anchor window
# ---------------------------------------------------------------------------


def test_stress_100mb_txt_paragraph_revision_via_window_is_byte_exact(
    real_engine, tmp_path, txt_case,
):
    """A paragraph at ~1/2 of a ~1.9M-line txt, revised through ONE D2
    anchor window; every other byte identical to the original."""
    require_stress_env()
    case = txt_case
    source = str(case.source)
    line_count = len(source.splitlines())
    assert line_count >= 1_000_000, (
        f"fixture drifted: expected a multi-million-line corpus, got "
        f"{line_count}"
    )
    span = _paragraph_at(case, 0.5)
    assert span.start_line > line_count // 3, (
        "fixture invariant: the edit sits at a DEEP offset"
    )
    payload = _append_payload(span)
    orig_lines = source.splitlines(keepends=True)
    snippet = _bracketed_insert_snippet(orig_lines, span, payload)
    golden = _golden_insert_after(orig_lines, span, payload)
    window = _expected_window(source, snippet)
    shift = payload.count("\n")

    def check(run):
        result = run.result
        m = metrics_tag(run)
        # The REAL model ran.
        assert run.merge_results, (
            f"engine.merge_auto was never invoked — no real LLM ran: {m}"
        )
        assert any(r.tokens_generated > 0 for r in run.merge_results), (
            f"model generated zero tokens — no real inference ran: {m}"
        )
        assert result.model_tokens > 0, f"pipeline accounted zero tokens: {m}"
        # The D2 WINDOW path ran: ONE anchor-derived window, accepted.
        assert result.chunk_regions == [window], (
            f"the edit must run through ONE D2 text-anchor window {window}, "
            f"got {result.chunk_regions} | {m}"
        )
        assert result.chunks_used == 1, f"{m}"
        assert result.chunks_rejected == 0, (
            f"the battery rejected the merge — kept the original: {m}"
        )
        assert result.parse_valid is True, f"{m}"
        assert result.retries <= 3, (
            f"retry-until-valid needed an unreasonable budget: {m}"
        )
        # The assembly trait gate's own predicate on the accepted merge
        # (the exact whole-file arithmetic the pipeline runs post-assembly).
        from fastedit.inference.chunked_merge import _derived_text_op
        from fastedit.text_heuristics import (
            TOLERANCE_MODEL_PROSE,
            validate_text_output,
        )

        op, layout_slack, removable = _derived_text_op(source, snippet)
        ok, reason = validate_text_output(
            source, op, result.merged_code, TOLERANCE_MODEL_PROSE,
            layout_slack=layout_slack, removable_traits=removable,
        )
        assert ok is True, (
            f"the assembly trait gate's predicate failed on the accepted "
            f"merge: {reason} | {m}"
        )
        # Untouched regions byte-identical (policy §0(b)): head before the
        # window and tail after it, at their post-edit positions.
        merged_lines = result.merged_code.splitlines(keepends=True)
        assert merged_lines[: window[0] - 1] == orig_lines[: window[0] - 1], (
            f"untouched head changed: {m}"
        )
        assert merged_lines[window[1] + shift:] == orig_lines[window[1]:], (
            f"untouched tail changed: {m}"
        )
        # THE golden assertion: full-file byte-exactness.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}"
        )

    with _StopWatch("(a) 100MB txt paragraph revision via window"):
        _run_until_converged(
            check, real_engine, tmp_path, "stress-txt-100mb", source, snippet,
        )


# ---------------------------------------------------------------------------
# (b) 100MB txt — undo full stack byte-exact (MCP harness pattern)
# ---------------------------------------------------------------------------


def test_stress_100mb_txt_two_anchored_appends_and_undo_full_stack(
    real_engine, mcp_harness, tmp_path, txt_case,
):
    """Two anchored appends at distant paragraphs through the MCP full-stack
    path (real writes, real 100MB backups), then two undos restore every
    intermediate state byte-exact — the C2 N-deep proof on the AST-less
    windowed path.

    Each anchored edit runs with bounded OUTER attempts (plan §0 policy
    (b)): a refused edit leaves the file byte-identical — the pipeline
    keeps the original when its battery exhausts and refuses the write —
    so re-issuing the same tool call is a clean outer attempt. The
    measured failure mode the retries absorb is the real model's
    occasional seam noise on a ~40-line prose window (an invented lead-in
    line around the declared payload), which the battery rightly rejects.
    """
    require_stress_env()
    from fastedit.mcp import tools_ast, tools_edit

    case = txt_case
    source = str(case.source)
    orig_lines = source.splitlines(keepends=True)

    span_a = _paragraph_at(case, 1 / 3)
    span_b = _paragraph_at(case, 2 / 3)
    assert span_a.end_line < span_b.start_line - 8, (
        "fixture invariant: the two append targets are distant paragraphs"
    )
    # Both snippets are built from the ORIGINAL lines: their anchors are
    # CONTENT lines (unique entry lines / the paragraph's own body), so the
    # same bytes anchor the edit on any state of the file.
    payload_a = _append_payload(span_a)
    payload_b = _append_payload(span_b)
    snippet_a = _bracketed_insert_snippet(orig_lines, span_a, payload_a)
    snippet_b = _bracketed_insert_snippet(orig_lines, span_b, payload_b)
    golden_a = _golden_insert_after(orig_lines, span_a, payload_a)
    # Edit 2's splice point sits in the POST-EDIT-1 file: everything after
    # span_a shifted down by edit 1's line delta (the payload's length).
    lines_a = golden_a.splitlines(keepends=True)
    golden_b = "".join(
        lines_a[: span_b.end_line + _PAYLOAD_LINES] + [payload_b]
        + lines_a[span_b.end_line + _PAYLOAD_LINES:],
    )

    target = tmp_path / "shiftlog_100mb.txt"
    target.write_bytes(source.encode("utf-8"))
    original_bytes = source.encode("utf-8")

    def edit_until_applied(snippet: str, attempts: int = OUTER_ATTEMPTS) -> str:
        """One anchored edit with bounded outer attempts (plan §0 policy (b)).

        A refused edit leaves the file byte-identical (the pipeline keeps
        the original on battery exhaustion and refuses the write), so
        re-issuing the SAME tool call is a clean outer attempt: fresh model
        calls, fresh validation, same declared op.
        """
        response = ""
        for attempt in range(1, attempts + 1):
            response = asyncio.run(tools_edit.fast_edit(
                file_path=str(target), edit_snippet=snippet,
            ))
            if response.startswith(f"Applied edit to {target}"):
                return response
        return response

    with _StopWatch("(b) 100MB txt two anchored appends + undo full stack"):
        response = edit_until_applied(snippet_a)
        assert response.startswith(f"Applied edit to {target}"), response
        assert "rejected" not in response and "Error" not in response, response
        got = target.read_bytes().decode("utf-8")
        assert got == golden_a, (
            f"edit 1 did not write the golden bytes: "
            f"{first_diff_tag(golden_a, got)}"
        )

        response = edit_until_applied(snippet_b)
        assert response.startswith(f"Applied edit to {target}"), response
        assert "rejected" not in response and "Error" not in response, response
        got = target.read_bytes().decode("utf-8")
        assert got == golden_b, (
            f"edit 2 did not write the composed golden bytes: "
            f"{first_diff_tag(golden_b, got)}"
        )

        # Undo #1 → the post-edit-1 state, byte-exact at 100MB.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes().decode("utf-8") == golden_a, (
            "undo #1 did not restore the post-edit-1 state byte-exactly: "
            f"{first_diff_tag(golden_a, target.read_bytes().decode('utf-8'))}"
        )

        # Undo #2 → the ORIGINAL bytes, exactly.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes() == original_bytes, (
            "undo #2 did not restore the original file byte-exactly: "
            f"{first_diff_tag(source, target.read_bytes().decode('utf-8'))}"
        )

        # The ledger is empty — a third undo fails loud, changes nothing.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith("Error: no undo history"), response
        assert target.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# (c) 100MB .md — the AST path (grammar-backed since B2)
# ---------------------------------------------------------------------------


def test_stress_100mb_md_section_replace_takes_the_ast_path(
    real_engine, tmp_path, md_case,
):
    """A markdown section replaced via ``replace=<heading>`` at 100MB.

    md is grammar-backed since B2, so the edit anchors on the section's
    HEADING — ``replace=<heading text>`` resolves through the markdown
    AST's heading-named ``section`` symbols, which the structureless path
    cannot do. WHICH path ran is asserted from the result's own shape: the
    grammar-backed direct-swap fast path reports ``chunks_used == 0`` /
    empty regions / ``model_tokens == 0`` / ``parse_valid is True`` (the
    RELATIVE md parse gate only runs for grammar-backed languages — the
    D2 window path never runs it), where the window path would report one
    ``<text anchor>`` region with model tokens and the model chunk path
    one AST region with model tokens. The golden is the independent
    oracle's byte-exact expectation, and the oracle's recorded inverse op
    regenerates the ORIGINAL bytes from the real pipeline output
    (plan req. 3).
    """
    require_stress_env()
    case = md_case
    source = str(case.source)
    manifest = case.source.manifest
    section_count = manifest.symbol_count
    assert section_count >= 100_000, (
        f"fixture drifted: expected a 100MB section corpus, got "
        f"{section_count} sections"
    )
    (op,) = case.ops
    assert op.kind == "replace_symbol_body"
    span = manifest.span(op.symbol)
    # Deep offset: the replaced section sits past the file's first third.
    assert span.start_line > len(source.splitlines()) // 3, (
        "fixture invariant: the replaced section sits at a deep offset"
    )
    golden = corpus.apply_op_oracle(source, op)

    def check(run):
        result = run.result
        m = metrics_tag(run)
        # WHICH path ran: the grammar-backed direct swap — zero chunks,
        # zero model tokens, and parse_valid computed by the RELATIVE md
        # parse gate (a structureless edit never runs that gate).
        assert result.chunk_regions == [], (
            f"the AST direct swap reports no chunk regions, got "
            f"{result.chunk_regions} | {m}"
        )
        assert result.chunks_used == 0, f"{m}"
        assert result.chunks_rejected == 0, f"{m}"
        assert result.parse_valid is True, f"{m}"
        # Deterministic: the direct swap consumed ZERO model tokens — a
        # fake engine could not prove this (the C2 (b) convention).
        assert result.model_tokens == 0, (
            f"the AST direct swap must be zero-token, got {m}"
        )
        # THE golden assertion: full-file byte-exactness vs the oracle.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}"
        )
        # Backward reconstruction (plan req. 3): the oracle's recorded
        # inverse op restores the ORIGINAL bytes from the real output.
        restored = corpus.apply_line_op(
            result.merged_code, corpus.inverse_op(op),
        )
        assert restored == source, (
            "the inverse op did not regenerate the original bytes: "
            f"{first_diff_tag(source, restored)}"
        )

    with _StopWatch("(c) 100MB md section replace via the AST path"):
        run = run_real_edit(
            source,
            op.new_text,
            file_path=str(tmp_path / "operations_100mb.md"),
            language="markdown",
            engine=real_engine,
            replace=op.symbol,
        )
        check(run)
