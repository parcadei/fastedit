"""Step D2 — text chunking through the REAL model (req. 1/2/9, llm tier).

Marked ``llm`` — deselected from the default tier; run explicitly with
``uv run pytest tests/test_real_llm_text_chunks.py -m llm``. There is NO
fake engine here: every test drives the trained fastedit mlx-8bit model
through :func:`llm_fixtures.run_real_edit` / the MCP full-stack harness
with ``language=None`` — the AST-less path Step D2 built (unique snippet
anchor → window chunk within the measured context budget).

WHAT D2 CHANGED FOR THE REAL TIER: before D2 a structureless file above
150 lines was refused outright (the whole-file gate); the D1 real tests
had to stay ≤37 lines. Here every fixture is thousands of lines and the
model only ever sees a ≤40-line window (``_MAX_TEXT_CHUNK_LINES``).

MEASURED MODEL BEHAVIOR this file pins (deterministic greedy probes
against the real mlx-8bit model, documented with the numbers that forced
each shape — the C3 probe doctrine):

* the D1-measured marker-FREE bracketed insertion (anchor adjacent to the
  insertion point + declared payload + next anchor) converges through the
  span-local battery and lands byte-exact vs the independent golden;
* the N-deep undo stack below rides the same measured shape through the
  MCP full-stack path (real writes, real backups);
* GIGO (req. 9): a pre-existing DUPLICATED paragraph elsewhere in the
  file survives byte-exact — the duplicate's body lines can never anchor
  a window (uniqueness matcher), and the trait/content batteries keep the
  model from "helpfully" deduplicating them.

MEASURED SHAPE CONSTRAINTS the snippets below honour (each one was a
deterministic non-convergence before it was fixed):

* **Prose merges must be marker-free.** The D1 measured doctrine (the
  model echoes preservation markers verbatim into prose output) re-measured
  for the windowed path: the marker-bearing paragraph-replace idiom echoed
  the marker on EVERY attempt across window budgets 20-40, payload styles
  and corrective notes — while the marker-free bracketed insertion
  converged. The ops here are therefore insertions.
* **Anchors must be file-wide unique.** Body sentences repeat across
  distant paragraphs; the D2 anchor matcher rightly skips them, and a
  snippet that OPENS on one mis-aligned the model (it matched two window
  lines and ate both). The bracketed snippets open on the line ADJACENT
  to the insertion point and close on the ENTRY LINE — the corpus's one
  file-wide-unique line per paragraph (see ``tests/corpus.py``).
* **The window the model re-emits may not repeat a line.** A repetitive
  chunk makes the model lossily compress it (echoed markers, dropped
  tails) and the battery rightly rejects every attempt — measured 4-10
  repeated lines per window under free ``rng.choice`` draws, 9/9 merge
  rejections; the corpus's stride-block draw
  (``tests/corpus.py::_TEXT_TEMPLATE_STRIDE``) makes every window
  repeat-free by construction, after which the same idioms converge.
* **Payload lines must be distinct from each other** — a payload of
  near-identical lines is the same lossy-compression trap inside the
  snippet itself.

Every case asserts the D2 WINDOW path ran (``chunk_regions`` is the
anchor-derived window, never the whole file), the REAL model ran, the
D2 assembly trait gate's own predicate accepts the output (the exact
whole-file arithmetic of ``_derived_text_op`` + ``validate_text_output``
— the gate the pipeline itself runs post-assembly), and the golden is
byte-exact within bounded outer attempts (plan §0 policy (b);
non-convergence is a product defect, reported with ``first_diff_tag``).

The corpus is the seeded text dialect of tests/corpus.py
(``recipe="text"``): blank-line separated paragraphs whose entry lines
are file-wide unique and whose windows are repeat-free by construction.
"""

from __future__ import annotations

import asyncio

import corpus
import pytest
from corpus import RECIPE_TEXT
from llm_fixtures import first_diff_tag, metrics_tag, run_real_edit

pytestmark = pytest.mark.llm

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b))."""

_PAYLOAD_LINES = 4
"""The anchored appends' declared payload size (a four-line addendum)."""


# ---------------------------------------------------------------------------
# Fixtures — the seeded text corpus, built once per module
# ---------------------------------------------------------------------------

_CASE_CACHE: dict[int, corpus.CorpusCase] = {}


def _text_case(target_bytes: int) -> corpus.CorpusCase:
    if target_bytes not in _CASE_CACHE:
        _CASE_CACHE[target_bytes] = corpus.build_corpus_case(
            "text", target_bytes, seed=f"d2-real-{target_bytes}",
            recipe=RECIPE_TEXT,
        )
    return _CASE_CACHE[target_bytes]


@pytest.fixture(scope="module")
def case_2000():
    """A ~3 500-line structureless txt (the (a)/(c) fixture)."""
    return _text_case(200_000)


@pytest.fixture(scope="module")
def case_5000():
    """A ~9 000-line structureless txt (the (b) undo-stack fixture)."""
    return _text_case(500_000)


def _paragraph_at(case: corpus.CorpusCase, fraction: float) -> corpus.SymbolSpan:
    """The manifest paragraph nearest ``fraction`` of the file (deterministic)."""
    manifest = case.source.manifest
    return manifest.symbols[int(manifest.symbol_count * fraction)]


def _append_payload(span: corpus.SymbolSpan) -> str:
    """The declared payload: a four-line, per-line-distinct addendum.

    Every payload line is DISTINCT — the C3 lossy-compression doctrine
    applies to the declared lines too: a payload of near-identical lines
    ("line 1 EDITED...", "line 2 EDITED...", ...) makes the model lose its
    place in its own copy and the battery rightly rejects every attempt.
    """
    return "".join(
        f"Revision addendum for {span.name}, note {j}: the duty supervisor "
        f"signed the corrected figures for this entry.\n"
        for j in range(1, _PAYLOAD_LINES + 1)
    )


def _bracketed_insert_snippet(
    case: corpus.CorpusCase, span: corpus.SymbolSpan, payload: str,
) -> str:
    """The measured converging marker-free bracketed insertion.

    The paragraph's last line (the position anchor ADJACENT to the
    insertion point — measured: a payload bracketed only by distant anchors
    lands in the wrong place), the declared payload, ONE declared blank
    seam line, then the next paragraph's entry line. The blank is DECLARED
    because the measured model drops an undeclared blank at the payload's
    tail seam (the layout it re-emits follows the snippet's own); the
    window-edge blanks the old shape also lost are gone entirely since the
    locator trims windows to content lines
    (``chunk_locator._trim_window_to_content``). The opening anchor is a
    body line, which the D2 anchor matcher skips (body sentences repeat
    file-wide by design) — the WINDOW then anchors on the unique next
    entry line, which still contains the whole target paragraph; inside
    the repeat-free window the opening line is unambiguous to the model.
    """
    lines = case.source.splitlines(keepends=True)
    anchor_last = lines[span.end_line - 1].rstrip("\r\n")
    next_anchor = lines[span.end_line + 1].rstrip("\r\n")
    return f"{anchor_last}\n{payload}\n{next_anchor}\n"


def _golden_insert_after(
    lines: list[str], span: corpus.SymbolSpan, payload: str,
) -> str:
    """Independent golden: the payload spliced after the paragraph's body.

    Measured (deterministic greedy probes): the bracketed insertion
    converges BYTE-EXACT with no seam blank on the repeat-free window —
    unlike the tail-anchored append, whose measured shape adds one layout
    blank at the file's end.
    """
    return "".join(
        lines[: span.end_line] + [payload] + lines[span.end_line:],
    )


def _expected_window(source: str, snippet: str) -> tuple[int, int]:
    """The single window the D2 locator must produce for this snippet.

    Computed through the locator's OWN window function — the test pins the
    real cut (anchors, cluster fit and the content-edge trim), never a
    second copy of the arithmetic.
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

    Each outer attempt is a COMPLETE real edit; non-convergence after the
    bound is a product defect per the plan, so the last failure is
    re-raised with metrics.
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


def _assert_window_path_ran(source, snippet, run, m):
    """The D2 window path ran: the region is the anchor-derived window."""
    result = run.result
    window = _expected_window(source, snippet)
    assert result.chunk_regions == [window], (
        f"the edit must run through ONE D2 text-anchor window {window}, "
        f"got {result.chunk_regions} | {m}"
    )
    assert result.chunks_used == 1, f"{m}"
    assert result.chunks_rejected == 0, (
        f"the battery rejected the merge — kept the original: {m}"
    )
    assert result.parse_valid is True, f"{m}"


def _assert_model_ran(run, m):
    assert run.merge_results, (
        f"engine.merge_auto was never invoked — no real LLM ran: {m}"
    )
    assert any(r.tokens_generated > 0 for r in run.merge_results), (
        f"model generated zero tokens — no real inference ran: {m}"
    )
    assert run.result.model_tokens > 0, f"pipeline accounted zero tokens: {m}"


def _assert_assembly_battery_accepted(source, snippet, run, m):
    """The D2 assembly trait gate's own predicate on the accepted merge.

    The battery is internal, but its engagement is directly observable:
    the pipeline accepted the merge and the exact whole-file arithmetic
    the assembly gate runs (_derived_text_op on the FULL original +
    snippet, the model-prose tolerance) accepts the output.
    """
    from fastedit.inference.chunked_merge import _derived_text_op
    from fastedit.text_heuristics import (
        TOLERANCE_MODEL_PROSE,
        validate_text_output,
    )

    op, layout_slack, removable = _derived_text_op(source, snippet)
    ok, reason = validate_text_output(
        source, op, run.result.merged_code, TOLERANCE_MODEL_PROSE,
        layout_slack=layout_slack, removable_traits=removable,
    )
    assert ok is True, (
        f"the D2 assembly trait gate's predicate failed on the accepted "
        f"merge: {reason} | {m}"
    )


def _assert_no_marker_leak(run, m):
    assert "# ... existing code ..." not in run.result.merged_code, (
        f"preservation marker leaked into the prose file: {m}\n"
        f"{run.result.merged_code}"
    )


# ---------------------------------------------------------------------------
# (a) ~3 500-line txt: mid-file paragraph revision via ONE window chunk
# ---------------------------------------------------------------------------


def test_real_2000_line_paragraph_revision_via_window_is_byte_exact(
    real_engine, tmp_path, case_2000,
):
    """A mid-file paragraph of a ~3 500-line txt is revised through ONE
    window chunk (a revision addendum anchored inside the paragraph);
    every other byte identical to the original.

    MEASURED OP CHOICE: the edit is the marker-FREE bracketed insertion —
    the D1 measured shape doctrine (tests/test_real_llm_text.py: the model
    echoes preservation markers verbatim into prose output on every
    attempt). Re-measured for the windowed path with deterministic greedy
    probes (Step D2): the marker-bearing paragraph-replace idiom echoes
    the marker on EVERY attempt across window budgets 20-40, payload
    styles and corrective notes, while the marker-free bracketed
    insertion converges. The op therefore inserts the paragraph's
    revision addendum instead of replacing its body — the window path,
    the deep offset, the single-window cut and the byte-exact golden are
    the properties under test.
    """
    case = case_2000
    source = str(case.source)
    line_count = len(source.splitlines())
    assert line_count >= 1800, (
        f"fixture drifted: expected ~3 500 lines, got {line_count}"
    )
    span = _paragraph_at(case, 0.5)
    payload = _append_payload(span)
    snippet = _bracketed_insert_snippet(case, span, payload)
    orig_lines = source.splitlines(keepends=True)
    golden = _golden_insert_after(orig_lines, span, payload)
    window = _expected_window(source, snippet)
    shift = payload.count("\n")

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_window_path_ran(source, snippet, run, m)
        _assert_assembly_battery_accepted(source, snippet, run, m)
        _assert_no_marker_leak(run, m)
        assert result.retries <= 3, (
            f"retry-until-valid needed an unreasonable budget: {m}"
        )
        # Untouched regions byte-identical (policy (b)): head before the
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

    _run_until_converged(
        check, real_engine, tmp_path, "replace-2000", source, snippet,
    )


# ---------------------------------------------------------------------------
# (b) ~9 000-line txt: two anchored appends → N-deep undo, full MCP stack
# ---------------------------------------------------------------------------


def test_real_5000_line_two_anchored_appends_and_ndeep_undo(
    real_engine, mcp_harness, tmp_path, case_5000,
):
    """Two anchored appends through the MCP full-stack path (real writes,
    real backups), then two undos restore every intermediate state
    byte-exact — the C2 N-deep proof on the AST-less windowed path."""
    from fastedit.mcp import tools_ast, tools_edit

    case = case_5000
    source = str(case.source)
    line_count = len(source.splitlines())
    assert line_count >= 4500, (
        f"fixture drifted: expected ~9 000 lines, got {line_count}"
    )
    orig_lines = source.splitlines(keepends=True)

    def golden_for_append(
        lines: list[str], span: corpus.SymbolSpan, offset: int,
    ) -> tuple[str, str, int]:
        """(snippet, golden, line_delta) for one anchored append.

        The snippet's anchors are matched by CONTENT (unique lines), so
        the same snippet works on any state of the file; the golden's
        splice point sits ``offset`` lines lower per prior edit.
        """
        payload = _append_payload(span)
        snippet = _bracketed_insert_snippet(case, span, payload)
        insert_at = span.end_line + offset  # 0-based splice index
        golden = "".join(lines[:insert_at] + [payload] + lines[insert_at:])
        return snippet, golden, payload.count("\n")

    span_a = _paragraph_at(case, 1 / 3)
    span_b = _paragraph_at(case, 2 / 3)
    assert span_a.end_line < span_b.start_line, (
        "fixture invariant: the two append targets are distant paragraphs"
    )
    snippet_a, golden_a, delta_a = golden_for_append(orig_lines, span_a, 0)
    # Edit 2 is declared against the POST-EDIT-1 file; its splice point
    # shifts by edit 1's line delta.
    lines_a = golden_a.splitlines(keepends=True)
    _snippet_b, golden_b, _delta_b = golden_for_append(
        lines_a, span_b, delta_a,
    )
    snippet_b = _bracketed_insert_snippet(
        case, span_b, _append_payload(span_b),
    )

    target = tmp_path / "append-5000.txt"
    target.write_text(source, encoding="utf-8")
    original_bytes = source.encode("utf-8")

    response = asyncio.run(tools_edit.fast_edit(
        file_path=str(target), edit_snippet=snippet_a,
    ))
    m = f"response={response!r}"
    assert response.startswith(f"Applied edit to {target}"), m
    assert "rejected" not in response and "Error" not in response, m
    got = target.read_bytes().decode("utf-8")
    assert got == golden_a, (
        f"edit 1 did not write the golden bytes: "
        f"{first_diff_tag(golden_a, got)}"
    )

    response = asyncio.run(tools_edit.fast_edit(
        file_path=str(target), edit_snippet=snippet_b,
    ))
    m = f"response={response!r}"
    assert response.startswith(f"Applied edit to {target}"), m
    assert "rejected" not in response and "Error" not in response, m
    got = target.read_bytes().decode("utf-8")
    assert got == golden_b, (
        f"edit 2 did not write the composed golden bytes: "
        f"{first_diff_tag(golden_b, got)}"
    )

    # Undo #1 → the post-edit-1 state, byte-exact at ~9 000 lines.
    response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
    assert response.startswith(f"Reverted {target}"), response
    assert target.read_bytes().decode("utf-8") == golden_a, (
        "undo #1 did not restore the post-edit-1 state byte-exactly"
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
# (c) GIGO: a pre-existing duplicated paragraph survives byte-exact
# ---------------------------------------------------------------------------


def test_real_gigo_duplicated_paragraph_survives_byte_exact(
    real_engine, tmp_path, case_2000,
):
    """req. 9 on the windowed path: the edit revises ONE paragraph; a
    DUPLICATED paragraph elsewhere must survive byte-exact — the trait
    oracle counts it, the content validator keeps it, and the uniqueness
    matcher never anchors a window on the duplicate's body lines."""
    case = case_2000
    source = str(case.source)
    lines = source.splitlines(keepends=True)

    # Duplicate the paragraph at ~3/4: splice a byte-copy of its block
    # after it, separated by the same blank line that separates the
    # original from what follows. The copy's START line is recorded so the
    # survival check is INDEX-based (the corpus's template body lines
    # repeat across distant paragraphs by design — only the entry line is
    # file-wide unique — so a substring-count assertion could never
    # distinguish the copy from that reuse).
    dup = _paragraph_at(case, 0.75)
    dup_block = lines[dup.start_line - 1: dup.end_line]
    separator = lines[dup.end_line]  # the blank line after the paragraph
    assert not separator.strip(), "fixture invariant: blank separator"
    duplicated = "".join(
        lines[: dup.end_line] + [separator] + dup_block + lines[dup.end_line:],
    )
    dup_copy_start = dup.end_line + 2  # 1-based first line of the copy
    dup_copy_end = dup.end_line + 1 + len(dup_block)
    assert duplicated.splitlines(keepends=True)[dup_copy_start - 1] == (
        dup_block[0]
    ), "fixture drift: the copy does not start where it was spliced"

    # The revised paragraph at ~1/2 precedes the duplicate, so the edit's
    # splice shifts every later line by exactly the payload's length — the
    # golden splices at the recorded index and the copy's post-edit index
    # shifts with it.
    span = _paragraph_at(case, 0.5)
    assert span.end_line < dup.start_line, (
        "fixture invariant: the edited paragraph precedes the duplicate"
    )
    payload = _append_payload(span)
    snippet = _bracketed_insert_snippet(case, span, payload)
    dup_lines = duplicated.splitlines(keepends=True)
    golden = _golden_insert_after(dup_lines, span, payload)
    shift = payload.count("\n")

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_window_path_ran(duplicated, snippet, run, m)
        _assert_assembly_battery_accepted(duplicated, snippet, run, m)
        _assert_no_marker_leak(run, m)
        assert result.retries <= 3, m
        # THE GIGO assertion: the duplicated paragraph's copy survives
        # BYTE-EXACT at its recorded post-edit position (the model never
        # "helpfully" deduplicates it), and so does the original block.
        merged_lines = result.merged_code.splitlines(keepends=True)
        assert (
            merged_lines[dup_copy_start - 1 + shift: dup_copy_end + shift]
            == dup_block
        ), (
            f"the duplicated paragraph was mutated (trait mutation): {m}"
        )
        assert (
            merged_lines[dup.start_line - 1 + shift: dup.end_line + shift]
            == dup_block
        ), f"the duplicated paragraph's original block changed: {m}"
        # And the commanded revision landed.
        assert payload in result.merged_code, (
            f"the declared revision did not land: {m}"
        )
        # Golden byte-exactness.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}"
        )

    _run_until_converged(
        check, real_engine, tmp_path, "gigo-dup", duplicated, snippet,
    )
