"""Step 17 — anchor disambiguation regressions (B24, B27, B28).

B24 — text_match bound each snippet context line to the FIRST forward
  original occurrence whose indent delta stayed within ±2 of the previous
  anchor's delta. A confidently-wrong-but-consistent repeat (``x = 1`` in
  two functions, a repeated ``return``) dragged the whole edit to the
  wrong region. The binding must consider ALL forward occurrences and
  prefer the one that keeps the snippet's anchor SEQUENCE contiguous with
  its neighbors AND matches the snippet line's indent; when two bindings
  remain equally valid the editor must DECLINE (return None) instead of
  silently picking the first.

B27 — ``_narrow_large_node`` accepted a sliding-window match at a stripped
  score as low as 0.2 and then cut ±padding RAW lines around it — mid-block
  cuts hand the model broken code to "repair". The acceptance threshold is
  a named tunable ≥ 0.6, and cuts happen at AST node edges (the enclosing
  block from the tree-sitter walk); without AST evidence the narrow is
  rejected (full node returned).

B28 — ``_find_insertion_region`` matched each context line to its FIRST
  occurrence and took ``max()`` as the anchor, so a common line steered
  the insertion anywhere. The anchor must be the RAREST context line
  (least frequent, then longest normalized content), and ALL context
  lines must land inside one consistent region — otherwise no region is
  returned and the caller's fallback runs.
"""

from __future__ import annotations

import pytest

from fastedit.inference import chunk_locator, snippet_analysis
from fastedit.inference.ast_utils import ASTNode
from fastedit.inference.chunk_locator import _narrow_large_node, locate_chunks
from fastedit.inference.text_match import deterministic_edit

# ===========================================================================
# B24 — deterministic_edit anchor disambiguation
# ===========================================================================


def test_repeated_context_line_binds_to_contiguous_sequence():
    """A repeated context line binds where the anchor SEQUENCE is contiguous.

    ``x = 1`` occurs in both alpha and beta; ``run(x)`` occurs only in beta,
    directly after beta's ``x = 1``. The old first-forward binding chose
    (alpha's x=1, beta's run(x)) — a binding spanning both functions — and
    the marker section then inserted the new ``telemetry(x)`` line at the
    TOP of that wrong gap, i.e. into alpha. The contiguous binding
    (5, 6) keeps both anchors together in beta.
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
        "    return x\n"
    )
    snippet = (
        "    x = 1\n"
        "    telemetry(x)\n"
        "    # ... existing code ...\n"
        "    run(x)\n"
    )

    result = deterministic_edit(original, snippet)

    assert result is not None, (
        "contiguous binding exists — the editor must apply the edit, "
        "not decline"
    )
    lines = result.splitlines()
    # The edit landed in beta: telemetry sits between beta's two anchors.
    assert lines[5] == "    x = 1"
    assert lines[6] == "    telemetry(x)"
    assert lines[7] == "    run(x)"
    # Alpha is untouched — the old binding spliced telemetry in here.
    assert lines[1] == "    x = 1"
    assert lines[2] == "    return x"
    assert "telemetry" not in "\n".join(lines[:5])


def test_equally_valid_bindings_decline_to_model():
    """Two equally valid contiguous bindings → decline, never pick the first.

    alpha and beta have byte-identical bodies; the snippet's anchors match
    both regions with identical contiguity and indent. The old code
    silently picked alpha (first forward occurrence).
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
    )
    snippet = (
        "    x = 1\n"
        "    probe()\n"
        "    run(x)\n"
    )

    assert deterministic_edit(original, snippet) is None, (
        "two equally valid anchor bindings must decline to the model path"
    )


def test_gate_fails_closed_on_ambiguous_binding():
    """The Step-6 gate consumes the same classifier — ambiguity fails closed.

    ``_deterministic_result_is_faithful`` must never ratify output whose
    anchor partition the editor itself would decline.
    """
    from fastedit.inference.chunked_merge import (
        _deterministic_result_is_faithful,
    )

    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
    )
    snippet = (
        "    x = 1\n"
        "    probe()\n"
        "    run(x)\n"
    )

    assert _deterministic_result_is_faithful(
        original, original, snippet,
    ) is False


def test_identical_indent_occurrence_wins_without_declining():
    """Same content at two indents: the indent-matching occurrence wins.

    ``x = 1`` / ``run(x)`` exist at indent 4 (alpha) and indent 8 (beta's
    guarded block). The snippet is written at indent 8 — both bindings are
    equally contiguous, so the identical-indent preference must resolve the
    tie toward beta WITHOUT declining (the old code bound alpha, the first
    forward occurrence).
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    if guard:\n"
        "        x = 1\n"
        "        run(x)\n"
    )
    snippet = (
        "        x = 1\n"
        "        audit(x)\n"
        "        run(x)\n"
    )

    result = deterministic_edit(original, snippet)

    assert result is not None, (
        "identical-indent preference must resolve the tie, not decline"
    )
    lines = result.splitlines()
    # The edit landed inside beta's guarded block.
    assert lines[6] == "        x = 1"
    assert lines[7] == "        audit(x)"
    assert lines[8] == "        run(x)"
    # Alpha (the shallow-indent occurrence) is untouched.
    assert lines[1] == "    x = 1"
    assert lines[2] == "    run(x)"
    assert "audit" not in "\n".join(lines[:6])


# ===========================================================================
# B27 — _narrow_large_node threshold + AST-edge cuts
# ===========================================================================


def _large_python_source() -> list[str]:
    """A 208-line function: two 100-line for-blocks around scalars.

    Lines (1-indexed): def=1, total=2, phase_one=3, for i=4, step=5..104,
    phase_two=105, for j=106, step2=107..206, audit=207, return=208.
    """
    lines = ["def big():", "    total = 0", "    phase_one()"]
    lines.append("    for i in range(100):")
    lines += [f"        step({i})" for i in range(100)]
    lines.append("    phase_two()")
    lines.append("    for j in range(100):")
    lines += [f"        step2({j})" for j in range(100)]
    lines.append("    audit(total)")
    lines.append("    return total")
    return lines


def _large_node() -> ASTNode:
    return ASTNode(
        name="big", kind="function",
        line_start=1, line_end=208, signature="def big():",
    )


def test_narrow_acceptance_threshold_is_named_and_at_least_0_6():
    """The B27 acceptance threshold is a module-level tunable ≥ 0.6."""
    assert chunk_locator._MIN_NARROW_SCORE >= 0.6


def test_narrow_low_overlap_rejected_returns_full_node():
    """A 0.5-score window must NOT narrow into a mid-block cut.

    The snippet matches only half its window inside the second for-block.
    Pre-fix the 0.2 threshold accepted it and cut (96, 208) — a window
    slicing into surrounding code. Post-fix the narrow is rejected and the
    full node range is returned (the caller keeps the whole function).
    """
    lines = _large_python_source()
    snippet = (
        "        step2(40)\n"
        "        BRAND_NEW_ALPHA = 1\n"
        "        BRAND_NEW_BETA = 2\n"
        "        step2(43)\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (1, 208), (
        f"low-overlap window narrowed to {result}; the 0.2-threshold "
        f"mid-block cut must be rejected (full node returned)"
    )


def test_narrow_high_overlap_cuts_at_block_edges():
    """Positive control: a ≥0.6 window still narrows — at AST node edges.

    The snippet matches the second for-block exactly, so the cut boundaries
    are the block's own edges (106..206), not block±padding raw lines.
    """
    lines = _large_python_source()
    snippet = (
        "        step2(40)\n"
        "        step2(41)\n"
        "        step2(42)\n"
        "        step2(43)\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (106, 206), (
        f"high-overlap window narrowed to {result}; expected the "
        f"enclosing block's AST edges (106, 206)"
    )


# Issue #12 e2e follow-up: the first matched line has NO enclosing block
# (top-level statement of the method body), but a LATER matched line sits
# inside a small try block. The narrow must walk forward through the
# matched lines instead of falling back to the FULL node.


def _ts_like_source_with_top_level_context() -> list[str]:
    """A 208-line function whose tail carries a small try block.

    Lines (1-indexed): def=1, total=2, for i=3 (body 4..103), phase_two=104,
    for j=105 (body 106..202), phase_three=203 (top level, NO block),
    try opener=204, body=205, handler=206..207, return=208.
    The try block spans 204..207 — inside it only ``final_step()`` and
    ``recover()`` are matched lines.
    """
    lines = ["def big():", "    total = 0"]
    lines.append("    for i in range(100):")
    lines += [f"        step({i})" for i in range(100)]
    lines.append("    phase_two()")
    lines.append("    for j in range(100):")
    lines += [f"        step2({j})" for j in range(97)]
    lines.append("    phase_three()")       # top-level, right before the try
    lines.append("    try:")
    lines.append("        final_step()")
    lines.append("    except Boom:")
    lines.append("        recover()")
    lines.append("    return total")
    return lines


def test_narrow_walks_to_a_later_matched_line_when_the_first_has_no_block():
    """The C3 anchor's enclosing-block lookup must not give up on the node.

    The snippet's first matched line is the method's top-level statement
    (no for/if/while/try encloses it — the first-block lookup returns
    None), while a LATER matched line sits inside a small try block. The
    old code returned the FULL node when the FIRST matched line had no
    block; it must walk forward through the remaining matched lines and
    cut at the block the snippet actually targets.
    """
    lines = _ts_like_source_with_top_level_context()
    node = ASTNode(
        name="big", kind="function",
        line_start=1, line_end=len(lines), signature="def big():",
    )
    snippet = (
        "    phase_three()\n"     # top-level statement — NO enclosing block
        "    try:\n"              # the small block the snippet targets
        "        final_step()\n"
        "    except Boom:\n"
        "        recover()\n"
        "        # ... existing code ...\n"
    )

    result = _narrow_large_node(
        node, snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result != (1, len(lines)), (
        f"narrow returned the full node {result} although a later matched "
        f"line pins the try block — the model must never see the whole "
        f"2,499-line symbol for a 5-line anchored edit"
    )
    start, end = result
    assert end - start < 40, (
        f"narrow cut {result} is not a small sub-block; expected the "
        f"try block the snippet's later matched lines pin"
    )


def test_narrow_still_rejects_when_no_matched_line_has_a_block():
    """Negative control for the walk: with NO matched line inside any
    block, the narrow still returns the full node (no raw-line cut)."""
    lines = _ts_like_source_with_top_level_context()
    node = ASTNode(
        name="big", kind="function",
        line_start=1, line_end=len(lines), signature="def big():",
    )
    snippet = (
        "    total = 0\n"
        "    phase_one()\n"
    )

    result = _narrow_large_node(
        node, snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (1, len(lines)), (
        f"no-block snippet narrowed to {result}; the full node must be "
        f"returned when no matched line sits inside a block"
    )


def test_narrow_without_enclosing_block_rejected():
    """No enclosing block → no raw-line fallback window; full node returned.

    The snippet matches the function's top-level statements (lines 2..3),
    which no for/if/while/try encloses. Pre-fix this produced the ±padding
    raw-line window (1, 14), slicing into the first for-block. Without AST
    evidence the narrow must be rejected.
    """
    lines = _large_python_source()
    snippet = (
        "    total = 0\n"
        "    phase_one()\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (1, 208), (
        f"window with no enclosing block narrowed to {result}; the "
        f"raw-line fallback cut must be rejected (full node returned)"
    )


def _deep_two_block_source() -> tuple[list[str], int, int]:
    """A >100-line function with two sibling for-blocks sharing ONE opener.

    The C3 deep-nesting corpus shape: block openers are byte-identical
    (`for _outer in range(2):`), inner openers and body statements differ
    per block. Returns (lines, block_b_start, block_b_end) 1-indexed.
    """
    lines = ["def big():", '    """docstring"""']
    lines.append("    for _outer in range(2):")  # block A opener
    lines.append("        if value_a > 1:")
    lines += ["            acc_a_i1 = 1", "            acc_a_i2 = 2"]
    lines += [f"        acc_a_{k:02d} = {k}" for k in range(1, 61)]
    block_b_start = len(lines) + 1
    lines.append("    for _outer in range(2):")  # block B opener — SAME text
    lines.append("        if value_a > 2:")
    lines += ["            acc_b_i1 = 1", "            acc_b_i2 = 2"]
    lines += [f"        acc_b_{k:02d} = {k}" for k in range(1, 61)]
    lines.append("    return total")
    return lines, block_b_start, block_b_start + 63


def test_narrow_anchors_on_first_matched_line_not_window_start():
    """C3: a leading unmatched snippet line must not slide the cut into the
    WRONG sub-block of a large node.

    The snippet's first line is the function's signature — it matches the
    node only at offset 0 — while the snippet's real context matches the
    SECOND for-block. The best-scoring window therefore starts one line
    ABOVE that block (the signature slot matches nothing there). Anchoring
    the enclosing-block lookup on the window's first line cut block ONE
    (the bystander block the snippet never mentioned); anchoring on the
    FIRST MATCHED line cuts block TWO — the block the snippet edits.
    """
    lines, b_start, b_end = _deep_two_block_source()
    node = ASTNode(
        name="big", kind="function",
        line_start=1, line_end=len(lines), signature="def big():",
    )
    snippet_lines = [
        "def big():",  # the auto-prepended signature (matches only line 1)
        "    for _outer in range(2):",  # block B opener — duplicate text
        "        if value_a > 2:",
        "            acc_b_i1 = 1",
        "            acc_b_i2 = 2",
        "        acc_b_01 = 1",
        "        acc_b_02 = 2",
        "        acc_b_03 = 3",
        "    with audit_lock:",  # the declared new guard
        "        # ... existing code ...",  # marker — excluded from matching
    ]
    snippet = "\n".join(snippet_lines) + "\n"

    result = _narrow_large_node(
        node, snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (b_start, b_end), (
        f"narrow cut {result}; the snippet edits block TWO at lines "
        f"{b_start}-{b_end} — a cut anchored on the window's unmatched "
        f"first line lands on block ONE (lines 3-66) and hands the model "
        f"the wrong sub-block"
    )


# ===========================================================================
# C3 — narrowed replace must not prepend the target's signature
# ===========================================================================


def _deep_two_block_python_file() -> tuple[str, tuple[int, int], tuple[int, int]]:
    """A file whose >100-line function carries two sibling for-blocks.

    Returns (source, block_a_span, block_b_span) 1-indexed inclusive.
    """
    lines, b_start, b_end = _deep_two_block_source()
    return "\n".join(lines) + "\n", (3, b_start - 1), (b_start, b_end)


def test_narrowed_replace_snippet_carries_no_prepended_signature():
    """C3: ``replace=`` on a >100-line symbol narrows the chunk to a
    sub-block — the auto-prepended signature would declare a line that is
    OUTSIDE the chunk the model sees, so the model echoes it into its merge
    and the validator rejects every attempt (the narrowed edit could never
    converge). When the narrow will engage, the signature must NOT be
    prepended, and a faithful sub-block merge must land on the first
    attempt through the (scripted) model path.
    """
    from types import SimpleNamespace

    from fastedit.inference.chunked_merge import chunked_merge

    source, _block_a, block_b = _deep_two_block_python_file()
    b_start, b_end = block_b
    lines = source.splitlines(keepends=True)
    block_lines = lines[b_start - 1: b_end]
    # The block-targeted snippet: the block's opener + first context lines,
    # the new guard, and a preservation marker. NO function signature.
    snippet = (
        "    for _outer in range(2):\n"
        "        if value_a > 2:\n"
        "            acc_b_i1 = 1\n"
        "        acc_b_01 = 1\n"
        "        with audit_lock:\n"
        "            # ... existing code ...\n"
    )
    # A faithful merge of the BLOCK chunk: every original line survives,
    # the body after the last declared context anchor moves one level
    # deeper inside the guard (uniform group shift).
    kept = block_lines[:5]
    wrapped = block_lines[5:]
    merged_chunk = (
        "".join(kept)
        + "        with audit_lock:\n"
        + "".join("    " + line for line in wrapped)
    )
    expected_file = "".join(lines[: b_start - 1]) + merged_chunk + "".join(
        lines[b_end:],
    )

    calls: list[tuple[str, str]] = []

    def merge_fn(chunk, snippet_text, language):
        calls.append((chunk, snippet_text))
        return SimpleNamespace(
            merged_code=merged_chunk, parse_valid=True, tokens_generated=9,
            latency_ms=1.0, truncated=False,
        )

    result = chunked_merge(
        original_code=source,
        snippet=snippet,
        file_path="narrow_probe.py",
        merge_fn=merge_fn,
        language="python",
        replace="big",
    )
    assert calls, "the model path never ran"
    chunk, sent_snippet = calls[0]
    assert "def big():" not in sent_snippet, (
        "the snippet handed to the model carries the auto-prepended "
        "signature — outside the narrowed chunk it declares a line the "
        "merge must not contain, so a narrowed edit can never converge"
    )
    assert chunk.startswith("    for _outer in range(2):"), (
        "the chunk must be the narrowed sub-block, not the whole symbol"
    )
    assert result.chunks_rejected == 0 and result.retries == 0
    assert result.chunk_regions == [block_b], result.chunk_regions
    assert result.merged_code == expected_file


# ===========================================================================
# C3 — the deep-recipe corpus narrows to the manifest-recorded FIRST block
# ===========================================================================


# The deep-narrow probe line per language: the ellipsis phrase rides inside a
# STRING, which drives ``_snippet_has_ellipsis`` (the narrow trigger) while
# the exact-match ``is_marker_line`` still classifies the line as CONTENT.
# (The stress suite's full op shape — including go's measured gofmt guard —
# lives in tests/test_stress_100mb_seams.py's ``_NARROW_TAIL``; the locator
# behavior under test here only depends on the block context + the probe.)
_DEEP_NARROW_PROBE = {
    "python": '        seam_probe = "... existing code ..."\n',
    "go": '\tseamProbe := "... existing code ..."\n',
    "typescript": '  const seamProbe = "... existing code ...";\n',
}


@pytest.mark.parametrize("language", ["python", "go", "typescript"])
def test_deep_recipe_corpus_narrows_to_recorded_first_block(language):
    """C3: on a deep-recipe corpus the ``replace=`` narrow must cut the
    manifest-recorded FIRST block of the targeted symbol.

    The deep corpus's sibling blocks are byte-identical (the B24 tie that
    keeps the deterministic editor off the edit) and the snippet's ellipsis
    phrase hides inside a string literal, so the narrow must engage on the
    phrase (``_snippet_has_ellipsis``), resolve its best-scoring window to
    the FIRST block (strictly-greater score keeps the first offset), and cut
    at that block's AST edges — the same ``result.chunk_regions`` assertion
    the 100MB seams stress makes, pinned here hermetically (no model).
    """
    import corpus

    from fastedit.inference.chunk_locator import _narrow_will_engage
    from fastedit.inference.markers import is_marker_line

    source = corpus.generate_big_source(language, 200_000, recipe="deep")
    manifest = source.manifest
    deep = next(sym for sym in manifest.symbols if sym.blocks)
    blk = deep.blocks[0]
    lines = source.splitlines(keepends=True)
    probe = _DEEP_NARROW_PROBE[language]
    snippet = (
        "".join(ln.rstrip("\r\n") + "\n" for ln in
                lines[blk.start_line - 1: blk.end_line])
        + probe
    )

    # The two load-bearing classifications of the probe line.
    assert "... existing code ..." in snippet
    assert _narrow_will_engage(snippet, deep.line_count), (
        f"{language}: the string-embedded ellipsis phrase must drive the "
        f"narrow on a {deep.line_count}-line symbol"
    )
    assert not is_marker_line(probe.rstrip("\n")), (
        f"{language}: the probe line must stay CONTENT (a declared new "
        f"line), not a preservation marker"
    )

    chunks = locate_chunks(
        snippet, str(source), f"deep_narrow_probe.{manifest.ext}",
        language=language, replace=deep.name,
    )

    assert len(chunks) == 1, (
        f"{language}: expected one narrowed region, got "
        f"{[(c.start_line, c.end_line) for c in chunks]}"
    )
    assert (chunks[0].start_line, chunks[0].end_line) == (
        blk.start_line, blk.end_line,
    ), (
        f"{language}: the narrow must cut the manifest-recorded FIRST block "
        f"{(blk.start_line, blk.end_line)} of '{deep.name}' "
        f"({deep.start_line}-{deep.end_line}, {len(deep.blocks)} recorded "
        f"blocks), got {(chunks[0].start_line, chunks[0].end_line)}"
    )


# ===========================================================================
# C3 — the realign repair must not re-indent the snippet's declared new lines
# ===========================================================================


def test_chunk_repair_leaves_declared_new_lines_untouched():
    """C3 seams stress: the chunk loop's indent repair protects declared lines.

    Measured model failure (go, narrowed 100MB chunk): the model echoed the
    chunk body one level shallow while echoing the snippet's declared new
    tail VERBATIM; the repair's uniform shift then restored the body and
    silently pushed the whole declared tail one level deeper — the new guard
    landed inside the enclosing block (a silent semantic change) and the
    content-level validator ratified it. The repair must restore the model's
    copy of the CHUNK while leaving the model's verbatim echo of the
    declared new lines alone, so the merge lands byte-exact on the declared
    op spec.
    """
    from types import SimpleNamespace

    from fastedit.inference.chunked_merge import chunked_merge

    # A >100-line function of two IDENTICAL sibling blocks (the B24 tie that
    # declines the deterministic editor) whose first block narrows the chunk.
    lines = ["def big():", '    """docstring"""']
    for _block in range(2):
        lines.append("    for _outer in range(2):")
        lines.append("        if value_a > 1:")
        lines += ["            acc_i1 = 1", "            acc_i2 = 2"]
        lines += [f"        acc_{k:02d} = {k}" for k in range(1, 55)]
    lines.append("    return total")
    source = "\n".join(lines) + "\n"
    block_a_start, block_a_end = 3, 3 + 57  # 58-line first for-block
    chunk = "\n".join(lines[block_a_start - 1: block_a_end]) + "\n"

    # The aligned snippet: the first block's context plus one declared new
    # line (the string-embedded ellipsis drives the narrow).
    declared = '    seam_probe = "... existing code ..."\n'
    snippet = "".join(ln + "\n" for ln in lines[block_a_start - 1: block_a_end]) + declared
    # The measured model failure: the chunk echo one level shallow (first
    # line at column 0), the declared new line verbatim.
    model_output = "\n".join(
        ln.removeprefix("    ") for ln in chunk.splitlines()
    ) + "\n" + declared
    expected = (
        "\n".join(lines[:block_a_end]) + "\n" + declared
        + "\n".join(lines[block_a_end:]) + "\n"
    )

    seen_chunks: list[str] = []

    def merge_fn(code, snippet_text, language):
        seen_chunks.append(code)
        return SimpleNamespace(
            merged_code=model_output, parse_valid=True, tokens_generated=7,
            latency_ms=1.0, truncated=False,
        )

    result = chunked_merge(
        original_code=source,
        snippet=snippet,
        file_path="repair_probe.py",
        merge_fn=merge_fn,
        language="python",
        replace="big",
    )
    assert seen_chunks == [chunk], (
        "the model must see exactly the narrowed first block"
    )
    assert result.chunks_rejected == 0 and result.retries == 0, (
        "the repaired merge must pass the battery"
    )
    assert result.merged_code == expected, (
        f"the repair re-indented the declared new line (or failed to restore "
        f"the chunk body):\n{result.merged_code!r}"
    )


# ===========================================================================
# B28 — _find_insertion_region rarest-context anchoring
# ===========================================================================


def _patch_snippet_defs(monkeypatch, node: ASTNode) -> None:
    monkeypatch.setattr(
        snippet_analysis, "_get_snippet_definitions",
        lambda _snippet, _language=None: [node],
    )


def test_insertion_region_anchors_on_rarest_context_line(monkeypatch):
    """A rare context line outranks a common one for the insertion anchor.

    ``rare_beta_hook()`` occurs once (line 2, the true neighborhood);
    ``setup()`` occurs three times, first at line 6. Pre-fix the anchor was
    max(first occurrences) = 6, steering the region to (5, 11). The rare
    line anchors (1, 6).
    """
    original_lines = [
        "def early():",
        "    rare_beta_hook()",
        "    compute_extras()",
        "",
        "def middle():",
        "    setup()",
        "",
        "def late():",
        "    setup()",
        "    setup()",
        "    return",
    ]
    ast_nodes = [
        ASTNode(name="early", kind="function",
                line_start=1, line_end=3, signature="def early():"),
        ASTNode(name="middle", kind="function",
                line_start=5, line_end=6, signature="def middle():"),
        ASTNode(name="late", kind="function",
                line_start=8, line_end=11, signature="def late():"),
    ]
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    rare_beta_hook()\n    setup()\ndef gamma():\n    pass\n"

    region = snippet_analysis._find_insertion_region(
        snippet, original_lines, ast_nodes, 11,
    )

    assert region is not None
    assert (region.start_line, region.end_line) == (1, 6), (
        f"common first occurrence steered the insertion region to "
        f"{(region.start_line, region.end_line)}; the rare context line "
        f"must anchor (1, 6)"
    )


def test_insertion_region_conflicting_context_returns_no_region(monkeypatch):
    """Context lines from two disjoint regions → no region at all.

    ``hook_one()`` lives at line 2, ``hook_two()`` at line 8; no single
    bracketed region contains both. Pre-fix max(first occurrences) = 8
    returned (7, 11) — a guess. Post-fix the function returns None so the
    caller's fallback runs.
    """
    original_lines = [
        "def top_a():",
        "    hook_one()",
        "",
        "def top_b():",
        "    filler()",
        "",
        "def bottom_a():",
        "    hook_two()",
        "",
        "def bottom_b():",
        "    filler()",
    ]
    ast_nodes = [
        ASTNode(name="top_a", kind="function",
                line_start=1, line_end=2, signature="def top_a():"),
        ASTNode(name="top_b", kind="function",
                line_start=4, line_end=5, signature="def top_b():"),
        ASTNode(name="bottom_a", kind="function",
                line_start=7, line_end=8, signature="def bottom_a():"),
        ASTNode(name="bottom_b", kind="function",
                line_start=10, line_end=11, signature="def bottom_b():"),
    ]
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    hook_one()\n    hook_two()\ndef gamma():\n    pass\n"

    region = snippet_analysis._find_insertion_region(
        snippet, original_lines, ast_nodes, 11,
    )

    assert region is None, (
        f"conflicting context regions produced {region}; expected no "
        f"region so the caller's fallback runs"
    )


def test_locate_chunks_conflicting_insertion_falls_back_to_whole_file(
    tmp_path, monkeypatch,
):
    """The caller's fallback for a declined insertion region is safe.

    locate_chunks must degrade to the conservative whole-file region when
    _find_insertion_region declines — never to a guessed sub-region.
    """
    src = """\
def top_a():
    hook_one()

def top_b():
    filler()

def bottom_a():
    hook_two()

def bottom_b():
    filler()
"""
    target = tmp_path / "svc.py"
    target.write_text(src)
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    hook_one()\n    hook_two()\ndef gamma():\n    pass\n"

    chunks = locate_chunks(snippet, src, str(target), language="python")

    assert len(chunks) == 1, chunks
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 11), (
        f"declined insertion region must fall back to the whole file, "
        f"got {(chunks[0].start_line, chunks[0].end_line)}"
    )
    assert chunks[0].matched_nodes == ["<unmatched>"]


# ===========================================================================
# Keep the Step-6 gate + editor partition identical (B24 critical note):
# the shared classifier means every decline above is observed by both
# consumers; test_gate_fails_closed_on_ambiguous_binding pins the gate side.
# ===========================================================================


_CONTIGUOUS_ORIGINAL = (
    "def alpha():\n"
    "    x = 1\n"
    "    return x\n"
    "\n"
    "def beta():\n"
    "    x = 1\n"
    "    run(x)\n"
    "    return x\n"
)
_CONTIGUOUS_SNIPPET = (
    "    x = 1\n"
    "    telemetry(x)\n"
    "    run(x)\n"
)


@pytest.mark.parametrize(
    "original,snippet",
    # Contiguous unique binding: gate sees the partition the editor used.
    [(_CONTIGUOUS_ORIGINAL, _CONTIGUOUS_SNIPPET)],
)
def test_classifier_is_single_source_for_editor_and_gate(original, snippet):
    """The classifier raises for ambiguity and binds uniquely otherwise.

    Calling it directly must yield exactly the binding the editor applied:
    the contiguous (beta) occurrence, never the first forward one.
    """
    from fastedit.inference.text_match import _classify_edit_lines

    classified = _classify_edit_lines(
        original.splitlines(), snippet.splitlines(),
    )
    contexts = [c for c in classified if c[0] == "context"]
    assert [(c[1], c[2]) for c in contexts] == [(0, 5), (2, 6)], (
        f"classifier bound the repeated anchor at the wrong occurrence: "
        f"{[(c[1], c[2]) for c in contexts]}"
    )
