"""Step D2 — AST-less text anchor chunking (hermetic, default tier).

MISSION under test: a STRUCTURELESS file (``language=None`` — ``.txt``,
``.log``, anything the grammar resolver honestly cannot parse) of ANY size
must be editable. Before D2 such files always took the whole-file merge
branch, whose >150-line gate refused everything bigger; D2 makes the
snippet's UNIQUE context lines act as anchors and extracts a WINDOW chunk
around each anchor, sized to the measured model context budget
(``chunk_locator._MAX_TEXT_CHUNK_LINES``, documented there — no tokenizer
dependency).

The doctrine locked down here (C2/C3 inheritance — the seams are where
corruption lives):

* **Window selection is declarative**: one anchor → one window (anchor ±
  the budget's context, clamped to the file); several anchors → several
  windows merged per the shared ``_merge_overlapping_regions``; a merged
  window is re-fitted around ITS anchors' span so the budget bounds the
  context, never the anchors themselves.
* **Anchor matching is uniqueness-based** (``_text_snippet_anchor_lines``):
  a line occurring more than once in the original cannot locate a window —
  the duplicated-paragraph trap (GIGO, req. 9) is made impossible instead
  of heuristic; the forward-scan ordering mirrors the battery's
  ``_classify_snippet`` cursor so an anchor the validator could never bind
  does not anchor a window either.
* **The fail-loud no-anchor policy is preserved**: with no unique anchor
  nothing declares where the edit goes, so the whole-file chunk (and its
  >150-line ``ValueError``) stays. An anchored window bypasses that gate
  even when it happens to span the whole file.
* **Battery coverage at BOTH levels** (Step D2 integration): each window
  merge runs the span-local battery (content faithfulness + D1 text
  traits, scripted merge_fn patterns from the A2/D1 suites), and the
  final ASSEMBLY runs the whole-file trait check with exact arithmetic
  (``_merge_rejection_reason``'s D1 gate on the full original vs the full
  merge output) — a rejected chunk's missing payload fails the whole
  merge loudly (rejection convention: the original file is kept, never a
  silently partial text edit).
* **Seam byte-identity**: windows are cut on ORIGINAL line indices and
  spliced in reverse — the scripted perfect model proves the untouched
  regions (head, tail, inter-window gaps) stay byte-identical and the
  composed golden is byte-exact; CRLF and no-final-EOL conventions pass
  through the windowed path unchanged.

Everything here is hermetic: scripted merge_fn, no model, no network.
The real-model counterparts live in tests/test_real_llm_text_chunks.py
(``llm`` tier) and tests/test_stress_100mb_txt.py (``llm stress``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastedit.inference.chunk_locator import (
    _MAX_TEXT_CHUNK_LINES,
    _TEXT_ANCHOR_TAG,
    _TEXT_WINDOW_CONTEXT,
    _text_anchor_windows,
    _text_snippet_anchor_lines,
    _trim_window_to_content,
    locate_chunks,
)
from fastedit.inference.chunked_merge import chunked_merge

# ---------------------------------------------------------------------------
# Synthetic text corpora — distinct lines (the C3 doctrine: the window the
# model re-emits must never repeat itself), blank-line paragraph separators
# ---------------------------------------------------------------------------


def _make_source(line_count: int, eol: str = "\n", final_eol: bool = True) -> str:
    """A distinct-line text file: prose-ish lines, blank line every 10th.

    Line layout invariant (used by the fixtures below): index ``i`` emits
    exactly ONE file line — a note when ``i % 10 != 9``, a blank line
    otherwise — so note ``i`` sits at 1-indexed file line ``i + 1`` and
    every emitted line is file-wide unique.
    """
    lines = []
    for i in range(line_count):
        if i % 10 == 9:
            lines.append("")
        else:
            lines.append(_note_line(i))
    text = eol.join(lines)
    if final_eol:
        text += eol
    return text


def _note_line(i: int) -> str:
    return (
        f"Note {i:05d}: the dock manifest copy ran at minute {i} "
        f"without a single warning from the scanner bank."
    )


def _note_line_no(i: int) -> int:
    """The 1-indexed file line of note ``i`` (see ``_make_source``)."""
    assert i % 10 != 9, f"note {i} is a blank-line slot in the fixture layout"
    return i + 1


def _note_exists(i: int) -> bool:
    return i % 10 != 9


def _expected_windows(source: str, snippet: str) -> list[tuple[int, int]]:
    """The documented D2 window formula applied to the snippet's anchors.

    The unit classes below pin the formula number-by-number on
    hand-computed fixtures; the end-to-end tests reuse it here so their
    expected regions stay readable while the fixtures' blank-line layout
    shifts anchor positions. Anchors come from the (separately unit-
    tested) matcher; merging uses the documented gap-20
    ``_merge_overlapping_regions`` behavior; the re-fit bounds the
    context by the budget around each cluster's anchor span.
    """
    lines = source.splitlines()
    anchors = [a for _si, a in _text_snippet_anchor_lines(snippet, lines)]
    assert anchors, "fixture invariant: the snippet must carry anchors"
    total = len(lines)
    windows = [
        (max(1, a - _TEXT_WINDOW_CONTEXT), min(total, a + _TEXT_WINDOW_CONTEXT))
        for a in anchors
    ]
    merged: list[list[int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + 20:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    fitted = []
    for start, end in merged:
        cluster = [a for a in anchors if start <= a <= end]
        lo, hi = min(cluster), max(cluster)
        fit_ctx = max(0, (_MAX_TEXT_CHUNK_LINES - (hi - lo + 1)) // 2)
        fitted.append(_trim_window_to_content(
            max(1, lo - fit_ctx), min(total, hi + fit_ctx), lines,
        ))
    return fitted


def _window_slice(source: str, window: tuple[int, int]) -> str:
    """The window's exact bytes (1-indexed inclusive line range)."""
    return "".join(source.splitlines(keepends=True)[window[0] - 1:window[1]])


def _StubMergeResult(merged_code, truncated=False):
    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=True,
        tokens_generated=7,
        latency_ms=1.0,
        truncated=truncated,
    )


def _run_merge(
    tmp_path, original, snippet, results, name="big.txt", **kwargs,
):
    """Drive chunked_merge with a scripted merge_fn (the A2/D1 pattern).

    Returns ``(result, calls)`` where each call records the EXACT chunk
    text and snippet the model would have seen.
    """
    calls = []

    def merge_fn(code, snip, lang):
        calls.append((code, snip, lang))
        return results[min(len(calls) - 1, len(results) - 1)]

    target = tmp_path / name
    target.write_text(original, encoding="utf-8")
    result = chunked_merge(
        original, snippet, str(target), merge_fn, language=None, **kwargs,
    )
    return result, calls


def _replace_line(source: str, old: str, new: str) -> str:
    assert old in source, f"fixture drift: {old!r} not in source"
    return source.replace(old, new)


# ---------------------------------------------------------------------------
# Anchor matcher — uniqueness, ordering, skips
# ---------------------------------------------------------------------------


class TestTextAnchorMatcher:
    def test_unique_lines_anchor_their_original_position(self):
        source = _make_source(40)
        anchors = _text_snippet_anchor_lines(
            f"{_note_line(3)}\nnew content line.\n{_note_line(30)}\n",
            source.splitlines(),
        )
        # (snippet line index, 1-indexed original line), forward-scan ordered.
        assert anchors == [(0, _note_line_no(3)), (2, _note_line_no(30))]

    def test_duplicated_line_is_never_an_anchor(self):
        # The GIGO trap: a line occurring twice cannot locate a window.
        source = "alpha paragraph line.\nbeta line.\nalpha paragraph line.\n"
        anchors = _text_snippet_anchor_lines(
            "alpha paragraph line.\n", source.splitlines(),
        )
        assert anchors == []

    def test_marker_and_blank_and_short_lines_are_skipped(self):
        source = _make_source(30)
        snippet = (
            "\n"
            "# ... existing code ...\n"
            "ok\n"
            f"{_note_line(5)}\n"
        )
        anchors = _text_snippet_anchor_lines(snippet, source.splitlines())
        assert anchors == [(3, _note_line_no(5))]

    def test_forward_scan_order_mirrors_the_battery_classifier(self):
        # An anchor BEHIND the cursor cannot bind (the validator would
        # classify it as a new line) — it must not anchor a window.
        source = _make_source(30)
        snippet = f"{_note_line(20)}\n{_note_line(5)}\n"
        anchors = _text_snippet_anchor_lines(snippet, source.splitlines())
        assert anchors == [(0, _note_line_no(20))]

    def test_lines_absent_from_the_original_do_not_anchor(self):
        source = _make_source(30)
        anchors = _text_snippet_anchor_lines(
            "This paragraph exists nowhere in the file at all.\n",
            source.splitlines(),
        )
        assert anchors == []


# ---------------------------------------------------------------------------
# Window selection — budget, clamping, merging, re-fitting (hand-computed)
# ---------------------------------------------------------------------------


class TestTextAnchorWindows:
    def test_single_anchor_window_is_budget_sized_and_clamped(self):
        source = _make_source(5000)
        windows = _text_anchor_windows(
            f"{_note_line(2500)}\nnew.\n", source.splitlines(),
        )
        # The raw fit is 2501±19 = 2482..2520; line 2520 is the fixture's
        # blank slot (i % 10 == 9), so the content-edge trim pulls the end
        # in to 2519.
        assert windows == [(2482, 2519)]
        for start, end in windows:
            assert end - start + 1 <= _MAX_TEXT_CHUNK_LINES

    def test_window_clamps_at_the_file_edges(self):
        source = _make_source(300)
        anchor_line = _note_line_no(2)  # 3
        windows = _text_anchor_windows(
            f"{_note_line(2)}\nnew.\n", source.splitlines(),
        )
        assert windows == [(1, anchor_line + _TEXT_WINDOW_CONTEXT)]

    def test_close_anchors_merge_into_one_window(self):
        # Anchors ~4 lines apart (the marker-idiom replacement shape:
        # line-before + marker + new + line-after) produce ONE window
        # covering both, via _merge_overlapping_regions, re-fitted so the
        # budget bounds the context around the anchors' span.
        source = _make_source(5000)
        lo = _note_line_no(1000)  # 1001
        hi = _note_line_no(1004)  # 1005
        snippet = (
            f"{_note_line(1000)}\n"
            "# ... existing code ...\n"
            "The replaced paragraph, declared fresh.\n"
            f"{_note_line(1004)}\n"
        )
        windows = _text_anchor_windows(snippet, source.splitlines())
        assert len(windows) == 1
        fit_ctx = (_MAX_TEXT_CHUNK_LINES - (hi - lo + 1)) // 2
        assert windows == [(lo - fit_ctx, hi + fit_ctx)]
        assert windows[0][1] - windows[0][0] + 1 <= _MAX_TEXT_CHUNK_LINES

    def test_distant_anchors_stay_separate_chunks(self):
        source = _make_source(5000)
        snippet = (
            f"{_note_line(1000)}\nnew A one.\nnew A two.\n"
            f"{_note_line(4000)}\nnew B one.\n"
        )
        windows = _text_anchor_windows(snippet, source.splitlines())
        # Both raw windows end on the fixture's blank slots (1020 / 4020,
        # i % 10 == 9); the content-edge trim pulls the ends in to 1019 /
        # 4019. The starts (982 / 3982) are content lines and stay.
        assert windows == [
            (_note_line_no(1000) - _TEXT_WINDOW_CONTEXT, 1019),
            (_note_line_no(4000) - _TEXT_WINDOW_CONTEXT, 4019),
        ]

    def test_no_anchor_means_no_windows(self):
        source = _make_source(300)
        assert _text_anchor_windows("all new content only.\n",
                                    source.splitlines()) == []

    def test_locate_chunks_tags_text_anchor_regions(self):
        source = _make_source(5000)
        regions = locate_chunks(
            f"{_note_line(2500)}\n# ... existing code ...\nnew.\n",
            source, "big.txt", 30, None,
        )
        # 2501±19 = 2482..2520, trimmed to the content edge 2519 (line
        # 2520 is the fixture's blank slot).
        assert [(r.start_line, r.end_line) for r in regions] == [(2482, 2519)]
        assert all(r.matched_nodes == [_TEXT_ANCHOR_TAG] for r in regions)


# ---------------------------------------------------------------------------
# End-to-end scripted merges — window extraction, battery, splicing
# ---------------------------------------------------------------------------

# The D1-measured converging marker idiom, window-shaped: anchor-before +
# marker + declared replacement + anchor-after (the old line sits directly
# before the second anchor, which is exactly the positional capacity the
# battery's deletion justification grants).


def _replace_snippet(before: int, new_line: str, after: int) -> str:
    return (
        f"{_note_line(before)}\n"
        "# ... existing code ...\n"
        f"{new_line}"
        f"{_note_line(after)}\n"
    )


def _merge_window(chunk: str, old: str, new: str) -> str:
    """The scripted PERFECT model: the window with the one swap applied."""
    merged = chunk.replace(old, new)
    assert merged != chunk, "fixture drift: the swap never engaged"
    return merged


class TestWindowedMergeEndToEnd:
    def test_single_window_replace_is_byte_exact_vs_independent_golden(
        self, tmp_path,
    ):
        source = _make_source(5000)
        old_line = _note_line(2500) + "\n"
        new_line = (
            "Note 02500: the dock manifest copy ran at minute 2500 "
            "with the revised rates.\n"
        )
        snippet = _replace_snippet(2498, new_line, 2501)
        # INDEPENDENT golden: explicit line-splice arithmetic, never fastedit.
        golden = _replace_line(source, old_line, new_line)
        (window,) = _expected_windows(source, snippet)

        result, calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(
                _merge_window(_window_slice(source, window), old_line, new_line),
            )],
        )

        # The window path ran: one chunk, the expected region, the model
        # saw EXACTLY the window bytes.
        assert result.chunks_used == 1
        assert result.chunk_regions == [window]
        assert len(calls) == 1
        assert calls[0][0] == _window_slice(source, window)
        # The window carried the FULL snippet (single chunk).
        assert calls[0][1] == snippet
        # Battery + gates all green, byte-exact vs the independent golden.
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.retries == 0
        assert result.merged_code == golden

    def test_seam_byte_identity_untouched_regions_untouched(self, tmp_path):
        source = _make_source(5000)
        old_line = _note_line(2500) + "\n"
        new_line = "Note 02500: rewritten by the op.\n"
        snippet = _replace_snippet(2498, new_line, 2501)
        (window,) = _expected_windows(source, snippet)
        result, _calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(_merge_window(
                _window_slice(source, window), old_line, new_line,
            ))],
        )
        merged_lines = result.merged_code.splitlines(keepends=True)
        orig_lines = source.splitlines(keepends=True)
        # Head before the window and tail after it: BYTE-IDENTICAL.
        assert merged_lines[: window[0] - 1] == orig_lines[: window[0] - 1]
        assert merged_lines[window[1]:] == orig_lines[window[1]:]

    def test_multi_anchor_multi_chunk_with_scoped_snippet_portions(
        self, tmp_path,
    ):
        source = _make_source(5000)
        payload_a = "".join(
            f"Inserted paragraph A line {j}: fresh manifest notes for the "
            f"afternoon shift.\n"
            for j in range(1, 5)
        )
        payload_b = "".join(
            f"Inserted paragraph B line {j}: the freezer bank audit trail "
            f"continues here.\n"
            for j in range(1, 5)
        )
        snippet = (
            f"{_note_line(1000)}\n"
            "\n"
            f"{payload_a}"
            f"{_note_line(4000)}\n"
            "\n"
            f"{payload_b}"
        )
        window_a, window_b = _expected_windows(source, snippet)

        def scripted(code: str) -> str:
            # The scripted model merges whichever window it was handed.
            if _note_line(1000) in code:
                return code.replace(
                    _note_line(1000) + "\n",
                    _note_line(1000) + "\n\n" + payload_a,
                )
            return code.replace(
                _note_line(4000) + "\n",
                _note_line(4000) + "\n\n" + payload_b,
            )

        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            return _StubMergeResult(scripted(code))

        target = tmp_path / "big.txt"
        target.write_text(source, encoding="utf-8")
        result = chunked_merge(
            source, snippet, str(target), merge_fn, language=None,
        )

        # Two windows, both the expected regions, both received exactly
        # their own slice and their OWN snippet portion (a foreign anchor
        # in the portion would classify as a new line and demand a
        # duplicate insertion — the portion scoping is the guard).
        assert result.chunks_used == 2
        assert result.chunk_regions == [window_a, window_b]
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        for window, anchor, payload, foreign in (
            (window_a, _note_line(1000), payload_a, _note_line(4000)),
            (window_b, _note_line(4000), payload_b, _note_line(1000)),
        ):
            window_text = _window_slice(source, window)
            matching = [c for c in calls if c[0] == window_text]
            assert len(matching) == 1, (
                f"expected exactly one model call on window {window}"
            )
            portion = matching[0][1]
            assert anchor in portion
            assert payload in portion
            assert foreign not in portion, (
                "a foreign anchor leaked into this window's snippet portion "
                "— the battery would demand a duplicate insertion"
            )

        # Composed golden: BOTH payloads inserted, byte-exact.
        golden = source.replace(
            _note_line(1000) + "\n",
            _note_line(1000) + "\n\n" + payload_a,
        ).replace(
            _note_line(4000) + "\n",
            _note_line(4000) + "\n\n" + payload_b,
        )
        assert result.merged_code == golden
        # The inter-window gap is untouched (seam byte-identity), at its
        # post-edit-1 position (window A's insertion shifted it down).
        shift_a = payload_a.count("\n") + 1  # payload lines + the blank
        merged_lines = result.merged_code.splitlines(keepends=True)
        orig_lines = source.splitlines(keepends=True)
        assert (
            merged_lines[window_a[1] + shift_a: window_b[0] - 1 + shift_a]
            == orig_lines[window_a[1]: window_b[0] - 1]
        )

    def test_anchored_300_line_file_bypasses_the_whole_file_gate(
        self, tmp_path,
    ):
        # An anchored snippet on a 300-line file (>150) BYPASSES the
        # whole-file gate: the anchor declared the edit site, so the merge
        # runs through the ~39-line window chunk and succeeds where the
        # pre-D2 pipeline raised.
        source = _make_source(300)
        old_line = _note_line(150) + "\n"
        new_line = "Note 00150: rewritten by the anchored op.\n"
        snippet = _replace_snippet(148, new_line, 151)
        golden = _replace_line(source, old_line, new_line)
        (window,) = _expected_windows(source, snippet)
        assert window[1] - window[0] + 1 <= _MAX_TEXT_CHUNK_LINES, (
            "fixture invariant: the window is budget-sized"
        )
        result, calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(_merge_window(
                _window_slice(source, window), old_line, new_line,
            ))],
        )
        assert result.chunks_used == 1
        assert result.chunks_rejected == 0
        assert len(calls) == 1
        assert result.merged_code == golden


class TestNoAnchorFailLoud:
    def test_no_anchor_over_150_lines_still_refused(self, tmp_path):
        # Nothing declares where the edit goes — the whole-file chunk and
        # its fail-loud gate are PRESERVED (documented D2 policy).
        source = _make_source(200)
        with pytest.raises(ValueError, match="150-line"):
            _run_merge(
                tmp_path, source, "A brand new paragraph, anchored nowhere.\n",
                [_StubMergeResult(source)],
            )

    def test_duplicated_anchor_lines_cannot_window_a_200_line_file(
        self, tmp_path,
    ):
        # Every content line occurs twice → no unique anchor → no window →
        # the gate refuses instead of guessing an occurrence (GIGO).
        source = "alpha paragraph line one.\nbeta line two.\n" * 100
        with pytest.raises(ValueError, match="150-line"):
            _run_merge(
                tmp_path, source,
                "alpha paragraph line one.\n"
                "# ... existing code ...\n"
                "new line.\n",
                [_StubMergeResult(source)],
            )


class TestWindowBattery:
    def test_corrupt_window_output_retries_with_the_reason_then_lands(
        self, tmp_path,
    ):
        # Attempt 1 drops a preserved window line (content-faithfulness
        # violation the span-local battery must catch); attempt 2 is the
        # perfect merge. The corrective note rides the retry.
        source = _make_source(5000)
        old_line = _note_line(2500) + "\n"
        new_line = "Note 02500: rewritten by the op.\n"
        snippet = _replace_snippet(2498, new_line, 2501)
        (window,) = _expected_windows(source, snippet)
        window_lines = _window_slice(source, window).splitlines(keepends=True)
        perfect = "".join(window_lines).replace(old_line, new_line)
        dropped = _note_line(2496) + "\n"
        assert dropped in perfect
        corrupt = "".join(
            ln for ln in window_lines if ln != dropped
        ).replace(old_line, new_line)
        assert corrupt != perfect

        result, calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(corrupt), _StubMergeResult(perfect)],
            max_validation_retries=2,
        )
        assert len(calls) == 2
        assert result.retries == 1
        assert result.chunks_rejected == 0
        assert "NOTE:" in calls[1][1], "the corrective note must ride the retry"
        assert calls[1][1].startswith(snippet)
        golden = _replace_line(source, old_line, new_line)
        assert result.merged_code == golden

    def test_blank_line_destruction_fails_the_span_local_trait_gate(
        self, tmp_path,
    ):
        # The D1 trait battery at WINDOW level: blank lines are the
        # content view's blind spot, the trait gate's sharpest witness.
        source = _make_source(5000)
        old_line = _note_line(2500) + "\n"
        new_line = "Note 02500: rewritten by the op.\n"
        snippet = _replace_snippet(2498, new_line, 2501)
        (window,) = _expected_windows(source, snippet)
        perfect = _merge_window(
            _window_slice(source, window), old_line, new_line,
        )
        blank_stripped = "".join(
            ln for ln in perfect.splitlines(keepends=True) if ln.strip()
        )
        assert blank_stripped != perfect

        result, calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(blank_stripped)],
            max_validation_retries=1,
        )
        assert len(calls) == 2
        assert result.retries == 1
        assert result.chunks_rejected == 1
        assert result.parse_valid is False
        # Rejection convention: the ORIGINAL file is kept.
        assert result.merged_code == source
        assert "text-trait check" in calls[1][1]
        assert "blank_lines" in calls[1][1]

    def test_rejected_chunk_fails_the_assembly_trait_gate_fail_loud(
        self, tmp_path,
    ):
        # Two windows; the second chunk's every attempt is corrupt → the
        # chunk is rejected → the ASSEMBLED file is missing that chunk's
        # declared payload → the whole-file trait check (exact arithmetic)
        # refuses the ENTIRE merge (original kept, parse_valid False,
        # every chunk reported rejected). A text merge never ships a
        # silently partial edit.
        source = _make_source(5000)
        payload_a = "".join(
            f"Inserted paragraph A line {j}: fresh manifest notes.\n"
            for j in range(1, 5)
        )
        payload_b = "".join(
            f"Inserted paragraph B line {j}: the freezer audit trail.\n"
            for j in range(1, 5)
        )
        snippet = (
            f"{_note_line(1000)}\n"
            "\n"
            f"{payload_a}"
            f"{_note_line(4000)}\n"
            "\n"
            f"{payload_b}"
        )

        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            if _note_line(1000) in code:
                merged = code.replace(
                    _note_line(1000) + "\n",
                    _note_line(1000) + "\n\n" + payload_a,
                )
            else:
                # Every attempt for window B drops a preserved line —
                # content-faithfulness rejects it on every retry.
                merged = code.replace(_note_line(4000) + "\n", "")
            return _StubMergeResult(merged)

        target = tmp_path / "big.txt"
        target.write_text(source, encoding="utf-8")
        result = chunked_merge(
            source, snippet, str(target), merge_fn, language=None,
            max_validation_retries=1,
        )

        # Window A succeeded (one attempt), window B exhausted its budget.
        assert len(calls) == 3  # A: 1 attempt; B: initial + 1 retry
        assert result.chunks_used == 2
        assert result.chunks_rejected == 2
        assert result.parse_valid is False
        # THE rejection convention: the original file, never a partial.
        assert result.merged_code == source


class TestEolThroughWindowedPath:
    def test_crlf_window_merge_preserves_the_convention(self, tmp_path):
        source = _make_source(5000, eol="\r\n")
        old_line = _note_line(2500) + "\r\n"
        new_line = "Note 02500: rewritten with crlf endings.\r\n"
        snippet = (
            f"{_note_line(2498)}\r\n"
            "# ... existing code ...\r\n"
            f"{new_line}"
            f"{_note_line(2501)}\r\n"
        )
        (window,) = _expected_windows(source, snippet)
        result, _calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(_merge_window(
                _window_slice(source, window), old_line, new_line,
            ))],
            name="crlf.txt",
        )
        golden = _replace_line(source, old_line, new_line)
        assert result.merged_code == golden
        assert "\r\n" in result.merged_code
        assert result.merged_code.count("\r\n") == golden.count("\r\n")

    def test_no_final_eol_window_merge_keeps_the_trailing_state(self, tmp_path):
        # The file's last line is a NOTE (index 4998) left unterminated,
        # and the anchor sits near EOF so the window includes that line:
        # the assembled file must end without a terminator exactly like
        # the original (B31 through the windowed path).
        source = _make_source(4999, final_eol=False)
        assert not source.endswith("\n"), (
            "fixture invariant: the last line is unterminated"
        )
        old_line = _note_line(4990) + "\n"
        new_line = "Note 04990: rewritten near the unterminated end.\n"
        snippet = _replace_snippet(4988, new_line, 4991)
        (window,) = _expected_windows(source, snippet)
        assert window[1] == len(source.splitlines(keepends=True)), (
            "fixture invariant: the window reaches the unterminated EOF line"
        )
        perfect = _merge_window(
            _window_slice(source, window), old_line, new_line,
        )
        assert not perfect.endswith("\n"), (
            "fixture invariant: the scripted merge keeps the EOF state"
        )
        result, _calls = _run_merge(
            tmp_path, source, snippet,
            [_StubMergeResult(perfect)],
            name="tail.txt",
        )
        golden = _replace_line(source, old_line, new_line)
        assert result.merged_code == golden
        assert not result.merged_code.endswith("\n")


class TestGrammarBackedFilesKeepTheAstPath:
    def test_markdown_never_takes_the_text_anchor_path(self):
        # md is grammar-backed since B2: even a snippet whose lines are
        # unique in the file goes through AST chunking (heading spans),
        # never the D2 text windows — the two strategies stay disjoint.
        # (Step D4 re-measured the alternative — routing AST-eligible
        # markdown comment edits through the D2 windows — against the real
        # model and REJECTED it: the window edges cut documents mid-fence,
        # and the model's fragment merges failed every attempt where the
        # whole-document merge converged. See
        # tests/test_real_llm_mixed_lang.py's docstring.)
        md = (
            "# Title\n\nintro paragraph one.\n\n## Section A\n\n"
            "body paragraph under A.\nsecond line of body.\n\n"
            "## Section B\n\nfinal paragraph.\n"
        )
        regions = locate_chunks(
            "body paragraph under A.\n# ... existing code ...\nnew.\n",
            md, "doc.md", 30, "markdown",
        )
        assert all(
            _TEXT_ANCHOR_TAG not in r.matched_nodes for r in regions
        ), (
            "a grammar-backed file must not be windowed by the D2 text "
            "anchor matcher"
        )
