"""Step D1 — CJK-aware trait heuristics for STRUCTURELESS text (req. 6 + 9).

Three product behaviors are locked down here:

1. ``fastedit.text_heuristics.text_traits`` — the trait definitions
   themselves (pure-zh, pure-en, mixed, emoji, tabs, CRLF, quotes,
   digits), each pinned to an exact number so the definitions cannot
   drift silently.
2. ``expected_traits_after_op`` + ``validate_text_output`` — the trait
   oracle: expected output traits are computed from INPUT + op spec
   (never from a "correct text" ideal), and acceptance is a per-trait
   band comparison under the documented tolerance policy.
3. The battery integration — ``_merge_rejection_reason`` grows a
   text-heuristics branch for STRUCTURELESS files (language is None or
   the grammar resolver honestly has no grammar), derived op spec from
   the snippet, and the GIGO semantics: preserved garbage passes,
   uncommanded changes fail.

GIGO (req. 9) at trait level, pinned by the unit tests:

* a DEFECTIVE input (duplicated paragraph, unbalanced quotes) is a
  trait to preserve — an unrelated edit that keeps it PASSES;
* an uncommanded paragraph drop FAILS with a reason naming the trait;
* a quote-count change FAILS.

The battery tests are hermetic (scripted merge_fn — no model, no
network), same doctrine as tests/test_relative_validation.py.
"""

from __future__ import annotations

import pytest

from fastedit.inference.chunked_merge import (
    _derived_text_op,
    _is_structureless_language,
    _merge_rejection_reason,
)
from fastedit.text_heuristics import (
    TOLERANCE_EXACT,
    TOLERANCE_MODEL_PROSE,
    TRAIT_NAMES,
    TextOp,
    expected_traits_after_op,
    text_traits,
    validate_text_output,
)

# ---------------------------------------------------------------------------
# Trait fixtures — every expected number is hand-derived from the
# documented definitions in fastedit/text_heuristics.py's docstring.
# ---------------------------------------------------------------------------

# pure zh: 3 maximal CJK runs (the fullwidth colon breaks the run),
# 15 CJK chars, 1 CJK punct, 3 words, 50 bytes.
F_ZH = "中文注释：快速编辑语料\n第二行文本\n"

# pure en: 9 whitespace words (each sentence-final "." rides its token),
# 2 ASCII punct, 55 bytes.
F_EN = "The warehouse sync runs at midnight.\nSecond line here.\n"

# mixed en/zh: "张伟" is ONE CJK word inside a latin sentence; the date
# contributes 8 digits. The hyphens are NOT in the ASCII punctuation class
# (the requirement names .,;:!? only) — only the final period counts.
F_MIX = "Created by 张伟 on 2024-01-02.\n"

# emoji: standalone emoji tokens contain no alphanumeric → NOT words;
# an emoji glued between letters does not split a whitespace token.
F_EMOJI = "good 👍 job\n🎉 party\n"

# tabs + CRLF + a whitespace-only line: splitlines sees 3 lines, one of
# them whitespace-only; \t separates words like any whitespace.
F_TABS = "a\tb\r\n\r\n  c\r\n"

# quotes: 3 straight (2 in "hi", 1 in It's), 2 curly (“ ”), the em dash
# is CJK-class punctuation, the periods are ASCII punctuation.
F_QUOTES = "He said \"hi\" — she replied “ok”.\nIt's fine.\n"

# blank/whitespace-only lines: 4 lines, 2 of them whitespace-only.
F_BLANKS = "para one\n\n   \n\tpara two\n"


class TestTextTraits:
    """The trait definitions, pinned number by number."""

    def test_pure_chinese(self):
        assert text_traits(F_ZH) == {
            "lines": 2,
            "blank_lines": 0,
            "words": 3,        # 中文注释 | 快速编辑语料 | 第二行文本
            "cjk_chars": 15,   # 4 + 6 + 5
            "punct_cjk": 1,    # ：
            "punct_ascii": 0,
            "quotes_straight": 0,
            "quotes_curly": 0,
            "digits": 0,
            "bytes": 50,       # (4+6+1)*3 + 1 + 5*3 + 1
        }

    def test_pure_english(self):
        assert text_traits(F_EN) == {
            "lines": 2,
            "blank_lines": 0,
            "words": 9,        # 6 + 3 tokens, each sentence-final "." rides it
            "cjk_chars": 0,
            "punct_cjk": 0,
            "punct_ascii": 2,  # two periods
            "quotes_straight": 0,
            "quotes_curly": 0,
            "digits": 0,
            "bytes": 55,
        }

    def test_mixed_english_chinese(self):
        assert text_traits(F_MIX) == {
            "lines": 1,
            "blank_lines": 0,
            "words": 5,        # 张伟 | Created | by | on | 2024-01-02.
            "cjk_chars": 2,    # 张伟
            "punct_cjk": 0,
            "punct_ascii": 1,  # the final period (hyphens are not .,;:!?)
            "quotes_straight": 0,
            "quotes_curly": 0,
            "digits": 8,       # 20240102
            "bytes": 33,
        }

    def test_emoji_is_not_a_word(self):
        assert text_traits(F_EMOJI) == {
            "lines": 2,
            "blank_lines": 0,
            "words": 3,        # good | job | party (standalone emoji: no alnum)
            "cjk_chars": 0,
            "punct_cjk": 0,
            "punct_ascii": 0,
            "quotes_straight": 0,
            "quotes_curly": 0,
            "digits": 0,
            "bytes": 25,
        }

    def test_tabs_crlf_and_whitespace_only_lines(self):
        assert text_traits(F_TABS) == {
            "lines": 3,        # ["a\tb", "", "  c"]
            "blank_lines": 1,
            "words": 3,        # a | b | c
            "cjk_chars": 0,
            "punct_cjk": 0,
            "punct_ascii": 0,
            "quotes_straight": 0,
            "quotes_curly": 0,
            "digits": 0,
            "bytes": 12,       # 5 + 2 + 5
        }

    def test_quote_classes(self):
        assert text_traits(F_QUOTES) == {
            "lines": 2,
            "blank_lines": 0,
            "words": 8,        # He|said|"hi"|she|replied|“ok”.|It's|fine.
            "cjk_chars": 0,
            "punct_cjk": 1,    # the em dash
            "punct_ascii": 2,  # two periods
            "quotes_straight": 3,  # "hi" (2) + It's (1)
            "quotes_curly": 2,     # “ok”
            "digits": 0,
            "bytes": 50,
        }

    def test_blank_and_whitespace_only_lines(self):
        traits = text_traits(F_BLANKS)
        assert traits["lines"] == 4
        assert traits["blank_lines"] == 2  # "" and "   "
        assert traits["words"] == 4
        assert traits["bytes"] == 24

    def test_empty_text_has_all_zero_traits(self):
        assert text_traits("") == {trait: 0 for trait in TRAIT_NAMES}

    def test_trait_names_are_stable(self):
        # The trait set is the validator's contract with its consumers
        # (the battery derives removable traits over exactly these keys).
        # ORDER is check order: blank_lines first — it is the sharpest
        # diagnostic for the content view's blind spot (layout lines),
        # and its reason pinpoints layout destruction.
        assert TRAIT_NAMES == (
            "blank_lines", "lines", "words", "cjk_chars",
            "punct_cjk", "punct_ascii", "quotes_straight",
            "quotes_curly", "digits", "bytes",
        )


# ---------------------------------------------------------------------------
# expected_traits_after_op — the trait oracle arithmetic (INPUT + op)
# ---------------------------------------------------------------------------

CODA = "Appended coda line.\n"


class TestExpectedTraitsAfterOp:
    def test_none_op_expects_the_original_traits(self):
        op = TextOp(kind="none")
        assert expected_traits_after_op(F_EN, op) == text_traits(F_EN)

    def test_insert_adds_exactly_the_payload_traits(self):
        op = TextOp(kind="insert", payload=CODA)
        expected = expected_traits_after_op(F_EN, op)
        original = text_traits(F_EN)
        payload = text_traits(CODA)
        for trait in TRAIT_NAMES:
            assert expected[trait] == original[trait] + payload[trait], trait
        assert expected["lines"] == 3
        assert expected["words"] == 12
        assert expected["bytes"] == 55 + 20

    def test_append_is_insert_arithmetic(self):
        # "append" is the EOF flavor of insert — the oracle arithmetic is
        # identical (the pipeline's splices are line-aligned either way).
        insert = expected_traits_after_op(F_EN, TextOp(kind="insert", payload=CODA))
        append = expected_traits_after_op(F_EN, TextOp(kind="append", payload=CODA))
        assert insert == append

    def test_replace_subtracts_the_removed_span_and_adds_the_payload(self):
        # Removed spans and payloads are line-aligned (each carries its own
        # terminator), so the byte arithmetic includes exactly one newline
        # per side.
        removed = "The warehouse sync runs at midnight.\n"
        payload = "The warehouse sync runs at dawn.\n"
        op = TextOp(kind="replace", payload=payload, removed=removed)
        expected = expected_traits_after_op(F_EN, op)
        original = text_traits(F_EN)
        assert expected["lines"] == original["lines"]  # 1 line out, 1 in
        assert expected["words"] == original["words"] - 6 + 6
        assert expected["punct_ascii"] == original["punct_ascii"]
        assert expected["bytes"] == original["bytes"] - 37 + 33

    def test_delete_subtracts_the_removed_span(self):
        op = TextOp(kind="delete", removed="Second line here.\n")
        expected = expected_traits_after_op(F_EN, op)
        assert expected["lines"] == 1
        assert expected["words"] == 6
        assert expected["bytes"] == 55 - 18
        assert expected["punct_ascii"] == 1

    def test_unknown_kind_fails_loud(self):
        with pytest.raises(ValueError, match="unknown TextOp kind"):
            TextOp(kind="teleport")


# ---------------------------------------------------------------------------
# validate_text_output — tolerance policy + GIGO semantics
# ---------------------------------------------------------------------------

# A DEFECTIVE input: the "body para alpha" block appears TWICE. The
# duplication is a trait — the tests below prove it is preserved through
# an unrelated append (PASS) and that dropping one copy is a FAIL.
GIGO_ORIGINAL = (
    "Intro line one.\n"
    "\n"
    "body para alpha one.\n"
    "body para alpha two.\n"
    "\n"
    "body para alpha one.\n"
    "body para alpha two.\n"
    "\n"
    "Closing line.\n"
)

GIGO_APPEND_OP = TextOp(kind="append", payload=CODA)


class TestValidateTextOutput:
    def test_faithful_output_passes_with_exact_tolerance(self):
        assert validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), F_EN + CODA,
            TOLERANCE_EXACT,
        ) == (True, "")

    def test_identity_output_passes_for_a_none_op(self):
        assert validate_text_output(
            F_EN, TextOp(kind="none"), F_EN, TOLERANCE_EXACT,
        ) == (True, "")

    # ── GIGO (req. 9): preserved garbage passes, uncommanded changes fail ──

    def test_duplicated_paragraph_preserved_through_unrelated_edit_passes(self):
        faithful = GIGO_ORIGINAL + CODA
        ok, reason = validate_text_output(
            GIGO_ORIGINAL, GIGO_APPEND_OP, faithful, TOLERANCE_EXACT,
        )
        assert ok is True, reason

    def test_uncommanded_paragraph_drop_fails_naming_the_trait(self):
        faithful = GIGO_ORIGINAL + CODA
        # The model "helpfully" deduplicates the repeated block.
        dropped = faithful.replace(
            "body para alpha one.\nbody para alpha two.\n\n", "", 1,
        )
        assert dropped != faithful
        ok, reason = validate_text_output(
            GIGO_ORIGINAL, GIGO_APPEND_OP, dropped, TOLERANCE_EXACT,
        )
        assert ok is False
        assert "lines" in reason
        assert "expected" in reason

    def test_quote_count_change_fails(self):
        original = "He said \"ready\" when asked.\n\nThen nothing.\n"
        output = "He said ready when asked.\n\nThen nothing.\n" + CODA
        ok, reason = validate_text_output(
            original, GIGO_APPEND_OP, output, TOLERANCE_EXACT,
        )
        assert ok is False
        assert "quotes_straight" in reason

    def test_cjk_paragraph_drop_fails_on_cjk_aware_counts(self):
        original = "中文笔记第一段。\n\n中文笔记第二段。\n"
        output = "中文笔记第一段。\n" + "中文补充段落。\n"
        ok, reason = validate_text_output(
            original, TextOp(kind="append", payload="中文补充段落。\n"),
            output, TOLERANCE_EXACT,
        )
        assert ok is False
        assert reason  # a trait name rides in the reason for the retry note

    # ── tolerance policy ──────────────────────────────────────────────────

    def test_exact_tolerance_rejects_off_by_one_word(self):
        drifted = F_EN + "Appended coda line two.\n"  # one extra word
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), drifted, TOLERANCE_EXACT,
        )
        assert ok is False
        assert "words" in reason

    def test_model_prose_tolerance_absorbs_a_small_count_drift(self):
        # ±2% with an absolute floor of 2, measured on a realistic-size
        # text so the relative band is meaningfully wider than the floor:
        # a one-word drift (and its few bytes) is inside the band, a
        # wholesale extra sentence is not.
        big = "The warehouse sync runs at midnight.\n" * 40
        op = TextOp(kind="insert", payload=CODA)
        within = big + "Appended coda line ok.\n"  # one extra word
        ok, reason = validate_text_output(
            big, op, within, TOLERANCE_MODEL_PROSE,
        )
        assert ok is True, reason
        beyond = big + CODA + "extra extra extra extra extra extra extra extra.\n"
        ok, reason = validate_text_output(
            big, op, beyond, TOLERANCE_MODEL_PROSE,
        )
        assert ok is False
        assert "words" in reason

    def test_line_count_is_exact_even_under_model_prose_tolerance(self):
        # The ±2% band never applies to lines: line counts are exact for
        # line-anchored ops. The model-prose preset carries ONE documented
        # seam-blank floor (a model may add or drop a single layout blank
        # at a merge seam — measured), so one undeclared blank passes and
        # two do not.
        one_seam_blank = F_EN + CODA + "\n"
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), one_seam_blank,
            TOLERANCE_MODEL_PROSE,
        )
        assert ok is True, reason
        two_seam_blanks = F_EN + CODA + "\n\n"
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), two_seam_blanks,
            TOLERANCE_MODEL_PROSE,
        )
        assert ok is False
        assert "lines" in reason

    def test_line_count_is_byte_exact_under_exact_tolerance(self):
        # Under TOLERANCE_EXACT even one undeclared blank is a regression:
        # the validator-level line rule is exact (the seam floor is a
        # battery-policy choice, not a trait definition).
        one_seam_blank = F_EN + CODA + "\n"
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), one_seam_blank,
            TOLERANCE_EXACT,
        )
        assert ok is False
        assert "lines" in reason

    def test_layout_slack_widens_lines_and_blank_lines_only(self):
        # The battery passes the snippet's own blank-line count: a model may
        # echo or drop declared layout blanks without failing the gate.
        # A CONTENT drift beyond the layout band still fails.
        with_blank = F_EN + "\n" + CODA
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), with_blank,
            TOLERANCE_MODEL_PROSE, layout_slack=1,
        )
        assert ok is True, reason
        # A dropped paragraph is NOT layout: words fall below the band.
        dropped = F_EN.replace("Second line here.\n", "") + CODA
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="insert", payload=CODA), dropped,
            TOLERANCE_MODEL_PROSE, layout_slack=1,
        )
        assert ok is False

    def test_prose_rewrap_relaxes_lines_only(self):
        # A declared rewrap op keeps its content traits while the line
        # breaks move: the op's payload declares the SAME content (the
        # span being re-wrapped) and the output carries the re-wrapped
        # breaks — lines get the documented rewrap band (here a 2-line
        # span re-wrapped into 4, byte-identical: spaces became newlines),
        # everything else stays tight.
        rewrapped = (
            "The\nwarehouse sync runs at\nmidnight.\nSecond line here.\n"
        )
        assert text_traits(rewrapped)["bytes"] == text_traits(F_EN)["bytes"]
        op = TextOp(
            kind="replace",
            payload=F_EN,
            removed=F_EN,
            prose_rewrap=True,
        )
        ok, reason = validate_text_output(
            F_EN, op, rewrapped, TOLERANCE_MODEL_PROSE,
        )
        assert ok is True, reason
        # Without the declared rewrap policy the same output is a line
        # regression: the op was line-anchored, so ±2 lines is outside the
        # model-prose seam floor.
        strict_op = TextOp(kind="replace", payload=F_EN, removed=F_EN)
        ok, reason = validate_text_output(
            F_EN, strict_op, rewrapped, TOLERANCE_MODEL_PROSE,
        )
        assert ok is False
        assert "lines" in reason

    def test_removable_traits_widen_the_floor_downward_only(self):
        # A snippet whose shape could justify removing a sentence lowers the
        # expected floor by that sentence's traits — never the ceiling.
        removed = "Second line here.\n"
        replaced = "Second revised line here.\n"
        op = TextOp(kind="insert", payload=replaced)  # insertion-shaped op
        removable = text_traits(removed)
        ok, reason = validate_text_output(
            F_EN, op, F_EN.replace(removed, replaced), TOLERANCE_EXACT,
            removable_traits=removable,
        )
        assert ok is True, reason
        # Without the removable allowance the same merge fails (the removed
        # sentence's traits are missing from the arithmetic).
        ok, reason = validate_text_output(
            F_EN, op, F_EN.replace(removed, replaced), TOLERANCE_EXACT,
        )
        assert ok is False

    def test_reason_is_human_readable_for_the_retry_note(self):
        ok, reason = validate_text_output(
            F_EN, TextOp(kind="none"), "changed\n", TOLERANCE_EXACT,
        )
        assert ok is False
        assert "lines" in reason
        assert "expected" in reason


# ---------------------------------------------------------------------------
# The structureless predicate — the resolver's honest answer
# ---------------------------------------------------------------------------


class TestStructurelessPredicate:
    def test_none_is_structureless(self):
        assert _is_structureless_language(None) is True

    def test_resolvable_language_is_not_structureless(self):
        assert _is_structureless_language("python") is False

    def test_unresolvable_language_is_structureless(self):
        # The resolver's honest answer: no grammar → the parse gate cannot
        # run, so the file is validated by the text-trait branch instead.
        assert _is_structureless_language("definitely-not-a-grammar") is True


# ---------------------------------------------------------------------------
# The battery's derived op spec (chunk-local, from the snippet)
# ---------------------------------------------------------------------------

NOTES_ORIGINAL = (
    "Project notes — warehouse sync.\n"
    "\n"
    "The nightly job copies stock levels at midnight.\n"
    "It skips items marked discontinued.\n"
    "\n"
    "Contact: ops@example.com for schedule changes.\n"
    "The on-call engineer reviews failures each morning.\n"
    "\n"
    "This file intentionally contains a duplicated sentence.\n"
    "This file intentionally contains a duplicated sentence.\n"
    "Keep the duplication: it is a trait to preserve.\n"
)

NOTES_APPEND_SNIPPET = (
    "The on-call engineer reviews failures each morning.\n"
    "# ... existing code ...\n"
    "\n"
    "Appended paragraph one.\n"
    "Appended paragraph two.\n"
)

NOTES_REPLACE_SNIPPET = (
    "Project notes — warehouse sync.\n"
    "# ... existing code ...\n"
    "The nightly job copies stock levels at dawn.\n"
    "It skips items marked discontinued.\n"
)

MIDNIGHT_LINE = "The nightly job copies stock levels at midnight.\n"
DAWN_LINE = "The nightly job copies stock levels at dawn.\n"


class TestDerivedTextOp:
    def test_append_snippet_declares_its_payload_and_layout(self):
        op, layout_slack, _removable = _derived_text_op(
            NOTES_ORIGINAL, NOTES_APPEND_SNIPPET,
        )
        assert op.kind == "insert"
        assert op.payload == "Appended paragraph one.\nAppended paragraph two.\n"
        assert op.removed == ""
        # The snippet's own blank line is the layout uncertainty.
        assert layout_slack == 1

    def test_marker_free_snippet_declares_no_removal_capacity(self):
        # Preserve-by-default: without a marker, prose lines are
        # identity-free and no deletion is ever justified.
        snippet = "The nightly job copies stock levels at dawn.\n"
        op, layout_slack, removable = _derived_text_op(NOTES_ORIGINAL, snippet)
        assert op.payload == snippet
        assert layout_slack == 0
        assert removable == {trait: 0 for trait in TRAIT_NAMES}

    def test_marker_segment_capacity_covers_the_adjacent_line(self):
        # The replace snippet's marker-bearing segment holds exactly the
        # midnight line; the declared new line after the marker gives the
        # segment one identity-free back-deletion of positional capacity.
        op, _layout_slack, removable = _derived_text_op(
            NOTES_ORIGINAL, NOTES_REPLACE_SNIPPET,
        )
        assert op.payload == DAWN_LINE
        assert removable["lines"] == 1
        assert removable["words"] == text_traits(MIDNIGHT_LINE)["words"]
        assert removable["bytes"] >= len(MIDNIGHT_LINE.encode("utf-8"))

    def test_no_new_lines_means_a_none_op(self):
        snippet = (
            "Project notes — warehouse sync.\n"
            "# ... existing code ...\n"
        )
        op, _slack, removable = _derived_text_op(NOTES_ORIGINAL, snippet)
        assert op.kind == "none"
        assert op.payload == ""
        assert removable == {trait: 0 for trait in TRAIT_NAMES}


# ---------------------------------------------------------------------------
# Battery integration — scripted merge_fn, hermetic (whole-file path)
# ---------------------------------------------------------------------------


def _StubMergeResult(merged_code, truncated=False):
    from types import SimpleNamespace

    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=True,
        tokens_generated=7,
        latency_ms=1.0,
        truncated=truncated,
    )


def _run_whole_file_merge(tmp_path, original, snippet, results, **kwargs):
    """Drive chunked_merge's whole-file branch with a scripted merge_fn."""
    from fastedit.inference.chunked_merge import chunked_merge

    calls = []

    def merge_fn(code, snip, lang):
        calls.append((code, snip, lang))
        return results[min(len(calls) - 1, len(results) - 1)]

    target = tmp_path / "notes.txt"
    target.write_text(original, encoding="utf-8")
    result = chunked_merge(
        original, snippet, str(target), merge_fn, language=None, **kwargs,
    )
    return result, calls


class TestBatteryTextTraitBranch:
    def test_battery_returns_the_trait_reason_directly(self):
        # The battery's single-reason-string contract (Step A2): the D1
        # branch is one more gate returning a reason string — asserted
        # here directly, without a loop around it.
        faithful = NOTES_ORIGINAL + "\n" + (
            "Appended paragraph one.\nAppended paragraph two.\n"
        )
        assert _merge_rejection_reason(
            NOTES_ORIGINAL, faithful, NOTES_APPEND_SNIPPET, None, False,
        ) is None
        blank_stripped = "".join(
            line for line in faithful.splitlines(keepends=True)
            if line.strip()
        )
        reason = _merge_rejection_reason(
            NOTES_ORIGINAL, blank_stripped, NOTES_APPEND_SNIPPET, None, False,
        )
        assert reason is not None
        assert "text-trait check" in reason
        assert "blank_lines" in reason
        # A truncated result still short-circuits before every gate.
        assert _merge_rejection_reason(
            NOTES_ORIGINAL, faithful, NOTES_APPEND_SNIPPET, None, True,
        ) == "truncated (model hit the token cap)"

    def test_faithful_append_lands_without_retries(self, tmp_path):
        merged = NOTES_ORIGINAL + "\n" + (
            "Appended paragraph one.\nAppended paragraph two.\n"
        )
        result, calls = _run_whole_file_merge(
            tmp_path, NOTES_ORIGINAL, NOTES_APPEND_SNIPPET,
            [_StubMergeResult(merged)],
        )
        assert len(calls) == 1  # clean on the first attempt — no retry
        assert result.retries == 0
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == merged

    def test_justified_sentence_replacement_passes_the_trait_gate(self, tmp_path):
        # The snippet's marker-bearing segment covers the midnight line and
        # declares one identity-free new line after the marker — the same
        # positional capacity the content validator grants. The trait gate
        # must honor it via the derived removable traits (without that
        # allowance the word/line arithmetic would reject this merge).
        merged = NOTES_ORIGINAL.replace(MIDNIGHT_LINE, DAWN_LINE)
        assert merged != NOTES_ORIGINAL
        result, calls = _run_whole_file_merge(
            tmp_path, NOTES_ORIGINAL, NOTES_REPLACE_SNIPPET,
            [_StubMergeResult(merged)],
        )
        assert len(calls) == 1
        assert result.chunks_rejected == 0
        assert result.merged_code == merged

    def test_blank_line_destruction_passes_content_but_fails_traits(
        self, tmp_path,
    ):
        # THE gap the trait gate exists for: the battery's content view
        # skips blank/whitespace-only lines entirely, so a merge that
        # deletes every blank line of a structureless file is invisible to
        # the content validator — and is caught by the blank_lines trait.
        faithful = NOTES_ORIGINAL + "\n" + (
            "Appended paragraph one.\nAppended paragraph two.\n"
        )
        blank_stripped = "".join(
            line for line in faithful.splitlines(keepends=True) if line.strip()
        )
        assert blank_stripped != faithful
        result, calls = _run_whole_file_merge(
            tmp_path, NOTES_ORIGINAL, NOTES_APPEND_SNIPPET,
            [_StubMergeResult(blank_stripped)],
            max_validation_retries=1,
        )
        assert len(calls) == 2  # initial + the pinned single retry
        assert result.retries == 1
        assert result.chunks_rejected == 1
        assert result.parse_valid is False
        # Rejection convention: the original file is kept, never the
        # blank-stripped payload.
        assert result.merged_code == NOTES_ORIGINAL
        # The corrective note carries the trait reason.
        assert "text-trait check" in calls[1][1]
        assert "blank_lines" in calls[1][1]

    def test_trailing_whitespace_loss_is_caught_by_the_bytes_trait(
        self, tmp_path,
    ):
        # Survivors compare STRIPPED in the content view, so a model that
        # trims trailing whitespace from an untouched line passes the
        # content validator — the byte trait is the second witness. The
        # trim (150 bytes) must exceed the snippet's removal capacity
        # (105 bytes: the two trailing content lines the snippet's
        # marker-declared new lines could justify removing) plus the
        # tolerance band — a trim INSIDE that allowance is the trait
        # oracle's documented blind spot (the op-side view cannot tell it
        # from a justified removal either; the content view governs line
        # survival, the trait view governs volume).
        original = NOTES_ORIGINAL.replace(
            "Contact: ops@example.com for schedule changes.\n",
            "Contact: ops@example.com for schedule changes."
            + " " * 150 + "\n",
        )
        trimmed = original.replace(
            "Contact: ops@example.com for schedule changes."
            + " " * 150 + "\n",
            "Contact: ops@example.com for schedule changes.\n",
        )
        trimmed += "\nAppended paragraph one.\nAppended paragraph two.\n"
        result, calls = _run_whole_file_merge(
            tmp_path, original, NOTES_APPEND_SNIPPET,
            [_StubMergeResult(trimmed)],
            max_validation_retries=1,
        )
        assert len(calls) == 2
        assert result.chunks_rejected == 1
        assert result.merged_code == original
        assert "text-trait check" in calls[1][1]
        assert "bytes" in calls[1][1]

    def test_language_known_files_skip_the_trait_branch(self, tmp_path):
        # The trait branch is STRUCTURELESS-only: a python file's blank
        # lines are not a text trait the battery judges (the parse gate
        # plus byte-exact splicing govern there).
        from fastedit.inference.chunked_merge import chunked_merge

        original = "def foo():\n\n    total = 1\n\n    return total\n"
        blank_stripped = "def foo():\n    total = 1\n    return total\n"
        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            return _StubMergeResult(blank_stripped)

        target = tmp_path / "mod.py"
        target.write_text(original, encoding="utf-8")
        snippet = (
            "def foo():\n"
            "# ... existing code ...\n"
            "    return total\n"
        )
        result = chunked_merge(
            original, snippet, str(target), merge_fn, language="python",
        )
        assert len(calls) == 1  # accepted on the first attempt
        assert result.chunks_rejected == 0
        assert result.merged_code == blank_stripped

    def test_trait_rejection_retries_then_rejects_fail_loud(self, tmp_path):
        # Every scripted attempt destroys the blank lines (the appended
        # paragraph still lands, so the CONTENT gate passes and the TRAIT
        # gate is what rejects): the loop retries with the corrective note
        # and finally rejects the site (original kept).
        blank_stripped = "".join(
            line
            for line in (
                NOTES_ORIGINAL + "\n"
                + "Appended paragraph one.\nAppended paragraph two.\n"
            ).splitlines(keepends=True)
            if line.strip()
        )
        result, calls = _run_whole_file_merge(
            tmp_path, NOTES_ORIGINAL, NOTES_APPEND_SNIPPET,
            [_StubMergeResult(blank_stripped)],
            max_validation_retries=2,
        )
        assert len(calls) == 3  # initial + 2 retries
        assert result.retries == 2
        assert result.chunks_rejected == 1
        assert result.merged_code == NOTES_ORIGINAL
        assert all("NOTE:" in c[1] for c in calls[1:])
        assert "text-trait check" in calls[1][1]


class TestStructurelessWholeFileGateBoundary:
    def test_anchored_200_line_text_file_now_windows_and_succeeds(
        self, tmp_path,
    ):
        # Step D2 lifted the D1 boundary: the snippet's unique anchor
        # ("note line 199") declares where the edit goes, so the file is
        # edited through a WINDOW chunk — the >150-line whole-file gate
        # no longer fires for anchored snippets. Full D2 window coverage
        # lives in tests/test_text_anchor_chunking.py; this pins the
        # exact snippet the D1 note used to refuse, with the scripted
        # model merging the WINDOW the locator extracts.
        from fastedit.inference.chunk_locator import (
            _TEXT_WINDOW_CONTEXT,
            locate_chunks,
        )
        from fastedit.inference.chunked_merge import chunked_merge

        original = "".join(f"note line {i}\n" for i in range(200))
        snippet = "note line 199\n# ... existing code ...\nAppended.\n"
        (window,) = [
            (r.start_line, r.end_line)
            for r in locate_chunks(snippet, original, "big.txt", 30, None)
        ]
        assert window == (200 - _TEXT_WINDOW_CONTEXT, 200), (
            "the anchor is the file's last line: the window clamps to it"
        )
        window_lines = original.splitlines(keepends=True)[window[0] - 1:]
        merged_window = "".join(window_lines) + "Appended.\n"
        merged = original + "Appended.\n"
        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            assert code == "".join(window_lines), (
                "the model must see exactly the window bytes"
            )
            return _StubMergeResult(merged_window)

        target = tmp_path / "big.txt"
        target.write_text(original, encoding="utf-8")
        result = chunked_merge(
            original, snippet, str(target), merge_fn, language=None,
        )
        assert len(calls) == 1
        assert result.chunks_used == 1
        assert result.chunks_rejected == 0
        assert result.merged_code == merged

    def test_anchorless_200_line_text_file_is_still_refused(self, tmp_path):
        # The D2 no-anchor policy: nothing in the snippet declares where
        # the edit goes (no unique context line), so only the whole-file
        # chunk remains and its fail-loud >150-line gate stands.
        from fastedit.inference.chunked_merge import chunked_merge

        original = "".join(f"note line {i}\n" for i in range(200))
        target = tmp_path / "big.txt"
        target.write_text(original, encoding="utf-8")
        with pytest.raises(ValueError, match="150-line"):
            chunked_merge(
                original,
                "A brand new paragraph, anchored nowhere in the file.\n",
                str(target),
                lambda *a: _StubMergeResult(original),
                language=None,
            )

    def test_anchored_150_line_text_file_merges_through_a_window(
        self, tmp_path,
    ):
        # At the limit an anchored snippet windows fine (Step D2): the
        # anchor "note line 149" is the file's last line, the window
        # clamps to it, and the battery (content + text traits) governs
        # the window-local merge exactly as the whole-file path used to.
        from fastedit.inference.chunk_locator import locate_chunks
        from fastedit.inference.chunked_merge import chunked_merge

        original = "".join(f"note line {i}\n" for i in range(150))
        snippet = (
            "note line 149\n"
            "# ... existing code ...\n"
            "\n"
            "Appended coda line.\n"
        )
        (window,) = [
            (r.start_line, r.end_line)
            for r in locate_chunks(snippet, original, "solo.txt", 30, None)
        ]
        window_lines = original.splitlines(keepends=True)[window[0] - 1:]
        merged_window = "".join(window_lines) + "\nAppended coda line.\n"
        merged = original + "\nAppended coda line.\n"
        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            return _StubMergeResult(merged_window)

        target = tmp_path / "solo.txt"
        target.write_text(original, encoding="utf-8")
        result = chunked_merge(
            original, snippet, str(target), merge_fn, language=None,
        )
        assert len(calls) == 1
        assert result.chunks_used == 1
        assert result.chunks_rejected == 0
        assert result.merged_code == merged
