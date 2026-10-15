"""Issue #6 — a structureless .gitignore one-line exact replacement is refused.

The reported failure: replacing ONE comment line in a .gitignore
(``# users machines`` → ``# users' machines``) was rejected after 8
validation retries — every attempt of the single text-anchor window was
refused, so the final assembly printed "1/1 window(s) rejected — the
structureless assembly gate refuses a partial text edit" and the file was
kept. A whole-file ``fastedit create --force`` was the only way through.

ROOT CAUSE (diagnosed with a faithful one-line replacement driven through
the real battery): for a structureless file the snippet is the op spec, and
the one-line replacement's snippet restates the whole span with the single
line changed. Both gates that judge the window refused the FAITHFUL merge:

  * the content-faithfulness validator
    (:func:`fastedit.inference.chunked_merge._check_hallucinations`) failed
    closed on the dropped original line — an identity-free line (no
    assignment LHS) with no preservation marker in the segment has no
    sanctioned deletion path, so the declared single-line replacement read
    as "unmentioned original line dropped";
  * the D1 text-trait gate
    (:func:`fastedit.inference.chunked_merge._derived_text_op`) derived an
    INSERT op from the snippet (payload = the new line, nothing removed),
    so the ``bytes``/``lines`` arithmetic expected original + new line and
    the replacement's byte drift (the old line's volume) exceeded the
    tolerance with no ``removable_traits`` credit.

THE FIX (both sides of the same declarative rule): when the snippet is a
COMPLETE RESTATEMENT of the span — every original content line except
exactly one is bound as a context anchor — and it declares exactly one new
line, the one uncovered line is the snippet's declared single-line
replacement:

  * the content validator justifies that one deletion (the merge is then
    byte-for-byte the snippet's declared content; nothing else could have
    survived anyway);
  * the op derivation credits the old line to ``removable_traits``, so the
    trait arithmetic is exact replace-one-line-with-one-line arithmetic.

Everything that made the old gate fail closed KEEPS failing closed, pinned
by the safety tests below: an incomplete restatement that drops a line
without a declared replacement (the dropped-paragraph corruption class),
a restatement minus a line with no declared new line at all, and the
ambiguous anchor-bracketed shape where the old line is simply missing from
the snippet (insert-vs-replace cannot be told apart — the gates decide,
never guess).

Hermetic: no model, no backend — the merge function is a stub that returns
the correctly-replaced window, i.e. what the real 1.7B model produces for
this shape once it converges.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastedit.inference.chunked_merge import (
    _check_hallucinations,
    _derived_text_op,
    _merge_rejection_reason,
    chunked_merge,
)
from fastedit.text_heuristics import TOLERANCE_MODEL_PROSE, validate_text_output

# The reported shape: a .gitignore with a comment line holding a typo.
GITIGNORE = (
    "# users machines\n"
    "*.pyc\n"
    "__pycache__/\n"
    "node_modules/\n"
    "dist/\n"
    ".env\n"
)
OLD_LINE = "# users machines"
NEW_LINE = "# users' machines"


def _replace_model(chunk: str, _snippet: str) -> str:
    """The faithful merge: swap the old line for the new one, keep the rest."""
    return "".join(
        NEW_LINE + "\n" if line.strip() == OLD_LINE else line
        for line in chunk.splitlines(keepends=True)
    )


def _run_window_merge(original_code: str, snippet: str):
    """Drive chunked_merge for a structureless file with a converging model."""
    return chunked_merge(
        original_code=original_code,
        snippet=snippet,
        file_path="/repo/.gitignore",
        merge_fn=lambda chunk, snip, _lang, **_kw: SimpleNamespace(
            merged_code=_replace_model(chunk, snip),
            parse_valid=True,
            tokens_generated=9,
            latency_ms=12.0,
            truncated=False,
        ),
        language=None,  # structureless: the D1/D2 path
    )


# ---------------------------------------------------------------------------
# The reported repro: the whole-file restatement with one line changed
# ---------------------------------------------------------------------------


class TestReportedRepro:
    def test_one_line_replacement_is_accepted(self):
        """The exact reported shape now lands: the comment line is replaced."""
        result = _run_window_merge(GITIGNORE, GITIGNORE.replace(OLD_LINE, NEW_LINE))

        assert result.chunks_rejected == 0
        assert result.parse_valid
        assert result.merged_code != GITIGNORE  # the edit actually landed
        assert NEW_LINE in result.merged_code
        assert OLD_LINE not in result.merged_code
        # Every untouched line survives byte-exact.
        for line in GITIGNORE.splitlines():
            if line.strip() != OLD_LINE:
                assert line in result.merged_code.splitlines()

    def test_mid_span_replacement_is_accepted(self):
        """Same rule with the replaced line between anchors, not at the edge."""
        snippet = GITIGNORE.replace("dist/", "build/")

        def mid_replace_model(chunk: str, _snippet: str) -> str:
            return "".join(
                "build/\n" if line.strip() == "dist/" else line
                for line in chunk.splitlines(keepends=True)
            )

        result = chunked_merge(
            original_code=GITIGNORE,
            snippet=snippet,
            file_path="/repo/.gitignore",
            merge_fn=lambda chunk, snip, _lang, **_kw: SimpleNamespace(
                merged_code=mid_replace_model(chunk, snip),
                parse_valid=True,
                tokens_generated=9,
                latency_ms=12.0,
                truncated=False,
            ),
            language=None,
        )

        assert result.chunks_rejected == 0
        assert result.parse_valid
        assert "build/" in result.merged_code.splitlines()
        assert "dist/" not in result.merged_code.splitlines()

    def test_battery_reason_is_none_for_the_faithful_merge(self):
        """The span-local battery accepts the faithful replacement outright."""
        snippet = GITIGNORE.replace(OLD_LINE, NEW_LINE)
        chunk = GITIGNORE
        merged = _replace_model(chunk, snippet)
        assert _merge_rejection_reason(chunk, merged, snippet, None, False) is None


# ---------------------------------------------------------------------------
# Gate-level units: both halves of the declarative rule
# ---------------------------------------------------------------------------


class TestContentGate:
    def test_complete_restatement_replacement_scores_clean(self):
        """The sanctioned single-line replacement, granted as the battery
        grants it — for a STRUCTURELESS file only (issue #6 scoping).

        The allowance is parameter-gated (:func:`_check_hallucinations`
        defaults to ``allow_complete_replacement=False``) so a code span
        keeps the strict B4 preserve-by-default contract; the battery
        passes ``allow_complete_replacement=structureless``. This test
        exercises the gate the same way the battery does."""
        snippet = GITIGNORE.replace(OLD_LINE, NEW_LINE)
        merged = _replace_model(GITIGNORE, snippet)
        assert _check_hallucinations(
            GITIGNORE, merged, snippet, allow_complete_replacement=True,
        ) == 1.0

    def test_code_span_keeps_the_strict_b4_contract(self):
        """The scoping direction that keeps B4 honest: the SAME
        complete-restatement shape on a code span (no grammar-free flow,
        marker-free) is still a preserve-by-default refusal — the flag
        defaults False for code, and the battery never grants it there."""
        code = "def foo():\n    x = 1\n    return x\n"
        snippet = "def foo():\n    x = 1\n    return x + 1\n"
        assert _check_hallucinations(code, snippet, snippet) == 0.0

    def test_incomplete_drop_without_replacement_still_rejected(self):
        """A restatement that just OMITS a line is not a replacement.

        The snippet restates every line except "dist/" and declares no new
        line at all; the model echoes it. The dropped line has no declared
        replacement, so the deletion stays unjustified.
        """
        dropped = GITIGNORE.replace("dist/\n", "")
        assert _check_hallucinations(GITIGNORE, dropped, dropped) == 0.0

    def test_insertion_idiom_with_dropped_gap_line_still_rejected(self):
        """The D2 insertion shape must keep its preserve-by-default guarantee.

        A marker-free bracketed snippet (anchor + declared payload + anchor)
        is an INSERTION; a model that additionally drops one gap line and
        emits the payload must stay rejected — the gap lines are not
        restated, so this is not a complete restatement.
        """
        big = (
            "header line\n"
            + "".join(f"gap line {i}\n" for i in range(30))
            + "tail line\n"
        )
        snippet = "header line\ninserted payload line\n"
        # Model output: dropped gap line 7 AND inserted the payload.
        merged = big.replace("gap line 7\n", "inserted payload line\n")
        assert _check_hallucinations(big, merged, snippet) == 0.0

    def test_insertion_idiom_without_drop_is_accepted(self):
        """The plain bracketed insertion keeps working (no regression)."""
        big = (
            "header line\n"
            + "".join(f"gap line {i}\n" for i in range(30))
            + "tail line\n"
        )
        snippet = "header line\ninserted payload line\n"
        merged = big.replace("header line\n", "header line\ninserted payload line\n")
        assert _check_hallucinations(big, merged, snippet) == 1.0

    def test_two_dropped_one_declared_still_rejected(self):
        """Complete restatement merging two lines into one: not one-for-one."""
        original = "alpha\nbeta\ngamma\ndelta\n"
        snippet = "alpha\nbeta-and-gamma\ndelta\n"  # two lines became one
        merged = snippet
        assert _check_hallucinations(original, merged, snippet) == 0.0


class TestTraitGate:
    def test_op_derivation_credits_the_replaced_line(self):
        """_derived_text_op must treat the declared replacement as a removal.

        The removable capacity carries exactly the old line (one line, its
        bytes), which makes the trait arithmetic the exact
        replace-one-line-with-one-line arithmetic.
        """
        snippet = GITIGNORE.replace(OLD_LINE, NEW_LINE)
        op, _slack, removable = _derived_text_op(GITIGNORE, snippet)

        assert removable["lines"] == 1
        assert removable["bytes"] == len((OLD_LINE + "\n").encode())
        ok, reason = validate_text_output(
            GITIGNORE,
            op,
            _replace_model(GITIGNORE, snippet),
            tolerance=TOLERANCE_MODEL_PROSE,
            layout_slack=0,
            removable_traits=removable,
        )
        assert ok, reason

    def test_incomplete_restatement_gets_no_removal_credit(self):
        """No completeness → no removal credit → the trait gate stays strict."""
        snippet = "header line\ninserted payload line\n"
        _op, _slack, removable = _derived_text_op(
            "header line\n" + "".join(f"gap line {i}\n" for i in range(5)),
            snippet,
        )
        assert not any(removable.values())  # nothing is justifiably removable
