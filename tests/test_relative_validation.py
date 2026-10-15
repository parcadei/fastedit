"""Step A2 — RELATIVE parse validation, unified retry-until-valid loop, GIGO.

Three product behaviors are locked down here (implementation plan Step A2,
requirements 5 and 9):

1. ``parse_diagnostics`` — ordered tree-sitter error traits (byte spans +
   kind) for a parse, with the bare-CR normalization contract the rest of
   the pipeline's AST tooling already follows.
2. ``merged_is_acceptable`` — the RELATIVE parse rule (req. 9,
   EDIT-NOT-CORRECT): acceptance is trait-based comparison against the
   ORIGINAL's own diagnostics, never an absolute well-formedness ideal.
3. The unified retry-until-valid loop in ``chunked_merge`` — one attempt
   loop for both the whole-file and per-chunk paths, one validation
   battery per attempt (relative parse + content faithfulness), the
   failure reason appended to the retry prompt, exhaustion → the existing
   rejection convention, and ``ChunkedMergeResult.retries`` accounting.

The GIGO tests pin the mechanism split that makes EDIT-NOT-CORRECT hold:

* a model that "helpfully fixes" untouched content is caught by the
  CONTENT faithfulness validator (``_check_hallucinations``) — the
  relative parse rule deliberately permits defect removal because the op
  itself may target the broken text;
* a merge that introduces a NEW parse error is caught by the RELATIVE
  parse rule — including the content-clean case (declared lines interact
  with kept lines into invalid syntax) that faithfulness cannot see;
* a faithful merge that preserves a pre-existing defect passes BOTH.

No network, no model: every loop test runs against a scripted merge_fn.
"""

from __future__ import annotations

import pytest

from fastedit.data_gen.ast_analyzer import parse_diagnostics, validate_parse
from fastedit.inference.chunked_merge import (
    _check_hallucinations,
    _max_validation_retries,
    _merge_rejection_reason,
    merged_is_acceptable,
)
from fastedit.split_join import normalize_bare_cr_for_ast

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _StubMergeResult(merged_code, truncated=False, parse_valid=True):
    """Minimal engine-result stand-in for scripted merge_fn sequences."""
    from types import SimpleNamespace

    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=parse_valid,
        tokens_generated=9,
        latency_ms=1.0,
        truncated=truncated,
    )


def _scripted(results):
    """A merge_fn replaying *results* (last one repeats), recording calls."""
    calls: list[tuple[str, str, str | None]] = []

    def merge_fn(code, snippet, language):
        calls.append((code, snippet, language))
        return results[min(len(calls) - 1, len(results) - 1)]

    return merge_fn, calls


# 30 filler lines above and below the target function, so the beta chunk
# never spans the whole file and the PER-CHUNK loop runs.
FILLER_TOP = "\n".join(f"x{i} = {i}" for i in range(30))
FILLER_BOTTOM = "\n".join(f"y{i} = {i}" for i in range(30))


def _target_file(body: str) -> str:
    """A file whose only function is `beta` with the given body."""
    return (
        FILLER_TOP
        + "\n\ndef beta():\n"
        + body
        + "\n"
        + FILLER_BOTTOM
        + "\n"
    )


# A pre-existing defect INSIDE the edit target: `retrun` is a syntax error
# tree-sitter reports as an ERROR trait on its own line.
BROKEN_BODY = "    retrun 1\n    return 0\n"
# The beta chunk keeps the broken line (a preserved trait) and applies the
# snippet's declared change to the last line.
BROKEN_TARGET_FILE = _target_file(BROKEN_BODY)
BROKEN_TARGET_SNIPPET = (
    "def beta():\n"
    "    retrun 1\n"
    "# ... existing code ...\n"
    "    return 99\n"
)
FAITHFUL_BROKEN_CHUNK = "def beta():\n    retrun 1\n    return 99\n"
HELPFUL_FIX_CHUNK = "def beta():\n    return 1\n    return 99\n"
BROKEN_MERGE_CHUNK = "def beta():\n    retrun 1\n    return 99\n    )\n"

CLEAN_BODY = "    return 1\n    return 0\n"
CLEAN_TARGET_FILE = _target_file(CLEAN_BODY)
CLEAN_TARGET_SNIPPET = (
    "def beta():\n"
    "    return 1\n"
    "# ... existing code ...\n"
    "    return 99\n"
)
FAITHFUL_CLEAN_CHUNK = "def beta():\n    return 1\n    return 99\n"

WHOLE_BROKEN_FILE = "def foo():\n    retrun 1\n"
WHOLE_BROKEN_SNIPPET = "def foo():\n# ... existing code ...\n"
WHOLE_HELPFUL_FIX = "def foo():\n    return 1\n"


def _write_and_merge(
    tmp_path, original, snippet, merge_fn, language="python", **kwargs,
):
    from fastedit.inference.chunked_merge import chunked_merge

    target = tmp_path / "mod.py"
    target.write_text(original)
    result = chunked_merge(
        original, snippet, str(target), merge_fn, language=language, **kwargs,
    )
    return result


# ---------------------------------------------------------------------------
# parse_diagnostics — ordered error traits
# ---------------------------------------------------------------------------


class TestParseDiagnostics:
    def test_clean_file_has_no_error_traits(self):
        src = (
            "import os\n\n\ndef alpha():\n    return 1\n\n\nclass C:\n"
            "    def m(self):\n        return 2\n"
        )
        diags = parse_diagnostics(src, "python")
        assert diags.errors == []
        assert diags.is_valid is True
        assert diags.source == src

    def test_missing_token_yields_one_span_inside_the_broken_construct(self):
        src = 'def alpha():\n    return "a"\n\n\ndef beta(:\n    total = 1\n'
        diags = parse_diagnostics(src, "python")
        assert diags.is_valid is False
        assert len(diags.errors) == 1
        start, end, kind = diags.errors[0]
        assert kind == "MISSING"
        # A MISSING token is zero-width and sits inside the broken def line.
        assert src.encode()[start:end] == b""
        broken_line_start = src.index("def beta(")
        assert broken_line_start <= start < src.index("\n", broken_line_start)

    def test_error_node_span_covers_the_broken_construct(self):
        src = 'def alpha():\n    return "a"\n\n\ndef foo()\n    return 1\n'
        diags = parse_diagnostics(src, "python")
        assert diags.is_valid is False
        assert len(diags.errors) == 1
        start, end, kind = diags.errors[0]
        assert kind == "ERROR"
        assert src.encode()[start:end].startswith(b"def foo()")

    def test_same_error_in_different_positions_moves_the_span(self):
        broken = "def foo()\n    return 1\n"
        prefix = "x = 1\ny = 2\n"
        d1 = parse_diagnostics(broken, "python")
        d2 = parse_diagnostics(prefix + broken, "python")
        assert len(d1.errors) == len(d2.errors) == 1
        s1, e1, k1 = d1.errors[0]
        s2, e2, k2 = d2.errors[0]
        assert k1 == k2 == "ERROR"
        shift = len(prefix.encode())
        assert s2 - s1 == shift
        assert e2 - e1 == shift
        # Both spans index the SAME broken construct text.
        assert broken.encode()[s1:e1] == (prefix + broken).encode()[s2:e2]

    def test_bare_cr_normalization_contract(self):
        # The pipeline's AST tooling parses the bare-CR-normalized copy
        # (split_join.normalize_bare_cr_for_ast; cf. chunk_locator).
        # parse_diagnostics normalizes the same way, and ``.source`` carries
        # exactly the text its byte spans index — consumers must read line
        # text from ``.source``, never re-derive lines from another
        # normalization (a bare CR changes where lines break).
        raw = "def foo():\r    return 1\r\ndef bar()\r    return 2\n"
        normalized = normalize_bare_cr_for_ast(raw)
        assert normalized != raw  # a bare CR was swapped for LF
        d_raw = parse_diagnostics(raw, "python")
        d_norm = parse_diagnostics(normalized, "python")
        assert d_raw.source == normalized
        assert d_raw.errors == d_norm.errors
        assert d_raw.is_valid is False
        # The 1-byte-for-1-byte substitution keeps every OFFSET identical:
        # spans point at the same byte positions in the raw text, but their
        # TEXT differs exactly where CRs were swapped — read it .source.
        raw_bytes = raw.encode()
        norm_bytes = normalized.encode()
        assert len(raw_bytes) == len(norm_bytes)
        for start, end, kind in d_raw.errors:
            assert d_raw.source.encode()[start:end] == norm_bytes[start:end]
            assert kind in ("ERROR", "MISSING")
        # Line text for defect identity comes from .source, never from the
        # raw text (a bare CR would change where the line breaks).
        start = d_raw.errors[0][0]
        ls = normalized.rfind("\n", 0, start) + 1
        le = normalized.find("\n", start)
        assert normalized[ls:le] == "def bar()"

    def test_is_valid_agrees_with_validate_parse(self):
        for src in (
            "def foo():\n    return 1\n",
            "def foo(:\n    pass\n",
            "def foo()\n    return 1\n",
            "x = (\n",
        ):
            diags = parse_diagnostics(src, "python")
            assert diags.is_valid == validate_parse(src, "python")

    def test_traits_are_ordered_by_document_position(self):
        src = (
            "def foo()\n    return 1\n\n\ndef bar()\n    return 2\n\n\n"
            "def baz(:\n    pass\n"
        )
        diags = parse_diagnostics(src, "python")
        starts = [s for s, _e, _k in diags.errors]
        assert starts == sorted(starts)
        assert len(diags.errors) >= 2


# ---------------------------------------------------------------------------
# merged_is_acceptable — the relative (trait-based) parse rule
# ---------------------------------------------------------------------------


class TestMergedIsAcceptable:
    def test_original_valid_merged_valid_is_accepted(self):
        ok, reason = merged_is_acceptable(
            parse_diagnostics("def f():\n    return 1\n", "python"),
            parse_diagnostics("def f():\n    return 2\n", "python"),
        )
        assert ok is True
        assert reason == ""

    def test_original_valid_merged_invalid_is_rejected_with_reason(self):
        ok, reason = merged_is_acceptable(
            parse_diagnostics("def f():\n    return 1\n", "python"),
            parse_diagnostics("def f():\n    return 1\n    )\n", "python"),
        )
        assert ok is False
        assert "parse error" in reason
        assert "bytes" in reason  # points at the offending span

    def test_original_invalid_defect_inherited_is_accepted(self):
        original = "def f():\n    retrun 1\n\ndef g():\n    return 2\n"
        merged = "def f():\n    retrun 1\n\ndef g():\n    return 3\n"
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(merged, "python"),
        )
        assert ok is True, reason
        assert reason == ""

    def test_original_invalid_defect_removed_is_accepted(self):
        # The op may target the broken text: the defect vanishing is fine.
        original = "def f():\n    retrun 1\n\ndef g():\n    return 2\n"
        merged = "def f():\n    return 1\n\ndef g():\n    return 2\n"
        ok, _reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(merged, "python"),
        )
        assert ok is True

    def test_original_invalid_new_error_is_rejected(self):
        original = "def f():\n    retrun 1\n\ndef g():\n    return 2\n"
        merged = "def f():\n    retrun 1\n\ndef g():\n    return 2\n    )\n"
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(merged, "python"),
        )
        assert ok is False
        assert "new parse error" in reason

    def test_edited_spans_excuse_errors_inside_them(self):
        # Inside an edited span the op spec governs: a caller that knows an
        # op legitimately declares malformed output in a region passes the
        # span and the rule excuses errors overlapping it.
        original = "def f():\n    retrun 1\n\ndef g():\n    return 2\n"
        merged = "def f():\n    retrun 1\n\ndef g():\n    return 2\n    )\n"
        merged_diags = parse_diagnostics(merged, "python")
        assert len(merged_diags.errors) == 2  # inherited trait + the new one
        new_span = merged_diags.errors[-1]
        ok, _ = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            merged_diags,
            edited_spans=[(new_span[0], new_span[1])],
        )
        assert ok is True
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"), merged_diags,
        )
        assert ok is False and "new parse error" in reason

    def test_multiset_matching_counts_identical_defect_lines(self):
        # Two textually identical broken lines; the op fixes ONE of them.
        original = "def f():\n    retrun 1\n\ndef g():\n    retrun 2\n"
        merged = "def f():\n    return 1\n\ndef g():\n    retrun 2\n"
        ok, _ = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(merged, "python"),
        )
        assert ok is True
        # The op fixes NEITHER and a third identical-looking defect appears
        # on a new line: the multiset exceeds the original's count → reject.
        worse = "def f():\n    retrun 1\n    retrun 5\n\ndef g():\n    retrun 2\n"
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(worse, "python"),
        )
        assert ok is False
        assert "new parse error" in reason

    def test_reason_is_human_readable_for_the_retry_note(self):
        original = "def f():\n    retrun 1\n\ndef g():\n    return 2\n"
        merged = "def f():\n    retrun 1\n\ndef g():\n    return 2\n    )\n"
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(merged, "python"),
        )
        assert ok is False
        # kind, byte span and offending line are all named
        assert "ERROR" in reason or "MISSING" in reason
        assert "bytes 49-50" in reason
        assert "'    )'" in reason


# ---------------------------------------------------------------------------
# The validation battery (shared gate order for both retry loops)
# ---------------------------------------------------------------------------


class TestMergeRejectionReason:
    def test_faithful_merge_preserving_pre_existing_defect_passes(self):
        # THE req. 9 headline: an edit next to a pre-existing syntax error
        # lands, the defect survives byte-exact, the battery is silent.
        original = "def alpha():\n    retrun 1\n\ndef beta():\n    return 2\n"
        snippet = "def beta():\n# ... existing code ...\n    return 3\n"
        merged = "def alpha():\n    retrun 1\n\ndef beta():\n    return 3\n"
        assert _merge_rejection_reason(
            original, merged, snippet, "python", False,
        ) is None

    def test_truncation_short_circuits_first(self):
        reason = _merge_rejection_reason(
            "def f():\n    return 1\n", "def f(:\n", "x", "python", True,
        )
        assert reason == "truncated (model hit the token cap)"

    def test_helpful_typo_fix_is_caught_by_faithfulness_not_by_the_parse_rule(self):
        # req. 9 corollary: a model that "fixes" an UNTOUCHED typo is a
        # trait mutation. The relative parse rule deliberately accepts it
        # (defect removal is allowed — the op may target the defect), so
        # the CONTENT faithfulness validator is the mechanism that rejects.
        original = "def alpha():\n    retrun 1\n\ndef beta():\n    return 2\n"
        snippet = "def beta():\n# ... existing code ...\n    return 2\n"
        helpful = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        # The content validator rejects...
        assert _check_hallucinations(original, helpful, snippet) == 0.0
        # ...while the parse rule alone would accept it (documented gap):
        ok, _ = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(helpful, "python"),
        )
        assert ok is True
        # Through the battery → the faithfulness reason wins.
        reason = _merge_rejection_reason(original, helpful, snippet, "python", False)
        assert reason is not None and "content-faithfulness" in reason

    def test_new_error_in_untouched_region_is_caught_by_both_gates(self):
        original = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        snippet = "def alpha():\n# ... existing code ...\n    return 9\n"
        corrupted = "def alpha():\n    return 9\n\ndef beta()\n    return 2\n"
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(corrupted, "python"),
        )
        assert ok is False and "parse error" in reason
        assert _check_hallucinations(original, corrupted, snippet) == 0.0

    def test_content_clean_but_parse_breaking_merge_is_caught_by_the_parse_rule(
        self,
    ):
        # Declared new lines interacting with kept lines into invalid
        # syntax: content-faithfulness alone ACCEPTS this merge (every line
        # is accounted for); only the relative parse rule sees it.
        original = "def alpha():\n    pass\n\ndef beta():\n    return 2\n"
        snippet = "def alpha():\n    x = (\n# ... existing code ...\n"
        broken = "def alpha():\n    x = (\n    pass\n\ndef beta():\n    return 2\n"
        assert _check_hallucinations(original, broken, snippet) == 1.0
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original, "python"),
            parse_diagnostics(broken, "python"),
        )
        assert ok is False and "parse error" in reason
        # The battery rejects with the parse reason, not faithfulness.
        reason = _merge_rejection_reason(original, broken, snippet, "python", False)
        assert reason is not None and "does not parse as python" in reason

    def test_battery_skips_parse_gate_without_language(self):
        # language=None (structureless/text files): no parse gate — a merge
        # that would fail an ABSOLUTE parse check passes the battery when
        # no language is known; the content validator still runs.
        from fastedit.data_gen.ast_analyzer import validate_parse

        broken = "def f(:\n    pass\n"
        assert not validate_parse(broken, "python")  # the skipped gate
        assert _merge_rejection_reason(
            broken, broken, broken, None, False,
        ) is None
        # Content mutations are still caught without a language.
        assert _merge_rejection_reason(
            "keep me\n", "changed me\n", "keep me\n", None, False,
        ) is not None


# ---------------------------------------------------------------------------
# Unified retry loop — per-chunk path
# ---------------------------------------------------------------------------


class TestChunkRetryLoop:
    def test_clean_first_attempt_needs_no_retry(self, tmp_path):
        merge_fn, calls = _scripted([_StubMergeResult(FAITHFUL_CLEAN_CHUNK)])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
        )
        assert len(calls) == 1
        assert result.retries == 0
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert "    return 99" in result.merged_code

    def test_broken_original_faithful_edit_lands_with_defect_preserved(
        self, tmp_path,
    ):
        # req. 9: the target function itself carries a pre-existing syntax
        # error; the snippet edits a VALID line of it. The edit must land
        # and the defect must survive byte-exact — with parse_valid True
        # (the relative rule), where the old absolute gate failed the chunk.
        merge_fn, calls = _scripted([_StubMergeResult(FAITHFUL_BROKEN_CHUNK)])
        result = _write_and_merge(
            tmp_path, BROKEN_TARGET_FILE, BROKEN_TARGET_SNIPPET, merge_fn,
        )
        assert len(calls) == 1, f"unexpected retries: {metrics(calls)}"
        assert result.retries == 0
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        # The defect survived byte-exact; the declared edit landed.
        assert "    retrun 1\n" in result.merged_code
        assert "    return 99\n" in result.merged_code

    def test_helpful_fix_of_untouched_defect_is_retried_then_rejected(
        self, tmp_path,
    ):
        # GIGO: the model keeps "fixing" the untouched `retrun` typo. The
        # battery's content-faithfulness gate rejects every attempt; the
        # loop retries with corrective notes and finally rejects the chunk
        # (original kept) instead of writing the "corrected" code.
        merge_fn, calls = _scripted([_StubMergeResult(HELPFUL_FIX_CHUNK)])
        result = _write_and_merge(
            tmp_path, BROKEN_TARGET_FILE, BROKEN_TARGET_SNIPPET, merge_fn,
            max_validation_retries=1,
        )
        assert len(calls) == 2  # initial + the pinned single retry
        assert result.retries == 1
        assert result.chunks_rejected == 1
        # The relative parse verdict describes the ASSEMBLED file — here the
        # untouched original (its own defect is a preserved trait), so it is
        # True; the rejection itself is surfaced via chunks_rejected (the
        # MCP/CLI gates consume that first, exactly as before Step A2).
        assert result.parse_valid is True
        # The untouched defective line survived — fastedit never corrects.
        assert result.merged_code == BROKEN_TARGET_FILE
        # Every retry prompt carried the failure reason as a NOTE.
        assert "NOTE:" in calls[1][1]
        assert "content-faithfulness" in calls[1][1]

    def test_valid_chunk_plus_broken_merge_is_retried_then_rejected(
        self, tmp_path,
    ):
        merge_fn, calls = _scripted([_StubMergeResult(BROKEN_MERGE_CHUNK)])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
            max_validation_retries=1,
        )
        assert len(calls) == 2
        assert result.retries == 1
        assert result.chunks_rejected == 1
        assert result.merged_code == CLEAN_TARGET_FILE
        assert "NOTE:" in calls[1][1]
        assert "does not parse as python" in calls[1][1]

    def test_transient_failure_then_clean_attempt_is_accepted(self, tmp_path):
        merge_fn, calls = _scripted([
            _StubMergeResult(BROKEN_MERGE_CHUNK),
            _StubMergeResult(FAITHFUL_CLEAN_CHUNK),
        ])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
        )
        assert len(calls) == 2
        assert result.retries == 1
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert "    return 99" in result.merged_code
        assert "NOTE:" in calls[1][1]
        # The FIRST call sees the plain snippet — no note contamination.
        assert "NOTE:" not in calls[0][1]

    def test_default_budget_is_eight_retries(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FASTEDIT_MAX_RETRIES", raising=False)
        merge_fn, calls = _scripted([_StubMergeResult(BROKEN_MERGE_CHUNK)])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
        )
        assert len(calls) == 9  # 1 initial + 8 retries
        assert result.retries == 8
        assert result.chunks_rejected == 1
        assert result.merged_code == CLEAN_TARGET_FILE

    def test_max_validation_retries_kwarg_pins_the_budget(self, tmp_path):
        merge_fn, calls = _scripted([_StubMergeResult(BROKEN_MERGE_CHUNK)])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
            max_validation_retries=3,
        )
        assert len(calls) == 4
        assert result.retries == 3
        assert result.chunks_rejected == 1

    def test_env_budget_and_kwarg_precedence(self, tmp_path, monkeypatch):
        from fastedit.inference.chunked_merge import chunked_merge

        target = tmp_path / "mod.py"
        target.write_text(CLEAN_TARGET_FILE)

        # env overrides the default…
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "0")
        merge_fn, calls = _scripted([_StubMergeResult(BROKEN_MERGE_CHUNK)])
        chunked_merge(
            CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, str(target), merge_fn,
            language="python",
        )
        assert len(calls) == 1  # budget 0 → single attempt, then reject
        assert calls[0][2] == "python"

        # …an explicit kwarg beats the env…
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "0")
        merge_fn, calls = _scripted([_StubMergeResult(BROKEN_MERGE_CHUNK)])
        chunked_merge(
            CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, str(target), merge_fn,
            language="python", max_validation_retries=2,
        )
        assert len(calls) == 3

        # …and a malformed env value fails loudly (repo convention).
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "bogus")
        with pytest.raises(ValueError):
            chunked_merge(
                CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, str(target),
                lambda *a: _StubMergeResult(FAITHFUL_CLEAN_CHUNK),
                language="python",
            )

    def test_attempt_tokens_are_all_accounted(self, tmp_path):
        # Honest accounting: every consumed attempt's tokens/latency sum
        # into the result, not just the last attempt's.
        merge_fn, calls = _scripted([
            _StubMergeResult(BROKEN_MERGE_CHUNK),
            _StubMergeResult(FAITHFUL_CLEAN_CHUNK),
        ])
        result = _write_and_merge(
            tmp_path, CLEAN_TARGET_FILE, CLEAN_TARGET_SNIPPET, merge_fn,
        )
        assert len(calls) == 2
        assert result.model_tokens == 18  # 9 per attempt × 2 attempts
        assert result.latency_ms == 2.0


# ---------------------------------------------------------------------------
# Unified retry loop — whole-file path
# ---------------------------------------------------------------------------


class TestWholeFileRetryLoop:
    def _merge(self, tmp_path, results, language="python", **kwargs):
        merge_fn, calls = _scripted(results)
        result = _write_and_merge(
            tmp_path, WHOLE_BROKEN_FILE, WHOLE_BROKEN_SNIPPET, merge_fn,
            language=language, **kwargs,
        )
        return result, calls

    def test_faithful_merge_of_broken_file_is_accepted(self, tmp_path):
        # req. 9: the whole-file path used to REJECT this edit (absolute
        # parse gate) or push the model to "repair" the untouched defect.
        # Now the preserved defect is a trait and the merge is accepted.
        result, calls = self._merge(tmp_path, [_StubMergeResult(WHOLE_BROKEN_FILE)])
        assert len(calls) == 1
        assert result.retries == 0
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == WHOLE_BROKEN_FILE

    def test_helpful_fix_is_retried_until_budget_exhaustion(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FASTEDIT_MAX_RETRIES", raising=False)
        result, calls = self._merge(tmp_path, [_StubMergeResult(WHOLE_HELPFUL_FIX)])
        assert len(calls) == 9
        assert result.retries == 8
        assert result.chunks_rejected == 1
        assert result.chunks_used == 1
        assert result.parse_valid is False
        assert result.merged_code == WHOLE_BROKEN_FILE
        # Retry prompts carry the faithfulness reason.
        assert all("NOTE:" in c[1] for c in calls[1:])
        assert "content-faithfulness" in calls[1][1]

    def test_bad_then_good_is_accepted_with_corrective_note(self, tmp_path):
        result, calls = self._merge(tmp_path, [
            _StubMergeResult(WHOLE_HELPFUL_FIX),
            _StubMergeResult(WHOLE_BROKEN_FILE),
        ])
        assert len(calls) == 2
        assert result.retries == 1
        assert result.parse_valid is True
        assert result.chunks_rejected == 0
        assert result.merged_code == WHOLE_BROKEN_FILE
        assert "NOTE:" not in calls[0][1]
        assert "NOTE:" in calls[1][1]

    def test_exhaustion_keeps_rejection_convention(self, tmp_path):
        result, calls = self._merge(
            tmp_path, [_StubMergeResult(WHOLE_HELPFUL_FIX)], max_validation_retries=0,
        )
        assert len(calls) == 1
        assert result.retries == 0
        assert result.chunks_rejected == 1
        assert result.parse_valid is False
        assert result.merged_code == WHOLE_BROKEN_FILE

    def test_attempt_tokens_are_all_accounted(self, tmp_path):
        result, calls = self._merge(tmp_path, [
            _StubMergeResult(WHOLE_HELPFUL_FIX),
            _StubMergeResult(WHOLE_BROKEN_FILE),
        ])
        assert len(calls) == 2
        assert result.model_tokens == 18
        assert result.latency_ms == 2.0


# ---------------------------------------------------------------------------
# Retry budget resolution (module-level env plumbing)
# ---------------------------------------------------------------------------


class TestMaxValidationRetries:
    def test_default_is_eight(self, monkeypatch):
        monkeypatch.delenv("FASTEDIT_MAX_RETRIES", raising=False)
        assert _max_validation_retries() == 8

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "3")
        assert _max_validation_retries() == 3

    def test_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "3")
        assert _max_validation_retries(1) == 1

    def test_malformed_env_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "bogus")
        with pytest.raises(ValueError):
            _max_validation_retries()

    def test_negative_values_are_rejected(self, monkeypatch):
        monkeypatch.setenv("FASTEDIT_MAX_RETRIES", "-1")
        with pytest.raises(ValueError):
            _max_validation_retries()
        with pytest.raises(ValueError):
            _max_validation_retries(-1)


# ---------------------------------------------------------------------------
# Deterministic gate — the relative rule on a PARTIALLY broken file
# ---------------------------------------------------------------------------


class TestDeterministicGateRelativeRule:
    def test_deterministic_edit_lands_on_partially_broken_file(self, tmp_path):
        # A file whose broken `alpha` swallows itself in error recovery but
        # whose later `locked` still parses (and is AST-resolvable): a
        # deterministic replace= edit of `locked` must LAND under the
        # relative rule — the old absolute gate discarded the identical,
        # content-faithful result because the merged FILE still carries
        # alpha's pre-existing error, and pushed the edit to the model.
        from fastedit.inference.chunked_merge import chunked_merge

        original = (
            "def alpha()\n"
            "    total = 1\n"
            "    return total\n"
            "\n"
            "\n"
            "def locked():\n"
            "    return 1\n"
        )
        snippet = (
            "def locked():\n"
            "    total = 5\n"
            "# ... existing code ...\n"
            "    return 1\n"
        )
        target = tmp_path / "mod.py"
        target.write_text(original)

        def merge_fn(code, snippet, language):
            raise AssertionError(
                "model path must not run — the deterministic result is "
                "relatively valid (only the pre-existing defect remains)"
            )

        result = chunked_merge(
            original, snippet, str(target), merge_fn,
            language="python", replace="locked",
        )
        assert result.model_tokens == 0
        assert result.chunks_used == 0
        assert result.retries == 0
        assert result.parse_valid is True
        # The edit landed in `locked`; the untouched defect survived
        # byte-exact (EDIT-NOT-CORRECT).
        assert "    total = 5\n" in result.merged_code
        assert "def alpha()\n" in result.merged_code
        assert "    return 1\n" in result.merged_code


# ---------------------------------------------------------------------------
# Step A3 — the two remaining ABSOLUTE gates go relative:
# the `after=` fast path and `_merge_preserve_siblings`
# ---------------------------------------------------------------------------


def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if called — both gates are zero-model."""
    raise AssertionError("merge_fn must NOT be called on these fast paths")


# A pre-existing defect the edit never comes near (broken `alpha`), while the
# anchor `beta` still parses and is AST-resolvable (same shape the
# deterministic-gate test above uses).
AFTER_BROKEN_FILE = (
    "def alpha()\n"
    "    total = 1\n"
    "    return total\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 1\n"
)
AFTER_CLEAN_FILE = "def beta():\n    return 1\n"
AFTER_CLEAN_SNIPPET = "def gamma():\n    return 2\n"
# A snippet that is itself malformed Python — the insert would introduce a
# NEW error trait the original does not have.
AFTER_BROKEN_SNIPPET = "def gamma(\n    return 2\n"


class TestAfterFastPathRelativeGate:
    def _merge(self, tmp_path, original, snippet, **kwargs):
        from fastedit.inference.chunked_merge import chunked_merge

        target = tmp_path / "mod.py"
        target.write_text(original)
        return chunked_merge(
            original, snippet, str(target), _no_model,
            language="python", after="beta", **kwargs,
        )

    def test_broken_original_insert_lands_with_defect_preserved(self, tmp_path):
        # req. 9: an INSERTION cannot fix a pre-existing error elsewhere, so
        # the relative rule applies directly — every original error trait
        # survives byte-exact and is inherited. The old ABSOLUTE gate refused
        # this exact edit (validate_parse(merged) is False) even though the
        # insert itself was clean.
        result = self._merge(tmp_path, AFTER_BROKEN_FILE, AFTER_CLEAN_SNIPPET)
        assert result.parse_valid is True
        assert result.model_tokens == 0
        assert result.retries == 0
        assert result.chunks_used == 0
        # The untouched defect survived byte-exact; the insert landed.
        assert "def alpha()\n" in result.merged_code
        assert "def gamma():\n    return 2\n" in result.merged_code

    def test_insert_that_breaks_a_clean_file_is_refused(self, tmp_path):
        # No loosening: a malformed snippet inserted into a VALID file is a
        # regression. The zero-model path keeps its refusal style — the
        # result is returned with parse_valid=False (no exception, no
        # retry); the MCP/CLI write gates then refuse the write.
        result = self._merge(tmp_path, AFTER_CLEAN_FILE, AFTER_BROKEN_SNIPPET)
        assert result.parse_valid is False
        assert result.model_tokens == 0
        assert "def gamma(\n" in result.merged_code

    def test_insert_cannot_introduce_a_new_error_on_a_broken_file(self, tmp_path):
        # The insertion zone is NOT an excused edited span: new code the op
        # declares must parse, even when the file was already broken. The
        # insert's error trait is new (the original's own defect is a
        # different line) → refused.
        result = self._merge(tmp_path, AFTER_BROKEN_FILE, AFTER_BROKEN_SNIPPET)
        assert result.parse_valid is False
        assert result.model_tokens == 0


# Kotlin class with a pre-existing syntax error INSIDE a sibling method that
# preserve_siblings carries over verbatim (the missing `)` is a MISSING trait
# on the `items.add(item` line — a byte-identical inherited trait after the
# splice). The class itself still parses and is AST-resolvable.
KT_BROKEN_SIBLING_FILE = """\
class Store {
    private val items: MutableList<String> = mutableListOf()

    fun add(item: String) {
        items.add(item
    }

    fun size(): Int {
        return items.size
    }
}
"""
KT_NARROW_SNIPPET = """\
class Store {
    private val items: MutableMap<Int, String> = mutableMapOf()
}
"""
KT_CLEAN_FILE = """\
class Store {
    private val items: MutableList<String> = mutableListOf()

    fun size(): Int {
        return items.size
    }
}
"""
# A snippet whose class SHELL is malformed — the splice would introduce a
# new error trait the clean original does not have.
KT_BAD_SHELL_SNIPPET = """\
class Store {
    private val items: MutableList<String> = mutableListOf(
}
"""


class TestPreserveSiblingsRelativeGate:
    def _merge(self, tmp_path, original, snippet, **kwargs):
        from fastedit.inference.chunked_merge import chunked_merge

        target = tmp_path / "Store.kt"
        target.write_text(original)
        return chunked_merge(
            original, snippet, str(target), _no_model,
            language="kotlin", replace="Store", preserve_siblings=True,
            **kwargs,
        )

    def test_broken_original_edit_lands_with_defect_preserved(self, tmp_path):
        # req. 9: the edit replaces the class shell; the broken sibling is
        # carried over VERBATIM, so its error trait is inherited (matched by
        # kind + containing line). The class span is the op's governed
        # region (passed as the edited span); the old ABSOLUTE gate refused
        # this exact edit because the merged FILE still carries the defect.
        result = self._merge(tmp_path, KT_BROKEN_SIBLING_FILE, KT_NARROW_SNIPPET)
        assert result.parse_valid is True
        assert result.model_tokens == 0
        assert result.retries == 0
        assert result.chunks_used == 0
        # The defect survived byte-exact inside the preserved sibling; the
        # declared field change and the untouched sibling both landed.
        assert "        items.add(item\n" in result.merged_code
        assert "    fun size(): Int {\n        return items.size\n    }\n" in result.merged_code
        assert "MutableMap<Int, String>" in result.merged_code

    def test_malformed_shell_cannot_introduce_a_new_error(self, tmp_path):
        # No loosening: on a CLEAN original the edited span excuses nothing
        # (the relative rule requires a clean merge of a clean file), so a
        # malformed snippet shell is refused — the path's existing refusal
        # style (parse_valid=False on the returned result, never an
        # exception; the MCP/CLI gates refuse the write).
        result = self._merge(tmp_path, KT_CLEAN_FILE, KT_BAD_SHELL_SNIPPET)
        assert result.parse_valid is False
        assert result.model_tokens == 0
        # The preserved sibling is still in the (unwritable) output.
        assert "        return items.size\n" in result.merged_code


def metrics(calls) -> str:
    """Tiny helper for failure messages in the loop tests."""
    return f"calls={len(calls)}"


# ---------------------------------------------------------------------------
# C2 stress defect #3 — tree-sitter-python silently recovers vanished suites
# ---------------------------------------------------------------------------

# Found by the 100MB real-LLM stress tier: tree-sitter-python's external
# INDENT/DEDENT machinery re-anchors a compound statement whose body is
# missing WITHOUT emitting ERROR or MISSING nodes, so parse_diagnostics
# reported a file CPython refuses with "expected an indented block" as
# parse-valid — and the pipeline wrote it (the MCP write gate trusts
# parse_valid). parse_diagnostics now consults a tokenizer-level
# suite-opener scan for colon-headed indentation languages.

EMPTY_SUITE_FILE = (
    "def f(x):\n"
    "    if x:\n"
    "    return 1\n"
)

# The same file, VALID — every suite opens.
INDENT_CLEAN_FILE = (
    "def f(x):\n"
    '    """Usage:\n'
    "\n"
    "        f(1)\n"
    '    """\n'
    "    d = {\n"
    '        "a": 1,\n'
    "    }\n"
    "    g = lambda v: v + 1\n"
    "    if x: return d\n"
    "    for k in d:\n"
    "        if k == 'a':\n"
    "            continue\n"
    "    return d\n"
)


def test_python_empty_suite_detected_despite_tree_sitter_recovery():
    """A vanished suite is an INDENT trait even though tree-sitter is clean.

    Pre-fix: parse_diagnostics returned is_valid=True for this file —
    CPython raises ``SyntaxError: expected an indented block`` — so the
    pipeline wrote a file that cannot even be imported. The CPython failure
    is asserted as the ground truth the scanner must agree with.
    """
    import ast

    with pytest.raises(SyntaxError, match="indented block"):
        ast.parse(EMPTY_SUITE_FILE)
    diags = parse_diagnostics(EMPTY_SUITE_FILE, "python")
    assert not diags.is_valid, (
        "the empty suite must be flagged despite tree-sitter's recovery"
    )
    assert any(kind == "INDENT" for _s, _e, kind in diags.errors), diags.errors


def test_python_suite_scanner_has_no_false_positives():
    """Valid python with colon-bearing strings, docstrings, dict literals,
    lambdas, inline suites and comments must stay parse-valid."""
    diags = parse_diagnostics(INDENT_CLEAN_FILE, "python")
    assert diags.is_valid, diags.errors


def test_validate_parse_sees_suite_defects():
    """The absolute check shares the strengthened defect set."""
    assert validate_parse(INDENT_CLEAN_FILE, "python") is True
    assert validate_parse(EMPTY_SUITE_FILE, "python") is False


def test_merge_introducing_empty_suite_is_rejected():
    """The relative rule refuses a merge whose only defect is the vanished
    suite the tree-sitter walk cannot see."""
    original_diags = parse_diagnostics(INDENT_CLEAN_FILE, "python")
    merged_diags = parse_diagnostics(EMPTY_SUITE_FILE, "python")
    ok, reason = merged_is_acceptable(original_diags, merged_diags)
    assert not ok, "an empty-suite merge of a clean file must be rejected"
    assert "parse error" in reason


def test_inherited_empty_suite_trait_is_preserved():
    """GIGO (req. 9): the defect is a trait like any other — a broken
    original's unrelated edit keeps it (and the relative rule accepts)."""
    broken_with_edit = EMPTY_SUITE_FILE.replace("    return 1\n", "    return 2\n")
    ok, _reason = merged_is_acceptable(
        parse_diagnostics(EMPTY_SUITE_FILE, "python"),
        parse_diagnostics(broken_with_edit, "python"),
    )
    assert ok, "a preserved pre-existing suite defect must stay acceptable"
    # ...and "helpfully" repairing it is also trait-consistent defect removal
    # (the CONTENT validator is the mechanism that blocks unwanted edits);
    # the relative rule must not call defect removal a regression either.
    ok_fixed, _reason = merged_is_acceptable(
        parse_diagnostics(EMPTY_SUITE_FILE, "python"),
        parse_diagnostics(INDENT_CLEAN_FILE, "python"),
    )
    assert ok_fixed
