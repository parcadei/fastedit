"""Step D5 — the GIGO hermetic pins (req. 9, default tier).

The real-LLM side of the GIGO matrix lives in
``tests/test_real_llm_gigo_e2e.py`` (marked ``llm``); this module pins —
hermetically, on the SAME committed fixtures — the two directions a real
model cannot be staged to produce on demand without fabricating behavior:

* **the validator refuses and repeats until correct** (plan req. 9, case c
  of the matrix): merge attempts that fail to produce the commanded
  well-formed frontmatter are rejected by the battery, retried with the
  corrective note, and only a well-formed merge is accepted; an output
  that NEVER converges is refused fail-loud with the original file kept.
  (Measured evidence: the real model applies the declared frontmatter on
  the first attempt of the converging shape — ``retries == 0`` on every
  measured trial — so the natural-failure branch is exercised here with a
  scripted ``merge_fn``, the A2 hermetic-twin pattern.)
* **the negative control** (case h of the matrix): a merge that silently
  "fixes" a typo in an UNTOUCHED region is a trait mutation and is
  rejected by the battery in-pipeline. The real-model side is case (a) of
  the e2e suite — the model preserved the typo live, byte-exact; this is
  the rejection side. (The same shape is pinned at validator level in
  ``test_relative_validation.py::test_helpful_fix_of_untouched_defect_is_retried_then_rejected``
  and at battery level for md in ``test_lang_attributes.py``.)

Fixtures and snippets are imported from the e2e suite so the two tiers can
never drift apart: one corpus, one op spec, one golden.
"""

from __future__ import annotations

from types import SimpleNamespace

from test_real_llm_gigo_e2e import (
    MD_FRONT_SNIPPET,
    MD_ORIGINAL,
    MD_WELLFORMED_BLOCK,
    PY_DEFECTS,
    PY_WRAP_SNIPPET,
    _golden,
    _py_wrap_oracle,
)

from fastedit.inference.chunked_merge import chunked_merge


def _StubMergeResult(merged_code: str, truncated: bool = False):
    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=True,
        tokens_generated=7,
        latency_ms=1.0,
        truncated=truncated,
    )


def _scripted(outputs: list[str]):
    """A scripted merge_fn: yields ``outputs`` in order, then holds the last."""
    calls: list[tuple[str, str, str | None]] = []

    def merge_fn(code: str, snippet: str, lang: str | None):
        calls.append((code, snippet, lang))
        return _StubMergeResult(outputs[min(len(calls) - 1, len(outputs) - 1)])

    return merge_fn, calls


# The model failure the retry loop must correct: the merge output keeps the
# MALFORMED frontmatter (the preserve-by-default instinct winning over the
# declared replacement) — i.e. the command did not land.
_MD_STILL_MALFORMED = MD_ORIGINAL


def test_commanded_frontmatter_fix_is_retried_until_correct(tmp_path):
    """req. 9, case (c) direction 1: refuse → corrective note → correct.

    The first two attempts keep the malformed frontmatter; the battery
    rejects each (the snippet's DECLARED new lines are missing from the
    merge — the content gate, whose line-exact view sees the missing
    declared block before the D3 semantic layer names it), retries with
    the corrective note, and accepts only the well-formed merge. The final
    state is the committed golden: frontmatter well-formed, rest of the
    file byte-exact, ``retries > 0``.
    """
    well_formed = _golden("md_frontmatter/expected_frontmatter_fix.md")
    merge_fn, calls = _scripted([
        _MD_STILL_MALFORMED,
        _MD_STILL_MALFORMED,
        well_formed,
    ])
    result = chunked_merge(
        MD_ORIGINAL, MD_FRONT_SNIPPET, str(tmp_path / "notes.md"), merge_fn,
        language="markdown",
    )
    assert len(calls) == 3, f"unexpected attempt count: {len(calls)}"
    assert result.retries == 2, f"the refusals must surface as retries: {result}"
    assert result.chunks_used == 1 and result.chunks_rejected == 0, str(result)
    # The final state MUST be well-formed — this is the binding requirement.
    assert result.merged_code == well_formed, "the fix did not land"
    assert result.merged_code.startswith(MD_WELLFORMED_BLOCK)
    assert "title: [unclosed list" not in result.merged_code
    # Every retry carried the failure reason as a corrective NOTE.
    assert "NOTE: the previous merge attempt was rejected" in calls[1][1]
    assert "content-faithfulness" in calls[1][1]
    assert "NOTE: the previous merge attempt was rejected" in calls[2][1]


def test_commanded_frontmatter_fix_that_never_converges_is_refused(tmp_path):
    """req. 9, case (c) direction 2: never-correct output is refused.

    Every attempt keeps the malformed frontmatter; the loop exhausts its
    budget and REJECTS the merge fail-loud — the original file is kept
    byte-for-byte and ``parse_valid`` is forced False (the universal
    do-not-persist signal the MCP/CLI gates read). The malformed
    frontmatter is never "half-fixed" into the file.
    """
    merge_fn, calls = _scripted([_MD_STILL_MALFORMED])
    result = chunked_merge(
        MD_ORIGINAL, MD_FRONT_SNIPPET, str(tmp_path / "notes.md"), merge_fn,
        language="markdown", max_validation_retries=1,
    )
    assert len(calls) == 2  # initial + the pinned single retry
    assert result.retries == 1
    assert result.chunks_rejected == 1
    assert result.parse_valid is False
    assert result.merged_code == MD_ORIGINAL, (
        "the refusing merge must never write its mutating payload"
    )
    assert "NOTE: the previous merge attempt was rejected" in calls[1][1]


def test_helpful_typo_fix_in_untouched_region_is_rejected_in_pipeline(tmp_path):
    """req. 9 corollary, case (h): the negative control, in-pipeline.

    The scripted merge output is FAITHFUL for the commanded wrap of
    ``beta`` but silently "fixes" the untouched ``def alpha()`` syntax
    error — a trait mutation. The battery's content-faithfulness gate
    rejects EVERY attempt (the mutated line is neither a survivor nor a
    declared new line), the loop retries with corrective notes, and on
    budget exhaustion the edit is refused fail-loud: the original file is
    kept byte-for-byte and the "fixed" line never reaches the output.

    The real-model complement is case (a) of the e2e suite: the REAL model
    preserved the typo live, byte-exact (golden). The validator-level pin
    of the same corollary is
    ``test_relative_validation.py::test_helpful_fix_of_untouched_defect_is_retried_then_rejected``.
    """
    helpful = _py_wrap_oracle(PY_DEFECTS).replace("def alpha()\n", "def alpha():\n")
    # Sanity: the scripted output DOES perform the commanded wrap — the only
    # unfaithful thing about it is the untouched typo it also "fixed".
    assert "    with audit_lock():\n" in helpful
    assert "def alpha():\n" in helpful and "def alpha()\n" not in helpful

    merge_fn, calls = _scripted([helpful])
    result = chunked_merge(
        PY_DEFECTS, PY_WRAP_SNIPPET, str(tmp_path / "report.py"), merge_fn,
        language="python", replace="beta", max_validation_retries=2,
    )
    assert len(calls) == 3  # initial + the two pinned retries
    assert result.retries == 2
    assert result.chunks_rejected == 1
    # The assembled file is the untouched original (its own defect is a
    # preserved trait, hence relatively parse-valid) — the refusal itself
    # is carried by the rejection bookkeeping, exactly the A2 contract.
    assert result.parse_valid is True
    assert result.merged_code == PY_DEFECTS, (
        "the 'helpfully fixed' merge must never be written — the original "
        "file is kept byte-for-byte"
    )
    assert "def alpha():\n" not in result.merged_code, (
        "the model's repair reached the output"
    )
    assert "NOTE: the previous merge attempt was rejected" in calls[1][1]
    assert "content-faithfulness" in calls[1][1]
