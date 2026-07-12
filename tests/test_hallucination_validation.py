"""Regression matrix for VAL-001: hallucination validator fidelity.

Issue #3 / VAL-001: the validator must accept a faithful line-replacing
merge and reject variants that

    * delete unrelated code,
    * omit required snippet code,
    * duplicate or reorder code,
    * invent content,
    * leak a marker, or
    * mishandle repeated identical lines via set collapse instead of
      multiplicity.

The validator under test is the private ``_check_hallucinations`` helper
in :mod:`fastedit.inference.chunked_merge`. It returns a float in
``[0.0, 1.0]``; downstream code retries when the score is below ``0.85``
and rejects the chunk entirely when it is below ``0.5``. These tests
treat any score at or above ``0.85`` as ACCEPT and anything strictly below
``0.85`` as REJECT, matching the production policy.

The exact issue #3 reproduction is a one-line return-value replacement:

    def foo():
        x = 1
        return x        <- original
        -> return x + 1 <- intent

The previous implementation scored this faithful merge ``0.0``, triggered
two model calls, rejected the chunk, and left the original file
unchanged. The tests below codify the behaviour the fix must satisfy.
"""

from __future__ import annotations

import pytest

from fastedit.inference.chunked_merge import _check_hallucinations


PASS_THRESHOLD = 0.85
REJECT_THRESHOLD = 0.5


def _score(original: str, merged: str, snippet: str) -> float:
    return _check_hallucinations(original, merged, snippet)


# ---------------------------------------------------------------------------
# Issue #3 — faithful replacement must be accepted
# ---------------------------------------------------------------------------


def test_issue_3_faithful_one_line_return_replacement_accepted():
    """Issue #3 reproduction: a faithful one-line return replacement is
    accepted by the validator (score >= 0.85). Previously scored 0.0."""
    original = (
        "def foo():\n"
        "    x = 1\n"
        "    return x\n"
    )
    snippet = (
        "def foo():\n"
        "    x = 1\n"
        "    return x + 1\n"
    )
    # Faithful merge == the snippet for this transform-only edit.
    merged = snippet

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Faithful one-line replacement scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Unrelated deletions — must be rejected
# ---------------------------------------------------------------------------


def test_faithful_edit_plus_unrelated_deletion_rejected():
    """A faithful replace that *also* drops an unrelated original line is
    rejected.  The unrelated ``C`` line is not declared as a replacement
    by the snippet, so removing it is an unauthorized deletion."""
    original = "A\nB\nC\n"
    snippet = "A\nX\nC\n"   # intent: replace B with X, preserve A and C
    merged = "A\nX\n"        # dropped C — unauthorized deletion

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Unrelated deletion scored {score:.3f}, expected < {PASS_THRESHOLD}"
    )


def test_one_to_many_with_bystander_deletion_rejected():
    """A one-to-many intended replacement that *also* deletes an
    unrelated bystander is rejected.  The snippet expands ``B`` into
    ``X1, X2``; dropping ``C`` on top of that is a bystander delete."""
    original = "A\nB\nC\n"
    snippet = "A\nX1\nX2\nC\n"
    merged = "A\nX1\nX2\n"  # dropped C — bystander deletion

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Bystander deletion scored {score:.3f}, expected < {PASS_THRESHOLD}"
    )


def test_additive_edit_with_unrelated_deletion_rejected():
    """An additive edit that drops an unrelated original line is
    rejected.  Snippet inserts ``X`` after ``A`` and appends ``Y``; the
    merge loses ``B`` even though the snippet never declared ``B`` as
    a replacement."""
    original = "A\nB\nC\n"
    snippet = "A\nX\nB\nC\nY\n"
    merged = "A\nX\nC\nY\n"  # dropped B — unauthorized deletion

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Additive edit dropped unrelated line; scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Multiplicity — duplicates are rejected when not declared
# ---------------------------------------------------------------------------


def test_duplicate_supported_line_rejected_when_not_intended():
    """A supported line duplicated beyond its declared multiplicity is
    rejected.  The snippet only declares ``A, X, B, C``; emitting
    ``A, X, B, B, C`` adds a second ``B`` that the snippet never
    declared."""
    original = "A\nB\nC\n"
    snippet = "A\nX\nB\nC\n"
    merged = "A\nX\nB\nB\nC\n"

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Unsupported duplicate scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_repeated_identical_original_lines_handled_by_multiplicity():
    """Repeated identical original lines are processed by multiplicity,
    not collapsed into a set.  Two scenarios in one test:

    (a) Snippet replaces the second ``A`` with ``X``; the merge keeps
        the first ``A`` and emits ``X`` in the slot of the second.  The
        expected multiset is ``{A, X, B}``, and the merge matches it
        exactly.

    (b) Snippet replaces *both* ``A``s with a single ``X``; the merge
        emits ``X, B``.  The expected multiset is ``{X, B}`` (one X, not
        two), and the merge matches.

    These two together prove the validator is multiplicity-aware and
    does not collapse duplicates into a set."""
    original = "A\nA\nB\n"

    # (a) replace-second-A
    snippet_a = "A\nX\nB\n"
    merged_a = "A\nX\nB\n"
    score_a = _score(original, merged_a, snippet_a)
    assert score_a >= PASS_THRESHOLD, (
        f"Replace-second-A scored {score_a:.3f}, expected >= {PASS_THRESHOLD}"
    )

    # (b) replace-both-As with a single X
    snippet_b = "X\nB\n"
    merged_b = "X\nB\n"
    score_b = _score(original, merged_b, snippet_b)
    assert score_b >= PASS_THRESHOLD, (
        f"Replace-both-As scored {score_b:.3f}, expected >= {PASS_THRESHOLD}"
    )

    # And a sanity check: dropping one of the two As without declaring
    # the replacement is still a hallucination.
    merged_c = "A\nB\n"          # drops one A without declaring X
    score_c = _score(original, merged_c, snippet_b)
    assert score_c < PASS_THRESHOLD, (
        f"Undeclared multiplicity loss scored {score_c:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Order — preserved lines must keep their relative order
# ---------------------------------------------------------------------------


def test_reordered_preserved_lines_rejected():
    """Preserved lines appearing in a different relative order is
    rejected.  The snippet only appends ``X``; the merge swaps ``A``
    and ``B`` even though every line is present with the correct
    multiplicity."""
    original = "A\nB\nC\n"
    snippet = "A\nB\nC\nX\n"  # intent: append X
    merged = "B\nA\nC\nX\n"   # swapped A and B

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Reordered preserved lines scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Missing required snippet code
# ---------------------------------------------------------------------------


def test_missing_required_snippet_code_rejected():
    """A merge that drops a required snippet line is rejected.  The
    snippet declares ``X`` as new content; emitting ``A, B, C`` is a
    silent omission of the edit."""
    original = "A\nB\nC\n"
    snippet = "A\nX\nB\nC\n"
    merged = "A\nB\nC\n"  # dropped X

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Missing required snippet code scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Invented output
# ---------------------------------------------------------------------------


def test_invented_output_rejected():
    """A merge that introduces lines from neither the original nor the
    snippet is rejected.  ``Z`` appears nowhere in the snippet; emitting
    it is fabrication."""
    original = "A\nB\nC\n"
    snippet = "A\nX\nB\nC\n"
    merged = "A\nX\nZ\nB\nC\n"  # Z invented

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Invented line scored {score:.3f}, expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Marker leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker_line",
    [
        "# ... existing code ...",
        "// ... existing code ...",
    ],
)
def test_marker_leak_rejected(marker_line: str):
    """If the model echoes a snippet placeholder into the merge output
    instead of replacing it with real code, the merge is rejected.
    Both the hash-form and the slash-form canonical markers are covered
    so the validator cannot hide a leak behind a comment-style choice."""
    original = "A\nB\nC\n"
    snippet = (
        f"A\n{marker_line}\nX\n{marker_line}\nB\nC\n"
    )
    # Merge forgot to substitute the second marker — it leaked through.
    merged = f"A\n{marker_line}\nX\nB\nC\n"

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Marker leak ({marker_line!r}) scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Sanity: ordinary no-op / unchanged merges stay accepted
# ---------------------------------------------------------------------------


def test_identical_original_and_merged_accepted():
    """An edit where the merged output is byte-for-byte identical to the
    original (model passed through unchanged) is accepted.  The snippet
    is also identical to both.  This is the most boring possible merge
    and must not be rejected as a hallucination."""
    body = "A\nB\nC\n"
    score = _score(body, body, body)
    assert score >= PASS_THRESHOLD, (
        f"Identity merge scored {score:.3f}, expected >= {PASS_THRESHOLD}"
    )


def test_pure_addition_with_marker_accepted():
    """A pure-addition merge that uses a marker to anchor existing code
    is accepted when the new content lands and the existing context
    survives with the correct order and multiplicity."""
    original = "A\nB\nC\n"
    snippet = (
        "A\n# ... existing code ...\nX\n# ... existing code ...\nB\nC\n"
    )
    merged = "A\nX\nB\nC\n"

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Pure-addition merge scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Fix-round-1 counterexamples — marker semantics and new-line order
# ---------------------------------------------------------------------------
#
# The original validator mis-treated every unmentioned original line as a
# replaceable slot and conflated preserved-line order with snippet-new-line
# order. The cases below pin the corrected behaviour:
#
#   * a marker at the END of a hunk is a preservation wildcard for the
#     original lines that follow the last context anchor — they MUST
#     appear in the merge in order, with multiplicity. Lines that the
#     snippet did NOT declare as a replacement MUST NOT be deleted;
#   * a marker at the START of a hunk is a preservation wildcard for
#     the original lines that come before the first context anchor;
#   * when the snippet only anchors the signature (line 0) and uses a
#     single marker, new lines are insertions, not replacements;
#   * new lines declared by the snippet must appear in the merge in
#     snippet order (the order check is on the snippet's new lines,
#     not on the original's preserved lines).
#
# These four properties together close the gap that the previous
# validator left open.


def test_faithful_marker_preserved_bystander_accepted():
    """Round-1 counterexample 1: a faithful merge where the trailing
    marker preserves the line AFTER the replaced original must be
    accepted. The snippet declares the replacement of ``old()`` with
    ``new()`` and uses a marker to preserve everything from that point
    onward, so ``must_survive()`` must appear in the merge."""
    original = (
        "def f():\n"
        "    context()\n"
        "    old()\n"
        "    must_survive()\n"
    )
    snippet = (
        "def f():\n"
        "    context()\n"
        "    new()\n"
        "# ... existing code ...\n"
    )
    merged = (
        "def f():\n"
        "    context()\n"
        "    new()\n"
        "    must_survive()\n"
    )

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Faithful marker-preserved bystander scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


def test_corrupt_marker_merge_drops_bystander_rejected():
    """Round-1 counterexample 2: a corrupt merge that drops the marker-
    preserved bystander must be rejected. The same snippet, but the
    merge forgets ``must_survive()``. The marker promised to preserve
    it, so the merge is hallucinated."""
    original = (
        "def f():\n"
        "    context()\n"
        "    old()\n"
        "    must_survive()\n"
    )
    snippet = (
        "def f():\n"
        "    context()\n"
        "    new()\n"
        "# ... existing code ...\n"
    )
    merged = (
        "def f():\n"
        "    context()\n"
        "    new()\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Bystander-deletion merge scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_faithful_top_insertion_preserves_marker_omitted_body_accepted():
    """Round-1 counterexample 3: a faithful top-insertion pattern where
    the marker preserves the entire original body must be accepted.
    The snippet only anchors the signature (``def f():``) and inserts
    ``added()`` at the top of the body, leaving ``first()`` and
    ``second()`` to be carried through verbatim."""
    original = (
        "def f():\n"
        "    first()\n"
        "    second()\n"
    )
    snippet = (
        "def f():\n"
        "    added()\n"
        "# ... existing code ...\n"
    )
    merged = (
        "def f():\n"
        "    added()\n"
        "    first()\n"
        "    second()\n"
    )

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Faithful top-insertion merge scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


def test_corrupt_new_line_reorder_rejected():
    """Round-1 counterexample 4: reordering the snippet's newly declared
    lines is rejected even when the multiset of merged lines is
    correct. ``X, Y`` in the snippet must appear in that order in the
    merged output; emitting ``Y`` before ``X`` violates the snippet's
    declared ordering."""
    original = "A\nB\n"
    snippet = "A\nX\nY\nB\n"
    merged = "A\nY\nX\nB\n"

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"New-line reorder scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_marker_preserves_multiple_omitted_lines_around_local_replacement():
    """A marker that brackets a multi-line original region and a local
    replacement both work in the same merge. The original lines outside
    the local replacement zone MUST be preserved verbatim, and the
    replacement MUST cover exactly the lines the snippet declares."""
    original = (
        "head1()\n"
        "head2()\n"
        "target()\n"
        "tail1()\n"
        "tail2()\n"
    )
    snippet = (
        "head1()\n"
        "head2()\n"
        "replacement()\n"
        "# ... existing code ...\n"
    )
    merged = (
        "head1()\n"
        "head2()\n"
        "replacement()\n"
        "tail1()\n"
        "tail2()\n"
    )

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Multi-line marker preservation scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


def test_ambiguous_replacement_key_fails_closed():
    """When two originals share the same replacement key (e.g. the
    same assignment LHS) and the snippet declares a single replacement
    with that key, the validator cannot decide which original was
    intended and MUST fail closed rather than guess. The expected
    behaviour is REJECT, because the validator's contract is to
    refuse ambiguous pairings — see the design-corrective
    ``required_design_corrections`` list."""
    original = (
        "self.x = 1\n"
        "self.x = 2\n"
        "self.y = 3\n"
    )
    # Two originals with the same LHS ``self.x``; the snippet declares
    # only ONE replacement for ``self.x``. Pairing is ambiguous.
    snippet = (
        "self.x = 99\n"
        "self.y = 3\n"
    )
    # Even an "obvious" pairing (replace the first self.x) is rejected
    # because the validator cannot prove that intent.
    merged = (
        "self.x = 99\n"
        "self.x = 2\n"
        "self.y = 3\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Ambiguous replacement-key merge scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD} (fail closed)"
    )


def test_new_line_reorder_within_preserved_block_rejected():
    """Reordering snippet-new lines that are NOT the only content of
    the snippet also rejects. Same shape as
    test_corrupt_new_line_reorder_rejected but exercising the case
    where the originals include bystanders between the new lines."""
    original = "A\nB1\nM\nC\nB2\n"
    snippet = "A\nX1\nX2\nB2\n"
    merged = "A\nX2\nX1\nB2\n"

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Reorder-among-bystanders scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Fix-round-2 counterexamples — a snippet-new line BEFORE a preservation
# marker is a pure insertion, not a forced replacement of the following
# original. See ``attempt_2_blocker`` in the work item and the marker
# semantics in ``text_match.deterministic_edit``.
# ---------------------------------------------------------------------------


def test_faithful_new_before_marker_insertion_accepted():
    """A snippet-new line placed BEFORE a preservation marker is an
    INSERTION: the marker-covered originals that follow are preserved by
    default, not consumed by the new line. ``deterministic_edit`` returns
    exactly this faithful merge — inserting ``z = 99`` before the gap and
    keeping BOTH ``b = 2`` and ``c = 3``.

    The previous validator scored this 0.0 because it forced ``z`` to
    consume ``b``. This regression pins the corrected insertion semantics
    and MUST be accepted (score >= 0.85)."""
    original = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    snippet = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "# ... existing code ...\n"
        "    return a\n"
    )
    # Faithful merge: z inserted before the preserved gap; b and c survive.
    merged = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Faithful new-before-marker insertion scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


def test_new_before_marker_mismatched_key_deletion_rejected():
    """The same shape as the faithful insertion, but the merge DELETES an
    unrelated marker-covered original (``b = 2``) while inserting
    ``z = 99``. The assignment key ``z`` does not justify deleting the
    assignment key ``b`` (different stable identities), and positional
    proximity must not override that mismatch. MUST be rejected."""
    original = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    snippet = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "# ... existing code ...\n"
        "    return a\n"
    )
    # Corrupt: b = 2 deleted even though z's key does not justify it.
    merged = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "    c = 3\n"
        "    return a\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Mismatched-key deletion scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_new_after_marker_insertion_accepted():
    """Symmetric to the new-before-marker case: a snippet-new line placed
    AFTER a preservation marker is a bottom insertion. The marker-covered
    originals before it are preserved; the new line lands after them."""
    original = (
        "def g():\n"
        "    p = 1\n"
        "    q = 2\n"
        "    return p\n"
    )
    snippet = (
        "def g():\n"
        "    p = 1\n"
        "# ... existing code ...\n"
        "    r = 3\n"
        "    return p\n"
    )
    merged = (
        "def g():\n"
        "    p = 1\n"
        "    q = 2\n"
        "    r = 3\n"
        "    return p\n"
    )

    score = _score(original, merged, snippet)

    assert score >= PASS_THRESHOLD, (
        f"Faithful new-after-marker insertion scored {score:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Parity matrix: every non-None deterministic_edit result is accepted.
# ---------------------------------------------------------------------------
#
# ``deterministic_edit`` is the established marker-semantics reference. The
# validator runs after model fallback but must never reject an output the
# deterministic engine itself produced. Each row below exercises one marker
# shape; the test asserts (a) deterministic_edit returns a concrete merge
# and (b) the validator accepts that exact merge.


_PARITY_MATRIX = [
    # (label, original, snippet)
    (
        "new-before-marker",
        "def foo():\n    a = 1\n    b = 2\n    c = 3\n    return a\n",
        "def foo():\n    a = 1\n    z = 99\n# ... existing code ...\n    return a\n",
    ),
    (
        "new-after-marker",
        "def g():\n    p = 1\n    q = 2\n    return p\n",
        "def g():\n    p = 1\n# ... existing code ...\n    r = 3\n    return p\n",
    ),
    (
        "marker-only-preservation",
        "def h():\n    a = 1\n    b = 2\n    return a\n",
        "def h():\n    a = 1\n# ... existing code ...\n    return a\n",
    ),
    (
        "no-marker-replacement",
        "def k():\n    x = 1\n    return x\n",
        "def k():\n    x = 1\n    return x + 1\n",
    ),
    (
        "top-insertion",
        "def t():\n    first()\n    second()\n",
        "def t():\n    added()\n# ... existing code ...\n",
    ),
    (
        "bottom-insertion",
        "def b():\n    one()\n    two()\n",
        "def b():\n# ... existing code ...\n    appended()\n",
    ),
    (
        "marker-key-replacement",
        "def m(self):\n    self._data = {}\n    self.count = 0\n    return self\n",
        "def m(self):\n    self._data = OrderedDict()\n"
        "# ... existing code ...\n    return self\n",
    ),
]


@pytest.mark.parametrize(
    "label,original,snippet",
    _PARITY_MATRIX,
    ids=[row[0] for row in _PARITY_MATRIX],
)
def test_deterministic_edit_parity_accepted(
    label: str, original: str, snippet: str
):
    """Every concrete ``deterministic_edit`` output is accepted by the
    validator. This is the anti-drift guard: the validator's marker
    semantics must not diverge from the engine that produces real
    faithful merges."""
    from fastedit.inference.text_match import deterministic_edit

    produced = deterministic_edit(original, snippet)
    assert produced is not None, (
        f"[{label}] deterministic_edit returned None; matrix row must "
        f"produce a concrete merge"
    )

    score = _score(original, produced, snippet)
    assert score >= PASS_THRESHOLD, (
        f"[{label}] validator rejected a faithful deterministic_edit "
        f"output (score {score:.3f}); produced:\n{produced}"
    )


# ---------------------------------------------------------------------------
# Fix-round-3 counterexamples — marker side-order invariant.
# ---------------------------------------------------------------------------
#
# A marker-protected segment preserves the originals it brackets, but the
# snippet's NEW lines still carry a *side* relative to the marker:
#
#   * a new line declared BEFORE the (first) marker is a top-of-gap
#     insertion — it must precede EVERY surviving original in the segment;
#   * a new line declared AFTER the (last) marker is a bottom-of-gap
#     insertion — it must follow EVERY surviving original;
#   * a new line declared BETWEEN two markers is position-ambiguous and
#     imposes no ordering constraint on the survivors.
#
# ``deterministic_edit`` emits top-insertions before the preserved gap and
# bottom-insertions after it (see the parity matrix above), so a faithful
# merge always satisfies this invariant. The round-2 validator compared the
# new-vs-new order and the survivor-vs-survivor order independently but
# dropped the relative order BETWEEN an unmatched new line and the matched
# survivors, so a new line teleported to the wrong side of a preserved gap
# still scored a clean 1.0. The three ``*_wrong_*`` cases below are the
# exact blocking counterexamples; they scored 1.0 before the fix and MUST
# now be rejected.


def test_after_marker_new_moved_before_gap_rejected():
    """Blocking case 1 (after-marker moved before gap). ``r = 3`` is
    declared AFTER the marker, so it must land after the preserved gap
    (``q = 2``). This corrupt merge places ``r = 3`` BEFORE ``q = 2`` —
    the wrong side of the gap — even though nothing is deleted and the
    new/survivor multisets match. It MUST be rejected (scored 1.0 before
    the fix)."""
    original = (
        "def g():\n"
        "    p = 1\n"
        "    q = 2\n"
        "    return p\n"
    )
    snippet = (
        "def g():\n"
        "    p = 1\n"
        "# ... existing code ...\n"
        "    r = 3\n"
        "    return p\n"
    )
    merged = (
        "def g():\n"
        "    p = 1\n"
        "    r = 3\n"
        "    q = 2\n"
        "    return p\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"After-marker line moved before the gap scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_before_marker_new_moved_after_gap_rejected():
    """Blocking case 2 (before-marker moved after gap). ``z = 99`` is
    declared BEFORE the marker, so it must precede the preserved gap
    (``b = 2``, ``c = 3``). This corrupt merge places ``z = 99`` AFTER the
    survivors — the wrong side — with no deletion. It MUST be rejected
    (scored 1.0 before the fix)."""
    original = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    snippet = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "# ... existing code ...\n"
        "    return a\n"
    )
    merged = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    z = 99\n"
        "    return a\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Before-marker line moved after the gap scored {score:.3f}, "
        f"expected < {PASS_THRESHOLD}"
    )


def test_identity_free_before_marker_new_moved_after_gap_rejected():
    """Blocking case 3 (identity-free wrong side). The new line ``new()``
    is declared BEFORE the marker and carries no assignment key, so it must
    precede the preserved gap (``old()``, ``keep()``). This corrupt merge
    appends ``new()`` after both survivors — the wrong side — and deletes
    nothing. With no replacement key to reason about, only the side-order
    invariant can reject it. It MUST be rejected (scored 1.0 before the
    fix)."""
    original = (
        "def f():\n"
        "    anchor()\n"
        "    old()\n"
        "    keep()\n"
        "    end()\n"
    )
    snippet = (
        "def f():\n"
        "    anchor()\n"
        "    new()\n"
        "# ... existing code ...\n"
        "    end()\n"
    )
    merged = (
        "def f():\n"
        "    anchor()\n"
        "    old()\n"
        "    keep()\n"
        "    new()\n"
        "    end()\n"
    )

    score = _score(original, merged, snippet)

    assert score < PASS_THRESHOLD, (
        f"Identity-free before-marker line moved after the gap scored "
        f"{score:.3f}, expected < {PASS_THRESHOLD}"
    )


def test_correct_side_marker_insertions_with_bystanders_accepted():
    """Mandatory positive: a correctly placed before-marker insertion
    (new line before the whole preserved gap) and a correctly placed
    after-marker insertion (new line after the whole preserved gap) are
    BOTH accepted with every bystander preserved. This is the accept half
    of the side-order invariant — the guard must not reject faithful
    insertions."""
    # before-marker: z = 99 precedes the entire preserved gap.
    original_b = (
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    snippet_b = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "# ... existing code ...\n"
        "    return a\n"
    )
    merged_b = (
        "def foo():\n"
        "    a = 1\n"
        "    z = 99\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    score_b = _score(original_b, merged_b, snippet_b)
    assert score_b >= PASS_THRESHOLD, (
        f"Correct-side before-marker insertion scored {score_b:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )

    # after-marker: r = 3 follows the entire preserved gap.
    original_a = (
        "def g():\n"
        "    p = 1\n"
        "    q = 2\n"
        "    return p\n"
    )
    snippet_a = (
        "def g():\n"
        "    p = 1\n"
        "# ... existing code ...\n"
        "    r = 3\n"
        "    return p\n"
    )
    merged_a = (
        "def g():\n"
        "    p = 1\n"
        "    q = 2\n"
        "    r = 3\n"
        "    return p\n"
    )
    score_a = _score(original_a, merged_a, snippet_a)
    assert score_a >= PASS_THRESHOLD, (
        f"Correct-side after-marker insertion scored {score_a:.3f}, "
        f"expected >= {PASS_THRESHOLD}"
    )


def test_identity_free_local_replacement_correct_side_accepted_wrong_side_rejected():
    """Mandatory pair: an identity-free before-marker new line may replace
    a justified front original and still precede the remaining survivor
    (accepted); moving that same new line PAST its surviving bystander is
    rejected. The replacement (``new()`` for ``old()``) is justified by
    positional front-adjacency, but the side-order invariant still requires
    ``new()`` to precede the survivor ``keep()``."""
    original = (
        "def f():\n"
        "    anchor()\n"
        "    old()\n"
        "    keep()\n"
        "    end()\n"
    )
    snippet = (
        "def f():\n"
        "    anchor()\n"
        "    new()\n"
        "# ... existing code ...\n"
        "    end()\n"
    )
    # Correct: new() replaces old() (front-justified) and precedes keep().
    merged_ok = (
        "def f():\n"
        "    anchor()\n"
        "    new()\n"
        "    keep()\n"
        "    end()\n"
    )
    score_ok = _score(original, merged_ok, snippet)
    assert score_ok >= PASS_THRESHOLD, (
        f"Correct-side identity-free local replacement scored "
        f"{score_ok:.3f}, expected >= {PASS_THRESHOLD}"
    )

    # Wrong: same replacement, but new() now trails the survivor keep().
    merged_bad = (
        "def f():\n"
        "    anchor()\n"
        "    keep()\n"
        "    new()\n"
        "    end()\n"
    )
    score_bad = _score(original, merged_bad, snippet)
    assert score_bad < PASS_THRESHOLD, (
        f"Wrong-side identity-free local replacement scored "
        f"{score_bad:.3f}, expected < {PASS_THRESHOLD}"
    )


def test_repeated_identical_preserved_originals_side_order_enforced():
    """Mandatory: repeated identical preserved originals must not let an
    LCS tie hide a wrong-side new occurrence. The gap holds two identical
    ``dup()`` lines; an after-marker ``r = 3`` is faithful only when it
    follows BOTH copies. A merge that slots ``r = 3`` between the two
    ``dup()`` survivors (an LCS-ambiguous position) is on the wrong side of
    the second survivor and MUST be rejected."""
    original = (
        "def f():\n"
        "    p = 1\n"
        "    dup()\n"
        "    dup()\n"
        "    return p\n"
    )
    snippet = (
        "def f():\n"
        "    p = 1\n"
        "# ... existing code ...\n"
        "    r = 3\n"
        "    return p\n"
    )
    # Faithful: r = 3 follows both dup() copies.
    merged_ok = (
        "def f():\n"
        "    p = 1\n"
        "    dup()\n"
        "    dup()\n"
        "    r = 3\n"
        "    return p\n"
    )
    score_ok = _score(original, merged_ok, snippet)
    assert score_ok >= PASS_THRESHOLD, (
        f"Faithful after-marker insertion past repeated survivors scored "
        f"{score_ok:.3f}, expected >= {PASS_THRESHOLD}"
    )

    # Wrong: r = 3 lands BETWEEN the two dup() survivors.
    merged_bad = (
        "def f():\n"
        "    p = 1\n"
        "    dup()\n"
        "    r = 3\n"
        "    dup()\n"
        "    return p\n"
    )
    score_bad = _score(original, merged_bad, snippet)
    assert score_bad < PASS_THRESHOLD, (
        f"After-marker line wedged between repeated survivors scored "
        f"{score_bad:.3f}, expected < {PASS_THRESHOLD}"
    )


def test_between_markers_new_line_placement_unconstrained():
    """Mandatory boundary regression: a new line declared BETWEEN two
    markers is position-ambiguous by design. The validator retains the
    invention/omission/deletion checks but does NOT impose a side
    constraint, so the new line is accepted whether it lands before,
    between, or after the preserved survivors. This documents the
    intentionally-unconstrained middle case (requirement 3)."""
    original = (
        "def fn():\n"
        "    head()\n"
        "    mid1()\n"
        "    mid2()\n"
        "    tail()\n"
    )
    snippet = (
        "def fn():\n"
        "    head()\n"
        "# ... existing code ...\n"
        "    inserted()\n"
        "# ... existing code ...\n"
        "    tail()\n"
    )
    placements = [
        # inserted() before both survivors
        "def fn():\n    head()\n    inserted()\n    mid1()\n    mid2()\n    tail()\n",
        # inserted() between the survivors
        "def fn():\n    head()\n    mid1()\n    inserted()\n    mid2()\n    tail()\n",
        # inserted() after both survivors
        "def fn():\n    head()\n    mid1()\n    mid2()\n    inserted()\n    tail()\n",
    ]
    for merged in placements:
        score = _score(original, merged, snippet)
        assert score >= PASS_THRESHOLD, (
            f"Between-markers placement scored {score:.3f}, expected "
            f">= {PASS_THRESHOLD} (position-ambiguous, must accept):\n{merged}"
        )


# ---------------------------------------------------------------------------
# Independent generated matrix — correct-side vs wrong-side placements.
# ---------------------------------------------------------------------------
#
# Beyond the hand-picked counterexamples, exhaustively enumerate single-
# marker segments with 1..3 preserved survivors and place the sole new line
# at every slot. For a before-marker snippet exactly ONE slot (ahead of all
# survivors) is correct; for an after-marker snippet exactly ONE slot
# (behind all survivors) is correct. Every other slot is a wrong-side
# placement. This is generated independently of the fix so it cannot be
# tuned to the implementation, and it stresses duplicate survivors where an
# LCS tie could otherwise mask a wrong side.


def _framed(body_lines: list[str]) -> str:
    """Wrap body lines in a ``def fn():`` shell at 4-space indent."""
    return "def fn():\n" + "".join(f"    {ln}\n" for ln in body_lines)


def _generate_side_order_matrix() -> list[tuple[str, str, str, str, bool]]:
    """Return ``(label, original, snippet, merged, expect_accept)`` rows.

    Survivor sets of size 1..3 (including an all-identical ``dup()`` set to
    exercise LCS ties) are bracketed by ``head()``/``tail()`` anchors and a
    single marker. The lone new line ``inserted()`` is swept across every
    inter-survivor slot; only the marker-consistent extreme is faithful.
    """
    rows: list[tuple[str, str, str, str, bool]] = []
    survivor_sets = {
        1: ["keep1()"],
        2: ["keep1()", "keep2()"],
        3: ["keep1()", "keep2()", "keep3()"],
        # Repeated identical survivors — LCS-ambiguous slots.
        "dup2": ["dup()", "dup()"],
    }
    for tag, survivors in survivor_sets.items():
        n = len(survivors)
        original = _framed(["head()", *survivors, "tail()"])
        snippet_before = _framed(
            ["head()", "inserted()", "# ... existing code ...", "tail()"]
        )
        snippet_after = _framed(
            ["head()", "# ... existing code ...", "inserted()", "tail()"]
        )
        for pos in range(n + 1):
            body = ["head()", *survivors[:pos], "inserted()", *survivors[pos:], "tail()"]
            merged = _framed(body)
            rows.append(
                (f"before-{tag}-pos{pos}", original, snippet_before, merged, pos == 0)
            )
            rows.append(
                (f"after-{tag}-pos{pos}", original, snippet_after, merged, pos == n)
            )
    return rows


def test_generated_side_order_matrix_accept_reject_counts():
    """Every generated correct-side placement is accepted and every
    wrong-side placement is rejected. Reports exact counts so the matrix
    coverage is auditable (visible with ``pytest -s``)."""
    rows = _generate_side_order_matrix()
    accepted_correct = 0
    rejected_wrong = 0
    mismatches: list[str] = []
    for label, original, snippet, merged, expect_accept in rows:
        score = _score(original, merged, snippet)
        is_accept = score >= PASS_THRESHOLD
        if expect_accept and is_accept:
            accepted_correct += 1
        elif not expect_accept and not is_accept:
            rejected_wrong += 1
        else:
            verdict = "ACCEPT" if is_accept else "REJECT"
            want = "ACCEPT" if expect_accept else "REJECT"
            mismatches.append(f"{label}: got {verdict} (score {score:.3f}), want {want}")

    total = len(rows)
    correct_total = sum(1 for r in rows if r[4])
    wrong_total = total - correct_total
    print(
        f"\n[side-order matrix] rows={total} "
        f"correct-side accepted={accepted_correct}/{correct_total} "
        f"wrong-side rejected={rejected_wrong}/{wrong_total}"
    )
    assert not mismatches, (
        "Generated side-order matrix mismatches:\n" + "\n".join(mismatches)
    )
    assert accepted_correct == correct_total
    assert rejected_wrong == wrong_total