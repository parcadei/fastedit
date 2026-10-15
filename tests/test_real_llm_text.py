"""Step D1 — structureless text edits through the REAL model (req. 6, llm tier).

Marked ``llm`` — deselected from the default tier; run explicitly with
``uv run pytest tests/test_real_llm_text.py tests/test_real_llm_sanity.py
-m llm``. There is NO fake engine here: every test drives the trained
fastedit mlx-8bit model through :func:`llm_fixtures.run_real_edit` with
``language=None`` — the structureless path the D1 text-trait battery
governs (no AST, no parse gate, whole-file merge branch).

THE ≤150-LINE BOUNDARY (Step D1 scope note): with no AST, a structureless
file always takes the whole-file merge branch, and that branch refuses
files above the 150-line safety limit (``ValueError``; pinned hermetically
in tests/test_text_heuristics.py). Step D2's AST-less chunking lifts this;
until then every real-LLM text fixture here stays well under the limit
(this file's largest is 37 lines).

MEASURED MODEL BEHAVIOR (real mlx-8bit runs; the shapes and goldens below
are pinned from those measurements, per the plan's non-determinism policy):

* A whole-file APPEND whose snippet ends with a tail preservation marker
  makes this model ECHO the marker verbatim into the prose output on
  every attempt (9/9 measured) — the battery rightly rejects every one.
  The battery-sanctioned marker-free append shape (context anchor + the
  new lines) converges on the FIRST attempt, byte-exact apart from one
  extra layout blank the model adds at the seam — which the D1 layout
  band exists to absorb (measured deltas: blank_lines +1, lines +1,
  bytes +1, everything else 0).
* A whole-file mid-file SENTENCE REPLACEMENT through the marker idiom
  (anchor + marker + new sentence + anchor) converges with a small
  number of validation retries (measured 1) and a byte-exact swap — the
  one sanctioned deletion is justified by the marker-adjacent positional
  capacity the D1 removable-traits allowance mirrors.
* A marker-free mid-file INSERTION between two context anchors converges
  on the first attempt with ZERO trait drift.

TOLERANCE CALIBRATION (the probe the plan asked for): across the real
runs above the trait deltas of ACCEPTED merges were exactly {0} or the
one-seam-blank {blank_lines +1, lines +1, bytes +1}; the
``TOLERANCE_MODEL_PROSE`` preset (±2% with floor 2, layout floor 1)
absorbs precisely that and nothing content-sized. Every test here also
re-runs the battery's own predicate
(:func:`fastedit.text_heuristics.validate_text_output` with the derived
op) on the accepted merge — the D1 gate's engagement is directly
observable, not inferred.

GIGO (req. 9): the fixtures carry a pre-existing typo (``recieve-33``)
and a quoted as-is note; an append elsewhere must preserve them
byte-exact — fastedit is an editor, not a correcter.
"""

from __future__ import annotations

import pytest
from llm_fixtures import first_diff_tag, metrics_tag, run_real_edit

pytestmark = pytest.mark.llm

# ── The corpus ────────────────────────────────────────────────────────────
# 37 lines of fully DISTINCT prose (the C3 doctrine: near-identical
# template lines make the real model lossily re-generate plausible
# variants instead of copying — measured on a templated draft, whose
# every attempt was rightly rejected). Two Chinese lines exercise the
# CJK-aware traits end to end; the "recieve-33" typo is the GIGO trait
# to preserve; the quoted note is the as-is explanation.
TXT_A = """\
Warehouse operations notes — spring rotation, week one.

Monday opened with the dock-two compressor still on the service list.
Dana walked the mezzanine and found two pallets of unlabeled returns.
The labeling printer on the north wall jammed twice before lunch.
Luis patched the manifest exporter and filed ticket OPS-1182.
By evening the cycle-count variance in aisle four was under one percent.

Tuesday brought the carrier audit: three manifests carried stale rates.
Priya renegotiated the express lane surcharge with the regional desk.
The mezzanine elevator passed its inspection with one advisory note.
中文备注：夜班组长确认 Dock7 的退货区域在闭店前完成复核。
A forklift battery died mid-shift; the spare was charged by midnight.

Wednesday's storm delayed the inbound freight by nearly four hours.
The dock crew split the backlog before the second shift started.
Dana noted the "recieve-33" typo in the returns ledger predates us.
It stays: fixing it silently would rewrite history nobody asked for.
The night crew inventoried the overflow cage down to the last carton.

Thursday the new scanner firmware bricked two handhelds at boot.
Luis rolled the fleet back and opened a case with the vendor.
Stock levels for the spring promo were locked at seventeen hundred.
The export to the planning sheet finished without a single warning.
Priya booked the training room for the Friday retro at ten sharp.

Friday's retro ran long but the action list came out clean.
The rota for the holiday week needs one more volunteer for Sundays.
The broken label applicator finally shipped out for repair.
中文备注：周五复盘确认库存差异已降至千分之三以下，无需追加盘点。
The week closed with every ticket either resolved or explicitly owned.

Next week the auditors return for the annual safety walkthrough.
The dock-two compressor is scheduled for its replacement part.
Dana will draft the rotation summary for the regional review.
Luis wants the manifest exporter's retry policy documented properly.
The overflow cage is due for a permanent shelf reconfiguration.
"""

GIGO_TYPO_LINE = 'Dana noted the "recieve-33" typo in the returns ledger predates us.\n'

ADDENDUM = (
    "Weekend addendum: the rotation checklist moved to the shared drive.\n"
    "The on-call handoff now starts at noon sharp, not at one as before.\n"
)

# Marker-free append: the context anchor is the file's last line, the two
# addendum lines are the declared payload. No tail marker — the measured
# model echoes a tail marker verbatim into prose output (see docstring),
# and the validator needs none: a trailing insertion zone is exactly what
# the preserve-by-default semantics already give this shape.
APPEND_SNIPPET = (
    "The overflow cage is due for a permanent shelf reconfiguration.\n"
    "\n"
    + ADDENDUM
)

# The independent golden (line-splice arithmetic, never fastedit): the
# original untouched, the declared payload appended after ONE blank
# separator line.
def _append_golden() -> str:
    return TXT_A + "\n" + ADDENDUM


# Mid-file sentence replacement through the marker idiom: the old line is
# NOT restated (an anchor must survive), the new sentence is declared
# between the marker and the following anchor — the one shape whose
# deletion the battery justifies positionally.
REPLACE_SNIPPET = (
    "Warehouse operations notes — spring rotation, week one.\n"
    "# ... existing code ...\n"
    "Tuesday brought the carrier audit: three manifests carried "
    "outdated rates.\n"
    "Priya renegotiated the express lane surcharge with the regional desk.\n"
)
STALE_LINE = "Tuesday brought the carrier audit: three manifests carried stale rates.\n"
OUTDATED_LINE = (
    "Tuesday brought the carrier audit: three manifests carried "
    "outdated rates.\n"
)

# Mid-file marker-free insertion between two context anchors (the measured
# first-attempt, zero-drift shape).
TXT_B = (
    "Meeting notes — ops sync.\n"
    "\n"
    "The warehouse sync runs at midnight every day.\n"
    "It skips items marked discontinued.\n"
    "\n"
    "Attendance: Dana, Luis, Priya and the on-call engineer.\n"
    "The projector in room 4 stays broken until Friday.\n"
)
TXT_B_INSERT_SNIPPET = (
    "Attendance: Dana, Luis, Priya and the on-call engineer.\n"
    "Apologies were recorded for Mateo who covers the night shift.\n"
    "The projector in room 4 stays broken until Friday.\n"
)
TXT_B_GOLDEN = TXT_B.replace(
    "Attendance: Dana, Luis, Priya and the on-call engineer.\n",
    "Attendance: Dana, Luis, Priya and the on-call engineer.\n"
    "Apologies were recorded for Mateo who covers the night shift.\n",
)

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b))."""


def _run_until_converged(check, real_engine, tmp_path, name, source, snippet):
    """Run the real edit with bounded outer attempts (plan §0 policy (b)).

    Each outer attempt is a COMPLETE real edit; ``check`` asserts the
    non-determinism policy. Non-convergence after the bound is a product
    defect per the plan, so the last failure is re-raised with metrics.
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


def _assert_battery_engaged(source, snippet, run, m):
    """The D1 text-trait battery's own predicate on the accepted merge.

    The battery is internal, but its engagement is directly observable:
    the loop accepted the merge (no rejection bookkeeping) and the trait
    validator — the exact gate the loop ran — accepts the output against
    the op derived from the same snippet the model saw.
    """
    from fastedit.inference.chunked_merge import _derived_text_op
    from fastedit.text_heuristics import (
        TOLERANCE_MODEL_PROSE,
        validate_text_output,
    )

    result = run.result
    assert result.chunks_rejected == 0, (
        f"the battery rejected the merge — kept the original: {m}"
    )
    assert result.parse_valid is True, f"{m}"
    op, layout_slack, removable = _derived_text_op(source, snippet)
    ok, reason = validate_text_output(
        source, op, result.merged_code, TOLERANCE_MODEL_PROSE,
        layout_slack=layout_slack, removable_traits=removable,
    )
    assert ok is True, f"the D1 trait gate's predicate failed on the accepted merge: {reason} | {m}"


def _assert_model_ran(run, m):
    assert run.merge_results, (
        f"engine.merge_auto was never invoked — no real LLM ran: {m}"
    )
    assert any(r.tokens_generated > 0 for r in run.merge_results), (
        f"model generated zero tokens — no real inference ran: {m}"
    )


def _assert_no_marker_leak(run, m):
    assert "# ... existing code ..." not in run.result.merged_code, (
        f"preservation marker leaked into the prose file: {m}\n"
        f"{run.result.merged_code}"
    )


# ── (a) append a paragraph to a structureless .txt (whole-file path) ──────


def test_real_append_paragraph_lands_byte_exact(real_engine, tmp_path):
    """The whole-file append converges; every original byte survives.

    The measured model shape adds ONE layout blank at the seam beyond the
    declared one — the D1 layout band absorbs exactly that, and the
    golden pins the measured byte shape.
    """
    golden = _append_golden()

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_battery_engaged(TXT_A, APPEND_SNIPPET, run, m)
        _assert_no_marker_leak(run, m)
        # Retry accounting is bounded: the loop converged, never crawled.
        assert result.retries <= 3, (
            f"retry-until-valid needed an unreasonable budget: {m}"
        )
        # Untouched regions byte-identical (policy (b)) — for an append
        # the ENTIRE original is the untouched prefix.
        assert result.merged_code.startswith(TXT_A), (
            f"untouched file head changed: "
            f"{first_diff_tag(TXT_A, result.merged_code[: len(TXT_A)])} | {m}"
            f"\n{result.merged_code}"
        )
        # Golden byte-exactness against the independent splice.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}\n"
            f"{result.merged_code}"
        )
        # The GIGO traits survive: the pre-existing typo and its as-is
        # explanation are untouched bytes.
        assert GIGO_TYPO_LINE in result.merged_code, (
            f"pre-existing typo was 'fixed' (trait mutation): {m}"
        )

    _run_until_converged(
        check, real_engine, tmp_path, "append", TXT_A, APPEND_SNIPPET,
    )


# ── (b) replace a sentence in the middle (marker idiom, model path) ───────


def test_real_sentence_replacement_lands_byte_exact(real_engine, tmp_path):
    """The mid-file replacement converges through the retry loop.

    The model path is forced (no AST for ``.txt`` → whole-file merge). The
    one sanctioned deletion rides the marker-adjacent positional capacity,
    which the D1 removable-traits allowance mirrors — measured: one
    validation retry before a byte-exact swap.
    """
    golden = TXT_A.replace(STALE_LINE, OUTDATED_LINE)
    assert golden != TXT_A

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_battery_engaged(TXT_A, REPLACE_SNIPPET, run, m)
        _assert_no_marker_leak(run, m)
        assert result.retries <= 3, (
            f"retry-until-valid needed an unreasonable budget: {m}"
        )
        # Attempt accounting: every site consumes at least one engine call
        # and `retries` counts the attempts beyond the first (the
        # whole-file branch is one site).
        assert len(run.merge_results) == result.retries + 1, (
            f"attempt accounting mismatch: {m}"
        )
        # The declared replacement landed; the stale wording is gone.
        assert OUTDATED_LINE in result.merged_code, (
            f"commanded replacement did not land: {m}\n{result.merged_code}"
        )
        assert STALE_LINE not in result.merged_code, (
            f"the replaced sentence survived the command: {m}"
        )
        # Untouched regions byte-identical (policy (b)) — head and tail
        # around the swapped line.
        head, _sep, tail = TXT_A.partition(STALE_LINE)
        assert result.merged_code.startswith(head), (
            f"untouched file head changed: {m}\n{result.merged_code}"
        )
        assert result.merged_code.endswith(tail), (
            f"untouched file tail changed: {m}\n{result.merged_code}"
        )
        # Golden byte-exactness.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}\n"
            f"{result.merged_code}"
        )
        # GIGO: the typo two paragraphs away survived the replacement.
        assert GIGO_TYPO_LINE in result.merged_code, (
            f"pre-existing typo was 'fixed' (trait mutation): {m}"
        )
        assert "中文备注" in result.merged_code, (
            f"CJK lines lost by the merge: {m}"
        )

    _run_until_converged(
        check, real_engine, tmp_path, "replace", TXT_A, REPLACE_SNIPPET,
    )


# ── (b2) marker-free mid-file insertion (measured zero-drift shape) ───────


def test_real_marker_free_insertion_is_byte_exact_first_attempt(
    real_engine, tmp_path,
):
    """A context-flanked insertion converges with zero trait drift.

    Measured: the model inserts exactly the declared line between the two
    anchors on the first attempt — the trait deltas are all zero and the
    golden is byte-exact without any outer slack.
    """

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_battery_engaged(TXT_B, TXT_B_INSERT_SNIPPET, run, m)
        _assert_no_marker_leak(run, m)
        assert result.retries == 0, (
            f"the measured zero-drift shape consumed retries: {m}"
        )
        assert result.merged_code == TXT_B_GOLDEN, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(TXT_B_GOLDEN, result.merged_code)} | {m}\n"
            f"{result.merged_code}"
        )

    _run_until_converged(
        check, real_engine, tmp_path, "insert", TXT_B, TXT_B_INSERT_SNIPPET,
    )


# ── (c) GIGO: a pre-existing typo elsewhere survives byte-exact ───────────


def test_real_gigo_typo_elsewhere_survives_byte_exact(real_engine, tmp_path):
    """req. 9 through the structureless path: the edit lands, the typo stays.

    The snippet anchors on the file's LAST line and declares the payload
    after it (the D1-measured converging append shape). MEASURED SHAPE
    NOTE (deterministic greedy probes, Step D2): the same edit declared
    from the SECOND-to-last line — anchor, blank, payload, payload meant
    to land at EOF past the undeclared last line — made the model drop
    that last line on every attempt once D2's window path took over the
    edit (the window ends at the anchor's file tail; the model reads the
    payload as the new tail and eats the line after the anchor), so the
    tail-anchored shape is the one this file pins.

    The untouched tail — including the ``recieve-33`` typo paragraph —
    must still come back byte-exact. The trait oracle expects exactly
    this: the typo's traits are part of the input vector, and the op
    never touches them.
    """
    snippet = (
        "The overflow cage is due for a permanent shelf reconfiguration.\n"
        "\n"
        "Weekend addendum: the rotation checklist moved to the shared "
        "drive.\n"
    )
    # Measured shape: the model appends the declared line after the file's
    # last line and adds ONE layout blank at the seam — the D1 layout band
    # absorbs exactly that, and the golden pins the measured byte shape
    # (the same convention test (a) pins).
    golden = (
        TXT_A
        + "\n"
        + "Weekend addendum: the rotation checklist moved to the shared "
        "drive.\n"
    )

    def check(run):
        result = run.result
        m = metrics_tag(run)
        _assert_model_ran(run, m)
        _assert_battery_engaged(TXT_A, snippet, run, m)
        _assert_no_marker_leak(run, m)
        # The typo and its as-is note survive BYTE-EXACT (the declared
        # insertion is elsewhere).
        assert GIGO_TYPO_LINE in result.merged_code, (
            f"pre-existing typo mutated by the model: {m}\n"
            f"{result.merged_code}"
        )
        assert (
            "It stays: fixing it silently would rewrite history nobody "
            "asked for.\n"
        ) in result.merged_code, (
            f"the as-is explanation was dropped: {m}"
        )
        # Golden byte-exactness.
        assert result.merged_code == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, result.merged_code)} | {m}\n"
            f"{result.merged_code}"
        )

    _run_until_converged(
        check, real_engine, tmp_path, "gigo", TXT_A, snippet,
    )
