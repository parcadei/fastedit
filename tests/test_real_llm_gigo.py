"""A2 GIGO: EDIT-NOT-CORRECT against the REAL model (req. 9, llm tier).

Marked ``llm`` — deselected from the default tier; run explicitly with
``uv run pytest tests/test_real_llm_gigo.py -m llm``. There is NO fake
engine here: every test drives the trained fastedit mlx-8bit model
through :func:`llm_fixtures.run_real_edit` (the cli.py wiring with the
session engine).

What is locked down (implementation plan Step A2 + req. 9):

(a) A file with a PRE-EXISTING syntax error (missing colon on ``alpha``)
    plus an LLM-path edit of a DIFFERENT function must LAND: the relative
    parse rule accepts the merged output because the original was already
    broken and the merge carries the SAME defect (a preserved trait, not a
    regression), and ``alpha``'s broken line survives byte-exact. Under
    the pre-A2 absolute parse gate this edit was refused/warned; the
    model was implicitly pushed to "repair" code the user never asked it
    to touch.

(b) The mirror op — explicitly replacing the broken ``alpha`` with a
    correct version — must produce a parse-error-free file: defect
    removal is exactly what the command declares.

(c) A HARD edit (wrap_block over an eight-line body — every body line
    must survive re-indented) exercises the unified retry-until-valid
    loop against real model noise: any attempt that mutates untouched
    content is rejected by the battery (content faithfulness) and retried
    with a corrective note until a faithful merge arrives. Retry
    engagement is asserted ONLY where naturally observable (the
    pipeline's own attempt accounting); nothing is staged or fabricated.

Pipeline notes that shape the fixtures (measured on the real pipeline,
pre-existing behavior — not changed by Step A2):

* On a file whose parse is broken, ``get_ast_map_from_source`` returns no
  AST map, ``replace=`` cannot resolve a tight chunk, and the edit takes
  the whole-file branch — so the battery validates the FULL original vs
  the FULL merge, and the snippet must carry the context anchors the
  full-file validator needs. (a) puts the edit target LAST so the
  untouched head is one segment; (b) anchors the region around the
  broken def line.
* The battery catches real model inventions: a first live run of the
  wrap edit invented a ``def locked():`` at EOF (the model "helpfully"
  defined the referenced name) and was rejected — the corpus now defines
  ``locked`` so the op spec is complete.

Non-determinism policy (plan §0): pipeline gates asserted every run
(relative parse-valid, nothing rejected), untouched regions byte-exact
vs the original and golden byte-exactness (via an independent
line-splice oracle, never fastedit) inside BOUNDED outer attempts;
measured metrics ride in every failure message via :func:`metrics_tag`.
"""

from __future__ import annotations

import ast
import asyncio

import pytest
from llm_fixtures import first_diff_tag, metrics_tag, run_real_edit

pytestmark = pytest.mark.llm

# ── The corpus ────────────────────────────────────────────────────────────
# `alpha` carries a deliberate pre-existing syntax error (missing colon) —
# the defect the GIGO tests must preserve (a)/(c) or replace on command
# (b). `beta` is the edit target, deliberately LAST so the untouched head
# forms one whole-file validator segment. Body lines are all distinct:
# repeated identical lines are a known model trap (a live run duplicated
# them and was correctly rejected as an invention).
SOURCE = '''\
#!/usr/bin/env python3
"""Warehouse report generator."""

import contextlib
import json


def alpha()
    total = 1
    return total


def locked():
    return contextlib.nullcontext()


def summarize(ledger):
    total = sum(ledger.entries.values())
    return f"total={total}"


def beta(items):
    lines = []
    for item in items:
        lines.append(str(item))
    lines.append("start")
    lines.append("middle")
    lines.append("finish")
    lines.append(str(len(items)))
    return " ".join(lines)
'''

BROKEN_ALPHA_LINE = "def alpha()\n"

# (a)/(c): marker-bearing wrap_block on `beta` — the model must wrap the
# existing body inside `with locked():` and re-indent it. This shape is
# genuinely deterministic-declined (see tests/test_add_guard_marker_position.py
# and the sanity test), so the REAL model path is forced.
BETA_WRAP_SNIPPET = """\
def beta(items):
    with locked():
        # ... existing code ...
"""

# (b): explicit replacement of the BROKEN alpha with a correct version.
# The `import json` anchor bounds the segment whose back line (the broken
# def) the op deletes; the marker between the anchors lets the validator
# justify that deletion positionally, and `def alpha():` is the declared
# replacement. alpha's FULL body is declared after the def — a live run
# with a body-only declaration made the model drop `return total` (an
# unjustified deletion, correctly rejected).
ALPHA_FIX_SNIPPET = """\
import json
# ... existing code ...
def alpha():
    total = 1
    return total
# ... existing code ...
"""

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for byte/golden assertions (plan §0 policy (b))."""


def _expected_wrap(source: str) -> str:
    """Independent golden oracle for (a)/(c): explicit line-splice arithmetic.

    NEVER calls fastedit: the expected file is the original with beta's
    body wrapped inside `with locked():` and uniformly re-indented —
    computed here by hand so the assertion cannot inherit a pipeline bug.
    """
    lines = source.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def beta"))
    body = lines[start + 1:]
    return "".join(
        lines[:start + 1] + ["    with locked():\n"] + ["    " + ln for ln in body],
    )


def _expected_alpha_fix(source: str) -> str:
    """Independent golden oracle for (b): one-line splice fixing the colon."""
    return source.replace("def alpha()\n", "def alpha():\n")


def _head_of(source: str, symbol: str) -> str:
    """Everything BEFORE the edit target's definition."""
    head, _sep, _tail = source.partition(f"def {symbol}")
    return head


def _run_until_converged(check, real_engine, tmp_path, snippet, replace):
    """Run the real edit with bounded outer attempts (plan §0 policy (b)).

    Each outer attempt is a COMPLETE real edit. ``check`` asserts the
    non-determinism policy (untouched regions byte-exact, op-spec
    conformance, validity). Non-convergence after the bound is a product
    defect per the plan, so the last failure is re-raised with metrics.
    """
    last_error: AssertionError | None = None
    last_metrics = ""
    run = None
    # The file is written once per test so the run's file_path exists on
    # disk exactly like cli.py's caller would have it (chunked_merge itself
    # never writes; the disk-untouched assertion in (a) relies on this).
    (tmp_path / "report.py").write_text(SOURCE, encoding="utf-8")
    for attempt in range(1, OUTER_ATTEMPTS + 1):
        run = run_real_edit(
            SOURCE,
            snippet,
            file_path=str(tmp_path / "report.py"),
            language="python",
            engine=real_engine,
            replace=replace,
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


# ── (a) unrelated LLM-path edit beside a pre-existing syntax error ───────


def test_unrelated_edit_lands_and_preserves_pre_existing_syntax_error(
    real_engine, tmp_path,
):
    """req. 9 headline: edit `beta`, keep `alpha` broken byte-exact."""
    head = _head_of(SOURCE, "beta")
    golden = _expected_wrap(SOURCE)

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        # The REAL model ran (wrap_block shape declines deterministic paths).
        assert run.merge_results, f"model path never invoked: {m}"
        assert any(r.tokens_generated > 0 for r in run.merge_results), (
            f"zero tokens — no real inference ran: {m}"
        )

        # Pipeline gates: the RELATIVE rule accepts the merge (original was
        # broken; the merge carries the SAME defect, nothing new), and the
        # unified loop converged without rejecting the chunk.
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )

        # The pre-existing defect survives BYTE-EXACT — fastedit is an
        # editor, not a correcter.
        assert BROKEN_ALPHA_LINE in merged, (
            f"pre-existing syntax error was 'fixed' (trait mutation): {m}\n{merged}"
        )
        assert "def alpha():" not in merged, (
            f"the model 'repaired' the untouched colon: {m}\n{merged}"
        )

        # Op spec: beta's body is wrapped inside the with-block, every
        # original body line preserved exactly once at the wrapped indent.
        assert "    with locked():" in merged, (
            f"wrap block missing from the merge: {m}\n{merged}"
        )
        for body_line in (
            "lines = []",
            "for item in items:",
            "lines.append(str(item))",
            'lines.append("start")',
            'lines.append("middle")',
            'lines.append("finish")',
            "lines.append(str(len(items)))",
            'return " ".join(lines)',
        ):
            indented = "        " + body_line
            assert merged.count(indented) == 1, (
                f"body line {body_line!r} not preserved exactly once under "
                f"the with-block: {m}\n{merged}"
            )

        # Untouched regions byte-identical (policy (b)).
        assert merged.startswith(head), (
            f"untouched file head changed: "
            f"{first_diff_tag(head, merged[: len(head)])} | {m}\n{merged}"
        )

        # Golden byte-exactness against the independent oracle.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )

        # Independent full-file structural check: the merged file must still
        # fail ast.parse EXACTLY at the preserved defect — the PASS condition
        # per req. 9 (a preserved pre-existing error is not a failure).
        try:
            ast.parse(merged)
        except SyntaxError as exc:
            defect_lineno = SOURCE.splitlines().index("def alpha()") + 1
            assert exc.lineno == defect_lineno, (
                f"unexpected syntax error at another position: {exc} | {m}\n{merged}"
            )
        else:
            raise AssertionError(
                "merged file parsed cleanly — the preserved defect "
                f"disappeared (trait mutation): {m}\n{merged}"
            )
        assert "# ... existing code ..." not in merged, (
            f"preservation marker leaked into the file: {m}\n{merged}"
        )

    run = _run_until_converged(check, real_engine, tmp_path, BETA_WRAP_SNIPPET, "beta")
    # Bounded convergence on the retry loop: a handful of validation
    # retries is legitimate (real model noise); exhaustion is not.
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )
    assert (tmp_path / "report.py").read_text(encoding="utf-8") == SOURCE, (
        "chunked_merge must not write; the caller (cli.py) owns the write"
    )


# ── (b) commanded replacement of the broken region ───────────────────────


def test_commanded_replacement_of_broken_region_produces_valid_file(
    real_engine, tmp_path,
):
    """req. 9 mirror: replace `alpha` on command → the defect is GONE."""
    golden = _expected_alpha_fix(SOURCE)

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )

        # The commanded fix landed: a well-formed alpha, the broken form gone.
        assert "def alpha():" in merged, (
            f"commanded replacement of the broken alpha did not land: {m}\n{merged}"
        )
        assert BROKEN_ALPHA_LINE not in merged, (
            f"the broken def line survived a command that replaced it: {m}\n{merged}"
        )
        assert merged.count("def alpha") == 1, (
            f"alpha duplicated or lost: {m}\n{merged}"
        )

        # Golden byte-exactness against the independent one-line splice.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )

        # With the defect replaced on command, the WHOLE file is now valid —
        # absolutely, independent of fastedit's own gate.
        try:
            ast.parse(merged)
        except SyntaxError as exc:
            raise AssertionError(
                f"merged file still does not parse after the commanded fix: "
                f"{exc} | {m}\n{merged}"
            )
        assert "# ... existing code ..." not in merged, (
            f"preservation marker leaked into the file: {m}\n{merged}"
        )

    run = _run_until_converged(check, real_engine, tmp_path, ALPHA_FIX_SNIPPET, "alpha")
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )


# ── (c) hard edit: the retry loop against real model noise ───────────────


def test_hard_edit_converges_and_the_attempt_accounting_reconciles(
    real_engine, tmp_path,
):
    """A wrap_block over the eight-line body must land through the loop.

    Every attempt that mutates the marker-preserved body (drops, invents,
    reorders or re-indents a line) is rejected by the battery — content
    faithfulness (req. 9's trait rules) — and retried with a corrective
    note. Retry engagement is read from the pipeline's own accounting
    (``result.retries`` vs the number of engine-level merge results);
    nothing here is staged to force a retry.
    """
    head = _head_of(SOURCE, "beta")
    golden = _expected_wrap(SOURCE)

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert run.merge_results, f"model path never invoked: {m}"
        assert any(r.tokens_generated > 0 for r in run.merge_results), (
            f"zero tokens — no real inference ran: {m}"
        )
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected after exhausting the retry budget: {m}\n{merged}"
        )

        # Op spec: every original body line survives exactly once, wrapped.
        assert "    with locked():" in merged, (
            f"wrap block missing: {m}\n{merged}"
        )
        for body_line in (
            "lines = []",
            "for item in items:",
            "lines.append(str(item))",
            'lines.append("start")',
            'lines.append("middle")',
            'lines.append("finish")',
            "lines.append(str(len(items)))",
            'return " ".join(lines)',
        ):
            indented = "        " + body_line
            assert merged.count(indented) == 1, (
                f"body line {body_line!r} not preserved exactly once under "
                f"the with-block (mutation or loss): {m}\n{merged}"
            )

        # Untouched regions byte-identical (policy (b)) — including the
        # pre-existing alpha defect, which the model must NOT touch.
        assert merged.startswith(head), (
            f"untouched file head changed: "
            f"{first_diff_tag(head, merged[: len(head)])} | {m}\n{merged}"
        )
        assert BROKEN_ALPHA_LINE in merged, (
            f"pre-existing defect mutated by the model: {m}\n{merged}"
        )
        assert "# ... existing code ..." not in merged, (
            f"preservation marker leaked into the file: {m}\n{merged}"
        )

        # Golden byte-exactness against the independent oracle.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )

    run = _run_until_converged(check, real_engine, tmp_path, BETA_WRAP_SNIPPET, "beta")
    # The unified loop's budget bound: convergence, never exhaustion.
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )
    # Attempt accounting, naturally observable from the pipeline itself:
    # every merge site consumes at least one engine call and `retries`
    # counts the attempts beyond the first per site. The whole-file path
    # (broken file → no AST map) is one site.
    assert run.result.chunks_used == 1, (
        f"expected the whole-file branch (broken original, no AST map): "
        f"{metrics_tag(run)}"
    )
    assert len(run.merge_results) == run.result.retries + run.result.chunks_used, (
        f"attempt accounting mismatch (merge calls vs retries+sites): "
        f"{metrics_tag(run)}"
    )


# ── (c-cont.) retry engagement, naturally observed: helpful invention ────

# The same corpus MINUS the `locked` definition, plus a wrap snippet that
# references it: the model "helpfully" invents `def locked():` on every
# attempt (measured 3/3 trials — 9/9 attempts — exhausting the budget),
# the battery rejects each attempt as an invention, and the pipeline
# rejects the edit fail-loud keeping the original file.
NO_LOCKED_SOURCE = SOURCE.replace(
    """def locked():
    return contextlib.nullcontext()


""",
    "",
)


def test_model_helpfulness_is_retried_then_rejected_fail_loud(
    real_engine, mcp_harness, tmp_path,
):
    """req. 9 corollary against the REAL model: an invention is never written.

    This is the retry loop ENGAGING under real model noise — ``retries``
    is naturally observable here (measured 8/8 on every trial), not
    staged: the model mutates the untouched content by INVENTING a
    `def locked():` the op never declares (and breaks the kept syntax
    beyond the preserved defect), the battery rejects every attempt, each
    retry carries the corrective note, and on budget exhaustion the edit
    is REJECTED fail-loud: the original file is kept byte-for-byte.

    THE FAIL-LOUD CONTRACT (plan §0 + req. 9) — pinned on both stacks:

    * pipeline level: ``chunks_rejected == 1`` (the do-not-persist signal
      the tool gates read) and ``merged_code == original bytes``. The
      routing here is the D2 text-anchor WINDOW path (the file's parse is
      broken, so no AST map exists and the snippet's unique ``def
      beta(items):`` line anchors one window spanning the file); that
      path's exhaustion convention keeps the assembled file structurally
      sound — the assembled file IS the untouched original — so
      ``parse_valid`` stays True and the refusal is carried by the
      rejection bookkeeping, never by a parse verdict.
    * MCP level (the write path): the tool gate reads exactly that
      bookkeeping and REFUSES — the full stack (real ``chunked_merge``,
      real ``_atomic_write``, real ``BackupStore``) writes NOTHING and the
      file on disk keeps the original bytes.

    If a future model converges here instead (a faithful no-invention
    wrap), that outcome is also correct product behavior and this test
    should be revisited with the model.
    """
    (tmp_path / "report.py").write_text(NO_LOCKED_SOURCE, encoding="utf-8")
    run = run_real_edit(
        NO_LOCKED_SOURCE,
        BETA_WRAP_SNIPPET,
        file_path=str(tmp_path / "report.py"),
        language="python",
        engine=real_engine,
        replace="beta",
    )
    result = run.result
    m = metrics_tag(run)

    # The retry loop ENGAGED: real model noise consumed real retries.
    assert result.retries > 0, (
        f"expected the validation loop to retry on the model's invented "
        f"`def locked():` — retries == 0 means the model converged: {m}"
    )
    # Fail-loud exhaustion: the invention never lands.
    assert result.chunks_rejected == 1, (
        f"expected the chunk rejection bookkeeping on exhaustion: {m}"
    )
    assert result.chunks_used == 1, f"{m}"
    assert result.parse_valid is True, (
        f"the kept file is the untouched original — structurally sound: {m}"
    )
    assert result.merged_code == NO_LOCKED_SOURCE, (
        f"the corrupted/inventing merge must never be written — the "
        f"original file is kept byte-for-byte: {m}\n{result.merged_code}"
    )
    assert "def locked()" not in result.merged_code, (
        f"the model's invented definition reached the output: {m}"
    )
    assert "# ... existing code ..." not in result.merged_code, (
        f"preservation marker leaked: {m}"
    )
    # Attempt accounting for the single merge site (one text-anchor window
    # spanning the file).
    assert len(run.merge_results) == result.retries + 1, (
        f"attempt accounting mismatch: {m}"
    )

    # ── THE MCP REFUSAL: the same op through the FULL MCP stack ──────────
    # The tool gate refuses an all-chunks-rejected merge (never
    # force-overridable), nothing is written, and the file on disk keeps
    # the original bytes — the fail-loud contract at the write path.
    from fastedit.mcp import tools_edit

    target = tmp_path / "mcp-refusal.py"
    target.write_text(NO_LOCKED_SOURCE, encoding="utf-8")
    response = asyncio.run(tools_edit.fast_edit(
        file_path=str(target),
        edit_snippet=BETA_WRAP_SNIPPET,
        replace="beta",
    ))
    assert response.startswith("Error: edit rejected"), (
        f"the MCP tool gate did not refuse the all-chunks-rejected merge: "
        f"{response!r}"
    )
    assert "File unchanged" in response, response
    assert target.read_text(encoding="utf-8") == NO_LOCKED_SOURCE, (
        "the refused merge modified the file on disk"
    )
    assert "def locked()" not in target.read_text(encoding="utf-8"), (
        "the model's invented definition reached the file on disk"
    )
