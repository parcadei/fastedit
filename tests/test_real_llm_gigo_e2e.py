"""Step D5 — the GIGO e2e suite: EDIT-NOT-CORRECT end-to-end (req. 9, real LLM).

Marked ``llm`` — deselected from the default tier; run explicitly with
``uv run pytest tests/test_real_llm_gigo_e2e.py -m llm``. There is NO fake
engine on any LLM path here: every commanded edit drives the trained
fastedit mlx-8bit model through :func:`llm_fixtures.run_real_edit` (the
cli.py wiring) and — wherever writing is involved — through the FULL MCP
stack (``mcp_harness``: real ``chunked_merge``, real ``_atomic_write``, real
``BackupStore``, real undo ledger), exactly like the C2/D4 suites.

MISSION (req. 9 end-to-end): prove through the full pipeline that fastedit
is an EDITOR, not a corrector. Input defects are PRESERVED unless the
command targets them; commanded fixes are REQUIRED by the validator and
retried until they land. Each case pairs a committed small fixture with an
independently committed expected file (``tests/golden/gigo/``); every test
re-derives the expected bytes with its own line-splice arithmetic — never
fastedit — and asserts the committed golden matches that oracle before
using it (the B3 oracle-agreement discipline).

Cases (the matrix of plan Step D5):

a. **py** — ``tests/golden/gigo/py_defects/``: function ``alpha`` carries a
   syntax error (missing colon), the misspelled identifier ``reuslt`` and
   two style crimes (a tab-indented block beside space-indented code, a
   trailing-whitespace line and a spaces-only line). The command wraps
   UNRELATED ``beta`` (LLM-path marker snippet — the deterministic editor
   declines the shape). ALL defects survive BYTE-EXACT (the committed
   golden is the proof — no "helpful fixes" anywhere), the edit lands, and
   the merged file still fails ``ast.parse`` exactly at the preserved
   colon. Real-model retries were MEASURED (1 validation retry on the
   converging run) and bounded. Then the FULL MCP stack: the edit writes
   the golden bytes and one ``fast_undo`` restores the original exactly.
b. **md** — ``tests/golden/gigo/md_frontmatter/``: frontmatter malformed
   TWICE (``title: [unclosed list``, no closing ``---``) + a body-paragraph
   edit → the malformed frontmatter is preserved byte-exact (req. 9's
   exact PASS example) and the D3 md structure battery does NOT reject the
   merge (its own predicate asserted green).
c. **md** — the SAME malformed fixture + the explicit command "replace the
   frontmatter with a well-formed version" (the snippet DECLARES the new
   block) → the output frontmatter is well-formed and the rest of the file
   is byte-exact; the validator REFUSES anything less and the retry loop
   repeats until correct — non-convergence after the bounded outer
   attempts is a product defect. MEASURED (plan §0 evidence): the real
   model applies the declared frontmatter on the first attempt of the
   converging shape (``retries == 0`` on every measured trial), so the
   refuse-and-repeat direction is pinned hermetically on this EXACT
   fixture in ``tests/test_gigo_hermetic.py`` (scripted merge_fn — the A2
   hermetic-twin pattern; the real model's own retry engagement is
   naturally observed in case (a) and pinned at the default tier by
   ``test_relative_validation.py``). The MCP stack write + undo are
   asserted byte-exact.
d. **md** — ``tests/golden/gigo/md_fence/``: the D4 corpus shape (unclosed
   backtick fence in an unrelated appendix) + a body edit. Measured model
   outcomes (D4 doctrine) are covered by ONE policy: refused → the tool
   gate keeps the original bytes; applied → the merge is the committed
   golden. EITHER WAY the unclosed-fence trait survives byte-exact on
   disk (a "helpful" fence repair never reaches the file).
e. **json** — ``tests/golden/gigo/json_trailing_comma/``: invalid JSON
   (trailing comma) + an edit of a DIFFERENT key. MEASURED routing: the
   json grammar resolves and tree-sitter's error recovery still yields an
   AST map on the broken input, so chunk location runs on that
   error-recovered map (one whole-file region) — NOT the D2 text-window
   path. The RELATIVE parse gate accepts the merge because the
   trailing-comma error trait is INHERITED (kind + containing line
   unchanged); the defect survives byte-exact and the file is still
   invalid JSON afterwards (a preserved defect, never a fix).
f. **typo preservation under replacement** —
   ``tests/golden/gigo/py_typo/``: the snippet replaces ``beta`` whose body
   CONTAINS the typo'd identifier. The typo inside the REPLACED region
   disappears (commanded — a complete re-definition, so the zero-token
   deterministic direct-swap path runs, ``0 tokens`` asserted) while the
   SAME typo in ``alpha`` survives byte-exact. Backward reconstruction and
   MCP undo regenerate the original.
g. **html** — the D3 malformed-attribute row (``<div lang>``, valueless) +
   an unrelated insertion → the valueless attribute survives byte-exact
   (the committed golden), with the D3 attribute gate's predicate green.

h. **negative control** — a model output that silently fixes an UNTOUCHED
   typo must be REJECTED by the battery. The real-model side is case (a)
   itself (the model preserved the typo live, byte-exact); the rejection
   side cannot be staged against the real model without fabricating
   behavior, so it is pinned in-pipeline with the scripted-merge_fn
   hermetic twin on THIS suite's corpus:
   ``tests/test_gigo_hermetic.py::test_helpful_typo_fix_in_untouched_region_is_rejected_in_pipeline``
   (the default-tier pin the task references:
   ``test_relative_validation.py::test_helpful_fix_of_untouched_defect_is_retried_then_rejected``).

Non-determinism policy (plan §0): pipeline gates asserted every run
(relative parse-valid, nothing rejected, op-spec conformance, untouched
regions byte-exact) and golden byte-exactness via the independent
line-splice oracle inside BOUNDED outer attempts (``OUTER_ATTEMPTS`` —
each attempt is a COMPLETE real edit; non-convergence is a product
defect, reported with ``first_diff_tag`` and metrics). Measured metrics
ride in every failure message via :func:`metrics_tag`.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest
from llm_fixtures import first_diff_tag, metrics_tag, run_real_edit

pytestmark = pytest.mark.llm

GIGO_DIR = Path(__file__).resolve().parent / "golden" / "gigo"

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b)).

Each outer attempt is a COMPLETE real edit (fresh model calls, fresh
validation, fresh file); non-convergence after the bound is a product
defect per plan §0, never absorbed as flakiness."""


def _golden(name: str) -> str:
    """A committed GIGO golden file (original or expected), strict UTF-8."""
    return (GIGO_DIR / name).read_text(encoding="utf-8")


def _splice(source: str, *pairs: tuple[str, str]) -> str:
    """The independent line-splice oracle: swap each pair's exact text."""
    for old, new in pairs:
        assert source.count(old) == 1, f"oracle splice needs a unique target: {old!r}"
        source = source.replace(old, new)
    return source


def _run_until_converged(check, real_engine, tmp_path, source, snippet, name,
                         language, **kwargs):
    """Run the real edit with bounded outer attempts (plan §0 policy (b)).

    Each outer attempt is a COMPLETE real edit against a FRESH file.
    ``check(run, target)`` asserts the non-determinism policy. After a
    converging attempt the helper also asserts the pipeline never wrote
    (chunked_merge never persists; the caller owns the write).
    Non-convergence after the bound is a product defect per the plan, so
    the last failure is re-raised with metrics.
    """
    last_error: AssertionError | None = None
    last_metrics = ""
    run = None
    for attempt in range(1, OUTER_ATTEMPTS + 1):
        target = tmp_path / f"{name}-{attempt}"
        target.write_text(source, encoding="utf-8")
        run = run_real_edit(
            source, snippet, file_path=str(target), language=language,
            engine=real_engine, **kwargs,
        )
        try:
            check(run, target)
            assert target.read_text(encoding="utf-8") == source, (
                "chunked_merge must not write; the caller (cli.py/MCP) owns "
                "the write — the pipeline modified the file on disk"
            )
            print(
                f"[D5 gigo] {name}: converged on outer attempt {attempt}, "
                f"retries={run.result.retries}, "
                f"model_tokens={run.result.model_tokens}",
            )
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


def _mcp_edit(path: Path, snippet: str, **kwargs) -> str:
    """One ``fast_edit`` through the FULL MCP stack (harness-installed)."""
    from fastedit.mcp import tools_edit

    return asyncio.run(tools_edit.fast_edit(
        file_path=str(path), edit_snippet=snippet, **kwargs,
    ))


def _mcp_undo(path: Path) -> str:
    from fastedit.mcp import tools_ast

    return asyncio.run(tools_ast.fast_undo(file_path=str(path)))


# ---------------------------------------------------------------------------
# Case (a) — py: syntax error + typo + style crimes preserved byte-exact
# ---------------------------------------------------------------------------

PY_DEFECTS = _golden("py_defects/original.py")
PY_WRAP_SNIPPET = (
    "def beta(items):\n"
    "    with audit_lock():\n"
    "        # ... existing code ...\n"
)
PY_BROKEN_DEF_LINE = "def alpha()\n"


def _py_wrap_oracle(source: str) -> str:
    """Independent golden oracle for (a): explicit line-splice arithmetic.

    NEVER calls fastedit: the expected file is the original with beta's
    body wrapped inside ``with audit_lock():`` and uniformly re-indented —
    computed here by hand so the assertion cannot inherit a pipeline bug.
    """
    lines = source.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def beta"))
    body = lines[start + 1:]
    return "".join(
        lines[:start + 1] + ["    with audit_lock():\n"]
        + ["    " + ln for ln in body],
    )


def _py_unwrap_oracle(wrapped: str) -> str:
    """The inverse op of :func:`_py_wrap_oracle` (backward reconstruction)."""
    lines = wrapped.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def beta"))
    guard = lines[start + 1]
    assert guard == "    with audit_lock():\n"
    body = [ln[4:] for ln in lines[start + 2:]]
    return "".join(lines[:start + 1] + body)


def test_a_py_defects_preserved_byte_exact(real_engine, mcp_harness, tmp_path):
    """req. 9 headline on the REAL model: edit ``beta``, keep every defect.

    The command wraps unrelated ``beta``; the syntax error, the typo and
    the style crimes in untouched ``alpha`` survive BYTE-EXACT (the
    committed golden is the proof — no "helpful fixes" anywhere), the edit
    lands, and retries stay within tolerance. Measured: one validation
    retry on the converging run (real model noise, bounded).
    """
    golden = _golden("py_defects/expected_wrap_beta.py")
    # Oracle agreement: the committed golden IS the independent splice.
    assert golden == _py_wrap_oracle(PY_DEFECTS)

    def check(run, _target):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        # The REAL model ran (the wrap shape declines deterministic paths).
        assert run.merge_results, (
            f"model path never invoked: {m}"
        )
        assert any(r.tokens_generated > 0 for r in run.merge_results), (
            f"zero tokens — no real inference ran: {m}"
        )

        # Pipeline gates: the RELATIVE parse rule accepts the merge (the
        # original was already broken; the merge carries the SAME defect),
        # and the retry loop converged without rejecting the chunk.
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )

        # Defect 1 — the syntax error survives byte-exact.
        assert PY_BROKEN_DEF_LINE in merged, (
            f"pre-existing syntax error was 'fixed' (trait mutation): "
            f"{m}\n{merged}"
        )
        assert "def alpha():\n" not in merged, (
            f"the model 'repaired' the untouched colon: {m}\n{merged}"
        )
        # Defect 2 — the typo survives byte-exact (tab-indented, as shipped).
        assert "\treuslt = 0\n" in merged and "\t\treuslt += 1\n" in merged, (
            f"the untouched typo'd identifier was 'corrected': {m}\n{merged}"
        )
        # Defect 3 — the style crimes survive byte-exact: a trailing-
        # whitespace content line, a spaces-only line, mixed tab/space
        # indentation (all covered by the golden; named here so a failure
        # points at the defect class).
        assert "\treturn reuslt  \n" in merged, (
            f"trailing whitespace on an untouched line was stripped: {m}"
        )
        assert "\n   \n" in merged, (
            f"the spaces-only line was normalized away: {m}\n{merged}"
        )

        # Golden byte-exactness against the independent oracle: NO helpful
        # fix of ANY kind anywhere in the file.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )

        # Op spec: beta's body wrapped, every original body line preserved
        # exactly once at the wrapped indent.
        assert "    with audit_lock():\n" in merged, (
            f"wrap block missing from the merge: {m}\n{merged}"
        )
        for body_line in (
            "lines = []",
            "for item in items:",
            "lines.append(str(item))",
            "lines.append(str(len(items)))",
            'return " ".join(lines)',
        ):
            assert merged.count("        " + body_line) == 1, (
                f"body line {body_line!r} not preserved exactly once under "
                f"the with-block: {m}\n{merged}"
            )
        assert "# ... existing code ..." not in merged, (
            f"preservation marker leaked into the file: {m}\n{merged}"
        )

        # Independent structural check: the merged file still fails
        # ast.parse EXACTLY at the preserved defect — a preserved
        # pre-existing error is a PASS per req. 9.
        try:
            ast.parse(merged)
        except SyntaxError as exc:
            defect_lineno = PY_DEFECTS.splitlines().index("def alpha()") + 1
            assert exc.lineno == defect_lineno, (
                f"unexpected syntax error at another position: {exc} | {m}"
                f"\n{merged}"
            )
        else:
            raise AssertionError(
                "merged file parsed cleanly — the preserved defect "
                f"disappeared (trait mutation): {m}\n{merged}"
            )

    run = _run_until_converged(
        check, real_engine, tmp_path, PY_DEFECTS, PY_WRAP_SNIPPET,
        "gigo-a", "python", replace="beta",
    )
    # Bounded convergence on the retry loop: a handful of validation
    # retries is legitimate (real model noise); exhaustion is not.
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )

    # Backward reconstruction (req. 3): the oracle's inverse op applied to
    # the EDITED bytes regenerates the original byte-for-byte.
    assert _py_unwrap_oracle(golden) == PY_DEFECTS, (
        "the inverse splice did not regenerate the original bytes"
    )

    # ── The FULL MCP stack: the write path + undo (req. 3) ──────────────
    target = tmp_path / "gigo-a-mcp.py"
    target.write_text(PY_DEFECTS, encoding="utf-8")
    response = _mcp_edit(target, PY_WRAP_SNIPPET, replace="beta")
    assert response.startswith(f"Applied edit to {target}"), (
        f"the MCP tool did not land the edit: {response!r}"
    )
    assert "rejected" not in response and "Error" not in response, response
    assert target.read_text(encoding="utf-8") == golden, (
        f"the file on disk is not the committed golden: "
        f"{first_diff_tag(golden, target.read_text(encoding='utf-8'))}"
    )
    undo = _mcp_undo(target)
    assert undo.startswith(f"Reverted {target}"), undo
    assert target.read_bytes() == PY_DEFECTS.encode("utf-8"), (
        "the MCP undo did not restore the original bytes exactly"
    )


# ---------------------------------------------------------------------------
# Case (b) — md: malformed frontmatter preserved by a body edit (req. 9's
# exact PASS example)
# ---------------------------------------------------------------------------

MD_ORIGINAL = _golden("md_frontmatter/original.md")
MD_MALFORMED_BLOCK = "---\ntitle: [unclosed list\ntags: [a, b\ndate: 2024-13-45\n\n"
MD_BODY_SNIPPET = (
    "# Field notes\n"
    "Body paragraph ONE was edited by the command.\n"
    "# ... existing code ...\n"
    "Body paragraph two records the drift.\n"
)
MD_PARA_ONE = "Body paragraph one opens the sensor log.\n"
MD_PARA_ONE_EDITED = "Body paragraph ONE was edited by the command.\n"


def test_b_md_malformed_frontmatter_preserved_by_body_edit(real_engine, tmp_path):
    """req. 9's exact example: a body edit keeps malformed frontmatter.

    The fixture's frontmatter is malformed TWICE — ``title: [unclosed
    list`` and a MISSING closing ``---`` — so the D3 md structure battery
    sees the ``(md_frontmatter_unclosed, '---')`` STATE trait. A body
    paragraph edit must land with the frontmatter preserved byte-exact
    (a PASS, not a failure), and the D3 battery must NOT reject the merge.
    """
    from fastedit.inference.chunked_merge import _attribute_rejection_reason
    from fastedit.lang_attributes import extract_attributes

    golden = _golden("md_frontmatter/expected_body_edit.md")
    assert golden == _splice(MD_ORIGINAL, (MD_PARA_ONE, MD_PARA_ONE_EDITED)), (
        "the committed golden disagrees with the independent splice oracle"
    )
    # Fixture invariant: the original really carries BOTH malformations.
    orig_keys = [t.key for t in extract_attributes(MD_ORIGINAL, "markdown")]
    assert ("md_frontmatter_unclosed", "---") in orig_keys, orig_keys
    assert not any(k[0] == "md_frontmatter_key" for k in orig_keys), (
        "an unclosed frontmatter block must yield no key traits (D3 spec)"
    )

    def check(run, _target):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert run.merge_results, f"model path never invoked: {m}"
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )

        # THE req. 9 assertion: the malformed frontmatter survives
        # BYTE-EXACT — both the unclosed key line and the missing closer.
        assert merged.startswith(MD_MALFORMED_BLOCK), (
            f"the malformed frontmatter was mutated (trait mutation): "
            f"{m}\n{merged}"
        )
        assert "title: [unclosed list\n" in merged, m
        assert "---\n" not in merged[len(MD_MALFORMED_BLOCK):], (
            f"the model 'helpfully' closed the frontmatter: {m}\n{merged}"
        )

        # The D3 md structure battery must NOT reject this merge — its own
        # predicate, fed exactly what the pipeline feeds it per attempt.
        assert _attribute_rejection_reason(
            MD_ORIGINAL, merged, MD_BODY_SNIPPET, "markdown",
        ) is None, (
            f"the D3 battery rejected a faithful body edit: {m}"
        )
        # Trait identity: the same structure traits, in the same order.
        assert [
            t.key for t in extract_attributes(merged, "markdown")
        ] == orig_keys, (
            f"the structure-trait inventory changed: {m}"
        )

        # The commanded edit landed; everything else byte-exact (golden).
        assert MD_PARA_ONE_EDITED in merged and MD_PARA_ONE not in merged, (
            f"the body edit did not land: {m}\n{merged}"
        )
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )
        assert "# ... existing code ..." not in merged, m

    run = _run_until_converged(
        check, real_engine, tmp_path, MD_ORIGINAL, MD_BODY_SNIPPET,
        "gigo-b", "markdown",
    )
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )

    # Backward reconstruction (req. 3): the inverse splice regenerates the
    # original bytes exactly.
    assert _splice(golden, (MD_PARA_ONE_EDITED, MD_PARA_ONE)) == MD_ORIGINAL, (
        "the inverse splice did not regenerate the original bytes"
    )


# ---------------------------------------------------------------------------
# Case (c) — md: the COMMANDED frontmatter fix must land well-formed
# ---------------------------------------------------------------------------

MD_FRONT_SNIPPET = (
    "---\n"
    "title: Release Notes\n"
    "tags: [a, b]\n"
    "date: 2024-01-15\n"
    "---\n"
    "\n"
    "# ... existing code ...\n"
    "## Appendix\n"
)
MD_WELLFORMED_BLOCK = (
    "---\ntitle: Release Notes\ntags: [a, b]\ndate: 2024-01-15\n---\n\n"
)


def test_c_md_commanded_frontmatter_fix_lands_well_formed(
    real_engine, mcp_harness, tmp_path,
):
    """req. 9's mirror: the command replaces the malformed frontmatter.

    The snippet DECLARES the well-formed block; the output frontmatter
    MUST be well-formed and the rest of the file byte-exact. The validator
    refuses anything less (the D3 gate names the missing declared trait)
    and the unified loop repeats until correct — non-convergence after the
    bounded outer attempts is a product defect. Measured evidence (plan
    §0): the real model applies the declared block on the first attempt of
    this shape (``retries == 0`` on every measured trial); the
    refuse-and-repeat direction — rejection, corrective note, retry,
    acceptance, and fail-loud refusal on never-converging output — is
    pinned on this EXACT fixture at the default tier:
    ``tests/test_gigo_hermetic.py``. The real model's retry engagement is
    naturally observed in case (a) (one validation retry, measured).
    """
    from fastedit.inference.chunked_merge import _attribute_rejection_reason
    from fastedit.lang_attributes import extract_attributes

    golden = _golden("md_frontmatter/expected_frontmatter_fix.md")
    assert golden == _splice(
        MD_ORIGINAL, (MD_MALFORMED_BLOCK, MD_WELLFORMED_BLOCK),
    ), "the committed golden disagrees with the independent splice oracle"

    def check(run, _target):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert run.merge_results, f"model path never invoked: {m}"
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"the validator refused the commanded fix: {m}\n{merged}"
        )

        # THE commanded fix landed: the frontmatter is WELL-FORMED.
        assert merged.startswith(MD_WELLFORMED_BLOCK), (
            f"the output frontmatter is not the declared well-formed block: "
            f"{m}\n{merged}"
        )
        assert "title: [unclosed list" not in merged, (
            f"the malformed title line survived a command that replaced it: "
            f"{m}\n{merged}"
        )
        assert "date: 2024-13-45" not in merged, (
            f"the malformed date line survived the commanded replacement: "
            f"{m}\n{merged}"
        )
        traits = extract_attributes(merged, "markdown")
        trait_keys = [t.key for t in traits]
        assert all(t.kind != "md_frontmatter_unclosed" for t in traits), (
            f"the frontmatter is still unclosed after the commanded fix: "
            f"{m}\n{merged}"
        )
        for declared in (
            ("md_frontmatter_key", "title: Release Notes"),
            ("md_frontmatter_key", "tags: [a, b]"),
            ("md_frontmatter_key", "date: 2024-01-15"),
        ):
            assert declared in trait_keys, (
                f"declared trait {declared} never landed: {m}\n{merged}"
            )

        # The D3 battery's own predicate accepts the declared fix.
        assert _attribute_rejection_reason(
            MD_ORIGINAL, merged, MD_FRONT_SNIPPET, "markdown",
        ) is None, f"the D3 battery rejected the declared fix: {m}"

        # The rest of the file is byte-exact (the independent golden).
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )
        assert "# ... existing code ..." not in merged, m

    run = _run_until_converged(
        check, real_engine, tmp_path, MD_ORIGINAL, MD_FRONT_SNIPPET,
        "gigo-c", "markdown",
    )
    # The retry loop's budget bound: convergence, never exhaustion. (The
    # refuse-and-repeat DIRECTION is pinned hermetically in
    # tests/test_gigo_hermetic.py on this exact fixture.)
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )

    # Backward reconstruction (req. 3): restoring the malformed block
    # regenerates the original bytes exactly.
    assert _splice(golden, (MD_WELLFORMED_BLOCK, MD_MALFORMED_BLOCK)) == (
        MD_ORIGINAL
    ), "the inverse splice did not regenerate the original bytes"

    # ── The FULL MCP stack: the write path + undo (req. 3) ──────────────
    target = tmp_path / "gigo-c-mcp.md"
    target.write_text(MD_ORIGINAL, encoding="utf-8")
    response = _mcp_edit(target, MD_FRONT_SNIPPET)
    assert response.startswith(f"Applied edit to {target}"), (
        f"the MCP tool did not land the commanded fix: {response!r}"
    )
    assert "rejected" not in response and "Error" not in response, response
    final = target.read_text(encoding="utf-8")
    assert final == golden, (
        f"the file on disk is not the committed golden: "
        f"{first_diff_tag(golden, final)}"
    )
    assert final.startswith(MD_WELLFORMED_BLOCK), (
        "the well-formed frontmatter did not reach the file on disk"
    )
    undo = _mcp_undo(target)
    assert undo.startswith(f"Reverted {target}"), undo
    assert target.read_bytes() == MD_ORIGINAL.encode("utf-8"), (
        "the MCP undo did not restore the original bytes exactly"
    )


# ---------------------------------------------------------------------------
# Case (d) — md: the unclosed fence (D4 corpus shape) survives both
# measured model outcomes
# ---------------------------------------------------------------------------

MD_FENCE_ORIGINAL = _golden("md_fence/original.md")
MD_FENCE_SNIPPET = (
    "# Field notes\n"
    "Body paragraph ONE was edited by the command.\n"
    "# ... existing code ...\n"
    "Body paragraph two lives here.\n"
)
MD_FENCE_PARA = "Body paragraph one lives here.\n"
MD_FENCE_PARA_EDITED = "Body paragraph ONE was edited by the command.\n"


def test_d_md_unclosed_fence_survives_both_model_outcomes(
    mcp_harness, tmp_path,
):
    """req. 9 at the trait level, under BOTH measured model outcomes.

    Measured on the real model (D4 probes + this suite's probe): with an
    unclosed fence at EOF the model sometimes "helpfully" closes it — a
    req. 9 trait mutation the battery rejects on every attempt (the edit
    is refused fail-loud) — and sometimes it converges faithfully. ONE
    policy covers both outcomes and is what this test pins:

    * refused → the tool gate refuses the all-chunks-rejected merge, the
      file keeps the ORIGINAL bytes, the trait survives;
    * applied → the file is the committed golden (every other byte
      identical) and the trait survives byte-exact.
    """
    from fastedit.lang_attributes import extract_attributes

    golden = _golden("md_fence/expected_body_edit.md")
    assert golden == _splice(
        MD_FENCE_ORIGINAL, (MD_FENCE_PARA, MD_FENCE_PARA_EDITED),
    ), "the committed golden disagrees with the independent splice oracle"
    assert ("md_fence_unclosed", "```text") in [
        t.key for t in extract_attributes(MD_FENCE_ORIGINAL, "markdown")
    ], "fixture invariant: the corpus carries the unclosed-fence trait"

    target = tmp_path / "gigo-d.md"
    target.write_text(MD_FENCE_ORIGINAL, encoding="utf-8")
    response = _mcp_edit(target, MD_FENCE_SNIPPET)
    final = target.read_text(encoding="utf-8")

    if response.startswith("Error: edit rejected"):
        # The fail-loud answer: nothing was written, the original bytes stay.
        assert "File unchanged" in response, response
        assert final == MD_FENCE_ORIGINAL, (
            "the rejected merge modified the file on disk"
        )
        print("[D5 gigo] case d: refused fail-loud, original kept")
    else:
        # The faithful answer: the edit landed byte-exact vs the golden.
        assert response.startswith(f"Applied edit to {target}"), response
        assert final == golden, (
            f"the applied merge is not the committed golden: "
            f"{first_diff_tag(golden, final)}\n{final}"
        )
        print("[D5 gigo] case d: applied byte-exact vs golden")

    # EITHER WAY: the malformed trait survives on disk BYTE-EXACT
    # (EDIT-NOT-CORRECT — a "helpful" fence repair never reaches the file)
    # and no undecodable byte ever becomes U+FFFD.
    assert final.endswith("```text\nleftover snippet\n"), (
        f"the unclosed fence's bytes were mutated: {final!r}"
    )
    assert ("md_fence_unclosed", "```text") in [
        t.key for t in extract_attributes(final, "markdown")
    ]
    assert "\ufffd" not in final


# ---------------------------------------------------------------------------
# Case (e) — json: the trailing comma is a preserved defect
# ---------------------------------------------------------------------------

JSON_ORIGINAL = _golden("json_trailing_comma/original.json")
JSON_SNIPPET = (
    "{\n"
    '  "name": "final report",\n'
    "# ... existing code ...\n"
    '  "totals": {\n'
)
JSON_OLD_LINE = '  "name": "report",\n'
JSON_NEW_LINE = '  "name": "final report",\n'
JSON_DEFECT_LINE = '    "sum": 42,\n'


def test_e_json_trailing_comma_preserved_by_unrelated_edit(real_engine, tmp_path):
    """req. 9 for json: editing one key keeps the trailing comma defect.

    MEASURED routing (documented per the plan): ``detect_language`` resolves
    json and tree-sitter's ERROR RECOVERY still yields an AST map on the
    broken input (the parser recovers ``key`` nodes around the defect), so
    chunk location runs on that error-recovered map — one whole-file
    region — NOT the D2 text-window path, and NOT a ``replace=`` path (the
    broken input has no resolvable target span for the deterministic
    editors). The battery's RELATIVE parse gate (json grammar) ACCEPTS the
    merge: the original is broken and the trailing-comma error trait is
    INHERITED (kind + containing line unchanged by the edit). The defect
    survives byte-exact and the file is still invalid JSON afterwards — a
    preserved defect, never a "fix".
    """
    from fastedit.data_gen.ast_analyzer import parse_diagnostics

    golden = _golden("json_trailing_comma/expected_edit.json")
    assert golden == _splice(JSON_ORIGINAL, (JSON_OLD_LINE, JSON_NEW_LINE)), (
        "the committed golden disagrees with the independent splice oracle"
    )
    # Fixture invariant: the input really is invalid JSON...
    orig_diag = parse_diagnostics(JSON_ORIGINAL, "json")
    assert not orig_diag.is_valid, orig_diag.errors
    # ...and the grammar resolves with an error-recovered AST map (the
    # measured routing documented in the docstring).
    from fastedit.inference.ast_utils import get_ast_map_from_source

    assert get_ast_map_from_source(JSON_ORIGINAL, "data.json", "json"), (
        "fixture invariant: tree-sitter error recovery yields nodes"
    )

    def check(run, _target):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert run.merge_results, f"model path never invoked: {m}"
        # THE relative gate accepts: the trailing-comma error trait is
        # inherited, so the broken original's defect is a preserved trait.
        assert result.parse_valid, (
            f"the relative parse gate rejected a defect-preserving merge: "
            f"{m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )
        # Measured routing: one whole-file region on the error-recovered map.
        assert result.chunks_used == 1, f"measured routing changed: {m}"
        assert result.chunk_regions == [(
            1, len(JSON_ORIGINAL.splitlines()),
        )], f"measured routing changed: {result.chunk_regions}"

        # The commanded edit landed; the DEFECT survives byte-exact.
        assert JSON_NEW_LINE in merged and JSON_OLD_LINE not in merged, (
            f"the key edit did not land: {m}\n{merged}"
        )
        assert JSON_DEFECT_LINE in merged, (
            f"the model 'fixed' the untouched trailing comma: {m}\n{merged}"
        )
        # The file is STILL invalid JSON — the defect is preserved, and the
        # only error trait is the inherited one on the same line.
        merged_diag = parse_diagnostics(merged, "json")
        assert not merged_diag.is_valid, (
            f"the merged file parses cleanly — the preserved defect "
            f"disappeared (trait mutation): {m}\n{merged}"
        )

        # Golden byte-exactness against the independent splice.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )
        assert "# ... existing code ..." not in merged, m

    run = _run_until_converged(
        check, real_engine, tmp_path, JSON_ORIGINAL, JSON_SNIPPET,
        "gigo-e", "json",
    )
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )

    # Backward reconstruction (req. 3).
    assert _splice(golden, (JSON_NEW_LINE, JSON_OLD_LINE)) == JSON_ORIGINAL, (
        "the inverse splice did not regenerate the original bytes"
    )


# ---------------------------------------------------------------------------
# Case (f) — typo preservation under replacement: the commanded removal is
# exact, the same typo elsewhere survives
# ---------------------------------------------------------------------------

PY_TYPO_ORIGINAL = _golden("py_typo/original.py")
PY_BETA_SNIPPET = (
    "def beta(items):\n"
    "    total = len(items)\n"
    "    return f\"count={total}\"\n"
)
PY_BETA_OLD_BLOCK = (
    "def beta(items):\n"
    "    reuslt = 0\n"
    "    for item in items:\n"
    "        reuslt += len(items)\n"
    "    return reuslt\n"
)
PY_BETA_NEW_BLOCK = (
    "def beta(items):\n"
    "    total = len(items)\n"
    "    return f\"count={total}\"\n"
)


def test_f_typo_preserved_outside_the_replaced_region(
    mcp_harness, tmp_path,
):
    """req. 9: a commanded replacement removes the typo ONLY where told.

    The snippet replaces ``beta`` — whose body CONTAINS the typo'd
    identifier — with a complete re-definition that does not use it. The
    typo inside the REPLACED region disappears (commanded); the SAME typo
    in untouched ``alpha`` survives byte-exact. The committed golden
    proves both. The snippet is a complete re-definition, so the ZERO-TOKEN
    deterministic direct-swap path runs (asserted via the MCP response's
    ``0 tokens`` metrics — no model call to fabricate or absorb noise).
    """
    golden = _golden("py_typo/expected_replace_beta.py")
    assert golden == _splice(
        PY_TYPO_ORIGINAL, (PY_BETA_OLD_BLOCK, PY_BETA_NEW_BLOCK),
    ), "the committed golden disagrees with the independent splice oracle"

    target = tmp_path / "gigo-f.py"
    target.write_text(PY_TYPO_ORIGINAL, encoding="utf-8")
    response = _mcp_edit(target, PY_BETA_SNIPPET, replace="beta")
    assert response.startswith(f"Applied edit to {target}"), (
        f"the MCP tool did not land the replacement: {response!r}"
    )
    assert "rejected" not in response and "Error" not in response, response
    assert "0 tokens" in response, (
        f"the complete re-definition must take the zero-token direct-swap "
        f"path: {response!r}"
    )

    final = target.read_text(encoding="utf-8")
    # The commanded removal: beta's typo'd lines are GONE.
    assert PY_BETA_NEW_BLOCK in final and PY_BETA_OLD_BLOCK not in final, (
        f"the commanded replacement did not land: {final}"
    )
    # The SAME typo elsewhere (alpha) survives byte-exact.
    assert "    reuslt = 0\n    return reuslt\n" in final, (
        "the model 'fixed' the untouched typo in alpha"
    )
    assert final.count("reuslt") == 2, (
        f"the typo must survive exactly twice (alpha only): {final}"
    )
    # Golden byte-exactness: nothing else changed anywhere.
    assert final == golden, (
        f"the file on disk differs from the committed golden: "
        f"{first_diff_tag(golden, final)}\n{final}"
    )

    # Backward reconstruction (req. 3): the inverse splice regenerates the
    # original bytes exactly.
    assert _splice(final, (PY_BETA_NEW_BLOCK, PY_BETA_OLD_BLOCK)) == (
        PY_TYPO_ORIGINAL
    ), "the inverse splice did not regenerate the original bytes"

    # The MCP undo restores the original bytes exactly (req. 3).
    undo = _mcp_undo(target)
    assert undo.startswith(f"Reverted {target}"), undo
    assert target.read_bytes() == PY_TYPO_ORIGINAL.encode("utf-8"), (
        "the MCP undo did not restore the original bytes exactly"
    )


# ---------------------------------------------------------------------------
# Case (g) — html: the valueless attribute (`<div lang>`) is a preserved
# malformed trait
# ---------------------------------------------------------------------------

HTML_ORIGINAL = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head><title>Notes</title></head>\n"
    "<body>\n"
    "  <div lang>Valueless attribute demo.</div>\n"
    "  <p>First paragraph.</p>\n"
    "  <p>Second paragraph.</p>\n"
    "</body>\n"
    "</html>\n"
)
HTML_SNIPPET = (
    "  <div lang>Valueless attribute demo.</div>\n"
    "# ... existing code ...\n"
    "  <p>First paragraph.</p>\n"
    "  <aside>Inserted by the command.</aside>\n"
    "  <p>Second paragraph.</p>\n"
)
HTML_INSERTED = "  <aside>Inserted by the command.</aside>\n"


def test_g_html_valueless_attribute_preserved_by_unrelated_edit(
    real_engine, tmp_path,
):
    """req. 9 + req. 7: the D3 malformed-attribute row survives an edit.

    ``<div lang>`` (the D3 spec's valueless-attribute row, trait identity
    ``(html_lang, '')``) is malformed-but-valid HTML. An unrelated
    insertion after the first paragraph must land with the valueless
    attribute preserved BYTE-EXACT, and the D3 attribute gate's own
    predicate must accept the merge.
    """
    from fastedit.inference.chunked_merge import _attribute_rejection_reason
    from fastedit.lang_attributes import extract_attributes

    golden = _splice(
        HTML_ORIGINAL, ("  <p>First paragraph.</p>\n",
                        "  <p>First paragraph.</p>\n" + HTML_INSERTED),
    )
    assert ("html_lang", "") in [
        t.key for t in extract_attributes(HTML_ORIGINAL, "html")
    ], "fixture invariant: the valueless attribute is a trait"

    def check(run, _target):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        assert run.merge_results, f"model path never invoked: {m}"
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}\n{merged}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator: {m}\n{merged}"
        )

        # THE malformed attribute survives BYTE-EXACT.
        assert "  <div lang>Valueless attribute demo.</div>\n" in merged, (
            f"the valueless attribute was 'repaired' or dropped: {m}\n"
            f"{merged}"
        )
        # The commanded insertion landed, exactly once.
        assert merged.count(HTML_INSERTED) == 1, (
            f"the declared insertion did not land exactly once: {m}\n{merged}"
        )
        # The D3 attribute gate's own predicate accepts the merge.
        assert _attribute_rejection_reason(
            HTML_ORIGINAL, merged, HTML_SNIPPET, "html",
        ) is None, f"the D3 attribute gate rejected a faithful merge: {m}"

        # Golden byte-exactness against the independent splice.
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}\n{merged}"
        )
        assert "# ... existing code ..." not in merged, m

    run = _run_until_converged(
        check, real_engine, tmp_path, HTML_ORIGINAL, HTML_SNIPPET,
        "gigo-g", "html",
    )
    assert run.result.retries <= 3, (
        f"retry-until-valid loop needed an unreasonable budget: "
        f"{metrics_tag(run)}"
    )
