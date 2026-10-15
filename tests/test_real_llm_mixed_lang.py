"""Step D4 — the mixed-language flagship e2e + language attributes (real LLM).

Marked ``llm`` — deselected from the default tier; run explicitly with
``uv run pytest tests/test_real_llm_mixed_lang.py -m llm``. There is NO
fake engine here: every test drives the trained fastedit mlx-8bit model
through the FULL MCP stack (``mcp_harness`` — real ``chunked_merge``, real
atomic writes, real ``BackupStore``, real undo ledger) exactly like the C2
suite does.

THE FLAGSHIP (req. 8): the committed mixed-language corpus
``tests/golden/mixed_md/`` is a realistic .md article — English intro,
Chinese paragraphs (。punctuation), fenced python + go code blocks whose
comments are MIXED zh/en, an embedded html widget with ``lang="zh-CN"``,
well-formed frontmatter, and ONE unclosed fence in an unrelated appendix
section (the req. 9 malformed trait, a ``~~~`` opener — see the measured
note below). The op: replace every Chinese comment with its DECLARED
English translation (the table lives in the corpus manifest, corpus.json),
every other byte preserved. The op is issued as a PER-COMMENT loop of
``fast_edit`` calls through the full MCP stack — closest to real usage —
and asserted byte-exact against the committed golden
(``expected_translate_comments.md``), derived by the corpus's INDEPENDENT
line-splice oracle (``tests/golden/mixed_md/_generate.py``, never
fastedit).

MEASURED MODEL BEHAVIOR this file pins (deterministic greedy probes
against the real mlx-8bit model; shapes pinned from those runs per plan
§0's non-determinism policy):

* The per-comment op shape that converges is the D shape: the DECLARED
  translation line stated BEFORE the preservation marker, bracketed by the
  two lines adjacent to the comment (``above / <declared line> /
  # ... existing code ... / below``), the anchors BYTE-EXACT against the
  file. Measured 4/4 comments, first attempt, zero retries, byte-exact —
  and the same shape restores the original Chinese line (the inverse op)
  on the fully-translated state. The anchor bytes are load-bearing (D4
  measured evidence): with a 4-space below anchor against the file's
  8-space ``total += user.order_amount`` line, the second python comment's
  edit was rejected 9/9 on a live run (content-faithfulness — the model
  must reconcile a snippet anchor that disagrees with the chunk bytes);
  with the byte-exact anchor the same op converges. The marker-first
  variant of the same snippet converges for the python comments but
  deterministically fails for the go comments once two translations have
  landed (the model duplicates the go block at EOF and stops without the
  ``</updated-code>`` wrapper — 9/9 attempts on both the plain-AR and the
  speculative path), so the D shape is the one this file pins. The
  battery justifies the old comment's deletion positionally in the
  marker-bearing segment either way; the D shape is the one the real
  model executes faithfully at every state.
* The corpus's malformed trait is an unclosed ``~~~`` fence — the SAME
  trait identity the D3 spec table and fixtures use
  (``(md_fence_unclosed, <opener line>)``). Measured on the real model:
  an unclosed fence at EOF sometimes drives the model to "helpfully" add
  the closing fence — a req. 9 trait mutation the battery rejects on
  every attempt (the D4 probes measured 9/9 non-convergence on the
  BACKTICK variant; the model also omits the ``</updated-code>`` wrapper,
  so B11 flags every attempt truncated) — and sometimes the model
  converges faithfully with the trait preserved (measured live: the
  committed corpus's per-comment edits APPLY). One policy covers both
  outcomes and is what this file pins
  (``test_real_unclosed_fence_trait_survives_both_model_outcomes`` on the
  backtick variant): either the MCP tool refuses and the file keeps the
  original bytes, or the edit lands byte-exact against the per-comment
  golden — and in BOTH cases the malformed trait survives on disk and no
  U+FFFD appears.
* Whole-file merges of this CJK-heavy corpus sit at the model's
  generation envelope: a 40-line multi-edit snippet made the model emit
  ONLY the snippet's region (dropping the rest of the document), so the
  flagship keeps the per-comment minimal-snippet loop. Assertions run on
  the file the loop CONVERGED on (bounded outer attempts) — an extra
  unconditional re-run would defeat the bound (measured: the flagship's
  second comment converged inside the outer attempts and rejected 9/9 on
  an additional re-run in the same session).

LANGUAGE-ATTRIBUTE e2e (req. 7) under the real model: html (lang
preserved through an unrelated paragraph edit; a DECLARED
``lang="fr"``→``lang="de"`` change lands exactly while every other lang
attribute stays), latex (an edit inside the ``\\selectlanguage{french}``
scope preserves the babel markers; ``detect_language(".tex") is None`` —
the structureless text path plus the D3 latex attribute battery), rtf (an
edit between ``\\langNNNN`` runs preserves the control words; the D2
text-window path plus the D1 trait gate and the D3 rtf battery), docx
(the committed container under ``tests/golden/docx/`` edited through the
adapter path — the merged document text is byte-exact against the
line-splice golden, every other zip entry keeps its payload, compression,
timestamp and order, and one MCP undo restores the ORIGINAL container
byte-for-byte).
"""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import pytest
from llm_fixtures import first_diff_tag

pytestmark = pytest.mark.llm

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for the golden byte-exactness assertion
(plan §0 policy (b)); non-convergence after the bound is a product defect."""

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# ---------------------------------------------------------------------------
# The committed mixed-language corpus (tests/golden/mixed_md/)
# ---------------------------------------------------------------------------

_MIXED_DIR = GOLDEN_DIR / "mixed_md"
ORIGINAL = (_MIXED_DIR / "original.md").read_text(encoding="utf-8")
EXPECTED = (_MIXED_DIR / "expected_translate_comments.md").read_text(
    encoding="utf-8",
)
MANIFEST = json.loads((_MIXED_DIR / "corpus.json").read_text(encoding="utf-8"))
TRANSLATIONS = MANIFEST["translations"]
"""Each entry: the exact zh comment line and its declared English line."""

# The per-comment D-shape anchors: the two lines adjacent to the comment in
# the file, byte-exact (the anchors are context lines the model preserves —
# an anchor that disagrees with the chunk's actual bytes destabilizes the
# merge: with a 4-space below anchor against the file's 8-space line the
# second python comment's edit was measured to reject 9/9 on one run).
_COMMENT_NEIGHBOURS = {
    "    # 遍历所有用户 / iterate all users": (
        "def summarize(users):",
        "    total = 0",
    ),
    "        # 累加订单金额 (sum the order amounts)": (
        "    for user in users:",
        "        total += user.order_amount",
    ),
    "    // 打开输出文件 / open the output file": (
        "func Export(rows []Row) error {",
        '    f, err := os.Create("report.csv")',
    ),
    "    // 逐行写入报表 (write the report row by row)": (
        "    w := csv.NewWriter(f)",
        "    for _, row := range rows {",
    ),
}


def _d_snippet(zh_line: str, en_line: str, reverse: bool = False) -> str:
    """The per-comment D-shape snippet (measured-converging).

    The DECLARED line (the translation — or, for the inverse op, the
    original Chinese line) is stated BEFORE the preservation marker,
    bracketed by the two adjacent original lines. The payload keeps its
    exact indentation: the snippet's declared lines are what the model is
    told to insert, and the golden depends on their exact bytes.
    """
    above, below = _COMMENT_NEIGHBOURS[zh_line]
    payload = en_line if not reverse else zh_line
    indent = zh_line[: len(zh_line) - len(zh_line.lstrip())]
    comment = (
        "# ... existing code ..."
        if zh_line.lstrip().startswith("#")
        else "// ... existing code ..."
    )
    return f"{above}\n{payload}\n{indent}{comment}\n{below}\n"


def _compose(source: str, *pairs: tuple[str, str]) -> str:
    """The line-splice oracle: swap each declared pair's lines, in order."""
    for old, new in pairs:
        source = source.replace(old + "\n", new + "\n")
    return source


def _forward_pairs() -> list[tuple[str, str]]:
    return [(entry["zh"], entry["en"]) for entry in TRANSLATIONS]


def _per_comment_goldens(source: str) -> list[str]:
    """The intermediate file after each single-comment translation."""
    goldens = []
    state = source
    for zh, en in _forward_pairs():
        state = state.replace(zh + "\n", en + "\n")
        goldens.append(state)
    return goldens


def _run_mcp_edit(path: Path, snippet: str) -> str:
    from fastedit.mcp import tools_edit

    return asyncio.run(tools_edit.fast_edit(
        file_path=str(path), edit_snippet=snippet,
    ))


def _run_translated_loop(path: Path) -> None:
    """The flagship op: one fast_edit call per Chinese comment, in order.

    Each edit is asserted against its per-comment golden ON DISK — the
    real write path's bytes, not just the merge result.
    """
    for i, (zh, en) in enumerate(_forward_pairs(), start=1):
        response = _run_mcp_edit(path, _d_snippet(zh, en))
        assert response.startswith(f"Applied edit to {path}"), (
            f"translation edit {i} did not land: {response!r}"
        )
        assert "rejected" not in response and "Error" not in response, (
            f"translation edit {i} refused: {response!r}"
        )
        want = _compose(ORIGINAL, *_forward_pairs()[:i])
        got = path.read_text(encoding="utf-8")
        assert got == want, (
            f"the file on disk is not the per-comment golden after edit {i}: "
            f"{first_diff_tag(want, got)}\n{got}"
        )


def _assert_trait_level_integrity(final: str, m: str) -> None:
    """The trait-level structural assertions on the composed file (req. 7+9)."""
    from fastedit.data_gen.ast_analyzer import parse_diagnostics
    from fastedit.inference.chunked_merge import _attribute_rejection_reason
    from fastedit.lang_attributes import extract_attributes
    from fastedit.text_heuristics import text_traits

    # The D3 trait inventory is UNCHANGED: same structure traits, same
    # order — the malformed unclosed fence, every fence info string and
    # marker length, the frontmatter delimiters and key lines.
    original_traits = [t.key for t in extract_attributes(ORIGINAL, "markdown")]
    final_traits = [t.key for t in extract_attributes(final, "markdown")]
    assert final_traits == original_traits, (
        f"the language-attribute/structure traits changed "
        f"(req. 7/9 violation): {m}\n"
        f"original={original_traits}\nfinal={final_traits}"
    )
    # The D3 battery's own predicate is GREEN on the composed op: with no
    # declared attribute change, every original trait must survive in
    # order — the exact check the final assembly runs for a multi-edit op.
    composed_snippet = "".join(
        _d_snippet(zh, en) for zh, en in _forward_pairs()
    )
    reason = _attribute_rejection_reason(
        ORIGINAL, final, composed_snippet, "markdown",
    )
    assert reason is None, (
        f"the D3 assembly gate rejected the composed op: {reason} | {m}"
    )
    # No U+FFFD anywhere: an undecodable byte must never become a
    # replacement character and get written back.
    assert "\ufffd" not in final, f"U+FFFD in the merged file: {m}"
    # CJK bytes intact: the CJK-aware trait counts of the final file equal
    # the golden's (the golden encodes the expected multibyte content).
    want, got = text_traits(EXPECTED), text_traits(final)
    for trait in ("cjk_chars", "punct_cjk", "bytes", "lines"):
        assert got[trait] == want[trait], (
            f"{trait} drift: got {got[trait]} want {want[trait]} | {m}"
        )
    # Structurally valid markdown (trait-level): the document parses with
    # the markdown grammar with zero error traits.
    diag = parse_diagnostics(final, "markdown")
    assert diag.is_valid, f"the merged file does not parse as markdown: {m}"


def _forward_loop_with_outer_attempts(path_factory, m_extra: str = "") -> Path:
    """Run the translated loop inside bounded outer attempts (plan §0 (b)).

    Returns the CONVERGED file — the one whose on-disk bytes the callers
    assert against. Each outer attempt is a complete fresh loop; a failed
    attempt discards that file and starts over. Non-convergence after the
    bound is a product defect.
    """
    last_error: AssertionError | None = None
    for attempt in range(1, OUTER_ATTEMPTS + 1):
        target = path_factory(attempt)
        try:
            _run_translated_loop(target)
            return target
        except AssertionError as exc:
            last_error = exc
    raise AssertionError(
        f"the flagship loop did not converge within {OUTER_ATTEMPTS} outer "
        f"attempts — non-convergence is a product defect (plan §0): "
        f"{last_error}{m_extra}"
    )


# ---------------------------------------------------------------------------
# 1. THE FLAGSHIP — translate every Chinese comment, byte-exact
# ---------------------------------------------------------------------------


def test_real_flagship_translate_chinese_comments_byte_exact(
    mcp_harness, tmp_path,
):
    """req. 8 headline: per-comment loop through the FULL MCP stack.

    Every Chinese comment becomes its DECLARED English translation; every
    other byte — frontmatter, CJK paragraphs, fence info strings, the
    embedded html widget with ``lang="zh-CN"``, the malformed unclosed
    fence — is identical to the original, proven against the committed
    line-splice golden. The assertions run on the file the loop CONVERGED
    on (bounded outer attempts, plan §0 policy (b)) — no extra
    unconditional re-run, which would defeat the bound.
    """

    def make(attempt: int) -> Path:
        target = tmp_path / f"flagship-{attempt}.md"
        target.write_text(ORIGINAL, encoding="utf-8")
        return target

    target = _forward_loop_with_outer_attempts(make)
    final = target.read_text(encoding="utf-8")
    m = "flagship"
    assert final == EXPECTED, (
        f"the composed file differs from the committed golden: "
        f"{first_diff_tag(EXPECTED, final)} | {m}\n{final}"
    )
    _assert_trait_level_integrity(final, m)
    for entry in TRANSLATIONS:
        assert entry["zh"] not in final, m
        assert entry["en"] in final, m


def test_real_backward_reconstruction_restores_original_comments(
    mcp_harness, tmp_path,
):
    """req. 3: the pipeline's output differs from the original by EXACTLY
    the declared op.

    Two proofs: (a) the model-driven inverse — restoring the first
    comment's original Chinese line through the same MCP stack lands
    byte-exact; (b) the corpus oracle's inverse line-splice applied to the
    pipeline's final output regenerates the ORIGINAL bytes exactly (the C2
    case-(e) doctrine: the golden's inverse op, never fastedit, executes
    the backward reconstruction). Assertions run on the file the forward
    loop CONVERGED on (bounded outer attempts, plan §0 policy (b)).
    """

    def make(attempt: int) -> Path:
        target = tmp_path / f"backward-{attempt}.md"
        target.write_text(ORIGINAL, encoding="utf-8")
        return target

    target = _forward_loop_with_outer_attempts(make)

    # (a) the model-driven inverse op: restore the first comment's zh line.
    zh, en = _forward_pairs()[0]
    response = _run_mcp_edit(target, _d_snippet(zh, en, reverse=True))
    assert response.startswith(f"Applied edit to {target}"), (
        f"the model-driven inverse op did not land: {response!r}"
    )
    after_restore = target.read_text(encoding="utf-8")
    assert after_restore == _compose(EXPECTED, (en, zh)), (
        f"the model-driven inverse op did not land byte-exact: "
        f"{first_diff_tag(_compose(EXPECTED, (en, zh)), after_restore)}\n"
        f"{after_restore}"
    )

    # (b) the oracle's inverse line-splice over the remaining comments
    # regenerates the ORIGINAL bytes exactly.
    oracle_inverse = _compose(after_restore, *[
        (en, zh) for zh, en in _forward_pairs()[1:]
    ])
    assert oracle_inverse == ORIGINAL, (
        f"the oracle's inverse op did not regenerate the original bytes: "
        f"{first_diff_tag(ORIGINAL, oracle_inverse)}\n{oracle_inverse}"
    )


def test_real_ndeep_undo_through_mcp_stack(mcp_harness, tmp_path):
    """req. 3: N-deep undo — every intermediate state back to the original,
    byte-exact, through the MCP backup stack.

    Backup semantics (B22/B38, measured): each of the N writes stores the
    file's pre-write bytes as a NEW backup, and every ``fast_undo`` pops the
    NEWEST one — so undo #k restores the state before edit ``N-k+1``: the
    three intermediate goldens g3, g2, g1, then the ORIGINAL, and the fifth
    call finds an empty ledger and fails loud.
    """
    def make(attempt: int) -> Path:
        fresh = tmp_path / f"undo-{attempt}.md"
        fresh.write_text(ORIGINAL, encoding="utf-8")
        return fresh

    target = _forward_loop_with_outer_attempts(make)

    from fastedit.mcp import tools_ast

    goldens = _per_comment_goldens(ORIGINAL)  # [g1, g2, g3, g4]
    states = list(reversed(goldens[:-1]))  # undo #1..#3 → g3, g2, g1
    for depth, want in enumerate(states, start=1):
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), (
            f"undo #{depth} failed: {response!r}"
        )
        got = target.read_text(encoding="utf-8")
        assert got == want, (
            f"undo #{depth} did not restore the intermediate state "
            f"byte-exactly: {first_diff_tag(want, got)}"
        )
    # The final undo restores the ORIGINAL bytes exactly.
    response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
    assert response.startswith(f"Reverted {target}"), response
    assert target.read_bytes() == ORIGINAL.encode("utf-8"), (
        "the last undo did not restore the original bytes"
    )
    # The ledger is empty — a further undo fails loud, writes nothing.
    response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
    assert response.startswith("Error: no undo history"), response
    assert target.read_bytes() == ORIGINAL.encode("utf-8")


def test_real_unclosed_fence_trait_survives_both_model_outcomes(
    mcp_harness, tmp_path,
):
    """req. 9 at the trait level, under BOTH measured model outcomes.

    Measured on the real model (module docstring): with an unclosed fence
    at EOF the model sometimes "helpfully" closes it — a req. 9 trait
    mutation the battery rejects on every attempt (the D4 probes measured
    9/9 non-convergence on the backtick variant; the committed ~~~ corpus
    conversely converged first-try in the flagship) — and sometimes it
    converges faithfully. One POLICY covers both outcomes and is what this
    test pins, on the BACKTICK variant of the corpus: the malformed trait
    is NEVER mutated on disk.

    * refused → the tool gate refuses the all-chunks-rejected merge, the
      file keeps the ORIGINAL bytes, the trait survives;
    * applied → the file is the per-comment golden (every other byte
      identical to the backtick variant) and the trait survives byte-exact.
    """
    from fastedit.lang_attributes import extract_attributes

    backtick = ORIGINAL.replace("~~~\n", "```\n")
    assert ("md_fence_unclosed", "```") in [
        t.key for t in extract_attributes(backtick, "markdown")
    ], "fixture invariant: the backtick variant carries the unclosed trait"
    target = tmp_path / "backtick.md"
    target.write_text(backtick, encoding="utf-8")
    zh, en = _forward_pairs()[0]
    snippet = _d_snippet(zh, en)

    response = _run_mcp_edit(target, snippet)
    final = target.read_text(encoding="utf-8")
    if response.startswith("Error: edit rejected"):
        # The fail-loud answer: nothing was written, the original bytes stay.
        assert "hallucinated" in response, response
        assert final == backtick, (
            "the rejected merge modified the file on disk"
        )
    else:
        # The faithful answer: the edit landed byte-exact against the
        # per-comment golden of the backtick variant.
        assert response.startswith(f"Applied edit to {target}"), response
        want = backtick.replace(zh + "\n", en + "\n")
        assert final == want, (
            f"the applied merge is not the per-comment golden: "
            f"{first_diff_tag(want, final)}\n{final}"
        )
    # EITHER WAY: the malformed trait survives on disk (EDIT-NOT-CORRECT —
    # a "helpful" fence repair never reaches the file) and no undecodable
    # byte ever becomes U+FFFD.
    traits = [t.key for t in extract_attributes(final, "markdown")]
    assert ("md_fence_unclosed", "```") in traits, traits
    assert "\ufffd" not in final


# ---------------------------------------------------------------------------
# 2. Language-attribute e2e under the real model (req. 7)
# ---------------------------------------------------------------------------

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lab notes</title>
</head>
<body>
  <section lang="fr" id="intro">
    <h1>Introduction</h1>
    <p>Le rapport couvre le premier trimestre.</p>
    <p>Les chiffres sont provisoires.</p>
  </section>
  <section lang="es" id="methodik">
    <h1>Metodologia</h1>
    <p>La muestra incluye 240 hogares.</p>
  </section>
  <a href="https://example.com/archive" hreflang="fr">archive</a>
</body>
</html>
"""

HTML_PRESERVED_GOLDEN = HTML.replace(
    "    <p>Les chiffres sont provisoires.</p>\n",
    "    <p>Les chiffres sont definitifs.</p>\n",
)
HTML_PRESERVED_SNIPPET = (
    "    <p>Le rapport couvre le premier trimestre.</p>\n"
    "# ... existing code ...\n"
    "    <p>Les chiffres sont definitifs.</p>\n"
    "  </section>\n"
)

HTML_DECLARED_GOLDEN = HTML.replace(
    '  <section lang="fr" id="intro">',
    '  <section lang="de" id="intro">',
)
HTML_DECLARED_SNIPPET = (
    "<body>\n"
    '  <section lang="de" id="intro">\n'
    "# ... existing code ...\n"
    "    <h1>Introduction</h1>\n"
)

LATEX = """\
\\documentclass{article}
\\usepackage[english,french]{babel}
\\begin{document}

\\selectlanguage{french}
Le rapport couvre le premier trimestre.
Les chiffres sont provisoires pour l'instant.
\\selectlanguage{english}
The figures become final next week.

\\end{document}
"""
LATEX_GOLDEN = LATEX.replace(
    "Les chiffres sont provisoires pour l'instant.\n",
    "Les chiffres seront definitifs lundi.\n",
)
LATEX_SNIPPET = (
    "Le rapport couvre le premier trimestre.\n"
    "# ... existing code ...\n"
    "Les chiffres seront definitifs lundi.\n"
    "\\selectlanguage{english}\n"
)

RTF = """\
{\\rtf1\\ansi\\deff0
{\\fonttbl{\\f0 Helvetica;}}
\\lang1036 Voici le rapport du trimestre.\\par
\\lang1033 The figures become final next week.\\par
\\lang1036 Les chiffres sont provisoires.\\par
}
"""
RTF_GOLDEN = RTF.replace(
    "\\lang1036 Voici le rapport du trimestre.\\par\n",
    "\\lang1036 Voici le rapport final du trimestre.\\par\n",
)
RTF_SNIPPET = (
    "{\\fonttbl{\\f0 Helvetica;}}\n"
    "# ... existing code ...\n"
    "\\lang1036 Voici le rapport final du trimestre.\\par\n"
    "\\lang1033 The figures become final next week.\\par\n"
)


def _mcp_edit_until_converged(
    tmp_path: Path, name: str, source: str, snippet: str, golden: str,
    check=None,
) -> str:
    """One MCP fast_edit with bounded outer attempts; returns the final text."""
    last_error: AssertionError | None = None
    for attempt in range(1, OUTER_ATTEMPTS + 1):
        target = tmp_path / f"{name}-{attempt}"
        target.write_text(source, encoding="utf-8")
        response = _run_mcp_edit(target, snippet)
        try:
            assert response.startswith(f"Applied edit to {target}"), (
                f"[outer attempt {attempt}] the edit did not land: {response!r}"
            )
            final = target.read_text(encoding="utf-8")
            assert final == golden, (
                f"[outer attempt {attempt}] merged file differs from the "
                f"golden: {first_diff_tag(golden, final)}\n{final}"
            )
            if check is not None:
                check(target, final, f"[outer attempt {attempt}]")
            return final
        except AssertionError as exc:
            last_error = exc
    raise AssertionError(
        f"the edit did not converge within {OUTER_ATTEMPTS} outer attempts "
        f"— non-convergence is a product defect (plan §0): {last_error}"
    )


def _assert_d3_traits(
    original_text: str, final: str, snippet: str, fmt: str, m: str,
) -> None:
    """The D3 battery's own predicate on the accepted merge (req. 7).

    Fed exactly what the pipeline feeds it per attempt: the original, the
    accepted merge, and the SNIPPET as the op spec.
    """
    from fastedit.inference.chunked_merge import _attribute_rejection_reason

    reason = _attribute_rejection_reason(original_text, final, snippet, fmt)
    assert reason is None, (
        f"the D3 attribute gate's predicate failed on the accepted merge: "
        f"{reason} | {m}"
    )


def test_real_html_lang_preserved_through_paragraph_edit(mcp_harness, tmp_path):
    """(a) html: an unrelated paragraph edit inside a ``lang="fr"`` section
    leaves EVERY lang attribute byte-exact (golden byte-exactness)."""
    from fastedit.data_gen.ast_analyzer import detect_language

    def check(_target, final, m):
        _assert_d3_traits(HTML, final, HTML_PRESERVED_SNIPPET, "html", m)
        for lang in ('lang="en"', 'lang="fr"', 'lang="es"', 'hreflang="fr"'):
            assert lang in final, f"{lang} lost by the merge | {m}"

    final = _mcp_edit_until_converged(
        tmp_path, "html-preserved", HTML, HTML_PRESERVED_SNIPPET,
        HTML_PRESERVED_GOLDEN, check,
    )
    assert detect_language("page.html") == "html"
    assert "\ufffd" not in final


def test_real_html_declared_lang_change_lands_exactly(mcp_harness, tmp_path):
    """(a) html: the snippet explicitly changes ``lang="fr"`` →
    ``lang="de"`` on the intro section; it lands EXACTLY and every other
    lang attribute is untouched."""
    def check(_target, final, m):
        assert 'lang="de" id="intro"' in final, f"declared change did not land | {m}"
        assert 'lang="fr" id="intro"' not in final, m
        # OTHER lang attributes untouched.
        assert '<html lang="en">' in final, m
        assert 'lang="es" id="methodik"' in final, m
        assert 'hreflang="fr"' in final, m
        _assert_d3_traits(HTML, final, HTML_DECLARED_SNIPPET, "html", m)

    _mcp_edit_until_converged(
        tmp_path, "html-declared", HTML, HTML_DECLARED_SNIPPET,
        HTML_DECLARED_GOLDEN, check,
    )


def test_real_latex_selectlanguage_scope_preserved(mcp_harness, tmp_path):
    """(b) latex: an edit INSIDE the ``\\selectlanguage{french}`` scope
    preserves the babel markers byte-exact.

    ``detect_language(".tex") is None`` — the structureless TEXT path runs
    (D1 text-trait battery) plus the D3 latex attribute battery (the fmt
    comes from the file suffix).
    """
    from fastedit.data_gen.ast_analyzer import detect_language
    from fastedit.inference.chunked_merge import _derived_text_op
    from fastedit.lang_attributes import format_for_path
    from fastedit.text_heuristics import (
        TOLERANCE_MODEL_PROSE,
        validate_text_output,
    )

    def check(_target, final, m):
        # The babel markers survived byte-exact.
        assert "\\selectlanguage{french}" in final, m
        assert "\\selectlanguage{english}" in final, m
        _assert_d3_traits(LATEX, final, LATEX_SNIPPET, "latex", m)
        # The D1 text-trait battery's own predicate (the structureless
        # path's gate) accepts the merge against the declared op.
        op, layout_slack, removable = _derived_text_op(LATEX, LATEX_SNIPPET)
        ok, reason = validate_text_output(
            LATEX, op, final, TOLERANCE_MODEL_PROSE,
            layout_slack=layout_slack, removable_traits=removable,
        )
        assert ok is True, f"the D1 trait gate's predicate failed: {reason} | {m}"

    assert detect_language("main.tex") is None
    assert format_for_path("main.tex") == "latex"
    _mcp_edit_until_converged(
        tmp_path, "latex-scope", LATEX, LATEX_SNIPPET, LATEX_GOLDEN, check,
    )


def test_real_rtf_lang_control_words_preserved(mcp_harness, tmp_path):
    """(c) rtf: an edit between ``\\langNNNN`` runs preserves the control
    words byte-exact. RTF is structureless to the grammar (the grammar
    resolver honestly cannot parse it): the D2 TEXT-WINDOW path runs, the
    D1 text-trait battery governs span-locally and at assembly, and the D3
    rtf attribute battery guards the control words — all asserted here."""
    from fastedit.data_gen.ast_analyzer import detect_language
    from fastedit.inference.chunk_locator import _text_anchor_windows
    from fastedit.inference.chunked_merge import _derived_text_op
    from fastedit.lang_attributes import format_for_path
    from fastedit.text_heuristics import (
        TOLERANCE_MODEL_PROSE,
        validate_text_output,
    )

    def check(target, final, m):
        # The control words survived byte-exact.
        assert "\\lang1036" in final and "\\lang1033" in final, m
        _assert_d3_traits(RTF, final, RTF_SNIPPET, "rtf", m)
        # The D1 text-trait battery's own predicate accepts the merge.
        op, layout_slack, removable = _derived_text_op(RTF, RTF_SNIPPET)
        ok, reason = validate_text_output(
            RTF, op, final, TOLERANCE_MODEL_PROSE,
            layout_slack=layout_slack, removable_traits=removable,
        )
        assert ok is True, f"the D1 trait gate's predicate failed: {reason} | {m}"
        # WHICH path ran: the D2 text-window path — the structureless file
        # has no AST, so the snippet's unique context lines anchored one
        # window, and the model was shown exactly that window's bytes (the
        # recording engine captures every chunk the model was asked to
        # merge; the tool layer does not surface chunk regions).
        windows = _text_anchor_windows(RTF_SNIPPET, RTF.splitlines())
        assert len(windows) == 1, f"fixture invariant: one window, got {windows}"
        (window,) = windows
        window_text = "".join(
            RTF.splitlines(keepends=True)[window[0] - 1 : window[1]],
        )
        recorded = mcp_harness.recorded_chunks
        assert recorded, f"the model never saw a chunk | {m}"
        assert window_text in recorded, (
            f"the model was not shown the anchored window {window} | {m}"
        )
        assert final == target.read_text(encoding="utf-8")

    assert detect_language("doc.rtf") is None
    assert format_for_path("doc.rtf") == "rtf"
    _mcp_edit_until_converged(
        tmp_path, "rtf-lang", RTF, RTF_SNIPPET, RTF_GOLDEN, check,
    )


# ---------------------------------------------------------------------------
# 3. docx — the adapter path through the full MCP stack (req. 7)
# ---------------------------------------------------------------------------

_DOCX_DIR = GOLDEN_DIR / "docx"
_DOCX_BYTES = (_DOCX_DIR / "report.docx").read_bytes()
_DOCX_GOLDEN_XML = (_DOCX_DIR / "expected_document.xml").read_text(encoding="utf-8")
_DOCX_SNIPPET = (
    "    <w:p>\n"
    "      <w:r>\n"
    '        <w:rPr><w:lang w:val="zh-Hans"/></w:rPr>\n'
    '        <w:t xml:space="preserve">'
    "All region data enters a temporary queue first.</w:t>\n"
    "      </w:r>\n"
    "    </w:p>\n"
)


def _zip_snapshot(data: bytes):
    """(names, {name: (payload, compress_type, date_time)}) of a container."""
    import io

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        return (
            [i.filename for i in infos],
            {
                i.filename: (zf.read(i.filename), i.compress_type, i.date_time)
                for i in infos
            },
        )


def test_real_docx_mcp_edit_preserves_container_and_undoes(
    mcp_harness, tmp_path,
):
    """(d) docx: the full MCP stack edits ``word/document.xml`` through the
    adapter — the merged document is byte-exact against the line-splice
    golden, every OTHER zip entry keeps its payload, compression method,
    timestamp and order, and one MCP undo restores the ORIGINAL container
    byte-for-byte."""
    from fastedit.lang_attributes import read_docx
    from fastedit.mcp import tools_ast

    target = tmp_path / "report.docx"
    target.write_bytes(_DOCX_BYTES)
    before_names, before = _zip_snapshot(_DOCX_BYTES)

    response = _run_mcp_edit(target, _DOCX_SNIPPET)
    assert response.startswith(f"Applied edit to {target}"), response
    assert "rejected" not in response and "Error" not in response, response

    # The document text is the declared translation, byte-exact.
    assert read_docx(target) == _DOCX_GOLDEN_XML, (
        f"the merged document differs from the golden:\n{read_docx(target)}"
    )
    # Every other entry is byte-identical (payload, compression, timestamp,
    # order); the adapter rebuilt only the changed document entry.
    after_names, after = _zip_snapshot(target.read_bytes())
    assert after_names == before_names, "zip entry order changed"
    for name in before_names:
        if name == "word/document.xml":
            continue
        assert after[name] == before[name], f"entry {name} was not preserved"

    # One undo restores the ORIGINAL container byte-for-byte (the B22
    # raw-bytes backup of the whole zip).
    response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
    assert response.startswith(f"Reverted {target}"), response
    assert target.read_bytes() == _DOCX_BYTES, (
        "the undo did not restore the original container byte-exactly"
    )
