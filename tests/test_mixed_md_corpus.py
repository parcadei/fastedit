"""Step D4 — the committed mixed-language corpus + the locator fix (hermetic).

Default-tier companion to ``tests/test_real_llm_mixed_lang.py`` (the real
model tier): everything here is scripted or pure arithmetic — no model,
no network.

What is locked down:

1. **Oracle agreement** (the B3 golden doctrine): the committed
   ``expected_translate_comments.md`` is re-derived from the authored
   ``original.md`` and the manifest's translation table with the corpus
   generator's INDEPENDENT line-splice oracle — a stale artifact fails
   loudly instead of pinning wrong behavior.
2. **Trait inventory**: the corpus's D3 language-attribute/structure
   traits match the manifest's declaration — well-formed frontmatter,
   three closed fences with their info strings, and ONE unclosed fence
   (the req. 9 malformed trait, preserved as ``(md_fence_unclosed,
   <opener line>)``).
3. **Battery coherence**: each per-comment translation op, driven through
   the REAL ``chunked_merge`` pipeline with a scripted faithful merge_fn,
   passes the whole battery (relative parse + content faithfulness + D3
   structure gate) and lands byte-exact; the four ops COMPOSE into the
   committed golden.
4. **The Step D4 locator regression** (the defect the real-model flagship
   exposed): a marker-bearing snippet anchored on a ``def`` line INSIDE a
   fenced code block of a markdown document used to be routed to a tail
   "insertion region" that does not contain the edit target — the
   language-blind definition regex fabricated a phantom "new definition"
   (``summarize``) that the markdown AST's section vocabulary could never
   match, the model was handed the wrong bytes, and the battery rejected
   every attempt (measured 9/9 against the real model) or would have
   ratified a wrong-region insertion. The fix makes the new-definition
   decision in the FILE's own symbol vocabulary
   (``_snippet_defines_new_symbols``); the regression pins that the
   snippet now lands on a region CONTAINING the edit target, and that
   code-language routing is unchanged (a genuinely new python function
   still gets an insertion region; an existing symbol still gets a tight
   chunk).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "golden" / "mixed_md"))
import _generate as mixed_md_oracle

from fastedit.inference.ast_utils import get_ast_map_from_source
from fastedit.inference.chunk_locator import locate_chunks
from fastedit.inference.chunked_merge import chunked_merge
from fastedit.inference.snippet_analysis import (
    _snippet_defines_new_symbols,
)
from fastedit.lang_attributes import extract_attributes

_MIXED_DIR = Path(__file__).resolve().parent / "golden" / "mixed_md"
ORIGINAL = (_MIXED_DIR / "original.md").read_text(encoding="utf-8")
EXPECTED = (_MIXED_DIR / "expected_translate_comments.md").read_text(
    encoding="utf-8",
)
MANIFEST = json.loads((_MIXED_DIR / "corpus.json").read_text(encoding="utf-8"))
TRANSLATIONS = MANIFEST["translations"]

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


def _snippet(zh_line: str, en_line: str) -> str:
    """The per-comment D-shape snippet (the shape measured to converge).

    The DECLARED translation line is stated BEFORE the preservation
    marker, bracketed by the two lines adjacent to the comment. The
    battery justifies the old comment's deletion positionally in the
    marker-bearing segment; the real model executes this shape faithfully
    at every loop state (measured 4/4, first attempt — see
    tests/test_real_llm_mixed_lang.py).
    """
    above, below = _COMMENT_NEIGHBOURS[zh_line]
    payload = en_line
    indent = zh_line[: len(zh_line) - len(zh_line.lstrip())]
    comment = (
        "# ... existing code ..."
        if zh_line.lstrip().startswith("#")
        else "// ... existing code ..."
    )
    return f"{above}\n{payload}\n{indent}{comment}\n{below}\n"


# ---------------------------------------------------------------------------
# 1. Oracle agreement + trait inventory
# ---------------------------------------------------------------------------


def test_committed_expected_agrees_with_the_line_splice_oracle():
    """The committed golden is exactly the manifest's declared line swaps."""
    expected_lines = mixed_md_oracle.oracle_swap_lines(
        ORIGINAL.splitlines(keepends=True), TRANSLATIONS,
    )
    assert "".join(expected_lines) == EXPECTED, (
        "committed expected_translate_comments.md drifted from the "
        "translation table — regenerate (tests/golden/mixed_md/_generate.py)"
    )
    # And the oracle's own identity checks hold on the committed artifact.
    assert EXPECTED != ORIGINAL
    assert len(EXPECTED.splitlines()) == len(ORIGINAL.splitlines())


def test_corpus_trait_inventory_matches_the_manifest():
    """The D3 traits of the fixture are exactly the manifest's declaration."""
    traits = extract_attributes(ORIGINAL, "markdown")
    keys = [t.key for t in traits]
    frontmatter = MANIFEST["traits"]["frontmatter"]
    assert keys[:3] == [
        ("md_frontmatter_delim", "---"),
        ("md_frontmatter_key", frontmatter["keys"][0]),
        ("md_frontmatter_key", frontmatter["keys"][1]),
    ], keys
    assert keys[3] == ("md_frontmatter_delim", "---")
    # Fence info strings, in document order, with the malformed trait last.
    infos = [v for k, v in keys if k == "md_fence_info"]
    assert infos == MANIFEST["traits"]["fence_info_strings"]
    assert keys[-1] == ("md_fence_unclosed", MANIFEST["traits"]["unclosed_fence_opener"])
    # The mixed-language content traits: CJK paragraphs and the embedded
    # html widget's language attribute are present verbatim.
    assert MANIFEST["traits"]["embedded_html_lang"] in ORIGINAL
    assert "遗留问题" in ORIGINAL and "。" in ORIGINAL


# ---------------------------------------------------------------------------
# 2. Battery coherence: each op lands byte-exact through the real pipeline
# ---------------------------------------------------------------------------


def _scripted_merge_fn(results: list[str]):
    """A merge_fn returning MergeResult-shaped scripted payloads."""
    calls = []

    def merge_fn(code, snip, lang):
        from types import SimpleNamespace

        calls.append((code, snip, lang))
        payload = results[min(len(calls) - 1, len(results) - 1)]
        return SimpleNamespace(
            merged_code=payload,
            tokens_generated=7,
            latency_ms=1.0,
            truncated=False,
        )

    return merge_fn, calls


@pytest.mark.parametrize("index", range(len(TRANSLATIONS)))
def test_each_translation_op_passes_the_battery_and_lands_byte_exact(
    tmp_path, index,
):
    """One per-comment op through the REAL pipeline (scripted merge_fn).

    The snippet anchors no AST node, so the pipeline routes the op through
    the whole-file merge branch, where the battery judges the FULL file
    (relative markdown parse, content faithfulness, D3 structure gate).
    The scripted merge returns the per-comment golden — exactly what a
    faithful model merge looks like. The op must be ACCEPTED on the first
    attempt (no rejection bookkeeping) and land byte-exact.
    """
    entry = TRANSLATIONS[index]
    golden = ORIGINAL.replace(entry["zh"] + "\n", entry["en"] + "\n")
    assert golden != ORIGINAL
    merge_fn, calls = _scripted_merge_fn([golden])
    target = tmp_path / "notes.md"
    target.write_text(ORIGINAL, encoding="utf-8")

    result = chunked_merge(
        original_code=ORIGINAL,
        snippet=_snippet(entry["zh"], entry["en"]),
        file_path=str(target),
        merge_fn=merge_fn,
        language="markdown",
    )
    assert result.chunks_rejected == 0, result.merged_code
    assert result.parse_valid is True
    assert result.retries == 0
    assert result.merged_code == golden
    # The model path ran (whole-file merge branch) — one scripted call.
    assert len(calls) == 1
    assert calls[0][0] == ORIGINAL


def test_translations_compose_into_the_committed_golden():
    """The four per-comment goldens chain into the committed expected file."""
    state = ORIGINAL
    for entry in TRANSLATIONS:
        state = state.replace(entry["zh"] + "\n", entry["en"] + "\n")
        assert state != ORIGINAL or entry is TRANSLATIONS[0]
    assert state == EXPECTED


# ---------------------------------------------------------------------------
# 3. The Step D4 locator regression (the flagship's exposed defect)
# ---------------------------------------------------------------------------

_PDef_SNIPPET = (
    "def summarize(users):\n"
    "    # ... existing code ...\n"
    "    # Iterate all users.\n"
    "    total = 0\n"
)


def test_markdown_comment_snippet_is_not_routed_to_a_tail_insertion_region():
    """THE D4 regression: the chunk must CONTAIN the edit target.

    Before the fix, ``locate_chunks`` routed this snippet to the file's
    last two sections (the "insertion region" for a phantom new python
    definition) — a region that does not contain the fenced block the
    snippet edits, so a compliant model merge would have inserted the
    snippet's lines into the WRONG section (battery-ratified corruption)
    and a faithful one was rejected 9/9.
    """
    chunks = locate_chunks(_PDef_SNIPPET, ORIGINAL, "notes.md", 30, "markdown")
    assert len(chunks) == 1
    start, end = chunks[0].start_line, chunks[0].end_line
    lines = ORIGINAL.splitlines()
    # The region must cover the fenced python block the snippet edits.
    block_start = lines.index("```python") + 1  # 1-indexed line of the fence
    block_end = lines.index("```", block_start) + 1
    assert start <= block_start and end >= block_end, (
        f"the chunk {start}-{end} does not contain the edit target "
        f"(the fenced block at {block_start}-{block_end})"
    )


def test_new_definition_decision_uses_the_file_vocabulary():
    """The fix's unit: no phantom definitions from foreign-vocabulary lines."""
    existing = {
        n.name
        for n in get_ast_map_from_source(ORIGINAL, "notes.md", "markdown")
    }
    # The snippet's `def summarize(users):` line is a python definition
    # inside a markdown fenced block — NOT a new markdown section.
    assert _snippet_defines_new_symbols(_PDef_SNIPPET, "markdown", existing) is False
    # A snippet that genuinely adds a markdown section IS a new definition.
    new_section_snippet = "# A brand new section\nSome text.\n"
    assert _snippet_defines_new_symbols(
        new_section_snippet, "markdown", existing,
    ) is True
    # Grammar-less languages keep the legacy regex answer.
    assert _snippet_defines_new_symbols(
        _PDef_SNIPPET, None, {"something"},
    ) is True


def test_code_language_routing_is_unchanged_by_the_vocabulary_fix():
    """The fix must not disturb code-language chunking (python here)."""
    py_source = 'def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n'
    # A genuinely new function still gets the insertion region.
    chunks = locate_chunks(
        "def gamma():\n    return 3\n", py_source, "m.py", 30, "python",
    )
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 6)]
    # An existing symbol still gets its tight chunk.
    chunks = locate_chunks(
        "def beta():\n    # ... existing code ...\n    return 9\n",
        py_source, "m.py", 30, "python",
    )
    assert [(c.start_line, c.end_line, c.matched_nodes) for c in chunks] == [
        (5, 6, ["beta"]),
    ]
