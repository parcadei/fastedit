"""Step D3 — language-attribute & structure-trait framework (req. 7 + 9).

Five product behaviors are locked down here:

1. ``fastedit.lang_attributes.extract_attributes`` — the declarative
   per-format spec table (one row per format, no if-chains): html
   ``lang``/``hreflang``, xml ``xml:lang``, latex babel commands, rtf
   ``\\langNNNN``/``\\langfeNNNN``, docx ``w:lang`` elements, md fence info
   strings + fence lengths + frontmatter. Extraction is AST-assisted where
   a grammar resolves (html/xml/docx) with a REGEX FALLBACK that must work
   without any grammar — both paths pinned, precedence documented by test.
2. MALFORMED traits (req. 9): ``lang=""``, a bare ``<div lang>``, an
   unclosed latex brace, ``\\lang`` with no number, a ``w:lang`` without
   ``w:val``, an unclosed md fence, frontmatter missing its closing
   ``---`` — all extracted AS-IS (value = raw source text where there is
   no value slot) so they can be PRESERVED as-is.
3. ``attributes_match_expectation`` — the trait oracle: every input trait
   must appear in the output verbatim at the same relative position,
   UNLESS the declared changes explicitly touch it (then the declared new
   value must appear exactly). Preserved / declared-change / undeclared
   change→FAIL / undeclared introduction→FAIL / justified removal all
   pinned, including the md structure family (fence state, frontmatter).
4. The docx adapter — stdlib ``zipfile``: ``read_docx`` extracts
   ``word/document.xml``; ``build_docx_bytes``/``write_docx`` rewrite the
   container preserving every other entry (payload bytes, compression
   method, timestamps, entry order). Corrupt zips / missing entries /
   non-UTF-8 XML fail loud (:class:`DocxError`).
5. Battery integration — the attribute/structure gate is one more branch
   of ``_merge_rejection_reason`` (beside the D1 text-trait gate), fed by
   the snippet-derived declared changes (``_derived_attribute_changes``,
   the D3 mirror of ``_derived_text_op``) and threaded by file suffix
   (``format_for_path``). Scripted merge_fn (hermetic — no model): a
   declared attribute change passes first try, an undeclared structure
   mutation is rejected to exhaustion with the attribute gate's own
   reason, the md flagship (malformed frontmatter + body edit) passes
   byte-exact, and an MCP-level docx edit round-trips through the real
   write path (real zip rebuild, real backup, undo → original bytes).

Relationship to the content gate (documented, tested): the battery's
content validator is LINE-exact — every output line is a byte-exact
survivor or an exact declared new line — so line-level attribute value
drift is already rejected there. The D3 gate is the SEMANTIC layer on
top: it names the attribute/structure trait in the retry note, enforces
the fence/frontmatter STATE (identity-free structure lines can be
deleted by positional capacity the content validator grants), and is the
only ASSEMBLY-level exact check a multi-window md/html/xml edit has.

Everything in this file is default-tier hermetic: scripted merges, real
tree-sitter grammars, real zip files on disk. No model, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import zipfile
from collections import defaultdict
from types import SimpleNamespace

import pytest

from fastedit.lang_attributes import (
    AttributeTrait,
    DocxError,
    attributes_match_expectation,
    build_docx_bytes,
    declared_changes_from_text,
    extract_attributes,
    format_for_path,
    read_docx,
    write_docx,
)

# ---------------------------------------------------------------------------
# Fixtures — every trait expectation is hand-derived from the spec table.
# ---------------------------------------------------------------------------

HTML_ORIGINAL = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    "  <title>Lab notes</title>\n"
    "</head>\n"
    "<body>\n"
    '  <p lang="en">First paragraph.</p>\n'
    '  <a href="https://example.com" hreflang="fr">archive</a>\n'
    "</body>\n"
    "</html>\n"
)

HTML_LANG_CHANGED = HTML_ORIGINAL.replace('lang="en"', 'lang="de"')
HTML_LANG_DECLARED = HTML_ORIGINAL.replace(
    '  <p lang="en">First paragraph.</p>',
    '  <p lang="fr">First paragraph.</p>',
)

XML_ORIGINAL = (
    '<?xml version="1.0"?>\n'
    '<catalog xml:lang="de">\n'
    '  <book id="b1" xml:lang="fr">\n'
    "    <title>T1</title>\n"
    "  </book>\n"
    "</catalog>\n"
)

LATEX_WELL_FORMED = (
    "\\documentclass{article}\n"
    "\\selectlanguage{french}\n"
    "Text \\foreignlanguage{german}{der Inhalt} more.\n"
    "\\begin{otherlanguage}{russian}\n"
    "body\n"
    "\\end{otherlanguage}\n"
    "\\begin{otherlanguage*}{greek}\n"
    "body\n"
    "\\end{otherlanguage*}\n"
)

LATEX_MALFORMED = (
    "\\selectlanguage{french\n"  # unclosed brace — malformed, preserved as-is
    "\\begin{otherlanguage}\n"   # missing language argument — malformed
)

RTF_WELL_FORMED = "{\\rtf1\\ansi\\lang1033\\langfe2057 Hello.\\par\n}\n"
RTF_MALFORMED = "{\\rtf1\\lang Hello.\\par\n}\n"

MD_FENCES = (
    "# Title\n"
    "\n"
    "```python\n"
    "print('hi')\n"
    "```\n"
    "\n"
    "````js\n"
    "const x = 1;\n"
    "````\n"
    "\n"
    "~~~\n"
    "tilde block\n"  # unclosed at EOF — the unclosed state is a trait
)

MD_FENCE_UNCLOSED = "```python\nprint('hi')\n"

MD_FRONTMATTER = (
    "---\n"
    "title: Lab Notes\n"
    "tags: [unclosed\n"  # malformed key line — a trait, preserved verbatim
    "---\n"
    "# Heading\n"
    "\n"
    "Body paragraph one.\n"
)

MD_FRONTMATTER_UNCLOSED = (
    "---\n"  # opener...
    "title: Lab Notes\n"
    "Body paragraph one.\n"  # ...and NO closing delimiter: malformed
)

DOCX_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
    "  <w:body>\n"
    "    <w:p>\n"
    "      <w:r>\n"
    '        <w:rPr><w:lang w:val="en-US"/></w:rPr>\n'
    "        <w:t>Hello laboratory.</w:t>\n"
    "      </w:r>\n"
    "    </w:p>\n"
    "  </w:body>\n"
    "</w:document>\n"
)

DOCX_DOCUMENT_MALFORMED = (
    '<?xml version="1.0"?>\n'
    "<w:document>\n"
    '  <w:p><w:rPr><w:lang w:eastAsia="zh"/></w:rPr></w:p>\n'
    "</w:document>\n"
)


def _keys(traits):
    """The (kind, value) identity sequence of a trait list."""
    return [t.key for t in traits]


def _no_grammar(monkeypatch):
    """Force the regex fallback: the AST-assisted path sees no grammar."""
    from fastedit.data_gen import ast_analyzer

    def _raise(*_a, **_kw):
        raise ast_analyzer.GrammarUnavailableError("html", "no grammar (test)")

    monkeypatch.setattr(ast_analyzer, "parse_code", _raise)


# ---------------------------------------------------------------------------
# 1. extract_attributes — html (AST-assisted, regex fallback, precedence)
# ---------------------------------------------------------------------------


class TestExtractHtml:
    def test_wellformed_ast_path(self):
        traits = extract_attributes(HTML_ORIGINAL, "html")
        assert _keys(traits) == [
            ("html_lang", "en"),
            ("html_lang", "en"),
            ("html_hreflang", "fr"),
        ]

    def test_trait_spans_and_raw(self):
        traits = extract_attributes(HTML_ORIGINAL, "html")
        lang = traits[0]
        assert isinstance(lang, AttributeTrait)
        assert lang.raw == 'lang="en"'
        assert HTML_ORIGINAL.encode()[lang.start:lang.end].decode() == 'lang="en"'
        href = traits[2]
        assert href.raw == 'hreflang="fr"'
        assert HTML_ORIGINAL.encode()[href.start:href.end].decode() == 'hreflang="fr"'

    def test_value_quote_styles(self):
        text = "<a lang='fr'>x</a>\n<p lang=de>y</p>\n"
        assert _keys(extract_attributes(text, "html")) == [
            ("html_lang", "fr"),
            ("html_lang", "de"),
        ]

    def test_regex_fallback_matches_ast_identity(self, monkeypatch):
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(HTML_ORIGINAL, "html")) == [
            ("html_lang", "en"),
            ("html_lang", "en"),
            ("html_hreflang", "fr"),
        ]

    def test_ast_precedence_ignores_script_text(self, monkeypatch):
        # THE precedence payoff: `el.lang = "fr"` inside a <script> block is
        # prose, not an attribute. The AST path (default) knows that; the
        # regex fallback (no grammar) cannot and reports it. Both behaviors
        # pinned so the precedence can never silently flip.
        text = (
            '<html lang="en">\n'
            "<script>\n"
            '  el.lang = "fr";\n'
            "</script>\n"
            "</html>\n"
        )
        assert _keys(extract_attributes(text, "html")) == [("html_lang", "en")]
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(text, "html")) == [
            ("html_lang", "en"),
            ("html_lang", "fr"),
        ]

    def test_malformed_empty_value_preserved_as_trait(self, monkeypatch):
        # req. 9: lang="" is extracted AS-IS (empty value is still a trait).
        text = '<p lang="">x</p>\n'
        assert _keys(extract_attributes(text, "html")) == [("html_lang", "")]
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(text, "html")) == [("html_lang", "")]

    def test_malformed_bare_attribute_preserved(self, monkeypatch):
        # `<div lang>` — no `=` at all: the AST reports the attribute with an
        # empty value; the regex fallback has a dedicated malformed row.
        text = "<div lang>x</div>\n"
        assert _keys(extract_attributes(text, "html")) == [("html_lang", "")]
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(text, "html")) == [("html_lang", "")]

    def test_xml_lang_not_an_html_trait(self, monkeypatch):
        text = '<body xml:lang="de">x</body>\n'
        assert extract_attributes(text, "html") == []
        _no_grammar(monkeypatch)
        assert extract_attributes(text, "html") == []

    def test_unknown_format_is_inert(self):
        assert extract_attributes(HTML_ORIGINAL, "sometxt") == []


# ---------------------------------------------------------------------------
# 2. extract_attributes — xml
# ---------------------------------------------------------------------------


class TestExtractXml:
    def test_wellformed_ast_path(self):
        assert _keys(extract_attributes(XML_ORIGINAL, "xml")) == [
            ("xml_lang", "de"),
            ("xml_lang", "fr"),
        ]

    def test_regex_fallback_matches_ast_identity(self, monkeypatch):
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(XML_ORIGINAL, "xml")) == [
            ("xml_lang", "de"),
            ("xml_lang", "fr"),
        ]

    def test_plain_lang_is_not_an_xml_trait(self, monkeypatch):
        text = '<catalog lang="de">x</catalog>\n'
        assert extract_attributes(text, "xml") == []
        _no_grammar(monkeypatch)
        assert extract_attributes(text, "xml") == []

    def test_malformed_empty_value_preserved(self, monkeypatch):
        text = "<catalog xml:lang=''>x</catalog>\n"
        assert _keys(extract_attributes(text, "xml")) == [("xml_lang", "")]
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(text, "xml")) == [("xml_lang", "")]


# ---------------------------------------------------------------------------
# 3. extract_attributes — latex (regex, line-oriented)
# ---------------------------------------------------------------------------


class TestExtractLatex:
    def test_babel_commands(self):
        assert _keys(extract_attributes(LATEX_WELL_FORMED, "latex")) == [
            ("latex_selectlanguage", "french"),
            ("latex_foreignlanguage", "german"),
            ("latex_otherlanguage", "russian"),
            ("latex_otherlanguage", "greek"),
        ]

    def test_malformed_traits_preserved_as_raw(self):
        traits = extract_attributes(LATEX_MALFORMED, "latex")
        assert _keys(traits) == [
            ("latex_selectlanguage", "\\selectlanguage{french"),
            ("latex_otherlanguage", "\\begin{otherlanguage}"),
        ]
        # The raw source is the identity: any byte change to the malformed
        # construct is a trait change (req. 9).
        assert traits[0].raw == "\\selectlanguage{french"

    def test_end_environment_is_not_an_opener(self):
        text = "\\begin{otherlanguage}{russian}\n\\end{otherlanguage}{x}\n"
        assert _keys(extract_attributes(text, "latex")) == [
            ("latex_otherlanguage", "russian"),
        ]

    def test_regex_only_no_grammar_dependency(self, monkeypatch):
        # latex/rtf/md declare regex rows only — they must work with no
        # grammar resolver at all (the B1 resolver is never consulted).
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(LATEX_WELL_FORMED, "latex")) == [
            ("latex_selectlanguage", "french"),
            ("latex_foreignlanguage", "german"),
            ("latex_otherlanguage", "russian"),
            ("latex_otherlanguage", "greek"),
        ]


# ---------------------------------------------------------------------------
# 4. extract_attributes — rtf
# ---------------------------------------------------------------------------


class TestExtractRtf:
    def test_lang_and_langfe(self):
        assert _keys(extract_attributes(RTF_WELL_FORMED, "rtf")) == [
            ("rtf_lang", "1033"),
            ("rtf_langfe", "2057"),
        ]

    def test_malformed_lang_without_number_preserved_as_raw(self):
        traits = extract_attributes(RTF_MALFORMED, "rtf")
        assert _keys(traits) == [("rtf_lang", "\\lang")]
        assert traits[0].raw == "\\lang"

    def test_langfe_is_not_double_counted_as_lang(self):
        # \langfe2057 must feed only the \langfe row.
        text = "{\\rtf1\\langfe2057 x\\par\n}\n"
        assert _keys(extract_attributes(text, "rtf")) == [("rtf_langfe", "2057")]


# ---------------------------------------------------------------------------
# 5. extract_attributes — md fences (structure traits)
# ---------------------------------------------------------------------------


class TestMarkdownFences:
    def test_info_string_and_fence_length_are_traits(self):
        assert _keys(extract_attributes(MD_FENCES, "markdown")) == [
            ("md_fence_info", "python"),
            ("md_fence_marker", "```"),
            ("md_fence_info", "js"),
            ("md_fence_marker", "````"),
            ("md_fence_info", ""),
            ("md_fence_marker", "~~~"),
            ("md_fence_unclosed", "~~~"),
        ]

    def test_unclosed_fence_trait(self):
        traits = extract_attributes(MD_FENCE_UNCLOSED, "markdown")
        assert _keys(traits) == [
            ("md_fence_info", "python"),
            ("md_fence_marker", "```"),
            ("md_fence_unclosed", "```python"),
        ]

    def test_fragment_mode_suppresses_unclosed_state(self):
        # A snippet/fragment is truncated by construction: its trailing
        # unclosed fence says nothing about the document's fence state.
        traits = extract_attributes(MD_FENCE_UNCLOSED, "markdown", fragment=True)
        assert _keys(traits) == [
            ("md_fence_info", "python"),
            ("md_fence_marker", "```"),
        ]

    def test_fragment_mode_emits_only_informative_fences(self):
        # In a fragment an empty-info fence line is ambiguous (opener in the
        # document, closer in the document) — it declares nothing. An info
        # string makes the opener definite.
        text = "```\n```python\n"
        assert _keys(extract_attributes(text, "markdown", fragment=True)) == [
            ("md_fence_info", "python"),
            ("md_fence_marker", "```"),
        ]

    def test_backtick_fence_with_backtick_info_is_not_a_fence(self):
        # CommonMark: a backtick info string may not contain backticks —
        # such a line is text, toggles no fence state; the later bare ```)
        # is then an (unclosed) opener of its own.
        text = "```python `tpl`\ncode\n```\n"
        assert _keys(extract_attributes(text, "markdown")) == [
            ("md_fence_info", ""),
            ("md_fence_marker", "```"),
            ("md_fence_unclosed", "```"),
        ]

    def test_closing_fence_with_info_does_not_close(self):
        # A "closing" fence carrying an info string is fence CONTENT.
        text = "```python\ncode\n``` not a closer\nmore\n"
        traits = extract_attributes(text, "markdown")
        assert ("md_fence_unclosed", "```python") in _keys(traits)

    def test_shorter_closing_fence_does_not_close(self):
        text = "````python\ncode\n```\nmore\n"
        traits = extract_attributes(text, "markdown")
        assert ("md_fence_unclosed", "````python") in _keys(traits)


# ---------------------------------------------------------------------------
# 6. extract_attributes — md frontmatter (structure traits)
# ---------------------------------------------------------------------------


class TestMarkdownFrontmatter:
    def test_delimiters_and_key_lines_are_traits(self):
        assert _keys(extract_attributes(MD_FRONTMATTER, "markdown")) == [
            ("md_frontmatter_delim", "---"),
            ("md_frontmatter_key", "title: Lab Notes"),
            ("md_frontmatter_key", "tags: [unclosed"),
            ("md_frontmatter_delim", "---"),
        ]

    def test_malformed_key_line_preserved_verbatim(self):
        # req. 9 flagship: `tags: [unclosed` is a trait whose value IS the
        # raw line — any byte change is a trait change.
        traits = extract_attributes(MD_FRONTMATTER, "markdown")
        key = traits[2]
        assert key.kind == "md_frontmatter_key"
        assert key.value == "tags: [unclosed"
        assert key.raw == "tags: [unclosed"

    def test_missing_closing_delimiter_is_a_trait(self):
        traits = extract_attributes(MD_FRONTMATTER_UNCLOSED, "markdown")
        assert _keys(traits) == [
            ("md_frontmatter_delim", "---"),
            ("md_frontmatter_unclosed", "---"),
        ]

    def test_unclosed_block_yields_no_key_traits(self):
        # With no closing delimiter the block extent is unknowable — key
        # lines are NOT extracted (a body edit must not become a phantom
        # frontmatter trait). The malformed STATE itself is the trait.
        keys = [
            t for t in extract_attributes(MD_FRONTMATTER_UNCLOSED, "markdown")
            if t.kind == "md_frontmatter_key"
        ]
        assert keys == []

    def test_close_delimiter_may_be_dots(self):
        text = "---\ntitle: x\n...\n"
        assert _keys(extract_attributes(text, "markdown")) == [
            ("md_frontmatter_delim", "---"),
            ("md_frontmatter_key", "title: x"),
            ("md_frontmatter_delim", "..."),
        ]

    def test_no_frontmatter_without_leading_delimiter(self):
        text = "# Heading\nkey: value\n"
        assert extract_attributes(text, "markdown") == []

    def test_fragment_key_shaped_lines_declare_keys(self):
        # A snippet that edits one frontmatter key rarely restates the whole
        # block: a key-SHAPED declared line counts as a declared key trait.
        text = "title: New Title\n"
        assert _keys(extract_attributes(text, "markdown", fragment=True)) == [
            ("md_frontmatter_key", "title: New Title"),
        ]
        # Prose with a colon is not key-shaped.
        assert extract_attributes("See page 12: details.\n", "markdown",
                                  fragment=True) == []


# ---------------------------------------------------------------------------
# 7. extract_attributes — docx (w:lang elements)
# ---------------------------------------------------------------------------


class TestExtractDocx:
    def test_w_lang_val(self):
        assert _keys(extract_attributes(DOCX_DOCUMENT, "docx")) == [
            ("docx_lang", "en-US"),
        ]

    def test_regex_fallback_matches_ast_identity(self, monkeypatch):
        _no_grammar(monkeypatch)
        assert _keys(extract_attributes(DOCX_DOCUMENT, "docx")) == [
            ("docx_lang", "en-US"),
        ]

    def test_malformed_w_lang_without_val_preserved_as_raw(self, monkeypatch):
        traits = extract_attributes(DOCX_DOCUMENT_MALFORMED, "docx")
        assert len(traits) == 1
        assert traits[0].kind == "docx_lang"
        assert traits[0].value == traits[0].raw
        assert traits[0].raw == '<w:lang w:eastAsia="zh"/>'
        _no_grammar(monkeypatch)
        traits = extract_attributes(DOCX_DOCUMENT_MALFORMED, "docx")
        assert traits[0].value == traits[0].raw

    def test_other_elements_are_not_traits(self):
        text = "<w:document><w:body/></w:document>\n"
        assert extract_attributes(text, "docx") == []


# ---------------------------------------------------------------------------
# 8. attributes_match_expectation — attribute traits
# ---------------------------------------------------------------------------


class TestMatchAttributes:
    def test_identical_output_passes(self):
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, HTML_ORIGINAL, "html",
        )
        assert ok is True and reason == ""

    def test_declared_change_passes(self):
        declared = declared_changes_from_text(
            '  <p lang="fr">First paragraph.</p>\n', "", "html",
        )
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, HTML_LANG_DECLARED, "html", declared,
        )
        assert ok is True, reason

    def test_undeclared_change_fails_with_reason(self):
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, HTML_LANG_CHANGED, "html",
        )
        assert ok is False
        assert "html_lang" in reason and "'en'" in reason

    def test_undeclared_introduction_fails(self):
        output = HTML_ORIGINAL.replace(
            "<body>\n", '<body>\n  <p lang="de">neu</p>\n',
        )
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, output, "html",
        )
        assert ok is False
        assert "html_lang" in reason and "'de'" in reason

    def test_model_applied_a_different_value_than_declared_fails(self):
        # THE value-level catch: the snippet declared lang="fr"; the model
        # applied lang="de" inside an equally keyed replacement. The line is
        # "justified" at key granularity — the trait gate sees the value.
        declared = declared_changes_from_text(
            '  <p lang="fr">First paragraph.</p>\n', "", "html",
        )
        output = HTML_ORIGINAL.replace(
            '  <p lang="en">First paragraph.</p>',
            '  <p lang="de">First paragraph.</p>',
        )
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, output, "html", declared,
        )
        assert ok is False
        assert "html_lang" in reason and "'de'" in reason

    def test_declared_addition_must_land(self):
        # The snippet declares a lang="fr" paragraph; the model dropped the
        # declared line. The declared value must appear exactly — fail.
        declared = declared_changes_from_text('<p lang="fr">x</p>\n', "", "html")
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, HTML_ORIGINAL, "html", declared,
        )
        assert ok is False
        assert "html_lang" in reason and "'fr'" in reason

    def test_declared_addition_that_lands_passes(self):
        declared = declared_changes_from_text('<p lang="fr">x</p>\n', "", "html")
        output = HTML_ORIGINAL.replace(
            "<body>\n", '<body>\n  <p lang="fr">x</p>\n',
        )
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, output, "html", declared,
        )
        assert ok is True, reason

    def test_relative_order_is_enforced(self):
        # Both traits survive but swapped: a reorder is a corruption. The
        # p-line's lang trait now comes AFTER the hreflang one.
        output = (
            "<!DOCTYPE html>\n"
            "<head>\n"
            "  <title>Lab notes</title>\n"
            '  <html lang="en">\n'
            "</head>\n"
            "<body>\n"
            '  <a href="https://example.com" hreflang="fr">archive</a>\n'
            '  <p lang="en">First paragraph.</p>\n'
            "</body>\n"
            "</html>\n"
        )
        ok, reason = attributes_match_expectation(HTML_ORIGINAL, output, "html")
        assert ok is False
        assert "html_" in reason

    def test_justified_removal_passes(self):
        # The op's shape justifies removing the line that carries the
        # hreflang trait (removable side of DeclaredChanges): the trait may
        # vanish without a replacement.
        declared = declared_changes_from_text(
            "", '  <a href="https://example.com" hreflang="fr">archive</a>\n',
            "html",
        )
        output = HTML_ORIGINAL.replace(
            '  <a href="https://example.com" hreflang="fr">archive</a>\n', "",
        )
        ok, reason = attributes_match_expectation(
            HTML_ORIGINAL, output, "html", declared,
        )
        assert ok is True, reason

    def test_unjustified_removal_fails(self):
        output = HTML_ORIGINAL.replace(
            '  <a href="https://example.com" hreflang="fr">archive</a>\n', "",
        )
        ok, reason = attributes_match_expectation(HTML_ORIGINAL, output, "html")
        assert ok is False
        assert "html_hreflang" in reason

    def test_rtf_declared_change_passes_and_undeclared_fails(self):
        declared = declared_changes_from_text("\\lang2057\n", "", "rtf")
        output = RTF_WELL_FORMED.replace("\\lang1033", "\\lang2057")
        ok, reason = attributes_match_expectation(
            RTF_WELL_FORMED, output, "rtf", declared,
        )
        assert ok is True, reason
        undeclared = RTF_WELL_FORMED.replace("\\lang1033", "\\lang1036")
        ok, reason = attributes_match_expectation(
            RTF_WELL_FORMED, undeclared, "rtf",
        )
        assert ok is False
        assert "rtf_lang" in reason

    def test_malformed_trait_mutation_fails(self):
        # A "helpful fix" of the malformed \lang (adding a number) is a
        # trait mutation — rejected like any other unfaithful edit.
        ok, _ = attributes_match_expectation(
            RTF_MALFORMED, RTF_MALFORMED.replace("\\lang ", "\\lang1033 "), "rtf",
        )
        assert ok is False

    def test_malformed_trait_preserved_passes(self):
        ok, reason = attributes_match_expectation(
            RTF_MALFORMED, RTF_MALFORMED, "rtf",
        )
        assert ok is True, reason

    def test_unknown_format_is_inert(self):
        ok, reason = attributes_match_expectation(
            "anything", "changed", "sometxt",
        )
        assert ok is True and reason == ""


# ---------------------------------------------------------------------------
# 9. attributes_match_expectation — md structure family
# ---------------------------------------------------------------------------


class TestMatchMarkdownStructure:
    def test_flagship_malformed_frontmatter_survives_body_edit(self):
        # req. 9: malformed frontmatter + a body edit → the frontmatter is
        # preserved malformed byte-exact and the edit PASSES.
        output = MD_FRONTMATTER.replace(
            "Body paragraph one.\n",
            "Body paragraph one.\nBody paragraph two.\n",
        )
        declared = declared_changes_from_text(
            "Body paragraph two.\n", "", "markdown",
        )
        ok, reason = attributes_match_expectation(
            MD_FRONTMATTER, output, "markdown", declared,
        )
        assert ok is True, reason

    def test_frontmatter_key_mutation_fails(self):
        # A model that "helpfully fixes" tags: [unclosed is a trait mutation.
        output = MD_FRONTMATTER.replace("tags: [unclosed", "tags: []")
        ok, reason = attributes_match_expectation(
            MD_FRONTMATTER, output, "markdown",
        )
        assert ok is False
        assert "md_frontmatter_key" in reason

    def test_declared_frontmatter_replacement_passes(self):
        # D5(c) shape: the command replaces the malformed frontmatter with a
        # well-formed block; the declared block must land exactly.
        block = "---\ntitle: Well Formed\ndate: 2024-01-01\n---\n"
        declared = declared_changes_from_text(block, "", "markdown")
        rest = MD_FRONTMATTER_UNCLOSED.split("\n", 2)[2]
        output = block + rest
        ok, reason = attributes_match_expectation(
            MD_FRONTMATTER_UNCLOSED, output, "markdown", declared,
        )
        assert ok is True, reason

    def test_declared_frontmatter_fix_that_did_not_land_fails(self):
        # The declared well-formed block was NOT applied (still unclosed):
        # the declared key lines are missing from the output — retry.
        block = "---\ntitle: Well Formed\ndate: 2024-01-01\n---\n"
        declared = declared_changes_from_text(block, "", "markdown")
        ok, reason = attributes_match_expectation(
            MD_FRONTMATTER_UNCLOSED, MD_FRONTMATTER_UNCLOSED, "markdown", declared,
        )
        assert ok is False
        assert "md_frontmatter_key" in reason

    def test_unclosed_fence_may_be_closed_only_by_declaration(self):
        # Closing a pre-existing unclosed fence without declaration is a
        # structure mutation; with the declared fence line it is the op.
        closed = MD_FENCE_UNCLOSED + "```\n"
        ok, reason = attributes_match_expectation(
            MD_FENCE_UNCLOSED, closed, "markdown",
        )
        assert ok is False
        assert "md_fence_unclosed" in reason
        declared = declared_changes_from_text("```\n", "", "markdown")
        ok, reason = attributes_match_expectation(
            MD_FENCE_UNCLOSED, closed, "markdown", declared,
        )
        assert ok is True, reason

    def test_deleting_a_closer_fails_without_declaration(self):
        output = MD_FENCES.replace("````\n", "", 1)
        ok, _reason = attributes_match_expectation(
            MD_FENCES, output, "markdown",
        )
        assert ok is False

    def test_fence_info_change_fails(self):
        output = MD_FENCES.replace("```python", "```ruby")
        ok, reason = attributes_match_expectation(MD_FENCES, output, "markdown")
        assert ok is False
        assert "md_fence_info" in reason

    def test_declared_fence_info_change_passes(self):
        declared = declared_changes_from_text("```ruby\n", "", "markdown")
        output = MD_FENCES.replace("```python", "```ruby")
        ok, reason = attributes_match_expectation(
            MD_FENCES, output, "markdown", declared,
        )
        assert ok is True, reason


# ---------------------------------------------------------------------------
# 10. format_for_path — the suffix → spec-format table
# ---------------------------------------------------------------------------


class TestFormatForPath:
    @pytest.mark.parametrize(
        ("path", "fmt"),
        [
            ("page.html", "html"),
            ("page.HTML", "html"),
            ("page.htm", "html"),
            ("doc.xml", "xml"),
            ("image.svg", "xml"),
            ("notes.md", "markdown"),
            ("notes.markdown", "markdown"),
            ("main.tex", "latex"),
            ("main.ltx", "latex"),
            ("doc.rtf", "rtf"),
            ("report.docx", "docx"),
            ("module.py", None),
            ("notes.txt", None),
            ("data.json", None),
        ],
    )
    def test_table(self, path, fmt):
        assert format_for_path(path) == fmt


# ---------------------------------------------------------------------------
# 11. The docx adapter — stdlib zip, byte-preserving rewrite
# ---------------------------------------------------------------------------

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    "<Default Extension='rels' ContentType='x'/>"
    "<Default Extension='xml' ContentType='y'/></Types>\n"
)
_RELS = '<?Relationships xmlns="http://x"><Relationship Id="r1"/></Relationships>\n'
_STYLES = "<w:styles xmlns:w='http://w'></w:styles>\n"


def _build_docx(path, document_xml=DOCX_DOCUMENT):
    """A small but real .docx container: mixed compression methods.

    Timestamps use even seconds — the DOS date field inside a zip stores
    seconds at 2-second granularity, so an odd second does not round-trip.
    """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)  # deflate
        stored = zipfile.ZipInfo("_rels/.rels", date_time=(2024, 1, 2, 3, 4, 4))
        stored.compress_type = zipfile.ZIP_STORED
        zf.writestr(stored, _RELS)
        zf.writestr("word/document.xml", document_xml)  # deflate
        styles = zipfile.ZipInfo("word/styles.xml", date_time=(2024, 6, 7, 8, 9, 10))
        styles.compress_type = zipfile.ZIP_STORED
        zf.writestr(styles, _STYLES)
    return path


def _zip_snapshot(path):
    """(names, {name: (payload, compress_type, date_time)}) of a container."""
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        return (
            [i.filename for i in infos],
            {
                i.filename: (
                    zf.read(i.filename),
                    i.compress_type,
                    i.date_time,
                )
                for i in infos
            },
        )


class TestDocxAdapter:
    def test_read_docx_extracts_document_xml(self, tmp_path):
        path = _build_docx(tmp_path / "report.docx")
        assert read_docx(path) == DOCX_DOCUMENT

    def test_read_docx_return_stat(self, tmp_path):
        path = _build_docx(tmp_path / "report.docx")
        xml, stat = read_docx(path, return_stat=True)
        assert xml == DOCX_DOCUMENT
        assert stat.st_size == path.stat().st_size

    def test_corrupt_zip_fails_loud(self, tmp_path):
        path = tmp_path / "broken.docx"
        path.write_bytes(b"this is not a zip container at all")
        with pytest.raises(DocxError, match="not a readable .docx"):
            read_docx(path)

    def test_missing_document_entry_fails_loud(self, tmp_path):
        path = tmp_path / "empty.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        with pytest.raises(DocxError, match="word/document.xml"):
            read_docx(path)

    def test_non_utf8_document_fails_loud(self, tmp_path):
        path = tmp_path / "latin.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", b"<a>\xff\xfe</a>")
        with pytest.raises(DocxError, match="UTF-8"):
            read_docx(path)

    def test_build_docx_bytes_preserves_every_other_entry(self, tmp_path):
        path = _build_docx(tmp_path / "report.docx")
        before_names, before = _zip_snapshot(path)
        updated = DOCX_DOCUMENT.replace('w:val="en-US"', 'w:val="en-GB"')
        blob = build_docx_bytes(path, updated)
        out = tmp_path / "out.docx"
        out.write_bytes(blob)
        after_names, after = _zip_snapshot(out)

        assert after_names == before_names  # entry ORDER preserved
        for name in before_names:
            if name == "word/document.xml":
                continue
            # Byte-for-byte: payload, compression method, timestamp.
            assert after[name] == before[name], name

        with zipfile.ZipFile(out) as zf:
            assert zf.read("word/document.xml").decode("utf-8") == updated

    def test_write_docx_round_trips_and_preserves_methods(self, tmp_path):
        path = _build_docx(tmp_path / "report.docx")
        original_bytes = path.read_bytes()
        updated = DOCX_DOCUMENT.replace('w:val="en-US"', 'w:val="en-GB"')
        write_docx(path, updated)
        assert read_docx(path) == updated
        names, entries = _zip_snapshot(path)
        assert names == [
            "[Content_Types].xml", "_rels/.rels",
            "word/document.xml", "word/styles.xml",
        ]
        assert entries["_rels/.rels"][1] == zipfile.ZIP_STORED
        assert entries["word/styles.xml"][1] == zipfile.ZIP_STORED
        assert entries["_rels/.rels"][2] == (2024, 1, 2, 3, 4, 4)
        assert path.read_bytes() != original_bytes

    def test_write_docx_fails_loud_on_corrupt_container(self, tmp_path):
        path = tmp_path / "broken.docx"
        path.write_bytes(b"junk")
        with pytest.raises(DocxError):
            write_docx(path, DOCX_DOCUMENT)
        # Nothing was written over the corrupt file.
        assert path.read_bytes() == b"junk"


# ---------------------------------------------------------------------------
# 12. MCP-level docx round-trip (real write path, merge stubbed — hermetic)
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeMcp:
    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


class _FakeBackend:
    """Yields a dummy engine from acquire(): chunked_merge is stubbed in
    every test here, so merge_fn is never invoked — but the tool's
    ``async with backend.acquire()`` must still be a real async context
    manager (the same shape tests/test_mcp_fast_edit_parse_gate.py uses)."""

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield SimpleNamespace(merge_auto=None)


def _install_fake_mcp(monkeypatch):
    from fastedit.mcp import tools_ast, tools_edit
    from fastedit.mcp.backup import BackupStore

    lifespan_context = {
        "backend_kind": "mlx",
        # Never actually merges: chunked_merge is stubbed in every test.
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    fake = _FakeMcp(lifespan_context)
    monkeypatch.setattr(tools_edit, "mcp", fake)
    monkeypatch.setattr(tools_ast, "mcp", fake)
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


def _stub_chunked_merge(monkeypatch, result):
    from fastedit.mcp import tools_edit

    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(tools_edit, "chunked_merge", _stub)
    return calls


def _merge_result(merged_code):
    from fastedit.inference.ast_utils import ChunkedMergeResult

    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=True,
        chunks_used=1,
        chunk_regions=[(1, 12)],
        model_tokens=12,
        latency_ms=1.0,
        chunks_rejected=0,
        retries=0,
    )


class TestDocxMcpRoundTrip:
    def test_fast_edit_round_trips_a_docx(self, tmp_path, monkeypatch):
        from fastedit.mcp import tools_ast, tools_edit

        _install_fake_mcp(monkeypatch)
        path = _build_docx(tmp_path / "report.docx")
        original_bytes = path.read_bytes()
        before_names, before = _zip_snapshot(path)
        updated = DOCX_DOCUMENT.replace('w:val="en-US"', 'w:val="en-GB"')
        calls = _stub_chunked_merge(monkeypatch, _merge_result(updated))

        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(path),
            edit_snippet='<w:rPr><w:lang w:val="en-GB"/></w:rPr>\n',
        ))

        assert response.startswith(f"Applied edit to {path}"), response
        # The adapter path mapped the container to its XML document.
        (_args, kwargs) = calls[0]
        assert kwargs.get("original_code") == DOCX_DOCUMENT
        assert kwargs.get("language") == "xml"
        # The container on disk: document.xml updated, EVERY other entry
        # byte-identical (payload, compression method, timestamp, order).
        after_names, after = _zip_snapshot(path)
        assert after_names == before_names
        for name in before_names:
            if name == "word/document.xml":
                continue
            assert after[name] == before[name], name
        assert read_docx(path) == updated

        # And the B22 raw-bytes backup lets undo restore the ORIGINAL zip.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(path)))
        assert response.startswith(f"Reverted {path}"), response
        assert path.read_bytes() == original_bytes

    def test_corrupt_docx_fails_loud_without_writing(self, tmp_path, monkeypatch):
        from fastedit.mcp import tools_edit

        _install_fake_mcp(monkeypatch)
        path = tmp_path / "broken.docx"
        path.write_bytes(b"junk-not-a-zip")
        _stub_chunked_merge(monkeypatch, _merge_result(DOCX_DOCUMENT))

        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(path),
            edit_snippet='<w:lang w:val="en-GB"/>\n',
        ))

        assert response.startswith("Error:"), response
        assert path.read_bytes() == b"junk-not-a-zip"


# ---------------------------------------------------------------------------
# 13. Battery integration — scripted merge_fn, hermetic
# ---------------------------------------------------------------------------


def _StubMergeResult(merged_code, truncated=False):
    return SimpleNamespace(
        merged_code=merged_code,
        parse_valid=True,
        tokens_generated=7,
        latency_ms=1.0,
        truncated=truncated,
    )


def _run_merge(tmp_path, name, original, snippet, results, **kwargs):
    from fastedit.inference.chunked_merge import chunked_merge

    calls = []

    def merge_fn(code, snip, lang):
        calls.append((code, snip, lang))
        return results[min(len(calls) - 1, len(results) - 1)]

    target = tmp_path / name
    target.write_text(original, encoding="utf-8")
    result = chunked_merge(original, snippet, str(target), merge_fn, **kwargs)
    return result, calls


class TestBatteryAttributeGate:
    def test_attribute_reason_carries_the_battery_prefix(self):
        # Direct battery call: the fence closer was deleted by a merge the
        # CONTENT validator ratifies (identity-free line, marker-adjacent
        # positional capacity) — the D3 gate is the one that catches the
        # structure mutation, with its own reason prefix.
        from fastedit.inference.chunked_merge import _merge_rejection_reason

        original = "# Title\n```python\nprint('hi')\n```\n"
        snippet = "print('hi')\n# ... existing code ...\nappended.\n"
        merged = "# Title\n```python\nprint('hi')\nappended.\n"
        reason = _merge_rejection_reason(
            original, merged, snippet, "markdown", False, fmt="markdown",
        )
        assert reason is not None
        assert "language-attribute/structure check" in reason
        assert "md_fence_unclosed" in reason
        # With the closer deleted the fence swallows the tail — the gate's
        # reason names the undeclared state change, not a generic failure.
        assert "```python" in reason

    def test_gate_is_suffix_gated(self):
        # Without a declared spec format (fmt=None — .txt, .py, ...) the
        # D3 gate is INERT: only formats with a spec row are judged.
        from fastedit.inference.chunked_merge import (
            _attribute_rejection_reason,
        )

        original = '<html lang="en">\n<body>\n</body>\n'
        merged = '<html lang="fr">\n<body>\n</body>\n'
        assert _attribute_rejection_reason(
            original, merged, original, None,
        ) is None
        reason = _attribute_rejection_reason(
            original, merged, original, "html",
        )
        assert reason is not None
        assert "language-attribute/structure check" in reason
        assert "html_lang" in reason

    def test_html_declared_change_passes_first_try(self, tmp_path):
        # The declared p-line replacement is keyed-justified in the content
        # gate AND carries the declared trait — the whole battery accepts.
        snippet = '  <p lang="fr">First paragraph.</p>\n'
        result, calls = _run_merge(
            tmp_path, "page.html", HTML_ORIGINAL, snippet,
            [_StubMergeResult(HTML_LANG_DECLARED)], language="html",
        )
        assert len(calls) == 1  # clean on the first attempt
        assert result.retries == 0
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == HTML_LANG_DECLARED

    def test_html_undeclared_change_retries_then_rejects(self, tmp_path):
        # The snippet declares lang="fr"; the scripted "model" applies
        # lang="de" — rejected (the battery's gates agree it is unfaithful)
        # to exhaustion, original kept.
        snippet = '  <p lang="fr">First paragraph.</p>\n'
        merged = HTML_ORIGINAL.replace(
            '  <p lang="en">First paragraph.</p>',
            '  <p lang="de">First paragraph.</p>',
        )
        result, calls = _run_merge(
            tmp_path, "page.html", HTML_ORIGINAL, snippet,
            [_StubMergeResult(merged)], language="html",
            max_validation_retries=2,
        )
        assert len(calls) == 3  # initial + 2 retries (pinned budget)
        assert result.retries == 2
        assert result.chunks_rejected == 1
        assert result.parse_valid is False
        # Rejection convention: the original file is kept, never the payload.
        assert result.merged_code == HTML_ORIGINAL
        # The corrective note carries a rejection reason.
        assert "NOTE: the previous merge attempt was rejected" in calls[1][1]

    def test_md_flagship_malformed_frontmatter_survives(self, tmp_path):
        # req. 9 flagship: malformed frontmatter + a body edit → PASS with
        # the malformed frontmatter preserved byte-exact.
        snippet = "Body paragraph one.\nBody paragraph two.\n"
        merged = MD_FRONTMATTER.replace(
            "Body paragraph one.\n",
            "Body paragraph one.\nBody paragraph two.\n",
        )
        result, calls = _run_merge(
            tmp_path, "notes.md", MD_FRONTMATTER, snippet,
            [_StubMergeResult(merged)], language="markdown",
        )
        assert len(calls) == 1
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == merged
        assert "tags: [unclosed" in result.merged_code  # byte-exact survival

    def test_md_frontmatter_mutation_is_rejected(self, tmp_path):
        # The same edit, but the scripted model "helpfully fixes" the
        # malformed key line: trait mutation → rejected to exhaustion.
        snippet = "Body paragraph one.\nBody paragraph two.\n"
        merged = MD_FRONTMATTER.replace(
            "Body paragraph one.\n",
            "Body paragraph one.\nBody paragraph two.\n",
        ).replace("tags: [unclosed", "tags: []")
        result, calls = _run_merge(
            tmp_path, "notes.md", MD_FRONTMATTER, snippet,
            [_StubMergeResult(merged)], language="markdown",
            max_validation_retries=1,
        )
        assert len(calls) == 2
        assert result.chunks_rejected == 1
        # Routing (measured): the markdown grammar RESOLVES, so the D2
        # AST-less text-window branch never engages and the snippet — which
        # anchors no markdown AST node — falls through to the WHOLE-FILE
        # merge branch. That branch's exhaustion convention is the fail-loud
        # contract: the original file is kept (never the mutating payload)
        # and parse_valid is forced False (the universal do-not-persist
        # signal); the mutation is ratifiable by no gate.
        assert result.parse_valid is False
        assert result.merged_code == MD_FRONTMATTER

    def test_md_undeclared_fence_state_change_is_rejected_by_the_gate(
        self, tmp_path,
    ):
        # Full pipeline: the closer's deletion is content-justified, so the
        # ATTRIBUTE gate is what rejects — its reason rides the corrective
        # note, and exhaustion keeps the original file.
        original = "# Title\n```python\nprint('hi')\n```\n"
        snippet = "print('hi')\n# ... existing code ...\nappended.\n"
        merged = "# Title\n```python\nprint('hi')\nappended.\n"
        result, calls = _run_merge(
            tmp_path, "notes.md", original, snippet,
            [_StubMergeResult(merged)], language="markdown",
            max_validation_retries=1,
        )
        assert len(calls) == 2
        assert result.chunks_rejected == 1
        # Routing (measured): the markdown grammar resolves and the snippet
        # anchors no markdown AST node, so the op takes the WHOLE-FILE merge
        # branch (chunk_regions == the whole file), whose exhaustion
        # convention keeps the original file and forces parse_valid False.
        assert result.parse_valid is False
        assert result.merged_code == original
        assert result.chunk_regions == [(1, len(original.splitlines()))]
        assert "language-attribute/structure check" in calls[1][1]
        assert "md_fence_unclosed" in calls[1][1]

    def test_latex_declared_language_change_passes_the_battery(self):
        # Marker-bearing replacement snippet (the D1 idiomatic shape): the
        # declared babel-language change passes content, D1 counts AND D3.
        from fastedit.inference.chunked_merge import _merge_rejection_reason

        original = LATEX_WELL_FORMED
        snippet = (
            "\\documentclass{article}\n"
            "# ... existing code ...\n"
            "\\selectlanguage{german}\n"
            "Text \\foreignlanguage{french}{der Inhalt} more.\n"
            "\\begin{otherlanguage}{russian}\n"
            "body\n"
            "\\end{otherlanguage}\n"
        )
        merged = original.replace(
            "\\selectlanguage{french}", "\\selectlanguage{german}",
        ).replace("\\foreignlanguage{german}", "\\foreignlanguage{french}")
        reason = _merge_rejection_reason(
            original, merged, snippet, None, False, fmt="latex",
        )
        assert reason is None, reason

    def test_rtf_declared_change_passes_the_battery(self):
        # Marker-bearing replacement snippet (the D1 idiomatic shape): the
        # declared \lang change passes content, D1 counts AND D3.
        from fastedit.inference.chunked_merge import _merge_rejection_reason

        original = RTF_WELL_FORMED
        merged = original.replace("\\lang1033", "\\lang2057")
        snippet = (
            merged.splitlines()[0] + "\n"
            "# ... existing code ...\n"
            "}\n"
        )
        reason = _merge_rejection_reason(
            original, merged, snippet, None, False, fmt="rtf",
        )
        assert reason is None, reason

    def test_docx_declared_w_lang_change_passes_pipeline(self, tmp_path):
        # Full pipeline (real chunked_merge, scripted merge_fn) on a .docx:
        # the adapter format is derived from the SUFFIX, the declared
        # w:lang change lands, and the battery accepts.
        from fastedit.inference.chunked_merge import chunked_merge

        updated = DOCX_DOCUMENT.replace('w:val="en-US"', 'w:val="en-GB"')
        calls = []

        def merge_fn(code, snip, lang):
            calls.append((code, snip, lang))
            return _StubMergeResult(
                code.replace('w:val="en-US"', 'w:val="en-GB"'),
            )

        target = tmp_path / "report.docx"
        target.write_bytes(b"unused: chunked_merge sees the extracted XML")
        result = chunked_merge(
            DOCX_DOCUMENT,
            '<w:rPr><w:lang w:val="en-GB"/></w:rPr>\n',
            str(target),
            merge_fn,
            language="xml",
        )
        assert calls, "the scripted model path must have run"
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == updated

    def test_docx_undeclared_w_lang_change_is_rejected(self, tmp_path):
        from fastedit.inference.chunked_merge import chunked_merge

        def merge_fn(code, snip, lang):
            # The "model" applied a lang value the snippet never declared.
            return _StubMergeResult(
                code.replace('w:val="en-US"', 'w:val="fr-FR"'),
            )

        target = tmp_path / "report.docx"
        target.write_bytes(b"unused")
        result = chunked_merge(
            DOCX_DOCUMENT,
            '<w:rPr><w:lang w:val="en-GB"/></w:rPr>\n',
            str(target),
            merge_fn,
            language="xml",
            max_validation_retries=1,
        )
        assert result.chunks_rejected == 1
        assert result.parse_valid is False
        assert result.merged_code == DOCX_DOCUMENT  # original kept


# ---------------------------------------------------------------------------
# 14. The assembly-level exact check (multi-window edits)
# ---------------------------------------------------------------------------


class TestAssemblyLevelCheck:
    SOURCE = (
        "---\n"
        "title: Lab Notes\n"
        "---\n"
        "# Alpha\n"
        "\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
        "\n"
        "# Beta\n"
        "\n"
        "tail paragraph.\n"
    )
    SNIPPET = "# Alpha\nAppended alpha.\n# Beta\nAppended beta.\n"

    def test_faithful_assembly_passes(self):
        from fastedit.inference.chunked_merge import (
            _attribute_rejection_reason,
        )

        merged = self.SOURCE.replace(
            "# Alpha\n", "# Alpha\nAppended alpha.\n",
        ).replace("# Beta\n", "# Beta\nAppended beta.\n")
        assert _attribute_rejection_reason(
            self.SOURCE, merged, self.SNIPPET, "markdown",
        ) is None

    def test_structure_breaking_assembly_is_refused(self):
        # Each window merged faithfully, but the assembled file LOST the
        # fence closer — the window batteries cannot see it (their spans
        # cut mid-structure); the assembly-level exact check is the gate
        # that refuses the write.
        from fastedit.inference.chunked_merge import (
            _attribute_rejection_reason,
        )

        merged = self.SOURCE.replace(
            "# Alpha\n", "# Alpha\nAppended alpha.\n",
        ).replace("# Beta\n", "# Beta\nAppended beta.\n")
        corrupted = merged.replace("```\n", "", 1)
        reason = _attribute_rejection_reason(
            self.SOURCE, corrupted, self.SNIPPET, "markdown",
        )
        assert reason is not None
        assert "language-attribute/structure check" in reason
        assert "md_fence_unclosed" in reason

    def test_inert_without_a_spec_format(self):
        from fastedit.inference.chunked_merge import (
            _attribute_rejection_reason,
        )

        assert _attribute_rejection_reason(
            self.SOURCE, "anything changed", self.SNIPPET, None,
        ) is None
