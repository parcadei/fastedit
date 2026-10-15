"""Step D3 — language-attribute & structure-trait framework (req. 7 + 9).

Req. 7 demands per-chunk verification that LANGUAGE ATTRIBUTES are
preserved exactly — or changed exactly as the edit declares. Req. 9
(EDIT-NOT-CORRECT) demands the same for MALFORMED structure: a broken
fence, a malformed frontmatter or an empty ``lang=""`` is a TRAIT of the
input, preserved as-is unless the edit targets it — never silently
"fixed". This module is the declarative spec layer that makes those
traits first-class data, plus the trait oracle that compares them.

THE SPEC TABLE (CLAUDE.md: no if-chains — one dataclass row per format).
:func:`extract_attributes` dispatches on :data:`_FORMAT_SPECS`; a new
format is one new row (patterns + optional AST rules + optional
structure scanner), never a new branch in the pipeline:

=================  ==============================================  ==========================
format             attribute kinds                                 extractor strategy
=================  ==============================================  ==========================
``html``           ``html_lang`` (``lang=``), ``html_hreflang``    AST-assisted (tree-sitter
                   (``hreflang=``)                                 html) with regex fallback
``xml``            ``xml_lang`` (``xml:lang=``)                    AST-assisted (xml grammar)
                                                                   with regex fallback
``latex``          ``latex_selectlanguage``,                       regex, line-oriented
                   ``latex_foreignlanguage`` (babel),
                   ``latex_otherlanguage`` (incl. ``*``)
``rtf``            ``rtf_lang`` (``\\langNNNN``),                  regex, line-oriented
                   ``rtf_langfe`` (``\\langfeNNNN``)
``markdown``       ``md_fence_info``/``md_fence_marker`` (fence    line scanner (structure
                   info string + fence length),                    traits — no attribute
                   ``md_frontmatter_delim``/``md_frontmatter_key`` patterns)
``docx``           ``docx_lang`` (``w:lang`` elements'             AST-assisted (xml grammar
                   ``w:val``) — via the zip adapter                on the extracted XML) with
                                                                   regex fallback
=================  ==============================================  ==========================

EXTRACTION PRECEDENCE (documented, pinned by tests): when the format
declares AST rules and the B1 grammar resolver resolves the format's
grammar, attributes are collected from the tree — exact spans, no false
positives from text content (``el.lang = "fr"`` inside a ``<script>``
block is prose, not an attribute). When the grammar is unavailable or
the parse raises, the regex rows run instead — they must work with no
grammar installed at all (latex/rtf/markdown declare regex rows only and
never consult the resolver). Both paths yield the same (kind, value)
identities; the precedence can never silently flip because both sides
of one comparison extract with the same method in the same process.

MALFORMED TRAITS (req. 9). A trait with no value slot keeps its RAW
source text as the value — an unclosed ``\\selectlanguage{french``, a
``\\lang`` with no number, a ``w:lang`` without ``w:val`` — so any byte
change to the malformed construct is a trait change. Attributes WITH a
value slot keep the decoded value (``lang=""`` is the trait
``(html_lang, "")``): still a trait, still preserved. Structure state is
a trait too: an unclosed md fence is ``(md_fence_unclosed, <opener
line>)``, frontmatter without its closing delimiter is
``(md_frontmatter_unclosed, "---")``. A model that "helpfully" repairs
any of them without declaring the repair fails the battery like any
other unfaithful merge.

THE ORACLE — :func:`attributes_match_expectation`. Every input trait
must appear in the output VERBATIM at the same relative position (a
greedy in-order alignment — the same relative-position requirement the
byte-exact splices satisfy by construction), UNLESS the declared changes
explicitly touch it, in which case the declared new value must appear
exactly. Declared changes come from the snippet (the op spec — the same
doctrine ``_derived_text_op`` documents): every attribute-bearing line
the snippet states is declared, and the lines the snippet's shape could
justify removing (``chunked_merge._snippet_justifiable_removals``)
excuse the traits they carry. Concretely, a merge fails when

* an original trait is missing/reordered with no declared replacement of
  the same kind, no declared structure fix, and no justified removal;
* the output carries a trait nobody declared (undeclared introduction);
* a declared trait never lands in the output (the declared change was
  not applied — the retry note says exactly what is missing).

Relationship to the battery's other gates (documented): the content
validator is LINE-exact — every output line is a byte-exact survivor or
an exact declared new line — so line-level attribute value drift is
already rejected there. This gate is the semantic layer above it: it
names the attribute in the retry note, enforces the fence/frontmatter
STATE (identity-free structure lines can be deleted by the positional
capacity the content validator grants), and — wired at the final
assembly — is the only ASSEMBLY-level exact check a multi-window
md/html/xml/docx edit has (each window is only content-checked
span-locally; a splice that breaks cross-window structure is invisible
to every span-local gate).

THE DOCX ADAPTER. ``.docx`` is a zip container, not text: :func:`read_docx`
extracts ``word/document.xml`` (strict UTF-8), :func:`build_docx_bytes`
rewrites the container with the new document while preserving every other
entry — payload bytes, compression method, timestamps, external
attributes and entry order all carried over from the original
:class:`zipfile.ZipInfo` (a fresh info object is built per entry: stale
CRC/sizes/flag bits are deliberately NOT copied — ``writestr`` recomputes
them). Deflated entries are re-deflated by the stdlib at the default
level; stored entries are copied bit-for-bit — there are no
re-compression surprises beyond the unavoidable re-deflate of a changed
archive. :class:`DocxError` (a ``ValueError``) fails loud for corrupt
containers, missing entries and non-UTF-8 XML, before anything is
written.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SUPPORTED_ATTRIBUTE_FORMATS",
    "AttributeTrait",
    "DeclaredChanges",
    "DocxError",
    "attributes_match_expectation",
    "build_docx_bytes",
    "declared_changes_from_text",
    "extract_attributes",
    "format_for_path",
    "read_docx",
    "write_docx",
]


# ---------------------------------------------------------------------------
# The trait
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeTrait:
    """One language attribute / structure trait of a document (Step D3).

    Attributes:
        kind: the spec-table kind (``html_lang``, ``md_fence_info``, ...).
        value: the trait identity value — the decoded attribute value when
            the construct has a value slot, else the RAW source text of the
            malformed construct (req. 9: malformed traits are preserved
            as-is, so their identity IS their bytes).
        start: byte offset of the trait in the extracted text.
        end: exclusive end byte offset.
        raw: the exact source text the trait was extracted from.
    """

    kind: str
    value: str
    start: int
    end: int
    raw: str

    @property
    def key(self) -> tuple[str, str]:
        """The (kind, value) identity the oracle matches on."""
        return (self.kind, self.value)


# Kind families for the markdown structure traits: the unclosed-STATE
# trait of a family is excused exactly when the op declared work in that
# family (a declared frontmatter replacement legitimately removes the
# ``md_frontmatter_unclosed`` state by closing the block).
_TRAIT_FAMILY: dict[str, str] = {
    "md_fence_info": "fence",
    "md_fence_marker": "fence",
    "md_fence_unclosed": "fence",
    "md_frontmatter_delim": "frontmatter",
    "md_frontmatter_key": "frontmatter",
    "md_frontmatter_unclosed": "frontmatter",
}
_UNCLOSED_KINDS = frozenset({"md_fence_unclosed", "md_frontmatter_unclosed"})
_FAMILY_KINDS: dict[str, tuple[str, ...]] = {
    "fence": ("md_fence_info", "md_fence_marker", "md_fence_unclosed"),
    "frontmatter": (
        "md_frontmatter_delim",
        "md_frontmatter_key",
        "md_frontmatter_unclosed",
    ),
}


# ---------------------------------------------------------------------------
# The declarative spec table — one row per format, no if-chains
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TraitPattern:
    """One regex trait row: how to find a trait and read its value.

    ``value_groups`` name the capture groups holding the value, tried in
    order (double-quoted, single-quoted, then bare). ``raw_group``, when
    set, contributes the trait's raw text AND span from that group instead
    of the whole match. ``raw_when_missing_group`` / ``raw_when_empty``
    switch a row into the malformed-trait mode: when the closer group is
    absent (unclosed latex brace) or the captured value is empty
    (``\\lang`` with no number, a ``w:lang`` without ``w:val``), the value
    becomes the raw source text — the malformed construct's identity is
    its bytes (req. 9).
    """

    kind: str
    regex: re.Pattern[str]
    value_groups: tuple[str, ...] = ()
    raw_group: str | None = None
    raw_when_missing_group: str | None = None
    raw_when_empty: bool = False


@dataclass(frozen=True)
class _ElementRule:
    """One AST element rule: a tag whose attribute value is a trait.

    Used for element-borne attributes (docx ``w:lang`` carries the
    language in its ``w:val`` attribute). When the attribute is absent the
    element is still a trait — with the element's own raw text as the
    value (malformed, preserved as-is).
    """

    tag: str
    attribute: str
    kind: str


@dataclass(frozen=True)
class _FormatSpec:
    """The declarative spec row for one format (Step D3).

    Attributes:
        format: the canonical format name.
        grammar: the tree-sitter language the AST-assisted path parses
            with, or ``None`` for regex-only formats (latex/rtf/markdown).
        ast_attribute_kinds: attribute name -> trait kind, collected from
            ``attribute``/``Attribute`` tree nodes. Empty for element-borne
            or regex-only formats.
        ast_element_rules: element rules (tag, attribute, kind) collected
            from start/self-closing tag nodes.
        patterns: the regex rows (the fallback when no grammar resolves,
            and the ONLY extractor for regex-only formats).
        structure: an optional structure scanner (markdown's fences +
            frontmatter). Signature: ``(text, *, fragment) -> traits``.
    """

    format: str
    grammar: str | None = None
    ast_attribute_kinds: dict[str, str] | None = None
    ast_element_rules: tuple[_ElementRule, ...] = ()
    patterns: tuple[_TraitPattern, ...] = ()
    structure: object = None


_FENCE_LINE_RE = re.compile(r"^ {0,3}(?P<run>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_FRONTMATTER_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:(?:\s|$)")
"""Key-shaped line: the fragment-mode declaration of a frontmatter key.

A snippet that edits one frontmatter key rarely restates the whole block
(``---`` delimiters included), so a declared key-SHAPED line counts as a
declared key trait. Prose with a colon ("See page 12: details.") does not
match — the key name must be a bare identifier at column 0.
"""


def _line_records(text: str) -> list[tuple[str, int, int]]:
    """(raw line without its EOL, byte start, byte end) per line."""
    records: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        width = len(raw.encode("utf-8"))
        records.append((raw, offset, offset + width))
        offset += len(line.encode("utf-8"))
    return records


def _md_structure_traits(
    text: str,
    *,
    fragment: bool = False,
) -> list[AttributeTrait]:
    """The markdown structure traits: frontmatter + fenced code blocks.

    Frontmatter (the file must START with ``---``): the delimiter lines
    and every line between them are traits (``md_frontmatter_delim`` /
    ``md_frontmatter_key``, value = raw line — any byte change to a key
    line is a trait change, so malformed keys like ``tags: [unclosed``
    are preserved verbatim). With NO closing delimiter the block extent
    is unknowable, so key lines are NOT extracted — the malformed STATE
    itself is the trait (``md_frontmatter_unclosed``); a body edit after
    a malformed unclosed block must not become a phantom key trait.

    Fences: every opening fence contributes its info string
    (``md_fence_info``) and its fence run (``md_fence_marker`` — the
    fence LENGTH is the value's length). A fence still open at EOF
    contributes ``md_fence_unclosed`` with the opening line as the value:
    the unclosed state is a trait, preserved unless the edit declares the
    fix. CommonMark details honored: a backtick fence whose info string
    contains a backtick is not a fence; a closing fence must use the same
    character, be at least as long, and carry no info string.

    ``fragment=True`` (snippet/declared-removal extraction): fragments are
    truncated by construction, so the trailing unclosed state says
    nothing about the document and is suppressed; an empty-info fence
    line is ambiguous in a fragment (opener in the document, closer in
    the document) and declares nothing; a non-``---``-starting fragment
    still declares key-SHAPED lines as frontmatter-key traits.
    """
    traits: list[AttributeTrait] = []
    records = _line_records(text)

    def add(kind: str, value: str, record: tuple[str, int, int]) -> None:
        raw, start, end = record
        traits.append(AttributeTrait(kind, value, start, end, raw))

    # --- frontmatter ---
    if records and records[0][0].rstrip() == "---":
        close_idx = next(
            (
                i for i in range(1, len(records))
                if records[i][0].rstrip() in ("---", "...")
            ),
            None,
        )
        add("md_frontmatter_delim", records[0][0], records[0])
        if close_idx is None:
            if not fragment:
                add("md_frontmatter_unclosed", records[0][0], records[0])
        else:
            for i in range(1, close_idx):
                add("md_frontmatter_key", records[i][0], records[i])
            add("md_frontmatter_delim", records[close_idx][0], records[close_idx])
    elif fragment:
        for record in records:
            if _FRONTMATTER_KEY_RE.match(record[0]):
                add("md_frontmatter_key", record[0], record)

    # --- fenced code blocks ---
    in_fence = False
    fence_char = ""
    fence_len = 0
    opener_record: tuple[str, int, int] | None = None
    for record in records:
        match = _FENCE_LINE_RE.match(record[0])
        if match is None:
            continue
        run = match.group("run")
        info = match.group("info")
        if not in_fence:
            if run[0] == "`" and "`" in info:
                continue  # CommonMark: backtick info may not contain backticks
            if fragment and not info.strip():
                continue  # ambiguous in a fragment — declares nothing
            add("md_fence_info", info.strip(), record)
            add("md_fence_marker", run, record)
            in_fence = True
            fence_char = run[0]
            fence_len = len(run)
            opener_record = record
        elif (
            run[0] == fence_char
            and len(run) >= fence_len
            and not info.strip()
        ):
            in_fence = False
    if in_fence and not fragment and opener_record is not None:
        add("md_fence_unclosed", opener_record[0], opener_record)

    traits.sort(key=lambda t: (t.start, t.end))
    return traits


def _attr_value_row(kind: str, name: str) -> _TraitPattern:
    """The quoted/unquoted attribute-value regex row for one attribute."""
    return _TraitPattern(
        kind=kind,
        regex=re.compile(
            r"(?<![-:\w])" + name + r"\s*=\s*"
            r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'|(?P<bare>[^\s>/]*))"
        ),
        value_groups=("dq", "sq", "bare"),
    )


def _valueless_attr_row(kind: str, name: str) -> _TraitPattern:
    """The malformed bare-attribute row (``<div lang>`` — no ``=``).

    Anchored inside one tag so prose containing the word never matches;
    the value is the empty string (consistent with the AST path, which
    sees the attribute node without a value child).
    """
    return _TraitPattern(
        kind=kind,
        regex=re.compile(
            r"<[a-zA-Z][^<>]*?(?<![-:\w])(?P<name>" + name + r")(?=[\s/>])[^<>]*?>"
        ),
        raw_group="name",
    )


_FORMAT_SPECS: dict[str, _FormatSpec] = {
    "html": _FormatSpec(
        format="html",
        grammar="html",
        ast_attribute_kinds={"lang": "html_lang", "hreflang": "html_hreflang"},
        patterns=(
            _attr_value_row("html_lang", "lang"),
            _attr_value_row("html_hreflang", "hreflang"),
            _valueless_attr_row("html_lang", "lang"),
            _valueless_attr_row("html_hreflang", "hreflang"),
        ),
    ),
    "xml": _FormatSpec(
        format="xml",
        grammar="xml",
        ast_attribute_kinds={"xml:lang": "xml_lang"},
        patterns=(_attr_value_row("xml_lang", "xml:lang"),),
    ),
    "latex": _FormatSpec(
        format="latex",
        patterns=(
            _TraitPattern(
                kind="latex_selectlanguage",
                regex=re.compile(
                    r"\\selectlanguage[ \t]*\{"
                    r"(?P<value>[^}\n]*)(?P<closer>\})?"
                ),
                value_groups=("value",),
                raw_when_missing_group="closer",
            ),
            _TraitPattern(
                kind="latex_foreignlanguage",
                regex=re.compile(
                    r"\\foreignlanguage[ \t]*\{"
                    r"(?P<value>[^}\n]*)(?P<closer>\})?"
                ),
                value_groups=("value",),
                raw_when_missing_group="closer",
            ),
            _TraitPattern(
                kind="latex_otherlanguage",
                regex=re.compile(
                    r"\\begin[ \t]*\{otherlanguage\*?\}[ \t]*\{"
                    r"(?P<value>[^}\n]*)(?P<closer>\})?"
                ),
                value_groups=("value",),
                raw_when_missing_group="closer",
            ),
            _TraitPattern(
                kind="latex_otherlanguage",
                # Missing language argument — malformed, preserved as-is.
                regex=re.compile(r"\\begin[ \t]*\{otherlanguage\*?\}(?![ \t]*\{)"),
                raw_when_empty=True,
            ),
        ),
    ),
    "rtf": _FormatSpec(
        format="rtf",
        patterns=(
            _TraitPattern(
                kind="rtf_lang",
                # (?!fe) keeps \langfe out of the \lang row; (?![a-zA-Z])
                # keeps other control words (\language, ...) out too.
                regex=re.compile(r"\\lang(?!fe)(?![a-zA-Z])(?P<value>\d*)"),
                value_groups=("value",),
                raw_when_empty=True,
            ),
            _TraitPattern(
                kind="rtf_langfe",
                regex=re.compile(r"\\langfe(?![a-zA-Z])(?P<value>\d*)"),
                value_groups=("value",),
                raw_when_empty=True,
            ),
        ),
    ),
    "markdown": _FormatSpec(
        format="markdown",
        structure=_md_structure_traits,
    ),
    "docx": _FormatSpec(
        format="docx",
        grammar="xml",
        ast_element_rules=(_ElementRule("w:lang", "w:val", "docx_lang"),),
        patterns=(
            _TraitPattern(
                kind="docx_lang",
                regex=re.compile(
                    r"<w:lang\b[^>]*?\bw:val\s*=\s*"
                    r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)')[^>]*>"
                ),
                value_groups=("dq", "sq"),
            ),
            _TraitPattern(
                kind="docx_lang",
                # A w:lang element without w:val — malformed, raw identity.
                regex=re.compile(r"<w:lang\b(?![^>]*\bw:val\s*=)[^>]*>"),
                raw_when_empty=True,
            ),
        ),
    ),
}

SUPPORTED_ATTRIBUTE_FORMATS = frozenset(_FORMAT_SPECS)
"""The formats the D3 gate judges (the spec table's keys)."""

_FORMAT_ALIASES: dict[str, str] = {
    "md": "markdown",
    "markdown": "markdown",
    "tex": "latex",
    "latex": "latex",
}

_PATH_SUFFIX_FORMATS: dict[str, str] = {
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".xml": "xml",
    ".svg": "xml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".tex": "latex",
    ".ltx": "latex",
    ".latex": "latex",
    ".rtf": "rtf",
    ".docx": "docx",
}


def _canonical_format(fmt: object) -> str:
    """Map any accepted spelling to its canonical spec-table name."""
    name = str(fmt).strip().lower()
    return _FORMAT_ALIASES.get(name, name)


def format_for_path(file_path: str | Path) -> str | None:
    """The spec format for a file path, or ``None`` when it has no spec row.

    This is the suffix gate that keeps the D3 battery branch inert for
    every format it does not declare (``.txt``, ``.py``, ...). ``.docx``
    maps to the adapter format — the container is binary, so
    ``detect_language`` (deliberately) never resolves it; only the adapter
    path brings a .docx into the pipeline.
    """
    suffix = Path(str(file_path)).suffix.lower()
    return _PATH_SUFFIX_FORMATS.get(suffix)


# ---------------------------------------------------------------------------
# Extraction — AST-assisted first, regex fallback, structure scanners
# ---------------------------------------------------------------------------

_AST_ATTRIBUTE_NODES = frozenset({"attribute", "Attribute"})
_AST_NAME_NODES = frozenset({"attribute_name", "Name"})
_AST_VALUE_NODES = frozenset({"attribute_value", "AttValue", "quoted_attribute_value"})
_AST_TAG_NODES = frozenset({"start_tag", "self_closing_tag", "STag", "EmptyElemTag"})
_AST_TAG_NAME_NODES = frozenset({"tag_name", "Name"})


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _attribute_value_text(source: bytes, node) -> str:
    """The decoded value of an attribute-value tree node (both grammars).

    html: ``attribute_value`` (bare) or the inner ``attribute_value`` of a
    ``quoted_attribute_value``; xml: ``AttValue`` whose text still carries
    the quote delimiters — stripped here.
    """
    if node.type == "attribute_value":
        return _node_text(source, node)
    text = _node_text(source, node)
    inner = next(
        (c for c in node.children if c.type == "attribute_value"), None,
    )
    if inner is not None:
        return _node_text(source, inner)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _ast_traits(text: str, spec: _FormatSpec) -> list[AttributeTrait] | None:
    """AST-assisted extraction, or ``None`` when no grammar resolves.

    Walks the tree iteratively (no recursion limit on deep documents):
    ``attribute``/``Attribute`` nodes whose name is in the spec's
    ``ast_attribute_kinds`` become traits (raw/span = the attribute node);
    tags matching an element rule become traits (raw/span = the whole tag,
    value = the rule's attribute, or the tag's own text when the
    attribute — or its value — is missing: malformed, preserved as-is).
    """
    try:
        from .data_gen.ast_analyzer import parse_code

        tree = parse_code(text, spec.grammar)
    except Exception:  # noqa: BLE001 — ANY resolution/parse failure IS the regex fallback's cue
        return None

    source = text.encode("utf-8")
    traits: list[AttributeTrait] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _AST_ATTRIBUTE_NODES and spec.ast_attribute_kinds:
            name_node = next(
                (c for c in node.children if c.type in _AST_NAME_NODES), None,
            )
            if name_node is not None:
                kind = spec.ast_attribute_kinds.get(_node_text(source, name_node))
                if kind is not None:
                    value_node = next(
                        (
                            c for c in node.children
                            if c.type in _AST_VALUE_NODES
                        ),
                        None,
                    )
                    value = (
                        _attribute_value_text(source, value_node)
                        if value_node is not None else ""
                    )
                    traits.append(AttributeTrait(
                        kind, value, node.start_byte, node.end_byte,
                        _node_text(source, node),
                    ))
        elif node.type in _AST_TAG_NODES and spec.ast_element_rules:
            tag_node = next(
                (c for c in node.children if c.type in _AST_TAG_NAME_NODES), None,
            )
            if tag_node is not None and _node_text(source, tag_node) in {
                rule.tag for rule in spec.ast_element_rules
            }:
                for rule in spec.ast_element_rules:
                    if _node_text(source, tag_node) != rule.tag:
                        continue
                    value = None
                    for child in node.children:
                        if child.type not in _AST_ATTRIBUTE_NODES:
                            continue
                        attr_name = next(
                            (
                                c for c in child.children
                                if c.type in _AST_NAME_NODES
                            ),
                            None,
                        )
                        if attr_name is None:
                            continue
                        if _node_text(source, attr_name) != rule.attribute:
                            continue
                        attr_value = next(
                            (
                                c for c in child.children
                                if c.type in _AST_VALUE_NODES
                            ),
                            None,
                        )
                        if attr_value is not None:
                            value = _attribute_value_text(source, attr_value)
                        break
                    traits.append(AttributeTrait(
                        rule.kind,
                        _node_text(source, node) if value is None else value,
                        node.start_byte, node.end_byte,
                        _node_text(source, node),
                    ))
        stack.extend(node.children)
    traits.sort(key=lambda t: (t.start, t.end))
    return traits


def _pattern_value(pattern: _TraitPattern, match: re.Match[str]) -> str:
    """Resolve a regex row's trait value (malformed rows keep their raw)."""
    value = ""
    for group in pattern.value_groups:
        captured = match.group(group)
        if captured is not None:
            value = captured
            break
    if (
        pattern.raw_when_missing_group
        and match.group(pattern.raw_when_missing_group) is None
    ):
        return match.group(0)
    if pattern.raw_when_empty and value == "":
        return match.group(0)
    return value


def _regex_traits(
    text: str,
    spec: _FormatSpec,
    *,
    fragment: bool = False,
) -> list[AttributeTrait]:
    traits: list[AttributeTrait] = []
    for pattern in spec.patterns:
        for match in pattern.regex.finditer(text):
            if pattern.raw_group is not None:
                raw = match.group(pattern.raw_group)
                start, end = match.span(pattern.raw_group)
            else:
                raw = match.group(0)
                start, end = match.span()
            traits.append(AttributeTrait(
                pattern.kind, _pattern_value(pattern, match), start, end, raw,
            ))
    if spec.structure is not None:
        traits.extend(spec.structure(text, fragment=fragment))
    traits.sort(key=lambda t: (t.start, t.end))
    return traits


def extract_attributes(
    text: str,
    format: str,
    *,
    fragment: bool = False,
) -> list[AttributeTrait]:
    """Extract the language-attribute/structure traits of ``text``.

    Args:
        text: the document text (whole file, chunk, snippet or fragment).
        format: the spec format (``html``/``xml``/``latex``/``rtf``/
            ``markdown``/``docx``, aliases ``md``/``tex`` accepted). An
            unknown format yields ``[]`` — no spec row, no traits.
        fragment: ``True`` when ``text`` is a snippet/removal fragment
            rather than a document — suppresses truncated-state traits
            (see :func:`_md_structure_traits`).

    Returns:
        The traits in document order. AST-assisted for html/xml/docx when
        the grammar resolves, the regex rows otherwise (documented
        precedence; both paths agree on the (kind, value) identities).
    """
    fmt = _canonical_format(format)
    spec = _FORMAT_SPECS.get(fmt)
    if spec is None or not text:
        return []
    if spec.grammar is not None and (spec.ast_attribute_kinds or spec.ast_element_rules):
        ast_traits = _ast_traits(text, spec)
        if ast_traits is not None:
            return ast_traits
    return _regex_traits(text, spec, fragment=fragment)


# ---------------------------------------------------------------------------
# The declared op side + the trait oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredChanges:
    """What the op declares about the language attributes (Step D3).

    Attributes:
        declared: the traits the SNIPPET states (the whole snippet is the
            op spec — context anchors and declared new lines alike). A
            declared trait whose (kind, value) already exists in the
            original is a RESTATEMENT ("this stays"); it imposes no
            landing requirement and carries no replacement power. A
            non-restated declared trait both excuses one missing original
            trait of the same kind AND must land in the output exactly.
        declared_families: structure families the snippet's lines touch
            (markdown only) — the shape-based evidence that lets a
            declared fix legitimately remove an unclosed-state trait.
        removable: the traits carried by the lines the snippet's shape
            could justify removing (``_snippet_justifiable_removals``) —
            they may vanish without a replacement.
    """

    declared: tuple[AttributeTrait, ...] = ()
    declared_families: frozenset[str] = frozenset()
    removable: tuple[AttributeTrait, ...] = ()


def declared_changes_from_text(
    declared_text: str,
    removable_text: str,
    format: str,
) -> DeclaredChanges:
    """Build :class:`DeclaredChanges` from the op's raw text (Step D3).

    ``declared_text`` is the snippet the merge was told to apply;
    ``removable_text`` the raw lines the snippet's shape could justify
    removing. Both are extracted with the fragment rules (a fragment's
    trailing unclosed state is truncation noise, not a document trait).
    """
    fmt = _canonical_format(format)
    if fmt not in _FORMAT_SPECS:
        return DeclaredChanges()
    declared = tuple(
        extract_attributes(declared_text, fmt, fragment=True)
        if declared_text else ()
    )
    removable = tuple(
        extract_attributes(removable_text, fmt, fragment=True)
        if removable_text else ()
    )
    families: frozenset[str] = frozenset()
    if fmt == "markdown" and declared_text:
        found: set[str] = set()
        for line in declared_text.splitlines():
            if _FENCE_LINE_RE.match(line):
                found.add("fence")
            stripped = line.strip()
            if stripped in ("---", "...") or _FRONTMATTER_KEY_RE.match(line):
                found.add("frontmatter")
        families = frozenset(found)
    return DeclaredChanges(declared, families, removable)


def _next_unconsumed(
    out: list[AttributeTrait],
    consumed: list[bool],
    key: tuple[str, str],
    start: int,
) -> int | None:
    for j in range(start, len(out)):
        if not consumed[j] and out[j].key == key:
            return j
    return None


def attributes_match_expectation(
    original_text: str,
    output_text: str,
    format: str,
    declared_changes: DeclaredChanges | None = None,
) -> tuple[bool, str]:
    """Compare the output's traits against the op-transformed input (req. 7).

    The check follows the trait model: expected = the ORIGINAL's traits,
    transformed only by what the snippet declares. Every original trait
    must appear in the output verbatim at the same relative position (the
    greedy in-order alignment below), unless it is

    * replaced — a non-restated declared trait of the same kind exists
      (then that declared value must land in the output exactly);
    * a structure state the op declared fixing (an unclosed fence /
      frontmatter trait may vanish when the declared lines touch that
      structure family);
    * on a line the snippet's shape could justify removing.

    Every output trait must likewise be accounted for: matched to an
    original trait or covered by a declared trait — an unaccounted trait
    is an undeclared introduction (or an undeclared structure change).

    Returns:
        ``(True, "")`` when acceptable, else ``(False, reason)`` naming
        the kind and value — the string the retry loop feeds back.
    """
    fmt = _canonical_format(format)
    if fmt not in _FORMAT_SPECS:
        return True, ""
    orig = extract_attributes(original_text, fmt)
    out = extract_attributes(output_text, fmt)
    declared = list(declared_changes.declared) if declared_changes else []
    declared_families = (
        declared_changes.declared_families if declared_changes else frozenset()
    )
    removable_counts = Counter(
        t.key for t in (declared_changes.removable if declared_changes else ())
    )

    orig_keys = [t.key for t in orig]
    out_keys = [t.key for t in out]
    if (
        not declared
        and not removable_counts
        and not declared_families
        and orig_keys == out_keys
    ):
        return True, ""

    orig_key_counts = Counter(orig_keys)
    # A declared trait whose identity already exists in the original is a
    # RESTATEMENT ("this stays"): no landing requirement, no replacement
    # power — it can neither excuse a missing original nor demand a second
    # landing that would double-book the same output occurrence.
    restated = {d.key for d in declared if orig_key_counts.get(d.key, 0) > 0}

    # Phase 1 — the relative-position alignment: every original trait must
    # appear in the output, in order, at or after the previous one.
    consumed = [False] * len(out)
    missing: list[AttributeTrait] = []
    cursor = 0
    for trait in orig:
        idx = _next_unconsumed(out, consumed, trait.key, cursor)
        if idx is None:
            missing.append(trait)
        else:
            consumed[idx] = True
            cursor = idx + 1
    surplus = [o for j, o in enumerate(out) if not consumed[j]]

    excuse_pool: dict[str, deque[AttributeTrait]] = {}
    for d in declared:
        if d.key not in restated:
            excuse_pool.setdefault(d.kind, deque()).append(d)

    def _excuse(trait: AttributeTrait) -> bool:
        queue = excuse_pool.get(trait.kind)
        if queue:
            queue.popleft()
            return True
        family = _TRAIT_FAMILY.get(trait.kind)
        if trait.kind in _UNCLOSED_KINDS and family is not None:
            if family in declared_families:
                return True
            for kind in _FAMILY_KINDS[family]:
                family_queue = excuse_pool.get(kind)
                if family_queue:
                    family_queue.popleft()
                    return True
        if removable_counts.get(trait.key, 0) > 0:
            removable_counts[trait.key] -= 1
            return True
        return False

    for trait in missing:
        if not _excuse(trait):
            return False, (
                f"{trait.kind} {trait.value!r} from the original is missing "
                f"from the merged output (changed or dropped without "
                f"declaration)"
            )

    # Phase 2 — every output trait nobody claimed must be declared.
    declared_key_pool = Counter(d.key for d in declared)
    for trait in surplus:
        if declared_key_pool.get(trait.key, 0) > 0:
            declared_key_pool[trait.key] -= 1
            continue
        return False, (
            f"{trait.kind} {trait.value!r} appears in the merged output "
            f"without declaration"
        )

    # Phase 3 — every non-restated declared trait must land in the output
    # exactly (in declared order): "changed exactly as the edit declares".
    landed = [False] * len(out)
    dcursor = 0
    for d in declared:
        if d.key in restated:
            continue
        idx = next(
            (
                j for j in range(dcursor, len(out))
                if out[j].key == d.key and not landed[j]
            ),
            None,
        )
        if idx is None:
            return False, (
                f"declared {d.kind} {d.value!r} is missing from the merged "
                f"output"
            )
        landed[idx] = True
        dcursor = idx + 1
    return True, ""


# ---------------------------------------------------------------------------
# The docx adapter — stdlib zip in, stdlib zip out
# ---------------------------------------------------------------------------

_DOCX_DOCUMENT_PATH = "word/document.xml"


class DocxError(ValueError):
    """A .docx container could not be read or rewritten (fail loud).

    Subclasses :class:`ValueError` so every existing clean-refusal path
    (``except ValueError``) treats it as a user-facing refusal. Raised
    BEFORE anything is written — the file on disk is untouched.
    """


def read_docx(
    path: str | Path,
    return_stat: bool = False,
) -> str | tuple[str, os.stat_result]:
    """Extract ``word/document.xml`` from a .docx container (Step D3).

    Mirrors :func:`fastedit.io_utils.read_source`'s contract: strict
    decode (no ``errors=``), and with ``return_stat=True`` the
    ``os.stat_result`` captured via ``os.fstat`` from the SAME open the
    bytes were read through — pass it to the write so the B37
    lost-update guard covers container writes too.

    Raises:
        DocxError: not a zip container, no ``word/document.xml`` entry, or
            the document is not valid UTF-8. The file is not modified.
        OSError: the file cannot be read.
    """
    target = Path(path)
    with open(target, "rb") as fh:
        stat = os.fstat(fh.fileno())
        try:
            archive = zipfile.ZipFile(fh)
        except zipfile.BadZipFile as exc:
            raise DocxError(
                f"{path}: not a readable .docx (zip) container ({exc}). "
                f"The file was not modified."
            ) from exc
        with archive:
            try:
                data = archive.read(_DOCX_DOCUMENT_PATH)
            except KeyError as exc:
                raise DocxError(
                    f"{path}: no {_DOCX_DOCUMENT_PATH} entry — not an "
                    f"editable Word document. The file was not modified."
                ) from exc
    try:
        xml_text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocxError(
            f"{path}: {_DOCX_DOCUMENT_PATH} is not valid UTF-8 ({exc}); "
            f"decoding with replacement would corrupt untouched bytes. "
            f"The file was not modified."
        ) from exc
    if return_stat:
        return xml_text, stat
    return xml_text


def _preserved_zip_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """A fresh ZipInfo carrying the original entry's preserved metadata.

    Filename, timestamp, compression method, external/internal attributes,
    create-system and comment are carried over; stale CRC / sizes / flag
    bits are deliberately NOT — ``ZipFile.writestr`` recomputes them for
    the payload actually written.
    """
    fresh = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    fresh.compress_type = info.compress_type
    fresh.external_attr = info.external_attr
    fresh.internal_attr = info.internal_attr
    fresh.create_system = info.create_system
    fresh.comment = info.comment
    return fresh


def build_docx_bytes(path: str | Path, xml_text: str) -> bytes:
    """Rebuild the .docx container with a new ``word/document.xml``.

    Every other entry is written from the ORIGINAL archive with its
    original :class:`zipfile.ZipInfo` (see :func:`_preserved_zip_info`) and
    byte-identical uncompressed payload; the entry ORDER is preserved.
    Stored entries are therefore copied bit-for-bit; deflated entries are
    re-deflated by the stdlib at the default level — the only compression
    that ever changes is the one entry that DID change.

    Raises:
        DocxError: the container on disk is not readable (corrupt zip,
            missing entry) — fail loud, nothing written.
        OSError: the file cannot be read.
    """
    document = xml_text.encode("utf-8")
    buffer = io.BytesIO()
    try:
        with open(path, "rb") as fh, zipfile.ZipFile(fh) as zin, \
                zipfile.ZipFile(buffer, "w") as zout:
            for info in zin.infolist():
                payload = (
                    document
                    if info.filename == _DOCX_DOCUMENT_PATH
                    else zin.read(info)
                )
                zout.writestr(_preserved_zip_info(info), payload)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DocxError(
            f"{path}: not a readable .docx (zip) container ({exc}); "
            f"the merged document was NOT written and the container on "
            f"disk is untouched."
        ) from exc
    return buffer.getvalue()


def write_docx(
    path: str | Path,
    xml_text: str,
    *,
    backups=None,
    expected_stat: os.stat_result | None = None,
) -> None:
    """Atomically write ``xml_text`` as the document of the .docx at *path*.

    Builds the container bytes (:func:`build_docx_bytes`) and funnels them
    through the shared atomic write — bytes content is written verbatim
    (B22: no decode/re-encode anywhere), the current file's raw bytes are
    stored as a NEW timestamped backup first (so ``fast_undo`` restores
    the original container byte-exactly), and *expected_stat* arms the B37
    lost-update guard.
    """
    from .mcp.backup import _atomic_write

    _atomic_write(
        Path(path),
        build_docx_bytes(path, xml_text),
        backups=backups,
        expected_stat=expected_stat,
    )
