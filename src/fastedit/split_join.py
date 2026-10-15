"""Split a file into per-chunk parts, and join those parts back into one.

Format-aware where the format allows a clean, lossless cut (a JSON array's
elements, a Markdown file's headings, CSV/TSV rows); a plain line-count
chunker otherwise. Every split writes a manifest.json into the output
directory so `join` can put the pieces back together without re-parsing
or re-serializing content -- reconstruction is byte-for-byte for every
mode except XML/HTML element splitting, which is lossy in general (see
below) and therefore read-only: split only, no join.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

MANIFEST_NAME = "manifest.json"

_MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
_CSV_EXTS = {".csv"}
_TSV_EXTS = {".tsv"}
_JSON_EXTS = {".json"}
_JSONL_EXTS = {".jsonl", ".ndjson"}
_XML_EXTS = {".xml"}
_HTML_EXTS = {".html", ".htm"}


class SplitJoinError(ValueError):
    """A requested mode doesn't fit the file (wrong --by, --rows on non-csv, ...)."""

def detect_format(path: Path) -> str:
    """Guess a split/join format from the file extension, defaulting to plain text."""
    ext = path.suffix.lower()
    if ext in _JSON_EXTS:
        return "json"
    if ext in _JSONL_EXTS:
        return "jsonl"
    if ext in _XML_EXTS:
        return "xml"
    if ext in _HTML_EXTS:
        return "html"
    if ext in _MARKDOWN_EXTS:
        return "markdown"
    if ext in _CSV_EXTS:
        return "csv"
    if ext in _TSV_EXTS:
        return "tsv"
    return "text"


def detect_line_ending(text: str) -> str:
    """Return the line-ending terminator that dominates text: CRLF, lone-CR, or LF.

    Counted directly (splitlines also treats several other Unicode characters
    as line breaks and would misreport a Python-only boundary as missing).

    Falls back to "\n" -- never os.linesep -- when text carries no line
    ending at all (empty string or a single line) or when two styles tie
    exactly. os.linesep is a HOST property ("\n" on macOS/Linux, "\r\n" on
    Windows): picking it here would make the SAME edit on the SAME file
    emit different bytes depending on which machine ran fastedit, breaking
    this tool's byte-exactness contract (TRDD-J8Q28MED). "\n" is the
    deterministic, host-independent choice: every caller of this function
    passes it the text being edited with no other signal to fall back on,
    so there is nothing platform-specific left to infer from -- LF is also
    the git/POSIX convention. DO NOT reintroduce os.linesep in either branch
    below.
    """
    crlf = text.count("\r\n")
    lone_cr = text.count("\r") - crlf
    lone_lf = text.count("\n") - crlf
    counts = {"\r\n": crlf, "\n": lone_lf, "\r": lone_cr}
    top = max(counts.values())
    if top == 0 or sum(1 for v in counts.values() if v == top) > 1:
        return "\n"
    return next(k for k, v in counts.items() if v == top)


def normalize_line_endings(text: str, target: str) -> str:
    """Rewrite every line ending in text to target (CRLF, lone-CR, or LF).

    Used to make snippet/inserted text match the prevailing convention of the
    file it is being spliced into. Callers apply this only to the new or
    replaced span, never to a whole file, so untouched regions stay byte-exact.
    """
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return canonical if target == "\n" else canonical.replace("\n", target)

def mask_string_spans(text: str, mask: str = "\x00") -> str:
    """A SAME-LENGTH copy of *text* with every character inside a string
    literal replaced by *mask* (NUL by default). Line terminators are never
    masked, so line counts, line numbers, and offsets are identical between
    *text* and the returned copy — callers can run line-oriented logic on
    the masked copy and index/slice the original safely.

    Shared by the marker normalizer (never rewrite a marker-looking line
    that lives inside an embedded template/script — B18) and the indent
    arithmetic (never shift a line whose text is string content — B7/B30).

    HEURISTIC, deliberately language-agnostic. ``normalize_markers`` and the
    indent helpers run on raw snippets/chunks with no language parameter and
    must stay pure string transforms, so a tree-sitter parse is not
    available here (and comment masking is out of scope entirely: comment
    syntax varies too much to guess safely). This scanner covers the
    delimiter families of the mainstream languages fastedit edits:

      * triple-quoted: ``\"\"\" ... \"\"\"`` and ``''' ... '''`` (multi-line);
      * single-line quoted: ``" ... "`` and ``' ... '`` with backslash
        escapes — an unterminated one ends at the line break, matching the
        C/Python/JS/Go rule that such strings cannot span lines;
      * backtick: ``` ... ``` (JS template literals, Go raw strings) —
        multi-line capable.

    A backslash consumes the following character everywhere (string escapes,
    escaped quotes/newlines — and, outside strings, C line continuations,
    where consuming the pair is harmless). The delimiter characters
    themselves stay unmasked, so a line whose FIRST character is masked
    begins inside a string and its leading whitespace is string content.
    Escaped characters inside a string stay unmasked too (the scanner skips
    the whole pair), and so do ``\\r``/``\\n`` everywhere.

    Over-masking (a stray unpaired delimiter) only makes the transform more
    conservative — content is preserved as-is; it can never corrupt it.

    PERFORMANCE (global-governor mission, PART 1): this used to build a
    per-character mutable copy (``list(text)`` — one Python object per
    character: ~40 MB of pointers for a 5 MB ASCII file, far more for
    multi-byte content) and walk it char by char in Python. Measured on a
    5 MB single-line input: 1.73 s wall / 42.9 MB peak allocation BEFORE,
    driven by exactly that list + per-char loop. The rewrite below is
    behavior-identical (byte-for-byte equivalence is asserted against a
    reference copy of the old scanner in tests/test_resource_hub.py,
    including a seeded 500-case fuzz over quote/escape/CRLF/multi-byte
    soup) but scans with compiled regexes and masks whole runs at C speed:
    the masked regions are collected as ``(start, end)`` spans where every
    character is provably maskable (the regex alternation matches escape
    pairs, closing delimiters and terminators FIRST, so a span never
    contains one), and the output is assembled from slices plus
    ``mask * run_length``. No per-character Python objects are ever built.
    """

    n = len(text)
    if n == 0:
        return text
    # Maskable runs collected as [start, end) spans. The interior scanner's
    # regex matches escape pairs, closing delimiters and terminators FIRST,
    # so a span never contains one and ``mask * span_length`` is exact.
    spans: list[tuple[int, int]] = []

    i = 0
    while i < n:
        m = _OUTER_TOKEN_RE.search(text, i)
        if m is None:
            break
        tok = m.group(0)
        if tok[0] == "\\":
            # Escape pair (or bare trailing backslash): both characters stay
            # unmasked, and the scan resumes past them — never a delimiter
            # start (the old scanner's ``i += 2``).
            i = m.end()
            continue
        # tok is a quote delimiter: """ / ''' (triple, multi-line) or one of
        # " ' ` (backtick multi-line, the others single-line).
        multi_line = len(tok) == 3 or tok == "`"
        i = _mask_string_interior(text, m.end(), tok, multi_line, spans)

    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(text[pos:start])
        out.append(mask * (end - start))
        pos = end
    out.append(text[pos:])
    return "".join(out)


_OUTER_TOKEN_RE = re.compile(r'\\.|"""|\'\'\'|["\'`]', re.DOTALL)
"""Tokens the outside-a-string scanner reacts to: an escape pair (``\\.``
with DOTALL also swallows an escaped newline — the C line continuation), a
triple-quote opener (tried BEFORE the single-quote alternative, exactly like
the old ``startswith(('\"\"\"', \"'''\"))`` check), or any quote character. A
bare trailing backslash matches none of them and is simply skipped unmasked
(the old ``i += 2`` clamping past EOF)."""


@functools.cache
def _interior_close_re(delim: str, multi_line: bool) -> re.Pattern[str]:
    """The inside-a-string event regex for ``delim``.

    Alternation order is load-bearing: an escape pair wins over the closing
    delimiter (``\\"`` must not close), the delimiter over the terminators,
    and a bare trailing backslash (``\\\\Z``) comes last — it only fires when
    ``\\.`` failed, i.e. the backslash is the final character. Terminators
    are alternatives so they pass through unmasked; in a single-line string
    ``\\n`` CLOSES (handled by the caller via the ``multi_line`` flag).
    """
    parts = [r"\\.", re.escape(delim)]
    if multi_line:
        parts.append(r"\r\n|\r|\n")
    else:
        parts.append(r"\n")
        parts.append(r"\r")  # lone CR passes through unmasked, never closes
    parts.append(r"\\\Z")
    return re.compile("|".join(parts), re.DOTALL)


def _mask_string_interior(
    text: str,
    i: int,
    delim: str,
    multi_line: bool,
    spans: list[tuple[int, int]],
) -> int:
    """Mask string-interior characters from ``i``; return the resume index.

    The gap before each regex event is a maskable run (it contains no
    escape pair, no delimiter, no terminator — the regex would have matched
    it first) and is appended to *spans*. Closing events — the delimiter
    itself, or the line break that terminates a single-line string — end
    the string; escapes and (multi-line) terminators pass through unmasked
    and scanning continues. The resume index sits just past the closing
    delimiter or past the closing line break (the old scanner returned the
    break's own index and the outer loop then stepped over it — the same
    resume point), or at EOF for an unterminated multi-line span.
    """
    n = len(text)
    close_re = _interior_close_re(delim, multi_line)
    while True:
        m = close_re.search(text, i)
        if m is None:
            if i < n:
                spans.append((i, n))
            return n
        if m.start() > i:
            spans.append((i, m.start()))
        tok = m.group(0)
        if tok == delim:
            return m.end()
        if tok == "\n" and not multi_line:
            return m.end()
        i = m.end()


STRING_MASK = "\x00"
"""The default mask character produced by :func:`mask_string_spans`.

A line whose first character equals this marker BEGINS inside a string
literal (its leading whitespace is string content, not code indentation).
"""


def normalize_bare_cr_for_ast(text: str) -> str:
    """Replace a lone CR (not part of CRLF) with LF -- same length, same positions.

    Tree-sitter and the external tldr binary both count rows by scanning for
    a plain LF; a bare CR (classic-Mac line ending) is invisible to them and
    collapses every line of a CR-only or CR-mixed file into one, corrupting
    any line_start/line_end an AST tool returns for it. This substitution is
    always a 1-byte-for-1-byte swap (it never touches an existing "\r\n" or
    "\n"), so line counts, line positions, and byte offsets computed against
    the result stay valid against the original text -- callers pass this
    normalized copy only to an AST/structure tool for line-number purposes,
    and keep slicing their own original text for the actual content.
    """
    import re

    return re.sub(r"\r(?!\n)", "\n", text)





# --- line-count chunking (works on any text file) ---


def split_by_lines(text: str, n: int) -> list[str]:
    """Chunk raw text into groups of exactly n lines each (last group may be short)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    return ["".join(lines[i : i + n]) for i in range(0, len(lines), n)]


# --- CSV/TSV row chunking ---


def split_csv_rows(text: str, n: int) -> list[str]:
    """Chunk CSV/TSV data rows into groups of n, repeating the header row in each.

    Naive and line-based: a data row is one physical line. A field value
    containing an embedded newline (legal, quoted, RFC 4180 CSV) would be
    split incorrectly -- a real CSV parser would be needed to handle that,
    out of scope for this pass.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    header, rows = lines[0], lines[1:]
    if not rows:
        return [header]
    return ["".join([header, *rows[i : i + n]]) for i in range(0, len(rows), n)]


def join_csv_chunks(chunks: list[bytes]) -> bytes:
    """Undo split_csv_rows: keep the first chunk whole, strip the repeated header from the rest."""
    if not chunks:
        return b""
    out = [chunks[0]]
    header = chunks[0].splitlines(keepends=True)[0] if chunks[0] else b""
    for chunk in chunks[1:]:
        lines = chunk.splitlines(keepends=True)
        if lines and lines[0] == header:
            lines = lines[1:]
        out.append(b"".join(lines))
    return b"".join(out)


# --- Markdown heading chunking ---


def split_markdown_headings(text: str, level: int) -> list[str]:
    """Cut markdown text right before each heading of the given level (## for level 2).

    Each chunk is a contiguous, unmodified slice of the original text, so
    concatenating the chunks in order reproduces the file exactly.
    """
    import itertools

    marker = "#" * level + " "
    lines = text.splitlines(keepends=True)
    boundaries = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not boundaries:
        raise SplitJoinError(f"no level-{level} headings ('{marker.strip()} ...') found")
    if boundaries[0] != 0:
        boundaries = [0, *boundaries]
    boundaries.append(len(lines))
    chunks = []
    for start, end in itertools.pairwise(boundaries):
        if start == end:
            continue
        chunks.append("".join(lines[start:end]))
    return chunks


# --- JSON array element splitting ---


def split_json_array(text: str) -> tuple[str, list[str], list[str], str]:
    """Split a top-level JSON array into (prefix, elements, separators, suffix).

    Elements and separators are exact substrings of *text* -- concatenating
    prefix + elements[0] + separators[0] + elements[1] + ... + elements[-1]
    + suffix reproduces *text* byte-for-byte, regardless of the source's
    original indentation or spacing.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise SplitJoinError(f"not valid JSON: {e}") from e
    if not isinstance(parsed, list):
        kind = "object" if isinstance(parsed, dict) else type(parsed).__name__
        raise SplitJoinError(f"--by element needs a JSON array at the top level, got a JSON {kind}")

    i = 0
    while text[i] in " \t\r\n":
        i += 1
    # json.loads already proved the top-level value is an array, so this is its '['.
    pos = i + 1
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    if not parsed:
        return text[: pos + 1] if text[pos] == "]" else text, [], [], ""

    decoder = json.JSONDecoder()
    elements: list[str] = []
    separators: list[str] = []
    prefix_end = pos
    while True:
        _, end = decoder.raw_decode(text, pos)
        elements.append(text[pos:end])
        k = end
        while k < len(text) and text[k] in " \t\r\n":
            k += 1
        if k < len(text) and text[k] == ",":
            m = k + 1
            while m < len(text) and text[m] in " \t\r\n":
                m += 1
            separators.append(text[end:m])
            pos = m
            continue
        if k < len(text) and text[k] == "]":
            return text[:prefix_end], elements, separators, text[end:]
        raise SplitJoinError("malformed JSON array (expected ',' or ']')")


def join_json_array(prefix: str, elements: list[bytes], separators: list[str], suffix: str) -> bytes:
    """Undo split_json_array."""
    body = []
    for idx, element in enumerate(elements):
        body.append(element)
        if idx < len(separators):
            body.append(separators[idx].encode("utf-8"))
    return prefix.encode("utf-8") + b"".join(body) + suffix.encode("utf-8")


# --- XML/HTML top-level-child splitting (read-only: split, never join) ---

_TAG_RE = re.compile(
    r"<!--.*?-->"  # comments
    r"|<!\[CDATA\[.*?\]\]>"  # CDATA
    r"|<\?.*?\?>"  # processing instructions / xml declaration
    r"|<!DOCTYPE[^>]*>"  # doctype
    r"|<(?P<close>/?)(?P<name>[a-zA-Z][a-zA-Z0-9:_.-]*)(?P<attrs>[^>]*?)(?P<selfclose>/?)>",
    re.DOTALL,
)

_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def split_markup_top_level_children(text: str) -> list[str]:
    """Best-effort split of an XML/HTML document into its root element's direct children.

    This is a naive tag-depth scanner, not a real (X)HTML parser: it does
    not understand script/style content with embedded angle brackets, or
    exotic markup. It is used only for `split`'s read-only element mode --
    there is deliberately no matching join, because a plucked-out fragment
    loses ancestor context (namespace declarations, xml:base, inherited
    attributes) that a real round-trip would need. Good enough for typical
    hand-written or generated markup; anything exotic should use --lines.
    """
    root_start = None
    root_name = None
    depth = 0
    children: list[str] = []
    child_start = None
    for m in _TAG_RE.finditer(text):
        tag = m.group(0)
        if tag.startswith(("<!--", "<![CDATA[", "<?", "<!DOCTYPE")):
            continue
        name = m.group("name")
        is_close = bool(m.group("close"))
        is_self_closing = bool(m.group("selfclose")) or name.lower() in _VOID_ELEMENTS

        if root_start is None:
            if is_close:
                continue
            root_start = m.start()
            root_name = name
            depth = 1
            if is_self_closing:
                return []  # a self-closing root has no children to split out
            continue

        if depth == 1 and not is_close:
            child_start = m.start()

        if is_close:
            depth -= 1
            if depth == 1 and child_start is not None:
                children.append(text[child_start : m.end()])
                child_start = None
            if depth == 0 and name == root_name:
                break
        elif not is_self_closing:
            depth += 1
        elif depth == 1 and child_start is not None:
            children.append(text[child_start : m.end()])
            child_start = None

    return children
