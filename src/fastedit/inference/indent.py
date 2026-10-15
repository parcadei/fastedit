"""Indentation alignment and tag escaping for chunked merge.

Handles aligning snippet indentation to match chunk context, re-aligning
model output, and escaping/unescaping <updated-code> tags.
"""

from __future__ import annotations

import secrets

from ..split_join import STRING_MASK, mask_string_spans
from .text_match import _dominant_indent_char, _indent_prefix, _indent_width

# ---------------------------------------------------------------------------
# Tag escaping — prevent model confusion from literal tags in file content
# ---------------------------------------------------------------------------

_TAG_OPEN = "<updated-code>"
_TAG_CLOSE = "</updated-code>"
# Legacy FIXED placeholders. They are collision-prone: user code that
# happens to contain the literal string round-trips into a literal tag
# (B33). Production call sites therefore pass a per-call random nonce; the
# one-arg form (legacy placeholders) is kept only for backward
# compatibility with existing callers/tests.
_TAG_OPEN_SAFE = "__FASTEDIT_TAG_OPEN__"
_TAG_CLOSE_SAFE = "__FASTEDIT_TAG_CLOSE__"


def _new_tag_nonce() -> str:
    """A random per-invocation nonce for tag placeholders (B33)."""
    return secrets.token_hex(8)


def _safe_placeholders(nonce: str) -> tuple[str, str]:
    """The placeholder pair for ``nonce``: nonced when given, legacy otherwise."""
    if not nonce:
        return _TAG_OPEN_SAFE, _TAG_CLOSE_SAFE
    return f"__FASTEDIT_TAG_OPEN_{nonce}__", f"__FASTEDIT_TAG_CLOSE_{nonce}__"


def _escape_tags(text: str, nonce: str = "") -> str:
    """Replace literal <updated-code> tags with safe placeholders.

    ``nonce`` suffixes the placeholders so they are unique per call: user
    text can no longer collide with them and come back as a literal tag
    (B33). Pair every escape with an unescape of the SAME nonce.
    """
    open_safe, close_safe = _safe_placeholders(nonce)
    return text.replace(_TAG_OPEN, open_safe).replace(_TAG_CLOSE, close_safe)


def _unescape_tags(text: str, nonce: str = "") -> str:
    """Restore the placeholders created by :func:`_escape_tags` (same nonce)
    back to literal <updated-code> tags. Placeholders carrying any OTHER
    nonce — e.g. the old fixed strings inside user text — are left alone."""
    open_safe, close_safe = _safe_placeholders(nonce)
    return text.replace(open_safe, _TAG_OPEN).replace(close_safe, _TAG_CLOSE)


# ---------------------------------------------------------------------------
# Indent alignment
# ---------------------------------------------------------------------------

def _strip_indent_columns(line: str, width: int) -> str:
    """Remove leading whitespace from ``line`` up to ``width`` columns.

    Columns are measured the way editors render them (``expandtabs(4)``),
    so removing 4 columns strips one tab from a tab-indented line or four
    spaces from a space-indented one — never ``width`` raw characters,
    which would mis-treat tabs as 1-column units (B7). Stops at the first
    full-whitespace character, so a line with less indent than ``width``
    loses only what it has.
    """
    if width <= 0:
        return line
    leading = line[: len(line) - len(line.lstrip(" \t"))]
    cut = 0
    for i in range(1, len(leading) + 1):
        if len(leading[:i].expandtabs(4)) <= width:
            cut = i
        else:
            break
    return line[cut:]


def _apply_indent_delta(line: str, delta: int, indent_char: str) -> str:
    """Shift one line's indentation by ``delta`` columns (signed).

    The shift is APPLIED in the file's own indent character: tabs render
    as whole 4-column levels via text_match._indent_prefix, spaces 1:1.
    Blank lines pass through untouched. Never mixes tab and space padding
    (B7).
    """
    if delta > 0:
        return _indent_prefix(delta, indent_char) + line
    if delta < 0:
        return _strip_indent_columns(line, min(-delta, _indent_width(line)))
    return line


def _string_interior_flags(text: str) -> list[bool]:
    """Per-line flags: True when the line BEGINS inside a string literal.

    Uses the shared :func:`split_join.mask_string_spans` masker. For such a
    line the leading whitespace is string CONTENT (it sits between the
    delimiters), so indent-shifting it would rewrite user data (B7/B30).
    """
    masked = mask_string_spans(text)
    return [m[:1] == STRING_MASK for m in masked.splitlines()]


def _has_string_interior_content(text: str) -> bool:
    """True when any non-blank line of ``text`` begins inside a string."""
    flags = _string_interior_flags(text)
    return any(
        interior and line.strip()
        for interior, line in zip(flags, text.splitlines())
    )


def _indent_char_of(text: str) -> str:
    """The dominant indent style of ``text`` (tab vs space).

    Delegates to text_match._dominant_indent_char (the B8 helper), voting
    on ``text``'s own lines with the first non-blank line as tie-breaker:
    the file's existing convention wins over anything the new text
    carries.
    """
    lines = text.splitlines()
    ref = next((ln for ln in lines if ln.strip()), "")
    return _dominant_indent_char(lines, ref)


def _align_snippet_indent(snippet: str, chunk_text: str) -> str:
    """Align snippet indentation to match the chunk's base indent.

    When `replace` scopes a class method, the chunk has class-level indent
    (e.g. 4 spaces) but Claude typically sends snippets at 0-indent.
    The 4B model sees the mismatch and may output at 0-indent, causing
    the method to fall outside the class.

    Fix: detect the indent delta and re-indent the snippet before merge.

    B7 rules:
      * the delta is computed in columns (``expandtabs(4)``) AND applied in
        the CHUNK's (the original's) own indent character — a tab-indented
        chunk shifts with tabs, never with injected spaces;
      * lines that begin inside a multi-line string are USER DATA — their
        leading whitespace belongs to the string's value — and pass through
        byte-identical (this includes a closing-delimiter line preceded by
        content spaces, whose indent IS the string's last content line).
    """
    def _base_indent(text: str) -> str:
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped:  # first non-blank line
                return line[: len(line) - len(stripped)]
        return ""

    chunk_indent = _base_indent(chunk_text)
    snippet_indent = _base_indent(snippet)

    if chunk_indent == snippet_indent:
        return snippet  # already aligned

    # Compute delta: how many columns to add (positive) or remove (negative)
    chunk_spaces = len(chunk_indent.expandtabs(4))
    snippet_spaces = len(snippet_indent.expandtabs(4))
    delta = chunk_spaces - snippet_spaces

    if delta == 0:
        return snippet  # same effective width (tab vs space equivalence)

    indent_char = _indent_char_of(chunk_text)
    interior = _string_interior_flags(snippet)

    result_lines = []
    for line, is_interior in zip(snippet.splitlines(keepends=True), interior):
        if not line.strip() or is_interior:
            result_lines.append(line)  # blank lines and string content: as-is
        else:
            result_lines.append(_apply_indent_delta(line, delta, indent_char))
    return "".join(result_lines)


def _realign_output(
    model_output: str,
    original_chunk: str,
    protected_lines: tuple[str, ...] | list[str] = (),
) -> str:
    """Re-align model output indentation to match the original chunk.

    Unlike _align_snippet_indent (which shifts all lines uniformly), this
    handles the common model failure mode where the first line loses its
    indent but subsequent lines keep theirs.  Strategy:

    1. Compare first non-blank line indent of output vs chunk (in columns).
    2. If they match, return as-is (model got it right).
    3. If the body lines are already at the right indent, only fix the
       first line.
    4. If the output contains a multi-line string, a forced uniform shift
       would move string-interior lines (user data) — only fix the first
       line there too (B30): the minimal safe correction beats a shift
       that rewrites the string or reverts the model's indent change.
    5. Otherwise, fall back to uniform shift via _align_snippet_indent
       (which itself never moves string content).

    ``protected_lines`` (C3 seams stress) lists the snippet's declared NEW
    lines in RAW form (indent included, order-sensitive — consumed by a
    cursor). These are bytes the aligned snippet told the model to insert,
    so a model echo of them already carries the right indent; the uniform
    shift computed from the model's drifted COPY of the chunk must never be
    applied to them (measured defect: the go model emitted the chunk body
    one level shallow while echoing the declared tail verbatim, and the
    blanket shift pushed the whole declared tail one level deeper — the
    validator's content-level view cannot see indent, so the corrupted
    merge was ratified and written). Lines that do not continue the
    declared sequence are repaired exactly as before, so a genuinely
    drifted declared line still gets the legacy repair.
    """
    def _first_line_indent(text: str) -> tuple[int, int]:
        """Return (indent_columns, line_index) of first non-blank line."""
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                return _indent_width(line), i
        return 0, 0

    chunk_indent, _ = _first_line_indent(original_chunk)
    output_indent, first_idx = _first_line_indent(model_output)

    if chunk_indent == output_indent:
        return model_output  # already correct

    delta = chunk_indent - output_indent

    # Check if body lines (after the first non-blank) are already correct.
    # This happens when the model strips only the def/class line's indent.
    output_lines = model_output.splitlines(keepends=True)
    chunk_lines = original_chunk.splitlines(keepends=True)

    # Find the second non-blank line indent in both
    def _second_line_indent(text_lines: list[str], after_idx: int) -> int:
        for line in text_lines[after_idx + 1:]:
            if line.strip():
                return _indent_width(line)
        return -1

    chunk_body_indent = _second_line_indent(chunk_lines, 0)
    output_body_indent = _second_line_indent(output_lines, first_idx)

    interior = _string_interior_flags(model_output)
    if first_idx < len(interior) and interior[first_idx]:
        # The output's first non-blank line is itself string content —
        # nothing can be re-indented safely. (first_idx is only ever out
        # of range for an EMPTY output, which has nothing to fix either.)
        return model_output

    if protected_lines:
        # Order-sensitive cursor over the declared new lines: an output
        # line is protected only while it continues the declared sequence
        # verbatim (EOL-insensitive — the funnel owns endings), so a chunk
        # line with the same bytes (a lone `}`) is still repaired, and a
        # genuinely drifted declared line falls back to the legacy repair
        # instead of being left corrupt.
        pending = [
            line.rstrip("\r\n") if line.endswith(("\r", "\n")) else line
            for line in protected_lines
        ]

        def _is_protected(line: str) -> bool:
            if pending and line.rstrip("\r\n") == pending[0]:
                pending.pop(0)
                return True
            return False

    else:
        def _is_protected(line: str) -> bool:
            return False

    body_matches = chunk_body_indent >= 0 and chunk_body_indent == output_body_indent
    if (body_matches or _has_string_interior_content(model_output)) and not (
        _is_protected(output_lines[first_idx])
    ):
        # Only the first non-blank line needs fixing — in the chunk's own
        # indent style (B7), and never a uniform shift over string
        # content (B30).
        result = list(output_lines)
        result[first_idx] = _apply_indent_delta(
            output_lines[first_idx], delta, _indent_char_of(original_chunk),
        )
        return "".join(result)

    # Body indent also wrong — uniform shift, never touching the declared
    # new lines the model echoed verbatim (blank and string-interior lines
    # pass through untouched, as in _align_snippet_indent).
    indent_char = _indent_char_of(original_chunk)
    result_lines = []
    for line, is_interior in zip(output_lines, interior):
        if line.strip() and not is_interior and not _is_protected(line):
            result_lines.append(_apply_indent_delta(line, delta, indent_char))
        else:
            result_lines.append(line)
    return "".join(result_lines)
