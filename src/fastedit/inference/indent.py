"""Indentation alignment and tag escaping for chunked merge.

Handles aligning snippet indentation to match chunk context, re-aligning
model output, and escaping/unescaping <updated-code> tags.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tag escaping — prevent model confusion from literal tags in file content
# ---------------------------------------------------------------------------

_TAG_OPEN = "<updated-code>"
_TAG_CLOSE = "</updated-code>"
_TAG_OPEN_SAFE = "__FASTEDIT_TAG_OPEN__"
_TAG_CLOSE_SAFE = "__FASTEDIT_TAG_CLOSE__"


def _escape_tags(text: str) -> str:
    """Replace literal <updated-code> tags with safe placeholders."""
    return text.replace(_TAG_OPEN, _TAG_OPEN_SAFE).replace(_TAG_CLOSE, _TAG_CLOSE_SAFE)


def _unescape_tags(text: str) -> str:
    """Restore safe placeholders back to literal <updated-code> tags."""
    return text.replace(_TAG_OPEN_SAFE, _TAG_OPEN).replace(_TAG_CLOSE_SAFE, _TAG_CLOSE)


# ---------------------------------------------------------------------------
# Indent alignment
# ---------------------------------------------------------------------------

def _align_snippet_indent(snippet: str, chunk_text: str) -> str:
    """Align snippet indentation to match the chunk's base indent.

    When `replace` scopes a class method, the chunk has class-level indent
    (e.g. 4 spaces) but Claude typically sends snippets at 0-indent.
    The 4B model sees the mismatch and may output at 0-indent, causing
    the method to fall outside the class.

    Fix: detect the indent delta and re-indent the snippet before merge.
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

    # Compute delta: how many spaces to add (positive) or remove (negative)
    chunk_spaces = len(chunk_indent.expandtabs(4))
    snippet_spaces = len(snippet_indent.expandtabs(4))
    delta = chunk_spaces - snippet_spaces

    if delta == 0:
        return snippet  # same effective width (tab vs space equivalence)

    result_lines = []
    for line in snippet.splitlines(keepends=True):
        if not line.strip():
            result_lines.append(line)  # preserve blank lines as-is
        elif delta > 0:
            result_lines.append(" " * delta + line)
        else:
            # Remove |delta| spaces from the start, but don't go negative
            remove = min(abs(delta), len(line) - len(line.lstrip()))
            result_lines.append(line[remove:])
    return "".join(result_lines)


def _realign_output(model_output: str, original_chunk: str) -> str:
    """Re-align model output indentation to match the original chunk.

    Unlike _align_snippet_indent (which shifts all lines uniformly), this
    handles the common model failure mode where the first line loses its
    indent but subsequent lines keep theirs.  Strategy:

    1. Compare first non-blank line indent of output vs chunk.
    2. If they match, return as-is (model got it right).
    3. If output first line has LESS indent than chunk, check if the body
       lines are already at the right indent. If so, only fix the first line.
    4. Otherwise, fall back to uniform shift via _align_snippet_indent.
    """
    def _first_line_indent(text: str) -> tuple[int, int]:
        """Return (indent_spaces, line_index) of first non-blank line."""
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                return len(line) - len(line.lstrip()), i
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
                return len(line) - len(line.lstrip())
        return -1

    chunk_body_indent = _second_line_indent(chunk_lines, 0)
    output_body_indent = _second_line_indent(output_lines, first_idx)

    if chunk_body_indent >= 0 and chunk_body_indent == output_body_indent:
        # Body is correct, only first line needs fixing — add delta to first line only
        result = []
        fixed_first = False
        for line in output_lines:
            if not fixed_first and line.strip():
                if delta > 0:
                    result.append(" " * delta + line)
                else:
                    remove = min(abs(delta), len(line) - len(line.lstrip()))
                    result.append(line[remove:])
                fixed_first = True
            else:
                result.append(line)
        return "".join(result)

    # Body indent also wrong — uniform shift
    return _align_snippet_indent(model_output, original_chunk)
