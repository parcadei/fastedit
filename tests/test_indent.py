"""Comprehensive tests for fastedit.inference.indent module.

Covers _escape_tags, _unescape_tags, _align_snippet_indent, and _realign_output
with thorough edge-case coverage.
"""

from __future__ import annotations

import pytest

from fastedit.inference.indent import (
    _escape_tags,
    _unescape_tags,
    _align_snippet_indent,
    _realign_output,
)


# ---------------------------------------------------------------------------
# Constants (mirrors the module's internal sentinel strings)
# ---------------------------------------------------------------------------
_TAG_OPEN = "<updated-code>"
_TAG_CLOSE = "</updated-code>"
_TAG_OPEN_SAFE = "__FASTEDIT_TAG_OPEN__"
_TAG_CLOSE_SAFE = "__FASTEDIT_TAG_CLOSE__"


# ===================================================================
# _escape_tags
# ===================================================================
class TestEscapeTags:
    """Tests for _escape_tags."""

    def test_no_tags_unchanged(self):
        text = "def foo():\n    return 42\n"
        assert _escape_tags(text) == text

    def test_open_tag_replaced(self):
        text = f"before {_TAG_OPEN} after"
        result = _escape_tags(text)
        assert _TAG_OPEN not in result
        assert _TAG_OPEN_SAFE in result

    def test_close_tag_replaced(self):
        text = f"before {_TAG_CLOSE} after"
        result = _escape_tags(text)
        assert _TAG_CLOSE not in result
        assert _TAG_CLOSE_SAFE in result

    def test_both_tags_replaced(self):
        text = f"{_TAG_OPEN}\ndef foo(): pass\n{_TAG_CLOSE}"
        result = _escape_tags(text)
        assert _TAG_OPEN not in result
        assert _TAG_CLOSE not in result
        assert _TAG_OPEN_SAFE in result
        assert _TAG_CLOSE_SAFE in result

    def test_multiple_open_tags(self):
        text = f"{_TAG_OPEN} and {_TAG_OPEN}"
        result = _escape_tags(text)
        assert result.count(_TAG_OPEN_SAFE) == 2
        assert _TAG_OPEN not in result

    def test_multiple_close_tags(self):
        text = f"{_TAG_CLOSE}{_TAG_CLOSE}{_TAG_CLOSE}"
        result = _escape_tags(text)
        assert result.count(_TAG_CLOSE_SAFE) == 3
        assert _TAG_CLOSE not in result

    def test_tags_embedded_in_string_literal(self):
        text = f'print("{_TAG_OPEN}hello{_TAG_CLOSE}")'
        result = _escape_tags(text)
        assert _TAG_OPEN not in result
        assert _TAG_CLOSE not in result
        assert f'print("{_TAG_OPEN_SAFE}hello{_TAG_CLOSE_SAFE}")' == result

    def test_empty_string(self):
        assert _escape_tags("") == ""

    def test_only_whitespace(self):
        text = "   \n\t\n  "
        assert _escape_tags(text) == text


# ===================================================================
# _unescape_tags
# ===================================================================
class TestUnescapeTags:
    """Tests for _unescape_tags."""

    def test_no_placeholders_unchanged(self):
        text = "def bar(): pass"
        assert _unescape_tags(text) == text

    def test_open_placeholder_restored(self):
        text = f"before {_TAG_OPEN_SAFE} after"
        result = _unescape_tags(text)
        assert _TAG_OPEN in result
        assert _TAG_OPEN_SAFE not in result

    def test_close_placeholder_restored(self):
        text = f"before {_TAG_CLOSE_SAFE} after"
        result = _unescape_tags(text)
        assert _TAG_CLOSE in result
        assert _TAG_CLOSE_SAFE not in result

    def test_both_placeholders_restored(self):
        text = f"{_TAG_OPEN_SAFE}\ncode\n{_TAG_CLOSE_SAFE}"
        result = _unescape_tags(text)
        assert result == f"{_TAG_OPEN}\ncode\n{_TAG_CLOSE}"

    def test_multiple_placeholders(self):
        text = f"{_TAG_OPEN_SAFE}{_TAG_OPEN_SAFE}"
        result = _unescape_tags(text)
        assert result == f"{_TAG_OPEN}{_TAG_OPEN}"

    def test_empty_string(self):
        assert _unescape_tags("") == ""


# ===================================================================
# Round-trip: escape -> unescape
# ===================================================================
class TestEscapeUnescapeRoundTrip:
    """Round-trip property: unescape(escape(x)) == x."""

    def test_round_trip_plain_text(self):
        text = "no tags here\njust code\n"
        assert _unescape_tags(_escape_tags(text)) == text

    def test_round_trip_with_both_tags(self):
        text = f"{_TAG_OPEN}\n    x = 1\n{_TAG_CLOSE}"
        assert _unescape_tags(_escape_tags(text)) == text

    def test_round_trip_tags_in_multiline(self):
        text = (
            f"line1\n"
            f"  {_TAG_OPEN} inline open\n"
            f"  {_TAG_CLOSE} inline close\n"
            f"line4\n"
        )
        assert _unescape_tags(_escape_tags(text)) == text

    def test_round_trip_empty(self):
        assert _unescape_tags(_escape_tags("")) == ""

    def test_round_trip_only_tags(self):
        text = f"{_TAG_OPEN}{_TAG_CLOSE}"
        assert _unescape_tags(_escape_tags(text)) == text

    def test_round_trip_multiple_interleaved_tags(self):
        text = f"{_TAG_OPEN}a{_TAG_CLOSE}b{_TAG_OPEN}c{_TAG_CLOSE}"
        assert _unescape_tags(_escape_tags(text)) == text


# ===================================================================
# _align_snippet_indent
# ===================================================================
class TestAlignSnippetIndent:
    """Tests for _align_snippet_indent."""

    def test_already_aligned_noop(self):
        snippet = "    def foo(self):\n        return 1\n"
        chunk = "    def bar(self):\n        return 2\n"
        assert _align_snippet_indent(snippet, chunk) == snippet

    def test_zero_to_four_indent(self):
        """Snippet at col-0, chunk at 4-space indent -> add 4 spaces."""
        snippet = "def foo():\n    return 1\n"
        chunk = "    def bar():\n        return 2\n"
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def foo():"
        assert lines[1] == "        return 1"

    def test_eight_to_four_indent(self):
        """Snippet at 8-space, chunk at 4-space -> remove 4 spaces."""
        snippet = "        def foo():\n            return 1\n"
        chunk = "    def bar():\n        return 2\n"
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def foo():"
        assert lines[1] == "        return 1"

    def test_blank_lines_preserved(self):
        """Blank lines should not get extra indentation added."""
        snippet = "def foo():\n\n    return 1\n"
        chunk = "    class C:\n        pass\n"
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines(keepends=True)
        # The blank line (index 1) should stay blank
        assert lines[1].strip() == ""

    def test_blank_lines_not_indented(self):
        """Blank lines remain exactly as they were (empty or whitespace-only)."""
        snippet = "def foo():\n    x = 1\n\n    return x\n"
        chunk = "    def bar():\n        pass\n"
        result = _align_snippet_indent(snippet, chunk)
        result_lines = result.splitlines()
        # Line index 2 is the blank line - must be empty
        assert result_lines[2] == ""

    def test_tab_based_indent_chunk(self):
        """Tab indent in chunk should be converted via expandtabs(4)."""
        snippet = "def foo():\n    return 1\n"
        chunk = "\tdef bar():\n\t\treturn 2\n"
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines()
        # Tab expands to 4 spaces, so delta = 4 - 0 = 4
        assert lines[0] == "    def foo():"

    def test_tab_based_indent_snippet(self):
        """Tab indent in snippet, space indent in chunk."""
        snippet = "\tdef foo():\n\t\treturn 1\n"
        chunk = "    def bar():\n        return 2\n"
        # Tab = 4 spaces, chunk = 4 spaces -> delta = 0
        # But the raw strings differ (tab vs space) so the first check
        # (chunk_indent == snippet_indent) fails, but expandtabs gives
        # delta == 0, so we hit the early return.
        result = _align_snippet_indent(snippet, chunk)
        assert result == snippet

    def test_single_line_snippet(self):
        snippet = "return 42\n"
        chunk = "        x = 1\n"
        result = _align_snippet_indent(snippet, chunk)
        assert result.splitlines()[0] == "        return 42"

    def test_empty_snippet(self):
        assert _align_snippet_indent("", "    def foo(): pass") == ""

    def test_empty_chunk(self):
        """Empty chunk means base indent is '', snippet should remain as-is if already at col-0."""
        snippet = "def foo():\n    pass\n"
        assert _align_snippet_indent(snippet, "") == snippet

    def test_snippet_all_blank_lines(self):
        """Snippet of only blank lines -> returned as-is (no non-blank line to compute base indent)."""
        snippet = "\n\n\n"
        chunk = "    def foo():\n        pass\n"
        result = _align_snippet_indent(snippet, chunk)
        assert result == snippet

    def test_mixed_indent_levels_shifted(self):
        """Each line in snippet shifts by the same delta."""
        snippet = "def outer():\n    def inner():\n        return 1\n"
        chunk = "    class C:\n        pass\n"
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def outer():"
        assert lines[1] == "        def inner():"
        assert lines[2] == "            return 1"

    def test_remove_more_than_available(self):
        """When delta is negative but a line has less indent than |delta|, only remove what's there."""
        snippet = "        x = 1\n    y = 2\n"  # 8-space and 4-space lines
        chunk = "z = 0\n"  # 0-space indent
        result = _align_snippet_indent(snippet, chunk)
        lines = result.splitlines()
        # delta = 0 - 8 = -8. First line: remove min(8, 8) = 8. Second: remove min(8, 4) = 4.
        assert lines[0] == "x = 1"
        assert lines[1] == "y = 2"

    def test_large_positive_delta(self):
        """12-space indent chunk, 0-space snippet."""
        snippet = "pass\n"
        chunk = "            deep_code()\n"
        result = _align_snippet_indent(snippet, chunk)
        assert result.splitlines()[0] == "            pass"


# ===================================================================
# _realign_output
# ===================================================================
class TestRealignOutput:
    """Tests for _realign_output."""

    def test_already_correct_noop(self):
        """Output indent matches chunk indent -> return as-is."""
        chunk = "    def foo(self):\n        return 1\n"
        output = "    def foo(self):\n        return 2\n"
        assert _realign_output(output, chunk) == output

    def test_first_line_lost_indent_body_correct(self):
        """Model stripped def line indent but body stays at correct indent.

        This is the common failure mode: model outputs the def/class line
        at col-0 but keeps the body at the right level.
        """
        chunk = "    def foo(self):\n        return 1\n"
        output = "def foo(self):\n        return 2\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def foo(self):"
        assert lines[1] == "        return 2"

    def test_first_line_lost_indent_with_leading_blank(self):
        """Leading blank line before the def line that lost its indent."""
        chunk = "    def foo(self):\n        return 1\n"
        output = "\ndef foo(self):\n        return 2\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        # Blank line stays blank
        assert lines[0] == ""
        # First non-blank line gets its indent fixed
        assert lines[1] == "    def foo(self):"
        # Body was already correct
        assert lines[2] == "        return 2"

    def test_uniform_shift_when_body_also_wrong(self):
        """Both first line and body are at wrong indent -> uniform shift."""
        chunk = "    def foo(self):\n        return 1\n"
        output = "def foo(self):\n    return 2\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        # Uniform shift by +4 spaces
        assert lines[0] == "    def foo(self):"
        assert lines[1] == "        return 2"

    def test_output_only_blank_lines(self):
        """Output with only blank lines -> all-blank has indent 0.

        chunk indent = 4, output indent = 0 -> delta = 4.
        No body lines exist, so body indent check returns -1 for both.
        chunk_body_indent (-1) != output_body_indent (-1)... actually both are -1,
        but the condition requires chunk_body_indent >= 0, so we fall through
        to uniform shift via _align_snippet_indent, which preserves blank lines.
        """
        chunk = "    def foo():\n        pass\n"
        output = "\n\n\n"
        result = _realign_output(output, chunk)
        # _align_snippet_indent with all-blank snippet returns it as-is
        assert result == "\n\n\n"

    def test_single_line_output(self):
        """Single non-blank line gets its indent fixed."""
        chunk = "        return 42\n"
        output = "return 99\n"
        result = _realign_output(output, chunk)
        assert result.splitlines()[0] == "        return 99"

    def test_output_more_indent_than_chunk(self):
        """Model added extra indent to first line, body correct."""
        chunk = "    x = 1\n    y = 2\n"
        output = "        x = 1\n    y = 2\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        # chunk indent = 4, output indent = 8, delta = -4
        # body: chunk body indent = 4, output body indent = 4 -> match
        # So only first line gets fixed: remove 4 spaces
        assert lines[0] == "    x = 1"
        assert lines[1] == "    y = 2"

    def test_class_method_def_line_stripped(self):
        """The common model failure: def line at col-0, body at class-method level.

        class Foo:
            def bar(self):       # chunk has 4-space indent
                return self.x    # body at 8-space indent

        Model outputs:
        def bar(self):           # 0-space (stripped)
                return self.x    # 8-space (correct)

        Expected: fix only the def line.
        """
        chunk = "    def bar(self):\n        return self.x\n"
        output = "def bar(self):\n        return self.x\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def bar(self):"
        assert lines[1] == "        return self.x"

    def test_multiline_body_first_line_only_fix(self):
        """Multi-line body, only first non-blank line is wrong."""
        chunk = (
            "    def compute(self):\n"
            "        a = 1\n"
            "        b = 2\n"
            "        return a + b\n"
        )
        output = (
            "def compute(self):\n"
            "        a = 10\n"
            "        b = 20\n"
            "        return a + b\n"
        )
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def compute(self):"
        assert lines[1] == "        a = 10"
        assert lines[2] == "        b = 20"
        assert lines[3] == "        return a + b"

    def test_empty_output(self):
        """Empty model output -> both indent 0 -> return as-is."""
        chunk = "    def foo():\n        pass\n"
        assert _realign_output("", chunk) == ""

    def test_empty_chunk(self):
        """Empty chunk -> chunk indent 0, output indent 0 -> return as-is."""
        output = "def foo():\n    pass\n"
        assert _realign_output(output, "") == output

    def test_both_empty(self):
        assert _realign_output("", "") == ""

    def test_deeply_nested_uniform_shift(self):
        """Both first line and body are uniformly wrong for deeply nested code."""
        chunk = "            if cond:\n                do_thing()\n"
        output = "    if cond:\n        do_thing()\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        # chunk first indent=12, output first indent=4 -> delta=8
        # chunk body indent=16, output body indent=8 -> mismatch -> uniform shift
        assert lines[0] == "            if cond:"
        assert lines[1] == "                do_thing()"

    def test_no_trailing_newline(self):
        """Output without trailing newline still works."""
        chunk = "    x = 1"
        output = "x = 2"
        result = _realign_output(output, chunk)
        assert result == "    x = 2"

    def test_first_line_fix_preserves_blank_lines(self):
        """When fixing only the first line, blank lines in body are preserved."""
        chunk = "    def foo():\n        a = 1\n\n        return a\n"
        output = "def foo():\n        a = 1\n\n        return a\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        assert lines[0] == "    def foo():"
        assert lines[1] == "        a = 1"
        assert lines[2] == ""
        assert lines[3] == "        return a"

    def test_output_indent_exceeds_chunk_uniform_shift(self):
        """Output at 8-space, chunk at 0-space, body also wrong -> uniform shift."""
        chunk = "def foo():\n    return 1\n"
        output = "        def foo():\n            return 1\n"
        result = _realign_output(output, chunk)
        lines = result.splitlines()
        assert lines[0] == "def foo():"
        assert lines[1] == "    return 1"

    def test_single_line_output_no_body(self):
        """Single line: no body to compare, falls through to uniform shift."""
        chunk = "    pass\n"
        output = "pass\n"
        result = _realign_output(output, chunk)
        assert result == "    pass\n"
