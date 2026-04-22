"""Comprehensive tests for fastedit.inference.text_match module.

Covers:
  - _is_marker: marker detection for various comment styles
  - _adjust_indent: indentation adjustment relative to context anchors
  - deterministic_edit: the main text-match editing function
"""

from __future__ import annotations

import textwrap

import pytest

from fastedit.inference.text_match import _adjust_indent, _is_marker, deterministic_edit


# ---------------------------------------------------------------------------
# _is_marker
# ---------------------------------------------------------------------------

class TestIsMarker:
    """Tests for the _is_marker helper."""

    def test_hash_existing_code(self):
        assert _is_marker("    # ... existing code ...") is True

    def test_slash_existing_code(self):
        assert _is_marker("    // ... existing code ...") is True

    def test_slash_shorthand(self):
        assert _is_marker("    // ...") is True

    def test_hash_shorthand(self):
        assert _is_marker("    # ...") is True

    def test_marker_no_leading_whitespace(self):
        assert _is_marker("# ... existing code ...") is True

    def test_marker_slash_no_leading_whitespace(self):
        assert _is_marker("// ...") is True

    def test_regular_code_line(self):
        assert _is_marker("    x = foo(bar)") is False

    def test_blank_line(self):
        assert _is_marker("") is False

    def test_whitespace_only(self):
        assert _is_marker("    ") is False

    def test_word_existing_not_marker(self):
        # Contains the word "existing" but not in the marker phrase format
        assert _is_marker("    existing = True") is False

    def test_comment_without_ellipsis(self):
        assert _is_marker("# some regular comment") is False

    def test_triple_dot_in_code(self):
        # Ellipsis literal in Python (the `...` object)
        assert _is_marker("    ...") is False

    def test_partial_marker_phrase(self):
        # "existing code" without the surrounding dots
        assert _is_marker("# existing code") is False

    def test_marker_embedded_in_longer_line(self):
        # Marker phrase is a substring
        assert _is_marker("    # ... existing code ... (keep this)") is True


# ---------------------------------------------------------------------------
# _adjust_indent
# ---------------------------------------------------------------------------

class TestAdjustIndent:
    """Tests for _adjust_indent indentation correction."""

    def test_same_indent_no_change(self):
        # Snippet and original both at 4 spaces
        orig_lines = ["    def foo():", "        pass"]
        snip_raw = ["    def foo():", "        new_line = 1"]
        result = _adjust_indent("        new_line = 1", 0, 0, snip_raw, orig_lines)
        assert result == "        new_line = 1"

    def test_original_more_indented(self):
        # Original at 8 spaces, snippet at 4 spaces -> new line gains 4 spaces
        orig_lines = ["        def foo():"]
        snip_raw = ["    def foo():"]
        # new_line has 8 spaces in snippet context
        result = _adjust_indent("        x = 1", 0, 0, snip_raw, orig_lines)
        # indent_diff = 8 - 4 = 4, curr_indent = 8, target = 12
        assert result == "            x = 1"

    def test_original_less_indented(self):
        # Original at 0 spaces, snippet at 4 spaces -> new line loses 4 spaces
        orig_lines = ["def foo():"]
        snip_raw = ["    def foo():"]
        # new_line has 8 spaces in snippet
        result = _adjust_indent("        x = 1", 0, 0, snip_raw, orig_lines)
        # indent_diff = 0 - 4 = -4, curr_indent = 8, target = 4
        assert result == "    x = 1"

    def test_never_negative_indent(self):
        # Even if math says negative, clamp to 0
        orig_lines = ["def foo():"]
        snip_raw = ["        def foo():"]
        # indent_diff = 0 - 8 = -8, curr_indent = 4, target = max(0, -4) = 0
        result = _adjust_indent("    x = 1", 0, 0, snip_raw, orig_lines)
        assert result == "x = 1"

    def test_new_line_zero_indent(self):
        # new_line already at 0, indent_diff is 4 -> gains 4
        orig_lines = ["    def foo():"]
        snip_raw = ["def foo():"]
        result = _adjust_indent("x = 1", 0, 0, snip_raw, orig_lines)
        # indent_diff = 4 - 0 = 4, curr_indent = 0, target = 4
        assert result == "    x = 1"

    def test_new_line_zero_indent_stays_zero(self):
        # new_line at 0, indent_diff is 0 -> stays at 0
        orig_lines = ["def foo():"]
        snip_raw = ["def foo():"]
        result = _adjust_indent("x = 1", 0, 0, snip_raw, orig_lines)
        assert result == "x = 1"

    def test_tab_stripping(self):
        # Tabs are treated as characters for lstrip, but the output uses spaces
        # _adjust_indent measures indent via len(line) - len(line.lstrip())
        orig_lines = ["    def foo():"]
        snip_raw = ["    def foo():"]
        result = _adjust_indent("    return 42", 0, 0, snip_raw, orig_lines)
        assert result == "    return 42"

    def test_uses_ref_indices_correctly(self):
        # ref_orig_idx and ref_snip_idx point into different positions
        orig_lines = ["class C:", "    def foo():", "        pass"]
        snip_raw = ["def foo():", "    new_line = 1"]
        # ref_orig_idx=1 (orig "    def foo():"), ref_snip_idx=0 (snip "def foo():")
        result = _adjust_indent("    new_line = 1", 1, 0, snip_raw, orig_lines)
        # indent_diff = 4 - 0 = 4, curr_indent = 4, target = 8
        assert result == "        new_line = 1"


# ---------------------------------------------------------------------------
# deterministic_edit
# ---------------------------------------------------------------------------

class TestDeterministicEditBasic:
    """Basic deterministic_edit tests."""

    def test_simple_single_line_addition(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                return x
        """)
        snippet = textwrap.dedent("""\
            def foo():
                x = 1
                y = 2
                return x
        """).rstrip("\n")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "y = 2" in result
        assert "x = 1" in result
        assert "return x" in result

    def test_returns_none_with_zero_context_anchors(self):
        original = "def foo():\n    return 1\n"
        snippet = "completely_different_code()\nanother_line()"
        result = deterministic_edit(original, snippet)
        assert result is None

    def test_returns_none_with_one_context_anchor(self):
        original = "def foo():\n    return 1\n"
        snippet = "def foo():\n    new_stuff()"
        result = deterministic_edit(original, snippet)
        # Only "def foo():" matches -- 1 anchor, need >= 2
        assert result is None

    def test_trailing_newline_preserved(self):
        original = "def foo():\n    x = 1\n    return x\n"
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert result.endswith("\n")

    def test_no_trailing_newline_when_original_has_none(self):
        original = "def foo():\n    x = 1\n    return x"
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert not result.endswith("\n")


class TestDeterministicEditReplaceMode:
    """Tests for replace mode (no marker, drop gap lines)."""

    def test_replace_single_line(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                y = old_value
                return x + y""")
        snippet = textwrap.dedent("""\
            def foo():
                x = 1
                y = new_value
                return x + y""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "y = new_value" in result
        assert "y = old_value" not in result

    def test_replace_drops_gap_lines(self):
        # Lines between two context anchors are dropped (replaced)
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                return a""")
        # Snippet keeps a=1 and return, drops b and c, adds z
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                z = 99
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "z = 99" in result
        assert "b = 2" not in result
        assert "c = 3" not in result

    def test_large_gap_without_marker_returns_none(self):
        # A gap larger than max_drop_gap without marker -> None
        lines = ["def foo():"]
        for i in range(25):
            lines.append(f"    line_{i} = {i}")
        lines.append("    return 0")
        original = "\n".join(lines)

        snippet = "def foo():\n    return 0"
        result = deterministic_edit(original, snippet, max_drop_gap=20)
        assert result is None

    def test_gap_at_max_drop_gap_succeeds(self):
        # Gap of exactly max_drop_gap should succeed
        lines = ["def foo():"]
        for i in range(20):
            lines.append(f"    line_{i} = {i}")
        lines.append("    return 0")
        original = "\n".join(lines)

        snippet = "def foo():\n    return 0"
        result = deterministic_edit(original, snippet, max_drop_gap=20)
        assert result is not None
        assert "return 0" in result

    def test_gap_just_over_max_drop_gap_returns_none(self):
        lines = ["def foo():"]
        for i in range(21):
            lines.append(f"    line_{i} = {i}")
        lines.append("    return 0")
        original = "\n".join(lines)

        snippet = "def foo():\n    return 0"
        result = deterministic_edit(original, snippet, max_drop_gap=20)
        assert result is None


class TestDeterministicEditMarkerMode:
    """Tests for marker mode (... existing code ... preserves gap)."""

    def test_marker_preserves_gap_lines(self):
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                return a""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ... existing code ...
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "b = 2" in result
        assert "c = 3" in result
        assert "a = 1" in result
        assert "return a" in result
        # Marker itself should NOT appear in the result
        assert "... existing code ..." not in result

    def test_new_lines_before_marker(self):
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                return a""")
        # New line before marker -> inserted before the gap
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                z = 99
                # ... existing code ...
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        z_idx = next(i for i, ln in enumerate(lines) if "z = 99" in ln)
        b_idx = next(i for i, ln in enumerate(lines) if "b = 2" in ln)
        assert z_idx < b_idx, "new line should appear before the preserved gap"

    def test_new_lines_after_marker(self):
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                return a""")
        # New line after marker -> inserted after the gap
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ... existing code ...
                z = 99
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        z_idx = next(i for i, ln in enumerate(lines) if "z = 99" in ln)
        c_idx = next(i for i, ln in enumerate(lines) if "c = 3" in ln)
        assert z_idx > c_idx, "new line should appear after the preserved gap"

    def test_marker_allows_large_gap(self):
        # With a marker, even large gaps are preserved (not rejected)
        lines = ["def foo():"]
        for i in range(50):
            lines.append(f"    line_{i} = {i}")
        lines.append("    return 0")
        original = "\n".join(lines)

        snippet = "def foo():\n    # ... existing code ...\n    return 0"
        result = deterministic_edit(original, snippet, max_drop_gap=5)
        assert result is not None
        # All 50 lines should be preserved
        for i in range(50):
            assert f"line_{i} = {i}" in result

    def test_slash_marker_works(self):
        original = textwrap.dedent("""\
            function foo() {
                let a = 1;
                let b = 2;
                let c = 3;
                return a;
            }""")
        snippet = textwrap.dedent("""\
            function foo() {
                let a = 1;
                // ... existing code ...
                return a;
            }""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "let b = 2;" in result
        assert "let c = 3;" in result

    def test_shorthand_marker_hash(self):
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                return a""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ...
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "b = 2" in result
        assert "c = 3" in result

    def test_shorthand_marker_slash(self):
        original = textwrap.dedent("""\
            function foo() {
                let a = 1;
                let b = 2;
                return a;
            }""")
        snippet = textwrap.dedent("""\
            function foo() {
                let a = 1;
                // ...
                return a;
            }""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "let b = 2;" in result

    def test_multiple_markers_in_snippet(self):
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                d = 4
                e = 5
                return a""")
        # Two markers preserving two separate gaps
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ... existing code ...
                c = 3
                # ... existing code ...
                return a""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "b = 2" in result
        assert "d = 4" in result
        assert "e = 5" in result


class TestDeterministicEditLeadingTrailing:
    """Tests for leading/trailing new lines outside context anchors."""

    def test_leading_new_lines_before_first_anchor(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                return x""")
        # "import os" is before the first context anchor "def foo():"
        snippet = "import os\ndef foo():\n    x = 1\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        # "import os" should appear before "def foo():"
        import_idx = next(i for i, ln in enumerate(lines) if "import os" in ln)
        def_idx = next(i for i, ln in enumerate(lines) if "def foo():" in ln)
        assert import_idx < def_idx

    def test_trailing_new_lines_after_last_anchor(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                return x""")
        snippet = "def foo():\n    x = 1\n    return x\n    print('done')"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "print('done')" in result
        lines = result.splitlines()
        ret_idx = next(i for i, ln in enumerate(lines) if "return x" in ln)
        print_idx = next(i for i, ln in enumerate(lines) if "print('done')" in ln)
        assert print_idx > ret_idx

    def test_trailing_marker_preserves_suffix(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                y = 2
                return x""")
        # Marker after last context anchor -> preserve rest of original
        snippet = textwrap.dedent("""\
            def foo():
                x = 1
                # ... existing code ...""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "y = 2" in result
        assert "return x" in result


class TestDeterministicEditIndentAdjustment:
    """Tests for indent adjustment in deterministic_edit."""

    def test_snippet_at_different_indent_adjusts_new_lines(self):
        # Original is indented at 4 spaces (method in a class)
        original = textwrap.dedent("""\
            class C:
                def foo(self):
                    x = 1
                    return x""")
        # Snippet refers to the same code but at 0-indent level
        snippet = "def foo(self):\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        y_line = next(ln for ln in lines if "y = 2" in ln)
        # y=2 should be at 8 spaces (same level as x=1 in original)
        assert y_line == "        y = 2"

    def test_no_indent_change_when_matching(self):
        original = "def foo():\n    x = 1\n    return x"
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        y_line = next(ln for ln in lines if "y = 2" in ln)
        assert y_line == "    y = 2"


class TestDeterministicEditRealWorld:
    """Real-world-like examples."""

    def test_add_logging_to_function(self):
        original = textwrap.dedent("""\
            def process(data):
                result = transform(data)
                validate(result)
                return result
        """)
        snippet = textwrap.dedent("""\
            def process(data):
                result = transform(data)
                logger.info("Transformed: %s", result)
                validate(result)
                return result""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert 'logger.info("Transformed: %s", result)' in result
        # Order preserved
        lines = result.splitlines()
        transform_idx = next(i for i, ln in enumerate(lines) if "transform(data)" in ln)
        log_idx = next(i for i, ln in enumerate(lines) if "logger.info" in ln)
        validate_idx = next(i for i, ln in enumerate(lines) if "validate(result)" in ln)
        assert transform_idx < log_idx < validate_idx

    def test_modify_return_statement(self):
        # To replace "return total" with "return total * 2", the snippet
        # must include the original return line as context so the new line
        # can be placed relative to it. Without the old line as context,
        # text-match treats the new line as an addition, not a replacement.
        original = textwrap.dedent("""\
            def compute(x, y):
                total = x + y
                avg = total / 2
                return total
        """)
        # Snippet: anchors are def, total, return total. Replace avg line.
        snippet = textwrap.dedent("""\
            def compute(x, y):
                total = x + y
                return total * 2
                return total""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "return total * 2" in result
        # The avg line was in the gap (no marker) so it's dropped (replace mode)
        assert "avg = total / 2" not in result

    def test_replace_line_by_dropping_gap(self):
        # The natural way to "replace" a line: include anchors on both sides,
        # new line replaces the gap between them.
        original = textwrap.dedent("""\
            def compute(x, y):
                total = x + y
                result = total
                return result
        """)
        snippet = textwrap.dedent("""\
            def compute(x, y):
                total = x + y
                result = total * 2
                return result""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "result = total * 2" in result
        assert "result = total\n" not in result

    def test_add_method_with_marker_in_class(self):
        original = textwrap.dedent("""\
            class Calculator:
                def __init__(self):
                    self.value = 0

                def add(self, n):
                    self.value += n
                    return self

                def reset(self):
                    self.value = 0
        """)
        snippet = textwrap.dedent("""\
                def add(self, n):
                    self.value += n
                    return self

                def subtract(self, n):
                    self.value -= n
                    return self

                def reset(self):
                    self.value = 0""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "def subtract(self, n):" in result
        assert "self.value -= n" in result
        # Original lines still present
        assert "class Calculator:" in result
        assert "def __init__(self):" in result
        assert "self.value = 0" in result

    def test_add_line_with_existing_code_marker(self):
        original = textwrap.dedent("""\
            def handler(request):
                user = get_user(request)
                perms = check_permissions(user)
                data = fetch_data(user)
                result = process(data)
                return jsonify(result)
        """)
        snippet = textwrap.dedent("""\
            def handler(request):
                user = get_user(request)
                # ... existing code ...
                result = process(data)
                log_result(result)
                return jsonify(result)""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # The gap (perms, data) should be preserved
        assert "perms = check_permissions(user)" in result
        assert "data = fetch_data(user)" in result
        # New line inserted
        assert "log_result(result)" in result
        lines = result.splitlines()
        data_idx = next(i for i, ln in enumerate(lines) if "fetch_data" in ln)
        log_idx = next(i for i, ln in enumerate(lines) if "log_result" in ln)
        result_idx = next(i for i, ln in enumerate(lines) if "process(data)" in ln)
        assert result_idx > data_idx
        assert log_idx > result_idx

    def test_prefix_lines_preserved(self):
        # Lines in the original before the first context anchor are kept
        original = textwrap.dedent("""\
            # Copyright 2026
            # License: MIT

            def foo():
                x = 1
                return x
        """)
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "# Copyright 2026" in result
        assert "# License: MIT" in result

    def test_suffix_lines_preserved(self):
        # Lines after the last context anchor are preserved
        original = textwrap.dedent("""\
            def foo():
                x = 1
                return x

            def bar():
                pass
        """)
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "def bar():" in result
        assert "    pass" in result

    def test_blank_lines_in_snippet_handled(self):
        original = textwrap.dedent("""\
            def foo():
                x = 1
                return x""")
        # Blank line between new additions
        snippet = "def foo():\n    x = 1\n\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "y = 2" in result

    def test_consecutive_context_lines_no_gap(self):
        # No gap between consecutive context anchors -> nothing to drop
        original = "def foo():\n    x = 1\n    return x"
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        expected = "def foo():\n    x = 1\n    y = 2\n    return x"
        assert result == expected


class TestDeterministicEditEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_original(self):
        result = deterministic_edit("", "some\ncode\nhere")
        assert result is None

    def test_empty_snippet(self):
        result = deterministic_edit("def foo():\n    pass", "")
        assert result is None

    def test_identical_original_and_snippet(self):
        code = "def foo():\n    x = 1\n    return x"
        result = deterministic_edit(code, code)
        assert result is not None
        assert result == code

    def test_max_drop_gap_zero(self):
        # max_drop_gap=0 means even 1 line gap without marker -> None
        original = "def foo():\n    a = 1\n    b = 2\n    return a"
        snippet = "def foo():\n    a = 1\n    return a"
        result = deterministic_edit(original, snippet, max_drop_gap=0)
        assert result is None

    def test_max_drop_gap_one(self):
        # max_drop_gap=1 allows exactly 1 line gap
        original = "def foo():\n    a = 1\n    b = 2\n    return a"
        snippet = "def foo():\n    a = 1\n    return a"
        result = deterministic_edit(original, snippet, max_drop_gap=1)
        assert result is not None
        assert "b = 2" not in result

    def test_only_markers_no_context(self):
        # Markers alone don't count as context anchors
        original = "def foo():\n    pass"
        snippet = "# ... existing code ...\n# ... existing code ..."
        result = deterministic_edit(original, snippet)
        assert result is None

    def test_forward_scan_order_preserving(self):
        # Context matching is forward-scan, order-preserving
        # If snippet reorders lines, later matches may fail
        original = "def foo():\n    a = 1\n    b = 2\n    c = 3\n    return a"
        # Reversed context -> "b = 2" would have to appear before "a = 1" in original
        # But forward scan will find a=1 at position 1, then b=2 at position 2
        # and c=3 at position 3 -- still works because the order in snippet matches original
        snippet = "def foo():\n    a = 1\n    c = 3\n    return a"
        result = deterministic_edit(original, snippet)
        assert result is not None
        # b=2 is in the gap between a=1 and c=3, dropped (replace mode)
        assert "b = 2" not in result

    def test_duplicate_lines_forward_scan(self):
        # Forward scan picks the first match from cursor
        original = "def foo():\n    x = 1\n    x = 1\n    return x"
        snippet = "def foo():\n    x = 1\n    y = 2\n    return x"
        result = deterministic_edit(original, snippet)
        assert result is not None
        # The first "x = 1" is matched, second "x = 1" is in the gap
        # Since no marker, gap is dropped (replace mode)
        assert "y = 2" in result

    def test_marker_in_trailing_section(self):
        # Marker after the last context anchor preserves suffix
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ... existing code ...""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "b = 2" in result
        assert "c = 3" in result

    def test_new_line_in_trailing_section(self):
        # New line after the last context anchor, no marker
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3
                d = 4""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "d = 4" in result
        lines = result.splitlines()
        c_idx = next(i for i, ln in enumerate(lines) if "c = 3" in ln)
        d_idx = next(i for i, ln in enumerate(lines) if "d = 4" in ln)
        assert d_idx == c_idx + 1

    def test_trailing_new_line_plus_marker(self):
        # Both a new line and a marker after last anchor
        # The marker emits the suffix, the new line is inserted at its position
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                z = 99
                # ... existing code ...""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # z=99 inserted before the preserved gap (b, c)
        lines = result.splitlines()
        z_idx = next(i for i, ln in enumerate(lines) if "z = 99" in ln)
        b_idx = next(i for i, ln in enumerate(lines) if "b = 2" in ln)
        assert z_idx < b_idx

    def test_new_line_after_trailing_marker(self):
        # New line AFTER the trailing marker
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                c = 3""")
        snippet = textwrap.dedent("""\
            def foo():
                a = 1
                # ... existing code ...
                d = 4""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # b, c preserved by marker; d=4 inserted after
        lines = result.splitlines()
        c_idx = next(i for i, ln in enumerate(lines) if "c = 3" in ln)
        d_idx = next(i for i, ln in enumerate(lines) if "d = 4" in ln)
        assert d_idx > c_idx

    def test_multiline_addition_between_anchors(self):
        original = "def foo():\n    start()\n    end()"
        snippet = "def foo():\n    start()\n    step_a()\n    step_b()\n    step_c()\n    end()"
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        start_idx = next(i for i, ln in enumerate(lines) if "start()" in ln)
        a_idx = next(i for i, ln in enumerate(lines) if "step_a()" in ln)
        b_idx = next(i for i, ln in enumerate(lines) if "step_b()" in ln)
        c_idx = next(i for i, ln in enumerate(lines) if "step_c()" in ln)
        end_idx = next(i for i, ln in enumerate(lines) if "end()" in ln)
        assert start_idx < a_idx < b_idx < c_idx < end_idx


class TestDeterministicEditMarkerGapIndentAdjust:
    """Bug 1 regression: preserved-gap lines must shift indent when a wrapper
    (try/except, if-guard, with-block) adds indentation around existing code.
    """

    def test_wrap_block_deeper_indent_try_except(self):
        # Wrapping body in try/except: the preserved gap must shift right
        # one indent level (4 spaces) to remain syntactically valid inside try.
        # We need >=2 context anchors surrounding the preserved gap with a
        # marker between them. Outer anchors (def ..., return ...) stay at
        # their original snippet-indent; the marker at the deeper indent
        # signals the gap must shift.
        original = textwrap.dedent("""\
            def load(path):
                opened = open_file(path)
                data = read(opened)
                cleaned = clean(data)
                return cleaned
        """)
        # Snippet: def + try: wrapper, marker preserves body, then return at
        # the orig indent level as a closing anchor.
        snippet = textwrap.dedent("""\
            def load(path):
                try:
                    # ... existing code ...
                except IOError:
                    cleaned = None
                return cleaned""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # The preserved gap (opened, data, cleaned) must be indented 8 spaces
        # (was 4), to sit correctly inside the try: block.
        assert "        opened = open_file(path)" in result
        assert "        data = read(opened)" in result
        assert "        cleaned = clean(data)" in result
        # Wrapper lines present at expected indent.
        assert "    try:" in result
        assert "    except IOError:" in result
        # Marker removed.
        assert "... existing code ..." not in result
        # Original 4-indent versions of the body should NOT appear (they got shifted).
        result_lines = result.splitlines()
        assert "    opened = open_file(path)" not in result_lines
        assert "    data = read(opened)" not in result_lines
        assert "    cleaned = clean(data)" not in result_lines

    def test_add_guard_shallower_indent_no_change(self):
        # A guard added at the SAME indent level as the anchor (no wrapping
        # shift) should not change the preserved gap lines.
        original = textwrap.dedent("""\
            def process(data):
                result = transform(data)
                validate(result)
                return result
        """)
        snippet = textwrap.dedent("""\
            def process(data):
                if data is None:
                    return None
                # ... existing code ...
                return result""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # Preserved lines stay at their original 4-space indent.
        assert "    result = transform(data)" in result
        assert "    validate(result)" in result

    def test_wrap_block_preserves_blank_lines(self):
        # Blank lines in the preserved gap must stay blank (not get indented).
        original = textwrap.dedent("""\
            def run():
                setup()

                work()
                finalize()
                return 0
        """)
        snippet = textwrap.dedent("""\
            def run():
                try:
                    # ... existing code ...
                except Exception:
                    return 1
                return 0""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        # The blank line between setup() and work() must remain blank (empty).
        assert "" in lines, "expected a blank line preserved in the gap"
        # Non-blank gap lines got shifted to 8-space indent.
        assert "        setup()" in result
        assert "        work()" in result
        assert "        finalize()" in result

    def test_trailing_marker_wrap_block_indents_suffix(self):
        # Same logic applies to the trailing-marker branch: when a wrapper
        # starts at deeper indent, preserved suffix lines must also shift.
        # Only one context anchor is needed for the trailing-marker branch when
        # there's a second anchor elsewhere (else deterministic_edit bails).
        # We use a 2-anchor snippet where the marker sits after the LAST ctx
        # anchor (so trailing-section logic applies) and extends indent deeper.
        original = textwrap.dedent("""\
            def go():
                prep()
                a = compute_a()
                b = compute_b()
                c = compute_c()
        """)
        # Snippet: wrap body starting AFTER prep() in a try block.
        # prep() + def go(): are context anchors; marker comes after prep().
        snippet = textwrap.dedent("""\
            def go():
                prep()
                try:
                    # ... existing code ...""")
        result = deterministic_edit(original, snippet)
        assert result is not None
        # Preserved suffix (a, b, c) must now be at 8-space indent (inside try:).
        assert "        a = compute_a()" in result
        assert "        b = compute_b()" in result
        assert "        c = compute_c()" in result
        assert "    try:" in result
        # Original 4-indent versions of the body should NOT appear (shifted).
        result_lines = result.splitlines()
        assert "    a = compute_a()" not in result_lines
        assert "    b = compute_b()" not in result_lines
