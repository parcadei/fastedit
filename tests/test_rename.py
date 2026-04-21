"""Tests for fastedit.inference.rename: do_rename and helpers."""

from __future__ import annotations

import textwrap

import pytest

from fastedit.inference.rename import (
    _collect_skip_ranges,
    _in_skip_zone,
    do_rename,
)


# ---------------------------------------------------------------------------
# _in_skip_zone tests
# ---------------------------------------------------------------------------


class TestInSkipZone:
    """Tests for _in_skip_zone()."""

    def test_inside_zone(self):
        """A range fully inside a skip zone returns True."""
        zones = [(10, 50)]
        assert _in_skip_zone(15, 20, zones) is True

    def test_outside_zone(self):
        """A range fully outside a skip zone returns False."""
        zones = [(10, 50)]
        assert _in_skip_zone(55, 60, zones) is False

    def test_exact_zone_boundaries(self):
        """A range matching the zone exactly returns True."""
        zones = [(10, 50)]
        assert _in_skip_zone(10, 50, zones) is True

    def test_partially_overlapping_not_contained(self):
        """A range that overlaps but is not fully contained returns False."""
        zones = [(10, 50)]
        # start is before zone start
        assert _in_skip_zone(5, 20, zones) is False

    def test_empty_zones(self):
        """No skip zones means nothing is skipped."""
        assert _in_skip_zone(0, 10, []) is False

    def test_multiple_zones_second_matches(self):
        """A range inside the second skip zone returns True."""
        zones = [(0, 10), (20, 40), (60, 80)]
        assert _in_skip_zone(25, 35, zones) is True


# ---------------------------------------------------------------------------
# _collect_skip_ranges tests (requires tree-sitter)
# ---------------------------------------------------------------------------


class TestCollectSkipRanges:
    """Tests for _collect_skip_ranges() using real tree-sitter parsing."""

    def _parse(self, code: str):
        """Parse Python code and return the root node."""
        from fastedit.data_gen.ast_analyzer import parse_code
        tree = parse_code(code, "python")
        return tree.root_node

    def test_string_creates_skip_range(self):
        """String literals create skip ranges."""
        code = 'x = "hello world"'
        root = self._parse(code)
        ranges: list[tuple[int, int]] = []
        _collect_skip_ranges(root, ranges)
        # There should be at least one skip range for the string content
        assert len(ranges) > 0
        # The string "hello world" should be within a skip range
        hello_start = code.encode("utf-8").index(b"hello world")
        hello_end = hello_start + len(b"hello world")
        assert any(s <= hello_start and e >= hello_end for s, e in ranges)

    def test_comment_creates_skip_range(self):
        """Comments create skip ranges."""
        code = "x = 1  # this is a comment"
        root = self._parse(code)
        ranges: list[tuple[int, int]] = []
        _collect_skip_ranges(root, ranges)
        # There should be a skip range covering the comment
        comment_start = code.encode("utf-8").index(b"# this is a comment")
        assert any(s <= comment_start for s, e in ranges)

    def test_no_strings_or_comments(self):
        """Code without strings or comments produces no skip ranges."""
        code = "x = 1 + 2"
        root = self._parse(code)
        ranges: list[tuple[int, int]] = []
        _collect_skip_ranges(root, ranges)
        assert len(ranges) == 0

    def test_multiline_string(self):
        """Triple-quoted strings create skip ranges."""
        code = 'x = """multi\nline\nstring"""'
        root = self._parse(code)
        ranges: list[tuple[int, int]] = []
        _collect_skip_ranges(root, ranges)
        assert len(ranges) > 0


# ---------------------------------------------------------------------------
# do_rename tests
# ---------------------------------------------------------------------------


class TestDoRename:
    """Tests for do_rename()."""

    def test_simple_rename(self):
        """Simple rename replaces all word-boundary matches."""
        code = textwrap.dedent("""\
        def get():
            return 1

        x = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch")
        assert "def fetch():" in renamed
        assert "x = fetch()" in renamed
        assert "def get():" not in renamed
        assert count == 2

    def test_word_boundary_preserves_longer_names(self):
        """Renaming 'get' does NOT touch 'get_all' or 'getter'."""
        code = textwrap.dedent("""\
        def get():
            pass

        def get_all():
            pass

        getter = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch")
        assert "def fetch():" in renamed
        assert "def get_all():" in renamed  # NOT renamed
        assert "getter = fetch()" in renamed  # get() call is renamed
        assert count == 2  # def get and get() call

    def test_preserves_strings_with_language(self):
        """With language='python', matches inside strings are skipped."""
        code = textwrap.dedent("""\
        def get():
            return 1

        msg = "call get here"
        result = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch", language="python")
        assert 'msg = "call fetch here"' not in renamed  # string preserved
        assert "result = fetch()" in renamed
        assert skipped >= 1

    def test_preserves_comments_with_language(self):
        """With language='python', matches inside comments are skipped."""
        code = textwrap.dedent("""\
        # call get to retrieve data
        def get():
            return 1

        x = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch", language="python")
        assert "# call get to retrieve data" in renamed  # comment preserved
        assert "def fetch():" in renamed
        assert "x = fetch()" in renamed
        assert skipped >= 1

    def test_no_matches_returns_original(self):
        """When there are no matches, returns original with 0 counts."""
        code = "x = 1\ny = 2\n"
        renamed, count, skipped = do_rename(code, "nonexistent", "replaced")
        assert renamed == code
        assert count == 0
        assert skipped == 0

    def test_multiple_occurrences_all_renamed(self):
        """Multiple code occurrences are all renamed."""
        code = textwrap.dedent("""\
        def calc(a, b):
            return calc_inner(a) + calc_inner(b)

        def calc_inner(x):
            return x * 2

        result = calc(1, 2)
        """)
        renamed, count, skipped = do_rename(code, "calc_inner", "compute")
        assert renamed.count("compute") == 3  # 2 calls + 1 def
        assert "calc_inner" not in renamed
        assert count == 3

    def test_language_none_falls_back_to_plain_regex(self):
        """With language=None, no skip zones are built (strings get renamed)."""
        code = textwrap.dedent("""\
        def get():
            return 1

        msg = "get value"
        x = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch", language=None)
        # Without tree-sitter, the string match is also renamed
        assert 'msg = "fetch value"' in renamed
        assert count >= 3  # def, string, call
        assert skipped == 0

    def test_realistic_mixed_code(self):
        """Realistic scenario: rename in definition, call, but not string/comment."""
        code = textwrap.dedent("""\
        def get_data(source):
            # get the data
            result = get(source)
            print("get done")
            return result
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch", language="python")

        # Only the call `get(source)` should be renamed
        assert "result = fetch(source)" in renamed
        # NOT these:
        assert "def get_data(source):" in renamed     # word boundary: get_data != get
        assert "# get the data" in renamed             # comment
        assert '"get done"' in renamed                 # string
        assert count == 1
        assert skipped >= 2  # comment + string

    def test_returns_correct_skipped_count(self):
        """Skipped count matches the number of matches inside skip zones."""
        code = textwrap.dedent("""\
        x = get()
        # get
        y = "get"
        z = get()
        """)
        renamed, count, skipped = do_rename(code, "get", "fetch", language="python")
        assert count == 2       # two get() calls
        assert skipped == 2     # comment + string

    def test_rename_preserves_indentation(self):
        """Renaming preserves leading whitespace on lines."""
        code = textwrap.dedent("""\
        class Foo:
            def bar(self):
                return bar()
        """)
        renamed, count, skipped = do_rename(code, "bar", "baz")
        assert "    def baz(self):" in renamed
        assert "        return baz()" in renamed

    def test_rename_with_special_regex_chars(self):
        """Names with characters that are special in regex are handled."""
        code = "x = __init__()\ny = __init__()\n"
        renamed, count, skipped = do_rename(code, "__init__", "__setup__")
        assert renamed.count("__setup__") == 2
        assert "__init__" not in renamed
        assert count == 2

    def test_empty_code(self):
        """Empty input returns empty output with 0 counts."""
        renamed, count, skipped = do_rename("", "get", "fetch")
        assert renamed == ""
        assert count == 0
        assert skipped == 0

    def test_unicode_content_preserved(self):
        """Unicode characters in the file are preserved during rename."""
        code = "# Calcul du co\u00fbt\ndef get():\n    return '\u00e9l\u00e8ve'\n\nx = get()\n"
        renamed, count, skipped = do_rename(code, "get", "fetch")
        assert "co\u00fbt" in renamed
        assert "\u00e9l\u00e8ve" in renamed
        assert "def fetch():" in renamed
        assert "x = fetch()" in renamed
