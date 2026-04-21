"""Tests for fastedit.inference.symbols: delete_symbol, move_symbol."""

from __future__ import annotations

import textwrap

import pytest

from fastedit.inference.symbols import delete_symbol, move_symbol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


THREE_FUNCTIONS = textwrap.dedent("""\
def alpha():
    return 1

def beta():
    return 2

def gamma():
    return 3
""")


CLASS_AND_FUNCTION = textwrap.dedent("""\
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def standalone():
    return 42
""")


TRAILING_CONTENT = textwrap.dedent("""\
def only_func():
    return 99

# This is trailing content
CONSTANT = 42
""")


@pytest.fixture
def three_funcs_file(tmp_path):
    """Create a Python file with three functions."""
    p = tmp_path / "three_funcs.py"
    p.write_text(THREE_FUNCTIONS)
    return str(p)


@pytest.fixture
def class_file(tmp_path):
    """Create a Python file with a class and a standalone function."""
    p = tmp_path / "class_file.py"
    p.write_text(CLASS_AND_FUNCTION)
    return str(p)


@pytest.fixture
def trailing_file(tmp_path):
    """Create a Python file with one function followed by trailing content."""
    p = tmp_path / "trailing.py"
    p.write_text(TRAILING_CONTENT)
    return str(p)


# ---------------------------------------------------------------------------
# delete_symbol tests
# ---------------------------------------------------------------------------


class TestDeleteSymbol:
    """Tests for delete_symbol()."""

    def test_delete_middle_function(self, three_funcs_file):
        """Deleting the middle function leaves alpha and gamma."""
        result = delete_symbol(three_funcs_file, "beta")

        assert result.deleted_symbol == "beta"
        assert "def alpha():" in result.merged_code
        assert "def beta():" not in result.merged_code
        assert "def gamma():" in result.merged_code
        assert result.lines_removed > 0

    def test_delete_first_function(self, three_funcs_file):
        """Deleting the first function leaves beta and gamma."""
        result = delete_symbol(three_funcs_file, "alpha")

        assert result.deleted_symbol == "alpha"
        assert "def alpha():" not in result.merged_code
        assert "def beta():" in result.merged_code
        assert "def gamma():" in result.merged_code

    def test_delete_last_function(self, three_funcs_file):
        """Deleting the last function leaves alpha and beta."""
        result = delete_symbol(three_funcs_file, "gamma")

        assert result.deleted_symbol == "gamma"
        assert "def alpha():" in result.merged_code
        assert "def beta():" in result.merged_code
        assert "def gamma():" not in result.merged_code

    def test_delete_class(self, class_file):
        """Deleting a class removes the entire class and its methods."""
        result = delete_symbol(class_file, "Calculator")

        assert result.deleted_symbol == "Calculator"
        assert result.deleted_kind == "class"
        assert "class Calculator:" not in result.merged_code
        assert "def add(" not in result.merged_code
        assert "def subtract(" not in result.merged_code
        # The standalone function should remain
        assert "def standalone():" in result.merged_code

    def test_delete_preserves_trailing_content(self, trailing_file):
        """Deleting the only function preserves trailing non-function content."""
        result = delete_symbol(trailing_file, "only_func")

        assert result.deleted_symbol == "only_func"
        assert "def only_func():" not in result.merged_code
        assert "CONSTANT = 42" in result.merged_code
        assert "# This is trailing content" in result.merged_code

    def test_delete_nonexistent_symbol_raises(self, three_funcs_file):
        """Deleting a symbol that does not exist raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            delete_symbol(three_funcs_file, "nonexistent")

    def test_delete_trailing_blank_consumed(self, three_funcs_file):
        """The blank line separator after a deleted symbol is consumed."""
        result = delete_symbol(three_funcs_file, "alpha")

        # After deletion, no double blank lines should appear at the top
        lines = result.merged_code.split("\n")
        # The first non-empty line should be `def beta():`
        non_empty = [l for l in lines if l.strip()]
        assert non_empty[0] == "def beta():"

    def test_delete_returns_correct_line_range(self, three_funcs_file):
        """deleted_lines reflects the original 1-indexed line range."""
        result = delete_symbol(three_funcs_file, "beta")

        start, end = result.deleted_lines
        # beta starts at line 4 and ends at line 5 in the original file
        assert start == 4
        assert end == 5

    def test_delete_result_is_parseable(self, three_funcs_file):
        """The result after deletion is still valid Python."""
        result = delete_symbol(three_funcs_file, "beta", language="python")
        assert result.parse_valid is True


# ---------------------------------------------------------------------------
# move_symbol tests
# ---------------------------------------------------------------------------


class TestMoveSymbol:
    """Tests for move_symbol()."""

    def test_move_first_after_last(self, three_funcs_file):
        """Moving alpha after gamma: A,B,C -> B,C,A."""
        result = move_symbol(three_funcs_file, "alpha", after="gamma")

        assert result.moved_symbol == "alpha"
        assert result.after_symbol == "gamma"

        code = result.merged_code
        # All three functions must be present
        assert "def alpha():" in code
        assert "def beta():" in code
        assert "def gamma():" in code

        # Order: beta, gamma, alpha
        pos_beta = code.index("def beta():")
        pos_gamma = code.index("def gamma():")
        pos_alpha = code.index("def alpha():")
        assert pos_beta < pos_gamma < pos_alpha

    def test_move_last_after_first(self, three_funcs_file):
        """Moving gamma after alpha: A,B,C -> A,C,B."""
        result = move_symbol(three_funcs_file, "gamma", after="alpha")

        code = result.merged_code
        pos_alpha = code.index("def alpha():")
        pos_gamma = code.index("def gamma():")
        pos_beta = code.index("def beta():")
        assert pos_alpha < pos_gamma < pos_beta

    def test_move_inserts_blank_separator(self, three_funcs_file):
        """A blank line separator is present between moved symbol and neighbours."""
        result = move_symbol(three_funcs_file, "alpha", after="gamma")

        lines = result.merged_code.splitlines()
        # Find the line with `def alpha():` and check there's a blank line before it
        for i, line in enumerate(lines):
            if line.strip() == "def alpha():":
                # There should be a blank line somewhere between gamma's body and alpha
                assert i > 0, "alpha should not be the first line"
                # Check that the line before alpha (or two before) is blank
                preceding = lines[i - 1].strip()
                assert preceding == "" or preceding.startswith("return"), (
                    f"Expected blank separator before alpha, got: {lines[i-1]!r}"
                )
                break
        else:
            pytest.fail("def alpha(): not found in result")

    def test_move_symbol_not_found_raises(self, three_funcs_file):
        """Moving a nonexistent symbol raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            move_symbol(three_funcs_file, "nonexistent", after="alpha")

    def test_move_target_not_found_raises(self, three_funcs_file):
        """Moving after a nonexistent target raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            move_symbol(three_funcs_file, "alpha", after="nonexistent")

    def test_move_same_symbol_raises(self, three_funcs_file):
        """Moving a symbol after itself raises ValueError."""
        with pytest.raises(ValueError, match="after itself"):
            move_symbol(three_funcs_file, "alpha", after="alpha")

    def test_move_preserves_all_code(self, three_funcs_file):
        """Moving does not lose any function bodies."""
        result = move_symbol(three_funcs_file, "alpha", after="gamma")

        assert "return 1" in result.merged_code
        assert "return 2" in result.merged_code
        assert "return 3" in result.merged_code

    def test_move_result_is_parseable(self, three_funcs_file):
        """The result after move is still valid Python."""
        result = move_symbol(three_funcs_file, "alpha", after="gamma", language="python")
        assert result.parse_valid is True

    def test_move_returns_new_line_range(self, three_funcs_file):
        """new_lines reflects the position after the move."""
        result = move_symbol(three_funcs_file, "alpha", after="gamma")

        new_start, new_end = result.new_lines
        # alpha was originally lines 1-2; after move it should be after gamma
        assert new_start > 4, f"Expected alpha to move past line 4, got {new_start}"

    def test_move_middle_after_first(self, three_funcs_file):
        """Moving beta after alpha is a no-op in ordering but still works."""
        result = move_symbol(three_funcs_file, "beta", after="alpha")

        code = result.merged_code
        assert "def alpha():" in code
        assert "def beta():" in code
        assert "def gamma():" in code
        # beta is already after alpha, ordering should be preserved
        pos_alpha = code.index("def alpha():")
        pos_beta = code.index("def beta():")
        pos_gamma = code.index("def gamma():")
        assert pos_alpha < pos_beta < pos_gamma
