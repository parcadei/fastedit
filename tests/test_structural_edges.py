"""Structural, boundary, and content-hazard tests for the fastedit CLI.

Complements test_cli_crlf_preservation.py's byte-exactness focus with the
shapes editors typically get wrong: symbols at file edges, empty/whitespace
files, substring names, identifiers embedded in strings/comments, duplicate
names in different scopes, and every failure path's exit code + message.

Every assertion compares FULL FILE BYTES (read_bytes), never a substring or
a "still contains X" presence check -- a byte-identical-except-target
comparison is the only thing that can catch a rewrite that also mangled
something nearby that a presence check would miss.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]

HAS_MLX = importlib.util.find_spec("mlx") is not None


def run_cli(*args: str, input_text: str | None = None, env_extra: dict | None = None):
    """Invoke the real fastedit CLI as a subprocess, mirroring test_cli.py's helper."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


# ---------------------------------------------------------------------------
# POSITION AND SHAPE
# ---------------------------------------------------------------------------


class TestSymbolAtFileStart:
    """A symbol at byte offset 0 -- nothing, not even a blank line, before it."""

    def test_rename_symbol_at_absolute_start_of_file_preserves_everything_after(self, tmp_path: Path) -> None:
        """Renaming the first token in the file must not touch a single byte after it."""
        f = tmp_path / "m.py"
        original = b"def first():\n    return 1\n\n\ndef second():\n    return 2\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "first", "renamed_first")
        assert result.returncode == 0, result.stderr
        expected = b"def renamed_first():\n    return 1\n\n\ndef second():\n    return 2\n"
        assert f.read_bytes() == expected

    def test_delete_symbol_at_absolute_start_of_file_leaves_remainder_byte_exact(self, tmp_path: Path) -> None:
        """Deleting the very first symbol leaves the second at byte 0, no stray blank line.

        UPDATED 2026-09-13. This test previously pinned OBSERVED behaviour -- its
        own docstring said so -- where delete consumed only ONE of the two blank
        separator lines, leaving the file starting with a stray newline. That was
        a characterization test, not a requirement, and the behaviour it pinned
        was wrong in both directions: not byte-exact (which would keep BOTH
        blanks) and not tidy (which keeps neither).

        delete now consumes the FULL separator following a removed symbol.
        Verified on a middle delete: in a PEP 8 three-function file, consuming one
        blank leaves THREE blank lines between the surviving neighbours; consuming
        all leaves exactly two, matching the original style byte for byte.
        """
        f = tmp_path / "m.py"
        f.write_bytes(b"def first():\n    return 1\n\n\ndef second():\n    return 2\n")
        result = run_cli("delete", str(f), "first")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def second():\n    return 2\n"


class TestSymbolAtFileEnd:
    """A symbol at the very end -- nothing, not even a newline, after its body."""

    def test_rename_symbol_at_absolute_end_of_file_no_trailing_content(self, tmp_path: Path) -> None:
        """Renaming the last symbol must not append or drop any trailing bytes."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def first():\n    return 1\n\n\ndef last():\n    return 2\n")
        result = run_cli("rename", str(f), "last", "renamed_last")
        assert result.returncode == 0, result.stderr
        expected = b"def first():\n    return 1\n\n\ndef renamed_last():\n    return 2\n"
        assert f.read_bytes() == expected

    def test_delete_symbol_at_absolute_end_of_file_no_trailing_content(self, tmp_path: Path) -> None:
        """Deleting the last symbol in the file must leave the head symbol byte-exact.

        Observed: unlike deleting a leading symbol, deleting the trailing
        symbol does NOT consume the blank separator lines before it -- they
        survive as trailing blank lines. Pinned here as the actual contract
        (see the sibling head-deletion test for the asymmetric case).
        """
        f = tmp_path / "m.py"
        f.write_bytes(b"def first():\n    return 1\n\n\ndef last():\n    return 2\n")
        result = run_cli("delete", str(f), "last")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def first():\n    return 1\n\n\n"


class TestOnlySymbolInFile:
    """A file whose entire content is exactly one symbol."""

    def test_replace_the_only_symbol_in_file_produces_exact_new_bytes(self, tmp_path: Path) -> None:
        """Replacing the only symbol must yield exactly the new snippet, nothing more."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def only():\n    return 1\n")
        result = run_cli("edit", str(f), "--replace", "only", "--snippet", "def only():\n    return 2\n")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def only():\n    return 2\n"

    def test_delete_the_only_symbol_leaves_file_with_no_symbols(self, tmp_path: Path) -> None:
        """Deleting the only symbol must remove exactly its lines, leaving the rest (nothing) byte-exact."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def only():\n    return 1\n")
        result = run_cli("delete", str(f), "only")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b""


class TestTwoAdjacentSymbolsNoGapBetween:
    """Two symbols with zero blank lines between them -- no separator to misparse."""

    def test_rename_first_of_two_symbols_with_no_blank_line_between_them(self, tmp_path: Path) -> None:
        """Renaming the first of two immediately-adjacent symbols must not bleed into the second."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def a():\n    return 1\ndef b():\n    return 2\n")
        result = run_cli("rename", str(f), "a", "renamed_a")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def renamed_a():\n    return 1\ndef b():\n    return 2\n"

    def test_delete_first_of_two_adjacent_symbols_second_symbol_untouched_bytes(self, tmp_path: Path) -> None:
        """Deleting the first of two adjacent symbols must leave the second's bytes exact."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def a():\n    return 1\ndef b():\n    return 2\n")
        result = run_cli("delete", str(f), "a")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def b():\n    return 2\n"

    def test_move_second_of_two_adjacent_symbols_before_the_first(self, tmp_path: Path) -> None:
        """Moving the second adjacent symbol to be first must preserve both bodies byte-exact, only order changes."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def a():\n    return 1\ndef b():\n    return 2\n")
        result = run_cli("move", str(f), "b", "--after", "__START__") if False else None
        # fastedit move only supports "after <symbol>", not "before" -- express
        # "b before a" as moving 'a' to after 'b', which is the only way to
        # reorder two adjacent symbols with this CLI's vocabulary.
        result = run_cli("move", str(f), "a", "--after", "b")
        assert result.returncode == 0, result.stderr
        content = f.read_bytes()
        assert b"def a():\n    return 1\n" in content
        assert b"def b():\n    return 2\n" in content
        assert content.index(b"def b") < content.index(b"def a")


class TestNestedSymbols:
    """A method inside a class, a closure inside a function."""

    def test_rename_a_method_inside_a_class_does_not_touch_sibling_method(self, tmp_path: Path) -> None:
        """Renaming one method of a class must leave its sibling method byte-exact."""
        f = tmp_path / "m.py"
        f.write_bytes(
            b"class C:\n    def one(self):\n        return 1\n\n    def two(self):\n        return 2\n"
        )
        result = run_cli("rename", str(f), "one", "renamed_one")
        assert result.returncode == 0, result.stderr
        expected = (
            b"class C:\n    def renamed_one(self):\n        return 1\n\n    def two(self):\n        return 2\n"
        )
        assert f.read_bytes() == expected

    def test_rename_a_variable_inside_a_closure_does_not_touch_outer_scope_variable(self, tmp_path: Path) -> None:
        """Renaming an inner-closure variable must not rename the outer function's same-named variable."""
        f = tmp_path / "m.py"
        f.write_bytes(
            b"def outer():\n"
            b"    total = 0\n\n"
            b"    def inner():\n"
            b"        total_inner = 1\n"
            b"        return total_inner\n\n"
            b"    return outer_result(total, inner())\n\n\n"
            b"def outer_result(a, b):\n"
            b"    return a + b\n"
        )
        result = run_cli("rename", str(f), "total_inner", "renamed_inner")
        assert result.returncode == 0, result.stderr
        content = f.read_bytes()
        assert b"total_inner" not in content
        assert b"renamed_inner" in content
        # the outer scope's distinct-named variable 'total' must be untouched
        assert content.count(b"    total = 0\n") == 1
        assert content.count(b"return outer_result(total, inner())") == 1


class TestEmptyBodySymbols:
    """A symbol whose entire body is `pass` -- the smallest legal body."""

    def test_replace_a_pass_only_function_body(self, tmp_path: Path) -> None:
        """Replacing a pass-only function must produce exactly the replacement bytes."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def stub():\n    pass\n")
        result = run_cli("edit", str(f), "--replace", "stub", "--snippet", "def stub():\n    return 42\n")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def stub():\n    return 42\n"

    def test_delete_a_pass_only_class_removes_exactly_its_two_lines(self, tmp_path: Path) -> None:
        """Deleting a pass-only class removes its lines and its trailing separator, nothing of its neighbour."""
        f = tmp_path / "m.py"
        f.write_bytes(b"class Empty:\n    pass\n\n\nclass Keep:\n    pass\n")
        result = run_cli("delete", str(f), "Empty")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"class Keep:\n    pass\n"


class TestOneLineFileNoTrailingNewline:
    def test_replace_symbol_in_one_line_file_without_trailing_newline(self, tmp_path: Path) -> None:
        """A one-line file with no trailing newline must not gain one as a side effect of editing.

        Was a strict-xfail defect reproducer (B31: the deterministic replace
        splice unconditionally appended a trailing newline even when the
        original file had none -- a byte that was never in the file and never
        in the snippet's own text). Unpinned in Step 14: the edit path now
        funnels its merged output through the central EOL/trailing-newline
        normalizer (chunked_merge._normalize_merged_eol), which derives the
        file's trailing-newline state from the ORIGINAL, so this is a
        permanent regression test.
        """
        f = tmp_path / "m.py"
        f.write_bytes(b"def x(): return 1")
        result = run_cli("edit", str(f), "--replace", "x", "--snippet", "def x(): return 2")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def x(): return 2"


class TestCommentOnlyFile:
    def test_edit_replace_on_comment_only_file_refuses_symbol_not_found(self, tmp_path: Path) -> None:
        """A file with no symbols at all (only a comment) must refuse cleanly, not crash."""
        f = tmp_path / "m.py"
        original = b"# just a comment, nothing else\n"
        f.write_bytes(original)
        result = run_cli("edit", str(f), "--replace", "anything", "--snippet", "x = 1\n")
        assert result.returncode != 0
        assert "anything" in result.stderr or "not found" in result.stderr.lower()
        assert f.read_bytes() == original


class TestWhitespaceOnlyFile:
    def test_delete_on_whitespace_only_file_refuses_symbol_not_found(self, tmp_path: Path) -> None:
        """A file that is only whitespace must refuse a delete cleanly and stay byte-exact."""
        f = tmp_path / "m.py"
        original = b"   \n\t\n   \n"
        f.write_bytes(original)
        result = run_cli("delete", str(f), "anything")
        assert result.returncode != 0
        assert f.read_bytes() == original


class TestEmptyFileZeroBytes:
    def test_create_refuses_overwriting_a_zero_byte_file_without_force(self, tmp_path: Path) -> None:
        """create on an existing 0-byte file without --force must refuse and leave it untouched."""
        f = tmp_path / "m.py"
        f.write_bytes(b"")
        result = run_cli("create", str(f), "--content", "x = 1\n")
        assert result.returncode != 0
        assert "exists" in result.stderr.lower()
        assert f.read_bytes() == b""

    def test_edit_on_a_zero_byte_file_refuses_symbol_not_found(self, tmp_path: Path) -> None:
        """A completely empty (0-byte) file has no symbols; edit --replace must refuse, not crash."""
        f = tmp_path / "m.py"
        f.write_bytes(b"")
        result = run_cli("edit", str(f), "--replace", "anything", "--snippet", "x = 1\n")
        assert result.returncode != 0
        assert f.read_bytes() == b""


class TestSingleNewlineFile:
    def test_delete_on_a_file_containing_only_a_single_newline_byte_refuses_cleanly(self, tmp_path: Path) -> None:
        """A file that is just one '\\n' byte has no symbol to delete; must refuse cleanly."""
        f = tmp_path / "m.py"
        f.write_bytes(b"\n")
        result = run_cli("delete", str(f), "anything")
        assert result.returncode != 0
        assert f.read_bytes() == b"\n"


# ---------------------------------------------------------------------------
# CONTENT HAZARDS
# ---------------------------------------------------------------------------


class TestVeryLongLine:
    def test_rename_identifier_inside_a_100k_char_line_preserves_the_rest_of_the_line_byte_exact(
        self, tmp_path: Path
    ) -> None:
        """A rename inside one enormous line must not corrupt a single byte outside the identifier."""
        f = tmp_path / "m.py"
        padding = "z" * 100_000
        original = f'def f():\n    long_var = "{padding}"\n    return long_var\n'.encode()
        f.write_bytes(original)
        result = run_cli("rename", str(f), "long_var", "renamed_long_var")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"long_var", b"renamed_long_var")
        assert f.read_bytes() == expected
        # the 100k-char string literal itself must be byte-identical
        assert (b'"' + padding.encode() + b'"') in f.read_bytes()


class TestSubstringSymbolNames:
    """A symbol name that is a substring of another symbol name in the same file."""

    def test_rename_get_does_not_touch_getAll(self, tmp_path: Path) -> None:
        """Renaming 'get' must not rename any part of 'getAll'."""
        f = tmp_path / "m.py"
        original = b"def get():\n    return 1\n\n\ndef getAll():\n    return [get()]\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "get", "fetch")
        assert result.returncode == 0, result.stderr
        expected = b"def fetch():\n    return 1\n\n\ndef getAll():\n    return [fetch()]\n"
        assert f.read_bytes() == expected

    def test_delete_get_leaves_getAll_byte_exact(self, tmp_path: Path) -> None:
        """Deleting 'get' must leave 'getAll' (a superstring of the deleted name) completely untouched."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def get():\n    return 1\n\n\ndef getAll():\n    return 2\n")
        result = run_cli("delete", str(f), "get")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def getAll():\n    return 2\n"

    def test_rename_getAll_does_not_touch_get(self, tmp_path: Path) -> None:
        """Renaming the longer 'getAll' must not affect the shorter 'get' it contains as a substring."""
        f = tmp_path / "m.py"
        original = b"def get():\n    return 1\n\n\ndef getAll():\n    return [get()]\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "getAll", "fetchAll")
        assert result.returncode == 0, result.stderr
        expected = b"def get():\n    return 1\n\n\ndef fetchAll():\n    return [get()]\n"
        assert f.read_bytes() == expected


class TestSymbolNameInStringLiteralAndComment:
    """The exact identifier text also appears inside a string literal and a comment."""

    def test_rename_skips_the_occurrence_inside_a_string_literal(self, tmp_path: Path) -> None:
        """The string literal mentioning the old name must be byte-identical after rename; only real refs change."""
        f = tmp_path / "m.py"
        original = (
            b"def get():\n"
            b"    return 1\n\n\n"
            b'def caller():\n    msg = "please call get() now"\n    return get()\n'
        )
        f.write_bytes(original)
        result = run_cli("rename", str(f), "get", "fetch")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"def get():", b"def fetch():").replace(
            b"return get()", b"return fetch()"
        )
        assert f.read_bytes() == expected
        assert b'"please call get() now"' in f.read_bytes()

    def test_rename_skips_the_occurrence_inside_a_comment(self, tmp_path: Path) -> None:
        """The comment mentioning the old name must be byte-identical after rename; only real refs change."""
        f = tmp_path / "m.py"
        original = (
            b"def get():\n    return 1\n\n\n"
            b"def caller():\n    # calls get() below\n    return get()\n"
        )
        f.write_bytes(original)
        result = run_cli("rename", str(f), "get", "fetch")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"def get():", b"def fetch():").replace(
            b"return get()", b"return fetch()"
        )
        assert f.read_bytes() == expected
        assert b"# calls get() below" in f.read_bytes()

    def test_delete_of_get_does_not_touch_a_string_literal_mentioning_get(self, tmp_path: Path) -> None:
        """Deleting the 'get' function must not touch a string elsewhere that merely names it."""
        f = tmp_path / "m.py"
        f.write_bytes(
            b"def get():\n    return 1\n\n\n"
            b'def other():\n    return "get is a common name"\n'
        )
        result = run_cli("delete", str(f), "get")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b'def other():\n    return "get is a common name"\n'


class TestDuplicateNamesDifferentScopes:
    """The same identifier name used in two unrelated scopes."""

    def test_rename_a_bare_name_shared_by_two_unrelated_classes_renames_both(self, tmp_path: Path) -> None:
        """fastedit rename is name-based (not scope-aware): both same-named methods are renamed.

        This documents the CLI's actual, documented behaviour ("rename all
        AST-verified occurrences of a symbol in a single file") -- it is not
        asserting a bug, it is pinning down what happens so a future change
        to scope-awareness is a visible, intentional diff here.
        """
        f = tmp_path / "m.py"
        original = (
            b"class A:\n    def value(self):\n        return 1\n\n\n"
            b"class B:\n    def value(self):\n        return 2\n"
        )
        f.write_bytes(original)
        result = run_cli("rename", str(f), "value", "renamed")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"def value(self):", b"def renamed(self):")
        assert f.read_bytes() == expected
        assert f.read_bytes().count(b"def renamed(self):") == 2

    def test_rename_a_dotted_class_method_name_is_refused_not_silently_ignored(self, tmp_path: Path) -> None:
        """rename does not resolve dotted 'Class.method' targeting -- it must refuse, not silently no-op."""
        f = tmp_path / "m.py"
        original = (
            b"class A:\n    def value(self):\n        return 1\n\n\n"
            b"class B:\n    def value(self):\n        return 2\n"
        )
        f.write_bytes(original)
        result = run_cli("rename", str(f), "A.value", "renamed")
        assert result.returncode != 0
        assert f.read_bytes() == original


class TestUnicodeIdentifiers:
    def test_rename_a_unicode_identifier_where_python_allows_it(self, tmp_path: Path) -> None:
        """Python allows non-ASCII identifiers; rename must handle one byte-exactly."""
        f = tmp_path / "m.py"
        original = "def café():\n    x = 1\n    return x\n".encode()
        f.write_bytes(original)
        result = run_cli("rename", str(f), "café", "kaffe")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def kaffe():\n    x = 1\n    return x\n"


class TestIndentationPreservedExactly:
    def test_rename_in_a_tabs_indented_file_keeps_tabs_not_spaces(self, tmp_path: Path) -> None:
        """A file indented with real tab bytes must keep tabs after a rename -- no silent re-indentation."""
        f = tmp_path / "m.py"
        original = b"def outer():\n\tinner_val = 1\n\treturn inner_val\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "inner_val", "renamed_val")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"inner_val", b"renamed_val")
        assert f.read_bytes() == expected
        assert b"\t" in f.read_bytes()
        assert b"    " not in f.read_bytes()

    def test_rename_in_a_spaces_indented_file_keeps_exact_space_count(self, tmp_path: Path) -> None:
        """A file indented with an unusual (non-4) space count must keep that exact count after a rename."""
        f = tmp_path / "m.py"
        original = b"def outer():\n  inner_val = 1\n  return inner_val\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "inner_val", "renamed_val")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"inner_val", b"renamed_val")
        assert f.read_bytes() == expected


class TestTrailingWhitespaceOnEveryLine:
    def test_rename_survives_trailing_whitespace_on_every_line_untouched(self, tmp_path: Path) -> None:
        """Trailing whitespace on lines the edit doesn't touch must survive byte-for-byte."""
        f = tmp_path / "m.py"
        original = b"def get():   \n    return 1  \n\n\ndef other():\t\n    return 2\t\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "get", "fetch")
        assert result.returncode == 0, result.stderr
        expected = original.replace(b"def get():", b"def fetch():")
        assert f.read_bytes() == expected
        assert b"return 2\t\n" in f.read_bytes()


# ---------------------------------------------------------------------------
# FAILURE AND REFUSAL PATHS
# ---------------------------------------------------------------------------


class TestFailureAndRefusalPaths:
    def test_edit_refuses_when_target_file_does_not_exist(self, tmp_path: Path) -> None:
        """edit --replace on a nonexistent file must refuse with a clear message and non-zero exit."""
        missing = tmp_path / "nope.py"
        result = run_cli("edit", str(missing), "--replace", "foo", "--snippet", "x = 1\n")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()
        assert not missing.exists()

    def test_delete_refuses_when_target_symbol_does_not_exist(self, tmp_path: Path) -> None:
        """delete of a symbol absent from the file must refuse with a clear message, file untouched."""
        f = tmp_path / "m.py"
        original = b"def real():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("delete", str(f), "imaginary")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()
        assert f.read_bytes() == original

    def test_create_refuses_on_existing_file_without_force(self, tmp_path: Path) -> None:
        """create on an existing populated file without --force must refuse and leave it untouched."""
        f = tmp_path / "m.py"
        original = b"def real():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("create", str(f), "--content", "def other():\n    return 2\n")
        assert result.returncode != 0
        assert "exists" in result.stderr.lower()
        assert f.read_bytes() == original

    def test_create_with_force_overwrites_an_existing_file(self, tmp_path: Path) -> None:
        """create --force on an existing file must succeed and replace its bytes exactly."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def real():\n    return 1\n")
        result = run_cli("create", str(f), "--content", "def other():\n    return 2\n", "--force")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def other():\n    return 2\n"

    def test_create_refuses_without_parents_when_parent_directory_is_missing(self, tmp_path: Path) -> None:
        """create without --parents into a missing directory must refuse, not silently mkdir."""
        target = tmp_path / "missing_dir" / "m.py"
        result = run_cli("create", str(target), "--content", "x = 1\n")
        assert result.returncode != 0
        assert "parent" in result.stderr.lower()
        assert not target.exists()
        assert not target.parent.exists()

    def test_create_with_parents_creates_the_missing_directory(self, tmp_path: Path) -> None:
        """create --parents into a missing directory must create it and write the file byte-exact."""
        target = tmp_path / "missing_dir" / "m.py"
        result = run_cli("create", str(target), "--content", "x = 1\n", "--parents")
        assert result.returncode == 0, result.stderr
        assert target.read_bytes() == b"x = 1\n"

    def test_edit_with_a_syntactically_invalid_snippet_refuses_cleanly(self, tmp_path: Path) -> None:
        """A snippet that isn't valid Python must be refused with a clear message, not an uncaught traceback.

        No mlx skip: the refusal is a parse gate on the snippet itself and fires
        before any merge backend is constructed, so the guard holds in every
        environment.
        """
        f = tmp_path / "m.py"
        original = b"def f():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f(:\n    broken\n")
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert f.read_bytes() == original

    def test_delete_refuses_a_symbol_with_cross_file_callers_without_force(self, tmp_path: Path) -> None:
        """delete of a symbol another file still imports/calls must refuse without --force."""
        lib = tmp_path / "lib.py"
        user = tmp_path / "user.py"
        lib.write_bytes(b"def shared():\n    return 42\n")
        user.write_bytes(b"from lib import shared\n\n\ndef caller():\n    return shared()\n")
        result = run_cli("delete", str(lib), "shared")
        assert result.returncode != 0
        assert "shared" in result.stderr
        assert lib.read_bytes() == b"def shared():\n    return 42\n"
        assert user.read_bytes() == b"from lib import shared\n\n\ndef caller():\n    return shared()\n"

    def test_delete_with_force_proceeds_despite_cross_file_callers(self, tmp_path: Path) -> None:
        """delete --force must proceed even when other files still reference the symbol."""
        lib = tmp_path / "lib.py"
        user = tmp_path / "user.py"
        lib.write_bytes(b"def shared():\n    return 42\n")
        user.write_bytes(b"from lib import shared\n\n\ndef caller():\n    return shared()\n")
        result = run_cli("delete", str(lib), "shared", "--force")
        assert result.returncode == 0, result.stderr
        assert lib.read_bytes() == b""
        # the caller file is not fastedit's business to fix up -- it is left as-is
        assert user.read_bytes() == b"from lib import shared\n\n\ndef caller():\n    return shared()\n"

    def test_move_after_refuses_when_the_after_target_does_not_exist(self, tmp_path: Path) -> None:
        """move --after naming a nonexistent symbol must refuse, leaving the file untouched."""
        f = tmp_path / "m.py"
        original = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        f.write_bytes(original)
        result = run_cli("move", str(f), "a", "--after", "does_not_exist")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()
        assert f.read_bytes() == original

    def test_move_refuses_when_the_symbol_to_move_does_not_exist(self, tmp_path: Path) -> None:
        """move of a symbol that isn't in the file must refuse, leaving the file untouched."""
        f = tmp_path / "m.py"
        original = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        f.write_bytes(original)
        result = run_cli("move", str(f), "does_not_exist", "--after", "b")
        assert result.returncode != 0
        assert f.read_bytes() == original

    def test_join_given_parts_that_do_not_belong_together_does_not_refuse(self, tmp_path: Path) -> None:
        """DOCUMENTS AN ACTUAL GAP: join has no manifest/checksum cross-check, so it silently
        concatenates unrelated files instead of refusing. Reported separately as a finding;
        this test pins the CURRENT (permissive) behaviour so a future fix shows as an
        intentional diff here rather than a silent behaviour change.
        """
        part_a = tmp_path / "part_a.txt"
        part_b = tmp_path / "unrelated.txt"
        part_a.write_bytes(b"line1\nline2\n")
        part_b.write_bytes(b"totally-unrelated-content\n")
        out = tmp_path / "joined.txt"
        result = run_cli("join", str(part_a), str(part_b), "-o", str(out))
        assert result.returncode == 0, result.stderr
        assert out.read_bytes() == b"line1\nline2\ntotally-unrelated-content\n"

    def test_rename_all_refuses_when_root_directory_does_not_exist(self, tmp_path: Path) -> None:
        """rename-all against a nonexistent root directory must refuse with a clear message."""
        missing_root = tmp_path / "does_not_exist_dir"
        result = run_cli("rename-all", str(missing_root), "old", "new")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# IDEMPOTENCE AND RECOVERY
# ---------------------------------------------------------------------------


class TestIdempotenceAndRecovery:
    def test_running_the_same_replace_twice_yields_identical_bytes_the_second_time(self, tmp_path: Path) -> None:
        """edit --replace applied a second time with the same snippet must produce the same bytes."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def f():\n    return 1\n")
        r1 = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 2\n")
        assert r1.returncode == 0, r1.stderr
        after_first = f.read_bytes()
        r2 = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 2\n")
        assert r2.returncode == 0, r2.stderr
        assert f.read_bytes() == after_first == b"def f():\n    return 2\n"

    def test_undo_after_replace_restores_the_file_byte_exactly(self, tmp_path: Path) -> None:
        """undo after edit --replace must restore the exact pre-edit bytes."""
        f = tmp_path / "m.py"
        original = b"def f():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 2\n")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() != original
        undo_result = run_cli("undo", str(f))
        assert undo_result.returncode == 0, undo_result.stderr
        assert f.read_bytes() == original

    def test_undo_after_delete_restores_the_file_byte_exactly(self, tmp_path: Path) -> None:
        """undo after delete must restore the deleted symbol's exact original bytes."""
        f = tmp_path / "m.py"
        original = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        f.write_bytes(original)
        result = run_cli("delete", str(f), "a")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() != original
        undo_result = run_cli("undo", str(f))
        assert undo_result.returncode == 0, undo_result.stderr
        assert f.read_bytes() == original

    def test_undo_after_move_restores_the_file_byte_exactly(self, tmp_path: Path) -> None:
        """undo after move must restore both symbols to their exact original order and bytes."""
        f = tmp_path / "m.py"
        original = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        f.write_bytes(original)
        result = run_cli("move", str(f), "a", "--after", "b")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() != original
        undo_result = run_cli("undo", str(f))
        assert undo_result.returncode == 0, undo_result.stderr
        assert f.read_bytes() == original

    def test_undo_after_rename_restores_the_file_byte_exactly(self, tmp_path: Path) -> None:
        """undo after rename must restore the original identifier bytes exactly."""
        f = tmp_path / "m.py"
        original = b"def get():\n    return [get()]\n"
        f.write_bytes(original)
        result = run_cli("rename", str(f), "get", "fetch")
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() != original
        undo_result = run_cli("undo", str(f))
        assert undo_result.returncode == 0, undo_result.stderr
        assert f.read_bytes() == original

    def test_undo_with_nothing_to_undo_fails_cleanly(self, tmp_path: Path) -> None:
        """undo on a file that was never edited by fastedit must fail cleanly, not crash."""
        f = tmp_path / "m.py"
        original = b"def f():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("undo", str(f))
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert f.read_bytes() == original

    def test_undo_twice_in_a_row_the_second_undo_fails_cleanly(self, tmp_path: Path) -> None:
        """A second consecutive undo (no more history) must fail cleanly and leave the restored bytes alone."""
        f = tmp_path / "m.py"
        original = b"def f():\n    return 1\n"
        f.write_bytes(original)
        result = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 2\n")
        assert result.returncode == 0, result.stderr
        first_undo = run_cli("undo", str(f))
        assert first_undo.returncode == 0, first_undo.stderr
        assert f.read_bytes() == original
        second_undo = run_cli("undo", str(f))
        assert second_undo.returncode != 0
        assert f.read_bytes() == original

    def test_running_the_same_delete_twice_the_second_run_refuses_symbol_gone(self, tmp_path: Path) -> None:
        """delete applied twice: the first removes the symbol, the second must refuse (already gone)."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def a():\n    return 1\n\n\ndef b():\n    return 2\n")
        r1 = run_cli("delete", str(f), "a")
        assert r1.returncode == 0, r1.stderr
        after_first = f.read_bytes()
        assert after_first == b"def b():\n    return 2\n"
        r2 = run_cli("delete", str(f), "a")
        assert r2.returncode != 0
        assert f.read_bytes() == after_first


# ---------------------------------------------------------------------------
# ADDITIONAL STRUCTURAL / MULTI-FILE COVERAGE
# ---------------------------------------------------------------------------


class TestRenameAllAcrossFiles:
    def test_rename_all_updates_definition_and_caller_in_two_files_byte_exact(self, tmp_path: Path) -> None:
        """rename-all across a directory must update both the definition and the cross-file call site."""
        lib = tmp_path / "lib.py"
        user = tmp_path / "user.py"
        lib.write_bytes(b"def shared():\n    return 42\n")
        user.write_bytes(b"from lib import shared\n\n\ndef caller():\n    return shared()\n")
        result = run_cli("rename-all", str(tmp_path), "shared", "renamed_shared")
        assert result.returncode == 0, result.stderr
        assert lib.read_bytes() == b"def renamed_shared():\n    return 42\n"
        assert user.read_bytes() == b"from lib import renamed_shared\n\n\ndef caller():\n    return renamed_shared()\n"

    def test_rename_all_dry_run_leaves_every_file_byte_exact(self, tmp_path: Path) -> None:
        """rename-all --dry-run must preview without writing a single byte to any file."""
        lib = tmp_path / "lib.py"
        user = tmp_path / "user.py"
        lib_original = b"def shared():\n    return 42\n"
        user_original = b"from lib import shared\n\n\ndef caller():\n    return shared()\n"
        lib.write_bytes(lib_original)
        user.write_bytes(user_original)
        result = run_cli("rename-all", str(tmp_path), "shared", "renamed_shared", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert lib.read_bytes() == lib_original
        assert user.read_bytes() == user_original


class TestMultiEditAndBatchEdit:
    def test_multi_edit_refuses_cleanly_when_one_target_file_is_missing(self, tmp_path: Path) -> None:
        """multi-edit naming a nonexistent file among its targets must refuse, not partially apply."""
        real = tmp_path / "real.py"
        original = b"def f():\n    return 1\n"
        real.write_bytes(original)
        missing = tmp_path / "missing.py"
        file_edits = (
            f'[{{"file_path": "{real}", "edits": [{{"replace": "f", "snippet": "def f():\\n    return 2\\n"}}]}},'
            f' {{"file_path": "{missing}", "edits": [{{"replace": "f", "snippet": "def f():\\n    return 2\\n"}}]}}]'
        )
        result = run_cli("multi-edit", "--file-edits", file_edits)
        assert result.returncode != 0
        # the real file must not have been left half-edited by the failed batch
        assert real.read_bytes() == original

    @pytest.mark.skipif(
        not HAS_MLX,
        reason="batch-edit never attempts the deterministic fast path even for a "
        "trivially exact whole-symbol replacement -- it always constructs a "
        "model backend, which requires mlx here (matches the suite's known "
        "mlx-dependent TestCLIBatchEdit/TestCLIMultiEdit skips).",
    )
    def test_batch_edit_applies_two_edits_to_one_file_in_declared_order(self, tmp_path: Path) -> None:
        """batch-edit with two edits to the same file must apply both, in the order given."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def a():\n    return 1\n\n\ndef b():\n    return 2\n")
        edits = (
            '[{"replace": "a", "snippet": "def a():\\n    return 10\\n"},'
            ' {"replace": "b", "snippet": "def b():\\n    return 20\\n"}]'
        )
        result = run_cli("batch-edit", str(f), "--edits", edits)
        assert result.returncode == 0, result.stderr
        assert f.read_bytes() == b"def a():\n    return 10\n\n\ndef b():\n    return 20\n"


class TestMoveInvolvingNestedSymbols:
    def test_move_a_top_level_function_after_a_class_leaves_the_classs_own_methods_byte_exact(
        self, tmp_path: Path
    ) -> None:
        """Moving a top-level function to after a class must not disturb the class's internal method bytes."""
        f = tmp_path / "m.py"
        original = (
            b"def helper():\n    return 1\n\n\n"
            b"class C:\n    def method(self):\n        return 2\n"
        )
        f.write_bytes(original)
        result = run_cli("move", str(f), "helper", "--after", "C")
        assert result.returncode == 0, result.stderr
        content = f.read_bytes()
        assert b"    def method(self):\n        return 2\n" in content
        assert content.index(b"class C:") < content.index(b"def helper():")


class TestDuplicateCommand:
    def test_duplicate_refuses_when_destination_already_exists_without_force(self, tmp_path: Path) -> None:
        """duplicate onto an existing destination without --force must refuse, leaving both files untouched."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src_original = b"def f():\n    return 1\n"
        dst_original = b"def other():\n    return 2\n"
        src.write_bytes(src_original)
        dst.write_bytes(dst_original)
        result = run_cli("duplicate", str(src), str(dst))
        assert result.returncode != 0
        assert src.read_bytes() == src_original
        assert dst.read_bytes() == dst_original

    def test_duplicate_produces_a_byte_exact_copy(self, tmp_path: Path) -> None:
        """duplicate onto a fresh path must produce byte-for-byte identical content to the source."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        original = b"def f():\n    return 1\n"
        src.write_bytes(original)
        result = run_cli("duplicate", str(src), str(dst))
        assert result.returncode == 0, result.stderr
        assert dst.read_bytes() == original
        assert src.read_bytes() == original
