"""Tests for AST-verified single-file rename (do_rename_ast).

Verifies:
- Matches inside strings, comments, and docstrings are NOT renamed.
- Matches inside code (definitions, calls, etc.) ARE renamed.
- The engine uses `tldr references --scope file`, so string/comment
  substrings are filtered at the AST layer rather than via regex skip
  zones.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fastedit.inference.rename import do_rename_ast


# ---------------------------------------------------------------------------
# Behavioral invariant (VAL-M1-002): strings/comments/docstrings preserved.
# ---------------------------------------------------------------------------


class TestDoRenameAstSkipsStringsAndComments:
    """Locks VAL-M1-002: substrings inside string/comment/docstring nodes
    must not be touched by the new AST-verified single-file rename."""

    def test_fast_rename_skips_strings_and_comments_python(self, tmp_path: Path):
        """Python: rename the function def + call, leave comment, string, and
        docstring substrings untouched."""
        path = tmp_path / "mod.py"
        path.write_text(textwrap.dedent("""\
        def old_name():
            \"\"\"docstring: old_name is mentioned here.\"\"\"
            return 1

        # old_name in comment
        msg = "old_name in string"
        x = old_name()
        """))

        new_content, count, skipped = do_rename_ast(path, "old_name", "new_name")

        # Code sites renamed:
        assert "def new_name():" in new_content
        assert "x = new_name()" in new_content

        # Non-code sites preserved verbatim:
        assert "docstring: old_name is mentioned here." in new_content
        assert "# old_name in comment" in new_content
        assert '"old_name in string"' in new_content

        # Exactly the def + the call were renamed.
        assert count == 2
        # There were 3 string/comment/docstring substring hits.
        assert skipped >= 3

    def test_fast_rename_skips_strings_and_comments_typescript(self, tmp_path: Path):
        """TypeScript: rename the function def + call, leave comment, string,
        and JSDoc substrings untouched."""
        path = tmp_path / "mod.ts"
        path.write_text(textwrap.dedent("""\
        /**
         * JSDoc mentions old_name in prose.
         */
        function old_name(): number {
          return 1;
        }

        // old_name in comment
        const msg: string = "old_name in string";
        const x = old_name();
        """))

        new_content, count, skipped = do_rename_ast(path, "old_name", "new_name")

        # Code sites renamed:
        assert "function new_name()" in new_content
        assert "const x = new_name();" in new_content

        # Non-code sites preserved verbatim:
        assert "JSDoc mentions old_name in prose." in new_content
        assert "// old_name in comment" in new_content
        assert '"old_name in string"' in new_content

        assert count == 2
        assert skipped >= 3


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestDoRenameAstBasics:
    def test_word_boundary_preserves_longer_names(self, tmp_path: Path):
        """Renaming 'get' must not touch 'get_all' or 'getter'."""
        path = tmp_path / "mod.py"
        path.write_text(textwrap.dedent("""\
        def get():
            pass

        def get_all():
            pass

        getter = get()
        """))

        new_content, count, _ = do_rename_ast(path, "get", "fetch")
        assert "def fetch():" in new_content
        assert "def get_all():" in new_content
        assert "getter = fetch()" in new_content
        assert count == 2

    def test_no_matches_returns_zero_count(self, tmp_path: Path):
        path = tmp_path / "mod.py"
        path.write_text("def other():\n    return 0\n")
        new_content, count, skipped = do_rename_ast(path, "missing", "replaced")
        assert new_content == path.read_text()
        assert count == 0

    def test_same_name_is_noop(self, tmp_path: Path):
        path = tmp_path / "mod.py"
        original = "def foo():\n    return foo()\n"
        path.write_text(original)
        new_content, count, _ = do_rename_ast(path, "foo", "foo")
        assert new_content == original
        assert count == 0

    def test_unicode_content_preserved(self, tmp_path: Path):
        """Unicode in strings/comments must round-trip intact."""
        path = tmp_path / "mod.py"
        path.write_text(
            "# Calcul du coût\n"
            "def get():\n"
            "    return 'élève'\n"
            "\n"
            "x = get()\n"
        )
        new_content, count, _ = do_rename_ast(path, "get", "fetch")
        assert "coût" in new_content
        assert "élève" in new_content
        assert "def fetch():" in new_content
        assert "x = fetch()" in new_content
        assert count == 2
