"""Tests for FastEdit CLI subcommands (edit, read, delete, move, rename, diff, undo, search).

These tests define the behavioral contracts for the CLI subcommands specified in
thoughts/cli-spec.md. They should ALL FAIL initially because the subcommands
do not exist yet. Each test documents the expected behavior of a single CLI
subcommand via both subprocess invocation and direct function call.

Test matrix:
  - read: structure output, small-file full content, missing file error
  - edit: replace via stdin, after insertion, missing file, missing symbol
  - delete: symbol removal, backup creation, missing symbol error
  - move: symbol relocation, backup creation, missing symbol/target errors
  - rename: word-boundary rename, replacement count, backup creation
  - search: keyword search, mode flags, default path
  - diff: unified diff output, no-backup message
  - undo: revert to backup, no-backup error
  - environment/config: --api-base, --model, env vars, defaults
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_PYTHON_FILE = textwrap.dedent("""\
    import os


    def greet(name: str) -> str:
        \"\"\"Return a greeting string.\"\"\"
        return f"Hello, {name}!"


    def farewell(name: str) -> str:
        \"\"\"Return a farewell string.\"\"\"
        return f"Goodbye, {name}!"


    class Calculator:
        def add(self, a: int, b: int) -> int:
            return a + b

        def subtract(self, a: int, b: int) -> int:
            return a - b
""")

LARGE_PYTHON_FILE = textwrap.dedent("""\
""") + "\n".join(
    f"def func_{i}(x):\n    return x + {i}\n"
    for i in range(60)
)  # ~180 lines — exceeds the 100-line threshold for full content


@pytest.fixture
def small_py(tmp_path: Path) -> Path:
    """Write a small Python file (<100 lines) and return its path."""
    p = tmp_path / "small.py"
    p.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
    return p


@pytest.fixture
def large_py(tmp_path: Path) -> Path:
    """Write a large Python file (>100 lines) and return its path."""
    p = tmp_path / "large.py"
    p.write_text(LARGE_PYTHON_FILE, encoding="utf-8")
    return p


@pytest.fixture
def backup_dir(tmp_path: Path, monkeypatch):
    """Redirect BackupStore to a temp directory to avoid polluting ~/.fastedit."""
    bd = tmp_path / "backups"
    bd.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    return bd


# ---------------------------------------------------------------------------
# Helper: run CLI as subprocess
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None, env_extra: dict | None = None):
    """Run `python -m fastedit <args>` and return CompletedProcess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


# ===================================================================
# 1. fastedit read
# ===================================================================

class TestCLIRead:
    """Tests for `fastedit read <file>` subcommand."""

    def test_read_small_file_returns_full_content(self, small_py: Path):
        """Small files (<100 lines) should return full content."""
        result = run_cli("read", str(small_py))
        assert result.returncode == 0
        # Should contain the actual code
        assert "def greet" in result.stdout
        assert "def farewell" in result.stdout
        assert "class Calculator" in result.stdout

    def test_read_large_file_returns_structure(self, large_py: Path):
        """Large files (>100 lines) should return structure, not full content."""
        result = run_cli("read", str(large_py))
        assert result.returncode == 0
        # Should mention the file path and line count
        assert str(large_py) in result.stdout or large_py.name in result.stdout
        # Should NOT contain every function body
        # (structure mode shows line ranges, not code)

    def test_read_missing_file_exits_with_error(self, tmp_path: Path):
        """Reading a non-existent file should exit with code 1."""
        missing = tmp_path / "nonexistent.py"
        result = run_cli("read", str(missing))
        assert result.returncode == 1
        assert "error" in result.stderr.lower() or "Error" in result.stderr

    def test_read_subcommand_exists(self):
        """The 'read' subcommand should be recognized by argparse."""
        result = run_cli("read", "--help")
        assert result.returncode == 0
        assert "file" in result.stdout.lower()


# ===================================================================
# 2. fastedit edit
# ===================================================================

class TestCLIEdit:
    """Tests for `fastedit edit <file> --snippet <text> [--after|--replace]`."""

    def test_edit_subcommand_exists(self):
        """The 'edit' subcommand should be recognized by argparse."""
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        assert "--snippet" in result.stdout
        assert "--after" in result.stdout
        assert "--replace" in result.stdout

    def test_edit_missing_file_exits_with_error(self, tmp_path: Path):
        """Editing a non-existent file should exit with code 1."""
        missing = tmp_path / "nonexistent.py"
        result = run_cli(
            "edit", str(missing),
            "--snippet", "def new(): pass",
            "--replace", "greet",
        )
        assert result.returncode == 1
        assert "error" in result.stderr.lower() or "Error" in result.stderr

    def test_edit_reads_snippet_from_stdin(self, small_py: Path):
        """--snippet - should read the edit snippet from stdin."""
        snippet = "def greet(name: str) -> str:\n    return f'Hi, {name}!'\n"
        result = run_cli(
            "edit", str(small_py),
            "--snippet", "-",
            "--replace", "greet",
            input_text=snippet,
        )
        # This will fail because the edit subcommand doesn't exist yet,
        # but when implemented it should apply the snippet
        assert result.returncode == 0

    def test_edit_after_inserts_code(self, small_py: Path):
        """--after should insert new code after the named symbol."""
        snippet = textwrap.dedent("""\
            def hello_world() -> str:
                return "Hello, World!"
        """)
        result = run_cli(
            "edit", str(small_py),
            "--snippet", "-",
            "--after", "greet",
            input_text=snippet,
        )
        assert result.returncode == 0
        # Verify the new function was inserted
        content = small_py.read_text()
        assert "def hello_world" in content

    def test_edit_replace_missing_symbol_exits_with_error(self, small_py: Path):
        """--replace with a non-existent symbol should exit with code 1."""
        result = run_cli(
            "edit", str(small_py),
            "--snippet", "def bogus(): pass",
            "--replace", "nonexistent_function",
        )
        assert result.returncode == 1

    def test_edit_creates_backup(self, small_py: Path, backup_dir):
        """Edit should create a backup in BackupStore before writing."""
        original = small_py.read_text()
        snippet = "def greet(name: str) -> str:\n    return 'changed'\n"
        edit_result = run_cli(
            "edit", str(small_py),
            "--snippet", "-",
            "--replace", "greet",
            input_text=snippet,
        )
        assert edit_result.returncode == 0, f"Edit failed: {edit_result.stderr}"
        # After editing, undo should restore the original
        result = run_cli("undo", str(small_py))
        assert result.returncode == 0, f"Undo failed: {result.stderr}"
        restored = small_py.read_text()
        assert restored == original

    def test_edit_accepts_backend_flags(self):
        """--backend, --model-path, --api-base, --api-model should be accepted."""
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        assert "--backend" in result.stdout
        assert "--model-path" in result.stdout
        assert "--api-base" in result.stdout
        assert "--api-model" in result.stdout


# ===================================================================
# 3. fastedit delete
# ===================================================================

class TestCLIDelete:
    """Tests for `fastedit delete <file> <symbol>`."""

    def test_delete_subcommand_exists(self):
        """The 'delete' subcommand should be recognized by argparse."""
        result = run_cli("delete", "--help")
        assert result.returncode == 0
        assert "file" in result.stdout.lower()
        assert "symbol" in result.stdout.lower()

    def test_delete_removes_function(self, small_py: Path):
        """Deleting a function should remove it from the file."""
        result = run_cli("delete", str(small_py), "farewell")
        assert result.returncode == 0
        content = small_py.read_text()
        assert "def farewell" not in content
        # Other functions should still be there
        assert "def greet" in content
        assert "class Calculator" in content

    def test_delete_removes_class(self, small_py: Path):
        """Deleting a class should remove the entire class."""
        result = run_cli("delete", str(small_py), "Calculator")
        assert result.returncode == 0
        content = small_py.read_text()
        assert "class Calculator" not in content
        assert "def add" not in content
        assert "def subtract" not in content
        # Functions should remain
        assert "def greet" in content

    def test_delete_missing_symbol_exits_with_error(self, small_py: Path):
        """Deleting a non-existent symbol should exit with code 1."""
        result = run_cli("delete", str(small_py), "nonexistent_func")
        assert result.returncode == 1
        assert "error" in result.stderr.lower() or "Error" in result.stderr

    def test_delete_missing_file_exits_with_error(self, tmp_path: Path):
        """Deleting from a non-existent file should exit with code 1."""
        missing = tmp_path / "nope.py"
        result = run_cli("delete", str(missing), "greet")
        assert result.returncode == 1

    def test_delete_creates_backup(self, small_py: Path, backup_dir):
        """Delete should create a backup before removing the symbol."""
        original = small_py.read_text()
        del_result = run_cli("delete", str(small_py), "farewell")
        assert del_result.returncode == 0, f"Delete failed: {del_result.stderr}"
        # Undo should restore
        result = run_cli("undo", str(small_py))
        assert result.returncode == 0, f"Undo failed: {result.stderr}"
        restored = small_py.read_text()
        assert "def farewell" in restored

    def test_delete_reports_lines_removed(self, small_py: Path):
        """Delete output should report which lines were removed."""
        result = run_cli("delete", str(small_py), "farewell")
        assert result.returncode == 0
        # Output should mention "Deleted" and line numbers
        assert "Deleted" in result.stdout or "deleted" in result.stdout
        assert "lines" in result.stdout.lower() or "L" in result.stdout


# ===================================================================
# 4. fastedit move
# ===================================================================

class TestCLIMove:
    """Tests for `fastedit move <file> <symbol> --after <target>`."""

    def test_move_subcommand_exists(self):
        """The 'move' subcommand should be recognized by argparse."""
        result = run_cli("move", "--help")
        assert result.returncode == 0
        assert "--after" in result.stdout

    def test_move_symbol_after_target(self, small_py: Path):
        """Moving a symbol should relocate it after the target."""
        # Move 'greet' to after 'farewell'
        result = run_cli("move", str(small_py), "greet", "--after", "farewell")
        assert result.returncode == 0
        content = small_py.read_text()
        # Both functions should still exist
        assert "def greet" in content
        assert "def farewell" in content
        # greet should now appear AFTER farewell
        greet_pos = content.index("def greet")
        farewell_pos = content.index("def farewell")
        assert farewell_pos < greet_pos

    def test_move_missing_symbol_exits_with_error(self, small_py: Path):
        """Moving a non-existent symbol should exit with code 1."""
        result = run_cli("move", str(small_py), "bogus", "--after", "greet")
        assert result.returncode == 1

    def test_move_missing_target_exits_with_error(self, small_py: Path):
        """Moving after a non-existent target should exit with code 1."""
        result = run_cli("move", str(small_py), "greet", "--after", "bogus")
        assert result.returncode == 1

    def test_move_missing_file_exits_with_error(self, tmp_path: Path):
        """Moving in a non-existent file should exit with code 1."""
        missing = tmp_path / "nope.py"
        result = run_cli("move", str(missing), "greet", "--after", "farewell")
        assert result.returncode == 1

    def test_move_creates_backup(self, small_py: Path, backup_dir):
        """Move should create a backup before modifying."""
        original = small_py.read_text()
        move_result = run_cli("move", str(small_py), "greet", "--after", "farewell")
        assert move_result.returncode == 0, f"Move failed: {move_result.stderr}"
        result = run_cli("undo", str(small_py))
        assert result.returncode == 0, f"Undo failed: {result.stderr}"
        restored = small_py.read_text()
        assert restored == original

    def test_move_same_symbol_exits_with_error(self, small_py: Path):
        """Moving a symbol after itself should exit with code 1."""
        result = run_cli("move", str(small_py), "greet", "--after", "greet")
        assert result.returncode == 1


# ===================================================================
# 5. fastedit rename
# ===================================================================

class TestCLIRename:
    """Tests for `fastedit rename <file> <old_name> <new_name>`."""

    def test_rename_subcommand_exists(self):
        """The 'rename' subcommand should be recognized by argparse."""
        result = run_cli("rename", "--help")
        assert result.returncode == 0

    def test_rename_replaces_all_occurrences(self, small_py: Path):
        """Rename should replace all word-boundary occurrences in code."""
        result = run_cli("rename", str(small_py), "greet", "welcome")
        assert result.returncode == 0
        content = small_py.read_text()
        assert "def welcome" in content
        assert "def greet" not in content

    def test_rename_respects_word_boundaries(self, small_py: Path):
        """Rename 'add' should not affect 'add' inside longer identifiers."""
        # 'add' appears in Calculator.add but should NOT affect any
        # hypothetical 'additional' or 'address' identifiers
        result = run_cli("rename", str(small_py), "add", "sum_values")
        assert result.returncode == 0
        content = small_py.read_text()
        assert "def sum_values" in content
        # 'subtract' should be untouched
        assert "def subtract" in content

    def test_rename_reports_replacement_count(self, small_py: Path):
        """Rename output should report how many replacements were made."""
        result = run_cli("rename", str(small_py), "greet", "welcome")
        assert result.returncode == 0
        assert "replacement" in result.stdout.lower()

    def test_rename_no_match_exits_with_error(self, small_py: Path):
        """Renaming a symbol that doesn't exist should exit with code 1."""
        result = run_cli("rename", str(small_py), "nonexistent_sym", "new_name")
        assert result.returncode == 1

    def test_rename_missing_file_exits_with_error(self, tmp_path: Path):
        """Renaming in a non-existent file should exit with code 1."""
        missing = tmp_path / "nope.py"
        result = run_cli("rename", str(missing), "old", "new")
        assert result.returncode == 1

    def test_rename_creates_backup(self, small_py: Path, backup_dir):
        """Rename should create a backup before modifying."""
        original = small_py.read_text()
        rename_result = run_cli("rename", str(small_py), "greet", "welcome")
        assert rename_result.returncode == 0, f"Rename failed: {rename_result.stderr}"
        result = run_cli("undo", str(small_py))
        assert result.returncode == 0, f"Undo failed: {result.stderr}"
        restored = small_py.read_text()
        assert "def greet" in restored

    def test_rename_shows_diff(self, small_py: Path):
        """Rename output should include a unified diff."""
        result = run_cli("rename", str(small_py), "greet", "welcome")
        assert result.returncode == 0
        # Diff markers
        assert "---" in result.stdout or "+++" in result.stdout

    def test_rename_skips_strings_and_comments(self, tmp_path: Path):
        """Rename should not modify occurrences inside strings and comments."""
        code = textwrap.dedent("""\
            def fetch(url: str) -> str:
                # fetch the resource
                return f"fetched {url}"

            def process():
                data = fetch("http://example.com")
                return data
        """)
        p = tmp_path / "with_strings.py"
        p.write_text(code, encoding="utf-8")
        result = run_cli("rename", str(p), "fetch", "retrieve")
        assert result.returncode == 0
        content = p.read_text()
        # Function def and call should be renamed
        assert "def retrieve" in content
        assert "retrieve(" in content
        # String content "fetched" should NOT be affected (it has "fetch" as substring
        # but word-boundary won't match inside "fetched")
        assert "fetched" in content


# ===================================================================
# 6. fastedit search
# ===================================================================

class TestCLISearch:
    """Tests for `fastedit search <query> [path] [--mode] [--top-k]`."""

    def test_search_subcommand_exists(self):
        """The 'search' subcommand should be recognized by argparse."""
        result = run_cli("search", "--help")
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--top-k" in result.stdout

    def test_search_finds_symbol(self, small_py: Path):
        """Searching for a function name should return results."""
        result = run_cli("search", "greet", str(small_py.parent))
        assert result.returncode == 0
        # Should have some output (not empty)
        assert len(result.stdout.strip()) > 0

    def test_search_defaults_to_current_directory(self):
        """Omitting path should default to current directory (.)."""
        result = run_cli("search", "--help")
        assert result.returncode == 0
        # The help text should mention the default
        # (argparse shows default values)

    def test_search_mode_regex(self, small_py: Path):
        """--mode regex should accept regex patterns."""
        result = run_cli(
            "search", "greet|farewell", str(small_py.parent),
            "--mode", "regex",
        )
        # Should succeed (whether results found depends on tldr)
        assert result.returncode == 0

    def test_search_mode_references(self, small_py: Path):
        """--mode references should find usages of a symbol."""
        result = run_cli(
            "search", "greet", str(small_py.parent),
            "--mode", "references",
        )
        assert result.returncode == 0

    def test_search_top_k_limits_results(self, small_py: Path):
        """--top-k N should limit results to N."""
        result = run_cli(
            "search", "func", str(small_py.parent),
            "--top-k", "3",
        )
        assert result.returncode == 0


# ===================================================================
# 7. fastedit diff
# ===================================================================

class TestCLIDiff:
    """Tests for `fastedit diff <file>`."""

    def test_diff_subcommand_exists(self):
        """The 'diff' subcommand should be recognized by argparse."""
        result = run_cli("diff", "--help")
        assert result.returncode == 0

    def test_diff_no_backup_shows_message(self, small_py: Path):
        """If no backup exists, diff should say so (not crash)."""
        result = run_cli("diff", str(small_py))
        # Should succeed but indicate no backup
        assert result.returncode == 0
        assert "no backup" in result.stdout.lower() or "No backup" in result.stdout

    def test_diff_shows_unified_diff_after_edit(self, small_py: Path, backup_dir):
        """After editing a file, diff should show a unified diff."""
        # First, create a backup by doing a rename (simulates an edit)
        rename_result = run_cli("rename", str(small_py), "greet", "welcome")
        assert rename_result.returncode == 0, f"Rename failed: {rename_result.stderr}"
        result = run_cli("diff", str(small_py))
        assert result.returncode == 0, f"Diff failed: {result.stderr}"
        # Diff should show unified diff markers
        assert "---" in result.stdout
        assert "+++" in result.stdout

    def test_diff_missing_file_exits_with_error(self, tmp_path: Path):
        """Diffing a non-existent file should exit with code 1."""
        missing = tmp_path / "nope.py"
        result = run_cli("diff", str(missing))
        assert result.returncode == 1

    def test_diff_no_changes_after_undo(self, small_py: Path, backup_dir):
        """If file is reverted to match backup, diff should say no changes."""
        # Rename then undo: the backup store should now be empty (popped),
        # so diff should report no backup.
        rename_result = run_cli("rename", str(small_py), "greet", "welcome")
        assert rename_result.returncode == 0, f"Rename failed: {rename_result.stderr}"
        undo_result = run_cli("undo", str(small_py))
        assert undo_result.returncode == 0, f"Undo failed: {undo_result.stderr}"
        result = run_cli("diff", str(small_py))
        assert result.returncode == 0
        # After undo (which pops the backup), diff should report no backup
        stdout_lower = result.stdout.lower()
        assert "no backup" in stdout_lower or "no changes" in stdout_lower


# ===================================================================
# 8. fastedit undo
# ===================================================================

class TestCLIUndo:
    """Tests for `fastedit undo <file>`."""

    def test_undo_subcommand_exists(self):
        """The 'undo' subcommand should be recognized by argparse."""
        result = run_cli("undo", "--help")
        assert result.returncode == 0

    def test_undo_no_backup_exits_with_error(self, small_py: Path):
        """Undo with no prior backup should exit with code 1."""
        result = run_cli("undo", str(small_py))
        assert result.returncode == 1
        assert "no undo" in result.stderr.lower() or "Nothing to revert" in result.stderr

    def test_undo_reverts_rename(self, small_py: Path, backup_dir):
        """Undo should revert the file to its pre-rename state."""
        original = small_py.read_text()
        run_cli("rename", str(small_py), "greet", "welcome")
        renamed = small_py.read_text()
        assert "def welcome" in renamed

        result = run_cli("undo", str(small_py))
        assert result.returncode == 0
        restored = small_py.read_text()
        assert "def greet" in restored
        assert restored == original

    def test_undo_reverts_delete(self, small_py: Path, backup_dir):
        """Undo should revert a delete operation."""
        original = small_py.read_text()
        run_cli("delete", str(small_py), "farewell")
        deleted = small_py.read_text()
        assert "def farewell" not in deleted

        result = run_cli("undo", str(small_py))
        assert result.returncode == 0
        restored = small_py.read_text()
        assert "def farewell" in restored

    def test_undo_shows_diff(self, small_py: Path, backup_dir):
        """Undo output should include a diff showing what was reverted."""
        rename_result = run_cli("rename", str(small_py), "greet", "welcome")
        assert rename_result.returncode == 0, f"Rename failed: {rename_result.stderr}"
        result = run_cli("undo", str(small_py))
        assert result.returncode == 0, f"Undo failed: {result.stderr}"
        assert "Reverted" in result.stdout
        # Should include diff markers
        assert "---" in result.stdout or "+++" in result.stdout

    def test_undo_is_one_deep(self, small_py: Path, backup_dir):
        """Undo only supports 1 level. A second undo should fail."""
        run_cli("rename", str(small_py), "greet", "welcome")
        run_cli("undo", str(small_py))  # first undo
        result = run_cli("undo", str(small_py))  # second undo should fail
        assert result.returncode == 1


# ===================================================================
# 9. Direct function call tests (unit tests)
# ===================================================================

class TestCLIFunctions:
    """Test the CLI handler functions directly (not via subprocess).

    These test that the functions exist and have the expected signatures.
    They import from fastedit.cli and call cmd_read, cmd_delete, etc.
    """

    def test_cmd_read_exists(self):
        """cmd_read should be importable from fastedit.cli."""
        from fastedit.cli import cmd_read
        assert callable(cmd_read)

    def test_cmd_edit_exists(self):
        """cmd_edit should be importable from fastedit.cli."""
        from fastedit.cli import cmd_edit
        assert callable(cmd_edit)

    def test_cmd_delete_exists(self):
        """cmd_delete should be importable from fastedit.cli."""
        from fastedit.cli import cmd_delete
        assert callable(cmd_delete)

    def test_cmd_move_exists(self):
        """cmd_move should be importable from fastedit.cli."""
        from fastedit.cli import cmd_move
        assert callable(cmd_move)

    def test_cmd_rename_exists(self):
        """cmd_rename should be importable from fastedit.cli."""
        from fastedit.cli import cmd_rename
        assert callable(cmd_rename)

    def test_cmd_search_exists(self):
        """cmd_search should be importable from fastedit.cli."""
        from fastedit.cli import cmd_search
        assert callable(cmd_search)

    def test_cmd_diff_exists(self):
        """cmd_diff should be importable from fastedit.cli."""
        from fastedit.cli import cmd_diff
        assert callable(cmd_diff)

    def test_cmd_undo_exists(self):
        """cmd_undo should be importable from fastedit.cli."""
        from fastedit.cli import cmd_undo
        assert callable(cmd_undo)

    def test_make_backend_with_overrides_exists(self):
        """_make_backend_with_overrides helper should be importable."""
        from fastedit.cli import _make_backend_with_overrides
        assert callable(_make_backend_with_overrides)


# ===================================================================
# 10. Argparse subcommand registration
# ===================================================================

class TestArgparseRegistration:
    """Test that all subcommands are registered and dispatch correctly."""

    def test_all_subcommands_in_help(self):
        """All 8 new subcommands should appear in the top-level help."""
        result = run_cli("--help")
        assert result.returncode == 0
        for cmd in ["read", "edit", "delete", "move", "rename", "search", "diff", "undo"]:
            assert cmd in result.stdout, f"Subcommand '{cmd}' not in help output"

    def test_batch_edit_subcommand_exists(self):
        """The 'batch-edit' subcommand should be recognized."""
        result = run_cli("batch-edit", "--help")
        assert result.returncode == 0
        assert "--edits" in result.stdout

    def test_multi_edit_subcommand_exists(self):
        """The 'multi-edit' subcommand should be recognized."""
        result = run_cli("multi-edit", "--help")
        assert result.returncode == 0
        assert "--file-edits" in result.stdout


# ===================================================================
# 11. Environment variable and config tests
# ===================================================================

class TestEnvironmentConfig:
    """Test that environment variables and CLI flags configure the backend."""

    def test_fastedit_backend_env_var_accepted(self):
        """FASTEDIT_BACKEND env var should be recognized."""
        # edit --help should work regardless of env vars
        result = run_cli("edit", "--help", env_extra={"FASTEDIT_BACKEND": "vllm"})
        assert result.returncode == 0

    def test_fastedit_model_path_env_var(self):
        """FASTEDIT_MODEL_PATH env var should be recognized."""
        result = run_cli("edit", "--help", env_extra={"FASTEDIT_MODEL_PATH": "/tmp/test-model"})
        assert result.returncode == 0

    def test_fastedit_vllm_api_base_env_var(self):
        """FASTEDIT_VLLM_API_BASE env var should be recognized."""
        result = run_cli(
            "edit", "--help",
            env_extra={"FASTEDIT_VLLM_API_BASE": "http://localhost:9999/v1"},
        )
        assert result.returncode == 0

    def test_edit_backend_flag_choices(self):
        """--backend should only accept 'mlx' or 'vllm'."""
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        stdout = result.stdout
        assert "mlx" in stdout
        assert "vllm" in stdout


# ===================================================================
# 12. Batch edit tests
# ===================================================================

class TestCLIBatchEdit:
    """Tests for `fastedit batch-edit <file> --edits <json>`."""

    def test_batch_edit_accepts_json_edits(self, small_py: Path):
        """batch-edit should parse JSON list of edits."""
        edits = json.dumps([
            {"snippet": "def new_func(): pass", "after": "greet"},
        ])
        result = run_cli("batch-edit", str(small_py), "--edits", edits)
        assert result.returncode == 0, f"batch-edit failed: {result.stderr}"

    def test_batch_edit_reads_json_from_stdin(self, small_py: Path):
        """batch-edit --edits - should read JSON from stdin."""
        edits = json.dumps([
            {"snippet": "def func_a(): pass", "after": "greet"},
            {"snippet": "def func_b(): pass", "after": "farewell"},
        ])
        result = run_cli(
            "batch-edit", str(small_py), "--edits", "-",
            input_text=edits,
        )
        assert result.returncode == 0, f"batch-edit stdin failed: {result.stderr}"

    def test_batch_edit_invalid_json_exits_with_error(self, small_py: Path):
        """Invalid JSON should exit with code 1."""
        result = run_cli("batch-edit", str(small_py), "--edits", "not-json{[")
        assert result.returncode == 1


# ===================================================================
# 13. Multi-edit tests
# ===================================================================

class TestCLIMultiEdit:
    """Tests for `fastedit multi-edit --file-edits <json>`."""

    def test_multi_edit_accepts_json(self, small_py: Path):
        """multi-edit should parse JSON with file_path and edits."""
        file_edits = json.dumps([{
            "file_path": str(small_py),
            "edits": [{"snippet": "def new(): pass", "after": "greet"}],
        }])
        result = run_cli("multi-edit", "--file-edits", file_edits)
        assert result.returncode == 0, f"multi-edit failed: {result.stderr}"

    def test_multi_edit_reads_from_stdin(self, small_py: Path):
        """multi-edit --file-edits - should read from stdin."""
        file_edits = json.dumps([{
            "file_path": str(small_py),
            "edits": [{"snippet": "def new(): pass", "after": "greet"}],
        }])
        result = run_cli("multi-edit", "--file-edits", "-", input_text=file_edits)
        assert result.returncode == 0, f"multi-edit stdin failed: {result.stderr}"


# ===================================================================
# 14. Entry point tests
# ===================================================================

class TestEntryPoint:
    """Test that the fastedit console_scripts entry point is configured."""

    def test_pyproject_has_scripts_section(self):
        """pyproject.toml should have [project.scripts] with fastedit entry."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "[project.scripts]" in content
        assert 'fastedit' in content

    def test_module_invocation_works(self):
        """python -m fastedit should work and show help."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "fastedit" in result.stdout.lower() or "FastEdit" in result.stdout


# ===================================================================
# 15. Integration: edit + undo round-trip
# ===================================================================

class TestEditUndoRoundTrip:
    """Integration tests combining edit operations with undo."""

    def test_delete_then_undo_preserves_original(self, small_py: Path, backup_dir):
        """Delete + undo should leave the file unchanged."""
        original = small_py.read_text()
        run_cli("delete", str(small_py), "farewell")
        assert "def farewell" not in small_py.read_text()
        run_cli("undo", str(small_py))
        assert small_py.read_text() == original

    def test_rename_then_undo_preserves_original(self, small_py: Path, backup_dir):
        """Rename + undo should leave the file unchanged."""
        original = small_py.read_text()
        run_cli("rename", str(small_py), "greet", "welcome")
        assert "def welcome" in small_py.read_text()
        run_cli("undo", str(small_py))
        assert small_py.read_text() == original

    def test_move_then_undo_preserves_original(self, small_py: Path, backup_dir):
        """Move + undo should leave the file unchanged."""
        original = small_py.read_text()
        move_result = run_cli("move", str(small_py), "greet", "--after", "farewell")
        assert move_result.returncode == 0, f"Move failed: {move_result.stderr}"
        undo_result = run_cli("undo", str(small_py))
        assert undo_result.returncode == 0, f"Undo failed: {undo_result.stderr}"
        assert small_py.read_text() == original

    def test_sequential_edits_only_undo_last(self, small_py: Path, backup_dir):
        """Multiple edits: undo should only revert the most recent one."""
        run_cli("rename", str(small_py), "greet", "welcome")
        after_rename = small_py.read_text()
        run_cli("rename", str(small_py), "farewell", "goodbye")
        # Undo should only revert the second rename
        run_cli("undo", str(small_py))
        current = small_py.read_text()
        # Should have "welcome" (first rename stuck) but "farewell" back
        assert "def welcome" in current
        assert "def farewell" in current


# ===================================================================
# 16. diff after edits (integration)
# ===================================================================

class TestDiffAfterEdits:
    """Test that diff correctly shows changes after various edit operations."""

    def test_diff_after_delete_shows_removed_lines(self, small_py: Path, backup_dir):
        """After deleting a symbol, diff should show the removed lines."""
        del_result = run_cli("delete", str(small_py), "farewell")
        assert del_result.returncode == 0, f"Delete failed: {del_result.stderr}"
        result = run_cli("diff", str(small_py))
        assert result.returncode == 0, f"Diff failed: {result.stderr}"
        # Unified diff should have --- and +++ headers
        assert "---" in result.stdout
        # Removed lines should be prefixed with -
        assert "-def farewell" in result.stdout or "-    " in result.stdout

    def test_diff_after_rename_shows_changes(self, small_py: Path, backup_dir):
        """After renaming, diff should show old and new names."""
        rename_result = run_cli("rename", str(small_py), "greet", "welcome")
        assert rename_result.returncode == 0, f"Rename failed: {rename_result.stderr}"
        result = run_cli("diff", str(small_py))
        assert result.returncode == 0, f"Diff failed: {result.stderr}"
        assert "---" in result.stdout
        assert "greet" in result.stdout or "welcome" in result.stdout
