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
import time
from importlib import metadata as importlib_metadata
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
def backup_dir(tmp_path: Path, monkeypatch) -> Path:
    """Isolates HOME for CLI subprocesses; backups go to the session directory set by tests/conftest.py."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return Path(os.environ["FASTEDIT_BACKUP_DIR"])


# ---------------------------------------------------------------------------
# Helper: run CLI as subprocess
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None, env_extra: dict | None = None,
            cwd: Path | None = None):
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
        check=False,
        cwd=None if cwd is None else str(cwd),
    )


import importlib.util

_MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None


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

    def test_read_directory_exits_with_clean_error(self, tmp_path: Path):
        """Reading a directory is a clean exit-1, not an IsADirectoryError traceback."""
        result = run_cli("read", str(tmp_path))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "not a regular file" in result.stderr

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

    def test_edit_replace_rust_full_function_snippet_lands_parse_valid(
        self, tmp_path: Path,
    ):
        """A brace-language full-function replace must land parse-valid (exit 0).

        Step 21 smoke regression: the text-match splice for a rust snippet like
        ``pub fn add(...) -> i32 { a.wrapping_add(b) }`` anchored on the
        signature + closing brace and kept the OLD body line next to the new
        one — a content-faithful but parse-invalid merge that the parse gate
        could only refuse (exit 1, edit never landed). The text-match branch
        now declines its own parse-invalid output (the same standard the
        direct-swap branch already applies) and chunked_merge's qualified
        direct-swap produces the whole-symbol replacement instead.
        """
        from fastedit.data_gen.ast_analyzer import validate_parse

        original = (
            "pub fn add(a: i32, b: i32) -> i32 {\n"
            "    a + b\n"
            "}\n"
            "\n"
            "pub fn mul(a: i32, b: i32) -> i32 {\n"
            "    a * b\n"
            "}\n"
        )
        target = tmp_path / "math.rs"
        target.write_text(original)

        snippet = (
            "pub fn add(a: i32, b: i32) -> i32 {\n"
            "    a.wrapping_add(b)\n"
            "}\n"
        )
        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--replace", "add",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Applied edit to" in result.stdout

        merged = target.read_text()
        assert "a.wrapping_add(b)" in merged, merged
        # The old body line must NOT survive next to the new one.
        assert "a + b" not in merged, merged
        # Sibling function untouched.
        assert "a * b" in merged
        assert validate_parse(merged, "rust") is True, merged

    def test_edit_accepts_backend_flags(self):
        """--backend, --model-path, --api-base, --api-model should be accepted."""
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        assert "--backend" in result.stdout
        assert "--model-path" in result.stdout
        assert "--api-base" in result.stdout
        assert "--api-model" in result.stdout


# ===================================================================
# 2b. Issue #9: '-' stdin handling (bounded, TTY-safe, single read)
# ===================================================================

class TestDashStdinHelper:
    """Unit tests for the shared '-' reader behind --snippet, --content,
    --content-file, --edits and --file-edits (issue #9)."""

    def test_reads_piped_stdin_once_and_retires_it(self, monkeypatch):
        import io as io_mod

        from fastedit import cli as cli_module

        monkeypatch.setattr(sys, "stdin", io_mod.StringIO("def new(): pass\n"))
        data = cli_module._read_dash_stdin("--snippet")
        assert data == "def new(): pass\n"
        # After the single read, stdin is exhausted: any later consumer gets
        # EOF immediately instead of blocking ("never touch stdin again").
        assert sys.stdin.read() == ""

    def test_tty_stdin_is_refused_without_reading(self, monkeypatch, capsys):
        from fastedit import cli as cli_module

        class FakeTTY:
            def isatty(self):
                return True

            def read(self, *a, **kw):  # pragma: no cover - must never run
                raise AssertionError("stdin must never be read when it is a TTY")

        monkeypatch.setattr(sys, "stdin", FakeTTY())
        with pytest.raises(SystemExit) as exc_info:
            cli_module._read_dash_stdin("--snippet")
        assert exc_info.value.code == 1
        assert "requires piped stdin" in capsys.readouterr().err

    def test_missing_stdin_is_refused_cleanly(self, monkeypatch, capsys):
        from fastedit import cli as cli_module

        monkeypatch.setattr(sys, "stdin", None)
        with pytest.raises(SystemExit) as exc_info:
            cli_module._read_dash_stdin("--snippet")
        assert exc_info.value.code == 1
        assert "requires piped stdin" in capsys.readouterr().err

    def test_never_closing_pipe_times_out_instead_of_hanging(self, monkeypatch, capsys):
        """A producer that never closes stdin (an agent harness holding the
        pipe open) must abort within the configured bound, not hang forever."""
        from fastedit import cli as cli_module

        r, w = os.pipe()
        stdin_obj = os.fdopen(r, "r", encoding="utf-8")
        monkeypatch.setattr(sys, "stdin", stdin_obj)
        monkeypatch.setenv("FASTEDIT_STDIN_TIMEOUT_S", "0.3")
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_module._read_dash_stdin("--snippet")
        finally:
            os.close(w)  # EOF unblocks the pump thread; it exits as a daemon
            stdin_obj.close()
        assert exc_info.value.code == 1
        assert "timed out" in capsys.readouterr().err

    def test_binary_read_returns_bytes(self, monkeypatch):
        import io as io_mod

        from fastedit import cli as cli_module

        class FakeStdin:
            def __init__(self):
                self.buffer = io_mod.BytesIO(b"x = 1\n")

            def isatty(self):
                return False

        monkeypatch.setattr(sys, "stdin", FakeStdin())
        data = cli_module._read_dash_stdin("--content-file", binary=True)
        assert data == b"x = 1\n"


class TestCreateStdinFallback:
    """`fastedit create` with NEITHER --content nor --content-file reads its
    content from piped stdin (issue #9 cross-check): the fallback must give
    the SAME bounded, TTY-safe, single-read contract the '-' flag call sites
    got -- not a bare sys.stdin.buffer.read() that blocks an interactive
    terminal (or a harness pipe that never closes) forever with zero output."""

    @staticmethod
    def _args(target: Path):
        from types import SimpleNamespace

        return SimpleNamespace(
            file=str(target), content=None, content_file=None,
            force=False, parents=False,
        )

    def test_tty_stdin_is_refused_with_create_guidance(
        self, tmp_path: Path, backup_dir, monkeypatch, capsys,
    ):
        from fastedit import cli as cli_module

        class FakeTTY:
            def isatty(self):
                return True

            def read(self, *a, **kw):  # pragma: no cover - must never run
                raise AssertionError("stdin must never be read when it is a TTY")

        monkeypatch.setattr(sys, "stdin", FakeTTY())
        target = tmp_path / "made.txt"
        with pytest.raises(SystemExit) as exc_info:
            cli_module.cmd_create(self._args(target))
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "requires piped stdin" in err
        # The guidance must name the create command, not the edit flag form.
        assert "fastedit create" in err
        assert not target.exists()

    def test_piped_stdin_is_read_once_as_bytes(
        self, tmp_path: Path, backup_dir, monkeypatch,
    ):
        import io as io_mod

        from fastedit import cli as cli_module

        class FakePipedStdin:
            def __init__(self, data: bytes):
                self.buffer = io_mod.BytesIO(data)

            def isatty(self):
                return False

        monkeypatch.setattr(sys, "stdin", FakePipedStdin(b"hello from stdin\n"))
        target = tmp_path / "made.txt"
        cli_module.cmd_create(self._args(target))
        assert target.read_text(encoding="utf-8") == "hello from stdin\n"


class TestBatchEditCleanRefusals:
    """batch-edit must report merge refusals the way edit and multi-edit do
    (issue #13 cross-check): the merge raises ValueError for knowable
    refusals (oversized snippet, oversized prompt, ambiguous symbol), and a
    bare traceback is not a refusal."""

    def test_oversized_snippet_is_a_clean_refusal_not_a_traceback(
        self, tmp_path: Path, backup_dir, monkeypatch, capsys,
    ):
        import json as json_mod
        from types import SimpleNamespace

        from fastedit import cli as cli_module
        from fastedit.inference.chunked_merge import _MAX_SNIPPET_CHARS

        target = tmp_path / "notes.txt"
        target.write_text("hello\n", encoding="utf-8")

        def _fake_backend(args):
            def _boom(*a, **kw):  # pragma: no cover - must never run
                raise AssertionError("the model must never be called")

            return "fake", SimpleNamespace(merge_auto=_boom)

        monkeypatch.setattr(cli_module, "_make_backend_with_overrides", _fake_backend)
        edits = json_mod.dumps([{"snippet": "x" * (_MAX_SNIPPET_CHARS + 1)}])
        args = SimpleNamespace(
            file=str(target), edits=edits, backend=None, model_path=None,
            api_base=None, api_model=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            cli_module._cmd_batch_edit_locked(args)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "snippet too large" in err
        assert "Traceback" not in err
        # Nothing was written.
        assert target.read_text(encoding="utf-8") == "hello\n"


class TestCLIEditStdinDash:
    """End-to-end issue #9 behavior for `edit --snippet -`."""

    def test_edit_snippet_dash_with_tty_stdin_fails_fast_cleanly(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        """`--snippet -` with a TTY must exit 1 immediately with a clean
        error -- never block on the terminal waiting for input."""
        from fastedit import cli as cli_module

        class FakeTTY:
            def isatty(self):
                return True

            def read(self, *a, **kw):  # pragma: no cover - must never run
                raise AssertionError("stdin must never be read when it is a TTY")

        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        monkeypatch.setattr(sys, "stdin", FakeTTY())
        monkeypatch.setattr(
            sys, "argv",
            ["fastedit", "edit", str(target), "--snippet", "-", "--after", "greet"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli_module.main()

        assert exc_info.value.code == 1
        assert "requires piped stdin" in capsys.readouterr().err
        # Nothing was edited.
        assert target.read_text(encoding="utf-8") == SMALL_PYTHON_FILE

    def test_edit_snippet_dash_never_closing_pipe_times_out_cleanly(self, tmp_path: Path):
        """Issue #9 repro: a pipe that never reaches EOF (an agent harness
        holding stdin open) must not hang forever -- the bounded read aborts
        with a clean exit 1 within FASTEDIT_STDIN_TIMEOUT_S."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        env["FASTEDIT_STDIN_TIMEOUT_S"] = "0.5"
        env["FASTEDIT_NO_UPDATE_CHECK"] = "1"
        target = tmp_path / "mod.md"
        original = "# Title\n\nsome text\n"
        target.write_text(original, encoding="utf-8")

        r, w = os.pipe()
        try:
            proc = subprocess.Popen(
                [*CLI_MODULE, "edit", str(target), "--snippet", "-"],
                stdin=r,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        finally:
            os.close(r)
        started = time.monotonic()
        try:
            _out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:  # pragma: no cover - only on a hang regression
                proc.kill()
                proc.communicate()
            os.close(w)
        elapsed = time.monotonic() - started

        assert proc.returncode == 1
        assert "timed out" in err
        assert elapsed < 20, f"the bounded read took {elapsed:.1f}s -- effectively a hang"
        assert target.read_text(encoding="utf-8") == original


# ===================================================================
# 2c. Issue #7: @file syntax + existing-file auto-detection (--snippet)
# ===================================================================

class TestSnippetArgResolver:
    """Unit tests for the shared --snippet/--content resolver (issue #7)."""

    def test_atfile_reads_the_file_without_a_note(self, tmp_path: Path, capsys):
        from fastedit import cli as cli_module

        snippet_file = tmp_path / "repl.py"
        snippet_file.write_text("def greet():\n    return 'from file'\n", encoding="utf-8")
        text = cli_module._resolve_snippet_text_arg(f"@{snippet_file}", "--snippet", literal=False)
        assert text == "def greet():\n    return 'from file'\n"
        assert capsys.readouterr().err == ""

    def test_atfile_missing_file_is_a_clean_error(self, tmp_path: Path, capsys):
        from fastedit import cli as cli_module

        with pytest.raises(SystemExit) as exc_info:
            cli_module._resolve_snippet_text_arg(
                f"@{tmp_path / 'nope.py'}", "--snippet", literal=False,
            )
        assert exc_info.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_existing_single_line_path_is_autodetected_with_note(self, tmp_path: Path, capsys):
        from fastedit import cli as cli_module

        snippet_file = tmp_path / "repl.py"
        snippet_file.write_text("def greet():\n    return 'from file'\n", encoding="utf-8")
        text = cli_module._resolve_snippet_text_arg(str(snippet_file), "--snippet", literal=False)
        assert text == "def greet():\n    return 'from file'\n"
        err = capsys.readouterr().err
        assert "note: --snippet resolved to an existing file" in err
        assert "--snippet-is-literal" in err

    def test_literal_flag_disables_detection_and_returns_raw(self, tmp_path: Path, capsys):
        from fastedit import cli as cli_module

        snippet_file = tmp_path / "repl.py"
        snippet_file.write_text("def greet():\n    return 'from file'\n", encoding="utf-8")
        text = cli_module._resolve_snippet_text_arg(str(snippet_file), "--snippet", literal=True)
        assert text == str(snippet_file)
        assert capsys.readouterr().err == ""

    def test_multiline_text_is_never_treated_as_a_path(self, tmp_path: Path, capsys):
        from fastedit import cli as cli_module

        raw = "def greet():\n    return 1\n"
        assert cli_module._resolve_snippet_text_arg(raw, "--snippet", literal=False) == raw
        assert capsys.readouterr().err == ""

    def test_nonexistent_path_text_is_verbatim_without_note(self, capsys):
        from fastedit import cli as cli_module

        raw = "def greet(): return 1"
        assert cli_module._resolve_snippet_text_arg(raw, "--snippet", literal=False) == raw
        assert capsys.readouterr().err == ""

    def test_empty_text_is_verbatim(self, capsys):
        from fastedit import cli as cli_module

        assert cli_module._resolve_snippet_text_arg("", "--snippet", literal=False) == ""
        assert capsys.readouterr().err == ""


class TestCLIEditSnippetFileArgs:
    """End-to-end issue #7 behavior for `edit --snippet`."""

    def test_edit_snippet_atfile_reads_file_content(self, tmp_path: Path):
        snippet_file = tmp_path / "replacement.py"
        snippet_file.write_text(
            "def greet(name: str) -> str:\n    return 'from file'\n", encoding="utf-8",
        )
        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--snippet", f"@{snippet_file}",
            "--replace", "greet",
        )

        assert result.returncode == 0, result.stderr
        content = target.read_text(encoding="utf-8")
        assert "return 'from file'" in content
        assert 'return f"Hello, {name}!"' not in content
        # The literal '@path' string must never be spliced into the file.
        assert f"@{snippet_file}" not in content

    def test_edit_snippet_atfile_missing_file_is_clean_error(self, tmp_path: Path):
        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--snippet", f"@{tmp_path / 'nope.py'}",
            "--replace", "greet",
        )

        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "not found" in result.stderr
        assert target.read_text(encoding="utf-8") == SMALL_PYTHON_FILE

    def test_edit_snippet_existing_path_autodetected_with_note(self, tmp_path: Path):
        """Issue #7 repro: the agent passed a PATH as the snippet text. The
        path string must not replace the section; the referenced file's
        content is used instead, with a stderr note."""
        snippet_file = tmp_path / "replacement.py"
        snippet_file.write_text(
            "def greet(name: str) -> str:\n    return 'from file'\n", encoding="utf-8",
        )
        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--snippet", str(snippet_file),
            "--replace", "greet",
        )

        assert result.returncode == 0, result.stderr
        content = target.read_text(encoding="utf-8")
        assert "return 'from file'" in content
        # The path string itself must NOT have been spliced in.
        assert str(snippet_file) not in content
        assert "note: --snippet resolved to an existing file" in result.stderr
        assert "--snippet-is-literal" in result.stderr

    def test_edit_snippet_is_literal_uses_the_path_string_verbatim(self, tmp_path: Path):
        """file exists AND --snippet-is-literal → verbatim: the PATH STRING
        itself is the snippet (a valid Python expression statement), and the
        referenced file's content must NOT be read."""
        snippet_file = tmp_path / "repl.py"
        snippet_file.write_text(
            "def greet(name: str) -> str:\n    return 'from file'\n", encoding="utf-8",
        )
        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--snippet", "repl.py",  # relative: a valid Python expression statement
            "--after", "greet",
            "--snippet-is-literal",
            cwd=tmp_path,
        )

        assert result.returncode == 0, result.stderr
        content = target.read_text(encoding="utf-8")
        assert "repl.py" in content
        assert "return 'from file'" not in content
        assert "resolved to an existing file" not in result.stderr

    def test_edit_multiline_snippet_is_never_detected_as_a_path(self, tmp_path: Path):
        """A multi-line snippet stays verbatim even when its first line's
        text names an existing file."""
        (tmp_path / "def greet").write_text("decoy\n", encoding="utf-8")
        target = tmp_path / "mod.py"
        target.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        snippet = "def hello_world() -> str:\n    return 'Hi'\n"

        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--after", "greet",
            cwd=tmp_path,
        )

        assert result.returncode == 0, result.stderr
        assert "def hello_world" in target.read_text(encoding="utf-8")
        assert "resolved to an existing file" not in result.stderr


# ===================================================================
# 3. fastedit delete
# ===================================================================

class TestCLIEditFormatDefinitionLineGuard:
    """GitHub issue #1 part A: markdown ``--replace`` with a
    heading-OMITTING snippet used to splice the snippet over the WHOLE
    section span (the AST anchor covers heading + body), deleting the
    heading and every body line the snippet did not restate — exit 0,
    silent corruption.

    The definition-line guard in ``_try_deterministic_replace`` covered
    only code definition kinds (``_DEFINITION_KINDS``); format symbols
    (markdown section, json key, css rule, ...) now declare the same
    requirement declaratively on their ``_FormatSymbolSpec`` row
    (``definition_line_required=True``), so the guard fires for them too
    while a heading-restating snippet keeps landing byte-exact.
    """

    MD_ORIGINAL = (
        "# T\n"
        "\n"
        "## Features\n"
        "\n"
        "- A **one**: first\n"
        "- B **two**: old wording here\n"
        "- C **three**: third\n"
        "\n"
        "## Next\n"
        "\n"
        "text\n"
    )

    def test_edit_replace_markdown_heading_omitted_snippet_refused(
        self, tmp_path: Path,
    ):
        """The issue-#1 repro shape: --replace Features with a snippet that
        restates only two body lines. Must exit 1 with the definition-line
        refusal and leave the file byte-for-byte unchanged."""
        target = tmp_path / "doc.md"
        target.write_text(self.MD_ORIGINAL)
        snippet = "- A **one**: first\n- B **two**: new wording here\n"

        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--replace", "Features",
        )

        assert result.returncode == 1, (
            f"expected refusal, stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert (
            "Error: snippet for 'Features' (kind: section) has no definition "
            "line of its own"
        ) in result.stderr, result.stderr
        assert target.read_text() == self.MD_ORIGINAL
        assert "Applied edit" not in result.stdout

    def test_edit_replace_markdown_whole_section_snippet_rewrites_byte_exact(
        self, tmp_path: Path,
    ):
        """The legal shape: the snippet restates the heading (the section's
        definition line) and the whole body, so the section swap lands
        byte-exact. Must keep working."""
        target = tmp_path / "doc.md"
        target.write_text(self.MD_ORIGINAL)
        snippet = (
            "## Features\n"
            "\n"
            "- A **one**: first\n"
            "- B **two**: new wording here\n"
            "- C **three**: third\n"
        )
        expected = (
            "# T\n"
            "\n"
            "## Features\n"
            "\n"
            "- A **one**: first\n"
            "- B **two**: new wording here\n"
            "- C **three**: third\n"
            "\n"
            "## Next\n"
            "\n"
            "text\n"
        )

        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--replace", "Features",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Applied edit to" in result.stdout
        assert target.read_text() == expected

    def test_edit_replace_json_value_wipe_snippet_refused(
        self, tmp_path: Path,
    ):
        """Single-line JSON key with a wiped value ('"name": ') — the value
        is lost and the fragment is not a valid JSON document, so the edit
        is refused and the file is unchanged."""
        original = '{\n  "name": "old",\n  "version": "1.0.0"\n}\n'
        target = tmp_path / "data.json"
        target.write_text(original)

        result = run_cli(
            "edit", str(target),
            "--snippet", '"name": ',
            "--replace", "name",
        )

        assert result.returncode == 1
        assert "not valid json" in result.stderr, result.stderr
        assert target.read_text() == original


class TestCLIEditDeterministicFaithfulness:
    """GitHub issue #1 part B, hole 1: the CLI's
    ``_try_deterministic_replace`` returned text-match and direct-swap
    results WITHOUT the content battery that ``chunked_merge`` runs on the
    same editor output — so a battery-failing splice landed unwritten-
    validated with exit 0.

    (a) a ``# ... existing code ...`` snippet consumed by the CLI text-match
        path used to splice an editor output that dropped the preserved-gap
        semantics (marker written literally / body duplicated). After the fix
        the result must be FAITHFUL: every unmentioned original body line
        survives exactly once (preserve-by-default), the marker never lands
        in the file.

    (b) a direct-swap snippet that restates the definition line but drops
        unmentioned body lines (a format symbol whose "signature" is its
        first line) must NOT be spliced: the deterministic branch declines
        and the CLI falls through to the validated model path. For a
        hermetic test the model path is stubbed via the established CLI stub
        pattern (``_make_backend_with_overrides`` → fake backend; model
        result must honor the battery) and the fall-through is proven by
        the stub having been reached with a zero-token result shape.
    """

    MD_ORIGINAL = (
        "# T\n"
        "\n"
        "## Features\n"
        "\n"
        "- A **one**: first\n"
        "- B **two**: old wording here\n"
        "- C **three**: third\n"
        "\n"
        "## Next\n"
        "\n"
        "text\n"
    )

    def test_edit_replace_marker_snippet_via_text_match_is_faithful(
        self, tmp_path: Path,
    ):
        """Case e of the repro: ``## Features`` + ``# ... existing code ...``
        + one changed line. The CLI text-match path consumes the snippet;
        the battery must ratify ONLY a preserve-by-default output: body
        lines preserved exactly once, no marker leaked, no duplicate."""
        target = tmp_path / "doc_e.md"
        target.write_text(self.MD_ORIGINAL)
        snippet = (
            "## Features\n"
            "# ... existing code ...\n"
            "- B **two**: new wording here\n"
        )

        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--replace", "Features",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Applied edit to" in result.stdout
        content = target.read_text()
        # The marker is a directive, never content:
        assert "existing code" not in content, content
        # Preserve-by-default: every original body line survives exactly once.
        assert content.count("- A **one**: first") == 1, content
        assert content.count("- C **three**: third") == 1, content
        # The changed line landed exactly once.
        assert content.count("- B **two**: new wording here") == 1, content
        # The old wording did not silently duplicate alongside the new one
        # beyond its one preserved/restated occurrence.
        assert content.count("- B **two**: old wording here") <= 1, content
        # Sections other than the target are byte-identical.
        assert "## Next\n\ntext\n" in content, content

    def test_edit_replace_direct_swap_dropping_body_lines_falls_through_to_model_path(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        """A markdown --replace snippet that restates the heading but drops
        unmentioned body lines cannot be served by a wholesale swap (the
        grammar gives a section no complete-re-definition proof). The
        deterministic branch must decline — fall through to the model path,
        which is validated. Hermetic: chunked_merge is stubbed (the
        established CLI stub pattern) and asserts the fall-through happened
        by serving a battery-honoring merge."""
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit import cli as cli_module
        from fastedit.inference.ast_utils import ChunkedMergeResult

        target = tmp_path / "doc_b.md"
        target.write_text(self.MD_ORIGINAL)
        # Restates the definition line (## Features) but drops the other two
        # body lines without any marker — the part-A flag passes (the
        # heading IS restated), so only the new battery wrap catches it.
        snippet = (
            "## Features\n"
            "- B **two**: new wording here\n"
        )

        deterministic_declined: list[bool] = []
        real_try = cli_module._try_deterministic_replace

        def spy_try(*a, **kw):
            result = real_try(*a, **kw)
            deterministic_declined.append(result is None)
            return result

        monkeypatch.setattr(cli_module, "_try_deterministic_replace", spy_try)

        MODEL_MERGED = (
            "# T\n"
            "\n"
            "## Features\n"
            "\n"
            "- A **one**: first\n"
            "- B **two**: new wording here\n"
            "- C **three**: third\n"
            "\n"
            "## Next\n"
            "\n"
            "text\n"
        )

        def stubbed_chunked_merge(*a, **kw):
            # The stub mimics the validated model path: the REAL model path
            # runs the full battery over its output before returning, and
            # the CLI's write gates run after. The stub hands back the
            # preserve-by-default merge a battery-honoring path would
            # produce for this edit.
            return ChunkedMergeResult(
                merged_code=MODEL_MERGED,
                parse_valid=True,
                chunks_used=1,
                chunk_regions=[(3, 7)],
                model_tokens=0,
                latency_ms=0.0,
            )

        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", stubbed_chunked_merge,
        )
        monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")

        # Drive cmd_edit in-process (the spy cannot cross a process boundary).
        argv_backup = sys.argv
        sys.argv = [
            "fastedit", "edit", str(target),
            "--snippet", snippet, "--replace", "Features",
        ]
        try:
            cli_module.main()
        finally:
            sys.argv = argv_backup

        # The fall-through DID happen: the deterministic branch declined.
        assert deterministic_declined == [True], (
            f"expected the deterministic branch to decline; got {deterministic_declined}"
        )
        # The validated model-path result was written.
        out = capsys.readouterr().out
        assert "Applied edit to" in out, out
        assert target.read_text() == MODEL_MERGED

    def test_edit_replace_direct_swap_keeps_working_for_complete_redefinition(
        self, tmp_path: Path,
    ):
        """Control: a snippet that restates the heading AND every body line
        is a complete re-definition — the direct swap keeps landing
        byte-exact (no battery regression on the wholesale path)."""
        target = tmp_path / "doc_g.md"
        target.write_text(self.MD_ORIGINAL)
        snippet = (
            "## Features\n"
            "\n"
            "- A **one**: first\n"
            "- B **two**: new wording here\n"
            "- C **three**: third\n"
        )
        expected = (
            "# T\n"
            "\n"
            "## Features\n"
            "\n"
            "- A **one**: first\n"
            "- B **two**: new wording here\n"
            "- C **three**: third\n"
            "\n"
            "## Next\n"
            "\n"
            "text\n"
        )

        result = run_cli(
            "edit", str(target),
            "--snippet", snippet,
            "--replace", "Features",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Applied edit to" in result.stdout
        assert target.read_text() == expected


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
# 5b. fastedit rename / rename-all — B37 concurrent-modification guard
# ===================================================================

class TestCLIRenameConcurrentWrite:
    """B37 lost-update guard for the rename verbs.

    Mirrors the multi-edit race tests (section 13): a deterministic seam (a
    wrapper around the rename engine or the per-file lock) interposes a
    NON-fastedit write between the verb's read and its write, because a real
    concurrent writer would be timing-dependent and flaky, while this
    reproduces "the file changed between being read and being written" on
    every run. Runs cli.main() IN-PROCESS (`cli.main()` with a patched
    `sys.argv`) rather than via subprocess, because the monkeypatched spy
    cannot reach across a `subprocess.run` process boundary. Nothing is
    stubbed — every test drives the real tldr-backed rename engine.
    """

    def test_rename_refuses_when_file_changed_after_read(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        """A non-fastedit write between rename's read and its write is refused:
        exit 1, and the EXTERNAL content — not the rename — stays on disk."""
        from fastedit import cli as cli_module
        from fastedit.inference import rename as rename_module

        target = tmp_path / "mod.py"
        target.write_text("def old_name():\n    return 1\n\nx = old_name()\n")
        external_bytes = b"# rewritten by a non-fastedit writer\n"

        real_do_rename_ast = rename_module.do_rename_ast
        engine_counts: list[int] = []

        def spy_do_rename_ast(path, old_name, new_name):
            renamed, count, skipped = real_do_rename_ast(path, old_name, new_name)
            engine_counts.append(count)
            # The external writer lands AFTER the engine's read of the file
            # (and after the CLI's own read), BEFORE the CLI's write.
            target.write_bytes(external_bytes)
            return renamed, count, skipped

        monkeypatch.setattr(rename_module, "do_rename_ast", spy_do_rename_ast)
        monkeypatch.setattr(
            sys, "argv",
            ["fastedit", "rename", str(target), "old_name", "new_name"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli_module.main()

        # Checked FIRST: the rename engine really ran and really found
        # references, so the fault was injected mid-run — not short-circuited
        # by an early "no code references" exit (which would make the refusal
        # below the wrong refusal and the setup itself broken).
        assert engine_counts and engine_counts[0] >= 1
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "file changed on disk" in captured.err
        assert "Nothing was written" in captured.err
        # The external content is what's on disk — the rename was NOT applied.
        assert target.read_bytes() == external_bytes

    def test_rename_all_refuses_changed_file_but_still_renames_the_rest(
        self, tmp_path: Path, monkeypatch, capsys,
    ):
        """rename-all: the file changed after the plan read it is refused; the
        other file still renames (the verb's partial per-file semantics)."""
        from fastedit import cli as cli_module
        from fastedit.inference import rename as rename_module

        body = "def old_sym():\n    return 1\n\nx = old_sym()\n"
        root = tmp_path / "proj"
        root.mkdir()
        survivor = root / "survivor.py"
        victim = root / "victim.py"
        survivor.write_text(body)
        victim.write_text(body)
        external_bytes = b"# rewritten by a non-fastedit writer\n"

        real_do_cross_file_rename = rename_module.do_cross_file_rename
        planned: dict = {}

        def ordered_do_cross_file_rename(root_dir, old_name, new_name, **kwargs):
            plan = real_do_cross_file_rename(root_dir, old_name, new_name, **kwargs)
            planned.update(plan)
            # Deterministic write order: survivor first, victim last (False
            # sorts before True), so the survivor's rename lands before the
            # victim's refusal.
            return {p: plan[p] for p in sorted(plan, key=lambda p: p == victim)}

        monkeypatch.setattr(
            rename_module, "do_cross_file_rename", ordered_do_cross_file_rename,
        )

        real_locked_for_edit = cli_module._locked_for_edit
        interposed: list[Path] = []

        def spy_locked_for_edit(path):
            if path == victim and not interposed:
                # The non-fastedit writer clobbers the victim after the plan
                # read it, before this file's locked write.
                interposed.append(path)
                victim.write_bytes(external_bytes)
            return real_locked_for_edit(path)

        monkeypatch.setattr(cli_module, "_locked_for_edit", spy_locked_for_edit)
        monkeypatch.setattr(
            sys, "argv",
            ["fastedit", "rename-all", str(root), "old_sym", "new_sym"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli_module.main()

        # Checked FIRST: both files were really planned (tldr found references
        # in each) and the external write really landed mid-run — not
        # short-circuited by an early no-plan exit.
        assert set(planned) == {survivor, victim}
        assert interposed == [victim]
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "file changed on disk" in captured.err
        assert "Nothing was written" in captured.err
        # The changed file keeps the EXTERNAL content — no rename applied to it.
        assert victim.read_bytes() == external_bytes
        # The other file still renamed: partial semantics consistent with the verb.
        survivor_text = survivor.read_text()
        assert "def new_sym" in survivor_text
        assert "def old_sym" not in survivor_text

    def test_rename_happy_path_still_writes(self, tmp_path: Path, monkeypatch):
        """With no concurrent writer, rename still applies and succeeds."""
        from fastedit import cli as cli_module

        target = tmp_path / "mod.py"
        target.write_text("def old_name():\n    return 1\n\nx = old_name()\n")
        monkeypatch.setattr(
            sys, "argv",
            ["fastedit", "rename", str(target), "old_name", "new_name"],
        )

        cli_module.main()  # no SystemExit: the command succeeded

        content = target.read_text()
        assert "def new_name" in content
        assert "def old_name" not in content

    def test_rename_all_happy_path_still_writes_every_planned_file(
        self, tmp_path: Path, monkeypatch,
    ):
        """With no concurrent writer, rename-all still renames every planned file."""
        from fastedit import cli as cli_module

        body = "def old_sym():\n    return 1\n\nx = old_sym()\n"
        root = tmp_path / "proj"
        root.mkdir()
        (root / "a.py").write_text(body)
        (root / "b.py").write_text(body)
        monkeypatch.setattr(
            sys, "argv",
            ["fastedit", "rename-all", str(root), "old_sym", "new_sym"],
        )

        cli_module.main()  # no SystemExit: the command succeeded

        for name in ("a.py", "b.py"):
            content = (root / name).read_text()
            assert "def new_sym" in content
            assert "def old_sym" not in content


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

    def test_help_epilog_names_installer_doctor_and_mcp_install(self):
        """Top-level --help points at the installer, the doctor, and MCP setup."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "scripts/install-dev.sh" in result.stdout
        assert "fastedit doctor" in result.stdout
        assert "fastedit mcp-install" in result.stdout

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

    def test_batch_edit_rejects_non_list_edits_cleanly(self, small_py: Path):
        """A JSON object (not a list) is a clean exit-1, not a traceback."""
        result = run_cli("batch-edit", str(small_py), "--edits", '{"snippet": "x = 1"}')
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "'snippet'" in result.stderr

    def test_batch_edit_rejects_item_missing_snippet_cleanly(self, small_py: Path):
        """An edit item without 'snippet' is a clean exit-1, not a KeyError."""
        result = run_cli("batch-edit", str(small_py), "--edits", '[{"after": "greet"}]')
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "'snippet'" in result.stderr

    def test_batch_edit_rejects_non_object_item_cleanly(self, small_py: Path):
        """A bare string inside the edits list is a clean exit-1."""
        result = run_cli("batch-edit", str(small_py), "--edits", '["just a string"]')
        assert result.returncode == 1
        assert "Traceback" not in result.stderr


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

    @pytest.mark.skipif(not _MLX_AVAILABLE, reason="multi-edit constructs a backend unconditionally; needs mlx")
    def test_multi_edit_refuses_all_writes_when_a_target_changes_after_being_read(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A file mutated after being read/merged but before the write phase aborts the WHOLE batch untouched.

        Uses a deterministic seam (wrapping `_refuse_if_edit_broke_parse`, which multi-edit
        already calls once per entry in list order) instead of real threads/sleeps: a genuine
        concurrent writer would be timing-dependent and flaky, while this reproduces "a target
        changed between being read and the write phase" on every run. Runs multi-edit IN-PROCESS
        (`cli.main()` with a patched `sys.argv`) rather than via subprocess, because the
        monkeypatched spy cannot reach across a `subprocess.run` process boundary. Neither target
        is stubbed or skipped -- both go through the real merge backend.
        """
        from fastedit import cli as cli_module

        first_py = tmp_path / "first.py"
        second_py = tmp_path / "second.py"
        first_py.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        second_py.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        original_second_bytes = second_py.read_bytes()
        mutator_bytes = b"# mutated by a concurrent writer\n"

        real_refuse = cli_module._refuse_if_edit_broke_parse
        call_count = 0

        def spy(path, original_code, merged_code, language):
            nonlocal call_count
            call_count += 1
            result = real_refuse(path, original_code, merged_code, language)
            if path == second_py:
                first_py.write_bytes(mutator_bytes)
            return result

        monkeypatch.setattr(cli_module, "_refuse_if_edit_broke_parse", spy)

        file_edits = json.dumps([
            {"file_path": str(first_py), "edits": [{"snippet": "def new(): pass", "after": "greet"}]},
            {"file_path": str(second_py), "edits": [{"snippet": "def new(): pass", "after": "greet"}]},
        ])
        monkeypatch.setattr(sys, "argv", ["fastedit", "multi-edit", "--file-edits", file_edits])

        with pytest.raises(SystemExit) as exc_info:
            cli_module.main()

        # Checked FIRST: `_refuse_if_edit_broke_parse` is called once per entry, in list order,
        # inside PHASE 2 -- so a count of 2 proves both targets were actually merged (the fault
        # was injected mid-run, not short-circuited by an early exit such as backend resolution
        # failing before PHASE 2 ever starts). A count mismatch here means the test setup itself
        # is broken, not that the fix under test is missing.
        assert call_count == 2
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "changed since being read" in captured.err
        assert str(first_py) in captured.err
        assert "no files were modified" in captured.err
        assert first_py.read_bytes() == mutator_bytes
        assert second_py.read_bytes() == original_second_bytes

    @pytest.mark.skipif(not _MLX_AVAILABLE, reason="multi-edit constructs a backend unconditionally; needs mlx")
    def test_multi_edit_refuses_all_writes_when_a_target_vanishes_after_being_read(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A target deleted after being read/merged but before the write phase aborts the WHOLE batch, no traceback.

        Same deterministic seam as the mutation test above, but the spy deletes the earlier
        target instead of overwriting it, exercising the FileNotFoundError re-read path in the
        verification phase. Neither target is stubbed or skipped -- both go through the real
        merge backend.
        """
        from fastedit import cli as cli_module

        first_py = tmp_path / "first.py"
        second_py = tmp_path / "second.py"
        first_py.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        second_py.write_text(SMALL_PYTHON_FILE, encoding="utf-8")
        original_second_bytes = second_py.read_bytes()

        real_refuse = cli_module._refuse_if_edit_broke_parse
        call_count = 0

        def spy(path, original_code, merged_code, language):
            nonlocal call_count
            call_count += 1
            result = real_refuse(path, original_code, merged_code, language)
            if path == second_py:
                first_py.unlink()
            return result

        monkeypatch.setattr(cli_module, "_refuse_if_edit_broke_parse", spy)

        file_edits = json.dumps([
            {"file_path": str(first_py), "edits": [{"snippet": "def new(): pass", "after": "greet"}]},
            {"file_path": str(second_py), "edits": [{"snippet": "def new(): pass", "after": "greet"}]},
        ])
        monkeypatch.setattr(sys, "argv", ["fastedit", "multi-edit", "--file-edits", file_edits])

        with pytest.raises(SystemExit) as exc_info:
            cli_module.main()

        # Checked FIRST -- see the sibling test above for why this must come before the exit
        # code / byte assertions.
        assert call_count == 2
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "changed since being read" in captured.err
        assert str(first_py) in captured.err
        assert "no files were modified" in captured.err
        assert not first_py.exists()
        assert second_py.read_bytes() == original_second_bytes


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


# ===================================================================
# 17. fastedit --version / -V
# ===================================================================

class TestCLIVersion:
    """`fastedit --version` / `-V`: the first thing anyone tries after installing.

    Contract: exits 0, prints ONLY the version line to stdout (so it can be
    captured by scripts), and reports the installed fastedits metadata version
    — with a "0.0.0+unknown" fallback for a source checkout without install.
    `--version` / `-V` are MAIN-parser flags: they print before any command
    dispatch, so `fastedit --version` needs no subcommand at all.
    """

    def _expected_version(self) -> str:
        """The version a correct `--version` line reports.

        importlib.metadata.version('fastedits') when the package is installed
        in this interpreter (the fastedit running under the test venv), else
        the same "0.0.0+unknown" fallback the CLI itself uses. Kept in lockstep
        with the CLI on purpose: the flag's whole job is to report this.
        """
        try:
            return importlib_metadata.version("fastedits")
        except importlib_metadata.PackageNotFoundError:
            return "0.0.0+unknown"

    def test_version_exits_zero_and_prints_the_metadata_version(self):
        result = run_cli("--version")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"fastedit {self._expected_version()}\n"

    def test_uppercase_v_alias_prints_the_same_line(self):
        short = run_cli("-V")
        assert short.returncode == 0, short.stderr
        assert short.stdout == f"fastedit {self._expected_version()}\n"

    def test_version_goes_to_stdout_not_stderr(self):
        """Scripts capture stdout; the version line must be there."""
        result = run_cli("--version")
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("fastedit ")
        assert result.stderr == ""

    def test_version_prints_nothing_else(self):
        """Exactly one line — no help text, no update notice, no banner."""
        result = run_cli("--version")
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.strip().splitlines()) == 1
        assert result.stdout.strip().splitlines()[0].startswith("fastedit ")

    def test_version_works_without_a_subcommand(self):
        """`fastedit --version` alone is valid argv — no 'the following arguments are required' error.

        The flag sits on the MAIN parser and argparse resolves it during
        parse_args, before the subparser's required-action check runs.
        """
        result = run_cli("--version")
        assert result.returncode == 0, result.stderr
        assert "required" not in result.stderr.lower()

    def test_version_takes_precedence_over_the_subcommand(self):
        """`fastedit --version read x.py` still prints the version and never touches the filesystem."""
        result = run_cli("--version", "read", "nonexistent-file-for-version-test.py")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"fastedit {self._expected_version()}\n"
        assert not (Path.cwd() / "nonexistent-file-for-version-test.py").exists()

    def test_version_is_absent_from_the_subcommand_parsers(self):
        """No subparser gains a --version: it stays a top-level flag, so
        `fastedit doctor --version` is NOT the version printer."""
        import argparse as argparse_mod

        from fastedit import cli as cli_mod

        original_parse = argparse_mod.ArgumentParser.parse_args
        captured: dict = {}
        sentinel = type("_Stop", (Exception,), {})

        def _capture(self, *a, **kw):
            captured["parser"] = self
            raise sentinel

        argparse_mod.ArgumentParser.parse_args = _capture
        try:
            try:
                cli_mod.main()
            except sentinel:
                pass
        finally:
            argparse_mod.ArgumentParser.parse_args = original_parse

        parser = captured["parser"]
        assert any(
            isinstance(a, argparse_mod._VersionAction) for a in parser._actions
        ), "the main parser must carry a version action"
        for action in parser._actions:
            if isinstance(action, argparse_mod._SubParsersAction):
                for name, sub in action.choices.items():
                    assert not any(
                        isinstance(a, argparse_mod._VersionAction) for a in sub._actions
                    ), f"subcommand '{name}' must not carry a --version flag"


# ---------------------------------------------------------------------------
# Issue #5 — addressable test blocks, constructors, and the imports block
# (printed slugs must be exactly the resolvable ones)
# ---------------------------------------------------------------------------

JEST_TS_FILE = (
    'import { describe, it, expect } from "vitest";\n'
    'import { createServer } from "./server";\n'
    "\n"
    "const PORT = 3111;\n"
    "\n"
    'describe("server", () => {\n'
    '  it("renders correctly", () => {\n'
    '    const el = renderBadge("hello");\n'
    '    expect(el.textContent).toBe("hello");\n'
    "  });\n"
    "\n"
    '  it("handles empty label", () => {\n'
    '    expect(renderBadge("")).toBeNull();\n'
    "  });\n"
    "});\n"
)

TS_CLASS_FILE = (
    "export class Server {\n"
    "  readonly port: number;\n"
    "\n"
    "  constructor(port: number) {\n"
    "    this.port = port;\n"
    "  }\n"
    "\n"
    "  start(): void {\n"
    "    console.log(this.port);\n"
    "  }\n"
    "}\n"
)


def _jest_read_file() -> str:
    """A >100-line jest file so `fastedit read` takes the structure path."""
    lines = [
        'import { describe, it, expect } from "vitest";',
        'import { createServer } from "./server";',
        "",
        'describe("server", () => {',
    ]
    for i in range(30):
        lines += [
            f'  it("test number {i}", () => {{',
            f"    expect({i}).toBe({i});",
            "  });",
            "",
        ]
    lines += ["});", ""]
    return "\n".join(lines)


class TestCLIIssue5AddressableSymbols:
    """`fastedit read` lists test blocks, constructors, and the imports
    block; every listed slug resolves through --replace/--after."""

    def test_read_lists_test_blocks_and_imports(self, tmp_path: Path):
        target = tmp_path / "server.test.ts"
        target.write_text(_jest_read_file(), encoding="utf-8")
        result = run_cli("read", str(target))
        assert result.returncode == 0, result.stderr
        assert "describe:server" in result.stdout
        assert "it:test-number-0" in result.stdout
        # the imports block is shown with its line span and its name
        assert "L1-2" in result.stdout
        assert "imports" in result.stdout

    def test_replace_test_block_byte_exact(self, tmp_path: Path):
        target = tmp_path / "badge.test.ts"
        target.write_text(JEST_TS_FILE, encoding="utf-8")
        snippet = (
            '  it("renders correctly", () => {\n'
            '    const el = renderBadge("hi");\n'
            '    expect(el.textContent).toBe("hi");\n'
            "  });\n"
        )
        result = run_cli(
            "edit", str(target),
            "--replace", "it:renders-correctly",
            "--snippet", snippet,
        )
        assert result.returncode == 0, result.stderr
        expected = JEST_TS_FILE.replace(
            'renderBadge("hello")', 'renderBadge("hi")'
        ).replace('toBe("hello")', 'toBe("hi")')
        assert target.read_text(encoding="utf-8") == expected

    def test_replace_test_block_body_only_snippet_is_refused(self, tmp_path: Path):
        """A body-only snippet would delete the naming it() line — the
        definition-line guard must refuse it for a test-block target."""
        target = tmp_path / "badge.test.ts"
        target.write_text(JEST_TS_FILE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace", "it:renders-correctly",
            "--snippet", '    expect(renderBadge("hi")).toBeNull();\n',
        )
        assert result.returncode == 1
        assert "has no definition" in result.stderr
        assert target.read_text(encoding="utf-8") == JEST_TS_FILE

    def test_replace_constructor(self, tmp_path: Path):
        target = tmp_path / "server.ts"
        target.write_text(TS_CLASS_FILE, encoding="utf-8")
        snippet = (
            "  constructor(port: number) {\n"
            "    this.port = port * 2;\n"
            "  }\n"
        )
        result = run_cli(
            "edit", str(target),
            "--replace", "constructor",
            "--snippet", snippet,
        )
        assert result.returncode == 0, result.stderr
        expected = TS_CLASS_FILE.replace(
            "this.port = port;", "this.port = port * 2;"
        )
        assert target.read_text(encoding="utf-8") == expected

    def test_after_imports_typescript(self, tmp_path: Path):
        target = tmp_path / "server.test.ts"
        target.write_text(JEST_TS_FILE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--after-imports",
            "--snippet", 'import { render } from "./test-utils";',
        )
        assert result.returncode == 0, result.stderr
        expected = JEST_TS_FILE.replace(
            'import { createServer } from "./server";\n',
            'import { createServer } from "./server";\n\n'
            'import { render } from "./test-utils";\n',
        )
        assert target.read_text(encoding="utf-8") == expected

    def test_after_name_imports_works_too(self, tmp_path: Path):
        """The import block is a plain resolvable symbol, so the documented
        `--after imports` form anchors on it as well."""
        target = tmp_path / "server.test.ts"
        target.write_text(JEST_TS_FILE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--after", "imports",
            "--snippet", 'import { render } from "./test-utils";',
        )
        assert result.returncode == 0, result.stderr
        expected = JEST_TS_FILE.replace(
            'import { createServer } from "./server";\n',
            'import { createServer } from "./server";\n\n'
            'import { render } from "./test-utils";\n',
        )
        assert target.read_text(encoding="utf-8") == expected

    def test_after_imports_python(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(
            "import os\nimport sys\n\n\ndef main() -> None:\n    print(os.name)\n",
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target),
            "--after-imports",
            "--snippet", "import json",
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text(encoding="utf-8") == (
            "import os\n"
            "import sys\n"
            "\n"
            "import json\n"
            "\n"
            "\n"
            "def main() -> None:\n"
            "    print(os.name)\n"
        )

    def test_after_imports_javascript(self, tmp_path: Path):
        target = tmp_path / "app.js"
        target.write_text(
            'import a from "a";\nimport b from "b";\nfunction f() {\n  return a;\n}\n',
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target),
            "--after-imports",
            "--snippet", 'import c from "c";',
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text(encoding="utf-8") == (
            'import a from "a";\n'
            'import b from "b";\n'
            "\n"
            'import c from "c";\n'
            "\n"
            "function f() {\n"
            "  return a;\n"
            "}\n"
        )

    def test_after_imports_without_imports_block_refuses(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--after-imports",
            "--snippet", "import json",
        )
        assert result.returncode == 1
        assert "Symbol 'imports' not found" in result.stderr
        assert target.read_text(encoding="utf-8") == "def main() -> None:\n    pass\n"

    def test_after_imports_conflicts_with_after(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("import os\n\n\ndef main():\n    pass\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--after", "main",
            "--after-imports",
            "--snippet", "import json",
        )
        assert result.returncode == 2

    def test_batch_edit_after_imports(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(
            "import os\nimport sys\n\n\ndef main():\n    print(os.name)\n",
            encoding="utf-8",
        )
        result = run_cli(
            "batch-edit", str(target),
            "--edits", '[{"snippet": "import json", "after": "imports"}]',
        )
        assert result.returncode == 0, result.stderr
        assert target.read_text(encoding="utf-8") == (
            "import os\n"
            "import sys\n"
            "\n"
            "import json\n"
            "\n"
            "\n"
            "def main():\n"
            "    print(os.name)\n"
        )


# ===================================================================
# 18. Issue #15 — `fastedit diff --base {last,all}` (default: last)
# ===================================================================

PY_ORIGINAL_DIFF = """def target():
    a = 1

def helper():
    b = 2
"""


class TestCLIDiffBaseFlag:
    """Issue #15: `fastedit diff` used to diff against the OLDEST backup
    (the issue-#8 loss-detection choice) so agents verifying an edit saw
    EVERY still-undoable hunk, not the one their edit just made.

    New contract: `--base last` (the DEFAULT) diffs against the NEWEST
    backup — exactly the last edit's change, what a verification step
    expects. `--base all` keeps the issue-#8 behavior: every change still
    in the undo history, for loss detection across multiple edits. The
    --help text states the baseline explicitly."""

    def _delete_helper(self, target: Path):
        deleted = run_cli("delete", str(target), "helper")
        assert deleted.returncode == 0, (
            f"stdout: {deleted.stdout!r} stderr: {deleted.stderr!r}"
        )

    def _rewrite_target(self, target: Path):
        result = run_cli(
            "edit", str(target),
            "--snippet", "def target():\n    a = 2\n",
            "--replace", "target",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )

    def test_default_base_is_last_shows_only_the_newest_hunk(
        self, tmp_path: Path, monkeypatch,
    ):
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL_DIFF)

        self._delete_helper(target)     # edit 1: removes the helper function
        self._rewrite_target(target)    # edit 2: a = 1 -> a = 2

        diff = run_cli("diff", str(target))
        assert diff.returncode == 0, diff.stderr
        # The last edit's hunk IS shown...
        assert "-    a = 1" in diff.stdout, diff.stdout
        assert "+    a = 2" in diff.stdout, diff.stdout
        # ...and the earlier edit's removal is NOT: those lines were still
        # present in the newest backup (edit 2's pre-state).
        assert "-def helper():" not in diff.stdout, diff.stdout
        assert "-    b = 2" not in diff.stdout, diff.stdout

    def test_base_last_flag_is_accepted_explicitly(
        self, tmp_path: Path, monkeypatch,
    ):
        """`--base last` spelled out behaves exactly like the default."""
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL_DIFF)

        self._delete_helper(target)
        self._rewrite_target(target)

        diff = run_cli("diff", str(target), "--base", "last")
        assert diff.returncode == 0, diff.stderr
        assert "-    a = 1" in diff.stdout, diff.stdout
        assert "+    a = 2" in diff.stdout, diff.stdout
        assert "-def helper():" not in diff.stdout, diff.stdout
        assert "-    b = 2" not in diff.stdout, diff.stdout

    def test_base_all_shows_every_still_undoable_change(
        self, tmp_path: Path, monkeypatch,
    ):
        """`--base all` keeps the issue-#8 loss-detection behavior: the
        diff base is the OLDEST surviving backup, so the removal from the
        EARLIER edit stays visible alongside the newest hunk."""
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL_DIFF)

        self._delete_helper(target)
        self._rewrite_target(target)

        diff = run_cli("diff", str(target), "--base", "all")
        assert diff.returncode == 0, diff.stderr
        # The removal from the EARLIER edit appears even though the newest
        # edit's pre-state still contained it.
        assert "-def helper():" in diff.stdout, diff.stdout
        assert "-    b = 2" in diff.stdout, diff.stdout
        # The newest edit's change is surfaced too.
        assert "-    a = 1" in diff.stdout, diff.stdout
        assert "+    a = 2" in diff.stdout, diff.stdout

    def test_base_last_after_single_edit_shows_that_edit(
        self, tmp_path: Path, monkeypatch,
    ):
        """One edit in history: `--base last` shows exactly that edit —
        the common verify-after-edit flow."""
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL_DIFF)

        self._rewrite_target(target)

        diff = run_cli("diff", str(target))
        assert diff.returncode == 0, diff.stderr
        assert "-    a = 1" in diff.stdout, diff.stdout
        assert "+    a = 2" in diff.stdout, diff.stdout

    def test_invalid_base_choice_is_rejected_by_argparse(
        self, tmp_path: Path, monkeypatch,
    ):
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL_DIFF)
        result = run_cli("diff", str(target), "--base", "oldest")
        assert result.returncode == 2

    def test_diff_help_states_the_baseline_explicitly(self):
        """`--help` must say which backup is the diff base — agents could
        not tell whether diff showed the last edit or the whole history."""
        result = run_cli("diff", "--help")
        assert result.returncode == 0
        assert "--base" in result.stdout
        help_text = result.stdout.lower()
        assert "last" in help_text
        assert "all" in help_text
        assert "newest" in help_text
        assert "oldest" in help_text


# ===================================================================
# 19. Issue #14 — hermetic literal line primitives on `fastedit edit`
# ===================================================================

RUST_BIG_FN = (
    "fn process_item(item: &Item) -> Result<(), Error> {\n"
    "    let raw = item.as_raw();\n"
    "    let checked = validate(&raw)?;\n"
    "    let parsed = parse(&checked)?;\n"
    "    let normalized = normalize(parsed);\n"
    "    let scored = score(&normalized);\n"
    "    let ranked = rank(scored);\n"
    "    let filtered = filter_valid(ranked);\n"
    "    let staged = stage(filtered);\n"
    "    let committed = commit(staged)?;\n"
    "    let indexed = index(committed);\n"
    "    let stored = store(indexed);\n"
    "    let packed = pack(stored);\n"
    "    let sealed = seal(packed);\n"
    "    let shipped = ship(sealed);\n"
    "    let logged = log(shipped);\n"
    "    let audited = audit(logged);\n"
    "    let archived = archive(audited);\n"
    "    let completed = finalize(archived);\n"
    "    Ok(completed)\n"
    "}\n"
)

RUST_UNIQUE_CLOSER = "    let audited = audit(logged);\n"


class TestCLIEditInsertAfterBefore:
    """Issue #14 refusal class 4 (one-line insertion inside a big Rust
    function): a hermetic, zero-model, deterministic line-insertion
    primitive keyed on a UNIQUE full-line literal.

    Semantics: the snippet (--snippet/@path/-) is spliced immediately
    AFTER (--insert-after) or BEFORE (--insert-before) the first line
    whose full stripped content equals the literal. 0 matches and >1
    matches are loud exit-1 refusals with the remedy spelled out."""

    def _literal(self, target: Path, flag: str, literal: str, snippet: str):
        return run_cli(
            "edit", str(target),
            "--snippet", snippet,
            flag, literal,
        )

    def test_rust_shaped_one_line_insertion_lands_zero_tokens(
        self, tmp_path: Path, backup_dir,
    ):
        """The issue's acceptance shape: a 200+-line function, one new line
        after a unique `...;` statement line. Lands with exit 0, 0 model
        tokens, and everything else byte-identical."""
        big = "fn big(x: i32) -> i32 {\n" + "".join(
            f"    let step_{i} = step_{i - 1} + 1;\n" for i in range(1, 199)
        ) + "    let result = step_198;\n    result\n}\n"
        assert len(big.splitlines()) >= 200
        target = tmp_path / "big.rs"
        target.write_text(big)
        unique = "    let step_42 = step_41 + 1;\n"

        result = self._literal(
            target, "--insert-after", "let step_42 = step_41 + 1;",
            "    let checkpoint = step_42;\n",
        )

        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text()
        lines = content.splitlines(keepends=True)
        idx = lines.index(unique)
        assert lines[idx + 1] == "    let checkpoint = step_42;\n"
        # Byte-identical apart from the one inserted line.
        assert content == big.replace(
            unique, unique + "    let checkpoint = step_42;\n",
        )
        # Hermetic: no model work happened.
        assert "0 tokens" in result.stdout, result.stdout

    def test_insert_after_uses_snippet_as_payload(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = self._literal(
            target, "--insert-after", "let audited = audit(logged);",
            "    audit_trail.record(audited);\n",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text(encoding="utf-8")
        assert content == RUST_BIG_FN.replace(
            RUST_UNIQUE_CLOSER,
            RUST_UNIQUE_CLOSER + "    audit_trail.record(audited);\n",
        )
        assert "0 tokens" in result.stdout

    def test_insert_before_puts_snippet_above_the_literal(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = self._literal(
            target, "--insert-before", "let audited = audit(logged);",
            "    audit_trail.begin();\n",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text(encoding="utf-8")
        assert content == RUST_BIG_FN.replace(
            RUST_UNIQUE_CLOSER,
            "    audit_trail.begin();\n" + RUST_UNIQUE_CLOSER,
        )
        assert "0 tokens" in result.stdout

    def test_literal_match_is_full_line_and_indent_insensitive(
        self, tmp_path: Path, backup_dir,
    ):
        """The literal matches the line's STRIPPED content — the caller
        need not reproduce the file's exact indentation."""
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = self._literal(
            target, "--insert-after", "let audited = audit(logged);",
            "    audit_trail.record(audited);\n",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text(encoding="utf-8")
        assert "    audit_trail.record(audited);\n" in content

    def test_zero_matches_refused_with_guidance(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = self._literal(
            target, "--insert-after", "let missing = absent();",
            "    x();\n",
        )
        assert result.returncode == 1
        assert "literal not found: let missing = absent();" in result.stderr
        assert "The edit was refused" in result.stderr
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_multiple_matches_refused_with_remedy(
        self, tmp_path: Path, backup_dir,
    ):
        """Two identical statement lines: ambiguous — refuse with the
        add-context / --lines remedy, never guess."""
        code = (
            "fn f() {\n"
            "    let a = 1;\n"
            "    let a = 1;\n"
            "}\n"
        )
        target = tmp_path / "dup.py"
        target.write_text(code, encoding="utf-8")
        result = self._literal(
            target, "--insert-after", "let a = 1;", "    b();\n",
        )
        assert result.returncode == 1
        assert "literal matches 2 lines" in result.stderr
        assert "include more context" in result.stderr
        assert "--lines" in result.stderr
        assert target.read_text(encoding="utf-8") == code

    def test_insert_after_conflicts_with_replace(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--snippet", "x\n",
            "--insert-after", "let audited = audit(logged);",
            "--replace", "process_item",
        )
        assert result.returncode == 1
        assert "--replace" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_insert_after_conflicts_with_after(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--snippet", "x\n",
            "--insert-after", "let audited = audit(logged);",
            "--after", "process_item",
        )
        assert result.returncode == 1
        assert "--after" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_insert_after_conflicts_with_after_imports(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--snippet", "x\n",
            "--insert-after", "let audited = audit(logged);",
            "--after-imports",
        )
        assert result.returncode == 1
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_insert_after_and_insert_before_are_mutually_exclusive(
        self, tmp_path: Path,
    ):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--snippet", "x\n",
            "--insert-after", "let audited = audit(logged);",
            "--insert-before", "let audited = audit(logged);",
        )
        assert result.returncode == 1
        assert "--insert" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_missing_snippet_with_insert_flag_refused(
        self, tmp_path: Path, backup_dir, monkeypatch, capsys,
    ):
        """The insertion flags are the mode; the snippet is the payload.
        With neither, the command cannot mean anything — refuse cleanly."""
        from fastedit import cli as cli_module
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        argv_backup = sys.argv
        sys.argv = [
            "fastedit", "edit", str(target),
            "--insert-after", "let audited = audit(logged);",
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_module.main()
        finally:
            sys.argv = argv_backup
        assert exc_info.value.code == 1
        assert "--snippet" in capsys.readouterr().err
        assert target.read_text(encoding="utf-8") == RUST_BIG_FN

    def test_snippet_dash_stdin_works_with_insert_after(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "app.py"
        target.write_text(RUST_BIG_FN, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--snippet", "-",
            "--insert-after", "let audited = audit(logged);",
            input_text="    audit_trail.record(audited);\n",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text(encoding="utf-8")
        assert content == RUST_BIG_FN.replace(
            RUST_UNIQUE_CLOSER,
            RUST_UNIQUE_CLOSER + "    audit_trail.record(audited);\n",
        )


class TestCLIEditLinesDelete:
    """Issue #14 refusal class 1: pure line-range deletion, no model.
    `--lines FROM:TO --delete` removes lines FROM..TO (1-indexed,
    inclusive); `--delete` is required so a malformed range can never
    silently become a deletion."""

    def test_lines_delete_removes_the_inclusive_range(
        self, tmp_path: Path, backup_dir,
    ):
        lines = [f"line {i}\n" for i in range(1, 11)]
        target = tmp_path / "doc.txt"
        target.write_text("".join(lines), encoding="utf-8")

        result = run_cli("edit", str(target), "--lines", "3:5", "--delete")

        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        expected = "".join(
            ln for i, ln in enumerate(lines, start=1) if i not in (3, 4, 5)
        )
        assert target.read_text(encoding="utf-8") == expected
        assert "Deleted lines 3-5" in result.stdout, result.stdout
        assert "0 tokens" in result.stdout

    def test_lines_delete_single_line_range(self, tmp_path: Path, backup_dir):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "2:2", "--delete")
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_text(encoding="utf-8") == "one\nthree\n"
        assert "Deleted lines 2-2" in result.stdout

    def test_lines_delete_creates_backup_for_undo(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "2:2", "--delete")
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        undo = run_cli("undo", str(target))
        assert undo.returncode == 0, undo.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_requires_delete_flag(self, tmp_path: Path):
        """--lines without --delete must not guess at a deletion."""
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "2:2")
        assert result.returncode == 1
        assert "--delete" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_requires_lines_flag(self, tmp_path: Path):
        """--delete without --lines is an incomplete mode — refuse."""
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--delete")
        assert result.returncode == 1
        assert "--lines" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_rejects_reversed_range(self, tmp_path: Path):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "5:3", "--delete")
        assert result.returncode == 1
        assert "invalid --lines range" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_rejects_out_of_bounds_range(self, tmp_path: Path):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "2:99", "--delete")
        assert result.returncode == 1
        assert "invalid --lines range" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_rejects_malformed_spec(self, tmp_path: Path):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        for bad in ("2", "2:3:4", "a:b", "2:"):
            result = run_cli("edit", str(target), "--lines", bad, "--delete")
            assert result.returncode == 1, bad
            assert "invalid --lines" in result.stderr, (bad, result.stderr)
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_conflicts_with_snippet(self, tmp_path: Path):
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--lines", "2:2", "--delete",
            "--snippet", "x\n",
        )
        assert result.returncode == 1
        assert "--snippet" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_empty_result_refused(self, tmp_path: Path):
        """Deleting every line of the file is not an edit, it's a remove —
        refuse (a future rm-style verb can own that)."""
        target = tmp_path / "doc.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = run_cli("edit", str(target), "--lines", "1:3", "--delete")
        assert result.returncode == 1
        assert "empty" in result.stderr, result.stderr
        assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"

    def test_lines_delete_preserves_crlf_endings(self, tmp_path: Path):
        """EOL/trailing-state preservation: a CRLF file keeps CRLF on every
        surviving line."""
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        result = run_cli("edit", str(target), "--lines", "2:2", "--delete")
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_bytes() == b"one\r\nthree\r\n"

    def test_lines_delete_no_trailing_newline_preserved(
        self, tmp_path: Path,
    ):
        """A file without a trailing terminator stays without one."""
        target = tmp_path / "tail.txt"
        target.write_bytes(b"one\ntwo\nthree")
        result = run_cli("edit", str(target), "--lines", "1:1", "--delete")
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_bytes() == b"two\nthree"


class TestCLIEditInsertEolPreservation:
    """EOL/trailing-state preservation for the insertion primitives."""

    def test_insert_after_preserves_crlf(self, tmp_path: Path):
        target = tmp_path / "crlf.py"
        target.write_bytes(b"def f():\r\n    a = 1\r\n    b = 2\r\n")
        result = run_cli(
            "edit", str(target),
            "--snippet", "    audit = true;\r\n",
            "--insert-after", "a = 1",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_bytes() == (
            b"def f():\r\n    a = 1\r\n    audit = true;\r\n    b = 2\r\n"
        )

    def test_insert_after_no_trailing_newline_state_kept(self, tmp_path: Path):
        target = tmp_path / "tail.py"
        target.write_bytes(b"def f():\n    a = 1\n    b = 2")
        result = run_cli(
            "edit", str(target),
            "--snippet", "    audit = true;\n",
            "--insert-after", "a = 1",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_bytes() == (
            b"def f():\n    a = 1\n    audit = true;\n    b = 2"
        )


class TestCLIEditLiteralHelp:
    def test_edit_help_lists_insert_and_lines_flags(self):
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        for flag in ("--insert-after", "--insert-before", "--lines", "--delete"):
            assert flag in result.stdout, flag
