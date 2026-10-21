"""End-to-end CLI tests for `fastedit create` and `fastedit duplicate`.

Runs the real CLI via subprocess (same convention as test_cli.py's
run_cli), never calling cmd_create/cmd_duplicate directly -- these two
subcommands were built but "never exercised end to end" per the phase-1
task, so the whole point is to actually run the binary.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


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
        timeout=30,
        env=env,
        check=False,
    )

def _make_png_bytes() -> bytes:
    """Build a real, minimal, valid PNG file in memory (8x8 IHDR + empty IDAT)."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", (8).to_bytes(4, "big") + (8).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0]))
    idat = chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 24))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


class TestCLICreate:
    def test_create_writes_content_from_flag(self, tmp_path: Path) -> None:
        """`create --content` writes exactly that content to a new file."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"
        assert "Created" in result.stdout

    def test_create_refuses_existing_file_without_force(self, tmp_path: Path) -> None:
        """An existing file is refused with a clear message and exit code 2, no --force."""
        new_file = tmp_path / "hello.py"
        new_file.write_text("old\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content", "new\n")
        assert result.returncode == 2
        assert "already exists" in result.stderr
        assert new_file.read_text(encoding="utf-8") == "old\n"

    def test_create_overwrites_existing_file_with_force(self, tmp_path: Path) -> None:
        """--force lets create overwrite an existing file."""
        new_file = tmp_path / "hello.py"
        new_file.write_text("old\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content", "new\n", "--force")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "new\n"

    def test_create_refuses_missing_parent_without_parents(self, tmp_path: Path) -> None:
        """A missing parent directory is refused with exit code 1, no --parents."""
        new_file = tmp_path / "sub" / "dir" / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n")
        assert result.returncode == 1
        assert "parent directory not found" in result.stderr
        assert not new_file.exists()

    def test_create_makes_missing_parents_with_parents(self, tmp_path: Path) -> None:
        """--parents creates the missing directory chain before writing."""
        new_file = tmp_path / "sub" / "dir" / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n", "--parents")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_refuses_content_and_content_file_together(self, tmp_path: Path) -> None:
        """--content and --content-file are mutually exclusive."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli(
            "create", str(new_file), "--content", "y = 2\n", "--content-file", str(content_file),
        )
        assert result.returncode == 1
        assert "mutually exclusive" in result.stderr
        assert not new_file.exists()

    def test_create_flag_error_precedes_existence_check(self, tmp_path: Path) -> None:
        """The --content/--content-file conflict is reported even when the
        target file also exists: an argv-only error must not be masked by a
        state error the user could 'fix' while the real problem stays."""
        new_file = tmp_path / "hello.py"
        new_file.write_text("old\n", encoding="utf-8")
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli(
            "create", str(new_file), "--content", "y = 2\n", "--content-file", str(content_file),
        )
        assert result.returncode == 1
        assert "mutually exclusive" in result.stderr
        assert "already exists" not in result.stderr
        assert new_file.read_text(encoding="utf-8") == "old\n"

    def test_create_reads_content_from_file(self, tmp_path: Path) -> None:
        """--content-file <path> reads the file's bytes as the new content."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content-file", str(content_file))
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_preserves_crlf_content_from_file(self, tmp_path: Path) -> None:
        """--content-file with CRLF line endings is written back byte-for-byte, not normalized to LF."""
        new_file = tmp_path / "hello.txt"
        content_file = tmp_path / "content.txt"
        original = b"line1\r\nline2\r\nline3\r\n"
        content_file.write_bytes(original)

        result = run_cli("create", str(new_file), "--content-file", str(content_file))

        assert result.returncode == 0
        assert new_file.read_bytes() == original

    def test_create_reads_content_file_dash_from_stdin(self, tmp_path: Path) -> None:
        """--content-file - reads content from stdin."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content-file", "-", input_text="x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_reads_bare_stdin_when_no_content_flag_given(self, tmp_path: Path) -> None:
        """With neither --content nor --content-file, create reads stdin directly."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), input_text="x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_refuses_binary_content_file(self, tmp_path: Path) -> None:
        """A NUL-bearing --content-file is refused as binary, with the reason in the message."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "payload.dat"
        content_file.write_bytes(b"abc\x00def")
        result = run_cli("create", str(new_file), "--content-file", str(content_file))
        assert result.returncode == 1
        assert "binary" in result.stderr
        assert "NUL" in result.stderr
        assert not new_file.exists()

    # ------------------------------------------------------------------
    # Issue #10: `--content -` must read PIPED stdin, not write a
    # literal '-' (edit documents the '-' convention; create honors it).
    # ------------------------------------------------------------------

    def test_create_reads_content_dash_from_stdin(self, tmp_path: Path) -> None:
        """Issue #10: --content - reads the piped content; the literal '-'
        must never be written as file content."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", "-", input_text="x = 1\n")
        assert result.returncode == 0, result.stderr
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"
        assert new_file.read_text(encoding="utf-8") != "-\n"

    @pytest.mark.skipif(os.name != "posix", reason="pty is POSIX-only")
    def test_create_content_dash_with_tty_stdin_errors_cleanly(self, tmp_path: Path) -> None:
        """Issue #9 semantics shared by create: `--content -` on a TTY must
        exit 1 fast with a clean error instead of blocking on the terminal,
        and must not create the file."""
        import pty

        master, slave = pty.openpty()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        new_file = tmp_path / "hello.py"
        try:
            proc = subprocess.Popen(
                [*CLI_MODULE, "create", str(new_file), "--content", "-"],
                stdin=slave,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        finally:
            os.close(slave)
        try:
            _out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:  # pragma: no cover - only on a regression
                proc.kill()
                proc.communicate()
            os.close(master)

        assert proc.returncode == 1
        assert b"requires piped stdin" in err
        assert not new_file.exists()

    def test_create_reports_symbols_for_a_parseable_language(self, tmp_path: Path) -> None:
        """A supported language's new file gets a symbol report, not just a line count."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", "def greet():\n    return 'hi'\n")
        assert result.returncode == 0
        assert "greet" in result.stdout

    def test_create_notes_unsupported_language(self, tmp_path: Path) -> None:
        """An extension fastedit/tldr doesn't parse still creates the file, with a note."""
        new_file = tmp_path / "notes.unsupportedext"
        result = run_cli("create", str(new_file), "--content", "hello\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "hello\n"
        assert "no fastedit/tldr language support" in result.stdout


class TestCLICreateAtFileContent:
    """Issue #7 applied to --content: explicit '@path' file syntax, an
    existing-file auto-detection with a stderr note, and
    --content-is-literal to opt out and use the text verbatim."""

    def test_create_content_atfile_reads_the_file(self, tmp_path: Path) -> None:
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content", f"@{content_file}")
        assert result.returncode == 0, result.stderr
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"
        # The explicit @form is not magic: no resolution note on stderr.
        assert "resolved to an existing file" not in result.stderr

    def test_create_content_atfile_missing_file_errors_cleanly(self, tmp_path: Path) -> None:
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", f"@{tmp_path / 'nope.txt'}")
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "not found" in result.stderr
        assert not new_file.exists()

    def test_create_content_existing_path_autodetected_with_note(self, tmp_path: Path) -> None:
        """A single-line --content naming an existing file is read from it,
        with the stderr note pointing at --content-is-literal."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")

        result = run_cli("create", str(new_file), "--content", str(content_file))

        assert result.returncode == 0, result.stderr
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"
        assert "note: --content resolved to an existing file" in result.stderr
        assert "--content-is-literal" in result.stderr

    def test_create_content_is_literal_keeps_path_string_verbatim(self, tmp_path: Path) -> None:
        """file exists AND --content-is-literal → the PATH STRING is the
        content, verbatim; the file's own content must NOT be read."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")

        result = run_cli(
            "create", str(new_file),
            "--content", str(content_file),
            "--content-is-literal",
        )

        assert result.returncode == 0, result.stderr
        assert new_file.read_text(encoding="utf-8") == str(content_file)
        assert "resolved to an existing file" not in result.stderr

    def test_create_content_multiline_text_is_never_detected_as_a_path(
        self, tmp_path: Path,
    ) -> None:
        """Multi-line content can never name a file: it stays verbatim even
        when a sibling file happens to share the first line's text."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "x = 1"
        content_file.write_text("decoy\n", encoding="utf-8")

        result = run_cli("create", str(new_file), "--content", "x = 1\ny = 2\n")

        assert result.returncode == 0, result.stderr
        assert new_file.read_text(encoding="utf-8") == "x = 1\ny = 2\n"
        assert "resolved" not in result.stderr


class TestCLIDuplicate:
    def test_duplicate_copies_content_exactly(self, tmp_path: Path) -> None:
        """duplicate writes the source's exact bytes to the destination."""
        source = tmp_path / "a.py"
        source.write_text("def f():\n    return 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
        assert "Duplicated" in result.stdout

    def test_duplicate_preserves_crlf_content(self, tmp_path: Path) -> None:
        """duplicate copies CRLF line endings byte-for-byte, not normalized to LF."""
        source = tmp_path / "a.txt"
        original = b"line1\r\nline2\r\nline3\r\n"
        source.write_bytes(original)
        dest = tmp_path / "b.txt"

        result = run_cli("duplicate", str(source), str(dest))

        assert result.returncode == 0
        assert dest.read_bytes() == original

    def test_duplicate_refuses_missing_source(self, tmp_path: Path) -> None:
        """A source that doesn't exist is refused with a clear message."""
        source = tmp_path / "missing.py"
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "source file not found" in result.stderr
        assert not dest.exists()

    def test_duplicate_refuses_directory_source_cleanly(self, tmp_path: Path) -> None:
        """A directory source is a clean exit-1 refusal, not an
        IsADirectoryError traceback from read_bytes()."""
        source = tmp_path / "srcdir"
        source.mkdir()
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "not a regular file" in result.stderr
        assert not dest.exists()

    def test_duplicate_copies_binary_source_bytes_exactly(self, tmp_path: Path) -> None:
        """A binary source (a real PNG) duplicates byte-for-byte -- no refusal, no decode."""
        png_bytes = _make_png_bytes()
        source = tmp_path / "logo.png"
        source.write_bytes(png_bytes)
        dest = tmp_path / "logo-old.png"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 0
        assert dest.read_bytes() == png_bytes
        assert "Duplicated" in result.stdout

    def test_duplicate_refuses_existing_destination_without_force(self, tmp_path: Path) -> None:
        """An existing destination is refused with exit code 2, no --force."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        dest.write_text("old\n", encoding="utf-8")
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 2
        assert "already exists" in result.stderr
        assert dest.read_text(encoding="utf-8") == "old\n"

    def test_duplicate_overwrites_destination_with_force(self, tmp_path: Path) -> None:
        """--force lets duplicate overwrite an existing destination."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        dest.write_text("old\n", encoding="utf-8")
        result = run_cli("duplicate", str(source), str(dest), "--force")
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == "x = 1\n"

    def test_duplicate_refuses_missing_destination_parent_without_parents(self, tmp_path: Path) -> None:
        """A missing destination directory is refused with exit code 1, no --parents."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "sub" / "dir" / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "parent directory not found" in result.stderr
        assert not dest.exists()

    def test_duplicate_makes_missing_destination_parents_with_parents(self, tmp_path: Path) -> None:
        """--parents creates the missing destination directory chain."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "sub" / "dir" / "b.py"
        result = run_cli("duplicate", str(source), str(dest), "--parents")
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == "x = 1\n"

    def test_duplicate_reports_symbols_for_a_parseable_language(self, tmp_path: Path) -> None:
        """The duplicated file gets its own symbol report (path-based, not the source's)."""
        source = tmp_path / "a.py"
        source.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 0
        assert "greet" in result.stdout
