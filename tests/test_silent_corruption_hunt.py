"""Silent-corruption hunt: byte-level regression tests for fastedit verbs.

Every assertion compares raw BYTES (never decoded strings) so that
line-ending / blank-line / trailing-newline defects cannot hide behind
str equality. See reports/corruption-hunt/ for the investigation writeup.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FASTEDIT = [sys.executable, "-m", "fastedit"]
FIXTURE_NAME = "mod." + "py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        FASTEDIT + args, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
    )


PEP8_SRC = (
    b"import os\n\n\n"
    b"def alpha():\n    return 1\n\n\n"
    b"def beta():\n    return 2\n\n\n"
    b"def gamma():\n    return 3\n"
)


def test_delete_preserves_two_blank_line_separator_style(tmp_path: Path):
    """fastedit delete must not leave a stray extra blank line when the
    file uses PEP 8's 2-blank-line separator between top-level defs."""
    target = tmp_path / FIXTURE_NAME
    target.write_bytes(PEP8_SRC)

    result = _run(["delete", str(target), "beta"], tmp_path)
    assert result.returncode == 0, result.stderr

    expected = (
        b"import os\n\n\n"
        b"def alpha():\n    return 1\n\n\n"
        b"def gamma():\n    return 3\n"
    )
    assert target.read_bytes() == expected

def test_delete_last_symbol_leaves_the_preceding_bytes_untouched(tmp_path: Path):
    """Deleting the LAST symbol must not rewrite the bytes that precede it.
    ADJUDICATED 2026-09-13, reversing an earlier draft fix. That draft trimmed
    the now-dangling separator blank lines when the deleted symbol ran to EOF,
    arguing they had nothing left to separate. It was DROPPED, because it popped
    entries off a list that includes original_lines[:start_idx] -- the PRESERVED
    PREFIX, i.e. bytes before the deleted span. Measured consequence: six
    byte-exactness tests in the encoding-matrix and CRLF-preservation suites went
    red, because on a CRLF file those trailing blanks are carriage-return entries
    and the preserved prefix stopped matching the original bytes.
    The rule this encodes: delete may consume the separator that FOLLOWS a
    removed symbol (that whitespace is part of the deleted region), but may never
    touch bytes BEFORE it. A file left ending in blank lines is untidy, not
    corrupt; a tool that silently rewrites bytes you did not ask it to change is
    the defect this whole suite exists to catch.
    """
    target = tmp_path / FIXTURE_NAME
    original = b"def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    target.write_bytes(original)

    result = _run(["delete", str(target), "beta"], tmp_path)
    assert result.returncode == 0, result.stderr

    # Byte-for-byte the original prefix, dangling blank lines included.
    assert target.read_bytes() == original[: original.index(b"def beta()")]
