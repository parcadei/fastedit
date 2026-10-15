"""Accuracy and coverage tests for the `fastedit --help` guide (fastedit.cli_help).

Contract:
  - EPILOG is rendered from structured data (SECTIONS / COMMANDS) — one source of truth.
  - Every example command in the data must be accepted by the REAL parser, built
    exactly the way fastedit.cli.main builds it. This is the no-invented-flags
    guarantee: argparse rejects unknown flags, bad choices, missing required
    arguments and unknown subcommands. Examples are parse-only — parse_args
    never touches the filesystem, and nothing is executed.
  - The data covers exactly the parser's subcommand set (no invented commands,
    none missing) and every command name appears in the rendered EPILOG.
  - Troubleshooting entries quote error substrings that really exist in src/.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fastedit.cli_help import COMMANDS, EPILOG, SECTION_HEADERS, SECTIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _StopParsing(Exception):
    """Sentinel that stops cli.main() right after the parser is built."""


@pytest.fixture(scope="module")
def real_parser():
    """The exact parser object fastedit.cli.main() builds.

    cli.main() constructs the parser and immediately calls parse_args(); a
    captor replaces argparse.ArgumentParser.parse_args, records `self`, and
    stops main() before any dispatch. The real method is restored afterwards,
    so the examples below go through genuine, unmodified argparse parsing.
    """
    from fastedit import cli

    captured: dict = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _StopParsing

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(_StopParsing):
            cli.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


def _subcommand_names(parser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("main parser has no subparsers action")


# Path-shaped example tokens are swapped for tmp dummies before parse_args.
# Parse-only either way: nothing is ever executed or written.
_PATH_TOKEN = re.compile(
    r"^(?:src/|tests/|docs/|data/|scripts/|parts/)|\.(?:py|md|json|csv|tsv|toml|txt|rs)$"
)


def _to_tmp_dummy(token: str, tmp_path: Path) -> str:
    if token.startswith("-") or not _PATH_TOKEN.search(token):
        return token
    return str(tmp_path / Path(token).name)


def test_example_commands_match_the_real_parser(real_parser, tmp_path):
    """Every example in the data must parse against the REAL parser."""
    failures = []
    for command in COMMANDS:
        for example in command.examples:
            argv = [_to_tmp_dummy(token, tmp_path) for token in example.argv]
            try:
                real_parser.parse_args(argv)
            except SystemExit as exc:
                failures.append(
                    f"{command.name}: {example.display!r} rejected (exit {exc.code})"
                )
    assert not failures, "examples the real parser rejects:\n" + "\n".join(failures)


def test_command_data_covers_exactly_the_real_subcommands(real_parser):
    """The data covers every real command, invents none, and renders all names."""
    real = _subcommand_names(real_parser)
    data = {command.name for command in COMMANDS}
    assert data == real, f"missing={sorted(real - data)} invented={sorted(data - real)}"
    missing = [name for name in sorted(data) if name not in EPILOG]
    assert not missing, f"command names absent from EPILOG: {missing}"


def test_required_section_headers_present():
    """All nine required sections exist, in order, and are rendered."""
    required = [
        "QUICKSTART",
        "THE THREE EDIT MODES",
        "COMMANDS",
        "PARAMETERS",
        "LIMITS",
        "LIMITATIONS",
        "BEST PRACTICES",
        "TROUBLESHOOTING",
        "INSTALL & UPDATE",
    ]
    assert [s.title for s in SECTIONS] == list(SECTION_HEADERS)
    for header in required:
        assert any(header == s.title for s in SECTIONS), f"missing section: {header}"
        assert header in EPILOG


def test_troubleshooting_quotes_real_error_strings():
    """Every error needle must exist in the real source AND in the rendered guide."""
    needles = [
        ("src/fastedit/file_lock.py", "another fastedit instance"),
        ("src/fastedit/mcp/backup.py", "file changed on disk since it was read"),
        ("src/fastedit/cli.py", "parse errors; refusing to write"),
        ("src/fastedit/cli.py", "Symbol '"),
        ("src/fastedit/cli.py", "contains a keep-marker but the edit"),
        ("src/fastedit/cli.py", "has no definition"),
        ("src/fastedit/model_download.py", "Model not found locally"),
        ("src/fastedit/inference/markers.py", "# ... existing code ..."),
    ]
    problems = []
    for rel_path, needle in needles:
        source = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        if needle not in source:
            problems.append(f"{needle!r} no longer in {rel_path} (source drifted)")
        if needle not in EPILOG:
            problems.append(f"{needle!r} missing from EPILOG")
    assert not problems, "\n".join(problems)


def test_epilog_lines_stay_under_100_chars():
    long_lines = [
        (number, line)
        for number, line in enumerate(EPILOG.splitlines(), 1)
        if len(line) > 100
    ]
    assert not long_lines, f"lines over 100 chars: {long_lines[:5]}"


def test_main_parser_uses_raw_formatter_and_data_driven_epilog(real_parser):
    assert real_parser.formatter_class is argparse.RawDescriptionHelpFormatter
    assert real_parser.epilog == EPILOG


def test_fastedit_help_subprocess_shows_the_guide():
    """`fastedit --help` exits 0 and shows the guide (and the legacy epilog contract)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "fastedit", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "QUICKSTART" in result.stdout
    assert "INSTALL & UPDATE" in result.stdout
    # Legacy epilog contract (TestArgparseRegistration): installer, doctor, MCP.
    assert "scripts/install-dev.sh" in result.stdout
    assert "fastedit doctor" in result.stdout
    assert "fastedit mcp-install" in result.stdout
