"""Behavior tests for `fastedit init` — the one-shot agent-skill installer.

Contract (src/fastedit/cli.py::cmd_init):
  - Stages the PACKAGED skill — the wheel's fastedit/skill/SKILL.md, the
    byte-identical copy of skills/fastedit/SKILL.md (drift-guarded by
    tests/test_fastedit_skill.py) — into a FRESH temp dir laid out as
    <tmp>/skills/fastedit/SKILL.md, then runs the Vercel skills CLI via npx
    with exactly this argv:
      npx --yes skills add <staged>/skills/fastedit -g -a claude-code -y
    No GitHub shorthand and no --skill filter — the staged directory IS the
    skill. The shorthand/tree-URL forms would resolve the fork's default
    branch (main), which still carries the legacy claude-skill content, and
    the tree-URL form fails outright in non-TTY mode; staging the packaged
    copy removes the branch question entirely, so the skill an agent reads
    always matches the installed fastedit. The subprocess is bounded
    (timeout=300) and un-checked (check=False) so cmd_init owns the failure
    reporting.
  - --skill-agent <agent> (default: claude-code) targets another agent; the
    only argv change is the -a value.
  - Success: removes the staging dir, prints the skills-CLI output tail, then
      "Agent skill installed (global, Claude Code). Next: fastedit pull
       --model mlx-8bit (Apple Silicon) · fastedit doctor"
    and points at `fastedit mcp-install` for the optional MCP entry without
    running it.
  - npx missing: exit 1 with guidance to install Node.js and re-run init —
    an init that did nothing must say so.
  - skills-CLI failure (nonzero exit, timeout, OSError) or a staging
    failure: exit 1 with the tool's own output shown, plus a manual command
    whose staged path is LEFT IN PLACE so the command is directly runnable.
    Never a silent no-op.

subprocess.run and shutil.which are stubbed (monkeypatch) throughout — no
test touches npx, the network, or the user's global agent config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastedit import cli as cli_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_SKILL = PROJECT_ROOT / "skills" / "fastedit" / "SKILL.md"
STAGING_PREFIX = "fastedit-init-skill-"

SKILLS_STDOUT = (
    "sSkills: resolving <staged>/skills/fastedit\n"
    "  installed fastedit (skill) -> ~/.claude/skills\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_which(monkeypatch, result="/usr/local/bin/npx") -> list[str]:
    """Stub shutil.which; records every binary name looked up."""
    looked_up: list[str] = []

    def fake_which(name):
        looked_up.append(name)
        return result

    monkeypatch.setattr(shutil, "which", fake_which)
    return looked_up


def _stub_run(
    monkeypatch,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> list[tuple[list[str], dict]]:
    """Stub subprocess.run globally (same seam test_fast_edit_impact uses)."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _staged_dir(argv: list[str]) -> Path:
    """The staged skills/fastedit directory cmd_init passed to npx."""
    assert argv[0:4] == ["npx", "--yes", "skills", "add"]
    staged_dir = Path(argv[4])
    assert staged_dir.name == "fastedit"
    assert staged_dir.parent.name == "skills"
    return staged_dir


def _staging_root(argv: list[str]) -> Path:
    """The fresh temp dir the run staged its skill into."""
    return _staged_dir(argv).parent.parent


def _cleanup_staging(argv: list[str]) -> None:
    """Remove the staging dir a failed run deliberately left in place."""
    shutil.rmtree(_staging_root(argv), ignore_errors=True)


def _run_init_inprocess(monkeypatch, *extra_args: str) -> None:
    """Run `fastedit init ...` through cli.main() with the update check off."""
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "argv", ["fastedit", "init", *extra_args])
    cli_module.main()


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m fastedit <args>` (for --help rendering checks only)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


class _StopParsing(Exception):
    """Sentinel that stops cli.main() right after the parser is built."""


@pytest.fixture(scope="module")
def real_parser():
    """The exact parser object fastedit.cli.main() builds (test_cli_help trick)."""
    captured: dict = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _StopParsing

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(_StopParsing):
            cli_module.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------

class TestInitArgparse:
    def test_init_subcommand_registered(self, real_parser):
        for action in real_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                assert "init" in action.choices
                break
        else:
            raise AssertionError("main parser has no subparsers action")

    def test_init_defaults_to_claude_code(self, real_parser):
        args = real_parser.parse_args(["init"])
        assert args.skill_agent == "claude-code"

    def test_init_help_names_the_skill_agent_flag(self):
        result = run_cli("init", "--help")
        assert result.returncode == 0
        assert "--skill-agent" in result.stdout
        assert "claude-code" in result.stdout

    def test_init_in_top_level_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "init" in result.stdout


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestInitInstallsTheSkill:
    def test_runs_the_skills_cli_against_a_staged_local_tree(self, monkeypatch, capsys):
        looked_up = _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch)

        assert "npx" in looked_up
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv[0:4] == ["npx", "--yes", "skills", "add"]
        staged_dir = _staged_dir(argv)
        # A FRESH temp dir under the system temp root, laid out skills/fastedit.
        staging_root = staged_dir.parent.parent
        assert staging_root.name.startswith(STAGING_PREFIX)
        assert staging_root.parent == Path(tempfile.gettempdir())
        # NO --skill filter and no GitHub source: the staged path IS the skill.
        assert "--skill" not in argv
        assert not any("github.com" in part or "/" == part for part in argv)
        assert not any(part.startswith(("http://", "https://")) for part in argv)
        assert argv[5:] == ["-g", "-a", "claude-code", "-y"]
        # Bounded, un-checked: cmd_init owns the failure reporting.
        assert kwargs["timeout"] == 300
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        # Success cleaned the staging dir up.
        assert not staging_root.exists()

    def test_staged_skill_is_the_packaged_resource(self, monkeypatch, capsys):
        """The staged SKILL.md is the packaged copy — byte-identical to the
        repo skill (the single source of truth)."""
        _stub_which(monkeypatch)
        staged_bytes: dict[str, bytes] = {}
        calls: list[tuple[list[str], dict]] = []

        def fake_run(argv, **kwargs):
            staged = _staged_dir(argv) / "SKILL.md"
            staged_bytes["data"] = staged.read_bytes()
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0, stdout=SKILLS_STDOUT, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        _run_init_inprocess(monkeypatch)

        assert staged_bytes["data"] == REPO_SKILL.read_bytes()
        assert len(calls) == 1
        # Success cleaned the staging dir up.
        assert not _staging_root(calls[0][0]).exists()

    def test_success_prints_output_tail_and_next_steps(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch)  # no SystemExit -> exit 0

        out = capsys.readouterr().out
        assert "installed fastedit (skill)" in out, out  # skills-CLI output tail
        assert "Agent skill installed (global, Claude Code)" in out, out
        assert "Next: fastedit pull --model mlx-8bit (Apple Silicon)" in out, out
        assert "fastedit doctor" in out, out
        # MCP is pointed at in the guidance text, never run by init.
        assert "fastedit mcp-install" in out, out
        assert len(calls) == 1

    def test_skill_agent_flag_only_changes_the_agent_value(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch, "--skill-agent", "codex")

        out = capsys.readouterr().out
        argv, _kwargs = calls[0]
        assert argv[0:4] == ["npx", "--yes", "skills", "add"]
        assert "--skill" not in argv
        assert argv[5:] == ["-g", "-a", "codex", "-y"]
        assert "Agent skill installed (global, codex)" in out, out


# ---------------------------------------------------------------------------
# Failure paths — all fail loud with exit 1
# ---------------------------------------------------------------------------

class TestInitFailurePaths:
    def test_npx_missing_exits_1_with_guidance_and_stages_nothing(self, monkeypatch, capsys):
        _stub_which(monkeypatch, result=None)
        calls = _stub_run(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "npx" in err
        assert "NOT installed" in err
        assert "Install Node.js" in err
        assert "fastedit init" in err  # re-run guidance
        assert calls == []  # nothing was executed

    def test_skills_cli_failure_exits_1_shows_output_and_keeps_the_staged_dir(
        self, monkeypatch, capsys
    ):
        _stub_which(monkeypatch)
        calls = _stub_run(
            monkeypatch,
            returncode=1,
            stdout="npm ERR! missing peer dependency",
            stderr="npm ERR! code ELIFECYCLE",
        )

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "npm ERR! missing peer dependency" in err, err  # output shown
        assert "npm ERR! code ELIFECYCLE" in err, err
        assert "NOT installed" in err
        # The manual command is the exact invocation, staged path included.
        argv, _kwargs = calls[0]
        assert " ".join(argv) in err
        # The staged dir is LEFT IN PLACE so the manual command is runnable.
        assert (_staged_dir(argv) / "SKILL.md").is_file()
        _cleanup_staging(argv)

    def test_skills_cli_timeout_exits_1_with_guidance(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "timed out" in err
        assert "NOT installed" in err
        assert " ".join(seen["argv"]) in err
        # The staged dir is LEFT IN PLACE so the manual command is runnable.
        assert (_staged_dir(seen["argv"]) / "SKILL.md").is_file()
        _cleanup_staging(seen["argv"])

    def test_npx_spawn_failure_exits_1_with_guidance(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            raise OSError("exec format error")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "could not run npx" in err
        assert " ".join(seen["argv"]) in err
        _cleanup_staging(seen["argv"])

    def test_staging_failure_exits_1_without_running_npx(self, monkeypatch, capsys):
        """A missing/broken packaged skill fails loud instead of a traceback."""
        _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch)

        def broken_packaged_bytes():
            raise FileNotFoundError("skill/SKILL.md missing from this install")

        monkeypatch.setattr(cli_module, "_packaged_skill_bytes", broken_packaged_bytes)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "stage" in err
        assert "NOT installed" in err
        assert calls == []  # npx was never invoked
