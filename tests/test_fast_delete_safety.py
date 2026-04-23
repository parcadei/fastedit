"""Tests for cross-file caller-safety check on fast_delete.

Milestone 2 (VAL-M2-001..003): fast_delete refuses to delete a symbol when
other files in the project still reference it, unless force=True.

These tests exercise the safety helper `check_cross_file_callers` that the
MCP tool and CLI both delegate to, plus the CLI end-to-end flow (since the
CLI is a thin wrapper we can drive synchronously in tests).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastedit.inference.caller_safety import (
    _find_project_root,
    check_cross_file_callers,
)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """_find_project_root walks up looking for .git / pyproject.toml /
    package.json markers, falling back to the file's parent if nothing is
    found."""

    def test_find_project_root_with_git_marker(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        file_path = nested / "x.py"
        file_path.write_text("pass\n")

        root = _find_project_root(file_path)
        assert root == tmp_path

    def test_find_project_root_with_pyproject_marker(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        nested = tmp_path / "pkg"
        nested.mkdir()
        file_path = nested / "x.py"
        file_path.write_text("pass\n")

        root = _find_project_root(file_path)
        assert root == tmp_path

    def test_find_project_root_fallback_when_no_marker(self, tmp_path: Path):
        # No markers anywhere — should fall back near the file.
        file_path = tmp_path / "lone.py"
        file_path.write_text("pass\n")

        root = _find_project_root(file_path)
        # Fallback contract is "reasonable root near the file"; at minimum
        # the file must be reachable from the returned directory.
        try:
            rel = file_path.resolve().relative_to(root.resolve())
        except ValueError:
            rel = None
        assert rel is not None


# ---------------------------------------------------------------------------
# Safety helper: check_cross_file_callers
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    """Create a mini project root with a .git marker so _find_project_root
    picks tmp_path and tldr scopes its search there."""
    (tmp_path / ".git").mkdir()
    return tmp_path


class TestCheckCrossFileCallers:
    """VAL-M2-001 / VAL-M2-002: helper returns the list of references in
    OTHER files. Self-references (inside the file being edited) and the
    definition itself are filtered out."""

    def test_reports_callers_in_other_files(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "a.py").write_text(textwrap.dedent("""\
        def foo():
            return 1
        """))
        (root / "b.py").write_text(textwrap.dedent("""\
        from a import foo

        def use():
            return foo()
        """))

        refs = check_cross_file_callers(
            file_path=root / "a.py",
            symbol="foo",
            project_root=root,
        )

        # At least one hit in b.py (the import or the call), none in a.py.
        assert len(refs) >= 1
        for r in refs:
            assert Path(r["file"]).resolve() != (root / "a.py").resolve()
        # b.py should appear among the caller files.
        caller_files = {Path(r["file"]).name for r in refs}
        assert "b.py" in caller_files

    def test_ignores_self_references(self, tmp_path: Path):
        """Self-references (calls to foo within the file defining foo)
        must be filtered out — deleting foo removes them anyway."""
        root = _make_project(tmp_path)
        (root / "a.py").write_text(textwrap.dedent("""\
        def foo():
            return 1

        def helper():
            # foo is called inside a.py but nowhere else.
            return foo() + 1
        """))

        refs = check_cross_file_callers(
            file_path=root / "a.py",
            symbol="foo",
            project_root=root,
        )

        assert refs == []

    def test_no_callers_returns_empty(self, tmp_path: Path):
        root = _make_project(tmp_path)
        (root / "a.py").write_text(textwrap.dedent("""\
        def orphan():
            return 1
        """))

        refs = check_cross_file_callers(
            file_path=root / "a.py",
            symbol="orphan",
            project_root=root,
        )

        assert refs == []

    def test_tldr_unavailable_falls_open(self, tmp_path: Path, monkeypatch):
        """VAL-M2-002 fall-open: if tldr can't run, we return [] rather
        than failing closed. The CLI/MCP layer surfaces a note separately."""
        root = _make_project(tmp_path)
        (root / "a.py").write_text("def foo(): return 1\n")

        # Force the subprocess call to blow up.
        import fastedit.inference.caller_safety as cs_mod

        def boom(*a, **kw):
            raise FileNotFoundError("tldr missing")

        monkeypatch.setattr(cs_mod.subprocess, "run", boom)

        refs = check_cross_file_callers(
            file_path=root / "a.py",
            symbol="foo",
            project_root=root,
        )
        assert refs == []


# ---------------------------------------------------------------------------
# End-to-end: `fastedit delete` CLI with and without --force
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Invoke the fastedit CLI via `python -m fastedit`."""
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


class TestFastDeleteCliSafety:
    """Exercise the CLI path so we lock down CLI + refuse message shape
    in the same suite. These tests cover VAL-M2-001 and VAL-M2-003."""

    def test_fast_delete_refuses_when_callers_exist(self, tmp_path: Path):
        """VAL-M2-001: def foo in a.py, foo() in b.py — delete should refuse
        without --force. With --force it deletes."""
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def foo():
            return 1
        """))
        b = root / "b.py"
        b.write_text(textwrap.dedent("""\
        from a import foo

        def use():
            return foo()
        """))

        # Without --force: refusal. a.py untouched.
        proc = _run_cli(["delete", str(a), "foo"], cwd=root)
        assert proc.returncode != 0, proc.stdout + proc.stderr
        combined = (proc.stdout + proc.stderr).lower()
        assert "refused" in combined or "reference" in combined
        assert "def foo" in a.read_text()  # still there

        # With --force: proceeds.
        proc2 = _run_cli(["delete", str(a), "foo", "--force"], cwd=root)
        assert proc2.returncode == 0, proc2.stdout + proc2.stderr
        assert "def foo" not in a.read_text()

    def test_fast_delete_allows_when_no_callers(self, tmp_path: Path):
        """VAL-M2-001 common case: no cross-file callers — proceeds as today."""
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def orphan():
            return 1

        def keep():
            return 2
        """))

        proc = _run_cli(["delete", str(a), "orphan"], cwd=root)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        content = a.read_text()
        assert "def orphan" not in content
        assert "def keep" in content

    def test_fast_delete_ignores_self_references(self, tmp_path: Path):
        """VAL-M2-001: symbol referenced only within its own file — delete
        proceeds without --force because self-refs are filtered."""
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def foo():
            return 1

        def helper():
            return foo() + 1
        """))

        proc = _run_cli(["delete", str(a), "foo"], cwd=root)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "def foo" not in a.read_text()
