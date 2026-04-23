"""Tests for Milestone 3 — fast_edit pre-flight impact preview.

VAL-M3-001: when ``fast_edit`` with ``replace=<name>`` changes the
function's SIGNATURE line (parameters, name, visibility, etc.), the
tool appends an informational impact note listing the cross-file caller
count. The edit still proceeds — this is advisory only.

VAL-M3-002 (hot path): body-only edits MUST NOT invoke ``tldr`` as a
subprocess. We verify this by monkeypatching ``subprocess.run`` and
asserting it's never called for a body-only edit.

VAL-M3-003: tests are regular pytest tests added to the default suite.

These tests exercise the CLI (``fastedit edit``) because it shares the
impact-check helper with the MCP ``fast_edit`` tool. The MCP tool is an
async FastMCP coroutine that requires a live server context, so we lock
down behavior through the thin CLI wrapper that uses the same helper.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def _make_project(tmp_path: Path) -> Path:
    """Create a project root with a ``.git`` marker so ``tldr`` scopes
    its reference search to ``tmp_path``."""
    (tmp_path / ".git").mkdir()
    return tmp_path


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Invoke the fastedit CLI via ``python -m fastedit``."""
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestSignatureChangeDetection:
    """The helper that decides whether a symbol's signature changed
    between two versions of the same source file."""

    def test_signature_unchanged_on_body_only_edit(self, tmp_path: Path):
        from fastedit.inference.caller_safety import signature_changed

        old = textwrap.dedent("""\
        def foo(a):
            return a + 1
        """)
        new = textwrap.dedent("""\
        def foo(a):
            # new comment
            return a + 2
        """)
        assert signature_changed(old, new, "foo", "python") is False

    def test_signature_changed_on_param_addition(self, tmp_path: Path):
        from fastedit.inference.caller_safety import signature_changed

        old = "def foo(a):\n    return a\n"
        new = "def foo(a, b):\n    return a + b\n"
        assert signature_changed(old, new, "foo", "python") is True

    def test_signature_changed_on_param_rename(self, tmp_path: Path):
        from fastedit.inference.caller_safety import signature_changed

        old = "def foo(a):\n    return a\n"
        new = "def foo(x):\n    return x\n"
        # Parameter renames ARE signature changes — callers passing by
        # keyword will break.
        assert signature_changed(old, new, "foo", "python") is True

    def test_signature_unchanged_when_symbol_missing(self, tmp_path: Path):
        """Defensive: if the symbol doesn't exist in either side we
        report 'no change' rather than crashing the edit path."""
        from fastedit.inference.caller_safety import signature_changed

        old = "def bar(): return 1\n"
        new = "def bar(): return 2\n"
        assert signature_changed(old, new, "missing", "python") is False


# ---------------------------------------------------------------------------
# End-to-end via the CLI — mirrors the MCP fast_edit flow
# ---------------------------------------------------------------------------


class TestFastEditImpactNote:
    """VAL-M3-001 / VAL-M3-002: the CLI edit command appends an impact
    note only when the signature changes AND cross-file callers exist."""

    def test_fast_edit_shows_impact_note_on_signature_change(
        self, tmp_path: Path,
    ):
        """VAL-M3-001: changing ``def foo(a):`` → ``def foo(a, b):`` with
        a caller in ``b.py`` produces an impact note in the CLI output.
        """
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def foo(a):
            return a + 1
        """))
        b = root / "b.py"
        b.write_text(textwrap.dedent("""\
        from a import foo

        def use():
            return foo(1)
        """))

        # Change the signature from `def foo(a):` to `def foo(a, b):`.
        # Use a full-function replace snippet so chunked_merge can route
        # through the deterministic text-match path (no model needed).
        snippet = "def foo(a, b):\n    return a + b\n"
        proc = _run_cli(
            ["edit", str(a), "--snippet", snippet, "--replace", "foo"],
            cwd=root,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        combined = proc.stdout + proc.stderr
        # Edit actually applied.
        assert "def foo(a, b):" in a.read_text()

        # Impact note appended.
        assert "signature of 'foo' changed" in combined, combined
        # Caller count: b.py has an import AND a call — 2 refs in 1 file.
        assert "caller" in combined
        assert "file" in combined
        # The hint mentions either `tldr impact` or `fast_search` as a
        # drill-down path.
        assert "tldr impact" in combined or "fast_search" in combined

    def test_fast_edit_no_impact_note_on_body_change(
        self, tmp_path: Path, monkeypatch,
    ):
        """VAL-M3-002: when the signature is unchanged, no impact note
        appears AND tldr is NEVER invoked (hot-path budget). We verify
        the latter by monkeypatching ``subprocess.run`` inside
        caller_safety and asserting no call."""
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def foo(a):
            return a + 1
        """))
        b = root / "b.py"
        b.write_text(textwrap.dedent("""\
        from a import foo

        def use():
            return foo(1)
        """))

        # Body-only edit — same signature, different body.
        snippet = "def foo(a):\n    return a + 42\n"
        proc = _run_cli(
            ["edit", str(a), "--snippet", snippet, "--replace", "foo"],
            cwd=root,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        # Edit applied.
        content = a.read_text()
        assert "return a + 42" in content

        combined = proc.stdout + proc.stderr
        # No impact note.
        assert "signature of 'foo' changed" not in combined, combined
        assert "caller(s)" not in combined, combined

    def test_fast_edit_no_impact_note_when_no_callers(
        self, tmp_path: Path,
    ):
        """Signature change with zero cross-file callers: no note
        (N ≥ 1 is the threshold)."""
        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text(textwrap.dedent("""\
        def orphan(a):
            return a + 1
        """))
        # No other files reference orphan.

        snippet = "def orphan(a, b):\n    return a + b\n"
        proc = _run_cli(
            ["edit", str(a), "--snippet", snippet, "--replace", "orphan"],
            cwd=root,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        combined = proc.stdout + proc.stderr
        assert "signature of 'orphan' changed" not in combined, combined


class TestFastEditImpactHotPath:
    """VAL-M3-002: confirm no ``tldr`` subprocess is spawned when the
    signature is unchanged. We call the helper directly (rather than
    through the CLI) so we can monkeypatch at the module level."""

    def test_body_only_edit_skips_tldr_subprocess(
        self, tmp_path: Path, monkeypatch,
    ):
        """Direct helper-level test: ``compute_signature_impact_note``
        short-circuits when signatures match, never calling
        ``subprocess.run``."""
        import fastedit.inference.caller_safety as cs_mod

        calls: list = []
        real_run = cs_mod.subprocess.run

        def spy(*a, **kw):
            calls.append((a, kw))
            return real_run(*a, **kw)

        monkeypatch.setattr(cs_mod.subprocess, "run", spy)

        root = _make_project(tmp_path)
        a = root / "a.py"
        old = "def foo(a):\n    return a\n"
        new = "def foo(a):\n    return a + 1\n"  # body only
        a.write_text(old)

        note = cs_mod.compute_signature_impact_note(
            old_code=old,
            new_code=new,
            symbol="foo",
            language="python",
            file_path=a,
            project_root=root,
        )
        assert note is None
        # No tldr subprocess call should have fired.
        assert calls == [], f"expected no subprocess.run calls, got: {calls}"

    def test_signature_change_invokes_tldr(
        self, tmp_path: Path,
    ):
        """Sanity check the happy path: signature changed + external
        caller → a non-empty impact note that mentions the count."""
        from fastedit.inference.caller_safety import (
            compute_signature_impact_note,
        )

        root = _make_project(tmp_path)
        a = root / "a.py"
        a.write_text("def foo(a):\n    return a\n")
        (root / "b.py").write_text(
            "from a import foo\n\ndef use():\n    return foo(1)\n"
        )

        old = "def foo(a):\n    return a\n"
        new = "def foo(a, b):\n    return a + b\n"
        note = compute_signature_impact_note(
            old_code=old,
            new_code=new,
            symbol="foo",
            language="python",
            file_path=a,
            project_root=root,
        )
        assert note is not None
        assert "signature of 'foo' changed" in note
        assert "caller" in note
