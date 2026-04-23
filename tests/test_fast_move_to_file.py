"""Tests for the cross-file move tool: fast_move_to_file / `fastedit move-to-file`.

Milestone 4 (VAL-M4-001..003): relocate a symbol from one file to another and
rewrite import statements in every dependent file.

Covers:
    VAL-M4-001 — happy path Python + TypeScript (symbol moved, caller imports rewritten)
    VAL-M4-002 — dry-run prints a plan and touches no files
    VAL-M4-003 — rejections: conflict in destination, same-file move

The tests exercise both the underlying ``move_to_file`` helper and the CLI
wrapper (``fastedit move-to-file``). We invoke the CLI via ``python -m fastedit``
so pytest-captured stdout reflects end-user output.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastedit.inference.move_to_file import (
    MoveToFilePlan,
    move_to_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    """Create a project root with a .git marker so tldr scopes its search."""
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
# VAL-M4-001 — happy path (Python)
# ---------------------------------------------------------------------------


def test_move_to_file_python_happy_path(tmp_path: Path):
    """Move ``foo`` from a.py to b.py, verify foo is gone from a.py,
    present in b.py, and imports in caller.py are rewritten from
    ``from a import foo`` -> ``from b import foo``."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    def foo():
        return 1


    def other():
        return 2
    """))
    b = root / "b.py"
    b.write_text(textwrap.dedent("""\
    def existing():
        return 99
    """))
    caller = root / "caller.py"
    caller.write_text(textwrap.dedent("""\
    from a import foo


    def use():
        return foo()
    """))

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )

    assert plan.applied is True
    assert plan.symbol == "foo"

    # Symbol removed from source; other symbols preserved.
    src_text = a.read_text()
    assert "def foo" not in src_text
    assert "def other" in src_text

    # Symbol present in destination; existing code preserved.
    dst_text = b.read_text()
    assert "def foo" in dst_text
    assert "def existing" in dst_text

    # Caller import rewritten. The old module reference should be gone;
    # the new module reference should be present; the symbol reference
    # in the body stays.
    caller_text = caller.read_text()
    assert "from a import foo" not in caller_text
    assert "from b import foo" in caller_text
    assert "return foo()" in caller_text

    # At least one import rewrite recorded in the plan.
    assert len(plan.import_rewrites) >= 1
    rewrites_files = {Path(r["file"]).name for r in plan.import_rewrites}
    assert "caller.py" in rewrites_files


# ---------------------------------------------------------------------------
# VAL-M4-001 — happy path (TypeScript)
# ---------------------------------------------------------------------------


def test_move_to_file_typescript_happy_path(tmp_path: Path):
    """Move ``foo`` from a.ts to b.ts (same directory) and verify the
    caller's ``import { foo } from "./a"`` gets rewritten to ``"./b"``."""
    root = _make_project(tmp_path)
    a = root / "a.ts"
    a.write_text(textwrap.dedent("""\
    export function foo() {
        return 1;
    }

    export function other() {
        return 2;
    }
    """))
    b = root / "b.ts"
    b.write_text(textwrap.dedent("""\
    export function existing() {
        return 99;
    }
    """))
    caller = root / "caller.ts"
    caller.write_text(textwrap.dedent("""\
    import { foo } from "./a";

    export function use() {
        return foo();
    }
    """))

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True

    src_text = a.read_text()
    assert "function foo" not in src_text
    assert "function other" in src_text

    dst_text = b.read_text()
    assert "function foo" in dst_text
    assert "function existing" in dst_text

    caller_text = caller.read_text()
    # Either `"./b"` replaced `"./a"`, or the whole line was split.
    # The core invariant: `"./a"` with `foo` must be gone, `./b` with
    # `foo` must be present.
    assert '"./a"' not in caller_text or "foo" not in caller_text.split('"./a"')[0].splitlines()[-1]
    assert '"./b"' in caller_text


# ---------------------------------------------------------------------------
# VAL-M4-002 — dry-run prints a plan and touches no files
# ---------------------------------------------------------------------------


def test_move_to_file_dry_run_no_write(tmp_path: Path):
    """Dry-run must not modify ANY file; it returns a plan with applied=False
    and a non-empty list of import rewrites."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    def foo():
        return 1
    """))
    b = root / "b.py"
    b.write_text(textwrap.dedent("""\
    def existing():
        return 99
    """))
    caller = root / "caller.py"
    caller.write_text(textwrap.dedent("""\
    from a import foo


    def use():
        return foo()
    """))

    a_before = a.read_text()
    b_before = b.read_text()
    caller_before = caller.read_text()

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=True,
    )

    assert isinstance(plan, MoveToFilePlan)
    assert plan.applied is False

    # No file was written.
    assert a.read_text() == a_before
    assert b.read_text() == b_before
    assert caller.read_text() == caller_before

    # Plan still describes the intended move.
    assert plan.source_span[0] >= 1
    assert plan.source_span[1] >= plan.source_span[0]
    assert len(plan.import_rewrites) >= 1


# ---------------------------------------------------------------------------
# VAL-M4-003 — rejection: symbol already exists in destination
# ---------------------------------------------------------------------------


def test_move_to_file_rejects_conflict(tmp_path: Path):
    """If the destination file already defines ``foo`` we reject with
    a clear message and no file is touched."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    def foo():
        return 1
    """))
    b = root / "b.py"
    b.write_text(textwrap.dedent("""\
    def foo():
        return 99
    """))

    a_before = a.read_text()
    b_before = b.read_text()

    with pytest.raises(ValueError) as excinfo:
        move_to_file(
            symbol="foo",
            from_file=str(a),
            to_file=str(b),
            after=None,
            project_root=root,
            dry_run=False,
        )

    msg = str(excinfo.value).lower()
    assert "foo" in msg
    assert "already" in msg or "conflict" in msg or "exists" in msg

    # No file touched.
    assert a.read_text() == a_before
    assert b.read_text() == b_before


# ---------------------------------------------------------------------------
# VAL-M4-003 — rejection: same-file move
# ---------------------------------------------------------------------------


def test_move_to_file_rejects_same_file(tmp_path: Path):
    """Moving within the same file is fast_move's job — reject with a hint."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    def foo():
        return 1


    def bar():
        return 2
    """))

    with pytest.raises(ValueError) as excinfo:
        move_to_file(
            symbol="foo",
            from_file=str(a),
            to_file=str(a),
            after=None,
            project_root=root,
            dry_run=False,
        )

    msg = str(excinfo.value).lower()
    assert "fast_move" in msg or "same" in msg
    # File untouched.
    assert "def foo" in a.read_text()
    assert "def bar" in a.read_text()


# ---------------------------------------------------------------------------
# Bonus: `after` anchor places the symbol immediately below the named
# symbol in the destination file rather than appending to end-of-file.
# ---------------------------------------------------------------------------


def test_move_to_file_with_after_anchor(tmp_path: Path):
    """``after='existing'`` places foo right after ``existing`` in b.py,
    not at end of file."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    def foo():
        return 1
    """))
    b = root / "b.py"
    b.write_text(textwrap.dedent("""\
    def existing():
        return 99


    def later():
        return 100
    """))

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after="existing",
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True

    b_text = b.read_text()
    # foo must appear between 'existing' and 'later'.
    pos_existing = b_text.find("def existing")
    pos_foo = b_text.find("def foo")
    pos_later = b_text.find("def later")
    assert pos_existing >= 0
    assert pos_foo > pos_existing
    assert pos_later > pos_foo


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_move_to_file_cli_dry_run(tmp_path: Path):
    """`fastedit move-to-file ... --dry-run` prints a plan and writes nothing."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.py"
    b.write_text("def existing():\n    return 99\n")
    caller = root / "caller.py"
    caller.write_text("from a import foo\n\ndef use():\n    return foo()\n")

    before = {p.name: p.read_text() for p in (a, b, caller)}

    proc = _run_cli(
        ["move-to-file", "foo", str(a), str(b), "--dry-run"],
        cwd=root,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "foo" in combined
    # Plan mentions the caller.
    assert "caller.py" in combined

    # Nothing changed on disk.
    for p in (a, b, caller):
        assert p.read_text() == before[p.name]


def test_move_to_file_cli_rejects_same_file(tmp_path: Path):
    """CLI same-file rejection surfaces the fast_move hint."""
    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")

    proc = _run_cli(
        ["move-to-file", "foo", str(a), str(a)],
        cwd=root,
    )

    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "fast_move" in combined or "same" in combined
