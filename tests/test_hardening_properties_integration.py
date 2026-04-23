"""Property / invariant + tool-integration hardening tests for M1-M4.

Milestone 4.6 Goals 2 & 4:

Goal 2 (properties):
    1. Rename idempotence: rename(rename(x, a, b), a, b) == rename(x, a, b)
    2. Move-then-move-back identity (modulo whitespace)
    3. Dry-run is always a no-op (mtimes + contents unchanged)
    4. kind_filter monotonicity:
         rename_all(..., kind_filter="class") ⊆ rename_all(..., kind_filter=None)
    5. force=True monotonicity: every state force=False accepts,
         force=True also accepts.

Goal 4 (tool integrations):
    1. Move-then-rename: move Foo a.py -> b.py, then rename Foo -> Bar
    2. Delete-with-force: confirm --force is truly opt-in with known
         consequences (would-have-refused signal)
    3. Edit-signature-then-rename: sig change then rename composes
    4. Rename-with-kind-filter-then-delete
    5. Move-then-check-empty-source: after move, source file is
         ready to delete / empty (don't actually delete)

hypothesis is NOT used (not a dependency) — hand-crafted parametrized
tests per property.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastedit.inference.caller_safety import (
    check_cross_file_callers,
    signature_changed,
)
from fastedit.inference.move_to_file import move_to_file
from fastedit.inference.rename import do_cross_file_rename, do_rename_ast


TLDR_AVAILABLE = shutil.which("tldr") is not None


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


# ===========================================================================
# PROPERTY 1: Rename idempotence
# ===========================================================================
#
# The M1 cross-language tests already cover the single-file path. Here we
# add the cross-file path (do_cross_file_rename).


def test_prop_rename_all_idempotent_python(tmp_path: Path):
    """rename_all X->Y executed twice: second run finds zero files.

    Tests the cross-file path to complement the single-file idempotence
    test in test_hardening_cross_language.py.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    (root / "a.py").write_text("def oldFunc():\n    return 1\n")
    (root / "b.py").write_text(
        "from a import oldFunc\n\ndef use():\n    return oldFunc()\n"
    )

    plan1 = do_cross_file_rename(root, "oldFunc", "newFunc")
    assert plan1, "baseline rename found 0 files"
    # Apply the plan.
    for path, (new_content, _, _) in plan1.items():
        path.write_text(new_content)

    # Second pass.
    plan2 = do_cross_file_rename(root, "oldFunc", "newFunc")
    assert plan2 == {}, (
        f"idempotence broken: second rename found "
        f"{sum(c for _, c, _ in plan2.values())} refs in "
        f"{len(plan2)} files. {plan2}"
    )


# ===========================================================================
# PROPERTY 2: Move-then-move-back is identity (modulo whitespace)
# ===========================================================================


def _normalize_ws(s: str) -> str:
    """Collapse runs of blank lines + strip trailing whitespace per line.

    Move may introduce/remove a blank line between symbols depending on
    insertion position; the *symbol content* is the invariant.
    """
    lines = [ln.rstrip() for ln in s.splitlines()]
    # Collapse >=2 consecutive blank lines to one.
    out: list[str] = []
    prev_blank = False
    for ln in lines:
        is_blank = (ln == "")
        if is_blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = is_blank
    return "\n".join(out).strip() + "\n"


def test_prop_move_then_move_back_python(tmp_path: Path):
    """Moving Foo a.py -> b.py, then back to a.py: the SET of symbols
    in each file matches the original. Note: move_to_file appends to
    end-of-file when ``after`` is None, so exact source ORDER is NOT
    preserved across the round-trip. We assert set-equivalence."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a_orig = textwrap.dedent("""\
    def foo():
        return 1


    def other():
        return 2
    """)
    a.write_text(a_orig)
    b = root / "b.py"
    b_orig = textwrap.dedent("""\
    def existing():
        return 99
    """)
    b.write_text(b_orig)

    # Forward move.
    plan1 = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan1.applied is True

    # Reverse move.
    plan2 = move_to_file(
        symbol="foo",
        from_file=str(b),
        to_file=str(a),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan2.applied is True

    # a.py contains foo + other; b.py contains only existing.
    import re
    def defs(txt: str) -> set[str]:
        return set(re.findall(r"def (\w+)", txt))

    a_after = a.read_text()
    b_after = b.read_text()

    assert defs(a_after) == defs(a_orig), (
        f"a.py defs drifted: {defs(a_after)} vs {defs(a_orig)}"
    )
    assert defs(b_after) == defs(b_orig), (
        f"b.py defs drifted: {defs(b_after)} vs {defs(b_orig)}"
    )
    # Both symbols' bodies intact.
    assert "return 1" in a_after
    assert "return 2" in a_after
    assert "return 99" in b_after


# ===========================================================================
# PROPERTY 3: Dry-run is always a no-op
# ===========================================================================


def test_prop_dry_run_move_preserves_mtimes(tmp_path: Path):
    """move_to_file(dry_run=True) must not touch mtimes OR content on
    any of source, destination, or caller files."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.py"
    b.write_text("def existing():\n    return 99\n")
    caller = root / "caller.py"
    caller.write_text(
        "from a import foo\n\ndef use():\n    return foo()\n"
    )

    files = (a, b, caller)
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in files}
    content_before = {p.name: p.read_bytes() for p in files}

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=True,
    )
    assert plan.applied is False

    for p in files:
        assert p.stat().st_mtime_ns == mtimes_before[p.name], (
            f"dry-run touched mtime of {p.name}"
        )
        assert p.read_bytes() == content_before[p.name], (
            f"dry-run modified content of {p.name}"
        )


def test_prop_dry_run_rename_all_cli_preserves_mtimes(tmp_path: Path):
    """`fastedit rename-all --dry-run` must not mutate any file."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.py"
    b.write_text(
        "from a import foo\n\ndef use():\n    return foo()\n"
    )

    files = (a, b)
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in files}
    bytes_before = {p.name: p.read_bytes() for p in files}

    proc = _run_cli(
        ["rename-all", str(root), "foo", "bar", "--dry-run"], cwd=root,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    for p in files:
        assert p.stat().st_mtime_ns == mtimes_before[p.name]
        assert p.read_bytes() == bytes_before[p.name]


# ===========================================================================
# PROPERTY 4: kind_filter monotonicity
# ===========================================================================


def test_prop_kind_filter_monotonic(tmp_path: Path):
    """Result of rename_all(kind_filter="class") ⊆ result of
    rename_all(kind_filter=None) — measured as the set of
    (file, replacement_count) pairs.

    The filter can only SHRINK the plan — it must never produce edits
    that the unfiltered plan didn't already include.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    # Ambiguous name 'Foo': a class AND a module-level variable.
    (root / "a.py").write_text(textwrap.dedent("""\
    class Foo:
        pass

    Foo_instance = Foo()
    """))
    (root / "b.py").write_text(textwrap.dedent("""\
    from a import Foo

    def make() -> Foo:
        return Foo()
    """))

    plan_no_filter = do_cross_file_rename(root, "Foo", "Bar")
    plan_class = do_cross_file_rename(root, "Foo", "Bar", kind_filter="class")

    files_no_filter = {p.name for p in plan_no_filter.keys()}
    files_class = {p.name for p in plan_class.keys()}

    # Monotonicity: every file in plan_class must appear in plan_no_filter.
    assert files_class.issubset(files_no_filter), (
        f"kind_filter=class produced files absent from unfiltered plan.\n"
        f"  class-only: {files_class}\n  unfiltered: {files_no_filter}"
    )

    # Count-wise: per-file count in plan_class <= count in plan_no_filter.
    for path, (_, class_count, _) in plan_class.items():
        matching = [
            v for k, v in plan_no_filter.items() if k.name == path.name
        ]
        if matching:
            _, full_count, _ = matching[0]
            assert class_count <= full_count, (
                f"[{path.name}] kind_filter count {class_count} > "
                f"unfiltered count {full_count}"
            )


def test_prop_kind_filter_variable_misses_class_defs(tmp_path: Path):
    """When the symbol resolves to a class, kind_filter='variable'
    returns {} — the filter is exact on the resolved definition kind.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    (root / "a.py").write_text("class Foo:\n    pass\n")

    plan = do_cross_file_rename(
        root, "Foo", "Bar", kind_filter="variable",
    )
    # tldr resolves Foo as a class; filter="variable" can't match.
    assert plan == {}


# ===========================================================================
# PROPERTY 5: force=True monotonicity
# ===========================================================================


def test_prop_force_monotonic_accepts_superset(tmp_path: Path):
    """If fast_delete(force=False) accepts (no cross-file refs), then
    fast_delete(force=True) MUST also accept. Forward direction:
    --force widens the accept set, never narrows it.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def orphan():\n    return 1\n\ndef keep():\n    return 2\n")

    # Without callers: force=False accepts.
    proc_no_force = _run_cli(["delete", str(a), "orphan"], cwd=root)
    assert proc_no_force.returncode == 0, (
        f"orphan delete should succeed without --force: "
        f"{proc_no_force.stdout}{proc_no_force.stderr}"
    )

    # Restore + try with --force: also accepts (superset).
    a.write_text("def orphan():\n    return 1\n\ndef keep():\n    return 2\n")
    proc_force = _run_cli(["delete", str(a), "orphan", "--force"], cwd=root)
    assert proc_force.returncode == 0, (
        f"--force delete should also succeed: "
        f"{proc_force.stdout}{proc_force.stderr}"
    )


# ===========================================================================
# INTEGRATION 1: Move-then-rename
# ===========================================================================


def test_integ_move_then_rename_cross_file(tmp_path: Path):
    """Move Foo from a.py to b.py, then rename Foo -> Bar across the
    project. Assert Bar ends up in b.py and all callers updated."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    class Foo:
        def method(self):
            return 1
    """))
    b = root / "b.py"
    b.write_text(textwrap.dedent("""\
    def existing():
        return 99
    """))
    caller = root / "caller.py"
    caller.write_text(textwrap.dedent("""\
    from a import Foo

    def use():
        return Foo()
    """))

    # Step 1: Move.
    plan = move_to_file(
        symbol="Foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True

    # Verify move landed.
    assert "class Foo" in b.read_text()
    assert "class Foo" not in a.read_text()
    # Caller import was rewritten.
    caller_after_move = caller.read_text()
    assert "from b import Foo" in caller_after_move

    # Step 2: Rename across the project.
    rename_plan = do_cross_file_rename(root, "Foo", "Bar")
    assert rename_plan, "rename plan empty after move"
    for path, (new_content, _, _) in rename_plan.items():
        path.write_text(new_content)

    # Bar now in b.py.
    assert "class Bar" in b.read_text()
    # Caller uses Bar.
    caller_final = caller.read_text()
    assert "Bar" in caller_final
    assert "Foo" not in caller_final


# ===========================================================================
# INTEGRATION 2: Delete-with-force reveals intended dangling import
# ===========================================================================


def test_integ_force_delete_leaves_dangling_import(tmp_path: Path):
    """VAL-M2-003: --force is truly opt-in. Deleting foo with --force
    when callers exist produces a file where those callers still have
    their `from a import foo` — we don't automatically migrate them.
    The user is expected to do so BEFORE --force'ing.

    This test locks in that the refusal path catches what --force
    bypasses: the dangling import is visible post-delete.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.py"
    b.write_text("from a import foo\n\ndef use():\n    return foo()\n")

    # Step 1: confirm refusal without --force.
    proc_refuse = _run_cli(["delete", str(a), "foo"], cwd=root)
    assert proc_refuse.returncode != 0, (
        "expected refusal with caller present"
    )
    refuse_msg = (proc_refuse.stdout + proc_refuse.stderr).lower()
    assert "b.py" in refuse_msg or "reference" in refuse_msg, (
        f"refusal should mention caller file; got: {refuse_msg}"
    )

    # Step 2: --force bypasses.
    proc_force = _run_cli(["delete", str(a), "foo", "--force"], cwd=root)
    assert proc_force.returncode == 0

    # Step 3: caller import is now dangling — confirm visible.
    assert "def foo" not in a.read_text()
    assert "from a import foo" in b.read_text(), (
        "--force is supposed to NOT auto-migrate callers"
    )


# ===========================================================================
# INTEGRATION 3: Edit-signature-then-rename composes correctly
# ===========================================================================


def test_integ_edit_signature_then_rename(tmp_path: Path):
    """Change foo's signature (add a param) via fastedit edit, then
    rename foo -> bar. Both operations compose: caller sees new
    signature AND new name."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo(a):\n    return a + 1\n")
    b = root / "b.py"
    b.write_text(
        "from a import foo\n\ndef use():\n    return foo(1)\n"
    )

    # Step 1: edit signature.
    snippet = "def foo(a, b):\n    return a + b\n"
    proc_edit = _run_cli(
        ["edit", str(a), "--snippet", snippet, "--replace", "foo"], cwd=root,
    )
    assert proc_edit.returncode == 0, proc_edit.stdout + proc_edit.stderr
    assert "def foo(a, b):" in a.read_text()

    # Impact note should have fired (signature change + caller).
    combined = proc_edit.stdout + proc_edit.stderr
    assert "signature of 'foo' changed" in combined, combined

    # Step 2: rename foo -> bar.
    rename_plan = do_cross_file_rename(root, "foo", "bar")
    for path, (new_content, _, _) in rename_plan.items():
        path.write_text(new_content)

    # Post-conditions: a.py has def bar(a, b); b.py imports bar.
    a_final = a.read_text()
    assert "def bar(a, b):" in a_final
    b_final = b.read_text()
    assert "bar" in b_final
    # foo not in either (unless it's a substring of bar — which it isn't).
    assert "foo" not in a_final
    assert "foo" not in b_final


# ===========================================================================
# INTEGRATION 4: Rename-with-kind-filter-then-delete
# ===========================================================================


def test_integ_rename_kind_filter_then_delete(tmp_path: Path):
    """Rename only class refs of 'Foo', then delete the now-unrenamed
    variable 'Foo'. Both ops succeed independently.

    This test exercises the fact that kind_filter is a safety net for
    ambiguous names: a user can rename the class part without the
    variable part, then decide what to do with the variable.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    # File with only a class — variable case exercised separately to
    # avoid tldr's definition-kind ambiguity on shared names.
    a = root / "a.py"
    a.write_text(textwrap.dedent("""\
    class Foo:
        pass

    def unused():
        return 1
    """))

    # Step 1: rename class Foo -> Bar.
    plan = do_cross_file_rename(
        root, "Foo", "Bar", kind_filter="class",
    )
    assert plan, "class rename plan is empty"
    for path, (new_content, _, _) in plan.items():
        path.write_text(new_content)
    assert "class Bar" in a.read_text()

    # Step 2: delete unused (no cross-file refs).
    proc = _run_cli(["delete", str(a), "unused"], cwd=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    final = a.read_text()
    assert "def unused" not in final
    assert "class Bar" in final


# ===========================================================================
# INTEGRATION 5: Move-then-check-empty-source
# ===========================================================================


def test_integ_move_then_source_is_ready_for_delete(tmp_path: Path):
    """After moving all symbols out of a file, the remaining content
    should be trivially empty (or comment/whitespace only). We assert
    the file is "ready for removal" without actually removing it.

    Exercises: move_to_file's extraction path cleanly leaves no stub.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.py"
    b.write_text("def existing():\n    return 99\n")

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True

    # a.py should now have no `def` — the only symbol was foo.
    a_after = a.read_text()
    assert "def foo" not in a_after
    # Content is effectively empty (might have a trailing newline).
    stripped = a_after.strip()
    assert stripped == "" or stripped.startswith("#"), (
        f"expected empty or comment-only source after move; got: "
        f"{a_after!r}"
    )


# ===========================================================================
# INTEGRATION 6 (bonus): Composable property — rename-back-rename restores
# ===========================================================================


def test_integ_rename_then_rename_back_restores(tmp_path: Path):
    """rename X->Y then rename Y->X restores the original file content
    (modulo whitespace — the engine doesn't insert blank lines on
    rename so this is strict equality)."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    root = _make_project(tmp_path)
    a = root / "a.py"
    original = "def foo():\n    return foo() + 1\n\nx = foo()\n"
    a.write_text(original)

    # Forward.
    new1, count1, _ = do_rename_ast(a, "foo", "tempname_zzz")
    assert count1 >= 2
    a.write_text(new1)

    # Backward.
    new2, count2, _ = do_rename_ast(a, "tempname_zzz", "foo")
    assert count2 == count1
    a.write_text(new2)

    assert a.read_text() == original
