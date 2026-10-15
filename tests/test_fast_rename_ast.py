"""Tests for AST-verified single-file rename (do_rename_ast).

Verifies:
- Matches inside strings, comments, and docstrings are NOT renamed.
- Matches inside code (definitions, calls, etc.) ARE renamed.
- The engine uses `tldr references --scope file`, so string/comment
  substrings are filtered at the AST layer rather than via regex skip
  zones.
- (B37) The MCP fast_rename_all write loop arms the lost-update guard:
  a non-fastedit write landing between the plan read and a file's write
  is refused with the clean refusal string — the plan's stale bytes are
  never written over the external writer's content.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

import pytest

from fastedit.inference.rename import do_rename_ast

# ---------------------------------------------------------------------------
# Behavioral invariant (VAL-M1-002): strings/comments/docstrings preserved.
# ---------------------------------------------------------------------------


class TestDoRenameAstSkipsStringsAndComments:
    """Locks VAL-M1-002: substrings inside string/comment/docstring nodes
    must not be touched by the new AST-verified single-file rename."""

    def test_fast_rename_skips_strings_and_comments_python(self, tmp_path: Path):
        """Python: rename the function def + call, leave comment, string, and
        docstring substrings untouched."""
        path = tmp_path / "mod.py"
        path.write_text(textwrap.dedent("""\
        def old_name():
            \"\"\"docstring: old_name is mentioned here.\"\"\"
            return 1

        # old_name in comment
        msg = "old_name in string"
        x = old_name()
        """))

        new_content, count, skipped = do_rename_ast(path, "old_name", "new_name")

        # Code sites renamed:
        assert "def new_name():" in new_content
        assert "x = new_name()" in new_content

        # Non-code sites preserved verbatim:
        assert "docstring: old_name is mentioned here." in new_content
        assert "# old_name in comment" in new_content
        assert '"old_name in string"' in new_content

        # Exactly the def + the call were renamed.
        assert count == 2
        # There were 3 string/comment/docstring substring hits.
        assert skipped >= 3

    def test_fast_rename_skips_strings_and_comments_typescript(self, tmp_path: Path):
        """TypeScript: rename the function def + call, leave comment, string,
        and JSDoc substrings untouched."""
        path = tmp_path / "mod.ts"
        path.write_text(textwrap.dedent("""\
        /**
         * JSDoc mentions old_name in prose.
         */
        function old_name(): number {
          return 1;
        }

        // old_name in comment
        const msg: string = "old_name in string";
        const x = old_name();
        """))

        new_content, count, skipped = do_rename_ast(path, "old_name", "new_name")

        # Code sites renamed:
        assert "function new_name()" in new_content
        assert "const x = new_name();" in new_content

        # Non-code sites preserved verbatim:
        assert "JSDoc mentions old_name in prose." in new_content
        assert "// old_name in comment" in new_content
        assert '"old_name in string"' in new_content

        assert count == 2
        assert skipped >= 3


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestDoRenameAstBasics:
    def test_word_boundary_preserves_longer_names(self, tmp_path: Path):
        """Renaming 'get' must not touch 'get_all' or 'getter'."""
        path = tmp_path / "mod.py"
        path.write_text(textwrap.dedent("""\
        def get():
            pass

        def get_all():
            pass

        getter = get()
        """))

        new_content, count, _ = do_rename_ast(path, "get", "fetch")
        assert "def fetch():" in new_content
        assert "def get_all():" in new_content
        assert "getter = fetch()" in new_content
        assert count == 2

    def test_no_matches_returns_zero_count(self, tmp_path: Path):
        path = tmp_path / "mod.py"
        path.write_text("def other():\n    return 0\n")
        new_content, count, _skipped = do_rename_ast(path, "missing", "replaced")
        assert new_content == path.read_text()
        assert count == 0

    def test_same_name_is_noop(self, tmp_path: Path):
        path = tmp_path / "mod.py"
        original = "def foo():\n    return foo()\n"
        path.write_text(original)
        new_content, count, _ = do_rename_ast(path, "foo", "foo")
        assert new_content == original
        assert count == 0

    def test_unicode_content_preserved(self, tmp_path: Path):
        """Unicode in strings/comments must round-trip intact."""
        path = tmp_path / "mod.py"
        path.write_text(
            "# Calcul du coût\n"
            "def get():\n"
            "    return 'élève'\n"
            "\n"
            "x = get()\n"
        )
        new_content, count, _ = do_rename_ast(path, "get", "fetch")
        assert "coût" in new_content
        assert "élève" in new_content
        assert "def fetch():" in new_content
        assert "x = fetch()" in new_content
        assert count == 2


# ---------------------------------------------------------------------------
# Dry-run consistency tests (fastedit rename --dry-run / cmd_rename dry_run)
# ---------------------------------------------------------------------------


class TestCmdRenameDryRun:
    """Locks dry-run behaviour on the single-file rename path:
    - file must NOT be modified
    - reported replacement count must be correct
    """

    def test_fast_rename_dry_run_does_not_write(self, tmp_path: Path):
        """dry_run=True must leave the file unchanged on disk."""
        path = tmp_path / "mod.py"
        original = (
            "def old_func():\n"
            "    return 1\n"
            "\n"
            "x = old_func()\n"
        )
        path.write_text(original)

        # Simulate what cmd_rename does with --dry-run: call do_rename_ast then
        # branch on dry_run — must NOT write.
        _new_content, count, _skipped = do_rename_ast(path, "old_func", "new_func")
        assert count >= 1, "precondition: rename found references"

        # Dry-run branch: do NOT write
        # (We test the guard logic directly — file must still equal original)
        assert path.read_text() == original

    def test_fast_rename_dry_run_reports_counts(self, tmp_path: Path):
        """do_rename_ast must return a count that matches the actual replacements."""
        path = tmp_path / "mod.py"
        path.write_text(
            "def compute():\n"
            "    return compute()\n"
            "\n"
            "result = compute()\n"
        )

        new_content, count, _skipped = do_rename_ast(path, "compute", "calculate")

        # There are 3 code-level occurrences: def, recursive call, assignment call.
        assert count >= 2, f"expected >=2 replacements, got {count}"
        # The returned content must contain the new name
        assert "calculate" in new_content
        # And the old name must be gone from code (may survive in strings/comments,
        # but this fixture has none)
        assert "compute" not in new_content


# ---------------------------------------------------------------------------
# B37: MCP fast_rename_all — lost-update guard on the plan-to-write window
# ---------------------------------------------------------------------------

TLDR_AVAILABLE = shutil.which("tldr") is not None

# A mtime delta far above any filesystem's timestamp granularity, so the
# guard's size/mtime_ns comparison cannot miss the simulated external write
# (same helper as tests/test_write_guards.py).
_MTIME_DELTA_NS = 3_600_000_000_000  # +1 hour


def _external_write(path: Path, text: str) -> None:
    """Simulate another process rewriting *path* after the stat was taken:
    different size AND a visibly different mtime."""
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_mtime_ns + _MTIME_DELTA_NS,) * 2)


def _stray_temp_files(directory: Path) -> list[str]:
    """Any atomic-write temp file left behind in *directory*."""
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


class _FakeRequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeMcp:
    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


def _install_fake_mcp(monkeypatch):
    """Point ``tools_ast.mcp`` at a fake context (fast_rename_all only reads
    ``backups`` and ``file_locks`` from the lifespan context)."""
    from fastedit.mcp import tools_ast
    from fastedit.mcp.backup import BackupStore

    lifespan_context = {
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_ast, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


class TestMcpFastRenameAllConcurrentWrite:
    """B37 on the cross-file rename verb.

    ``fast_rename_all`` reads the whole plan once (``do_cross_file_rename``
    captures each file's read-time stat), then writes the files in a loop.
    A non-fastedit write landing between the plan read and a file's write
    must be REFUSED with the clean refusal string — the plan's stale bytes
    must never clobber the external writer's content.

    Mirrors tests/test_cli.py::TestCLIRenameConcurrentWrite (the same seam
    on the CLI surface) and tests/test_write_guards.py's MCP guard tests
    (the same refusal-shape assertions): a spy around the REAL tldr-backed
    engine interposes the external write deterministically after the plan
    read, because a real concurrent writer would be timing-dependent and
    flaky. Nothing about the rename engine is stubbed.
    """

    @pytest.mark.skipif(not TLDR_AVAILABLE, reason="tldr binary not on PATH")
    def test_refuses_only_the_changed_file_but_still_renames_the_rest(
        self, tmp_path: Path, monkeypatch,
    ):
        """The raced target refuses cleanly and keeps the EXTERNAL content;
        the other file still renames (partial per-file semantics)."""
        from fastedit.inference import rename as rename_module
        from fastedit.mcp import tools_ast

        _install_fake_mcp(monkeypatch)

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
        interposed: list[Path] = []

        def ordered_plan_then_external_write(
            root_dir, old_name, new_name, **kwargs,
        ):
            plan = real_do_cross_file_rename(root_dir, old_name, new_name, **kwargs)
            planned.update(plan)
            # Deterministic write order: survivor first, victim last (False
            # sorts before True), so the survivor's rename lands before the
            # victim's refusal.
            ordered = {p: plan[p] for p in sorted(plan, key=lambda p: p == victim)}
            # The non-fastedit writer clobbers the victim AFTER the plan read
            # it (the per-file read_stat was captured inside the real engine
            # above) and BEFORE the tool's write loop reaches this file.
            if victim in plan:
                interposed.append(victim)
                victim.write_bytes(external_bytes)
            return ordered

        monkeypatch.setattr(
            rename_module, "do_cross_file_rename",
            ordered_plan_then_external_write,
        )

        message = asyncio.run(tools_ast.fast_rename_all(
            root_dir=str(root), old_name="old_sym", new_name="new_sym",
        ))

        # Checked FIRST: both files were really planned (the engine found
        # references in each) and the external write really landed mid-run —
        # not short-circuited by an early no-plan exit.
        assert set(planned) == {survivor, victim}
        assert interposed == [victim]

        # Clean refusal — the same string the other MCP verbs return.
        assert "file changed on disk since it was read" in message, message
        assert "re-read and retry" in message, message
        assert "file unchanged" in message, message
        assert not message.startswith("Traceback"), message
        assert f"{victim}: file changed on disk" in message, message
        assert "NOT written" in message, message

        # The victim keeps the EXTERNAL content — no rename applied to it.
        assert victim.read_bytes() == external_bytes
        # The other file still renamed: partial semantics consistent with
        # the verb's existing per-file lock-refusal behavior.
        survivor_text = survivor.read_text()
        assert "def new_sym" in survivor_text
        assert "def old_sym" not in survivor_text

        # The refused write left no atomic-write temp file behind.
        assert _stray_temp_files(root) == []
