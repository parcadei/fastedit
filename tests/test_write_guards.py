"""Step 19 (B37): the lost-update guard on ``_atomic_write``.

The bug: a file was read once under a per-process asyncio lock, the merge
took seconds, and any external write landing in that read-to-write window
was silently overwritten — the write had no idea the file had changed since
the read.

Contract locked down here:

1. ``_atomic_write(..., expected_stat=<stale stat>)`` on a file that changed
   on disk since the stat was captured raises ``ConcurrentModificationError``
   and leaves the file byte-for-byte as the external writer left it (and no
   temp file behind).
2. A matching stat writes normally.
3. ``expected_stat=None`` behaves exactly as before (backward compat), and
   remains the default for callers with no read-time stat.
4. A destination that vanished since the read is a concurrent modification
   too (the write would otherwise resurrect it).
5. ``io_utils.read_source(..., return_stat=True)`` hands back the stat
   captured from the SAME open the bytes came from; the default stays the
   2-tuple so existing callers are untouched.
6. The MCP edit tools return a clean refusal (no traceback, nothing
   written); the CLI refuses with its usual ``Error: ...`` + exit 1.
7. With N-deep backups (B38), two consecutive CLI edits are BOTH undoable,
   step by step.
8. undo/diff decode the popped/peeked bytes FOR DISPLAY ONLY — a latin-1
   file's backup shows its real characters in the diff and the file's bytes
   are restored exactly; the display decode is never written back.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from fastedit.inference.ast_utils import ChunkedMergeResult
from fastedit.io_utils import read_source
from fastedit.mcp import tools_edit
from fastedit.mcp.backup import (
    BackupStore,
    ConcurrentModificationError,
    _atomic_write,
)

PY_ORIGINAL = "def existing():\n    return 1\n"
PY_EXTERNAL = "def existing():\n    return 'EXTERNAL'\n"
PY_MERGED = "def existing():\n    return 2\n"

# A mtime delta far above any filesystem's timestamp granularity, so the
# guard's size/mtime_ns comparison cannot miss the simulated external write.
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


# ---------------------------------------------------------------------------
# _atomic_write(expected_stat=...) — the guard itself
# ---------------------------------------------------------------------------


class TestAtomicWriteLostUpdateGuard:
    def test_stale_stat_refuses_write_and_leaves_file_untouched(self, tmp_path):
        """B37: a write whose expected_stat no longer matches the file on
        disk is refused; the external writer's content is left exactly."""
        target = tmp_path / "f.py"
        target.write_text("original\n", encoding="utf-8")
        stale = target.stat()

        _external_write(target, "EXTERNAL WRITER\n")

        with pytest.raises(ConcurrentModificationError):
            _atomic_write(target, "merged\n", expected_stat=stale)

        assert target.read_text(encoding="utf-8") == "EXTERNAL WRITER\n"
        assert _stray_temp_files(tmp_path) == []  # temp file cleaned up

    def test_matching_stat_writes_normally(self, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("original\n", encoding="utf-8")
        st = target.stat()

        _atomic_write(target, "merged\n", expected_stat=st)

        assert target.read_text(encoding="utf-8") == "merged\n"

    def test_expected_stat_none_behaves_exactly_as_before(self, tmp_path):
        """Backward compat: the default (and an explicit None) must not
        change behavior for callers with no read-time stat."""
        target = tmp_path / "f.py"
        _atomic_write(target, "created\n", expected_stat=None)
        assert target.read_text(encoding="utf-8") == "created\n"

        _atomic_write(target, "again\n")  # no-arg default form
        assert target.read_text(encoding="utf-8") == "again\n"

    def test_deleted_destination_is_a_concurrent_modification(self, tmp_path):
        """A vanished destination must not be silently resurrected by a
        write computed from stale content."""
        target = tmp_path / "f.py"
        target.write_text("original\n", encoding="utf-8")
        stale = target.stat()
        target.unlink()

        with pytest.raises(ConcurrentModificationError):
            _atomic_write(target, "merged\n", expected_stat=stale)

        assert not target.exists()

    def test_bytes_content_is_guarded_too(self, tmp_path):
        target = tmp_path / "f.py"
        target.write_bytes(b"original\n")
        stale = target.stat()

        _external_write(target, "EXTERNAL\n")

        with pytest.raises(ConcurrentModificationError):
            _atomic_write(target, b"merged\n", expected_stat=stale)
        assert target.read_bytes() == b"EXTERNAL\n"

    def test_backup_is_raw_bytes_even_when_a_guarded_write_is_refused(
        self, tmp_path,
    ):
        """The refused write's backup stores what was on disk (the external
        content) as raw bytes — never a decode of it."""
        store = BackupStore()
        target = tmp_path / "f.py"
        target.write_bytes(b"# caf\xe9\nold = 1\n")
        stale = target.stat()

        external = b"# caf\xe9\nEXTERNAL = 1\n"
        target.write_bytes(external)
        st = os.stat(target)
        os.utime(target, ns=(st.st_mtime_ns + _MTIME_DELTA_NS,) * 2)

        with pytest.raises(ConcurrentModificationError):
            _atomic_write(target, "new\n", backups=store, expected_stat=stale)

        assert store.pop(str(target)) == external


# ---------------------------------------------------------------------------
# read_source(return_stat=True) — capturing the stat at READ time
# ---------------------------------------------------------------------------


class TestReadSourceReturnStat:
    def test_default_return_is_still_a_two_tuple(self, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("x = 1\n", encoding="utf-8")
        assert read_source(target) == ("x = 1\n", "utf-8")

    def test_return_stat_adds_the_stat_of_the_same_open(self, tmp_path):
        target = tmp_path / "f.py"
        target.write_text("x = 1\n", encoding="utf-8")

        text, encoding, st = read_source(target, return_stat=True)

        assert (text, encoding) == ("x = 1\n", "utf-8")
        on_disk = os.stat(target)
        assert (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) == (
            on_disk.st_size, on_disk.st_mtime_ns, on_disk.st_ino, on_disk.st_dev,
        )

    def test_return_stat_works_for_the_empty_file_branch(self, tmp_path):
        target = tmp_path / "f.py"
        target.write_bytes(b"")

        text, encoding, st = read_source(target, return_stat=True)

        assert (text, encoding, st.st_size) == ("", "utf-8", 0)

    def test_read_to_write_window_refuses_after_an_external_write(self, tmp_path):
        """The B37 scenario end-to-end at the io/backup boundary: read with
        its stat, an external write lands, the guarded write refuses."""
        target = tmp_path / "f.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")
        _text, _encoding, st = read_source(target, return_stat=True)

        _external_write(target, PY_EXTERNAL)

        with pytest.raises(ConcurrentModificationError):
            _atomic_write(target, PY_MERGED, expected_stat=st)
        assert target.read_text(encoding="utf-8") == PY_EXTERNAL


# ---------------------------------------------------------------------------
# MCP boundary: clean refusals from the edit tools
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeEngine:
    """Engine handed out by the fake backend. Never actually used: the merge
    is stubbed in every tool test, so merge_auto is never invoked."""

    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError("merge_fn must not be called; the merge is stubbed")


class _FakeBackend:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield _FakeEngine()


class _FakeMcp:
    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


def _install_fake_mcp(monkeypatch) -> dict:
    """Point ``tools_edit.mcp`` at a fake context; disable the update check."""
    lifespan_context = {
        "backend_kind": "mlx",
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_edit, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


def _result(merged_code: str, parse_valid: bool = True) -> ChunkedMergeResult:
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=1,
        chunk_regions=[(1, 5)],
        model_tokens=0,
        latency_ms=0.0,
    )


def _stub_merge_with_external_write(monkeypatch, target: Path, external: str):
    """A merge stub that simulates an external writer landing its change
    during the (long) merge, then hands back a clean merged result."""
    def racing_merge(*a, **kw):
        _external_write(target, external)
        return _result(PY_MERGED)

    monkeypatch.setattr(tools_edit, "chunked_merge", racing_merge)


def _stub_batch_merge_with_external_write(monkeypatch, target: Path, external: str):
    """Same as :func:`_stub_merge_with_external_write` for the batch merge
    the batch/multi tools call."""
    def racing_batch(*a, **kw):
        _external_write(target, external)
        return _result(PY_MERGED)

    monkeypatch.setattr(tools_edit, "batch_chunked_merge", racing_batch)


class TestMcpConcurrentModificationRefusals:
    def test_fast_edit_refuses_cleanly_and_writes_nothing(
        self, tmp_path, monkeypatch,
    ):
        lifespan = _install_fake_mcp(monkeypatch)
        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")
        _stub_merge_with_external_write(monkeypatch, target, PY_EXTERNAL)

        message = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="x", after="existing",
        ))

        assert "file changed on disk since it was read" in message, message
        assert "re-read and retry" in message, message
        assert "file unchanged" in message, message
        assert not message.startswith("Traceback"), message
        assert target.read_text(encoding="utf-8") == PY_EXTERNAL
        assert _stray_temp_files(tmp_path) == []
        # The merge result was never persisted into the undo history as the
        # tool's own edit: the newest backup is the external content.
        assert lifespan["backups"].peek(str(target)) == PY_EXTERNAL.encode("utf-8")

    def test_fast_batch_edit_refuses_cleanly_and_writes_nothing(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")
        _stub_batch_merge_with_external_write(monkeypatch, target, PY_EXTERNAL)

        message = asyncio.run(tools_edit.fast_batch_edit(
            file_path=str(target),
            edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        assert "file changed on disk since it was read" in message, message
        assert "file unchanged" in message, message
        assert target.read_text(encoding="utf-8") == PY_EXTERNAL
        assert _stray_temp_files(tmp_path) == []

    def test_fast_multi_edit_refuses_only_the_changed_target(
        self, tmp_path, monkeypatch,
    ):
        """Partial-batch semantics: the raced target is refused, the clean
        target still writes, and the summary says a file was not written."""
        _install_fake_mcp(monkeypatch)
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL, encoding="utf-8")
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL, encoding="utf-8")

        results = iter([
            _result(PY_MERGED),  # A: merge ok, but the stub writes behind our back
            _result(PY_MERGED),  # B: clean
        ])

        def racing_batch(*a, **kw):
            result = next(results)
            if kw.get("file_path") == str(target_a):
                _external_write(target_a, PY_EXTERNAL)
            return result

        monkeypatch.setattr(tools_edit, "batch_chunked_merge", racing_batch)

        message = asyncio.run(tools_edit.fast_multi_edit(
            file_edits=json.dumps([
                {"file_path": str(target_a), "edits": [{"snippet": "x"}]},
                {"file_path": str(target_b), "edits": [{"snippet": "y"}]},
            ]),
        ))

        assert "file changed on disk since it was read" in message, message
        assert f"{target_a}: file changed on disk" in message, message
        assert "not written" in message, message
        assert target_a.read_text(encoding="utf-8") == PY_EXTERNAL  # untouched by us
        assert target_b.read_text(encoding="utf-8") == PY_MERGED  # still written

    def test_two_consecutive_fast_edits_leave_both_backups_restorable(
        self, tmp_path, monkeypatch,
    ):
        """B38 end-to-end through the real tool: the second edit must not
        destroy the first backup; undo can walk back step by step."""
        lifespan = _install_fake_mcp(monkeypatch)
        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")
        backups: BackupStore = lifespan["backups"]

        asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="def added_one():\n    return 1\n",
            after="existing",
        ))
        after_first = target.read_text(encoding="utf-8")
        asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="def added_two():\n    return 2\n",
            after="added_one",
        ))

        assert backups.pop(str(target)) == after_first.encode("utf-8")
        assert backups.pop(str(target)) == PY_ORIGINAL.encode("utf-8")


# ---------------------------------------------------------------------------
# CLI boundary: clean refusal + N-deep undo + display-only decode
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the real fastedit CLI (python -m fastedit) as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # the return code IS the assertion target
        env=env,
    )


class TestCliConcurrentModificationRefusal:
    def test_cmd_edit_refuses_cleanly_when_the_file_changed_since_read(
        self, tmp_path, monkeypatch, capsys,
    ):
        """In-process cmd_edit: an external write lands during the merge;
        the CLI refuses with its usual Error style and exit 1."""
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_edit

        target = tmp_path / "f.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")

        real_merge = chunked_merge_module.chunked_merge

        def racing_merge(*a, **kw):
            _external_write(target, PY_EXTERNAL)
            return real_merge(*a, **kw)

        monkeypatch.setattr(chunked_merge_module, "chunked_merge", racing_merge)

        args = argparse.Namespace(
            file=str(target),
            snippet="def appended():\n    return 0\n",
            replace="", after="existing",
            backend=None, model_path=None, api_base=None, api_model=None,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_edit(args)

        assert exc.value.code == 1
        stderr = capsys.readouterr().err
        assert "file changed on disk since it was read" in stderr, stderr
        assert "Traceback" not in stderr, stderr
        assert target.read_text(encoding="utf-8") == PY_EXTERNAL  # untouched by us
        assert _stray_temp_files(tmp_path) == []


class TestCliNDeepUndo:
    def test_two_edits_are_both_undoable_step_by_step(self, tmp_path):
        """B38 end-to-end: undo after two edits reverts the second, then the
        first — the old 1-deep store destroyed the first backup instead."""
        f = tmp_path / "m.py"
        f.write_bytes(b"def f():\n    return 1\n")

        r1 = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 2\n")
        assert r1.returncode == 0, r1.stderr
        r2 = run_cli("edit", str(f), "--replace", "f", "--snippet", "def f():\n    return 3\n")
        assert r2.returncode == 0, r2.stderr

        u1 = run_cli("undo", str(f))
        assert u1.returncode == 0, u1.stderr
        assert f.read_bytes() == b"def f():\n    return 2\n"  # reverts edit 2

        u2 = run_cli("undo", str(f))
        assert u2.returncode == 0, u2.stderr
        assert f.read_bytes() == b"def f():\n    return 1\n"  # reverts edit 1 (B38)


LATIN1_ORIGINAL = b"# caf\xe9\ndef a():\n    return 'caf\xe9 value'\n\n\ndef b():\n    return 2\n"
LATIN1_EDIT = ("def a():\n    return 'café 2'\n")


class TestCliDisplayOnlyDecode:
    """undo/diff pop/peek bytes and decode them FOR DISPLAY ONLY — never
    written; a latin-1 file shows its real characters in the diff (the é
    sits on an edited line so it appears in unified_diff's changed lines),
    and the file's bytes are restored exactly."""

    def test_undo_restores_latin1_bytes_and_shows_real_characters(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_bytes(LATIN1_ORIGINAL)
        edit = run_cli("edit", str(f), "--replace", "a", "--snippet", LATIN1_EDIT)
        assert edit.returncode == 0, edit.stderr

        undo = run_cli("undo", str(f))
        assert undo.returncode == 0, undo.stderr
        assert f.read_bytes() == LATIN1_ORIGINAL  # byte-for-byte restore
        assert "caf\xe9" in undo.stdout  # é decoded for display, not mojibake
        assert "\ufffd" not in undo.stdout

    def test_diff_decodes_the_backup_for_display_only(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_bytes(LATIN1_ORIGINAL)
        edit = run_cli("edit", str(f), "--replace", "a", "--snippet", LATIN1_EDIT)
        assert edit.returncode == 0, edit.stderr

        diff = run_cli("diff", str(f))
        assert diff.returncode == 0, diff.stderr
        assert "caf\xe9" in diff.stdout
        assert "\ufffd" not in diff.stdout
        # Display only: the file on disk still carries the latin-1 bytes.
        assert f.read_bytes().count(b"\xe9") == 2
        assert f.read_bytes().count(b"\xef\xbf\xbd") == 0

    def test_diff_after_undo_reports_no_backup(self, tmp_path):
        """Popping the last backup on undo removes the file's undo history,
        so diff afterwards reports no backup (existing behavior kept)."""
        f = tmp_path / "m.py"
        f.write_bytes(LATIN1_ORIGINAL)
        edit = run_cli("edit", str(f), "--replace", "b", "--snippet", "def b():\n    return 20\n")
        assert edit.returncode == 0, edit.stderr
        undo = run_cli("undo", str(f))
        assert undo.returncode == 0, undo.stderr

        diff = run_cli("diff", str(f))
        assert diff.returncode == 0
        assert "no backup" in diff.stdout.lower()


# ---------------------------------------------------------------------------
# G2 audit: ``io_utils.write_all`` — descriptor writes must never silently
# truncate. A single os.write may legally write fewer bytes than requested
# (POSIX returns the count written; a >2 GiB single write is capped by the
# kernel on Linux, and signal interruption can stop one early) — ignoring
# the return value truncates the payload in exactly those cases. Every
# whole-file descriptor write (backups, undo, temp copies) now goes through
# the looping helper; these tests pin the loop against simulated partial
# writes.
# ---------------------------------------------------------------------------


def test_write_all_completes_when_os_write_is_partial(monkeypatch, tmp_path):
    from fastedit import io_utils

    real_write = os.write
    calls: list[int] = []

    def partial_write(fd, data):
        chunk = bytes(data)[:5]  # never more than 5 bytes per call
        calls.append(len(chunk))
        return real_write(fd, chunk)

    monkeypatch.setattr(io_utils.os, "write", partial_write)
    payload = b"0123456789" * 7  # 70 bytes -> 14 partial calls
    target = tmp_path / "partial.bin"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        io_utils.write_all(fd, payload)
    finally:
        os.close(fd)
    assert target.read_bytes() == payload
    assert len(calls) > 1  # the loop actually looped


def test_write_all_refuses_a_zero_byte_write(monkeypatch, tmp_path):
    from fastedit import io_utils

    def zero_write(_fd, _data):
        return 0

    monkeypatch.setattr(io_utils.os, "write", zero_write)
    fd = os.open(tmp_path / "zero.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        with pytest.raises(OSError, match="0 bytes written"):
            io_utils.write_all(fd, b"nonempty")
    finally:
        os.close(fd)


def test_backup_store_round_trips_through_partial_writes(
    monkeypatch, tmp_path,
):
    """Integration: BackupStore's raw-bytes record (the undo path) survives
    simulated partial os.write -- the worst place to lose bytes silently."""
    from fastedit.mcp.backup import BackupStore

    real_write = os.write

    def partial_write(fd, data):
        return real_write(fd, bytes(data)[:3])

    monkeypatch.setattr(os, "write", partial_write)
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    store = BackupStore()
    payload = bytes(range(256)) * 41  # 10496 bytes
    store["f.py"] = payload
    assert store.pop("f.py") == payload


def test_atomic_write_round_trips_through_partial_writes(
    monkeypatch, tmp_path,
):
    """Integration: _atomic_write's fsync'd payload lands complete even when
    the descriptor write is repeatedly partial."""
    from fastedit.mcp.backup import _atomic_write

    real_write = os.write

    def partial_write(fd, data):
        return real_write(fd, bytes(data)[:7])

    monkeypatch.setattr(os, "write", partial_write)
    target = tmp_path / "dest.txt"
    payload = b"atomic-payload-" * 300
    _atomic_write(target, payload)
    assert target.read_bytes() == payload
