"""Tests for BackupStore: prune race handling, test-session isolation, and
the Step 19 persistence contract (B38 N-deep + B22 raw bytes).

Step 19 additions lock down:

* **B38** — the old store kept ONE backup per file, so a second edit
  overwrote the first and a corrupted second edit could only be undone one
  step. The store now keeps ``_MAX_BACKUPS_PER_FILE`` timestamped backups
  per file; ``pop`` returns the newest and older steps stay restorable.
* **B22** — the store used to decode/encode text at ``__setitem__``/``pop``.
  It now stores RAW BYTES with no codec anywhere in the store, so a
  latin-1 file's backup round-trips byte-exact.
* **B39** — ``_atomic_write`` fsyncs the file before close and the directory
  after ``os.replace`` (asserted by counting ``os.fsync`` calls, not by
  mocking the OS deeply).
* **B39 (store)** — ``BackupStore.__setitem__`` fsyncs the backup fd before
  close/replace, fsyncs the store directory after the rename (best-effort),
  and the ``.meta`` write is fsynced too: a power cut must not keep the main
  file's new content while losing the just-written backup, which would leave
  that edit without an undo step.
* **Stale-temp cleanup** — a SIGKILL bypasses every exception-path unlink and
  leaves the hidden ``.tmp`` residue behind with nobody to clean it up. The
  next ``_atomic_write`` to the same target (and ``BackupStore.__init__``
  for the backups dir) sweeps EXACTLY fastedit's own mkstemp shapes —
  ``.{target}.{8 x [a-z0-9_]}.tmp`` and the bare ``{8 x [a-z0-9_]}.tmp`` —
  and only past a conservative age threshold, so a LIVE run's in-flight temp
  is never touched and nothing that merely looks similar is ever deleted.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path

import pytest

from fastedit.mcp.backup import _MAX_BACKUPS_PER_FILE, BackupStore, _atomic_write


def test_prune_old_survives_a_backup_removed_between_listing_and_unlink(tmp_path, monkeypatch):
    """_prune_old must not raise when a stale backup vanishes mid-loop (concurrent process)."""
    # Bypass __init__ (which resolves the real per-user backup dir unless
    # overridden) so this test exercises the race in _prune_old alone, fully
    # decoupled from where BackupStore's directory normally comes from.
    store = BackupStore.__new__(BackupStore)
    store._dir = tmp_path

    # Two stale backups: one is removed by a "concurrent process" right
    # before we get to it, the other must still be pruned normally.
    raced = tmp_path / "raced.bak"
    raced.write_text("old", encoding="utf-8")
    survivor = tmp_path / "survivor.bak"
    survivor.write_text("old", encoding="utf-8")
    stale_mtime = time.time() - BackupStore._MAX_AGE_SECS - 10
    os.utime(raced, (stale_mtime, stale_mtime))
    os.utime(survivor, (stale_mtime, stale_mtime))

    real_glob = Path.glob

    def racy_glob(self, pattern):
        results = list(real_glob(self, pattern))
        if self == tmp_path and pattern == "*.bak":
            # Simulate another process deleting `raced` right after our
            # listing snapshot was taken, before we get to stat/unlink it.
            raced.unlink()
        return results

    monkeypatch.setattr(Path, "glob", racy_glob)

    store._prune_old()  # must not raise FileNotFoundError

    assert not survivor.exists()


def test_backup_store_root_is_under_tmp_not_the_real_home_store():
    """The test session's BackupStore root must live under a tmp path, never the real per-user store."""
    real_home_store = Path.home() / ".fastedit" / "backups"
    store = BackupStore()
    assert store._dir != real_home_store
    assert "FASTEDIT_BACKUP_DIR" in os.environ
    assert str(store._dir) == os.environ["FASTEDIT_BACKUP_DIR"]


def test_backup_store_rejects_a_relative_backup_dir(monkeypatch):
    """BackupStore.__init__ raises ValueError when FASTEDIT_BACKUP_DIR is not an absolute path."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", "relative/backups")
    with pytest.raises(ValueError, match="absolute path"):
        BackupStore()


# ---------------------------------------------------------------------------
# Step 19 (B38): N-deep timestamped backups
# ---------------------------------------------------------------------------


def test_two_consecutive_stores_are_both_restorable(tmp_path):
    """B38: a second edit must not overwrite the first backup — pop returns
    the newest first, then the previous one, then nothing."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"state one\n"
    store[str(target)] = b"state two\n"

    assert store.pop(str(target)) == b"state two\n"  # newest first
    assert store.pop(str(target)) == b"state one\n"  # the older step survived
    with pytest.raises(KeyError):
        store.pop(str(target))


def test_five_edits_keep_exactly_max_backups_per_file(tmp_path):
    """B38: per-file depth is capped at _MAX_BACKUPS_PER_FILE; older backups
    are pruned on write and newest-first pop order is preserved."""
    store = BackupStore()
    target = tmp_path / "f.py"
    for i in range(8):
        store[str(target)] = f"state {i}\n".encode()

    # Exactly _MAX_BACKUPS_PER_FILE .bak files kept for this one file (the
    # session backup dir is shared, so scope the count to this file).
    assert len(store._key_paths(str(target))) == _MAX_BACKUPS_PER_FILE

    popped = [store.pop(str(target)) for _ in range(_MAX_BACKUPS_PER_FILE)]
    assert popped == [
        b"state 7\n", b"state 6\n", b"state 5\n", b"state 4\n", b"state 3\n",
    ]
    assert str(target) not in store  # undo history fully consumed


def test_peek_returns_the_newest_backup_without_removing_it(tmp_path):
    """The diff commands must look at the newest backup without consuming
    the undo history (peek is non-destructive; pop still sees it)."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"older\n"
    store[str(target)] = b"newest\n"

    assert store.peek(str(target)) == b"newest\n"
    assert store.peek(str(target)) == b"newest\n"  # still there
    assert str(target) in store
    assert store.pop(str(target)) == b"newest\n"


def test_meta_records_the_original_path_and_goes_with_the_last_backup(tmp_path):
    """One <hash>.meta per file (per-backup metas would duplicate the same
    original path N times and need extra prune bookkeeping); it is removed
    only when the file's LAST backup is consumed."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"one\n"
    store[str(target)] = b"two\n"

    meta = store._meta_path(str(target))
    assert meta.read_text(encoding="utf-8") == str(target)

    store.pop(str(target))
    assert meta.exists()  # a backup remains -> undo history exists
    store.pop(str(target))
    assert not meta.exists()  # last one gone -> no undo history at all


# ---------------------------------------------------------------------------
# Step 19 (B22): raw bytes — no decode/encode anywhere in the store
# ---------------------------------------------------------------------------


def test_latin1_backup_round_trips_byte_exact_no_decode_anywhere(tmp_path):
    """B22: a latin-1 file's 0xE9 byte survives the backup round-trip
    byte-exact. A UTF-8 decode would raise on it; errors="replace" would
    corrupt it into EF BF BD; neither may happen in the store."""
    store = BackupStore()
    target = tmp_path / "f.py"
    raw = b"# caf\xe9\nx = 1\n"  # 0xE9 is not valid UTF-8
    store[str(target)] = raw

    assert store.peek(str(target)) == raw
    assert store.pop(str(target)) == raw


def test_atomic_write_stores_the_current_file_as_raw_bytes(tmp_path):
    """B22: _atomic_write's backup of the CURRENT file is the file's raw
    bytes — byte-exact for a latin-1 file, independent of any codec."""
    store = BackupStore()
    target = tmp_path / "f.py"
    raw = b"# caf\xe9\nold = 1\n"
    target.write_bytes(raw)

    _atomic_write(target, "new = 2\n", backups=store, encoding="latin-1")

    assert store.pop(str(target)) == raw


# ---------------------------------------------------------------------------
# Step 19 (B39): fsync before close + best-effort directory fsync
# ---------------------------------------------------------------------------


def test_atomic_write_fsyncs_the_file_and_the_directory(tmp_path, monkeypatch):
    """B39: the written file is fsync'd before close and the directory is
    fsync'd after os.replace. Counted via os.fsync; the directory fd is
    identified by intercepting os.open rather than mocking the OS deeply."""
    real_open = os.open
    real_fsync = os.fsync
    opened_dirs: set[int] = set()
    fsync_targets: list[str] = []

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_dirs.add(fd)
        return fd

    def counting_fsync(fd):
        fsync_targets.append("dir" if fd in opened_dirs else "file")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", counting_fsync)

    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    _atomic_write(target, "new\n", expected_stat=None)

    assert fsync_targets.count("file") == 1  # content durability before close
    assert fsync_targets.count("dir") == 1  # rename durability after replace
    assert target.read_text(encoding="utf-8") == "new\n"


def test_directory_fsync_failure_does_not_break_the_write(tmp_path, monkeypatch):
    """B39: the directory fsync is best-effort — a filesystem that refuses
    it must not fail an otherwise good write (content fsync already done)."""
    real_open = os.open
    real_fsync = os.fsync
    opened_dirs: set[int] = set()

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_dirs.add(fd)
        return fd

    def refusing_dir_fsync(fd):
        if fd in opened_dirs:
            raise OSError(45, "Operation not supported")  # e.g. some filesystems
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", refusing_dir_fsync)

    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    _atomic_write(target, "new\n")  # must not raise

    assert target.read_text(encoding="utf-8") == "new\n"


# ---------------------------------------------------------------------------
# B39 for the store: __setitem__ durability (backup fd + store dir + .meta)
#
# _atomic_write already fsyncs its own writes (B39), but the backup it stores
# first went through BackupStore.__setitem__, which did write_all + close +
# replace with NO fsync: a power-loss window could keep the main file's
# fsync'd new content while losing the just-written backup, leaving that edit
# without an undo step. The <hash>.meta write (plain write_text) had the same
# gap. The recorder below is the B39 counting pattern extended with
# write/close/replace ordering so fsync-vs-rename order can be asserted
# without mocking the OS deeply.
# ---------------------------------------------------------------------------


def _install_fsync_recorder(monkeypatch) -> list[tuple]:
    """Record fd-level open/write/fsync/close ordering plus every
    ``os.replace`` target. Every fake delegates to the real syscall, so the
    real work still happens. Returns *events*, an ordered list of
    ``("open", fd, token, path)`` / ``("write", fd, token)`` /
    ``("fsync", fd, token, is_dir)`` / ``("close", fd, token)`` /
    ``("replace", src, dst)`` tuples.

    *token* identifies the open session the event belongs to, and *is_dir*
    is resolved with ``fstat`` at fsync time: fd NUMBERS are reused the
    instant a descriptor is closed (a directory fd closed after one rename
    can come back as the next tmp file's fd), so classifying by fd number
    alone would misattribute a file fsync to a directory or mix two
    sessions."""
    real_open, real_write = os.open, os.write
    real_fsync, real_close, real_replace = os.fsync, os.close, os.replace
    events: list[tuple] = []
    tokens: dict[int, int] = {}
    next_token = [0]

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        next_token[0] += 1
        tokens[fd] = next_token[0]
        events.append(("open", fd, next_token[0], os.fspath(path)))
        return fd

    def fake_write(fd, data):
        events.append(("write", fd, tokens.get(fd)))
        return real_write(fd, data)

    def fake_fsync(fd):
        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        events.append(("fsync", fd, tokens.get(fd), is_dir))
        return real_fsync(fd)

    def fake_close(fd):
        events.append(("close", fd, tokens.pop(fd, None)))
        return real_close(fd)

    def fake_replace(src, dst, *args, **kwargs):
        events.append(("replace", os.fspath(src), os.fspath(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "write", fake_write)
    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "close", fake_close)
    monkeypatch.setattr(os, "replace", fake_replace)
    return events


def _first_replace_index(events: list[tuple], suffix: str) -> int | None:
    """Index of the first os.replace whose destination ends with *suffix*
    (".bak" = the backup rename, ".meta" = the meta rename)."""
    for i, event in enumerate(events):
        if event[0] == "replace" and event[2].endswith(suffix):
            return i
    return None


def _fd_sessions(events: list[tuple], fd: int, kind: str) -> dict:
    """Indexes of *kind* events for descriptor *fd*, grouped by open session
    token, so a reused fd number cannot mix two open sessions."""
    grouped: dict = {}
    for i, ev in enumerate(events):
        if ev[0] == kind and ev[1] == fd:
            grouped.setdefault(ev[2], []).append(i)
    return grouped


def _assert_fsynced_after_last_write_before_close(
    events: list[tuple], fd: int, what: str,
) -> None:
    """The durability ordering contract for one content descriptor: within
    each of its open sessions, at least one fsync AFTER the descriptor's
    last write and BEFORE it is closed."""
    writes = _fd_sessions(events, fd, "write")
    fsyncs = _fd_sessions(events, fd, "fsync")
    closes = _fd_sessions(events, fd, "close")
    assert writes, f"{what}: descriptor {fd} was never written ({events})"
    for token, write_idxs in writes.items():
        session = f"descriptor {fd} (open #{token})"
        session_fsyncs = fsyncs.get(token, [])
        assert session_fsyncs, f"{what}: {session} was never fsync'd ({events})"
        assert max(write_idxs) < min(session_fsyncs), (
            f"{what}: {session} fsync must follow the last write ({events})"
        )
        session_closes = closes.get(token, [])
        assert session_closes, f"{what}: {session} was never closed ({events})"
        assert min(session_fsyncs) < min(session_closes), (
            f"{what}: {session} fsync must precede close ({events})"
        )


def test_setitem_fsyncs_the_backup_fd_before_close_and_replace(
    tmp_path, monkeypatch,
):
    """B39 for the store: the backup's content fd is fsync'd after its last
    write and before it is closed and renamed. Without it a power cut can
    keep the main file's fsync'd new content while losing the just-written
    backup — that edit would have no undo step."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    store = BackupStore()
    events = _install_fsync_recorder(monkeypatch)

    store[str(tmp_path / "f.py")] = b"state one\n"

    bak_idx = _first_replace_index(events, ".bak")
    assert bak_idx is not None, f"no .bak replace observed: {events}"
    pre = events[:bak_idx]
    written_fds = {ev[1] for ev in pre if ev[0] == "write"}
    # Only the backup tmp descriptor is written before the backup rename
    # (the .meta write comes after it).
    assert len(written_fds) == 1, pre
    _assert_fsynced_after_last_write_before_close(
        pre, written_fds.pop(), "backup fd",
    )


def test_setitem_fsyncs_the_store_directory_after_the_replace(
    tmp_path, monkeypatch,
):
    """B39 for the store: after os.replace the store DIRECTORY is fsync'd so
    the rename itself is durable (same contract as _atomic_write) — the
    content fsync alone does not make the rename survive a power cut."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    store = BackupStore()
    events = _install_fsync_recorder(monkeypatch)

    store[str(tmp_path / "f.py")] = b"state one\n"

    bak_idx = _first_replace_index(events, ".bak")
    assert bak_idx is not None, f"no .bak replace observed: {events}"
    dir_fsyncs_after = [
        i for i, ev in enumerate(events)
        if ev[0] == "fsync" and ev[3] and i > bak_idx
    ]
    assert dir_fsyncs_after, (
        f"the store directory was never fsync'd after the rename: {events}"
    )


def test_setitem_tolerates_a_refusing_directory_fsync(tmp_path, monkeypatch):
    """B39 for the store: the directory fsync is best-effort — a filesystem
    that refuses it must not fail an otherwise good backup write (the
    content fd was already fsync'd), matching _atomic_write's contract."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    real_fsync = os.fsync

    def refusing_dir_fsync(fd):
        # fstat at fsync time: fd numbers are reused the instant a
        # descriptor is closed (the dir fd closed after one rename can come
        # back as the next tmp file's fd), so a set of once-dir fds would
        # misfile a later FILE fsync and break an otherwise good write.
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(45, "Operation not supported")  # e.g. some filesystems
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", refusing_dir_fsync)

    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"state one\n"  # must not raise

    # The write landed complete, meta included, despite the refusing
    # directory fsync.
    assert store._meta_path(str(target)).read_text(
        encoding="utf-8",
    ) == str(target)
    assert store.pop(str(target)) == b"state one\n"


def test_meta_write_is_fsynced_too(tmp_path, monkeypatch):
    """B39 for the store: the <hash>.meta write (the original path used for
    undo/diff display) is fsynced as well, through a real descriptor write
    path — and content + encoding stay exactly what the old
    ``Path.write_text(file_path, encoding="utf-8")`` produced."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    store = BackupStore()
    target = tmp_path / "f.py"
    events = _install_fsync_recorder(monkeypatch)

    store[str(target)] = b"state one\n"

    bak_idx = _first_replace_index(events, ".bak")
    assert bak_idx is not None, f"no .bak replace observed: {events}"
    post = events[bak_idx + 1:]
    written_fds = {ev[1] for ev in post if ev[0] == "write"}
    assert written_fds, (
        "no descriptor-level write observed after the backup rename: the "
        f".meta write bypasses the fsyncable write path ({post})"
    )
    for fd in written_fds:
        _assert_fsynced_after_last_write_before_close(post, fd, ".meta write")

    # Behavior identical: the meta still records the original path, utf-8.
    meta = store._meta_path(str(target))
    assert meta.exists()
    assert meta.read_text(encoding="utf-8") == str(target)


# ---------------------------------------------------------------------------
# Stale-temp cleanup: identify fastedit's OWN temps exactly, sweep only the
# residue of crashed runs, never anything else.
#
# tempfile.mkstemp names its temps with an 8-character random segment drawn
# from "abcdefghijklmnopqrstuvwxyz0123456789_" (verified:
# tempfile._RandomNameSequence().characters), so fastedit's shapes are:
#   * ``.{target}.{random8}.tmp``  — _atomic_write's temp next to its target
#   * ``{random8}.tmp``            — BackupStore's backup temp (no prefix)
# Anything that differs by even one character of shape is NOT fastedit's and
# must never be deleted. All residue below is aged with os.utime (no
# sleeps); a "fresh" temp has mtime-now, i.e. it may be a LIVE run's
# in-flight write.
# ---------------------------------------------------------------------------

_CRASH_AGE_SECS = 2 * 3600  # crashed-run residue is hours old


def _age(path: Path, *, secs: float = _CRASH_AGE_SECS,
         follow_symlinks: bool = True) -> None:
    """Backdate *path*'s mtime so it counts as crashed-run residue."""
    old = time.time() - secs
    os.utime(path, (old, old), follow_symlinks=follow_symlinks)


def test_crashed_residue_for_this_target_is_swept_on_the_next_write(
    tmp_path, caplog,
):
    """A SIGKILL mid-``_atomic_write`` leaves ``.{name}.{random}.tmp`` behind;
    the next write to the SAME target removes it before creating its own
    temp, logging the sweep (count + names) at WARNING — the deletion happened
    in a directory the USER owns."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    residue = tmp_path / ".f.py.ab12cd34.tmp"
    residue.write_bytes(b"half-written by a killed run")
    _age(residue)

    with caplog.at_level(logging.INFO, logger="fastedit.backup"):
        _atomic_write(target, "new\n")

    assert not residue.exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Swept 1" in message and ".f.py.ab12cd34.tmp" in message
        for message in messages
    ), messages


def test_user_tree_sweep_summary_is_warning_cli_visible(tmp_path, caplog):
    """Site A deletes a file in a directory the USER owns: the summary is a
    WARNING. A CLI run configures no logging, so WARNING is the first level
    that reaches stderr (logging.lastResort) — at INFO the deletion of a
    user-tree file would be invisible exactly where it matters most."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    residue = tmp_path / ".f.py.ab12cd34.tmp"
    residue.write_bytes(b"half-written by a killed run")
    _age(residue)

    with caplog.at_level(logging.INFO, logger="fastedit.backup"):
        _atomic_write(target, "new\n")

    assert not residue.exists()
    swept = [r for r in caplog.records if "Swept 1" in r.getMessage()]
    assert swept, [r.getMessage() for r in caplog.records]
    assert swept[0].levelno == logging.WARNING


def test_store_dir_sweep_summary_stays_info_internal_housekeeping(
    tmp_path, monkeypatch, caplog,
):
    """Site B sweeps fastedit's OWN backup store — internal housekeeping in a
    fastedit-owned directory: the summary stays INFO (observable on the MCP
    server's DEBUG stderr) and no WARNING is emitted for it."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(backup_dir))
    stale = backup_dir / "ab12cd34.tmp"
    stale.write_bytes(b"crashed backup temp")
    _age(stale)

    with caplog.at_level(logging.INFO, logger="fastedit.backup"):
        BackupStore()

    assert not stale.exists()
    swept = [r for r in caplog.records if "Swept 1" in r.getMessage()]
    assert swept, [r.getMessage() for r in caplog.records]
    assert swept[0].levelno == logging.INFO
    assert all(
        r.levelno < logging.WARNING
        for r in caplog.records
        if r.name == "fastedit.backup"
    )


def test_fresh_same_shape_temp_survives_the_sweep(tmp_path):
    """A same-shape temp with mtime NOW may be a LIVE concurrent fastedit's
    in-flight write; the age threshold must protect it even though the shape
    matches exactly."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    live = tmp_path / ".f.py.deadbeef.tmp"
    live.write_bytes(b"in-flight temp of a live run")

    _atomic_write(target, "new\n")

    assert live.exists()
    assert live.read_bytes() == b"in-flight temp of a live run"
    assert target.read_text(encoding="utf-8") == "new\n"


def test_near_miss_temp_names_survive_the_sweep(tmp_path):
    """The matcher is EXACT: a name only counts as fastedit's own temp on a
    full ``.{target}.{8 x [a-z0-9_]}.tmp`` match. A missing middle segment,
    a wrong-length middle, uppercase letters, a non-hidden spelling, a wrong
    suffix, .bak/.meta/.lock siblings, a DIRECTORY with a matching name, and
    a symlink with a matching name all survive — however old they are (every
    case below is aged, so age is not what spares them)."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")

    near_misses = [
        ".f.py.tmp",                # missing the random middle segment
        ".f.py.ab1.tmp",            # wrong-length middle (3 chars)
        ".f.py.ab12cd345.tmp",      # wrong-length middle (9 chars)
        ".f.py.AB12CD99.tmp",       # uppercase: tempfile's alphabet is [a-z0-9_]
        "f.py.ab12cd34.tmp",        # non-hidden variant (no leading dot)
        ".f.py.ab12cd34.bak",       # wrong suffix
        "f.py.bak",                 # a .bak sibling
        "f.py.meta",                # a .meta sibling
        "f.py.lock",                # a .lock sibling
        "a1b2c3d4e5f60718293a4b5c6d7e8f90.lock",  # lock-file shape
    ]
    for name in near_misses:
        p = tmp_path / name
        p.write_bytes(b"not fastedit's temp")
        _age(p)

    # A DIRECTORY named exactly like the pattern.
    directory_shaped = tmp_path / ".f.py.ab12cd34.tmp"
    directory_shaped.mkdir()
    (directory_shaped / "marker.txt").write_bytes(b"inside")
    _age(directory_shaped)

    # A SYMLINK named exactly like the pattern (points at a real file that
    # must never be followed or touched).
    decoy = tmp_path / "decoy.txt"
    decoy.write_bytes(b"decoy target")
    _age(decoy)
    symlink_shaped = tmp_path / ".f.py.deadbeef.tmp"
    os.symlink(decoy, symlink_shaped)
    _age(symlink_shaped, follow_symlinks=False)

    _atomic_write(target, "new\n")

    for name in near_misses:
        assert (tmp_path / name).exists(), name
    assert directory_shaped.is_dir()  # still a directory, contents intact
    assert (directory_shaped / "marker.txt").read_bytes() == b"inside"
    assert symlink_shaped.is_symlink()  # still a symlink, never followed
    assert decoy.read_bytes() == b"decoy target"
    assert target.read_text(encoding="utf-8") == "new\n"


def test_other_targets_temps_survive_this_targets_sweep(tmp_path):
    """A write to f.py sweeps only f.py-shaped residue: another target's
    temp (``.{other}.tmp`` shape) and a bare backups-style ``{8}.tmp`` in
    the same directory are not this target's temps and stay put."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    other = tmp_path / ".other.py.aa11bb22.tmp"
    other.write_bytes(b"another target's residue")
    _age(other)
    bare = tmp_path / "cc22dd33.tmp"  # bare form belongs to the backups dir
    bare.write_bytes(b"backups-dir shape, wrong directory")
    _age(bare)

    _atomic_write(target, "new\n")

    assert other.exists()
    assert other.read_bytes() == b"another target's residue"
    assert bare.exists()
    assert bare.read_bytes() == b"backups-dir shape, wrong directory"


def test_store_init_sweeps_stale_bare_temps_and_keeps_bak(tmp_path, monkeypatch):
    """BackupStore.__init__ sweeps bare ``{8 x [a-z0-9_]}.tmp`` residue in
    the store dir (crashed ``__setitem__``): aged temps go, fresh temps and
    the real .bak backups stay."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(backup_dir))

    stale = backup_dir / "ab12cd34.tmp"
    stale.write_bytes(b"crashed backup temp")
    _age(stale)
    fresh = backup_dir / "zz99xx11.tmp"  # mtime now: possibly a live run's
    fresh.write_bytes(b"in-flight backup temp")
    # AB12CD99, not AB12CD34: APFS is case-insensitive, so AB12CD34 would be
    # the SAME directory entry as the lowercase stale temp above.
    stale_upper = backup_dir / "AB12CD99.tmp"  # not tempfile's alphabet
    stale_upper.write_bytes(b"uppercase near-miss")
    _age(stale_upper)
    bak = backup_dir / "0123456789abcdef-00000000000000000042.bak"
    bak.write_bytes(b"real backup")

    store = BackupStore()

    assert store._dir == backup_dir
    assert not stale.exists()
    assert fresh.exists()
    assert fresh.read_bytes() == b"in-flight backup temp"
    assert stale_upper.exists()
    assert bak.exists()
    assert bak.read_bytes() == b"real backup"


def test_sweep_unlink_failure_is_swallowed_and_logged_not_fatal(
    tmp_path, monkeypatch, caplog,
):
    """A residue temp that cannot be removed (permission denied) must not
    break the write: the sweep logs the failure in the repo's error style
    and ``_atomic_write`` completes normally."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    stuck = tmp_path / ".f.py.ab12cd34.tmp"
    stuck.write_bytes(b"unremovable residue")
    _age(stuck)

    real_unlink = Path.unlink

    def refusing_unlink(self, missing_ok=False):
        if self.name == ".f.py.ab12cd34.tmp":
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refusing_unlink)

    with caplog.at_level(logging.INFO, logger="fastedit.backup"):
        _atomic_write(target, "new\n")  # must not raise

    assert target.read_text(encoding="utf-8") == "new\n"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "could not remove" in message.lower() and "ab12cd34" in message
        for message in messages
    ), messages


def test_write_still_lands_end_to_end_after_sweeping(tmp_path):
    """Sweeping never disturbs the write itself: the residue is gone, the
    new content is in place, and no temp of ours is left behind."""
    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    residue = tmp_path / ".f.py.ab12cd34.tmp"
    residue.write_bytes(b"half-written")
    _age(residue)

    _atomic_write(target, "new content\n", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new content\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.py"]


def test_non_utf8_path_backs_up_instead_of_crashing(tmp_path, monkeypatch):
    """A POSIX filename that is not valid UTF-8 (created on e.g. ext4, then
    handed to fastedit) arrives as a str carrying surrogates; the store must
    hash it (surrogateescape, like file_lock.lock_file_for) and store the
    backup + meta instead of raising UnicodeEncodeError mid-__setitem__ —
    which used to fail the caller's edit after the temp file was already
    created, and again via the meta write AFTER the backup had landed.

    The surrogate path is never created on disk here: APFS (macOS) refuses
    such filenames outright, but the STORE must not crash on the str
    regardless of which filesystem produced it.
    """
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "store"))
    store = BackupStore()

    surrogate_path = str(tmp_path / "caf\udce9.py")  # \udce9 = raw byte 0xE9
    store[surrogate_path] = b"two\n"

    # Backup stored and keyed by the surrogate path (the old _hash raised).
    assert surrogate_path in store
    assert store.pop(surrogate_path) == b"two\n"
    assert store._meta_path(surrogate_path).exists() is False  # consumed by pop

    store[surrogate_path] = b"four\n"
    # Meta content: the surrogateescape encoding of the original path —
    # written as pre-encoded bytes so the strict-utf-8 meta encode can no
    # longer fail after the backup is durable.
    meta = store._meta_path(surrogate_path)
    assert meta.read_bytes() == surrogate_path.encode("utf-8", "surrogateescape")


# ---------------------------------------------------------------------------
# GitHub issue #11: _atomic_write must preserve the target's permission mode.
#
# mkstemp creates the temp with 0o600 and os.replace keeps the TEMP's mode,
# so every rewrite used to silently drop the executable bit and group/other
# read bits from the file it rewrote. Existing targets keep their mode; a
# NEW target gets the standard creat default (0o666 & ~umask).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-mode semantics")
def test_atomic_write_preserves_mode_0755(tmp_path):
    """Issue #11 repro: rewriting a 0o755 script keeps 0o755 — the
    executable bit and group/other read survive the mkstemp+replace round
    trip instead of collapsing to 0o600."""
    target = tmp_path / "script.sh"
    target.write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(target, 0o755)

    _atomic_write(target, "#!/bin/sh\necho bye\n")

    assert target.read_text(encoding="utf-8") == "#!/bin/sh\necho bye\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-mode semantics")
def test_atomic_write_preserves_mode_0644(tmp_path):
    """A plain 0o644 source file keeps 0o644 across a rewrite."""
    target = tmp_path / "f.py"
    target.write_bytes(b"old = 1\n")
    os.chmod(target, 0o644)

    _atomic_write(target, "new = 2\n")

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-mode semantics")
def test_atomic_write_preserves_mode_even_under_restrictive_umask(tmp_path):
    """An existing target's mode comes from the FILE, never from the
    process umask: 0o755 stays 0o755 even while umask 0o077 is in force
    (the old 0o600 leak and a naive 0o644&~umask default would both fail)."""
    target = tmp_path / "script.sh"
    target.write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(target, 0o755)

    old_umask = os.umask(0o077)
    try:
        _atomic_write(target, "#!/bin/sh\necho bye\n")
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-mode semantics")
def test_atomic_write_new_file_gets_creat_default_not_mkstemp_0600(tmp_path):
    """A brand-new target gets the standard creat default — 0o666 & ~umask —
    not mkstemp's 0o600. Under umask 0o027 that is 0o640 (rw-r-----)."""
    target = tmp_path / "new.py"

    old_umask = os.umask(0o027)
    try:
        _atomic_write(target, "x = 1\n")
    finally:
        os.umask(old_umask)

    assert target.read_text(encoding="utf-8") == "x = 1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-mode semantics")
def test_atomic_write_new_file_default_follows_the_current_umask(tmp_path):
    """The new-file default tracks the umask in force at write time: under
    the default 0o022 the file lands 0o644 (world/group readable), which is
    what every standard tool (cat >, tee, touch) produces."""
    target = tmp_path / "new2.py"

    old_umask = os.umask(0o022)
    try:
        _atomic_write(target, "x = 1\n")
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
