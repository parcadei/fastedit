"""Cross-process edit locking — one fastedit instance per file at a time.

The gap this closes: every previous guard was IN-PROCESS. ``_atomic_write``'s
expected-stat guard (B37) and the MCP server's ``file_locks`` asyncio locks
serialize writers inside one process, but a CLI run and an MCP server (or two
CLIs) editing the same file interleave whole read-merge-write cycles, and the
second silently destroys the first's change. ``acquire_edit_lock`` is the
cross-process counterpart, held across each file's read→merge→write window.

WHY FLOCK, NOT O_EXCL LOCKFILES
    The lock is ``flock(2)`` (POSIX) / ``msvcrt.locking`` (Windows) on a
    central lock file — the KERNEL owns the lock, so it is released when the
    holder dies for any reason, crash included. There is NO stale-lock
    problem by construction, which is exactly what an O_EXCL create-step
    lockfile cannot promise: its creator's death strands the marker file
    forever, and every later run must guess whether the holder is alive.

NEVER UNLINK THE LOCK FILE
    Release is unlock + close ONLY. Deleting the file is the classic
    unlink race: a waiter blocked on the old inode and a fresh creator of a
    same-named file would both believe they hold the lock. The files are
    permanent, ever-reusable rendezvous points (one per target path,
    ``<sha256(realpath)>.lock`` under ``~/.fastedit/locks/`` — central, like
    BackupStore, and never inside the user's tree). Their content —
    ``pid=<pid>\\nstarted=<epoch>\\n`` — is a courtesy stamp for the refusal
    message, never the lock itself.

REENTRANCY REGISTRY
    flock excludes per open-file-description, so two fds of the SAME process
    conflict with each other — a nested acquire (multi-edit's phases, MCP +
    CLI sharing one process) would deadlock against itself. A process-level
    registry maps realpath → held record: a same-process re-acquire is a
    no-op returning the existing record, and only the OUTERMOST release
    unlocks. The whole check→flock→insert (and delete→unlock) sequence runs
    under one threading lock, atomically: fastedit both runs MCP tools on
    the event loop and drives merges through asyncio.to_thread workers, so
    two THREADS can race on one path — a racer that missed another's
    registry insert would open its OWN fd, fail the non-blocking flock
    against itself, and raise a false cross-process refusal naming its own
    pid.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from pathlib import Path

__all__ = [
    "EditFileLock",
    "FileLockedError",
    "acquire_edit_lock",
    "edit_lock_or_refusal",
    "lock_dir",
    "lock_file_for",
]

# Platform-guarded locking primitives: exactly one of these exists on a
# given platform (POSIX: fcntl; Windows: msvcrt).
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover — POSIX
    msvcrt = None  # type: ignore[assignment]


class FileLockedError(RuntimeError):
    """Another fastedit process holds the edit lock for the target file.

    The message is user-facing verbatim on both surfaces (CLI stderr after
    ``Error: `` and the MCP tools' ``Error: `` response): it names the
    holder pid and how long it has held the lock when the stamp is
    readable, always names the target path, and always ends with the
    "file unchanged" assurance. Nothing is read, merged, or written.
    """


class EditFileLock:
    """One held cross-process edit lock (per target path, per process).

    Yielded by :func:`acquire_edit_lock`; carries the target path and the
    lock file location for diagnostics. Release semantics live in
    :meth:`release`.
    """

    def __init__(self, target: str, lock_file: Path, fd: int):
        self.target = target
        self.lock_file = lock_file
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        """Drop one reentrancy level; unlock + close only at depth 0.

        The lock FILE is never unlinked (module docstring: the unlink race
        would let a waiter on the old inode and a fresh creator both
        believe they hold the lock). The kernel drops the flock when this
        process dies, so a leaked record is self-healing.

        The FINAL release — depth reset, registry removal, unlock, close —
        runs under ``_REGISTRY_LOCK``, the same lock :func:`_acquire_record`
        holds across its check→flock→insert. Split it, and an acquirer
        could observe the record already deleted while the kernel lock is
        still held (flock excludes per open-file-description even within
        one process) and false-refuse against its own pid. Every depth
        mutation is under the lock too, so concurrent acquire/release
        threads cannot lose an increment or double-unlock. The unlock is a
        non-blocking syscall, so the mutex is never held on a wait.
        """
        with _REGISTRY_LOCK:
            if self._depth > 1:
                self._depth -= 1
                return
            if self._depth <= 0:  # pragma: no cover — double-release guard
                return
            self._depth = 0
            if _REGISTRY.get(self.target) is self:
                del _REGISTRY[self.target]
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(self._fd)


# Process-level registry: realpath -> held record. Guarded by a threading
# lock because fastedit both runs MCP tools on the event loop and drives
# merges through asyncio.to_thread worker threads.
_REGISTRY: dict[str, EditFileLock] = {}
_REGISTRY_LOCK = threading.Lock()


def lock_dir() -> Path:
    """The central lock directory.

    ``FASTEDIT_LOCK_DIR`` relocates it (absolute path required, a leading
    ``~`` is expanded) — the same override pattern as BackupStore's
    ``FASTEDIT_BACKUP_DIR``, so tests and sandboxes can isolate it. Default:
    ``~/.fastedit/locks`` — central, like the backup store, never inside the
    user's tree.
    """
    override = os.environ.get("FASTEDIT_LOCK_DIR")
    if override:
        directory = Path(override).expanduser()
        if not directory.is_absolute():
            raise ValueError(
                "FASTEDIT_LOCK_DIR must be an absolute path "
                f"(a leading ~ is expanded); got: {override!r}"
            )
    else:
        directory = Path.home() / ".fastedit" / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def lock_file_for(path: Path | str) -> Path:
    """The lock file for *path*: ``<sha256(realpath)>.lock`` in lock_dir().

    realpath is the key, so spellings that resolve to the same file
    (symlinks, ``..`` segments) share one lock; surrogateescape keeps
    paths that are not valid UTF-8 hashable.
    """
    key = os.path.realpath(str(path))
    digest = hashlib.sha256(
        key.encode("utf-8", "surrogateescape"),
    ).hexdigest()
    return lock_dir() / f"{digest}.lock"


def _open_and_lock(lock_file: Path) -> int:
    """Open (or create) *lock_file* and take the kernel lock, non-blocking."""
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:
            # Windows byte-range locks refuse to lock a region the file does
            # not have, and a freshly created lock file is 0 bytes here (the
            # full holder stamp is only written AFTER the lock is taken).
            # Give the file its first byte BEFORE LK_NBLCK — and only when
            # it has none, so a conflicted acquirer never tramples the
            # holder's stamp. POSIX flock is per-fd and needs no bytes, so
            # the fcntl branch stays write-free.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\n")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover — neither primitive exists
            raise RuntimeError("no file-locking primitive on this platform")
    except OSError:
        os.close(fd)
        raise
    return fd


def _stamp_holder(fd: int) -> None:
    """Write the courtesy holder record (pid + start time) into the file."""
    payload = f"pid={os.getpid()}\nstarted={time.time():.6f}\n".encode()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        # Truncate AFTER the write, to exactly the payload length: a longer
        # stamp left by a previous holder loses its tail, and — the Windows
        # shape — the truncation never removes byte 0, where the msvcrt
        # byte-range lock lives. The old ftruncate(0)-first order would
        # delete the locked byte itself (a _chsize over a locked region can
        # fail or drop the lock; untestable on macOS, so the order that is
        # safe on both platforms runs unconditionally).
        os.ftruncate(fd, len(payload))
    except OSError:
        # The stamp is informational; the flock is the lock. A read-only
        # lock file must not fail an otherwise-successful acquisition.
        pass


def _read_holder(lock_file: Path) -> tuple[int | None, float | None]:
    """Best-effort (pid, started) from the lock file's stamp.

    Any failure — missing file, unreadable bytes, garbage, truncated
    values — degrades to (None, None): the refusal message then simply does
    not name a holder. Content is a courtesy; it must never be load-bearing.
    """
    try:
        raw = lock_file.read_bytes()[:512]
    except OSError:
        return None, None
    pid: int | None = None
    started: float | None = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("pid="):
            with contextlib.suppress(ValueError):
                pid = int(line[4:].strip())
        elif line.startswith("started="):
            with contextlib.suppress(ValueError):
                started = float(line[8:].strip())
    return pid, started


def _conflict_message(target: str, lock_file: Path) -> str:
    """The user-facing refusal: holder pid + age (when readable) + path."""
    pid, started = _read_holder(lock_file)
    if pid is not None and started is not None:
        age = max(time.time() - started, 0.0)
        holder = f" (pid {pid}, running {age:.1f}s)"
    elif pid is not None:
        holder = f" (pid {pid})"
    else:
        holder = ""
    return (
        f"another fastedit instance{holder} is editing {target}; "
        f"wait for it to finish — file unchanged"
    )


def _acquire_record(target: str, path: Path | str) -> EditFileLock:
    """Check-then-insert the reentrancy registry for *target*, ATOMICALLY.

    Everything — the registry check, the depth bump of an existing record,
    the open+flock of a fresh one, the insert, and the courtesy stamp —
    runs under ``_REGISTRY_LOCK``. It must: flock excludes per
    open-file-description even WITHIN one process, so a thread that missed
    another thread's insert would open its OWN fd, fail the non-blocking
    flock, and raise a false cross-process ``FileLockedError`` naming its
    own pid; two racers that both inserted would each hold a record whose
    release unlocks while a sibling still "holds". The critical section is
    microseconds (the kernel acquisition is non-blocking — it never waits),
    so the mutex never queues in practice.
    """
    with _REGISTRY_LOCK:
        held = _REGISTRY.get(target)
        if held is not None:
            held._depth += 1
            return held
        lock_file = lock_file_for(path)
        try:
            fd = _open_and_lock(lock_file)
        except OSError as e:
            raise FileLockedError(
                _conflict_message(str(path), lock_file),
            ) from e
        record = EditFileLock(target, lock_file, fd)
        _REGISTRY[target] = record
        _stamp_holder(fd)
        return record


@contextlib.contextmanager
def acquire_edit_lock(path: Path | str):
    """Hold *path*'s cross-process edit lock for a read→merge→write window.

    Non-blocking: if another PROCESS holds the lock, raises
    :class:`FileLockedError` immediately (the CLI turns that into
    ``Error: ...`` + exit 1; the MCP tools return it as their ``Error: ``
    response) — fastedit never queues behind another instance.

    Reentrant per process: acquiring the same path again in the SAME
    process returns the existing record (a no-op) — required because flock
    excludes per open-file-description and would otherwise make nested
    acquires self-conflict. Only the outermost release unlocks. The
    registry check→flock→insert is atomic across threads
    (:func:`_acquire_record`), so same-process threads racing on one path
    share ONE record with correct depth accounting instead of a racer
    falsely refusing against its own pid.
    """
    target = os.path.realpath(str(path))
    record = _acquire_record(target, path)
    try:
        yield record
    finally:
        record.release()


@contextlib.asynccontextmanager
async def edit_lock_or_refusal(path: Path | str):
    """MCP-tool guard: yield ``None`` when locked, else the refusal message.

    An async context manager so it can share the tools' existing
    ``async with file_locks[...]`` statement. The underlying acquisition is
    the synchronous non-blocking flock (microseconds; never waits), so
    calling it on the event loop is safe. The tools return strings, never
    raise at the user — the guard yields the raw :class:`FileLockedError`
    text (callers add their surface's own ``Error: `` prefix or embed it
    per-target) instead of raising. The ``acquired`` flag keeps a failure
    raised by the guarded BODY (which is not an acquisition conflict)
    propagating unchanged.
    """
    acquired = False
    try:
        with acquire_edit_lock(path):
            acquired = True
            yield None
    except FileLockedError as e:
        if acquired:
            raise  # the guarded body itself failed — not ours to convert
        yield str(e)
