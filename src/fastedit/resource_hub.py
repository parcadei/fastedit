"""Global resource governor — cross-process slots coordinating ALL fastedit
instances through one central hub under ``~/.fastedit``.

THE PROBLEM
    fastedit spawns in-process MLX engines (multi-GB of RAM while loaded,
    seconds of 100%-CPU prefill per merge) and rewrites multi-MB files. A
    host agent that fires several `fastedit edit` runs at once — a CLI, an
    MCP server, a batch — launches one engine EACH and processes several big
    files CONCURRENTLY, with nothing coordinating the instances. Memory and
    CPU scale with the fan-out and there was no ceiling.

THE GOVERNOR
    Every model load / heavy file rewrite first acquires a SLOT from the
    hub: ``~/.fastedit/limits/model.slot.N`` (engine loads) and
    ``~/.fastedit/limits/heavy.slot.N`` (big-file windows), N < limit. The
    slots are ``flock(2)`` locks on permanent files — the same mechanism as
    :mod:`fastedit.file_lock`, so the kernel releases a dead holder's slot
    (crash-safe, no stale-slot problem) and the limit is global by
    construction: two processes contending for one hub directory serialize
    through the kernel.

    When every slot is held, the acquirer QUEUES: one stderr status line
    (``waiting for a model slot: 2/2 busy (pids 123, 456) — held 3s...``)
    and a 0.5 s poll that re-attempts each slot in order and re-reads the
    holders' pid stamps (so the wait line names live pids), until a slot
    frees or the ``slot_wait_timeout_s`` budget elapses → loud failure
    naming the remedy (raise the limit in ``~/.fastedit/limits.json``).

    Release is unlock + close ONLY — the slot file is never unlinked (the
    same unlink race file_lock.docstring documents). Acquisition is
    reentrant per process via a registry keyed (kind, realpath-of-slot-file):
    flock excludes per open-file-description, so a nested acquire of the
    SAME slot in one process would self-deadlock without it; only the
    outermost release unlocks.

OBSERVABILITY
    While held, each slot carries a JSON state file
    ``state/<kind>-<slot>.json`` ({pid, argv, file, started_epoch}) so
    ``fastedit doctor`` and ``install-dev.sh --check`` can report who holds
    what. ``read_hub_state()`` filters out (and prunes) holders whose pid is
    dead — a crashed holder's flock dies with it, but its state file would
    otherwise linger as a ghost.

LIMITS
    Defaults: max_model_instances=2, max_heavy_jobs=2, heavy_file_bytes=
    10_000_000, slot_wait_timeout_s=600. ``~/.fastedit/limits.json``
    overrides per key; ``FASTEDIT_MAX_MODEL_INSTANCES`` /
    ``FASTEDIT_MAX_HEAVY_JOBS`` / ``FASTEDIT_HEAVY_FILE_BYTES`` /
    ``FASTEDIT_SLOT_WAIT_TIMEOUT_S`` override the json (CLI > file, the
    file_lock/BackupStore env-override convention). Unknown keys, non-numeric
    values and out-of-range values fail LOUD (ValueError) — a typo'd limit
    must never silently degrade to a default.

WINDOWS
    The locking primitive is the same platform split file_lock uses (fcntl
    on POSIX, msvcrt byte-range locks on Windows). The hub dir override and
    all paths go through :class:`pathlib.Path` (``Path.home()`` works on
    both platforms); the two-process tests are skipped on Windows where the
    demo's fd semantics differ.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

__all__ = [
    "Limits",
    "SlotLease",
    "SlotWaitTimeout",
    "acquire_heavy_slot",
    "acquire_heavy_slot_for_path",
    "acquire_model_slot",
    "hub_dir",
    "is_heavy_file",
    "limits_dir",
    "limits_file",
    "load_limits",
    "read_hub_state",
    "state_dir",
]

# Platform-guarded locking primitives — the same split file_lock.py uses.
try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover — POSIX
    msvcrt = None  # type: ignore[assignment]

_MODEL = "model"
_HEAVY = "heavy"
_POLL_INTERVAL_S = 0.5

# Env override names, one per limit key (the file_lock/BackupStore
# FASTEDIT_*_DIR convention: the dedicated override, not HOME).
_ENV_KEYS = {
    "max_model_instances": "FASTEDIT_MAX_MODEL_INSTANCES",
    "max_heavy_jobs": "FASTEDIT_MAX_HEAVY_JOBS",
    "heavy_file_bytes": "FASTEDIT_HEAVY_FILE_BYTES",
    "slot_wait_timeout_s": "FASTEDIT_SLOT_WAIT_TIMEOUT_S",
}


class SlotWaitTimeout(TimeoutError):
    """Raised when no slot freed within ``slot_wait_timeout_s``.

    User-facing verbatim (the CLI prints it after ``Error: ``): names the
    kind, the budget and the remedy. Nothing was read, merged, or written
    by the failed acquisition.
    """


@dataclass(frozen=True)
class Limits:
    """The governor's knobs (see module docstring for precedence)."""

    max_model_instances: int
    max_heavy_jobs: int
    heavy_file_bytes: int
    slot_wait_timeout_s: float


_DEFAULT_LIMITS = Limits(
    max_model_instances=2,
    max_heavy_jobs=2,
    heavy_file_bytes=10_000_000,
    slot_wait_timeout_s=600.0,
)

# Field name -> (validator, error label). Declarative on purpose: adding a
# limit is one dataclass field, one env key, one row here — no new branches.
_INT_VALIDATORS = {
    "max_model_instances": (lambda v: v >= 1, "at least 1"),
    "max_heavy_jobs": (lambda v: v >= 1, "at least 1"),
    "heavy_file_bytes": (lambda v: v >= 0, "non-negative"),
    "slot_wait_timeout_s": (lambda v: v > 0, "positive"),
}


def hub_dir() -> Path:
    """The central hub root: ``~/.fastedit`` (``FASTEDIT_HUB_DIR`` overrides).

    The override must be absolute (a leading ``~`` is expanded) — the same
    rule ``file_lock.lock_dir`` applies, so a relative override can never
    silently relocate the hub into the current working directory. Created
    on use, like every fastedit central directory.
    """
    override = os.environ.get("FASTEDIT_HUB_DIR")
    if override:
        directory = Path(override).expanduser()
        if not directory.is_absolute():
            raise ValueError(
                "FASTEDIT_HUB_DIR must be an absolute path "
                f"(a leading ~ is expanded); got: {override!r}"
            )
    else:
        directory = Path.home() / ".fastedit"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def limits_dir() -> Path:
    """``<hub>/limits`` — slot files and limits.json live here (created)."""
    d = hub_dir() / "limits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir() -> Path:
    """``<hub>/state`` — per-holder JSON state files live here (created)."""
    d = hub_dir() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def limits_file() -> Path:
    """The ``limits.json`` path (the file itself is created only by the user)."""
    return limits_dir() / "limits.json"


def _coerce_limit(raw: str, field: str) -> int | float:
    """Parse one env var into a number, failing loud (never silent-default).

    Floats are accepted only for ``slot_wait_timeout_s`` (seconds); the
    count/byte fields must be integers — a float there would be a caller
    confusion worth refusing.
    """
    try:
        if field == "slot_wait_timeout_s":
            value = float(raw)
        else:
            value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{_ENV_KEYS[field]} must be a number, got {raw!r}"
        ) from e
    validator, requirement = _INT_VALIDATORS[field]
    if not validator(value):
        raise ValueError(
            f"{_ENV_KEYS[field]} must be {requirement}, got {value!r}"
        )
    return value


def load_limits() -> Limits:
    """Resolve the limits: defaults < limits.json < environment.

    Any malformed input — unparseable JSON, a non-object document, an
    unknown key, an out-of-range value, a malformed env var — raises
    ValueError naming the offending key. Fail loud: a limit that silently
    fell back to its default would defeat the whole governor.
    """
    values: dict[str, int | float] = {
        "max_model_instances": _DEFAULT_LIMITS.max_model_instances,
        "max_heavy_jobs": _DEFAULT_LIMITS.max_heavy_jobs,
        "heavy_file_bytes": _DEFAULT_LIMITS.heavy_file_bytes,
        "slot_wait_timeout_s": _DEFAULT_LIMITS.slot_wait_timeout_s,
    }

    path = limits_file()
    if path.exists():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"unreadable limits file {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"invalid JSON in limits file {path}: {e} — fix or remove "
                f"the file (keys: {', '.join(_DEFAULT_LIMITS.__dataclass_fields__)})"
            ) from e
        if not isinstance(document, dict):
            raise ValueError(
                f"limits file {path} must contain a JSON object, got "
                f"{type(document).__name__}"
            )
        known = set(_DEFAULT_LIMITS.__dataclass_fields__)
        for key, value in document.items():
            if key not in known:
                raise ValueError(
                    f"unknown limits key {key!r} in {path} (known keys: "
                    f"{', '.join(sorted(known))})"
                )
            if key == "slot_wait_timeout_s":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"limits key {key!r} must be a number, got {value!r}"
                    )
            else:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"limits key {key!r} must be an integer, got {value!r}"
                    )
            validator, requirement = _INT_VALIDATORS[key]
            if not validator(value):
                raise ValueError(
                    f"limits key {key!r} must be {requirement}, got {value!r}"
                )
            values[key] = value

    for field, env_name in _ENV_KEYS.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        values[field] = _coerce_limit(raw.strip(), field)

    return Limits(
        max_model_instances=int(values["max_model_instances"]),
        max_heavy_jobs=int(values["max_heavy_jobs"]),
        heavy_file_bytes=int(values["heavy_file_bytes"]),
        slot_wait_timeout_s=float(values["slot_wait_timeout_s"]),
    )


def is_heavy_file(path: Path | str) -> bool:
    """True when *path*'s size reaches ``heavy_file_bytes`` (missing → False).

    ``>=`` semantics: a file exactly AT the threshold is heavy, matching the
    spec's "≥ heavy_file_bytes". A missing/unstatable file is not heavy
    (nothing to process).
    """
    try:
        size = os.stat(path).st_size
    except OSError:
        return False
    return size >= load_limits().heavy_file_bytes


# ---------------------------------------------------------------------------
# Slot files + leases
# ---------------------------------------------------------------------------


class SlotLease:
    """One held slot: release drops one reentrancy level (unlock at 0).

    Mirrors :class:`file_lock.EditFileLock` semantics: never unlinks the
    slot file, prunes the state file at the OUTERMOST release only, and the
    final unlock+close runs under ``_REGISTRY_LOCK`` so a racing acquire
    thread can never observe a half-released record.
    """

    def __init__(self, kind: str, slot: int, slot_file: Path, fd: int):
        self.kind = kind
        self.slot = slot
        self.slot_file = slot_file
        self._fd = fd
        self._depth = 1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        with _REGISTRY_LOCK:
            if self._depth > 1:
                self._depth -= 1
                return
            if self._depth <= 0:  # pragma: no cover — double-release guard
                return
            self._depth = 0
            key = (self.kind, str(self.slot_file))
            if _REGISTRY.get(key) is self:
                del _REGISTRY[key]
            state_file = _state_file_for(self.kind, self.slot)
            try:
                state_file.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass  # observability only — never fail the release
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(self._fd)


# Process-level registry: (kind, slot-file-path) -> held lease. Guarded by a
# threading lock — fastedit runs MCP tools on the event loop AND merges on
# asyncio.to_thread workers, so two threads can race one slot.
_REGISTRY: dict[tuple[str, str], SlotLease] = {}
_REGISTRY_LOCK = threading.Lock()


def _slot_file(kind: str, slot: int) -> Path:
    return limits_dir() / f"{kind}.slot.{slot}"


def _state_file_for(kind: str, slot: int) -> Path:
    return state_dir() / f"{kind}-{slot}.json"


def _open_and_lock(slot_file: Path) -> int:
    """Open (or create) *slot_file* and take its kernel lock, non-blocking."""
    fd = os.open(slot_file, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:
            # Windows byte-range locks need the region to exist (see
            # file_lock._open_and_lock): give a fresh file its first byte
            # BEFORE LK_NBLCK, only when it has none, so a conflicted
            # acquirer never tramples the holder's stamp.
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
    """Write the courtesy pid/started stamp into the slot file."""
    payload = f"pid={os.getpid()}\nstarted={time.time():.6f}\n".encode()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        # Truncate AFTER the write (file_lock's order): a longer previous
        # stamp loses its tail, and byte 0 — where the msvcrt lock lives —
        # is never removed.
        os.ftruncate(fd, len(payload))
    except OSError:
        pass  # the stamp is informational; the flock is the lock


def _read_slot_pids(kind: str, max_slots: int) -> list[int]:
    """Best-effort holder pids of *kind*'s slots 0..max_slots-1, in order.

    Read from the slot files' pid stamps (refreshed by each waiting poll,
    so the queue line names the CURRENT holders). Missing/garbled stamps
    contribute nothing — the stamp is a courtesy, never load-bearing (the
    file_lock convention).
    """
    pids: list[int] = []
    for slot in range(max_slots):
        try:
            raw = _slot_file(kind, slot).read_bytes()[:512]
        except OSError:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if line.startswith("pid="):
                with contextlib.suppress(ValueError):
                    pids.append(int(line[4:].strip()))
                break
    return pids


def _write_state(kind: str, slot: int, file: str | None) -> None:
    """Write this holder's state file (best-effort — observability only)."""
    payload = {
        "pid": os.getpid(),
        "argv": sys.argv[:8] if sys.argv else [],
        "file": file,
        "started_epoch": time.time(),
        "kind": kind,
        "slot": slot,
    }
    try:
        _state_file_for(kind, slot).write_text(
            json.dumps(payload), encoding="utf-8",
        )
    except OSError:
        pass  # the state file is a courtesy; never fail the acquisition


def _try_each_slot(kind: str, count: int) -> SlotLease | None:
    """Attempt a non-blocking flock on each slot 0..count-1, in order.

    Returns the lease for the first free slot, or None when all are busy.
    A same-process reentrant acquire of an already-held slot is resolved by
    the registry BEFORE this loop (flock would self-conflict per-fd).
    """
    for slot in range(count):
        slot_file = _slot_file(kind, slot)
        try:
            fd = _open_and_lock(slot_file)
        except OSError:
            continue  # held by someone (or unusable) — try the next slot
        lease = SlotLease(kind, slot, slot_file, fd)
        _REGISTRY[(kind, str(slot_file))] = lease
        _stamp_holder(fd)
        return lease
    return None


def _status_line(kind: str, count: int, waited: float, pids: list[int]) -> str:
    """The one-line stderr queue status (spec wording)."""
    pid_note = f" (pids {', '.join(str(p) for p in pids)})" if pids else ""
    return (
        f"waiting for a {kind} slot: {count}/{count} busy{pid_note} — "
        f"held {waited:.0f}s..."
    )


def acquire_slot(kind: str, file: str | None = None) -> SlotLease:
    """Acquire one slot of *kind*, queueing (polling) when all are busy.

    Non-blocking first pass over every slot; when all are held, print the
    one-line queue status to stderr and poll every 0.5 s — re-attempting
    each slot in order and refreshing the holder pids — until a slot frees
    or ``slot_wait_timeout_s`` elapses (→ :class:`SlotWaitTimeout`, loud).
    Reentrant per process for the SAME slot file via the registry.

    *file* is recorded in the state file for observability (the target
    path of the edit that took the slot).
    """
    count = (
        load_limits().max_model_instances
        if kind == _MODEL
        else load_limits().max_heavy_jobs
    )
    # Reentrancy: any slot of this kind already held by THIS process → the
    # caller asked twice (stacked CLI+MCP windows); bump the existing
    # lease's depth. Distinct slot FILES stay distinct leases (a process
    # may hold heavy.slot.0 and heavy.slot.1 for different files).
    with _REGISTRY_LOCK:
        for (held_kind, _path), lease in _REGISTRY.items():
            if held_kind == kind:
                lease._depth += 1
                return lease

    deadline = time.monotonic() + load_limits().slot_wait_timeout_s
    waited = 0.0
    announced = False
    while True:
        lease = _try_each_slot(kind, count)
        if lease is not None:
            _write_state(kind, lease.slot, file)
            return lease
        if not announced:
            # _try_each_slot ran under the registry lock only for its own
            # bookkeeping; the pid read here is a courtesy snapshot.
            print(
                _status_line(
                    kind, count, waited, _read_slot_pids(kind, count),
                ),
                file=sys.stderr,
                flush=True,
            )
            announced = True
        if time.monotonic() >= deadline:
            raise SlotWaitTimeout(
                f"queued too long for a {kind} slot ({waited:.0f}s waited, "
                f"budget {load_limits().slot_wait_timeout_s:g}s) — try again "
                f"or raise limits in {limits_file()}."
            )
        time.sleep(_POLL_INTERVAL_S)
        waited += _POLL_INTERVAL_S
        # The queue line repeats at a slow cadence so a long wait stays
        # observable without spamming stderr (the agent/host sees liveness).
        if waited >= 30 and int(waited) % 30 == 0:
            print(
                _status_line(
                    kind, count, waited, _read_slot_pids(kind, count),
                ),
                file=sys.stderr,
                flush=True,
            )


@contextlib.contextmanager
def acquire_model_slot(file: str | None = None):
    """Context manager over :func:`acquire_slot` for the model kind."""
    lease = acquire_slot(_MODEL, file)
    try:
        yield lease
    finally:
        lease.release()


@contextlib.contextmanager
def acquire_heavy_slot(file: str | None = None):
    """Context manager over :func:`acquire_slot` for the heavy kind."""
    lease = acquire_slot(_HEAVY, file)
    try:
        yield lease
    finally:
        lease.release()


@contextlib.contextmanager
def acquire_heavy_slot_for_path(path: Path | str):
    """Heavy slot only when *path* is heavy; a no-op window otherwise.

    The single entry point CLI/MCP call sites use: small files (the vast
    majority) skip the hub entirely, big files serialize across ALL
    fastedit processes for their read→merge→write window.
    """
    if is_heavy_file(path):
        with acquire_heavy_slot(file=str(path)):
            yield
    else:
        yield


def read_hub_state() -> list[dict]:
    """The active holders across ALL processes: one dict per state file.

    Filters out (and prunes) holders whose pid is dead — ``os.kill(pid, 0)``
    succeeds or raises ESRCH for a live/dead pid; a crashed holder's slot
    flock died with it, so its state file is a ghost. Corrupt or unreadable
    state files are pruned too. Each entry carries the payload plus the
    parsed ``kind``/``slot`` from its filename (authoritative — the file
    name IS the slot identity).
    """
    holders: list[dict] = []
    directory = state_dir()
    for path in sorted(directory.glob("*.json")):
        stem = path.stem  # e.g. "model-0" / "heavy-1"
        try:
            kind, slot_text = stem.rsplit("-", 1)
            slot = int(slot_text)
        except ValueError:
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        if not isinstance(payload, dict):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        pid = payload.get("pid")
        alive = False
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True  # exists, owned by someone else
            except OSError:
                alive = True  # conservative: treat as alive
        if not alive:
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        holders.append({**payload, "kind": kind, "slot": slot})
    return holders


# Re-exported names used by doctor (keeps the doctor import surface small).
def _hub_report_rows() -> list[str]:
    """Human-readable hub rows for ``fastedit doctor`` (read-only)."""
    limits = load_limits()
    rows = [
        (
            f"limits: max_model_instances={limits.max_model_instances} "
            f"max_heavy_jobs={limits.max_heavy_jobs} "
            f"heavy_file_bytes={limits.heavy_file_bytes} "
            f"slot_wait_timeout_s={limits.slot_wait_timeout_s:g}"
        ),
        f"hub dir: {hub_dir()}",
    ]
    holders = read_hub_state()
    if holders:
        for h in holders:
            file_note = f" file={h.get('file')}" if h.get("file") else ""
            rows.append(
                f"active: {h['kind']}-{h['slot']} pid={h.get('pid')}"
                f"{file_note}"
            )
    else:
        rows.append("active holders: none")
    return rows
