"""Standalone backup and atomic write utilities -- no MCP dependency.

BackupStore and _atomic_write are pure filesystem utilities used by both
the MCP server and the CLI. Extracted here so the CLI can import them
without pulling in the `mcp` package.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import stat
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from ..io_utils import UnsupportedEncodingError, write_all

logger = logging.getLogger("fastedit.backup")

# B38: how many backups are kept per file. Every write stores a NEW
# timestamped backup and prunes the oldest beyond this depth, so a second
# edit never destroys the first backup and a corrupted edit can be undone
# step by step.
_MAX_BACKUPS_PER_FILE = 5

# Stale-temp cleanup: ``_atomic_write`` and ``BackupStore.__setitem__`` create
# hidden ".tmp" siblings (mkstemp) and unlink them on every exception path --
# but a SIGKILL or power cut skips all of those, leaving residue nobody owns.
# The sweep sites (``_atomic_write`` before it creates its own temp, and
# ``BackupStore.__init__`` for the store dir) may only delete files that are
# BOTH exactly fastedit's own mkstemp shape AND at least _TEMP_MIN_AGE_SECS
# old. The age threshold is deliberately conservative: a LIVE concurrent
# fastedit's in-flight temp is seconds old (mkstemp -> write -> fsync ->
# rename is milliseconds), so nothing under it can ever be swept mid-write;
# site A additionally runs inside the per-file edit lock, so no legitimate
# fastedit can hold an in-flight temp for that target anyway -- the threshold
# is what guards against pathological same-shape collisions from OTHER tools.
_TEMP_MIN_AGE_SECS = 60

# tempfile's random segment is EXACTLY 8 characters drawn from
# "abcdefghijklmnopqrstuvwxyz0123456789_" (tempfile._RandomNameSequence
# .characters -- never uppercase, never another length), verified against the
# CPython 3.x implementation. Both matchers below anchor on that with
# fullmatch, so a name counts as fastedit's own temp only on a FULL match: a
# missing/extra/longer random segment, uppercase letters, a different suffix,
# or a non-hidden spelling never matches, and .bak/.meta/.lock files cannot
# match at all.
_TEMP_RANDOM_SEGMENT = r"[a-z0-9_]{8}"
# Bare form: BackupStore's backup temp, tempfile.mkstemp(dir=..., suffix=
# ".tmp") with no prefix -> "<random8>.tmp" in the backups directory.
_BARE_TEMP_RE = re.compile(rf"{_TEMP_RANDOM_SEGMENT}\.tmp")


class ConcurrentModificationError(RuntimeError):
    """B37: the destination changed on disk between the caller's read and
    this write; the write was refused before the destination was touched.

    RuntimeError rather than ValueError/OSError: the content was never bad
    (not a value problem) and nothing failed at the OS level -- this is
    fastedit's own refusal, and callers catch it EXPLICITLY to turn it into
    a clean "re-read and retry" message. Nothing was written; the file on
    disk is exactly as the external writer left it.
    """


def _stale_temp_files(
    directory: Path, prefix: str | None = None,
) -> Iterator[Path]:
    """Yield fastedit's own STALE temp files in *directory* (no recursion).

    A file is yielded only when BOTH hold:

    * its name FULL-matches the hidden mkstemp shape this module creates --
      prefixed form ``.{prefix}.<random8>.tmp`` when *prefix* is given (the
      ``_atomic_write`` temp for a target named *prefix*), bare form
      ``<random8>.tmp`` otherwise (the ``BackupStore.__setitem__`` backup
      temp in the store dir) -- where ``<random8>`` is tempfile's 8-char
      ``[a-z0-9_]`` segment, and
    * it is at least ``_TEMP_MIN_AGE_SECS`` old, so a LIVE run's in-flight
      temp (seconds old) can never match.

    Everything else is skipped by construction: directories and symlinks
    (lstat semantics -- nothing is ever followed), vanished entries,
    near-miss names (never delete on a partial match), and young temps.
    Never recurses and never yields a path outside *directory*.
    """
    now = time.time()
    if prefix is None:
        matcher: re.Pattern[str] = _BARE_TEMP_RE
    else:
        matcher = re.compile(
            rf"\.{re.escape(prefix)}\.{_TEMP_RANDOM_SEGMENT}\.tmp",
        )
    try:
        with os.scandir(directory) as scan:
            entries = list(scan)
    except OSError as e:
        logger.info("Temp sweep: cannot list %s: %s", directory, e)
        return
    for entry in entries:
        try:
            # lstat semantics: a symlink to a regular file is still a
            # symlink here and is skipped, never followed.
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue  # vanished mid-sweep: a concurrent fastedit got it
        if not stat.S_ISREG(st.st_mode):
            continue
        if matcher.fullmatch(entry.name) is None:
            continue
        if now - st.st_mtime < _TEMP_MIN_AGE_SECS:
            continue  # seconds old: possibly a live run's in-flight temp
        yield Path(entry.path)


def _sweep_stale_temps(directory: Path, prefix: str | None = None) -> None:
    """Remove every stale fastedit temp :func:`_stale_temp_files` yields in
    *directory*, logging the sweep (count + names).

    The summary's level follows the SITE. The prefixed form (``_atomic_write``
    sweeping next to a real target) deletes files in a directory the USER
    owns, so it logs at WARNING -- the first level a CLI run with no logging
    configuration emits (``logging.lastResort``), keeping the deletion visible
    exactly where it matters most. The bare form (``BackupStore.__init__``
    sweeping its own store dir) is internal housekeeping in a fastedit-owned
    directory and stays at INFO.

    Best-effort by contract: an unlink failure (permissions, a concurrent
    removal) is logged and swallowed -- a cleanup sweep must never be the
    thing that fails an otherwise good write.
    """
    # The level is DERIVED from the site, not passed in: prefix is None
    # exactly when the sweep is the store dir's own housekeeping (the bare
    # mkstemp shape only ever occurs there), so the level cannot drift from
    # what is actually being swept.
    level = logging.INFO if prefix is None else logging.WARNING
    removed: list[str] = []
    for path in _stale_temp_files(directory, prefix=prefix):
        try:
            path.unlink()
        except FileNotFoundError:
            continue  # a concurrent fastedit removed it first
        except OSError as e:
            logger.info("Temp sweep: could not remove %s: %s", path, e)
            continue
        removed.append(path.name)
    if removed:
        logger.log(
            level,
            "Swept %d stale temp file(s) left by crashed run(s) in %s: %s",
            len(removed), directory, ", ".join(sorted(removed)),
        )


class BackupStore:
    """Disk-backed N-deep undo store (B38: ``_MAX_BACKUPS_PER_FILE`` per file).

    Survives server restarts and Escape cancels. Backups live in
    ~/.fastedit/backups/ named ``<sha256-16>-<timestamp>.bak``: a hash of the
    original path plus a nanosecond timestamp, so consecutive edits stack
    instead of overwriting each other (the old 1-deep store reused a single
    name per file and a second edit destroyed the first backup). One
    ``<sha256-16>.meta`` file per ORIGINAL path records that path for
    display; one meta per file (not per backup) is enough because the hash
    is stable per original path -- per-backup metas would duplicate the same
    string N times and need extra bookkeeping to stay in sync with pruning.

    Backups store RAW BYTES (B22): there is no decode or encode anywhere in
    the store, so any file's backup -- latin-1, BOM'd, whatever -- round-trips
    byte-exact. On startup, backups older than 24h are pruned; on every
    write, only the newest ``_MAX_BACKUPS_PER_FILE`` per file are kept.
    """

    _MAX_AGE_SECS = 86400  # 24 hours

    def __init__(self):
        override = os.environ.get("FASTEDIT_BACKUP_DIR")
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    "FASTEDIT_BACKUP_DIR must be an absolute path "
                    f"(a leading ~ is expanded); got: {override!r}"
                )
            self._dir = path
        else:
            self._dir = Path.home() / ".fastedit" / "backups"
        self._dir.mkdir(parents=True, exist_ok=True)
        # Sweep site B: crashed __setitem__ temps (bare "<random8>.tmp") have
        # no exception path left to clean them up. The store dir is
        # fastedit-owned and a live backup write from another process takes
        # milliseconds, so anything in the bare shape older than the
        # threshold is SIGKILL residue.
        _sweep_stale_temps(self._dir)
        self._prune_old()

    def _hash(self, file_path: str) -> str:
        """Stable hash prefix from the original file path.

        ``surrogateescape`` mirrors ``file_lock.lock_file_for``: a POSIX
        filename that is not valid UTF-8 arrives as a str carrying
        surrogates, and the old strict ``encode()`` raised
        UnicodeEncodeError mid-write (the backup could never be stored).
        Identical to the strict encoding for every valid-UTF-8 path, so
        existing backup hashes are unchanged.
        """
        return hashlib.sha256(
            file_path.encode("utf-8", "surrogateescape"),
        ).hexdigest()[:16]

    def _key_paths(self, file_path: str) -> list[Path]:
        """All backup files for *file_path*, NEWEST first. Timestamps are
        zero-padded to a fixed width, so lexicographic order is numeric."""
        prefix = self._hash(file_path)
        return sorted(
            self._dir.glob(f"{prefix}-*.bak"),
            key=lambda p: p.name,
            reverse=True,
        )

    def _meta_path(self, file_path: str) -> Path:
        return self._dir / f"{self._hash(file_path)}.meta"

    def __setitem__(self, file_path: str, data: bytes) -> None:
        """Store *data* as a NEW timestamped backup for *file_path*.

        *data* is RAW BYTES and is stored exactly as given (B22): callers
        pass what they read off disk or hold in memory; the store never
        decodes or encodes. Only the newest ``_MAX_BACKUPS_PER_FILE``
        backups per file are kept (B38).

        Durability (B39, the same contract as ``_atomic_write``): the
        backup content is fsync'd before close, and the store directory is
        fsync'd (best-effort) after the rename. Without it a power cut can
        keep the main file's fsync'd new content while losing the
        just-written backup -- that edit would have no undo step, since
        ``fastedit undo`` restores from backups, not from the main file.
        """
        ts = time.time_ns()
        target = self._dir / f"{self._hash(file_path)}-{ts:020d}.bak"
        while target.exists():
            # Same-nanosecond collision (clock adjustment, two processes):
            # bump the timestamp instead of silently overwriting an existing
            # backup -- that overwrite was the B38 failure mode.
            ts += 1
            target = self._dir / f"{self._hash(file_path)}-{ts:020d}.bak"
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        closed = False
        try:
            write_all(fd, data)
            # B39: fsync the content BEFORE close/replace -- the rename
            # alone does not make the bytes durable.
            os.fsync(fd)
            closed = True
            os.close(fd)
            os.replace(tmp, target)  # atomic on POSIX
            _fsync_directory(self._dir)  # B39: durable rename (best-effort)
        except BaseException:
            if not closed:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        # Metadata (original path) so undo/diff can display it. Funneled
        # through _atomic_write (B39) so the meta is fsync'd too: same
        # content as the old ``write_text(file_path, encoding="utf-8")``,
        # plus a temp+replace write so a crash cannot leave a half-written
        # meta. Written as PRE-ENCODED bytes with surrogateescape (the
        # _hash convention): a non-UTF-8 path must not fail the edit AFTER
        # its backup has already been stored.
        _atomic_write(
            self._meta_path(file_path),
            file_path.encode("utf-8", "surrogateescape"),
        )
        self._prune_per_file(file_path)

    def __contains__(self, file_path: str) -> bool:
        return bool(self._key_paths(file_path))

    def peek(self, file_path: str) -> bytes:
        """Return the NEWEST backup for *file_path* WITHOUT removing it.

        Used by the diff commands, which must look at the last backup
        without consuming the undo history. Raises KeyError when no backup
        exists.
        """
        paths = self._key_paths(file_path)
        if not paths:
            raise KeyError(file_path)
        return paths[0].read_bytes()

    def oldest(self, file_path: str) -> bytes:
        """Return the OLDEST backup for *file_path* WITHOUT removing it.

        The diff commands diff against this base (issue #8): peek's
        NEWEST backup is the LAST edit's pre-state, so a line an EARLIER
        edit had dropped was identical on both sides of that base and
        vanished from the rendered diff as soon as any later edit landed
        — `fastedit diff` showed only the newest intended hunk. The
        oldest surviving backup is the pre-state of the whole still-
        undoable change set, so every change in the undo history is
        surfaced (undo pops walk the same chain one step at a time).
        Raises KeyError when no backup exists.
        """
        paths = self._key_paths(file_path)
        if not paths:
            raise KeyError(file_path)
        return paths[-1].read_bytes()

    def pop(self, file_path: str) -> bytes:
        """Return and remove the NEWEST backup for *file_path* (undo
        semantics: one pop = one step back), as the RAW BYTES stored (B22).

        Raises KeyError when no backup exists.
        """
        paths = self._key_paths(file_path)
        if not paths:
            raise KeyError(file_path)
        newest = paths[0]
        data = newest.read_bytes()
        newest.unlink()
        if len(paths) == 1:
            # Last backup consumed: the file has no undo history left, so
            # its meta goes too (matching __contains__ going False).
            self._meta_path(file_path).unlink(missing_ok=True)
        return data

    def _prune_per_file(self, file_path: str) -> None:
        """Keep only the newest _MAX_BACKUPS_PER_FILE backups (B38)."""
        for stale in self._key_paths(file_path)[_MAX_BACKUPS_PER_FILE:]:
            with contextlib.suppress(OSError):
                stale.unlink()

    def _prune_old(self) -> None:
        """Delete backups older than _MAX_AGE_SECS, and each affected
        file's meta once its last backup is gone."""
        now = time.time()
        pruned = 0
        stale_hashes: set[str] = set()
        for p in self._dir.glob("*.bak"):
            try:
                if now - p.stat().st_mtime > self._MAX_AGE_SECS:
                    p.unlink()
                    # Backup names are "<hash>-<timestamp>.bak": the hash
                    # prefix groups a file's backups for meta cleanup.
                    stale_hashes.add(p.name.split("-")[0])
                    pruned += 1
            except FileNotFoundError:
                # A concurrent fastedit run or parallel test already removed
                # this .bak between our glob() listing and stat()/unlink().
                continue
        for h in stale_hashes:
            if not any(self._dir.glob(f"{h}-*.bak")):
                (self._dir / f"{h}.meta").unlink(missing_ok=True)
        if pruned:
            logger.info("Pruned %d stale backup(s) older than 24h", pruned)


def _refuse_if_changed_on_disk(
    path: Path, expected_stat: os.stat_result | None,
) -> None:
    """B37: raise when *path* no longer matches the stat captured at read
    time. Called immediately before ``os.replace`` -- the narrowest window
    available without kernel-level locking. Size or ``st_mtime_ns`` differing
    means something else wrote the file in the read-to-write window; the
    documented residual TOCTOU is a same-size write landing between this
    stat and the rename itself."""
    if expected_stat is None:
        return
    try:
        current = os.stat(path)
    except FileNotFoundError as e:
        raise ConcurrentModificationError(
            f"{path}: file changed on disk since it was read (it no longer "
            f"exists); re-read and retry. Nothing was written."
        ) from e
    if (current.st_size, current.st_mtime_ns) != (
        expected_stat.st_size, expected_stat.st_mtime_ns,
    ):
        raise ConcurrentModificationError(
            f"{path}: file changed on disk since it was read (size or mtime "
            f"differ); re-read and retry. Nothing was written."
        )


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync after ``os.replace`` (B39): makes the
    rename itself durable, not just the file content. Some filesystems
    refuse directory fsync and Windows cannot open one this way, so any
    OSError is swallowed -- the content is already fsync'd; this only
    hardens the rename."""
    try:
        dfd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_write(
    path: Path,
    content: str | bytes,
    backups=None,
    encoding: str = "utf-8",
    expected_stat: os.stat_result | None = None,
) -> None:
    """Write content to a file atomically via temp file + rename.

    Prevents corrupt files if the process is interrupted mid-write.

    If *backups* is provided and the file already exists, the current file's
    RAW BYTES are stored as a NEW timestamped backup before overwriting
    (B22: no decode anywhere in the backup path, so a latin-1 file's backup
    is byte-exact regardless of codec; B38: consecutive edits stack instead
    of overwriting each other). This enables N-deep undo via ``fast_undo`` /
    ``fastedit undo``. Backups persist to disk (~/.fastedit/backups/).

    *content* may be str (every caller except duplicate/undo) or bytes (used
    by ``fastedit duplicate`` for a byte-for-byte copy that never decodes
    the source, and by undo, which restores the stored raw backup bytes
    verbatim). str content is encoded with *encoding* (B23) -- the SAME
    codec that decoded the file, passed down by every edit verb from its
    ``read_source`` call, so untouched bytes round-trip exactly. The default
    remains "utf-8" for callers that have no read-time codec. bytes content
    is written as-is: no BOM restore, no re-encoding.

    A UTF-8 BOM is a file-level marker, not line content: every write verb
    funnels its final write through here, so restoring a BOM the on-disk
    file already had -- when *content* is str and did not already keep
    it -- is done once, here, rather than in every caller. This matters
    because a merge/replace/delete/move touching the symbol at the very
    top of the file naturally rewrites line 1 as ordinary text and has no
    reason to know a BOM was riding on it; str content is the only case
    this applies to (bytes content is never touched).
    A "-sig" *encoding* supplies its own BOM at encode time; a leading
    U+FEFF still present in the text would encode as a SECOND BOM, so it
    is stripped first (the codec restores the marker).

    *expected_stat* (B37 lost-update guard) is the ``os.stat_result``
    captured from the SAME open the caller read the file through -- see
    ``io_utils.read_source(..., return_stat=True)``. When provided, the
    destination is re-stat'ed immediately BEFORE ``os.replace``: if its size
    or ``st_mtime_ns`` no longer match -- something else wrote the file in
    the read-to-write window -- the write is refused with
    ``ConcurrentModificationError`` and the destination is untouched. A
    destination that no longer exists is a concurrent modification too
    (this write would otherwise resurrect it). ``None`` (the default) skips
    the guard, preserving the behavior for callers with no read-time stat
    (new files, undo).

    Before creating its own temp, crashed-run residue for THIS target --
    ``.{path.name}.<random8>.tmp`` files at least ``_TEMP_MIN_AGE_SECS`` old
    in the destination's directory -- is swept (best-effort, logged; see
    :func:`_sweep_stale_temps`). A SIGKILL bypasses every exception-path
    unlink, so this is the only cleanup those temps ever get.

    Permission mode (issue #11): an EXISTING destination keeps its current
    permission mode -- the executable bit and group/other read bits survive
    a rewrite instead of being silently dropped by os.replace keeping
    mkstemp's private 0o600 temp mode. A NEW destination gets the standard
    creat default, ``0o666 & ~umask``.

    Raises:
        UnsupportedEncodingError: str content that *encoding* cannot
            represent. Raised BEFORE any temp file is created and before
            anything is written; the file on disk is untouched.
        ConcurrentModificationError: *expected_stat* was provided and the
            destination changed on disk since it was captured. The temp
            file is cleaned up; the destination is untouched.
    """
    if backups is not None and path.exists():
        # B22: raw bytes -- the store never decodes, so the backup is
        # exactly what is on disk, whatever the codec.
        backups[str(path)] = path.read_bytes()
    if isinstance(content, bytes):
        data = content
    else:
        if encoding.endswith("-sig") and content.startswith("\ufeff"):
            content = content[1:]  # the codec adds the BOM; don't double it
        try:
            data = content.encode(encoding)
        except UnicodeEncodeError as e:
            raise UnsupportedEncodingError(
                f"{path}: content cannot be encoded with {encoding!r} ({e}); "
                f"the file was not modified."
            ) from e
    # Sweep site A: remove crashed-run residue for THIS target before
    # creating our own temp. Every call site holds the per-file edit lock,
    # so no legitimate concurrent fastedit can have an in-flight temp for
    # this target here; the matcher's exact shape plus _TEMP_MIN_AGE_SECS
    # keep any other tool's same-shape file safe regardless.
    _sweep_stale_temps(path.parent, prefix=path.name)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    closed = False
    try:
        if not isinstance(content, bytes) and path.exists():
            try:
                had_bom = path.read_bytes()[:3] == b"\xef\xbb\xbf"
            except OSError:
                had_bom = False
            if had_bom and not data.startswith(b"\xef\xbb\xbf"):
                data = b"\xef\xbb\xbf" + data
        write_all(fd, data)
        # B39: the content must survive a power cut, not just the rename.
        os.fsync(fd)
        closed = True
        os.close(fd)
        # B37: checked here, immediately before the rename, so the
        # read-to-write window is closed as tightly as userspace allows.
        _refuse_if_changed_on_disk(path, expected_stat)
        # Issue #11: mkstemp creates the temp with 0o600 and os.replace
        # keeps the TEMP's mode, so every rewrite used to silently drop the
        # target's executable bit and group/other read bits (0755 -> 0600,
        # 0644 -> 0600). Give the temp the mode the TARGET must end up
        # with, chmod'ed before the rename so mode and content land
        # together:
        #   * target exists  -> its own current mode (chmod before replace,
        #     so the rename lands the mode atomically with the content);
        #   * target missing -> the standard creat default, 0o666 & ~umask,
        #     exactly what `cat >`, `tee` or an editor would produce --
        #     never mkstemp's private 0o600.
        # The umask read is the portable os.umask(0)+restore dance; the
        # window where the process umask reads 0 is microseconds and every
        # caller of _atomic_write holds the per-file edit lock.
        try:
            target_mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mask = os.umask(0)
            os.umask(mask)
            target_mode = 0o666 & ~mask
        try:
            os.chmod(tmp, target_mode)
        except OSError as e:
            # Best-effort on platforms where chmod cannot express the mode:
            # the content write itself must never fail because of it.
            logger.warning(
                "Could not set mode %#o on temp for %s: %s", target_mode, path, e,
            )
        os.replace(tmp, path)  # atomic on POSIX
        _fsync_directory(path.parent)  # B39: durable rename (best-effort)
    except BaseException:
        if not closed:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
