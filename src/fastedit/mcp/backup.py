"""Standalone backup and atomic write utilities -- no MCP dependency.

BackupStore and _atomic_write are pure filesystem utilities used by both
the MCP server and the CLI. Extracted here so the CLI can import them
without pulling in the `mcp` package.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("fastedit.backup")


class BackupStore:
    """Disk-backed 1-deep undo store. Survives server restarts and Escape cancels.

    Backups live in ~/.fastedit/backups/ as files named by a hash of the
    original path. On startup, any backups older than 24h are pruned.
    Only the most recent backup per file is kept (1-deep).
    """

    _MAX_AGE_SECS = 86400  # 24 hours

    def __init__(self):
        self._dir = Path.home() / ".fastedit" / "backups"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prune_old()

    def _key_path(self, file_path: str) -> Path:
        """Stable filename from the original file path."""
        import hashlib
        h = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return self._dir / f"{h}.bak"

    def _meta_path(self, file_path: str) -> Path:
        import hashlib
        h = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return self._dir / f"{h}.meta"

    def __setitem__(self, file_path: str, content: str) -> None:
        key = self._key_path(file_path)
        # Write backup atomically
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, key)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        # Write metadata (original path) so we can display it
        self._meta_path(file_path).write_text(file_path, encoding="utf-8")

    def __contains__(self, file_path: str) -> bool:
        return self._key_path(file_path).exists()

    def pop(self, file_path: str) -> str:
        key = self._key_path(file_path)
        if not key.exists():
            raise KeyError(file_path)
        content = key.read_text(encoding="utf-8")
        key.unlink()
        meta = self._meta_path(file_path)
        if meta.exists():
            meta.unlink()
        return content

    def _prune_old(self) -> None:
        """Delete backups older than _MAX_AGE_SECS."""
        import time
        now = time.time()
        pruned = 0
        for p in self._dir.glob("*.bak"):
            if now - p.stat().st_mtime > self._MAX_AGE_SECS:
                p.unlink()
                meta = p.with_suffix(".meta")
                if meta.exists():
                    meta.unlink()
                pruned += 1
        if pruned:
            logger.info("Pruned %d stale backup(s) older than 24h", pruned)


def _atomic_write(path: Path, content: str, backups=None) -> None:
    """Write content to a file atomically via temp file + rename.

    Prevents corrupt files if the process is interrupted mid-write.

    If *backups* is provided and the file already exists, the current content
    is saved into backups before overwriting. This enables 1-deep undo via
    ``fast_undo``. Backups persist to disk (~/.fastedit/backups/).
    """
    if backups is not None and path.exists():
        backups[str(path)] = path.read_text(encoding="utf-8")
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        if not closed:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
