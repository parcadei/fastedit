"""Shared test fixtures.

Isolates BackupStore's on-disk backup directory during the test session so
the suite never writes into (or prunes) the real per-user backup store.

Deliberately NOT done via monkeypatching HOME: several model-dependent
tests locate the local FastEdit model relative to the home directory
(see fastedit.model_download), and changing HOME mid-session can trigger
a ~1.7 GB model re-download. FASTEDIT_BACKUP_DIR is a dedicated override
consumed by BackupStore.__init__ instead, so only the backup store moves.

Also re-syncs the packaged agent skill copy at session start (see
_sync_packaged_skill_copy below).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Fixture re-exports: the real-LLM engine fixture LIVES in llm_fixtures.py
# (implementation plan Phase A) and the MCP full-stack harness fixture lives
# in test_stress_100mb_llm.py (Step C2, reused by C3's seams suite). Importing
# them here puts them in conftest's fixture namespace so every llm-tier test
# can request `real_engine`/`mcp_harness` by name without re-importing them
# (a per-module import would shadow-collide with the test functions' fixture
# parameters — ruff F811, and the same collision the C2 module hit).
from llm_fixtures import real_engine  # noqa: F401 -- fixture discovery re-export
from test_stress_100mb_llm import (
    mcp_harness,  # noqa: F401 -- fixture discovery re-export
)

# ---------------------------------------------------------------------------
# Packaged-skill auto-sync (runs at session start, before collection)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKILL_SOURCE = PROJECT_ROOT / "skills" / "fastedit" / "SKILL.md"
_SKILL_PACKAGED = PROJECT_ROOT / "src" / "fastedit" / "skill" / "SKILL.md"


def _sync_packaged_skill_copy() -> None:
    """Copy skills/fastedit/SKILL.md -> src/fastedit/skill/SKILL.md, one-way.

    The agent skill exists twice BY NECESSITY, not by accident:
      - skills/fastedit/SKILL.md is the single source of truth — the local
        directory the Vercel skills CLI reads (`npx skills add <repo>/skills/
        fastedit` is the measured-good install shape).
      - src/fastedit/skill/SKILL.md is package data: the copy shipped in the
        wheel and staged by `fastedit init`, read via importlib.resources.

    A build-time hatchling force-include cannot replace the second copy: the
    same importlib seam must keep working under EDITABLE installs (uv sync /
    uv run), and there importlib.resources anchors fastedit at src/ —
    measured in an editable sandbox install: the force-included file lands in
    site-packages, yet files('fastedit')/'skill'/'SKILL.md' resolves to
    src/fastedit, where it does not exist (is_file() == False).

    So the copy is refreshed here instead, on every test session, BEFORE any
    test runs: a desync can never survive a session, and a direct edit to the
    src/ copy is overwritten by the source of truth. In-sync sessions write
    nothing (byte compare first), so git status stays clean. If the source of
    truth is missing, do nothing — tests/test_fastedit_skill.py fails loudly
    naming the real problem.
    """
    if not _SKILL_SOURCE.is_file():
        return
    if _SKILL_PACKAGED.is_file() and _SKILL_PACKAGED.read_bytes() == _SKILL_SOURCE.read_bytes():
        return
    _SKILL_PACKAGED.parent.mkdir(parents=True, exist_ok=True)
    _SKILL_PACKAGED.write_bytes(_SKILL_SOURCE.read_bytes())


_sync_packaged_skill_copy()


@pytest.fixture(autouse=True, scope="session")
def _isolated_backup_store(tmp_path_factory):
    """Point every BackupStore at a session-scoped tmp dir, not the real one."""
    # The suite sets FASTEDIT_BACKUP_DIR so tests never write into or prune
    # the real per-user backup store at ~/.fastedit/backups.
    backup_root = tmp_path_factory.mktemp("fastedit-backups")
    previous = os.environ.get("FASTEDIT_BACKUP_DIR")
    os.environ["FASTEDIT_BACKUP_DIR"] = str(backup_root)
    yield backup_root
    if previous is None:
        del os.environ["FASTEDIT_BACKUP_DIR"]
    else:
        os.environ["FASTEDIT_BACKUP_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolated_lock_dir(tmp_path_factory):
    """Point every cross-process edit lock at a session-scoped tmp dir.

    The suite sets FASTEDIT_LOCK_DIR (the BackupStore-style override consumed
    by fastedit.file_lock) so lock contention tests exercise a real central
    directory shared with spawned subprocess holders, while the real
    per-user lock store at ~/.fastedit/locks is never touched.
    """
    lock_root = tmp_path_factory.mktemp("fastedit-lockdir")
    previous = os.environ.get("FASTEDIT_LOCK_DIR")
    os.environ["FASTEDIT_LOCK_DIR"] = str(lock_root)
    yield lock_root
    if previous is None:
        del os.environ["FASTEDIT_LOCK_DIR"]
    else:
        os.environ["FASTEDIT_LOCK_DIR"] = previous
