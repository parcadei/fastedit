"""PyPI version check — surfaces new releases to callers.

Primary consumers:

* MCP server: checks on startup (async), attaches a one-line banner to the
  first successful ``fast_edit`` / ``fast_batch_edit`` response so the host
  LLM can relay the notice to the human operator. The model cannot run
  ``pip install``, so the notice is aimed at the user through the model.
* CLI: prints a stderr notice before exit when invoked by a human.

Behavior:

* Caches the result in ``~/.cache/fastedit/update-check.json`` for 24 hours
  so we don't hit PyPI on every call.
* Silent on any failure — network, parse error, missing cache dir. Never
  raises.
* Disabled when ``FASTEDIT_NO_UPDATE_CHECK=1`` is set.

Uses stdlib ``urllib`` — no new dependency.
"""
from __future__ import annotations

import json
import os
import time
from importlib import metadata
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/fastedits/json"
CACHE_TTL_S = 24 * 60 * 60  # 24 h
CACHE_PATH = Path.home() / ".cache" / "fastedit" / "update-check.json"
FETCH_TIMEOUT_S = 2.5


def _installed_version() -> str | None:
    try:
        return metadata.version("fastedits")
    except metadata.PackageNotFoundError:
        return None


def _parse_version(v: str) -> tuple[int, ...]:
    """PEP-440-ish tuple parse — enough to compare dotted release numbers.
    Anything non-numeric (e.g. ``0.2.6rc1``) collapses to the numeric prefix
    so ``0.2.6rc1`` < ``0.2.6``, which is the conservative direction."""
    parts: list[int] = []
    for chunk in v.split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        parts.append(int(num))
    return tuple(parts)


def _read_cache() -> dict | None:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(payload: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass


def _fetch_latest_from_pypi() -> str | None:
    try:
        from urllib.request import Request, urlopen

        req = Request(PYPI_URL, headers={"User-Agent": "fastedit-update-check"})
        with urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.load(resp)
        return data.get("info", {}).get("version")
    except Exception:
        return None


def get_update_notice() -> str | None:
    """Return a one-line notice string if a newer release exists, else None.

    Hits the cache first; only contacts PyPI when the cache is missing or
    older than ``CACHE_TTL_S``. Safe to call from sync code paths.
    """
    if os.environ.get("FASTEDIT_NO_UPDATE_CHECK") == "1":
        return None

    current = _installed_version()
    if not current:
        return None

    cached = _read_cache()
    now = int(time.time())
    latest: str | None = None

    if cached and now - cached.get("checked_at", 0) < CACHE_TTL_S:
        latest = cached.get("latest")
    else:
        latest = _fetch_latest_from_pypi()
        if latest:
            _write_cache({"latest": latest, "checked_at": now})
        elif cached:
            # Fall back to the stale cache if the fetch failed — better
            # than telling the user nothing.
            latest = cached.get("latest")

    if not latest:
        return None

    try:
        if _parse_version(latest) <= _parse_version(current):
            return None
    except Exception:
        return None

    return (
        f"[fastedit {latest} available — you're on {current}. "
        f"Run: pip install -U fastedits  (restart the MCP server after)]"
    )


async def get_update_notice_async() -> str | None:
    """Async-safe wrapper for :func:`get_update_notice`.

    Runs the (potentially blocking) PyPI fetch in a worker thread so the
    MCP server's event loop stays responsive during startup.
    """
    import asyncio

    return await asyncio.to_thread(get_update_notice)
