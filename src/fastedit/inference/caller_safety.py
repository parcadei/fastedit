"""Cross-file caller-safety checks for destructive operations (fast_delete).

When a caller asks to delete a symbol we first run `tldr references` at
project scope and refuse if other files still call / import the symbol.
The MCP tool and CLI both delegate here so the behavior stays consistent.

Fall-open policy (VAL-M2-002): if the `tldr` binary is missing or errors,
the check returns an empty list rather than failing closed. The caller
surfaces a "check skipped" note and proceeds. Deletion is a user-driven
action — we don't want infra issues to block it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Project-root markers, in priority order. First hit wins when walking up
# from the file being edited.
_PROJECT_ROOT_MARKERS = (".git", "pyproject.toml", "package.json")


def _find_project_root(file_path: Path) -> Path:
    """Walk up from `file_path` looking for a project-root marker.

    Markers: ``.git`` directory, ``pyproject.toml``, or ``package.json``.
    If none is found we fall back to the file's parent directory so the
    safety check still has a sensible scope rather than erroring out.
    """
    file_path = Path(file_path)
    for parent in [file_path.parent, *file_path.parents]:
        for marker in _PROJECT_ROOT_MARKERS:
            if (parent / marker).exists():
                return parent
    return file_path.parent


def _run_tldr_references_workspace(symbol: str, root: Path) -> dict:
    """Invoke `tldr references <symbol> <root>` at workspace scope.

    Returns the parsed JSON payload, or an empty {"references": []} on any
    failure (binary missing, timeout, non-zero exit, bad JSON). We mirror
    the fall-open semantics of ``rename._run_tldr_references`` — infra
    issues must not fail-close a destructive operation.
    """
    cmd = [
        "tldr", "references", symbol, str(root),
        "--format", "json",
        "--scope", "workspace",
        "--min-confidence", "0.9",
        "--limit", "10000",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"references": []}

    stdout = result.stdout or ""
    if not stdout.strip():
        return {"references": []}

    # tldr sometimes emits diagnostic lines before the JSON object. Strip
    # to the first '{' to be lenient.
    start = stdout.find("{")
    if start == -1:
        return {"references": []}
    try:
        data = json.loads(stdout[start:])
    except (json.JSONDecodeError, ValueError):
        return {"references": []}
    if not isinstance(data, dict):
        return {"references": []}
    data.setdefault("references", [])
    return data


def check_cross_file_callers(
    file_path: Path,
    symbol: str,
    project_root: Path,
) -> list[dict]:
    """Return references to `symbol` that live OUTSIDE `file_path`.

    References inside the file being edited are filtered out — they're
    about to go away along with the symbol itself, so they don't
    represent a caller-safety concern. The definition entry is also
    dropped; tldr may include it alongside the usage list.

    On infra failure (tldr missing/erroring) returns an empty list
    (fall-open).

    Each element is the tldr reference dict, augmented with 'file',
    'line', 'kind' keys at minimum.
    """
    file_path = Path(file_path).resolve()
    data = _run_tldr_references_workspace(symbol, project_root)

    out: list[dict] = []
    for ref in data.get("references") or []:
        ref_file = ref.get("file")
        if not ref_file:
            continue
        # Drop self-references: same file as the one being edited.
        try:
            if Path(ref_file).resolve() == file_path:
                continue
        except OSError:
            # Path resolution on weird paths: skip conservatively.
            continue
        # Drop the definition entry explicitly (belt and suspenders —
        # tldr usually exposes it via data["definition"], not as a ref,
        # but some versions emit a "definition"-kind row too).
        if ref.get("kind") == "definition":
            continue
        out.append(ref)
    return out


def format_refusal_message(
    symbol: str,
    refs: list[dict],
    force_hint: str,
) -> str:
    """Build the structured refusal message shown to users.

    `force_hint` is the caller-specific override hint (e.g.
    "Pass force=True (MCP) / --force (CLI) to delete anyway").
    Truncates the reference list at 10 lines.
    """
    files = {r.get("file") for r in refs if r.get("file")}
    lines = [
        f"Refused: '{symbol}' has {len(refs)} reference(s) "
        f"in {len(files)} other file(s):",
    ]
    shown = refs[:10]
    for r in shown:
        fp = r.get("file", "?")
        line_no = r.get("line", "?")
        kind = r.get("kind", "ref")
        lines.append(f"  {fp}:{line_no} [{kind}]")
    remaining = len(refs) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    lines.append(force_hint)
    return "\n".join(lines)
