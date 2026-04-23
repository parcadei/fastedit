"""One-liner MCP config installer for Claude Code.

Writes or updates the fastedit entry in `~/.claude.json` (user scope) or
`$(pwd)/.mcp.json` (project scope). Safe on fresh systems (creates the file
and the `mcpServers` block if missing) and idempotent on existing ones
(detects a matching entry and skips). Backs up the existing config before
modifying.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


FASTEDIT_ENTRY = {
    "command": "fastedit-mcp",
    "type": "stdio",
}


def _target_path(scope: str) -> Path:
    if scope == "user":
        return Path.home() / ".claude.json"
    if scope == "project":
        return Path.cwd() / ".mcp.json"
    raise ValueError(f"unknown scope: {scope!r}")


def install_mcp_config(scope: str = "user") -> int:
    path = _target_path(scope)

    if path.exists():
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
            return 1
        if not isinstance(config, dict):
            print(f"error: {path} is not a JSON object", file=sys.stderr)
            return 1
    else:
        config = {}

    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        print(f"error: {path} has non-object 'mcpServers' field", file=sys.stderr)
        return 1

    existing = servers.get("fastedit")
    if existing == FASTEDIT_ENTRY:
        print(f"{path} — already configured, nothing to do")
        return 0

    if existing:
        print(f"{path} — existing fastedit entry:")
        print(json.dumps(existing, indent=2))
        print("  → replacing with fastedit-mcp entry point.")

    if shutil.which("fastedit-mcp") is None:
        print(
            "warning: `fastedit-mcp` is not on PATH — installing config anyway, "
            "but Claude Code will fail to launch the server until you "
            "`uv tool install 'fastedits[mcp]'` (or add the mcp extra to an "
            "existing install).",
            file=sys.stderr,
        )

    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(path.name + f".bak-{stamp}")
        shutil.copy(path, backup)
        print(f"backup: {backup}")

    servers["fastedit"] = FASTEDIT_ENTRY
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{path} — fastedit MCP entry installed")
    print("\nRestart Claude Code for the change to take effect.")
    return 0
