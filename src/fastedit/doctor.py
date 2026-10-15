"""Self-diagnostic for FastEdit installations.

Checks binaries on PATH, Python version, package version, optional backend
dependencies, model cache state, Claude Code MCP config sanity, and tldr
availability. Prints a tabular report and exits non-zero if any required
check fails.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK = "\033[32m✓\033[0m"
WARN = "\033[33m!\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[90m–\033[0m"


def _status(ok: bool, required: bool) -> str:
    if ok:
        return OK
    return FAIL if required else WARN


class Report:
    """Rows print immediately per section so headers and rows interleave."""

    def __init__(self) -> None:
        self.failed_required = False

    def add(self, check: str, status: str, detail: str, required: bool = False) -> None:
        print(f"  {status}  {check.ljust(24)}{detail}")
        if status == FAIL and required:
            self.failed_required = True

    def section(self, title: str) -> None:
        print(f"\n{title}")


def _check_binaries(r: Report) -> None:
    for name, required in [("fastedit", True), ("fastedit-mcp", False), ("fastedit-hook", False)]:
        path = shutil.which(name)
        if path:
            r.add(name, OK, path, required=required)
        else:
            r.add(name, _status(False, required), "not on PATH", required=required)


def _check_python(r: Report) -> None:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    r.add("python", _status(ok, required=True), f"{v.major}.{v.minor}.{v.micro}", required=True)


def _check_package_version(r: Report) -> None:
    from .update_check import _parse_version, get_version_info

    current, latest = get_version_info()
    if not current:
        r.add("fastedits package", FAIL, "not installed (running from source?)", required=True)
        return

    if not latest:
        r.add("fastedits package", OK, f"{current} (remote version check unavailable)")
        return

    # Version parsing is best-effort: any hiccup falls through to "up to date".
    with contextlib.suppress(Exception):
        if _parse_version(latest) > _parse_version(current):
            r.add(
                "fastedits package",
                WARN,
                f"{current} → {latest} available (uv tool upgrade fastedits)",
            )
            return

    r.add("fastedits package", OK, f"{current} (up to date)")


def _check_optional_dep(r: Report, label: str, module: str, extra: str) -> bool:
    try:
        importlib.import_module(module)
        r.add(label, OK, "importable")
        return True
    except ImportError:
        r.add(label, WARN, f"not installed (reinstall with [{extra}] extra)")
        return False


def _check_backend_deps(r: Report) -> None:
    has_mlx = _check_optional_dep(r, "mlx backend", "mlx", "mlx")
    has_vllm = _check_optional_dep(r, "vllm backend", "vllm", "vllm")
    _check_optional_dep(r, "mcp server (fastmcp)", "fastmcp", "mcp")

    if not (has_mlx or has_vllm):
        r.add(
            "local inference",
            WARN,
            "no backend installed — external LLM server mode only",
        )


def _check_model_cache(r: Report) -> None:
    try:
        from .model_download import DEFAULT_CACHE_DIR, MODELS
    except Exception as e:  # noqa: BLE001 -- diagnostics: report-and-continue, the doctor must never crash on an optional-dep import failure
        r.add("model cache", WARN, f"skipped ({e})")
        return

    env_path = os.environ.get("FASTEDIT_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and any(p.glob("*.safetensors")):
            r.add("model cache (env)", OK, f"{p} ({_dir_size(p)})")
        else:
            r.add("model cache (env)", FAIL, f"{p} — FASTEDIT_MODEL_PATH set but no weights")
        return

    found_any = False
    for name in MODELS:
        p = DEFAULT_CACHE_DIR / name
        if p.is_dir() and any(p.glob("*.safetensors")):
            r.add(f"model cache ({name})", OK, f"{p} ({_dir_size(p)})")
            found_any = True
    if not found_any:
        r.add(
            "model cache",
            WARN,
            f"no models in {DEFAULT_CACHE_DIR} — run `fastedit pull --model mlx-8bit` (or bf16)",
        )


def _dir_size(p: Path) -> str:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{total / 1024 / 1024 / 1024:.1f} GB" if total >= 1024**3 else f"{total / 1024 / 1024:.0f} MB"


def _check_mcp_config(r: Report) -> None:
    """Read Claude Code configs and flag stale fastedit MCP entries."""
    targets = [
        ("user config", Path.home() / ".claude.json"),
        ("project config (cwd)", Path.cwd() / ".mcp.json"),
    ]
    for label, path in targets:
        if not path.exists():
            r.add(label, SKIP, f"not found: {path}")
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 -- diagnostics: any read/parse failure is reported as WARN, never propagated (json can raise more than OSError/ValueError, e.g. RecursionError)
            r.add(label, WARN, f"{path} not parseable ({e})")
            continue

        fastedit = (data.get("mcpServers") or {}).get("fastedit")
        if not fastedit:
            r.add(label, SKIP, f"{path} — no fastedit entry (run `fastedit mcp-install`)")
            continue

        cmd = fastedit.get("command", "")
        args = fastedit.get("args", [])
        if cmd == "fastedit-mcp" and not args:
            r.add(label, OK, f"{path} — uses fastedit-mcp entry point")
        elif cmd and shutil.which(cmd):
            r.add(label, OK, f"{path} — command on PATH ({cmd})")
        elif cmd.startswith("/"):
            if Path(cmd).exists():
                r.add(label, OK, f"{path} — absolute path exists")
            else:
                r.add(label, FAIL, f"{path} — stale path: {cmd}", required=True)
        else:
            r.add(label, WARN, f"{path} — command may not resolve: {cmd}")


def _check_tldr(r: Report) -> None:
    path = shutil.which("tldr")
    if not path:
        r.add(
            "tldr",
            FAIL,
            "not on PATH — required for fast_read / fast_search / fast_edit guard",
            required=True,
        )
        return
    version = "unknown"
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            version = out.stdout.strip().split()[-1]
    except (subprocess.TimeoutExpired, OSError):
        pass
    r.add("tldr", OK, f"{path} (v{version})")


def _check_resource_hub(r: Report) -> None:
    """The global resource governor's limits + live holders (read-only).

    Diagnostics, never a gate: a malformed limits.json or an unreadable hub
    directory is a WARN row (the governor itself fails loud at acquisition
    time; here the report must survive any hub state). Holders are listed
    one per row — pid, slot, target file — so a stuck run can be identified
    (and killed) from this report alone.
    """
    from . import resource_hub

    try:
        limits = resource_hub.load_limits()
    except ValueError as e:
        r.add("limits", WARN, f"malformed limits: {e}")
        return
    r.add(
        "limits",
        OK,
        f"max_model_instances={limits.max_model_instances} "
        f"max_heavy_jobs={limits.max_heavy_jobs} "
        f"heavy_file_bytes={limits.heavy_file_bytes} "
        f"queue_wait={limits.slot_wait_timeout_s:g}s",
    )
    try:
        holders = resource_hub.read_hub_state()
    except OSError as e:
        r.add("hub state", WARN, f"unreadable hub dir: {e}")
        return
    if not holders:
        r.add("active holders", OK, "none")
        return
    for h in holders:
        file_note = f" file={h.get('file')}" if h.get("file") else ""
        r.add(
            f"{h['kind']}-{h['slot']}",
            OK,
            f"pid={h.get('pid')} since epoch {h.get('started_epoch')}{file_note}",
        )


def run_doctor() -> int:
    r = Report()

    print("\nfastedit doctor")
    r.section("binaries")
    _check_binaries(r)
    r.section("runtime")
    _check_python(r)
    _check_package_version(r)
    r.section("backends")
    _check_backend_deps(r)
    r.section("model")
    _check_model_cache(r)
    r.section("mcp config")
    _check_mcp_config(r)
    r.section("helpers")
    _check_tldr(r)
    r.section("resource hub")
    _check_resource_hub(r)

    print()
    if r.failed_required:
        print("Required checks failed — see ✗ rows above.")
        return 1
    print("All required checks passed.")
    return 0
