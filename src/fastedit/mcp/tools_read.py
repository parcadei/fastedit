"""MCP tools for reading and searching: fast_read, fast_search, fast_diff."""

from __future__ import annotations

import difflib
import json
import subprocess
from pathlib import Path

from .server import mcp


@mcp.tool(
    description=(
        "Read a file's structure (functions, classes, methods with line ranges) "
        "without reading the full file content. Returns a compact map of symbols "
        "you can target with fast_edit (after/replace), fast_delete, or fast_move. "
        "Use this FIRST to understand a file before editing — saves input tokens "
        "vs reading the whole file. For small files (<100 lines), just read the file directly."
    ),
)
def fast_read(file_path: str) -> str:
    """Read a file's structure without reading the full content."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    content = path.read_text(encoding="utf-8", errors="replace")
    total_lines = content.count("\n") + (
        1 if content and not content.endswith("\n") else 0
    )

    # Small files: just return the content directly
    if total_lines <= 100:
        return f"{file_path} ({total_lines} lines — small file, showing full content)\n\n{content}"

    try:
        result = subprocess.run(
            ["tldr", "structure", file_path, "--format", "compact"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return f"Error: tldr structure failed for {file_path}: {result.stderr.strip()}"

        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return f"Error: tldr structure timed out for {file_path}"
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return f"Error: {e}"

    files = data.get("files", [])
    if not files:
        return f"{file_path} ({total_lines} lines) — no structure detected"

    file_info = files[0]
    language = data.get("language", "unknown")
    definitions = file_info.get("definitions", [])
    imports = file_info.get("imports", [])

    lines: list[str] = [f"{file_path} ({language}, {total_lines} lines)"]
    lines.append("")

    # Imports summary (deduplicated)
    if imports:
        seen: set[str] = set()
        import_names: list[str] = []
        for imp in imports:
            mod = imp.get("module", "")
            names = imp.get("names", [])
            label = f"{mod} ({', '.join(names)})" if names else mod
            if label not in seen:
                seen.add(label)
                import_names.append(label)
        lines.append(f"Imports: {', '.join(import_names)}")
        lines.append("")

    # Definitions with line ranges — grouped by kind for readability
    if definitions:
        # Track class membership for indentation
        class_ranges: list[tuple[int, int, str]] = []
        for d in definitions:
            if d.get("kind") == "class":
                class_ranges.append((d["line_start"], d["line_end"], d["name"]))

        for d in definitions:
            name = d.get("name", "?")
            kind = d.get("kind", "?")
            ls = d.get("line_start", 0)
            le = d.get("line_end", 0)
            sig = d.get("signature", "")

            # Indent methods that are inside a class
            indent = ""
            if kind == "method":
                for cs, ce, _cn in class_ranges:
                    if cs <= ls <= ce:
                        indent = "  "
                        break

            # Use signature if available, else name
            label = sig if sig else f"{kind} {name}"
            lines.append(f"{indent}L{ls}-{le:<4} {label}")

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Search codebase for functions, classes, and symbols. "
        "Modes: 'search' (keyword, default), 'regex' (pattern match), "
        "'hybrid' (keyword + regex_filter), 'references' (find usages). "
        "Returns compact results: name, file, line range, and signature. Instant, no model."
    ),
)
def fast_search(
    query: str,
    path: str = ".",
    mode: str = "search",
    top_k: int = 10,
    regex_filter: str = "",
) -> str:
    """Search codebase for functions, symbols, and references."""
    if mode == "references":
        try:
            result = subprocess.run(
                ["tldr", "references", query, path, "--format", "text", "--limit", str(top_k)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"Error: tldr references failed: {result.stderr.strip()}"
            return result.stdout.strip() or "No references found."
        except subprocess.TimeoutExpired:
            return "Error: references search timed out"
        except FileNotFoundError:
            return "Error: tldr not found on PATH"
    # search, regex, or hybrid mode — use --no-callgraph for speed
    cmd = ["tldr", "search", query, path, "--format", "text", "--top-k", str(top_k), "--no-callgraph"]
    if mode == "regex":
        cmd.append("--regex")
    elif mode == "hybrid" and regex_filter:
        cmd.extend(["--hybrid", regex_filter])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return f"Error: tldr search failed: {result.stderr.strip()}"
        return _compact_search(result.stdout.strip()) or "No results found."
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except FileNotFoundError:
        return "Error: tldr not found on PATH"

def _compact_search(text: str) -> str:
    """Strip code previews and callers/callees from search results.

    Keeps: header line, numbered results with signature.
    Drops: 'Called by:', 'Calls:', and '---' preview blocks.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_preview = False
    for line in lines:
        if line.startswith("   ---"):
            in_preview = True
            continue
        if in_preview:
            # Preview ends at next numbered result or blank line
            if line and (line[0].isdigit() or not line.startswith("   ")):
                in_preview = False
            else:
                continue
        stripped = line.strip()
        if stripped.startswith("Called by:") or stripped.startswith("Calls:"):
            continue
        out.append(line)
    return "\n".join(out).strip()


@mcp.tool(
    description=(
        "Show a unified diff of changes made by the last fast_edit or "
        "fast_batch_edit call on this file. Returns a compact diff — much "
        "cheaper than re-reading the whole file to verify an edit."
    ),
)
def fast_diff(file_path: str) -> str:
    """Show the diff between the pre-edit snapshot and current file."""
    ctx = mcp.get_context()
    snapshots: dict = ctx.request_context.lifespan_context["snapshots"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    if file_path not in snapshots:
        return f"No prior edit recorded for {file_path}. Call fast_edit first."

    original = snapshots[file_path]
    current = path.read_text(encoding="utf-8", errors="replace")

    if original == current:
        return f"No changes detected in {file_path}."

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    return "".join(diff) or f"No changes detected in {file_path}."
