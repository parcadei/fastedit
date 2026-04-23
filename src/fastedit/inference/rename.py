"""Pure rename logic: word-boundary symbol rename with tree-sitter skip zones.

Extracted from mcp/tools_ast.py so both the MCP server and CLI can use it
without any MCP dependency.
"""

from __future__ import annotations

import re

# Tree-sitter node types that represent literal text (not code).
# Matches in these ranges are skipped during rename to avoid
# renaming inside strings, comments, and docstrings.
_SKIP_NODE_TYPES = frozenset({
    # Comments (all languages)
    "comment", "line_comment", "block_comment",
    # String content (language-specific names for the text inside quotes)
    "string_content",                       # Python, Ruby, Rust
    "string_fragment",                      # JavaScript, TypeScript, Java
    "interpreted_string_literal_content",   # Go
})


def _collect_skip_ranges(
    node, ranges: list[tuple[int, int]],
) -> None:
    """Walk tree-sitter AST and collect byte ranges of strings/comments."""
    if node.type in _SKIP_NODE_TYPES:
        ranges.append((node.start_byte, node.end_byte))
        return  # Don't recurse into children (they're covered)
    for child in node.children:
        _collect_skip_ranges(child, ranges)


def _in_skip_zone(
    start: int, end: int, ranges: list[tuple[int, int]],
) -> bool:
    """Check if a byte range overlaps any skip zone."""
    for zone_start, zone_end in ranges:
        if start >= zone_start and end <= zone_end:
            return True
    return False


def do_rename(
    original: str,
    old_name: str,
    new_name: str,
    language: str | None = None,
) -> tuple[str, int, int]:
    """Rename all word-boundary occurrences of old_name to new_name.

    Uses tree-sitter to skip matches inside strings, comments, and docstrings.
    Falls back to plain regex if the language is not supported.

    Args:
        original: The original file content.
        old_name: The symbol name to find.
        new_name: The replacement name.
        language: Optional language identifier for tree-sitter parsing.

    Returns:
        (renamed_text, replacement_count, skipped_count)
    """
    pattern = re.compile(r"\b" + re.escape(old_name) + r"\b")

    # Build skip zones from tree-sitter (strings, comments, docstrings)
    skip_ranges: list[tuple[int, int]] = []
    if language:
        try:
            from ..data_gen.ast_analyzer import parse_code
            tree = parse_code(original, language)
            _collect_skip_ranges(tree.root_node, skip_ranges)
        except Exception:
            pass  # Fall back to unfiltered regex

    # Replace only matches outside skip zones
    original_bytes = original.encode("utf-8")
    parts: list[bytes] = []
    last_end = 0
    count = 0

    for m in pattern.finditer(original):
        # Convert char offsets to byte offsets for tree-sitter comparison
        byte_start = len(original[:m.start()].encode("utf-8"))
        byte_end = len(original[:m.end()].encode("utf-8"))

        if _in_skip_zone(byte_start, byte_end, skip_ranges):
            continue

        parts.append(original_bytes[last_end:byte_start])
        parts.append(new_name.encode("utf-8"))
        last_end = byte_end
        count += 1

    parts.append(original_bytes[last_end:])
    renamed = b"".join(parts).decode("utf-8")

    skipped = len(list(pattern.finditer(original))) - count
    return renamed, count, skipped

# ---------------------------------------------------------------------------
# Cross-file rename
# ---------------------------------------------------------------------------

# Default directory names to skip when walking. Rename passes that walk into
# vendor trees are almost always a bug, and the performance cost of re-renaming
# a million vendored files is real, so we prune aggressively. Users who need
# to rename inside these can point rename-all directly at the subdirectory.
DEFAULT_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".jj",
    "node_modules", ".npm", "bower_components", ".yarn",
    "__pycache__", ".venv", "venv", "env", ".env",
    ".mypy_cache", ".pytest_cache", ".tox", ".ruff_cache", ".hypothesis",
    "target", ".cargo",
    "build", "dist", "out", ".next", ".nuxt", ".output",
    "vendor", "Godeps",
    ".gradle", ".idea", ".vscode",
    "coverage", ".coverage", "htmlcov",
})

# Valid values for kind_filter. These map to tldr's `definition.kind` field
# for the symbol being renamed. When set, the rename only applies if the
# resolved definition matches — catches same-name collisions like a class
# Foo vs a local variable foo.
_VALID_KIND_FILTERS = frozenset({"class", "function", "method", "variable"})

# tldr `references` output assigns `kind: "other"` with confidence 0.5 to
# string/comment substring hits. We already filter via `--min-confidence 0.9`,
# but keep these as defensive client-side filters in case tldr's confidence
# scoring changes across versions.
_REFERENCE_KINDS_TO_RENAME = frozenset({
    "call", "read", "write", "import", "type", "definition",
})


def _iter_code_files(
    root: "Path",
    supported_exts: set[str],
    ignore_dirs: set[str],
):
    """Yield Path objects for files under root whose suffix is in supported_exts.

    Prunes ignore_dirs at every directory level without descending into them.
    Also prunes any directory containing a Python venv marker (pyvenv.cfg),
    which catches non-standard venv names like .venv311, myenv, virtualenv
    that aren't in DEFAULT_IGNORE_DIRS. Uses os.walk so we can mutate
    dirnames in place (Path.rglob does not). followlinks is False so
    symlinks aren't descended — prevents double-rename when a project has
    a symlink pointing back into its own tree.
    """
    import os
    from pathlib import Path

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        # Prune by basename and by PEP 405 venv marker.
        dirnames[:] = [
            d for d in dirnames
            if d not in ignore_dirs and not (dp / d / "pyvenv.cfg").exists()
        ]
        for name in filenames:
            if Path(name).suffix.lower() in supported_exts:
                yield dp / name


def _extract_json_object(text: str) -> str:
    """Return the first balanced {...} object in text, or '' if none.

    tldr's references command can print trailing 'No references found.' /
    'Suggestions:' lines after the JSON on stdout when there are zero hits.
    We want just the first top-level JSON object.
    """
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _run_tldr_references(old_name: str, root: "Path") -> dict:
    """Invoke `tldr references` and return the parsed JSON payload.

    Returns an empty payload ({"references": []}) on any failure — tldr
    missing, timeout, non-zero exit, or unparseable output. The rename then
    becomes a no-op rather than crashing the caller.
    """
    import json
    import subprocess

    cmd = [
        "tldr", "references", old_name, str(root),
        "--format", "json",
        "--min-confidence", "0.9",
        "--limit", "10000",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"references": []}

    payload = _extract_json_object(result.stdout or "")
    if not payload:
        return {"references": []}
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return {"references": []}
    if not isinstance(data, dict):
        return {"references": []}
    data.setdefault("references", [])
    return data


def _char_col_to_byte_offset(line_bytes: bytes, char_col_1_indexed: int) -> int:
    """Convert tldr's 1-indexed character column to a 0-indexed byte offset
    within the given line bytes. Columns beyond the line length clamp to end.
    """
    if char_col_1_indexed <= 1:
        return 0
    try:
        line_text = line_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # ASCII fallback: char col == byte col.
        return min(char_col_1_indexed - 1, len(line_bytes))
    # Character index = col - 1, clamped.
    char_idx = min(char_col_1_indexed - 1, len(line_text))
    return len(line_text[:char_idx].encode("utf-8"))


def _apply_refs_to_content(
    content: str, refs: list[dict], old_name: str, new_name: str,
) -> tuple[str, int]:
    """Apply a list of tldr reference hits to file content.

    Each ref is {line: int (1-indexed), column: int (1-indexed),
                 end_column: int, kind: str}. We rewrite right-to-left per
    line so earlier edits don't shift later column positions.

    Returns (new_content, replacement_count).
    """
    if not refs:
        return content, 0

    # Split preserving line endings. splitlines(keepends=True) round-trips.
    lines = content.splitlines(keepends=True)
    # Dedupe refs by (line, column) — tldr sometimes emits both a
    # 'definition' and a usage-kind row at the same position.
    seen: set[tuple[int, int]] = set()
    unique: list[dict] = []
    for r in refs:
        key = (r.get("line", 0), r.get("column", 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Bucket refs per line index (0-indexed).
    per_line: dict[int, list[dict]] = {}
    for r in unique:
        line_no = r.get("line", 0)
        if line_no < 1 or line_no > len(lines):
            continue
        per_line.setdefault(line_no - 1, []).append(r)

    count = 0
    old_bytes = old_name.encode("utf-8")
    new_bytes = new_name.encode("utf-8")

    for idx, line_refs in per_line.items():
        raw = lines[idx].encode("utf-8")
        # Sort by column descending so replacements don't invalidate earlier
        # column offsets.
        line_refs.sort(key=lambda r: r.get("column", 0), reverse=True)
        for ref in line_refs:
            col = ref.get("column", 0)
            end_col = ref.get("end_column", col + len(old_name))
            byte_start = _char_col_to_byte_offset(raw, col)
            byte_end = _char_col_to_byte_offset(raw, end_col)
            # Verify the slice we're about to replace actually equals
            # old_name. Guards against column-math surprises on exotic
            # unicode or tldr version skew.
            if raw[byte_start:byte_end] != old_bytes:
                continue
            raw = raw[:byte_start] + new_bytes + raw[byte_end:]
            count += 1
        lines[idx] = raw.decode("utf-8")

    return "".join(lines), count


def do_cross_file_rename(
    root_dir,
    old_name: str,
    new_name: str,
    supported_exts: set[str] | None = None,
    ignore_dirs: set[str] | None = None,
    kind_filter: str | None = None,
) -> dict:
    """Rename old_name -> new_name across all supported code files under root_dir.

    Drives edits from `tldr references <old_name> <root> --format json`, which
    uses tree-sitter + language-aware name resolution to distinguish real
    references from coincidental substrings in strings, comments, and
    docstrings. The previous text-based word-boundary engine is no longer the
    primary match source — it used regex plus a tree-sitter skip-zone filter
    and could not tell a variable `foo` from a class `Foo` at the same name.

    Does NOT write any files — returns a plan that callers can apply
    atomically or preview (dry-run).

    Args:
        root_dir: Path (or str) to walk.
        old_name: Symbol name to find.
        new_name: Replacement name.
        supported_exts: File suffixes to consider (.py/.ts/...). Defaults to
            fastedit's EXTENSION_TO_LANGUAGE keys.
        ignore_dirs: Directory basenames to skip. Defaults to
            DEFAULT_IGNORE_DIRS.
        kind_filter: When set, restrict the rename to targets whose
            tldr-resolved `definition.kind` matches. Valid values:
            'class', 'function', 'method', 'variable'. Semantics: this
            filters on the *definition's* kind (not per-reference usage kind)
            because tldr groups all refs under one resolved definition. An
            unmatched filter returns an empty plan — the caller can retry
            without the filter or pick a different kind.

    Returns:
        Dict mapping Path -> (new_content, replacement_count, skipped_count)
        for every file where replacement_count > 0. Files with zero matches
        are omitted. Binary / unreadable files are silently skipped. An
        empty dict is returned when old_name == new_name (no-op guard), when
        tldr is unavailable, or when kind_filter doesn't match.
    """
    from pathlib import Path

    from ..data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE

    if old_name == new_name:
        return {}

    if kind_filter is not None and kind_filter not in _VALID_KIND_FILTERS:
        raise ValueError(
            f"kind_filter must be one of {sorted(_VALID_KIND_FILTERS)} "
            f"or None, got {kind_filter!r}"
        )

    root = Path(root_dir)
    exts = supported_exts if supported_exts is not None else set(EXTENSION_TO_LANGUAGE.keys())
    ignore = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS

    # Walk once, keeping both a resolved->original map (so we can return the
    # caller-facing Path as the dict key, not the realpath — tmp_path on
    # macOS contains a /private prefix after resolve()) and an allowed set.
    # The intersection with tldr's output enforces fastedit's pruning rules
    # (pyvenv.cfg marker, symlink non-following, supported_exts filter) which
    # are stricter than tldr's built-in ignore list.
    resolved_to_original: dict[Path, Path] = {}
    try:
        for p in _iter_code_files(root, exts, ignore):
            try:
                resolved_to_original[p.resolve()] = p
            except OSError:
                resolved_to_original[p] = p
    except OSError:
        pass

    data = _run_tldr_references(old_name, root)

    # kind_filter applies to the definition's kind. When the filter doesn't
    # match the resolved definition, we return {} rather than silently
    # dropping refs — avoids accidentally renaming a same-name variable
    # when the caller asked for a class-only rename.
    if kind_filter is not None:
        definition = data.get("definition") or {}
        if definition.get("kind") != kind_filter:
            return {}

    # Group refs by resolved path, dropping any with disallowed kind.
    refs_by_file: dict[Path, list[dict]] = {}
    for ref in data.get("references") or []:
        kind = ref.get("kind")
        if kind not in _REFERENCE_KINDS_TO_RENAME:
            continue
        file_str = ref.get("file")
        if not file_str:
            continue
        try:
            resolved = Path(file_str).resolve()
        except (OSError, ValueError):
            continue
        if resolved not in resolved_to_original:
            continue
        refs_by_file.setdefault(resolved, []).append(ref)

    word_pattern = re.compile(r"\b" + re.escape(old_name) + r"\b")
    plan: dict = {}
    for resolved, refs in refs_by_file.items():
        original_path = resolved_to_original[resolved]
        try:
            original = original_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_content, count = _apply_refs_to_content(
            original, refs, old_name, new_name,
        )
        if count == 0:
            continue
        # 'skipped' is the number of raw word-boundary occurrences that tldr
        # did *not* verify as a reference — i.e. hits that live inside
        # strings, comments, or docstrings. Computed client-side because
        # tldr's output doesn't carry a per-file skip count.
        raw_hits = len(word_pattern.findall(original))
        skipped = max(0, raw_hits - count)
        plan[original_path] = (new_content, count, skipped)
    return plan
