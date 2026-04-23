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


def do_cross_file_rename(
    root_dir,
    old_name: str,
    new_name: str,
    supported_exts: set[str] | None = None,
    ignore_dirs: set[str] | None = None,
) -> dict:
    """Rename old_name -> new_name across all supported code files under root_dir.

    Uses word-boundary matching via do_rename, skipping strings/comments via
    tree-sitter per file. Does NOT write any files — returns a plan that
    callers can apply atomically or preview (dry-run).

    Args:
        root_dir: Path (or str) to walk.
        old_name: Symbol name to find.
        new_name: Replacement name.
        supported_exts: File suffixes to consider (.py/.ts/...). Defaults to
            fastedit's EXTENSION_TO_LANGUAGE keys.
        ignore_dirs: Directory basenames to skip. Defaults to DEFAULT_IGNORE_DIRS.

    Returns:
        Dict mapping Path -> (new_content, replacement_count, skipped_count)
        for every file where replacement_count > 0. Files with zero matches
        are omitted. Binary / unreadable files are silently skipped. An
        empty dict is returned when old_name == new_name (no-op guard).
    """
    from pathlib import Path

    from ..data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE, detect_language

    if old_name == new_name:
        return {}

    root = Path(root_dir)
    exts = supported_exts if supported_exts is not None else set(EXTENSION_TO_LANGUAGE.keys())
    ignore = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS

    plan: dict = {}
    for path in _iter_code_files(root, exts, ignore):
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if old_name not in original:
            continue
        language = detect_language(path)
        renamed, count, skipped = do_rename(original, old_name, new_name, language)
        if count > 0:
            plan[path] = (renamed, count, skipped)
    return plan
