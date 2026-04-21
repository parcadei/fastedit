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
