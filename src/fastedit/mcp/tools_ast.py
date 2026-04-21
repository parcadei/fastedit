"""MCP tools for AST operations: fast_delete, fast_move, fast_rename, fast_undo."""

from __future__ import annotations

import difflib
from pathlib import Path

from ..data_gen.ast_analyzer import detect_language
from ..inference.chunked_merge import delete_symbol, move_symbol
from ..inference.rename import do_rename
from .server import _atomic_write, mcp


@mcp.tool(
    description=(
        "Delete a function, method, or class from a source file by name. "
        "Uses AST analysis to find the exact line range — no model inference, "
        "instant and 100% accurate. Supports all major languages. "
        "Use this instead of fast_edit when removing entire symbols."
    ),
)
async def fast_delete(file_path: str, symbol: str) -> str:
    """Remove a function, method, or class from a file using AST analysis."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    language = detect_language(path)

    async with file_locks[file_path]:
        try:
            result = delete_symbol(
                file_path=file_path,
                symbol=symbol,
                language=language,
            )
        except ValueError as e:
            return f"Error: {e}"

        if language and not result.parse_valid:
            _atomic_write(path, result.merged_code, backups=backups)
            return (
                f"Warning: parse errors after deleting {result.deleted_kind} "
                f"'{result.deleted_symbol}' from {file_path}. "
                f"Removed L{result.deleted_lines[0]}-{result.deleted_lines[1]} "
                f"({result.lines_removed} lines). Wrote anyway. 0 model tokens."
            )

        _atomic_write(path, result.merged_code, backups=backups)
        return (
            f"Deleted {result.deleted_kind} '{result.deleted_symbol}' "
            f"from {file_path}. "
            f"Removed L{result.deleted_lines[0]}-{result.deleted_lines[1]} "
            f"({result.lines_removed} lines). 0 model tokens."
        )


@mcp.tool(
    description=(
        "Move a function, method, or class to after another symbol in the "
        "same file. Pure AST operation — no model inference, instant and "
        "deterministic. Use this for code reorganization and refactoring."
    ),
)
async def fast_move(file_path: str, symbol: str, after: str) -> str:
    """Move a symbol to after another symbol using AST analysis."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    language = detect_language(path)

    async with file_locks[file_path]:
        try:
            result = move_symbol(
                file_path=file_path,
                symbol=symbol,
                after=after,
                language=language,
            )
        except ValueError as e:
            return f"Error: {e}"

        if language and not result.parse_valid:
            _atomic_write(path, result.merged_code, backups=backups)
            return (
                f"Warning: parse errors after moving {result.moved_kind} "
                f"'{result.moved_symbol}' after '{result.after_symbol}' "
                f"in {file_path}. Wrote anyway. 0 model tokens."
            )

        _atomic_write(path, result.merged_code, backups=backups)
        return (
            f"Moved {result.moved_kind} '{result.moved_symbol}' "
            f"from L{result.from_lines[0]}-{result.from_lines[1]} "
            f"to after '{result.after_symbol}' "
            f"(now L{result.new_lines[0]}-{result.new_lines[1]}) "
            f"in {file_path}. 0 model tokens."
        )


@mcp.tool(
    description=(
        "Rename all occurrences of a symbol in a file. Uses word-boundary matching "
        "to rename functions, variables, classes, or parameters without affecting "
        "partial matches (renaming 'get' won't touch 'get_all'). Instant, no model."
    ),
)
async def fast_rename(file_path: str, old_name: str, new_name: str) -> str:
    """Rename all occurrences of a symbol in a file using word-boundary regex.

    Uses tree-sitter to skip matches inside strings, comments, and docstrings.
    Falls back to plain regex if the language isn't supported by tree-sitter.
    """
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    async with file_locks[file_path]:
        original = path.read_text(encoding="utf-8")
        language = detect_language(path)

        renamed, count, skipped = do_rename(original, old_name, new_name, language)

        if count == 0:
            return (
                f"Error: no code occurrences of '{old_name}' found in {file_path} "
                f"(word-boundary match, excluding strings/comments)."
            )

        _atomic_write(path, renamed, backups=backups)

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            renamed.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
        diff_text = "".join(diff)

        skip_note = f" (skipped {skipped} in strings/comments)" if skipped else ""

        return (
            f"Renamed '{old_name}' -> '{new_name}' in {file_path}: "
            f"{count} replacement(s).{skip_note} 0 model tokens.\n\n{diff_text}"
        )


@mcp.tool(
    description=(
        "Undo the last edit to a file. Restores the file to its state before "
        "the most recent fast_edit, fast_batch_edit, fast_delete, fast_move, "
        "or fast_rename operation. One level of undo per file. Instant, no model."
    ),
)
async def fast_undo(file_path: str) -> str:
    """Revert the last edit to a file. Backups persist to disk across restarts."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups = lc["backups"]
    file_locks: dict = lc["file_locks"]

    if file_path not in backups:
        return f"Error: no undo history for {file_path}. Nothing to revert."

    path = Path(file_path)

    async with file_locks[file_path]:
        backup_content = backups.pop(file_path)

        current = path.read_text(encoding="utf-8") if path.exists() else ""

        # Write backup WITHOUT passing backups — undo itself must not create
        # a backup-of-backup (no undo-of-undo).
        _atomic_write(path, backup_content)

        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            backup_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
        diff_text = "".join(diff)

        return f"Reverted {file_path} to previous state.\n\n{diff_text}"
