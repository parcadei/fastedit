"""MCP tools for AST operations: fast_delete, fast_move, fast_rename, fast_rename_all, fast_undo."""

from __future__ import annotations

import difflib
from pathlib import Path

from ..data_gen.ast_analyzer import detect_language
from ..inference.chunked_merge import delete_symbol, move_symbol
from ..inference.rename import do_rename_ast
from .server import _atomic_write, mcp


@mcp.tool(
    description=(
        "Delete a function, method, or class from a source file by name. "
        "Uses AST analysis to find the exact line range — no model inference, "
        "instant and 100% accurate. Supports all major languages. "
        "Use this instead of fast_edit when removing entire symbols."
    ),
)
async def fast_delete(file_path: str, symbol: str, force: bool = False) -> str:
    """Remove a function, method, or class from a file using AST analysis.
    When ``force`` is False (default) the tool first runs `tldr references`
    at project scope. If the symbol is still called/imported from other
    files the delete is REFUSED with a structured message listing those
    callers (truncated at 10). Pass ``force=True`` to override, or run
    `fast_rename_all` to migrate callers first.
    If `tldr` is unavailable (missing binary, timeout, etc.) the check
    falls open — a note is appended to the success message and the delete
    proceeds. We don't fail-close on infra issues.
    """
    from ..inference.caller_safety import (
        _find_project_root,
        check_cross_file_callers,
        format_refusal_message,
    )
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    language = detect_language(path)

    async with file_locks[file_path]:
        # M2: cross-file caller-safety check. Skipped on force=True.
        if not force:
            project_root = _find_project_root(path)
            refs = check_cross_file_callers(
                file_path=path, symbol=symbol, project_root=project_root,
            )
            if refs:
                return format_refusal_message(
                    symbol,
                    refs,
                    "Pass force=True (MCP) / --force (CLI) to delete "
                    "anyway, or run fast_rename_all to migrate callers first.",
                )

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
        "Rename all AST-verified references to a symbol in a single file. "
        "Drives matching through `tldr references --scope file`, so the rename "
        "skips substrings inside strings, comments, and docstrings and never "
        "touches partial matches (renaming 'get' won't touch 'get_all'). "
        "Use fast_rename_all for cross-file renames. Instant, no model. "
        "Pass dry_run=True to preview without writing."
    ),
)
async def fast_rename(file_path: str, old_name: str, new_name: str, dry_run: bool = False) -> str:
    """Rename all AST-verified references to a symbol in a single file.

    Drives matching through ``tldr references <name> <file> --scope file``,
    so only real code references are renamed — substrings inside strings,
    comments, and docstrings are skipped. When tldr is unavailable the call
    becomes a no-op (count=0) rather than falling back to regex, matching
    the safety stance of fast_rename_all.

    Pass ``dry_run=True`` to preview what would change without writing any file.
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

        renamed, count, skipped = do_rename_ast(path, old_name, new_name)

        if count == 0:
            return (
                f"Error: no code references to '{old_name}' found in {file_path} "
                f"(AST-verified via tldr references --scope file; matches inside "
                f"strings/comments/docstrings are not counted)."
            )

        if dry_run:
            skip_note = (
                f" (skipping {skipped} in strings/comments)" if skipped else ""
            )
            return (
                f"Dry run: would rename '{old_name}' -> '{new_name}' in "
                f"1 file, {count} replacement(s){skip_note}: {file_path}"
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
        "Rename a symbol across every supported code file under a directory "
        "(cross-file rename). Uses word-boundary matching + tree-sitter per "
        "file to skip strings, comments, and docstrings. Prunes .git, "
        "node_modules, __pycache__, target, dist, vendor, and other common "
        "vendor/build dirs. Pass dry_run=True to preview which files would "
        "change without writing. Not scope-aware — renames every matching "
        "identifier, so unique names are safer than short common ones. For "
        "scope-aware refactors use an LSP-backed tool. Instant, no model."
    ),
)
async def fast_rename_all(
    root_dir: str,
    old_name: str,
    new_name: str,
    dry_run: bool = False,
    kind_filter: str | None = None,
) -> str:
    """Rename all occurrences of a symbol across a directory tree."""
    from ..inference.rename import do_cross_file_rename

    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    root = Path(root_dir)
    if not root.is_dir():
        return f"Error: directory not found: {root_dir}"

    plan = do_cross_file_rename(
        root, old_name, new_name, kind_filter=kind_filter,
    )
    if not plan:
        return (
            f"No occurrences of '{old_name}' found under {root_dir} "
            f"(word-boundary match, excluding strings/comments/vendor dirs)."
        )

    total_count = sum(count for _, count, _ in plan.values())
    total_skipped = sum(skipped for _, _, skipped in plan.values())

    if dry_run:
        lines = [
            f"Dry run: would rename '{old_name}' -> '{new_name}' in "
            f"{len(plan)} file(s), {total_count} replacement(s)"
            f"{f' (skipping {total_skipped} in strings/comments)' if total_skipped else ''}:",
            "",
        ]
        for path, (_, count, skipped) in sorted(plan.items()):
            skip_note = f" ({skipped} skipped)" if skipped else ""
            lines.append(f"  {path} — {count} replacement(s){skip_note}")
        return "\n".join(lines)

    # Apply. Lock each file individually so concurrent callers on unrelated
    # files don't serialize through a single global lock.
    for path, (new_content, _, _) in plan.items():
        async with file_locks[str(path)]:
            _atomic_write(path, new_content, backups=backups)

    skip_note = f" (skipped {total_skipped} in strings/comments)" if total_skipped else ""
    return (
        f"Renamed '{old_name}' -> '{new_name}' in {len(plan)} file(s), "
        f"{total_count} replacement(s).{skip_note} 0 model tokens."
    )


@mcp.tool(
    description=(
        "Undo the last edit to a file. Restores the file to its state before "
        "the most recent fast_edit, fast_batch_edit, fast_delete, fast_move, "
        "fast_rename, or fast_rename_all operation. One level of undo per file. Instant, no model."
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


@mcp.tool(
    description=(
        "Move a function, method, or class from one file to another and "
        "automatically rewrite `from X import symbol` / "
        "`import { symbol } from \"./X\"` statements in every dependent "
        "file. Uses tldr for importer discovery — instant, 0 model tokens. "
        "Use this for cross-file refactors. For same-file reorganisation "
        "use fast_move instead. Pass dry_run=True to preview the plan."
    ),
)
async def fast_move_to_file(
    symbol: str,
    from_file: str,
    to_file: str,
    after: str | None = None,
    dry_run: bool = False,
) -> str:
    """Move a symbol across files and rewrite every consumer's imports.

    Rejects same-file moves (hint: use fast_move). Rejects when the
    destination file already defines the symbol (conflict). Emits a plan
    message listing every importer it rewrote + a manual-review tail for
    cases the auto-rewriter can't handle (wildcard imports, re-exports,
    non-standard module specifiers).
    """
    from ..inference.caller_safety import _find_project_root
    from ..inference.move_to_file import move_to_file

    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    file_locks: dict = lc["file_locks"]

    from_path = Path(from_file)
    to_path = Path(to_file)

    if not from_path.exists():
        return f"Error: source file not found: {from_file}"
    if not to_path.exists():
        return f"Error: target file not found: {to_file}"

    project_root = _find_project_root(from_path)

    # Hold locks on BOTH files for the duration of the move so a
    # concurrent edit doesn't interleave with our two-file write.
    async with file_locks[from_file]:
        async with file_locks[to_file]:
            try:
                plan = move_to_file(
                    symbol=symbol,
                    from_file=str(from_path),
                    to_file=str(to_path),
                    after=after,
                    project_root=project_root,
                    dry_run=dry_run,
                )
            except ValueError as e:
                return f"Error: {e}"

    return plan.message
