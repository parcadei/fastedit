"""MCP tools for AST operations: fast_delete, fast_move, fast_rename, fast_rename_all, fast_undo."""

from __future__ import annotations

import asyncio
import difflib
import os
from pathlib import Path

from ..data_gen.ast_analyzer import detect_language
from ..file_lock import edit_lock_or_refusal
from ..inference.chunked_merge import delete_symbol, move_symbol
from ..inference.rename import do_rename_ast
from ..io_utils import UnsupportedEncodingError, read_source
from .server import ConcurrentModificationError, _atomic_write, mcp

_CONCURRENT_REFUSAL = (
    "file changed on disk since it was read; "
    "re-read and retry — file unchanged."
)


def _persist_ast(
    path: Path, content: str, *, backups, expected_stat: os.stat_result,
) -> str | None:
    """Write an AST verb's result with the B37 lost-update guard armed.

    Returns ``None`` on success, else the same clean refusal string the
    edit tools return (B37: nothing was written; the file on disk is
    exactly as the external writer left it).
    """
    try:
        _atomic_write(
            path, content, backups=backups, expected_stat=expected_stat,
        )
    except ConcurrentModificationError:
        return _CONCURRENT_REFUSAL
    return None

_UNDO_DIFF_LINE_BUDGET = 50_000
"""The embedded undo diff's input budget (lines per side).

``difflib``'s SequenceMatcher is quadratic on repetitive content: the D2
100MB txt stress measured ~3 hours inside this one diff for a 5-line
change (two ~1.8M-line template-heavy shift logs), while the revert
itself is a byte copy. The diff is a display convenience, never a gate —
the restore below is byte-exact regardless of size — so past the budget
the response carries an omitted-diff note instead of a full unified diff.
Small files (the overwhelming case, and every hermetic test) keep their
full diff.
"""


@mcp.tool(
    description=(
        "Delete a function, method, or class from a source file by name. "
        "Uses AST analysis to find the exact line range — no model inference, "
        "instant and 100% accurate. Supports all major languages. "
        "Runs a cross-file caller-safety check via tldr references first and "
        "REFUSES to delete if the symbol is still called/imported from other "
        "files; pass force=True to override. "
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

    # In-process asyncio lock + cross-process flock: a CLI `fastedit delete`
    # on the same file must be refused while this tool writes it.
    async with file_locks[file_path], edit_lock_or_refusal(path) as _lock_refusal:
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # M2: cross-file caller-safety check. Skipped on force=True. The
        # tldr subprocess (its own timeout) runs on a worker thread so the
        # event loop keeps serving other requests while it runs.
        if not force:
            project_root = _find_project_root(path)
            refs = await asyncio.to_thread(
                check_cross_file_callers,
                file_path=path, symbol=symbol, project_root=project_root,
            )
            if refs:
                return format_refusal_message(
                    symbol,
                    refs,
                    "Pass force=True (MCP) / --force (CLI) to delete "
                    "anyway, or run fast_rename_all to migrate callers first.",
                )

        # B37: stat captured immediately before the verb's own read of the
        # file, so the write below refuses when the file changed on disk in
        # the read-to-write window (fast_edit/batch/multi arm the same guard).
        try:
            read_stat = os.stat(path)
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        try:
            result = await asyncio.to_thread(
                delete_symbol,
                file_path=file_path,
                symbol=symbol,
                language=language,
            )
        except ValueError as e:
            return f"Error: {e}"

        if language and not result.parse_valid:
            error = _persist_ast(
                path, result.merged_code, backups=backups,
                expected_stat=read_stat,
            )
            if error:
                return error
            return (
                f"Warning: parse errors after deleting {result.deleted_kind} "
                f"'{result.deleted_symbol}' from {file_path}. "
                f"Removed L{result.deleted_lines[0]}-{result.deleted_lines[1]} "
                f"({result.lines_removed} lines). Wrote anyway. 0 model tokens."
            )

        error = _persist_ast(
            path, result.merged_code, backups=backups,
            expected_stat=read_stat,
        )
        if error:
            return error
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

    # In-process asyncio lock + cross-process flock (see fast_delete).
    async with file_locks[file_path], edit_lock_or_refusal(path) as _lock_refusal:
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # B37: stat captured immediately before the verb's own read of the
        # file (same guard as fast_delete/fast_edit).
        try:
            read_stat = os.stat(path)
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        try:
            result = await asyncio.to_thread(
                move_symbol,
                file_path=file_path,
                symbol=symbol,
                after=after,
                language=language,
            )
        except ValueError as e:
            return f"Error: {e}"

        if language and not result.parse_valid:
            error = _persist_ast(
                path, result.merged_code, backups=backups,
                expected_stat=read_stat,
            )
            if error:
                return error
            return (
                f"Warning: parse errors after moving {result.moved_kind} "
                f"'{result.moved_symbol}' after '{result.after_symbol}' "
                f"in {file_path}. Wrote anyway. 0 model tokens."
            )

        error = _persist_ast(
            path, result.merged_code, backups=backups,
            expected_stat=read_stat,
        )
        if error:
            return error
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

    # In-process asyncio lock + cross-process flock (see fast_delete).
    async with file_locks[file_path], edit_lock_or_refusal(path) as _lock_refusal:
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # B21: strict-decode read via read_source — a bare utf-8 read_text
        # raised an uncaught UnicodeDecodeError on a latin-1 (or UTF-16)
        # file instead of a clean refusal; read_source refuses those with
        # UnsupportedEncodingError. B37: the read-time stat guards the
        # write below. do_rename_ast re-reads the file itself with a
        # strict utf-8 decode, so a non-UTF-8 file yields count=0 (the
        # "no code references" response) and is never written.
        try:
            original, _encoding, read_stat = read_source(path, return_stat=True)
        except UnsupportedEncodingError as e:
            return f"Error: {e}"

        # tldr subprocess + tree walk: worker thread, off the event loop.
        renamed, count, skipped = await asyncio.to_thread(
            do_rename_ast, path, old_name, new_name,
        )

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

        # renamed is do_rename_ast's strict-utf-8 decode of the file
        # (a UTF-8 BOM rides on it and _atomic_write keeps it exact);
        # the default utf-8 codec is therefore the correct write codec.
        error = _persist_ast(
            path, renamed, backups=backups, expected_stat=read_stat,
        )
        if error:
            return error

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
        "(cross-file rename). AST-verified via tree-sitter per file — skips "
        "strings, comments, and docstrings structurally. Optional kind_filter "
        "('class'|'function'|'method'|'variable') narrows to targets whose "
        "definition kind matches (uses tldr for AST-verified lookup). Prunes "
        ".git, node_modules, __pycache__, target, dist, vendor, and other "
        "common vendor/build dirs. Pass dry_run=True to preview which files "
        "would change without writing. Not scope-aware — renames every "
        "matching identifier, so unique names are safer than short common "
        "ones. For scope-aware refactors use an LSP-backed tool. Instant, no "
        "model."
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

    # Directory walk + tldr subprocess: worker thread, off the event loop.
    plan = await asyncio.to_thread(
        do_cross_file_rename,
        root, old_name, new_name, kind_filter=kind_filter,
    )
    if not plan:
        return (
            f"No occurrences of '{old_name}' found under {root_dir} "
            f"(AST-verified via tldr — strings/comments/vendor dirs excluded)."
        )

    total_count = sum(count for _new_content, count, _skipped, _stat in plan.values())
    total_skipped = sum(skipped for _new_content, _count, skipped, _stat in plan.values())

    if dry_run:
        lines = [
            f"Dry run: would rename '{old_name}' -> '{new_name}' in "
            + f"{len(plan)} file(s), {total_count} replacement(s)"
            + f"{f' (skipping {total_skipped} in strings/comments)' if total_skipped else ''}:",
            "",
        ]
        for path, (_new_content, count, skipped, _stat) in sorted(plan.items()):
            skip_note = f" ({skipped} skipped)" if skipped else ""
            lines.append(f"  {path} — {count} replacement(s){skip_note}")
        return "\n".join(lines)

    # Apply. Lock each file individually so concurrent callers on unrelated
    # files don't serialize through a single global lock. A target held by
    # another fastedit PROCESS is skipped and named, not silently dropped.
    refused: list[str] = []
    for path, (new_content, _count, _skipped, _read_stat) in plan.items():
        async with (
            file_locks[str(path)],
            edit_lock_or_refusal(path) as _lock_refusal,
        ):
            if _lock_refusal:
                refused.append(f"{path}: {_lock_refusal}")
                continue
            # B37: _read_stat is the stat do_cross_file_rename captured when
            # it read this file for the plan. A non-fastedit write landing
            # between that read and this write is REFUSED here (not silently
            # overwritten from the plan's stale bytes); files already written
            # stay written, matching the verb's partial per-file semantics.
            # Same guard + refusal string as the other AST verbs.
            error = _persist_ast(
                path, new_content, backups=backups, expected_stat=_read_stat,
            )
            if error:
                refused.append(f"{path}: {error}")
                continue

    skip_note = f" (skipped {total_skipped} in strings/comments)" if total_skipped else ""
    refused_note = ""
    if refused:
        # Reason-neutral header: each entry below carries its own refusal —
        # a cross-process lock ("another fastedit instance (pid ...)") or a
        # B37 lost-update refusal ("file changed on disk since it was read").
        refused_note = (
            f"\n{len(refused)} file(s) NOT written:\n" + "\n".join(refused)
        )
    return (
        f"Renamed '{old_name}' -> '{new_name}' in {len(plan)} file(s), "
        f"{total_count} replacement(s).{skip_note} 0 model tokens."
        f"{refused_note}"
    )


@mcp.tool(
    description=(
        "Undo the last edit to a file. Restores the file to its state before "
        "the most recent fast_edit, fast_batch_edit, fast_delete, fast_move, "
        "fast_rename, or fast_rename_all operation. Walks back one step per "
        "call through the backups kept per file. Instant, no model."
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

    # The undo WRITES the file, so it takes the same cross-process lock as
    # the verbs that created the backup (in-process asyncio lock + flock).
    async with file_locks[file_path], edit_lock_or_refusal(path) as _lock_refusal:
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # B22/B38: backups are raw bytes; pop returns (and removes) the
        # NEWEST one. The restore below writes them back byte-for-byte
        # (bytes content skips _atomic_write's str/BOM path).
        backup_bytes = backups.pop(file_path)
        # Display-only decode (never written): utf-8 with replacement
        # characters, so a non-UTF-8 file's diff degrades gracefully.
        backup_content = backup_bytes.decode("utf-8", errors="replace")

        current = (
            path.read_text(encoding="utf-8", errors="replace")
            if path.exists() else ""
        )

        # Write backup WITHOUT passing backups — undo itself must not create
        # a backup-of-backup (no undo-of-undo).
        _atomic_write(path, backup_bytes)

        # Display-only diff, capped (see _UNDO_DIFF_LINE_BUDGET): the
        # restore above is byte-exact at any size; only the embedded diff
        # is size-gated.
        current_lines = current.splitlines(keepends=True)
        backup_lines = backup_content.splitlines(keepends=True)
        if max(len(current_lines), len(backup_lines)) > _UNDO_DIFF_LINE_BUDGET:
            return (
                f"Reverted {file_path} to previous state. "
                f"(diff omitted: the file exceeds "
                f"{_UNDO_DIFF_LINE_BUDGET} lines)"
            )

        diff = difflib.unified_diff(
            current_lines,
            backup_lines,
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
    # concurrent edit doesn't interleave with our two-file write — both the
    # in-process asyncio locks and the cross-process flocks.
    async with (
        file_locks[from_file],
        file_locks[to_file],
        edit_lock_or_refusal(from_path) as _from_refusal,
        edit_lock_or_refusal(to_path) as _to_refusal,
    ):
        if _from_refusal:
            return f"Error: {_from_refusal}"
        if _to_refusal:
            return f"Error: {_to_refusal}"
        try:
            # Importer discovery runs tldr subprocesses per consumer file:
            # worker thread, off the event loop (locks are held across the
            # await, exactly as they were held across the sync call).
            plan = await asyncio.to_thread(
                move_to_file,
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
