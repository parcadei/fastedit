"""FastEdit CLI — edit, read, delete, move, rename, search, diff, undo.

Usage:
    fastedit read src/app.py
    fastedit edit src/app.py --snippet '...' --replace greet
    fastedit edit src/app.py --snippet - --after greet < snippet.py
    fastedit batch-edit src/app.py --edits '[{"snippet": "...", "replace": "fn"}]'
    fastedit delete src/app.py old_function
    fastedit move src/app.py my_func --after other_func
    fastedit rename src/app.py old_name new_name
    fastedit diff src/app.py
    fastedit undo src/app.py
    fastedit search "query" src/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend helpers (for model-using subcommands: edit, batch-edit, multi-edit)
# ---------------------------------------------------------------------------

def _make_backend_with_overrides(args):
    """Build the inference backend; CLI args override environment variables."""
    import os

    backend_kind = getattr(args, "backend", None) or os.environ.get("FASTEDIT_BACKEND", "mlx")

    if backend_kind == "vllm":
        from .inference.vllm_engine import VLLMEngine
        api_base = (
            getattr(args, "api_base", None)
            or os.environ.get("FASTEDIT_VLLM_API_BASE", "http://127.0.0.1:8000/v1")
        )
        model = (
            getattr(args, "api_model", None)
            or os.environ.get("FASTEDIT_VLLM_MODEL", "/root/fastedit-merged")
        )
        return backend_kind, VLLMEngine(
            api_base=api_base,
            model=model,
            api_key=os.environ.get("FASTEDIT_VLLM_API_KEY", "not-needed"),
            max_tokens=int(os.environ.get("FASTEDIT_VLLM_MAX_TOKENS", "16384")),
        )
    else:
        from .inference.mlx_engine import MLXEngine
        from .model_download import get_model_path
        model_path = getattr(args, "model_path", None) or get_model_path()
        return backend_kind, MLXEngine(model_path)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _format_small_file(path: str, content: str, total_lines: int) -> str:
    return f"{path} ({total_lines} lines — small file, showing full content)\n\n{content}"


def _format_structure(path: str, data: dict, total_lines: int) -> str:
    files = data.get("files", [])
    if not files:
        return f"{path} ({total_lines} lines) — no structure detected"

    file_info = files[0]
    language = data.get("language", "unknown")
    definitions = file_info.get("definitions", [])
    imports = file_info.get("imports", [])

    lines: list[str] = [f"{path} ({language}, {total_lines} lines)"]
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

    # Definitions with line ranges
    if definitions:
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

            indent = ""
            if kind == "method":
                for cs, ce, _cn in class_ranges:
                    if cs <= ls <= ce:
                        indent = "  "
                        break

            label = sig if sig else f"{kind} {name}"
            lines.append(f"{indent}L{ls}-{le:<4} {label}")

    return "\n".join(lines)


def cmd_read(args):
    """Show a file's structure (functions, classes, line ranges)."""
    import json as json_mod
    import subprocess

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = path.read_text(encoding="utf-8", errors="replace")
    total_lines = content.count("\n") + (
        1 if content and not content.endswith("\n") else 0
    )

    # Small files: return the content directly
    if total_lines <= 100:
        print(_format_small_file(args.file, content, total_lines))
        return

    try:
        result = subprocess.run(
            ["tldr", "structure", args.file, "--format", "compact"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"Error: tldr structure failed for {args.file}: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

        data = json_mod.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"Error: tldr structure timed out for {args.file}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        # tldr not found -- fall back to simple line count
        print(f"{args.file} ({total_lines} lines)")
        return
    except json_mod.JSONDecodeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(_format_structure(args.file, data, total_lines))


def _try_deterministic_replace(path, original_code, original_lines, snippet, replace_sym, language, backups):
    from .data_gen.ast_analyzer import validate_parse
    from .inference.chunked_merge import (
        ChunkedMergeResult,
        _qualified_symbol_names,
        _resolve_symbol,
        get_ast_map,
    )
    from .inference.text_match import deterministic_edit

    total_lines = len(original_lines)
    ast_nodes = get_ast_map(str(path), total_lines)
    target_node = _resolve_symbol(replace_sym, ast_nodes or [])
    if target_node is None:
        available = _qualified_symbol_names(ast_nodes or [])
        print(
            f"Error: Symbol '{replace_sym}' not found in {path}. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    func_start = target_node.line_start - 1  # 0-indexed
    func_end = target_node.line_end  # exclusive
    original_func = "".join(original_lines[func_start:func_end])

    # Try deterministic text-match first
    edited = deterministic_edit(original_func, snippet)
    if edited is not None:
        edited_lines = edited.splitlines(keepends=True)
        if edited_lines and not edited_lines[-1].endswith("\n"):
            edited_lines[-1] += "\n"
        result_lines = list(original_lines)
        result_lines[func_start:func_end] = edited_lines
        merged = "".join(result_lines)
        parse_valid = True
        if language:
            parse_valid = validate_parse(merged, language)
        return ChunkedMergeResult(
            merged_code=merged, parse_valid=parse_valid,
            chunks_used=0, chunk_regions=[], model_tokens=0, latency_ms=0.0,
        )

    # Direct replacement: snippet IS the new symbol. Swap line ranges.
    snippet_text = snippet.rstrip("\n") + "\n"
    snippet_lines = snippet_text.splitlines(keepends=True)
    result_lines = list(original_lines)
    result_lines[func_start:func_end] = snippet_lines
    merged = "".join(result_lines)
    parse_valid = True
    if language:
        parse_valid = validate_parse(merged, language)
    if parse_valid:
        return ChunkedMergeResult(
            merged_code=merged, parse_valid=True,
            chunks_used=0, chunk_regions=[], model_tokens=0, latency_ms=0.0,
        )
    # Parse invalid: fall through to model-based chunked_merge
    return None


def cmd_edit(args):
    """Apply an edit snippet to a file using the FastEdit model."""
    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import chunked_merge
    from .mcp.backup import BackupStore, _atomic_write

    snippet = sys.stdin.read() if args.snippet == "-" else args.snippet
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backups = BackupStore()
    original_code = path.read_text(encoding="utf-8", errors="replace")
    language = detect_language(path)
    original_lines = original_code.splitlines(keepends=True)

    replace_sym = args.replace or None
    after_sym = args.after or None

    if replace_sym and not after_sym:
        result = _try_deterministic_replace(
            path, original_code, original_lines, snippet, replace_sym, language, backups,
        )
        if result is not None:
            _atomic_write(path, result.merged_code, backups=backups)
            print(f"Applied edit to {args.file}. latency: 0ms, 0 tok/s, 0 tokens")
            return

    # Lazy backend: only loaded when merge_fn is actually called.
    # Deterministic paths (after=, replace= with text-match) never call merge_fn.
    _backend_cache = {}

    def _lazy_merge_fn(*a, **kw):
        if "engine" not in _backend_cache:
            _, engine = _make_backend_with_overrides(args)
            _backend_cache["engine"] = engine
        return _backend_cache["engine"].merge_auto(*a, **kw)

    try:
        result = chunked_merge(
            original_code=original_code,
            snippet=snippet,
            file_path=args.file,
            merge_fn=_lazy_merge_fn,
            language=language,
            after=after_sym,
            replace=replace_sym,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _atomic_write(path, result.merged_code, backups=backups)

    tok_per_sec = (
        result.model_tokens / (result.latency_ms / 1000)
        if result.latency_ms > 0 else 0
    )
    metrics = (
        f"latency: {result.latency_ms:.0f}ms, "
        f"{tok_per_sec:.0f} tok/s, {result.model_tokens} tokens"
    )
    if result.chunks_used > 1:
        metrics += f", {result.chunks_used} chunk(s)"
    if getattr(result, "chunks_rejected", 0):
        print(
            f"Warning: {result.chunks_rejected}/{result.chunks_used} chunk(s) rejected. "
            f"Partial edit applied. {metrics}"
        )
    elif language and not result.parse_valid:
        print(f"Warning: merged output has parse errors. Wrote anyway. {metrics}")
    else:
        print(f"Applied edit to {args.file}. {metrics}")


def cmd_batch_edit(args):
    """Apply multiple sequential edits to one file."""
    import json as json_mod

    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import BatchEdit, batch_chunked_merge
    from .mcp.backup import BackupStore, _atomic_write

    edits_json = sys.stdin.read() if args.edits == "-" else args.edits
    try:
        edits_list = json_mod.loads(edits_json)
    except json_mod.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    batch = [
        BatchEdit(
            snippet=e["snippet"],
            after=e.get("after") or None,
            replace=e.get("replace") or None,
        )
        for e in edits_list
    ]

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backend_kind, backend = _make_backend_with_overrides(args)
    backups = BackupStore()
    original_code = path.read_text(encoding="utf-8", errors="replace")
    language = detect_language(path)

    result = batch_chunked_merge(
        original_code=original_code,
        edits=batch,
        file_path=args.file,
        merge_fn=backend.merge_auto,
        language=language,
    )
    _atomic_write(path, result.merged_code, backups=backups)
    print(
        f"Applied {len(batch)} edits to {args.file}. "
        f"latency: {result.latency_ms:.0f}ms, {result.model_tokens} tokens"
    )


def cmd_multi_edit(args):
    """Apply edits across multiple files."""
    import json as json_mod

    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import BatchEdit, batch_chunked_merge
    from .mcp.backup import BackupStore, _atomic_write

    file_edits_json = sys.stdin.read() if args.file_edits == "-" else args.file_edits
    try:
        file_edits_list = json_mod.loads(file_edits_json)
    except json_mod.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    backend_kind, backend = _make_backend_with_overrides(args)
    backups = BackupStore()

    for entry in file_edits_list:
        file_path = entry["file_path"]
        edits = entry["edits"]
        path = Path(file_path)
        if not path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        batch = [
            BatchEdit(
                snippet=e["snippet"],
                after=e.get("after") or None,
                replace=e.get("replace") or None,
            )
            for e in edits
        ]

        original_code = path.read_text(encoding="utf-8", errors="replace")
        language = detect_language(path)

        result = batch_chunked_merge(
            original_code=original_code,
            edits=batch,
            file_path=file_path,
            merge_fn=backend.merge_auto,
            language=language,
        )
        _atomic_write(path, result.merged_code, backups=backups)
        print(
            f"Applied {len(batch)} edits to {file_path}. "
            f"latency: {result.latency_ms:.0f}ms, {result.model_tokens} tokens"
        )


def cmd_delete(args):
    """Delete a function, method, or class from a file using AST analysis."""
    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import delete_symbol
    from .mcp.backup import BackupStore, _atomic_write

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    language = detect_language(path)
    backups = BackupStore()

    try:
        result = delete_symbol(
            file_path=args.file,
            symbol=args.symbol,
            language=language,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _atomic_write(path, result.merged_code, backups=backups)

    warn = ""
    if language and not result.parse_valid:
        warn = " (warning: parse errors after delete)"
    print(
        f"Deleted {result.deleted_kind} '{result.deleted_symbol}' from {args.file}. "
        f"Removed L{result.deleted_lines[0]}-{result.deleted_lines[1]} "
        f"({result.lines_removed} lines). 0 model tokens.{warn}"
    )


def cmd_move(args):
    """Move a symbol to after another symbol in the same file."""
    from .data_gen.ast_analyzer import detect_language
    from .inference.chunked_merge import move_symbol
    from .mcp.backup import BackupStore, _atomic_write

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    language = detect_language(path)
    backups = BackupStore()

    try:
        result = move_symbol(
            file_path=args.file,
            symbol=args.symbol,
            after=args.after,
            language=language,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _atomic_write(path, result.merged_code, backups=backups)

    warn = ""
    if language and not result.parse_valid:
        warn = " (warning: parse errors after move)"
    print(
        f"Moved {result.moved_kind} '{result.moved_symbol}' "
        f"from L{result.from_lines[0]}-{result.from_lines[1]} "
        f"to after '{result.after_symbol}' "
        f"(now L{result.new_lines[0]}-{result.new_lines[1]}) "
        f"in {args.file}. 0 model tokens.{warn}"
    )


def cmd_rename(args):
    """Rename all occurrences of a symbol in a file using word-boundary regex."""
    import difflib

    from .data_gen.ast_analyzer import detect_language
    from .inference.rename import do_rename
    from .mcp.backup import BackupStore, _atomic_write

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    original = path.read_text(encoding="utf-8")
    language = detect_language(path)

    renamed, count, skipped = do_rename(original, args.old_name, args.new_name, language)

    if count == 0:
        print(
            f"Error: no code occurrences of '{args.old_name}' found in {args.file} "
            f"(word-boundary, excluding strings/comments).",
            file=sys.stderr,
        )
        sys.exit(1)

    backups = BackupStore()
    _atomic_write(path, renamed, backups=backups)

    skip_note = f" (skipped {skipped} in strings/comments)" if skipped else ""

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        renamed.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print(
        f"Renamed '{args.old_name}' -> '{args.new_name}' in {args.file}: "
        f"{count} replacement(s).{skip_note} 0 model tokens."
    )
    print("".join(diff), end="")


def _format_search_results(stdout: str, mode: str) -> str:
    text = stdout.strip()
    if text:
        return text
    if mode == "references":
        return "No references found."
    return "No results found."


def cmd_search(args):
    """Search codebase for functions, symbols, and references."""
    import subprocess

    if args.mode == "references":
        cmd = ["tldr", "references", args.query, args.path,
               "--format", "text", "--limit", str(args.top_k)]
        error_label = "references"
    else:
        cmd = ["tldr", "search", args.query, args.path,
               "--format", "text", "--top-k", str(args.top_k)]
        if args.mode == "regex":
            cmd.append("--regex")
        elif args.mode == "hybrid" and args.regex_filter:
            cmd.extend(["--hybrid", args.regex_filter])
        error_label = "search"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"Error: tldr {error_label} failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        print(_format_search_results(result.stdout, args.mode))
    except subprocess.TimeoutExpired:
        print(f"Error: {error_label} search timed out", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: tldr not found on PATH", file=sys.stderr)
        sys.exit(1)


def cmd_diff(args):
    """Show unified diff between the last backup and the current file content."""
    import difflib

    from .mcp.backup import BackupStore

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    backups = BackupStore()

    if args.file not in backups:
        print(f"No backup recorded for {args.file}. Run an edit command first.")
        return

    # Peek without popping: read .bak file directly
    backup_content = backups._key_path(args.file).read_text(encoding="utf-8")
    current = path.read_text(encoding="utf-8", errors="replace")

    if backup_content == current:
        print(f"No changes detected in {args.file}.")
        return

    diff = difflib.unified_diff(
        backup_content.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print("".join(diff), end="")


def cmd_undo(args):
    """Revert the last edit to a file using BackupStore."""
    import difflib

    from .mcp.backup import BackupStore, _atomic_write

    backups = BackupStore()
    path = Path(args.file)

    if args.file not in backups:
        print(f"Error: no undo history for {args.file}. Nothing to revert.", file=sys.stderr)
        sys.exit(1)

    backup_content = backups.pop(args.file)
    current = path.read_text(encoding="utf-8") if path.exists() else ""

    # Write backup WITHOUT passing backups -- no backup-of-backup
    _atomic_write(path, backup_content)

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        backup_content.splitlines(keepends=True),
        fromfile=f"a/{path.name}",
        tofile=f"b/{path.name}",
    )
    print(f"Reverted {args.file} to previous state.")
    print("".join(diff), end="")


def cmd_pull(args):
    """Pull the merge model from HuggingFace."""
    from .model_download import get_model_path
    path = get_model_path(model_name=args.model)
    print(f"Model ready at: {path}")


# ---------------------------------------------------------------------------
# Argparse setup and main dispatch
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="fastedit",
        description="FastEdit — AST-aware code editing via CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # --- read (no model) ---
    read_p = sub.add_parser("read", help="Show file structure (functions, classes, line ranges)")
    read_p.add_argument("file", help="Path to source file")

    # --- search (no model) ---
    search_p = sub.add_parser("search", help="Search codebase for symbols and functions")
    search_p.add_argument("query", help="Search query or symbol name")
    search_p.add_argument("path", nargs="?", default=".", help="Directory to search (default: .)")
    search_p.add_argument(
        "--mode",
        choices=["search", "regex", "hybrid", "references"],
        default="search",
        help="Search mode (default: search)",
    )
    search_p.add_argument("--top-k", type=int, default=10, help="Max results (default: 10)")
    search_p.add_argument("--regex-filter", default="", help="Regex filter for hybrid mode")

    # --- diff (no model) ---
    diff_p = sub.add_parser("diff", help="Show diff between last backup and current file")
    diff_p.add_argument("file", help="Path to source file")

    # --- edit (model) ---
    edit_p = sub.add_parser("edit", help="Apply an edit snippet to a file")
    edit_p.add_argument("file", help="Path to source file")
    edit_p.add_argument("--snippet", required=True, help="Edit snippet or '-' for stdin")
    edit_p.add_argument("--after", default="", help="Insert new code after this symbol")
    edit_p.add_argument("--replace", default="", help="Replace this symbol with the snippet")
    edit_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    edit_p.add_argument("--model-path", default=None, help="MLX model path (overrides FASTEDIT_MODEL_PATH)")
    edit_p.add_argument("--api-base", default=None, help="vLLM API base URL")
    edit_p.add_argument("--api-model", default=None, help="vLLM model name")

    # --- batch-edit (model) ---
    be_p = sub.add_parser("batch-edit", help="Apply multiple edits to one file")
    be_p.add_argument("file", help="Path to source file")
    be_p.add_argument(
        "--edits", required=True,
        help=(
            'JSON list of edits. Each item: {"snippet": "...", "after": "sym"} '
            'or {"snippet": "...", "replace": "sym"}. Use \'-\' for stdin.'
        ),
    )
    be_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    be_p.add_argument("--model-path", default=None)
    be_p.add_argument("--api-base", default=None)
    be_p.add_argument("--api-model", default=None)

    # --- multi-edit (model) ---
    me_p = sub.add_parser("multi-edit", help="Apply edits across multiple files")
    me_p.add_argument(
        "--file-edits", required=True,
        help=(
            'JSON list. Each item: {"file_path": "...", "edits": [...]}. '
            'Use \'-\' for stdin.'
        ),
    )
    me_p.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    me_p.add_argument("--model-path", default=None)
    me_p.add_argument("--api-base", default=None)
    me_p.add_argument("--api-model", default=None)

    # --- delete (no model) ---
    del_p = sub.add_parser("delete", help="Delete a function/class/method by name")
    del_p.add_argument("file", help="Path to source file")
    del_p.add_argument("symbol", help="Symbol name to delete (e.g. 'my_func' or 'MyClass.method')")

    # --- move (no model) ---
    mv_p = sub.add_parser("move", help="Move a symbol to after another symbol")
    mv_p.add_argument("file", help="Path to source file")
    mv_p.add_argument("symbol", help="Symbol to move")
    mv_p.add_argument("--after", required=True, help="Move after this symbol")

    # --- rename (no model) ---
    rn_p = sub.add_parser("rename", help="Rename all occurrences of a symbol")
    rn_p.add_argument("file", help="Path to source file")
    rn_p.add_argument("old_name", help="Current symbol name")
    rn_p.add_argument("new_name", help="New symbol name")

    # --- undo (no model) ---
    undo_p = sub.add_parser("undo", help="Revert the last edit to a file")
    undo_p.add_argument("file", help="Path to source file")

    # pull
    pull_p = sub.add_parser("pull", help="Pull the merge model from HuggingFace (~3GB)")
    pull_p.add_argument("--model", required=True, choices=["mlx-8bit", "bf16"],
                        help="Model to download. Use mlx-8bit on Apple Silicon (MLX), bf16 on Linux GPU (vLLM).")

    args = parser.parse_args()

    if args.command == "read":
        cmd_read(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "edit":
        cmd_edit(args)
    elif args.command == "batch-edit":
        cmd_batch_edit(args)
    elif args.command == "multi-edit":
        cmd_multi_edit(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "move":
        cmd_move(args)
    elif args.command == "rename":
        cmd_rename(args)
    elif args.command == "undo":
        cmd_undo(args)
    elif args.command == "pull":
        cmd_pull(args)
    else:
        parser.print_help()

    # Passive update notice on exit. Silent when up-to-date, network-down,
    # or FASTEDIT_NO_UPDATE_CHECK=1. Runs after the command so it never
    # delays user-visible output.
    try:
        from .update_check import get_update_notice
        notice = get_update_notice()
        if notice:
            sys.stderr.write("\n" + notice + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
