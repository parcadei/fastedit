"""MCP tools for code editing: fast_edit, fast_batch_edit, fast_multi_edit."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..data_gen.ast_analyzer import detect_language
from ..inference.chunked_merge import BatchEdit, batch_chunked_merge, chunked_merge
from .server import _atomic_write, mcp


@mcp.tool(
    description=(
        "Edit code via fast local 1.7B model. Always set `replace` or `after`. "
        "Editing inside a function (1-2s): replace='func_name' + snippet with "
        "'# ... existing code ...' markers around changed lines. "
        "Works on ANY size function — even 500+ lines. The model only sees "
        "the ~35-line region around your change, not the whole function. "
        "Inserting new code (instant): after='existing_func', snippet is the new code. "
        "Replacing a whole function: replace='func_name', snippet is the full replacement. "
        "Without replace/after, the model processes the entire file (slow)."
    ),
)
async def fast_edit(file_path: str, edit_snippet: str, after: str = "", replace: str = "") -> str:
    """Apply an edit snippet to a file using the local FastEdit model."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backend_kind: str = lc["backend_kind"]
    backend = lc["backend"]
    snapshots: dict = lc["snapshots"]
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    needs_model = not (after and not replace)  # after= is pure text insert, no model

    async with file_locks[file_path]:
        original_code = path.read_text(encoding="utf-8", errors="replace")
        snapshots[file_path] = original_code
        language = detect_language(path)
        if language is None:
            return (
                f"Error: unsupported file type '{path.suffix}'. "
                "FastEdit supports: .py .js .jsx .ts .tsx .rs .go .java "
                ".c .h .cpp .cc .cxx .hpp .hh .rb .swift .kt .kts .cs .php "
                ".ex .exs. "
                "Use the Edit tool for this file."
            )

        try:
            if needs_model:
                if backend_kind == "mlx":
                    async with backend.acquire() as engine:
                        result = chunked_merge(
                            original_code=original_code,
                            snippet=edit_snippet,
                            file_path=file_path,
                            merge_fn=engine.merge_auto,
                            language=language,
                            after=after or None,
                            replace=replace or None,
                        )
                else:
                    result = await asyncio.to_thread(
                        chunked_merge,
                        original_code=original_code,
                        snippet=edit_snippet,
                        file_path=file_path,
                        merge_fn=backend.merge_auto,
                        language=language,
                        after=after or None,
                        replace=replace or None,
                    )
            else:
                # after= fast path: 0 model tokens, no engine needed
                result = chunked_merge(
                    original_code=original_code,
                    snippet=edit_snippet,
                    file_path=file_path,
                    merge_fn=lambda *a, **k: None,  # never called
                    language=language,
                    after=after or None,
                    replace=replace or None,
                )
        except ValueError as e:
            return f"Error: {e}"

        tok_per_sec = (
            result.model_tokens / (result.latency_ms / 1000)
            if result.latency_ms > 0 else 0
        )
        chunks_info = (
            f"{result.chunks_used} chunk(s)"
            if result.chunks_used > 1
            else ""
        )
        metrics = (
            f"latency: {result.latency_ms:.0f}ms, "
            f"{tok_per_sec:.0f} tok/s, "
            f"{result.model_tokens} tokens"
        )
        if chunks_info:
            metrics += f", {chunks_info}"

        # If all chunks were rejected due to hallucination, don't write garbage
        if getattr(result, 'chunks_rejected', 0) > 0 and result.chunks_rejected >= result.chunks_used:
            return (
                f"Error: edit rejected — model hallucinated on {result.chunks_rejected} chunk(s). "
                f"File unchanged. The function may be too large ({result.chunks_used} chunk(s)) "
                f"for the 1.7B model. Try a smaller edit or split the function. {metrics}"
            )

        if getattr(result, 'chunks_rejected', 0) > 0:
            _atomic_write(path, result.merged_code, backups=backups)
            return (
                f"Warning: {result.chunks_rejected}/{result.chunks_used} chunk(s) rejected "
                f"due to hallucination. Partial edit applied. {metrics}"
            )

        if language and not result.parse_valid:
            _atomic_write(path, result.merged_code, backups=backups)
            return (
                f"Warning: merged output has parse errors in {language}. "
                f"Wrote to {file_path} anyway. {metrics}"
            )

        _atomic_write(path, result.merged_code, backups=backups)
        return f"Applied edit to {file_path}. {metrics}"


@mcp.tool(
    description=(
        "Apply multiple edits to one file in a single call. `edits` is a JSON list "
        "of objects with `snippet` and optional `after`/`replace` keys. "
        "Edits are applied sequentially — each sees the result of the previous. "
        "One round-trip instead of N separate fast_edit calls."
    ),
)
async def fast_batch_edit(file_path: str, edits: str) -> str:
    """Apply multiple sequential edits to a file in one call."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backend_kind: str = lc["backend_kind"]
    backend = lc["backend"]
    snapshots: dict = lc["snapshots"]
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    try:
        edits_list = json.loads(edits)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON in edits parameter: {e}"

    if not isinstance(edits_list, list) or not edits_list:
        return "Error: edits must be a non-empty JSON list"

    batch = []
    for i, entry in enumerate(edits_list):
        if not isinstance(entry, dict) or "snippet" not in entry:
            return f"Error: edit {i} must be an object with a 'snippet' key"
        batch.append(BatchEdit(
            snippet=entry["snippet"],
            after=entry.get("after") or None,
            replace=entry.get("replace") or None,
        ))

    async with file_locks[file_path]:
        original_code = path.read_text(encoding="utf-8", errors="replace")
        snapshots[file_path] = original_code
        language = detect_language(path)
        if language is None:
            return (
                f"Error: unsupported file type '{path.suffix}'. "
                "FastEdit supports: .py .js .jsx .ts .tsx .rs .go .java "
                ".c .h .cpp .cc .cxx .hpp .hh .rb .swift .kt .kts .cs .php "
                ".ex .exs. "
                "Use the Edit tool for this file."
            )

        try:
            if backend_kind == "mlx":
                async with backend.acquire() as engine:
                    result = batch_chunked_merge(
                        original_code=original_code,
                        edits=batch,
                        file_path=file_path,
                        merge_fn=engine.merge_auto,
                        language=language,
                    )
            else:
                result = await asyncio.to_thread(
                    batch_chunked_merge,
                    original_code=original_code,
                    edits=batch,
                    file_path=file_path,
                    merge_fn=backend.merge_auto,
                    language=language,
                )
        except ValueError as e:
            return f"Error: {e}"

        tok_per_sec = (
            result.model_tokens / (result.latency_ms / 1000)
            if result.latency_ms > 0 else 0
        )
        metrics = (
            f"latency: {result.latency_ms:.0f}ms, "
            f"{tok_per_sec:.0f} tok/s, "
            f"{result.model_tokens} tokens, "
            f"{result.chunks_used} chunk(s), "
            f"{len(batch)} edit(s)"
        )

        if language and not result.parse_valid:
            _atomic_write(path, result.merged_code, backups=backups)
            return (
                f"Warning: parse errors after {len(batch)} edits to {file_path}. "
                f"Wrote to {file_path} anyway. {metrics}"
            )

        _atomic_write(path, result.merged_code, backups=backups)
        return f"Applied {len(batch)} edits to {file_path}. {metrics}"


@mcp.tool(
    description=(
        "Apply edits across multiple files in one call. `file_edits` is a JSON list "
        "of objects with `file_path` and `edits` (same format as fast_batch_edit). "
        "Files are processed sequentially so cross-file dependencies work correctly."
    ),
)
async def fast_multi_edit(file_edits: str) -> str:
    """Apply sequential edits across multiple files in one call."""
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    backend_kind: str = lc["backend_kind"]
    backend = lc["backend"]
    snapshots: dict = lc["snapshots"]
    backups: dict = lc["backups"]
    file_locks: dict = lc["file_locks"]

    try:
        file_edits_list = json.loads(file_edits)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON in file_edits parameter: {e}"

    if not isinstance(file_edits_list, list) or not file_edits_list:
        return "Error: file_edits must be a non-empty JSON list"

    results: list[str] = []
    total_tokens = 0
    total_latency = 0.0
    total_edits = 0

    for fi, file_entry in enumerate(file_edits_list):
        if not isinstance(file_entry, dict):
            return f"Error: file_edits[{fi}] must be an object"
        fp = file_entry.get("file_path")
        edits_raw = file_entry.get("edits")
        if not fp or not edits_raw or not isinstance(edits_raw, list):
            return f"Error: file_edits[{fi}] needs 'file_path' and 'edits' (list)"

        path = Path(fp)
        if not path.exists():
            return f"Error: file not found: {fp}"

        batch = []
        for i, entry in enumerate(edits_raw):
            if not isinstance(entry, dict) or "snippet" not in entry:
                return f"Error: file_edits[{fi}].edits[{i}] needs a 'snippet' key"
            batch.append(BatchEdit(
                snippet=entry["snippet"],
                after=entry.get("after") or None,
                replace=entry.get("replace") or None,
            ))

        # Lock each file individually as we process it sequentially
        async with file_locks[fp]:
            original_code = path.read_text(encoding="utf-8", errors="replace")
            snapshots[fp] = original_code
            language = detect_language(path)
            if language is None:
                return (
                    f"Error: unsupported file type '{path.suffix}' for {fp}. "
                    "FastEdit supports: .py .js .jsx .ts .tsx .rs .go .java "
                    ".c .h .cpp .cc .cxx .hpp .hh .rb .swift .kt .kts .cs .php "
                    ".ex .exs. "
                    "Use the Edit tool for this file."
                )

            try:
                if backend_kind == "mlx":
                    async with backend.acquire() as engine:
                        result = batch_chunked_merge(
                            original_code=original_code,
                            edits=batch,
                            file_path=fp,
                            merge_fn=engine.merge_auto,
                            language=language,
                        )
                else:
                    result = await asyncio.to_thread(
                        batch_chunked_merge,
                        original_code=original_code,
                        edits=batch,
                        file_path=fp,
                        merge_fn=backend.merge_auto,
                        language=language,
                    )
            except ValueError as e:
                return f"Error on {fp}: {e}"

            _atomic_write(path, result.merged_code, backups=backups)
            total_tokens += result.model_tokens
            total_latency += result.latency_ms
            total_edits += len(batch)

            status = "ok" if result.parse_valid else "parse_errors"
            results.append(f"{fp}: {len(batch)} edit(s), {status}")

    tok_per_sec = (
        total_tokens / (total_latency / 1000)
        if total_latency > 0 else 0
    )
    summary = (
        f"Applied {total_edits} edit(s) across {len(file_edits_list)} file(s). "
        f"latency: {total_latency:.0f}ms, {tok_per_sec:.0f} tok/s, "
        f"{total_tokens} tokens"
    )
    detail = "\n".join(results)
    return f"{summary}\n{detail}"
