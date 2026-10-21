"""MCP tools for code editing: fast_edit, fast_batch_edit, fast_multi_edit."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..data_gen.ast_analyzer import detect_language
from ..file_lock import edit_lock_or_refusal
from ..inference.chunked_merge import (
    BatchEdit,
    _validation_retries_metric,
    batch_chunked_merge,
    chunked_merge,
)
from ..io_utils import UnsupportedEncodingError, read_source
from ..lang_attributes import DocxError, build_docx_bytes, read_docx
from ..update_check import get_update_notice_async
from ..write_gates import _all_chunks_rejected, _rejection_refusal
from .server import ConcurrentModificationError, _atomic_write, mcp

# Once-per-server-session flag — attaches the update banner to the first
# successful edit response so the host LLM can relay it to the human.
_UPDATE_NOTICE_SHOWN = False
_update_notice_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def heavy_window_or_refusal(path: Path):
    """MCP heavy-slot guard for one target file (the global governor).

    Yields ``None`` when the file is small (below ``heavy_file_bytes`` — the
    overwhelmingly common case pays nothing) or when the heavy slot was
    acquired; yields the refusal MESSAGE when every slot was busy for longer
    than the queue budget (the tools return ``Error: ...`` strings, never
    raise at the host LLM). The flock attempt loop is non-blocking, so the
    wait runs on a worker thread and the event loop keeps serving.
    """
    from ..resource_hub import SlotWaitTimeout, acquire_slot, is_heavy_file

    if not is_heavy_file(path):
        yield None
        return
    try:
        lease = await asyncio.to_thread(
            acquire_slot, "heavy", str(path),
        )
    except SlotWaitTimeout as e:
        yield str(e)
        return
    try:
        yield None
    finally:
        lease.release()


def _merge_under_model_slot(merge_callable, /, **kwargs):
    """Run one merge under a global model slot (the non-pool backends).

    The mlx backend serializes through ``ModelPool.acquire`` (which wraps
    the hub's model slot around engine load + merge). Backend kinds with no
    pool — vllm (an HTTP engine) and llm — bypass the pool, so their merge
    call sites wrap THIS helper instead: the slot spans exactly the model
    work, keeping the machine-wide instance ceiling true for every backend.
    Runs inside a worker thread (the caller's ``asyncio.to_thread``), so the
    hub's queue poll never blocks the event loop.
    """
    from ..resource_hub import acquire_slot

    with acquire_slot("model"):
        return merge_callable(**kwargs)


async def _maybe_append_update_notice(message: str) -> str:
    """Append a one-line PyPI update notice to the tool response, at most
    once per server session. Silent on any failure."""
    global _UPDATE_NOTICE_SHOWN
    if _UPDATE_NOTICE_SHOWN:
        return message
    async with _update_notice_lock:
        if _UPDATE_NOTICE_SHOWN:
            return message
        try:
            notice = await get_update_notice_async()
        except Exception:  # noqa: BLE001 -- deliberate: docstring "Silent on any failure" -- an update-check hiccup must never perturb the tool response
            notice = None
        _UPDATE_NOTICE_SHOWN = True
        if notice:
            return f"{message}\n\n{notice}"
    return message


# ---------------------------------------------------------------------------
# Step 18 (B34): shared per-file write gates.
#
# fast_edit, fast_batch_edit and fast_multi_edit must enforce the SAME
# signals in the SAME order: the fail-loud hallucination refusal first
# (never overridable), then the parse gate (force=True opt-in). The two
# gate helpers above are imported from fastedit.write_gates so the CLI's
# edit/batch-edit/multi-edit tails enforce the identical predicate with
# the identical refusal wording — the single-edit refusals below are
# quoted verbatim by the batch tools. The response-shaped helpers that
# follow (_partial_rejection_warning, _parse_refusal,
# _concurrent_modification_error) stay MCP-local: the CLI prints its own
# established message shapes.
# ---------------------------------------------------------------------------


def _partial_rejection_warning(result, metrics: str) -> str:
    """Write-with-warning response for a partially rejected merge."""
    return (
        f"Warning: {result.chunks_rejected}/{result.chunks_used} chunk(s) rejected "
        f"due to hallucination. Partial edit applied. {metrics}"
    )


def _parse_refusal(file_path: str, language: str, metrics: str) -> str:
    """Fail-loud refusal for a parse-invalid merge (the CLI's
    ``_refuse_if_edit_broke_parse`` voice). ``force=True`` is the explicit
    opt-in escape hatch."""
    return (
        f"Error: merged output for {file_path} has parse errors in {language}; "
        f"refusing to write. The file is unchanged. Break the edit into smaller "
        f"edits, or pass force=True to write anyway. {metrics}"
    )


def _concurrent_modification_error() -> str:
    """B37: uniform clean response when a write was refused because the file
    changed on disk between the tool's read and its write (the read-time
    stat no longer matches at write time). Nothing was written; the file on
    disk is exactly as the external writer left it."""
    return (
        "Error: file changed on disk since it was read; "
        "re-read and retry — file unchanged."
    )


def _persist_merge(
    path: Path,
    merged_code: str,
    *,
    backups,
    encoding: str,
    expected_stat,
) -> str | None:
    """Write the merge result and return an error string on refusal.

    Step D3: ``.docx`` containers go through the adapter — the merged
    document XML is rebuilt into the zip (every other entry preserved
    byte-for-byte) and written as BYTES; every other file writes the str
    content with the codec the file was read with (B23). Both paths keep
    the B22 backup and the B37 expected-stat guard. Returns ``None`` on
    success, else the fail-loud error string (corrupt container).
    """
    try:
        if path.suffix.lower() == ".docx":
            content: str | bytes = build_docx_bytes(path, merged_code)
        else:
            content = merged_code
    except DocxError as e:
        return f"Error: {e}"
    try:
        _atomic_write(
            path, content, backups=backups, encoding=encoding,
            expected_stat=expected_stat,
        )
    except ConcurrentModificationError:
        return _concurrent_modification_error()
    return None


@mcp.tool(
    description=(
        "Apply a code edit via tree-sitter AST + a fast local 1.7B merge model.\n"
        "\n"
        "USE CASES (choose the pattern, then write the minimal snippet):\n"
        "• add-guard / prelude (insert at TOP):  replace='func', snippet='<new_lines>\\n#...\\n'\n"
        "• append (insert at BOTTOM):            replace='func', snippet='#...\\n<new_lines>\\n'\n"
        "• edit in the MIDDLE:                   replace='func', snippet='<anchor>\\n<new_lines>\\n#...\\n'\n"
        "• replace the whole function:           replace='func', snippet=<full new body>\n"
        "• new symbol after an existing one:     after='existing', snippet=<new code>   (pure AST, 0 tokens)\n"
        "• surgical class-member edit:           replace='Class', preserve_siblings=True, snippet=<class shell + changed members>\n"
        "\n"
        "MARKERS (all accepted — pick the shortest):\n"
        "• #...   — 2 tokens (Python / Ruby / Elixir)\n"
        "• //...  — 2 tokens (JS / TS / Rust / Go / Java / C / C++ / Swift / Kotlin / C# / PHP)\n"
        "• …      — 1 token (U+2026, language-agnostic)\n"
        "• Legacy '# ... existing code ...' / '// ... existing code ...' still work.\n"
        "\n"
        "SIGNATURE: replace='name' auto-preserves the target's def/fn/class signature — do NOT "
        "repeat it in the snippet. Just send the new body (or body fragment + marker).\n"
        "\n"
        "Always set replace or after. Omitting both triggers a whole-file merge — slow and "
        "unreliable on files > 150 lines. Works on functions of any size — the model sees only "
        "a ~35-line region around your change, never the full file.\n"
        "\n"
        "When replace=<name> changes the target function's signature, the response "
        "includes a caller-impact note summarising how many call sites reference it — "
        "informational only, does not block the edit.\n"
        "\n"
        "PARSE SAFETY: if the merged output fails a parse check, the edit is refused, "
        "nothing is written, and the response suggests smaller edits. Pass force=True "
        "to override that refusal and write anyway — use only when you intend it. "
        "force never overrides a hallucination rejection (all chunks rejected). "
        "Validation is relative (edit-not-correct): a pre-existing syntax error "
        "elsewhere in the file is preserved, never auto-fixed, and rejected merge "
        "attempts retry up to FASTEDIT_MAX_RETRIES times (default 8) before refusing."
    ),
)
async def fast_edit(
    file_path: str,
    edit_snippet: str,
    after: str = "",
    replace: str = "",
    preserve_siblings: bool = False,
    force: Annotated[bool, Field(
        description=(
            "write even when the merged output fails a parse check; "
            "use only when you intend it"
        ),
    )] = False,
) -> str:
    """Apply an edit snippet to a file using the local FastEdit model.

    Refuses to write when the merged output fails the parse check (mirrors
    the CLI's ``_refuse_if_edit_broke_parse``); ``force=True`` restores the
    old write-anyway-with-warning escape hatch. The chunks-rejected
    hallucination refusal is never overridable.
    """
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

    # Three zero-model paths: `after=` (pure insert), and `replace=` with
    # `preserve_siblings=True` (pure AST splice). Everything else may reach
    # the model after the deterministic fast paths.
    needs_model = not (
        (after and not replace)
        or (replace and preserve_siblings)
    )

    # Three lock layers: the in-process asyncio lock (concurrent requests to
    # THIS server), the cross-process per-file flock (a CLI run or a second
    # MCP server instance), and the GLOBAL governor's heavy window for a
    # big file (a no-op for small files). The heavy guard yields the
    # queue-timeout refusal instead of raising — same shape as the edit
    # lock's refusal below.
    async with (
        file_locks[file_path],
        heavy_window_or_refusal(path) as _heavy_refusal,
        edit_lock_or_refusal(path) as _lock_refusal,
    ):
        if _heavy_refusal:
            return f"Error: {_heavy_refusal}"
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # B21: strict-decode read -- an undecodable byte must never become
        # U+FFFD and get written back as EF BF BD. UTF-16/binary files are
        # refused here, before the merge (and before tree-sitter). B37: the
        # stat of that same open rides along so the write below can refuse
        # when the file changed on disk in the read-to-write window.
        #
        # Step D3: ``.docx`` is the ONE suffix with an adapter read/write
        # path — a zip container, not text, so ``read_source`` (rightly)
        # refuses it and ``detect_language`` (rightly) cannot resolve it.
        # The adapter extracts ``word/document.xml`` (strict UTF-8,
        # fail-loud on corrupt containers via :class:`DocxError`) and the
        # pipeline edits/validates it as XML; the write side rebuilds the
        # container around the merged document preserving every other
        # entry byte-for-byte (``build_docx_bytes``). This is suffix
        # registration ONLY for the adapter path — no other behavior
        # changes for any other suffix.
        is_docx = path.suffix.lower() == ".docx"
        read_stat = None
        encoding = "utf-8"
        if is_docx:
            try:
                original_code, read_stat = read_docx(path, return_stat=True)
            except DocxError as e:
                return f"Error: {e}"
            language = "xml"
        else:
            try:
                original_code, encoding, read_stat = read_source(
                    path, return_stat=True,
                )
            except UnsupportedEncodingError as e:
                return f"Error: {e}"
            language = detect_language(path)
        snapshots[file_path] = original_code
        # Step D2: ``language=None`` is a SUPPORTED path, not a refusal —
        # the AST-less structureless pipeline (D1 trait battery + D2 text
        # anchor windows) edits any strictly-decoded text file. The
        # pipeline's own gates are the honest support boundary now (the
        # >150-line no-anchor whole-file gate fails loud with the file's
        # symbols; the battery refuses unfaithful merges), so the old
        # extension allowlist here — which made ``.txt``/``.log`` edits
        # impossible at the door — is gone.

        try:
            if needs_model:
                if backend_kind == "mlx":
                    async with backend.acquire() as engine:
                        # The merge (tree-sitter + MLX inference) is seconds
                        # of synchronous work; run it on a worker thread so
                        # the event loop keeps serving other requests. Same
                        # shape as the LLM branch below (and the only way a
                        # multi-engine pool actually parallelizes, B40).
                        result = await asyncio.to_thread(
                            chunked_merge,
                            original_code=original_code,
                            snippet=edit_snippet,
                            file_path=file_path,
                            merge_fn=engine.merge_auto,
                            language=language,
                            after=after or None,
                            replace=replace or None,
                            preserve_siblings=preserve_siblings,
                        )
                else:
                    result = await asyncio.to_thread(
                        _merge_under_model_slot,
                        lambda **kw: chunked_merge(
                            merge_fn=backend.merge_auto, **kw,
                        ),
                        original_code=original_code,
                        snippet=edit_snippet,
                        file_path=file_path,
                        language=language,
                        after=after or None,
                        replace=replace or None,
                        preserve_siblings=preserve_siblings,
                    )
            else:
                # Zero-model fast path (after= insert, or replace= with
                # preserve_siblings=True): no engine needed. Still pure-Python
                # AST work over the whole file — thread it like the model paths
                # so a huge file cannot stall the loop.
                result = await asyncio.to_thread(
                    chunked_merge,
                    original_code=original_code,
                    snippet=edit_snippet,
                    file_path=file_path,
                    merge_fn=lambda *a, **k: None,  # never called
                    language=language,
                    after=after or None,
                    replace=replace or None,
                    preserve_siblings=preserve_siblings,
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
        # Step A3: retries consumed by the validation loop surface in the
        # metrics segment (empty string when none — shape stays stable).
        metrics += _validation_retries_metric(getattr(result, "retries", 0))

        # If all chunks were rejected due to hallucination, don't write garbage
        if _all_chunks_rejected(result):
            return _rejection_refusal(result, metrics)

        if getattr(result, "chunks_rejected", 0) > 0:
            error = _persist_merge(
                path, result.merged_code, backups=backups, encoding=encoding,
                expected_stat=read_stat,
            )
            if error:
                return error
            return _partial_rejection_warning(result, metrics)

        # B10: parity with the CLI's _refuse_if_edit_broke_parse — THIS edit
        # producing a parse-invalid merge must not be persisted with a mere
        # warning. Refuse without writing; force=True is the explicit opt-in
        # escape hatch that restores the old write-with-warning behavior.
        if language and not result.parse_valid and not force:
            return _parse_refusal(file_path, language, metrics)

        if language and not result.parse_valid:
            error = _persist_merge(
                path, result.merged_code, backups=backups, encoding=encoding,
                expected_stat=read_stat,
            )
            if error:
                return error
            return (
                f"Warning: merged output has parse errors in {language}. "
                f"Wrote to {file_path} anyway. {metrics}"
            )

        # B23: write with the codec the file was read with, so untouched
        # bytes (a latin-1 é, a BOM) round-trip exactly. B37: the read-time
        # stat guards against clobbering an external write.
        error = _persist_merge(
            path, result.merged_code, backups=backups, encoding=encoding,
            expected_stat=read_stat,
        )
        if error:
            return error

        # VAL-M3-001: pre-flight impact note. When replace=<name> and
        # the signature line actually changed, surface the cross-file
        # caller count so the user knows callers may break. Hot-path
        # discipline (VAL-M3-002): the helper short-circuits on matching
        # signatures without invoking tldr. Swallow any exception so an
        # infra hiccup can't fail a successful edit.
        impact_suffix = ""
        if replace:
            try:
                from ..inference.caller_safety import (
                    _find_project_root,
                    compute_signature_impact_note,
                )
                project_root = _find_project_root(path)
                # May invoke tldr (a subprocess with its own timeout);
                # thread it so the event loop is not blocked for its run.
                note = await asyncio.to_thread(
                    compute_signature_impact_note,
                    old_code=original_code,
                    new_code=result.merged_code,
                    symbol=replace,
                    language=language,
                    file_path=path,
                    project_root=project_root,
                )
                if note:
                    impact_suffix = "\n" + note
            except Exception:  # noqa: BLE001 -- deliberate: the edit has already landed successfully; we swallow any exception so an infra hiccup (tldr/AST) cannot fail the success response
                impact_suffix = ""

        return await _maybe_append_update_notice(
            f"Applied edit to {file_path}. {metrics}{impact_suffix}"
        )


@mcp.tool(
    description=(
        "Apply multiple edits to one file in a single call. `edits` is a JSON list of "
        "objects, each with a `snippet` key plus optional `after` / `replace` / "
        "`preserve_siblings`. Edits apply sequentially — each sees the result of the "
        "previous. One round-trip instead of N separate fast_edit calls.\n"
        "\n"
        "Each edit follows the same minimal-snippet patterns as fast_edit: see its "
        "description for USE CASES + MARKERS. Short markers #... / //... / … are accepted; "
        "replace= auto-preserves the target's signature (do not repeat it in the snippet).\n"
        "\n"
        "PARSE SAFETY: if the merged output fails a parse check, nothing is written "
        "and the response explains why. Pass force=True to override that refusal and "
        "write anyway — use only when you intend it. force never overrides a "
        "hallucination rejection (all chunks rejected): a fully-rejected merge is "
        "always refused and the file is left unchanged."
    ),
)
async def fast_batch_edit(
    file_path: str,
    edits: str,
    force: Annotated[bool, Field(
        description=(
            "write even when the merged output fails a parse check; "
            "use only when you intend it"
        ),
    )] = False,
) -> str:
    """Apply multiple sequential edits to a file in one call.

    Step 18 (B34): mirrors fast_edit's write gates — a merge whose chunks
    were all rejected is refused (never overridable) and a parse-invalid
    merge is refused unless ``force=True``.
    """
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
            preserve_siblings=bool(entry.get("preserve_siblings", False)),
        ))

    # In-process asyncio lock + cross-process per-file flock + the GLOBAL
    # governor's heavy window (see fast_edit above).
    async with (
        file_locks[file_path],
        heavy_window_or_refusal(path) as _heavy_refusal,
        edit_lock_or_refusal(path) as _lock_refusal,
    ):
        if _heavy_refusal:
            return f"Error: {_heavy_refusal}"
        if _lock_refusal:
            return f"Error: {_lock_refusal}"
        # B21: strict-decode read; UTF-16/binary refused before the merge.
        # B37: the read-time stat rides along to the write below.
        try:
            original_code, encoding, read_stat = read_source(path, return_stat=True)
        except UnsupportedEncodingError as e:
            return f"Error: {e}"
        snapshots[file_path] = original_code
        # Step D2: ``language=None`` rides into the pipeline (the AST-less
        # structureless path); the pipeline's own gates govern support —
        # see the note on the fast_edit gate above.
        language = detect_language(path)

        try:
            if backend_kind == "mlx":
                async with backend.acquire() as engine:
                    # Off-loop merge: same rationale as fast_edit's mlx branch.
                    result = await asyncio.to_thread(
                        batch_chunked_merge,
                        original_code=original_code,
                        edits=batch,
                        file_path=file_path,
                        merge_fn=engine.merge_auto,
                        language=language,
                    )
            else:
                result = await asyncio.to_thread(
                    _merge_under_model_slot,
                    lambda **kw: batch_chunked_merge(
                        merge_fn=backend.merge_auto, **kw,
                    ),
                    original_code=original_code,
                    edits=batch,
                    file_path=file_path,
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
        # Step A3: validation-retry count in the metrics segment (stable
        # shape when none were consumed).
        metrics += _validation_retries_metric(getattr(result, "retries", 0))

        # Step 18 (B34): same gate order as fast_edit — the fail-loud
        # hallucination refusal first (never force-overridable), then the
        # parse gate (force=True opt-in).
        if _all_chunks_rejected(result):
            return _rejection_refusal(result, metrics)

        if getattr(result, "chunks_rejected", 0) > 0:
            try:
                _atomic_write(
                    path, result.merged_code, backups=backups, encoding=encoding,
                    expected_stat=read_stat,
                )
            except ConcurrentModificationError:
                return _concurrent_modification_error()
            return _partial_rejection_warning(result, metrics)

        if language and not result.parse_valid and not force:
            return _parse_refusal(file_path, language, metrics)

        if language and not result.parse_valid:
            try:
                _atomic_write(
                    path, result.merged_code, backups=backups, encoding=encoding,
                    expected_stat=read_stat,
                )
            except ConcurrentModificationError:
                return _concurrent_modification_error()
            return (
                f"Warning: parse errors after {len(batch)} edits to {file_path}. "
                f"Wrote to {file_path} anyway. {metrics}"
            )

        try:
            _atomic_write(
                path, result.merged_code, backups=backups, encoding=encoding,
                expected_stat=read_stat,
            )
        except ConcurrentModificationError:
            return _concurrent_modification_error()
        return await _maybe_append_update_notice(
            f"Applied {len(batch)} edits to {file_path}. {metrics}"
        )


@mcp.tool(
    description=(
        "Apply edits across multiple files in one call. `file_edits` is a JSON list "
        "of objects with `file_path` and `edits` (same format as fast_batch_edit). "
        "Files are processed sequentially so cross-file dependencies work correctly.\n"
        "\n"
        "PARSE SAFETY: each file's write follows the same gates as fast_edit — a "
        "merge whose chunks were all rejected as hallucinations is never written "
        "(not even with force=True), and a parse-invalid merge is not written "
        "unless force=True. Refused files are left unchanged while the remaining "
        "targets still process; the per-file statuses in the response tell you "
        "exactly what was written and what was refused."
    ),
)
async def fast_multi_edit(
    file_edits: str,
    force: Annotated[bool, Field(
        description=(
            "write even when the merged output fails a parse check; "
            "use only when you intend it"
        ),
    )] = False,
) -> str:
    """Apply sequential edits across multiple files in one call.

    Step 18 (B34): per-target write gates mirror fast_edit — a rejected or
    parse-invalid target is left untouched (the remaining targets still
    write) and the summary distinguishes ok / rejected / parse-errors.
    """
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
    written_files = 0
    refused_files = 0
    total_tokens = 0
    total_latency = 0.0
    total_edits = 0
    total_retries = 0

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
                preserve_siblings=bool(entry.get("preserve_siblings", False)),
            ))

        # Lock each file individually as we process it sequentially, plus
        # the GLOBAL governor's heavy window per big target (a no-op for
        # small files).
        async with (
            file_locks[fp],
            heavy_window_or_refusal(path) as _heavy_refusal,
            edit_lock_or_refusal(path) as _lock_refusal,
        ):
            if _heavy_refusal:
                # Queue budget exhausted for a big target: refuse it like
                # any other gated target; the remaining targets still write.
                results.append(f"{fp}: {_heavy_refusal}")
                refused_files += 1
                continue
            if _lock_refusal:
                # Another fastedit process holds this target: refuse it like
                # any other gated target; the remaining targets still write.
                results.append(f"{fp}: {_lock_refusal}")
                refused_files += 1
                continue
            # B21: strict-decode read; UTF-16/binary refused before the merge.
            # B37: the read-time stat rides along to this file's write below.
            try:
                original_code, encoding, read_stat = read_source(path, return_stat=True)
            except UnsupportedEncodingError as e:
                return f"Error on {fp}: {e}"
            snapshots[fp] = original_code
            # Step D2: ``language=None`` rides into the pipeline (the
            # AST-less structureless path); the pipeline's own gates govern
            # support — see the note on the fast_edit gate above.
            language = detect_language(path)

            try:
                if backend_kind == "mlx":
                    async with backend.acquire() as engine:
                        # Off-loop merge: same rationale as fast_edit's mlx branch.
                        result = await asyncio.to_thread(
                            batch_chunked_merge,
                            original_code=original_code,
                            edits=batch,
                            file_path=fp,
                            merge_fn=engine.merge_auto,
                            language=language,
                        )
                else:
                    result = await asyncio.to_thread(
                        _merge_under_model_slot,
                        lambda **kw: batch_chunked_merge(
                            merge_fn=backend.merge_auto, **kw,
                        ),
                        original_code=original_code,
                        edits=batch,
                        file_path=fp,
                        language=language,
                    )
            except ValueError as e:
                return f"Error on {fp}: {e}"

            # Step 18 (B34): per-target write gates, mirroring fast_edit's
            # order — the fail-loud hallucination refusal first (never
            # force-overridable), then the parse gate (force=True opt-in).
            # A refused target is left untouched; the remaining targets
            # still process (partial-batch semantics — the pre-existing
            # whole-call aborts for hard errors above are unchanged).
            if _all_chunks_rejected(result):
                results.append(
                    f"{fp}: {len(batch)} edit(s), rejected — model hallucinated on "
                    f"{result.chunks_rejected} chunk(s). File unchanged. "
                    f"Try a smaller edit or split the function."
                )
                refused_files += 1
            elif language and not result.parse_valid and not force:
                results.append(
                    f"{fp}: {len(batch)} edit(s), parse_errors — merged output has "
                    f"parse errors in {language}; refusing to write. File unchanged. "
                    f"Break the edit into smaller edits, or pass force=True to write "
                    f"anyway."
                )
                refused_files += 1
            else:
                try:
                    _atomic_write(
                        path, result.merged_code, backups=backups,
                        encoding=encoding, expected_stat=read_stat,
                    )
                except ConcurrentModificationError:
                    # B37: this target changed on disk since it was read —
                    # it is refused like any other gated target while the
                    # remaining targets still write.
                    results.append(
                        f"{fp}: file changed on disk since it was read; "
                        f"re-read and retry — file unchanged."
                    )
                    refused_files += 1
                else:
                    written_files += 1
                    total_edits += len(batch)
                    rejected = getattr(result, "chunks_rejected", 0)
                    if rejected > 0:
                        results.append(
                            f"{fp}: {len(batch)} edit(s), ok (warning: {rejected}/"
                            f"{result.chunks_used} chunk(s) rejected due to hallucination "
                            f"— partial edit applied)"
                        )
                    elif language and not result.parse_valid:
                        results.append(
                            f"{fp}: {len(batch)} edit(s), parse_errors — written with "
                            f"force=True despite parse errors in {language}."
                        )
                    else:
                        results.append(f"{fp}: {len(batch)} edit(s), ok")
            # The merge ran either way — its cost stays in the totals.
            total_tokens += result.model_tokens
            total_latency += result.latency_ms
            # Step A3: retries aggregate into the summary's metrics segment.
            total_retries += getattr(result, "retries", 0)

    tok_per_sec = (
        total_tokens / (total_latency / 1000)
        if total_latency > 0 else 0
    )
    # Step 18 (B34): the all-clean header keeps its exact existing shape;
    # with refusals the counts narrow to what was actually written and the
    # not-written files are called out instead of silently vanishing.
    if refused_files:
        header = (
            f"Applied {total_edits} edit(s) across {written_files} of "
            f"{len(file_edits_list)} file(s); {refused_files} file(s) not written. "
        )
    else:
        header = (
            f"Applied {total_edits} edit(s) across {len(file_edits_list)} file(s). "
        )
    summary = (
        f"{header}"
        f"latency: {total_latency:.0f}ms, {tok_per_sec:.0f} tok/s, "
        f"{total_tokens} tokens"
        f"{_validation_retries_metric(total_retries)}"
    )
    detail = "\n".join(results)
    return f"{summary}\n{detail}"
