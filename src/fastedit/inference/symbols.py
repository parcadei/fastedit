"""Deterministic symbol operations: delete and move.

Pure AST-based operations that require no model inference.
Uses tldr to find exact line ranges and splices code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .ast_utils import (
    BatchEdit,
    ChunkedMergeResult,
    DeleteResult,
    MoveResult,
    _qualified_symbol_names,
    _resolve_symbol,
    get_ast_map,
    get_ast_map_from_source,
)


def delete_symbol(
    file_path: str,
    symbol: str,
    language: str | None = None,
) -> DeleteResult:
    """Delete a function, method, or class from a file using AST line ranges.

    Pure deterministic operation -- no model inference. Uses in-memory
    tree-sitter (via get_ast_map_from_source) for authoritative line
    ranges. Works across all 16 languages supported by tree-sitter.

    Prefers the in-memory path over get_ast_map (which shells out to
    tldr structure) because tldr's structure extractor has a known
    bug where @decorator-wrapped Python functions are mis-reported:
    the decorated function's span is missing and the NEXT function
    inherits its line numbers, silently corrupting deletes. See
    docs/testing-matrix.md for the repro. The in-memory walker correctly
    spans the decorated_definition AST node including the decorator.

    Args:
        file_path: Path to the source file.
        symbol: Name of the function, method, or class to delete.
        language: Optional language for parse validation.

    Returns:
        DeleteResult with the modified code.

    Raises:
        ValueError: If the symbol is not found in the file's AST.
    """
    # io_utils/split_join are deliberately function-body imports (kept out
    # of the module import surface); get_ast_map_from_source itself needs no
    # local import -- it is already imported at module level above.
    from ..io_utils import read_source
    from ..split_join import normalize_bare_cr_for_ast

    path = Path(file_path)
    # B21: strict decode -- an undecodable byte must never become U+FFFD in
    # the spliced output. UnsupportedEncodingError (a ValueError) propagates
    # to the caller's existing refusal handling. The codec is the caller's
    # write concern: the writer re-reads/holds the encoding for _atomic_write.
    original_code, _ = read_source(path)
    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    # In-memory AST -- bypasses tldr's decorator-span bug. Feed it an
    # LF-normalized copy so a lone CR (invisible to tree-sitter's row
    # counting) does not collapse the whole file into one line; the
    # returned line numbers stay valid against original_lines either way.
    # B3: the caller's `language` rides along as an explicit hint so
    # extension-unwired languages (all-grammars extras) still resolve.
    ast_nodes = get_ast_map_from_source(
        normalize_bare_cr_for_ast(original_code), file_path, language,
    )
    if not ast_nodes:
        ast_nodes = get_ast_map(file_path, total_lines)

    # Find the target node (supports 'Class.method' qualification)
    target = _resolve_symbol(symbol, ast_nodes)

    if target is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Symbol '{symbol}' not found in {file_path}. "
            f"Available: {available}"
        )

    # For methods, delete just the method. For top-level items, find
    # enclosing boundaries to clean up properly.
    start_idx = target.line_start - 1  # 0-indexed
    end_idx = target.line_end           # exclusive

    # Strip ALL trailing blank lines that separated the deleted symbol from
    # whatever follows it. Consuming only a single blank line (the previous
    # behavior) left a stray extra blank line behind whenever the file used
    # a 2-blank-line separator style (e.g. PEP 8 top-level defs): the
    # leading blank lines already preserved before start_idx become the new
    # separator, so any trailing blanks left uncollapsed are pure surplus.
    # This was a silent, deterministic (exit 0) formatting corruption.
    while end_idx < total_lines and original_lines[end_idx].strip() == "":
        end_idx += 1

    result_lines = original_lines[:start_idx] + original_lines[end_idx:]
    merged_code = "".join(result_lines)
    lines_removed = end_idx - start_idx

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged_code, language)

    return DeleteResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        deleted_symbol=target.name,
        deleted_kind=target.kind,
        deleted_lines=(target.line_start, target.line_end),
        lines_removed=lines_removed,
    )


def move_symbol(
    file_path: str,
    symbol: str,
    after: str,
    language: str | None = None,
) -> MoveResult:
    """Move a function, method, or class to after another symbol.

    Pure deterministic operation -- no model inference. Uses the in-memory
    tree-sitter map (get_ast_map_from_source, same as delete_symbol) to find
    the exact line ranges and splices the code. Handles decorators, trailing
    blank lines, and proper spacing.

    Args:
        file_path: Path to the source file.
        symbol: Name of the symbol to move.
        after: Name of the symbol to insert after.
        language: Optional language for parse validation.

    Returns:
        MoveResult with the modified code.

    Raises:
        ValueError: If either symbol is not found, or they are the same.
    """
    from ..io_utils import read_source
    from ..split_join import detect_line_ending, normalize_bare_cr_for_ast

    if symbol == after:
        raise ValueError(f"Cannot move '{symbol}' after itself.")

    path = Path(file_path)
    # B21: strict decode -- see delete_symbol. For a UTF-8-BOM file the
    # utf-8-sig codec strips the BOM here; the writer restores it at the
    # true start of the file via the codec/BOM policy in _atomic_write.
    original_code, _ = read_source(path)
    # A UTF-8 BOM is a file-level marker, not part of line 1's content.
    # Strip it before splitting into lines so it can never ride along as
    # embedded text on whichever symbol happens to occupy line 1 -- _atomic_write
    # restores it at the true start of the file once the move is written.
    # (No-op on the normal read_source path -- utf-8-sig already stripped
    # it -- but kept for text that still carries a leading U+FEFF.)
    had_bom = original_code.startswith("﻿")
    if had_bom:
        original_code = original_code[1:]
    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)
    line_ending = detect_line_ending(original_code)

    # B35: in-memory AST — the same authoritative source delete_symbol uses.
    # The disk-based get_ast_map consults the tldr daemon, whose cache can
    # hold pre-write line numbers for a file that was just rewritten; a move
    # spliced from stale coordinates corrupts both the moved span and its
    # neighbours. A bare CR is swapped for LF by a same-length, same-position
    # substitution first (tree-sitter counts rows by scanning for "\n"), so
    # the returned line numbers stay valid against original_lines. B3: the
    # caller's `language` rides along as an explicit hint.
    ast_nodes = get_ast_map_from_source(
        normalize_bare_cr_for_ast(original_code), file_path, language,
    )
    if not ast_nodes:
        # Unsupported extension / missing grammar — fall back to the tldr
        # path as delete_symbol does.
        ast_nodes = get_ast_map(file_path, total_lines)

    # Find both nodes (supports 'Class.method' qualification)
    source_node = _resolve_symbol(symbol, ast_nodes)
    target_node = _resolve_symbol(after, ast_nodes)

    if source_node is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Symbol '{symbol}' not found in {file_path}. "
            f"Available: {available}"
        )
    if target_node is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Target '{after}' not found in {file_path}. "
            f"Available: {available}"
        )

    # Extract the source symbol's lines (0-indexed, exclusive end)
    src_start = source_node.line_start - 1
    src_end = source_node.line_end

    # Include trailing blank line separator if present
    if src_end < total_lines and original_lines[src_end].strip() == "":
        src_end += 1

    extracted = original_lines[src_start:src_end]

    # Remove source from original
    remaining = original_lines[:src_start] + original_lines[src_end:]

    # Recalculate target position in the remaining lines.
    # The target may have shifted if it was after the source.
    shift = src_end - src_start
    tgt_end_0 = target_node.line_end  # 1-indexed end -> 0-indexed exclusive
    if target_node.line_start > source_node.line_end:
        tgt_end_0 -= shift

    # Terminate the line preceding the insertion point when it lacks a
    # terminator: only the file's LAST line can be unterminated
    # (splitlines(keepends=True) keeps every other line terminated), and
    # splicing after it would concatenate that line with the moved block's
    # first line — silent corruption, exit 0 (a file whose final symbol
    # line carries no newline, then moving a symbol to EOF).
    if tgt_end_0 > 0 and not remaining[tgt_end_0 - 1].endswith(("\n", "\r")):
        remaining[tgt_end_0 - 1] += line_ending

    # Ensure blank line separator before inserted code
    if tgt_end_0 < len(remaining) and remaining[tgt_end_0 - 1].strip() != "" and extracted[0].strip() != "":
            extracted = [line_ending] + extracted

    # Ensure trailing blank line after inserted code
    if extracted and extracted[-1].strip() != "":
        extracted.append(line_ending)

    # Insert after target
    result_lines = remaining[:tgt_end_0] + extracted + remaining[tgt_end_0:]
    merged_code = "".join(result_lines)
    if had_bom:
        merged_code = "﻿" + merged_code

    # Calculate new position
    new_start = tgt_end_0 + 1  # 1-indexed
    new_end = new_start + (source_node.line_end - source_node.line_start)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged_code, language)

    return MoveResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        moved_symbol=source_node.name,
        moved_kind=source_node.kind,
        from_lines=(source_node.line_start, source_node.line_end),
        after_symbol=target_node.name,
        new_lines=(new_start, new_end),
    )


def batch_chunked_merge(
    original_code: str,
    edits: list[BatchEdit],
    file_path: str,
    merge_fn,
    language: str | None = None,
    padding: int = 30,
) -> ChunkedMergeResult:
    """Apply multiple edits to a file sequentially in one call.

    Each edit is applied via chunked_merge, with the result fed into the
    next edit. A temp file is used for AST analysis between edits so the
    original file is untouched until all edits succeed.

    Args:
        original_code: Full original file content.
        edits: List of BatchEdit operations to apply in order.
        file_path: Path to the file (for language detection / suffix).
        merge_fn: Callable(original_chunk, snippet, language) -> MergeResult.
        language: Optional language for validation.
        padding: Lines of context padding around edit regions.

    Returns:
        ChunkedMergeResult with all edits applied. ``chunks_rejected``
        accumulates every per-edit hallucination rejection so batch
        callers (MCP fast_batch_edit/fast_multi_edit, CLI) can apply the
        same fail-loud gates a single fast_edit applies.
    """
    # Import here to avoid circular import
    from .chunked_merge import _normalize_merged_eol, chunked_merge

    if not edits:
        return ChunkedMergeResult(
            merged_code=original_code,
            parse_valid=True,
            chunks_used=0,
            chunk_regions=[],
            model_tokens=0,
            latency_ms=0.0,
            chunks_rejected=0,
        )

    current_code = original_code
    total_tokens = 0
    total_latency = 0.0
    total_rejected = 0
    total_retries = 0
    all_regions: list[tuple[int, int]] = []

    # Temp file for AST analysis between edits
    suffix = Path(file_path).suffix
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8",
    ) as f:
        tmp_path = f.name
        f.write(current_code)

    try:
        for i, edit in enumerate(edits):
            result = chunked_merge(
                original_code=current_code,
                snippet=edit.snippet,
                file_path=tmp_path,
                merge_fn=merge_fn,
                language=language,
                padding=padding,
                after=edit.after,
                replace=edit.replace,
                preserve_siblings=edit.preserve_siblings,
            )
            current_code = result.merged_code
            total_tokens += result.model_tokens
            total_latency += result.latency_ms
            total_rejected += result.chunks_rejected
            total_retries += result.retries
            all_regions.extend(result.chunk_regions)

            # Update temp file for next edit's AST analysis
            if i < len(edits) - 1:
                Path(tmp_path).write_text(current_code, encoding="utf-8")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Step 14 (B41): single choke point for every batch caller (CLI
    # batch-edit/multi-edit and MCP fast_batch_edit/fast_multi_edit all
    # compose chunked_merge through here). Each per-edit result is already
    # funneled inside chunked_merge against ITS input; this final funnel
    # guarantees the assembled output is EOL- and trailing-newline-consistent
    # with the ORIGINAL file the batch started from — bare-LF pieces a
    # merge dropped can no longer survive to the write path. Idempotent on
    # an already-consistent result.
    current_code = _normalize_merged_eol(current_code, original_code)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(current_code, language)

    return ChunkedMergeResult(
        merged_code=current_code,
        parse_valid=parse_valid,
        chunks_used=len(all_regions),
        chunk_regions=all_regions,
        model_tokens=total_tokens,
        latency_ms=total_latency,
        chunks_rejected=total_rejected,
        retries=total_retries,
    )
