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
)


def delete_symbol(
    file_path: str,
    symbol: str,
    language: str | None = None,
) -> DeleteResult:
    """Delete a function, method, or class from a file using AST line ranges.

    Pure deterministic operation — no model inference. Uses in-memory
    tree-sitter (via ``get_ast_map_from_source``) for authoritative line
    ranges. Works across all 16 languages supported by tree-sitter.

    Prefers the in-memory path over ``get_ast_map`` (which shells out to
    ``tldr structure``) because tldr's structure extractor has a known
    bug where ``@decorator``-wrapped Python functions are mis-reported:
    the decorated function's span is missing and the NEXT function
    inherits its line numbers, silently corrupting deletes. See
    docs/testing-matrix.md for the repro. The in-memory walker correctly
    spans the ``decorated_definition`` AST node including the decorator.

    Args:
        file_path: Path to the source file.
        symbol: Name of the function, method, or class to delete.
        language: Optional language for parse validation.

    Returns:
        DeleteResult with the modified code.

    Raises:
        ValueError: If the symbol is not found in the file's AST.
    """
    from .ast_utils import get_ast_map_from_source

    path = Path(file_path)
    original_code = path.read_text(encoding="utf-8", errors="replace")
    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    # In-memory AST — bypasses tldr's decorator-span bug.
    ast_nodes = get_ast_map_from_source(original_code, file_path)
    if not ast_nodes:
        # Fall back to the tldr-backed path for any language / parser
        # combination the in-memory walker doesn't support yet.
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

    # Strip trailing blank lines that belong to the deleted symbol
    while end_idx < total_lines and original_lines[end_idx].strip() == "":
        end_idx += 1
        # Only consume one trailing blank line as separator
        break

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

    Pure deterministic operation — no model inference. Uses tldr to find
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
    if symbol == after:
        raise ValueError(f"Cannot move '{symbol}' after itself.")

    path = Path(file_path)
    original_code = path.read_text(encoding="utf-8", errors="replace")
    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

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
    tgt_end_0 = target_node.line_end  # 1-indexed end → 0-indexed exclusive
    if target_node.line_start > source_node.line_end:
        tgt_end_0 -= shift

    # Ensure blank line separator before inserted code
    if tgt_end_0 < len(remaining) and remaining[tgt_end_0 - 1].strip() != "" and not extracted[0].startswith("\n"):
            extracted = ["\n"] + extracted

    # Ensure trailing blank line after inserted code
    if extracted and extracted[-1].strip() != "":
        extracted.append("\n")

    # Insert after target
    result_lines = remaining[:tgt_end_0] + extracted + remaining[tgt_end_0:]
    merged_code = "".join(result_lines)

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
        ChunkedMergeResult with all edits applied.
    """
    # Import here to avoid circular import
    from .chunked_merge import chunked_merge

    if not edits:
        return ChunkedMergeResult(
            merged_code=original_code,
            parse_valid=True,
            chunks_used=0,
            chunk_regions=[],
            model_tokens=0,
            latency_ms=0.0,
        )

    current_code = original_code
    total_tokens = 0
    total_latency = 0.0
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
            all_regions.extend(result.chunk_regions)

            # Update temp file for next edit's AST analysis
            if i < len(edits) - 1:
                Path(tmp_path).write_text(current_code, encoding="utf-8")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

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
    )
