"""Chunked merge: AST-guided windowed editing for large files.

Instead of asking the model to rewrite an entire 1000+ line file,
we use tree-sitter AST analysis (via tldr) to locate the edit region,
extract a small chunk around it, merge just that chunk, and splice back.

The model only ever sees ~100-300 lines, which is its sweet spot.
Works across all 16 languages supported by tldr.
"""

from __future__ import annotations

# --- Re-export all public types and functions for backward compatibility ---
# All existing `from .inference.chunked_merge import X` imports continue to work.
from .ast_utils import (  # noqa: F401
    ASTNode,
    BatchEdit,
    ChunkedMergeResult,
    ChunkRegion,
    DeleteResult,
    MoveResult,
    _qualified_symbol_names,
    _resolve_symbol,
    get_ast_map,
)
from .chunk_locator import (  # noqa: F401
    _find_enclosing_block,
    _find_enclosing_parent,
    _narrow_large_node,
    locate_chunks,
)
from .indent import (  # noqa: F401
    _align_snippet_indent,
    _escape_tags,
    _realign_output,
    _unescape_tags,
)
from .snippet_analysis import (  # noqa: F401
    _MARKER_RE,
    _extract_identifiers,
    _extract_snippet_names,
    _find_import_region,
    _find_insertion_region,
    _find_matching_nodes,
    _get_import_line_set,
    _get_snippet_definitions,
    _has_import_changes,
    _merge_overlapping_regions,
    _regex_extract_names,
    _split_snippet,
    _try_tldr_snippet_parse,
)
from .symbols import (  # noqa: F401
    batch_chunked_merge,
    delete_symbol,
    move_symbol,
)
from .text_match import deterministic_edit  # noqa: F401

# Phrases that mark "keep everything here" in snippets
_MARKER_PHRASES = ("... existing code ...", "// ...", "# ...")


def _is_marker_line(line: str) -> bool:
    """Check if a line is an ellipsis marker (not real code)."""
    return any(m in line for m in _MARKER_PHRASES)


def _check_hallucinations(
    original_chunk: str,
    merged_chunk: str,
    snippet: str,
) -> float:
    """Score merge quality: 1.0 = clean, lower = hallucinated.

    Combines two signals:
    - Anchor preservation: did the model keep lines it wasn't asked to change?
    - Invention rate: did the model add lines that exist in neither original nor snippet?

    Returns a score where <0.85 typically indicates hallucination.
    """
    orig_lines = original_chunk.splitlines()
    merged_lines = merged_chunk.splitlines()
    snippet_lines = snippet.splitlines()

    # Non-marker snippet lines = the intended changes
    snippet_code: set[str] = set()
    for sl in snippet_lines:
        s = sl.strip()
        if s and not _is_marker_line(sl):
            snippet_code.add(s)

    orig_set = {line.strip() for line in orig_lines if line.strip()}

    # Anchor lines: original lines NOT in the snippet's change set
    anchors: list[str] = []
    for ol in orig_lines:
        s = ol.strip()
        if not s:
            continue
        if s in snippet_code:
            continue
        anchors.append(s)

    if not anchors:
        return 1.0

    merged_set = {line.strip() for line in merged_lines}
    survived = sum(1 for a in anchors if a in merged_set)
    anchor_score = survived / len(anchors)

    # Invention penalty: lines in output that don't come from original or snippet
    merged_real = [line.strip() for line in merged_lines if line.strip()]
    if merged_real:
        invented = sum(1 for m in merged_real if m not in orig_set and m not in snippet_code)
        invention_rate = invented / len(merged_real)
    else:
        invention_rate = 0.0

    # Marker leak: if the model left "... existing code ..." in output,
    # it didn't actually merge — it echoed the snippet placeholders
    marker_leak = sum(1 for m in merged_real if _is_marker_line(m))
    if marker_leak > 0:
        return max(0.0, anchor_score - invention_rate - 0.3)

    # Combined: anchor preservation minus invention penalty
    return max(0.0, anchor_score - invention_rate)


# ---------------------------------------------------------------------------
# Core merge function — the only logic that remains in this file
# ---------------------------------------------------------------------------

def _merge_preserve_siblings(
    original_code: str,
    snippet: str,
    file_path: str,
    replace: str,
    language: str | None,
) -> ChunkedMergeResult:
    """Replace `replace` (a class) with `snippet`, carrying over any named
    children of the original class that don't appear in the snippet.

    Pure-AST, zero-token operation. Used when the caller passes
    `preserve_siblings=True` on a `replace=` edit.

    The merged output contains:
      - the snippet's class shell and any members it defines
      - followed by every sibling (method, nested class, ...) present in
        the original class but missing from the snippet, spliced verbatim
        before the closing brace of the snippet.

    Sibling boundaries come from tldr's structure/extract output. Field
    declarations aren't exposed as separate nodes by tldr in Java/Kotlin/
    Swift/TS, which is why the field change lives in the snippet itself.
    """
    import logging

    _log = logging.getLogger("fastedit.chunked_merge")

    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    ast_nodes = get_ast_map(file_path, total_lines) or []
    target = _resolve_symbol(replace, ast_nodes)
    if target is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Symbol '{replace}' not found in {file_path}. "
            f"Available: {available}"
        )

    class_start = target.line_start  # 1-indexed
    class_end = target.line_end      # 1-indexed inclusive

    # Collect named children of the class: any AST node strictly within
    # the class's line span, excluding the class itself.
    child_nodes = [
        n for n in ast_nodes
        if n.name != replace
        and n.line_start >= class_start
        and n.line_end <= class_end
    ]

    # Parse the snippet to discover which children it names.
    snippet_child_names: set[str] = set()
    snippet_nodes = _get_snippet_definitions(snippet, language)
    for n in snippet_nodes:
        if n.name != replace:
            snippet_child_names.add(n.name)

    # Missing = children in original but not mentioned by the snippet.
    missing = [n for n in child_nodes if n.name not in snippet_child_names]

    # Re-indent the snippet so its class header matches the original's
    # class header indent level.
    class_first_line = original_lines[class_start - 1] if class_start - 1 < total_lines else ""
    indent = class_first_line[: len(class_first_line) - len(class_first_line.lstrip())]

    snippet_lines = snippet.splitlines(keepends=True)
    snippet_first_nonblank = next(
        (ln for ln in snippet_lines if ln.strip()), class_first_line
    )
    snippet_indent = snippet_first_nonblank[
        : len(snippet_first_nonblank) - len(snippet_first_nonblank.lstrip())
    ]
    if snippet_indent != indent:
        new_snippet_lines: list[str] = []
        for line in snippet_lines:
            if line.strip() and line.startswith(snippet_indent):
                new_snippet_lines.append(indent + line[len(snippet_indent):])
            else:
                new_snippet_lines.append(line)
        snippet_lines = new_snippet_lines

    # Strip ellipsis marker lines from the snippet — preserve_siblings
    # subsumes their role.
    snippet_lines = [ln for ln in snippet_lines if not _MARKER_RE.match(ln)]

    # Build preserved blocks from the original (in original source order).
    preserved_blocks: list[str] = []
    for child in sorted(missing, key=lambda n: n.line_start):
        cstart = child.line_start - 1  # 0-indexed
        cend = child.line_end          # exclusive
        preserved_blocks.append("".join(original_lines[cstart:cend]))

    # Locate the closing brace line of the snippet's class body. We
    # assume languages with `{ ... }` class syntax (Java/Kotlin/Swift/TS
    # all qualify). Scan from the end for the first line that is exactly
    # `}` (after whitespace).
    close_idx: int | None = None
    for i in range(len(snippet_lines) - 1, -1, -1):
        if snippet_lines[i].strip() == "}":
            close_idx = i
            break

    if close_idx is None:
        raise ValueError(
            "preserve_siblings=True requires a class body with a `}` closing "
            "brace in the snippet — no such line found. This path currently "
            "supports Java/Kotlin/Swift/TypeScript (and similar brace-delimited "
            "languages)."
        )

    assembled: list[str] = []
    assembled.extend(snippet_lines[:close_idx])
    # Blank-line separator before preserved siblings if the snippet doesn't
    # already end with a blank line.
    if preserved_blocks and assembled and assembled[-1].strip() != "":
        assembled.append("\n")
    for i, block in enumerate(preserved_blocks):
        assembled.append(block)
        # Blank-line separator between preserved siblings (not after last).
        if i < len(preserved_blocks) - 1 and not block.endswith("\n\n"):
            assembled.append("\n")
    assembled.extend(snippet_lines[close_idx:])

    # Splice the assembled class body back into the original file.
    result_lines = list(original_lines)
    result_lines[class_start - 1:class_end] = assembled
    merged = "".join(result_lines)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged, language)
        if not parse_valid:
            _log.warning(
                "preserve_siblings produced parse-invalid output for "
                "replace='%s' in %s (language=%s)",
                replace, file_path, language,
            )

    _log.info(
        "preserve_siblings for replace='%s' (L%d-L%d): "
        "%d preserved sibling(s), 0 model tokens",
        replace, class_start, class_end, len(preserved_blocks),
    )
    return ChunkedMergeResult(
        merged_code=merged,
        parse_valid=parse_valid,
        chunks_used=0,
        chunk_regions=[],
        model_tokens=0,
        latency_ms=0.0,
        chunks_rejected=0,
    )


def chunked_merge(
    original_code: str,
    snippet: str,
    file_path: str,
    merge_fn,
    language: str | None = None,
    padding: int = 30,
    after: str | None = None,
    replace: str | None = None,
    preserve_siblings: bool = False,
) -> ChunkedMergeResult:
    """Merge a snippet into a large file using chunked extraction.

    Args:
        original_code: Full original file content.
        snippet: The edit snippet.
        file_path: Path to the file (for AST extraction).
        merge_fn: Callable(original_chunk, snippet, language) -> MergeResult.
        language: Optional language for validation.
        padding: Lines of context padding around edit regions.
        after: Optional symbol name — insert new code after this function/class.
        replace: Optional symbol name — replace this function/class/method entirely.
        preserve_siblings: When True alongside `replace=ClassName`, carry over
            any named sibling members (methods, nested classes) that exist in
            the original class but aren't mentioned in the snippet. Lets you
            edit a subset of a class's members without enumerating the rest.
            Only valid with `replace=`; raises ValueError otherwise.

    Returns:
        ChunkedMergeResult with the fully merged file.
    """
    # preserve_siblings is only meaningful on a `replace=` edit. Fail
    # early and loudly so callers don't silently no-op.
    if preserve_siblings and not replace:
        raise ValueError(
            "preserve_siblings=True requires replace=ClassName. "
            "The flag controls how `replace=` behaves when the snippet "
            "describes only a subset of the class's members."
        )

    if preserve_siblings and replace:
        return _merge_preserve_siblings(
            original_code=original_code,
            snippet=snippet,
            file_path=file_path,
            replace=replace,
            language=language,
        )

    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    import logging
    _log = logging.getLogger("fastedit.chunked_merge")

    # Fast path: `after` means pure text insertion — no model needed.
    # The snippet IS the new code; just splice it after the anchor symbol.
    if after:
        ast_nodes = get_ast_map(file_path, total_lines)
        anchor_node = _resolve_symbol(after, ast_nodes or [])
        if anchor_node is None:
            available = _qualified_symbol_names(ast_nodes or [])
            raise ValueError(
                f"Symbol '{after}' not found in {file_path}. "
                f"Available: {available}"
            )

        # Insert after the anchor's last line
        anchor_end = anchor_node.line_end  # 1-indexed
        before = original_lines[:anchor_end]
        after_lines = original_lines[anchor_end:]

        # Align snippet indent to match the anchor's indent level.
        anchor_start_idx = anchor_node.line_start - 1
        anchor_first_line = original_lines[anchor_start_idx] if anchor_start_idx < total_lines else ""
        snippet_text = snippet.rstrip("\n") + "\n"
        snippet_text = _align_snippet_indent(snippet_text, anchor_first_line)
        snippet_parts = snippet_text.splitlines(keepends=True)

        # Ensure blank line separator before and after the new code
        separator = ["\n"] if before and before[-1].strip() != "" else []
        trailing = ["\n"] if after_lines and after_lines[0].strip() != "" else []

        result_lines = before + separator + snippet_parts + trailing + after_lines
        merged = "".join(result_lines)

        parse_valid = True
        if language:
            from ..data_gen.ast_analyzer import validate_parse
            parse_valid = validate_parse(merged, language)

        _log.info(
            "Fast-path insert after '%s' (L%d): %d snippet lines, 0 model tokens",
            after, anchor_end, len(snippet_parts),
        )
        return ChunkedMergeResult(
            merged_code=merged,
            parse_valid=parse_valid,
            chunks_used=0,
            chunk_regions=[],
            model_tokens=0,
            latency_ms=0.0,
        )

    # Guard: `replace=X` means "replace X with the snippet". The snippet
    # must therefore define at most X itself — not X plus additional new
    # symbols. Multi-symbol snippets under `replace=` silently force the
    # model to extend the chunk with code it has no draft for, which
    # breaks speculative decoding and burns minutes of AR generation.
    # If you want to replace X AND add Y, use fast_batch_edit with two
    # entries: {replace: 'X', ...} and {after: 'X', snippet: 'def Y...'}.
    if replace:
        snippet_names = _extract_snippet_names(snippet, language)
        extras = [n for n in snippet_names if n != replace]
        if extras:
            raise ValueError(
                f"replace='{replace}' snippet defines additional symbol(s) "
                f"{extras}. One fast_edit call targets one symbol. "
                f"Use fast_batch_edit to replace '{replace}' and add "
                f"{extras} in a single round-trip: "
                f"[{{'replace': '{replace}', 'snippet': '...'}}, "
                f"{{'after': '{replace}', 'snippet': 'def {extras[0]}...'}}]"
            )

        # Fast path: deterministic text-match — 0 model tokens, instant.
        # Classifies snippet lines as context (matches original) vs new (the edit),
        # then splices new lines between context anchors. Falls back to model
        # if <2 context anchors or unsafe gap detected.
        from .text_match import deterministic_edit

        ast_nodes = get_ast_map(file_path, total_lines)
        target_node = _resolve_symbol(replace, ast_nodes or [])
        if target_node:
            func_start = target_node.line_start - 1  # 0-indexed
            func_end = target_node.line_end  # 1-indexed inclusive
            original_func = "".join(original_lines[func_start:func_end])

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
                    from ..data_gen.ast_analyzer import validate_parse
                    parse_valid = validate_parse(merged, language)

                _log.info(
                    "Deterministic text-match for replace='%s': "
                    "0 model tokens, %d context anchors",
                    replace, sum(1 for _ in edited.splitlines()),
                )
                return ChunkedMergeResult(
                    merged_code=merged,
                    parse_valid=parse_valid,
                    chunks_used=0,
                    chunk_regions=[],
                    model_tokens=0,
                    latency_ms=0.0,
                )
            # Direct-swap fast-path: when deterministic_edit can't anchor
            # (every body line changed), but the snippet is a complete
            # re-definition of the target symbol, we can do a pure AST
            # boundary replacement — zero model tokens. Covers patterns
            # like change_signature, extend_literal, and full-function
            # wrap_block.
            #
            # Conditions:
            #   1. Snippet contains no marker lines — markers imply
            #      "preserve original chunks" which is inherently
            #      incompatible with whole-symbol swap.
            #   2. Snippet parses cleanly (tldr structure) and reports
            #      exactly one top-level definition.
            #   3. That definition's name equals `replace` (already
            #      verified by the extras-check above, but re-verify
            #      here against the parsed AST to guard against
            #      regex-only name extraction false positives).
            if not any(_is_marker_line(ln) for ln in snippet.splitlines()):
                from pathlib import Path as _Path
                snippet_parse = _try_tldr_snippet_parse(
                    snippet, _Path(file_path).suffix,
                )
                if (
                    snippet_parse
                    and len(snippet_parse) == 1
                    and snippet_parse[0] == replace
                ):
                    # Align snippet indent to match the target function's
                    # indent in the original.
                    anchor_first_line = (
                        original_lines[func_start]
                        if func_start < total_lines
                        else ""
                    )
                    snippet_text = snippet.rstrip("\n") + "\n"
                    snippet_text = _align_snippet_indent(
                        snippet_text, anchor_first_line,
                    )
                    snippet_lines = snippet_text.splitlines(keepends=True)
                    if snippet_lines and not snippet_lines[-1].endswith("\n"):
                        snippet_lines[-1] += "\n"

                    result_lines = list(original_lines)
                    result_lines[func_start:func_end] = snippet_lines
                    merged = "".join(result_lines)

                    parse_valid = True
                    if language:
                        from ..data_gen.ast_analyzer import validate_parse
                        parse_valid = validate_parse(merged, language)

                    _log.info(
                        "Direct-swap for replace='%s' (L%d-L%d): "
                        "0 model tokens, %d snippet lines",
                        replace, func_start + 1, func_end, len(snippet_lines),
                    )
                    return ChunkedMergeResult(
                        merged_code=merged,
                        parse_valid=parse_valid,
                        chunks_used=0,
                        chunk_regions=[],
                        model_tokens=0,
                        latency_ms=0.0,
                    )

            _log.info(
                "Deterministic text-match and direct-swap failed for "
                "replace='%s', falling back to model",
                replace,
            )

    chunks = locate_chunks(
        snippet, original_code, file_path, padding, language,
        replace=replace,
    )

    _log.info(
        "locate_chunks returned %d chunk(s) for %d-line file: %s",
        len(chunks), total_lines,
        [(c.start_line, c.end_line, c.matched_nodes) for c in chunks],
    )

    # Reject whole-file merge on large files — model will truncate
    _MAX_WHOLE_FILE_LINES = 150  # noqa: N806
    is_whole_file = (
        len(chunks) == 1
        and chunks[0].start_line == 1
        and chunks[0].end_line == total_lines
    )
    if is_whole_file and total_lines > _MAX_WHOLE_FILE_LINES:
        # Build a helpful list of available symbols
        ast_nodes = get_ast_map(file_path, total_lines)
        available = _qualified_symbol_names(ast_nodes or [])
        sym_hint = ""
        if available:
            preview = available[:8]
            sym_hint = (
                f" Available symbols: {preview}"
                + (f" (+{len(available) - 8} more)" if len(available) > 8 else "")
            )
        raise ValueError(
            f"Whole-file merge rejected: {total_lines} lines exceeds "
            f"{_MAX_WHOLE_FILE_LINES}-line safety limit. "
            f"Use `after='symbol'` to insert new code, or "
            f"`replace='symbol'` to modify lines inside an existing function "
            f"(context markers work — model sees only the function, not the whole file)."
            f"{sym_hint}"
        )

    # If only one chunk covering the whole file, just do a normal merge
    if is_whole_file:
        safe_code = _escape_tags(original_code)
        safe_snippet = _escape_tags(snippet)
        result = merge_fn(safe_code, safe_snippet, language)
        merged = _unescape_tags(result.merged_code)
        # Retry once on parse failure
        if language:
            from ..data_gen.ast_analyzer import validate_parse
            if not validate_parse(merged, language):
                _log.warning("Whole-file merge parse invalid, retrying once")
                retry = merge_fn(safe_code, safe_snippet, language)
                retry_merged = _unescape_tags(retry.merged_code)
                if validate_parse(retry_merged, language):
                    result = retry
                    merged = retry_merged
        return ChunkedMergeResult(
            merged_code=merged,
            parse_valid=result.parse_valid,
            chunks_used=1,
            chunk_regions=[(1, total_lines)],
            model_tokens=result.tokens_generated,
            latency_ms=result.latency_ms,
        )

    # For multi-chunk edits, split the snippet so each chunk only sees
    # its relevant portion (prevents import chunk from getting code, etc.)
    import_snippet = snippet
    code_snippet = snippet
    if len(chunks) > 1:
        has_import_chunk = any("<imports>" in c.matched_nodes for c in chunks)
        if has_import_chunk:
            import_snippet, code_snippet = _split_snippet(snippet, language)
            # If splitting produced empty parts, fall back to full snippet
            if not import_snippet.strip():
                import_snippet = snippet
            if not code_snippet.strip():
                code_snippet = snippet

    # Process each chunk independently and splice back.
    # Work backwards so line numbers don't shift.
    result_lines = list(original_lines)
    total_tokens = 0
    total_latency = 0.0
    rejected_chunks = 0
    chunk_regions = []

    for chunk in reversed(chunks):
        start_idx = chunk.start_line - 1  # 0-indexed
        end_idx = chunk.end_line           # exclusive

        chunk_text = "".join(original_lines[start_idx:end_idx])
        chunk_text = _escape_tags(chunk_text)

        # Use the appropriate snippet portion for this chunk
        if len(chunks) > 1 and "<imports>" in chunk.matched_nodes:
            chunk_snippet = import_snippet
        elif len(chunks) > 1:
            chunk_snippet = code_snippet
        else:
            chunk_snippet = snippet

        safe_chunk_snippet = _escape_tags(chunk_snippet)
        safe_chunk_snippet = _align_snippet_indent(safe_chunk_snippet, chunk_text)
        result = merge_fn(chunk_text, safe_chunk_snippet, language)

        # Retry once on parse failure
        merged_chunk_code = _unescape_tags(result.merged_code)
        if language:
            from ..data_gen.ast_analyzer import validate_parse
            if not validate_parse(merged_chunk_code, language):
                _log.warning(
                    "Chunk %d-%d parse invalid, retrying once",
                    chunk.start_line, chunk.end_line,
                )
                retry = merge_fn(chunk_text, safe_chunk_snippet, language)
                retry_code = _unescape_tags(retry.merged_code)
                if validate_parse(retry_code, language):
                    result = retry
                    merged_chunk_code = retry_code

        # Re-align output indent to match the original chunk.
        merged_chunk_code = _realign_output(merged_chunk_code, chunk_text)

        # Check for hallucinations: verify anchor lines survived.
        # If the model dropped too many original lines, retry once.
        raw_chunk = _unescape_tags(chunk_text)
        anchor_score = _check_hallucinations(raw_chunk, merged_chunk_code, chunk_snippet)
        if anchor_score < 0.85:
            _log.warning(
                "Chunk %d-%d anchor score %.1f%% — retrying (hallucination likely)",
                chunk.start_line, chunk.end_line, anchor_score * 100,
            )
            retry = merge_fn(chunk_text, safe_chunk_snippet, language)
            retry_code = _unescape_tags(retry.merged_code)
            retry_code = _realign_output(retry_code, chunk_text)
            retry_score = _check_hallucinations(raw_chunk, retry_code, chunk_snippet)
            if retry_score > anchor_score:
                merged_chunk_code = retry_code
                result = retry
                anchor_score = retry_score
                _log.info("Retry improved anchor score: %.1f%% → %.1f%%", anchor_score * 100, retry_score * 100)
            else:
                _log.info("Retry didn't improve (%.1f%%), keeping original", retry_score * 100)

        # If both attempts failed badly, keep the original chunk unchanged
        # rather than writing corrupted code to disk.
        if anchor_score < 0.5:
            _log.error(
                "Chunk %d-%d anchor score %.1f%% — rejecting edit (keeping original)",
                chunk.start_line, chunk.end_line, anchor_score * 100,
            )
            rejected_chunks += 1
            total_tokens += result.tokens_generated
            total_latency += result.latency_ms
            chunk_regions.append((chunk.start_line, chunk.end_line))
            continue

        total_tokens += result.tokens_generated
        total_latency += result.latency_ms

        merged_chunk_lines = merged_chunk_code.splitlines(keepends=True)
        if merged_chunk_lines and not merged_chunk_lines[-1].endswith("\n"):
            merged_chunk_lines[-1] += "\n"

        result_lines[start_idx:end_idx] = merged_chunk_lines
        chunk_regions.append((chunk.start_line, chunk.end_line))

    merged_code = "".join(result_lines)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged_code, language)

    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=len(chunks),
        chunk_regions=list(reversed(chunk_regions)),
        model_tokens=total_tokens,
        latency_ms=total_latency,
        chunks_rejected=rejected_chunks,
    )
