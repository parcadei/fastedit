"""Deterministic text-match editing: zero model tokens.

Classifies snippet lines as context (matches original) or new (the edit).
Forward-scans through the original to find context positions, then splices
new lines between them.

Markers control gap behavior:
  - With marker: keep original lines between context anchors.
    New lines go at their position relative to the marker
    (before marker = before gap, after marker = after gap).
  - Without marker: drop original lines (replace mode).

Falls back to None (model required) when <2 context anchors found or
when a large gap would be dropped without a marker (likely missing marker).
"""

from __future__ import annotations

import logging

_log = logging.getLogger("fastedit.text_match")

_MARKER_PHRASES = ("... existing code ...", "// ...", "# ...")

# Lines that are too ambiguous to use as context anchors — they match
# too many positions and cause false anchors (e.g., closing braces).
_AMBIGUOUS_LINES = frozenset([
    "}", "{", "end", "]", ")", "];", "});", "});",
    "else:", "else {", "else", "pass", "break", "continue",
    "return", "return;", "return None", "return nil",
])

# Minimum non-whitespace characters for a line to be a context anchor.
_MIN_ANCHOR_LENGTH = 4


def _is_marker(line: str) -> bool:
    return any(m in line for m in _MARKER_PHRASES)


def _is_ambiguous_anchor(stripped: str) -> bool:
    """Check if a stripped line is too short/common to be a reliable anchor."""
    if stripped in _AMBIGUOUS_LINES:
        return True
    if len(stripped) < _MIN_ANCHOR_LENGTH:
        return True
    return False


def _adjust_indent(
    new_line: str,
    ref_orig_idx: int,
    ref_snip_idx: int,
    snip_raw: list[str],
    orig_lines: list[str],
) -> str:
    """Adjust indentation of a new line relative to the nearest context anchor."""
    ref_orig = orig_lines[ref_orig_idx]
    ref_snip = snip_raw[ref_snip_idx]

    orig_indent = len(ref_orig) - len(ref_orig.lstrip())
    snip_indent = len(ref_snip) - len(ref_snip.lstrip())
    indent_diff = orig_indent - snip_indent

    curr_indent = len(new_line) - len(new_line.lstrip())
    target_indent = max(0, curr_indent + indent_diff)

    # Preserve tab vs space indentation from the original
    indent_char = "\t" if ref_orig and ref_orig[0] == "\t" else " "
    return indent_char * target_indent + new_line.lstrip()


def deterministic_edit(
    original_func: str,
    snippet: str,
    max_drop_gap: int = 20,
) -> str | None:
    """Apply an edit via pure text matching — no model needed.

    Args:
        original_func: The original function/symbol code.
        snippet: The edit snippet (context lines + new lines + optional markers).
        max_drop_gap: Maximum number of original lines to silently drop in a
            gap without a marker. If exceeded, returns None (fall back to model).

    Returns:
        The edited function text, or None if text matching can't confidently
        apply the edit (not enough context anchors or unsafe gap).
    """
    orig_lines = original_func.splitlines()
    orig_stripped = [ln.strip() for ln in orig_lines]
    snip_raw = snippet.splitlines()

    # Step 1: Classify each snippet line via forward scan
    # Types: "context" (matches orig), "new" (the edit), "marker", "blank"
    classified: list[tuple[str, int, int | None, str]] = []
    orig_cursor = 0

    for si, sl in enumerate(snip_raw):
        stripped = sl.strip()
        if not stripped:
            classified.append(("blank", si, None, sl))
            continue
        if _is_marker(sl):
            classified.append(("marker", si, None, sl))
            continue

        # Forward-scan: find this line in the original (order-preserving)
        # Skip ambiguous lines (e.g., lone `}`) as anchors ONLY when they
        # appear in the middle of the snippet (more lines after them).
        # At the end of the snippet, they're valid anchors (closing brace).
        found_idx = None
        remaining_significant = any(
            snip_raw[j].strip() and not _is_marker(snip_raw[j])
            for j in range(si + 1, len(snip_raw))
        )
        skip_as_ambiguous = _is_ambiguous_anchor(stripped) and remaining_significant

        if not skip_as_ambiguous:
            for oi in range(orig_cursor, len(orig_stripped)):
                if orig_stripped[oi] == stripped:
                    # ── Fix D: Indent consistency check ──
                    # If we already have a context anchor, verify that the
                    # indent relationship is consistent. A line at indent 8
                    # in the snippet matching a line at indent 4 in the
                    # original (when the first anchor pair has indent_diff=0)
                    # is likely a false match (reused line in a new block).
                    prev_contexts = [c for c in classified if c[0] == "context"]
                    if prev_contexts:
                        ref_ctx = prev_contexts[-1]
                        ref_orig_indent = len(orig_lines[ref_ctx[2]]) - len(orig_lines[ref_ctx[2]].lstrip())
                        ref_snip_indent = len(snip_raw[ref_ctx[1]]) - len(snip_raw[ref_ctx[1]].lstrip())
                        expected_diff = ref_orig_indent - ref_snip_indent

                        orig_indent = len(orig_lines[oi]) - len(orig_lines[oi].lstrip())
                        snip_indent = len(sl) - len(sl.lstrip())
                        actual_diff = orig_indent - snip_indent

                        # Allow ±2 tolerance for minor indent variations
                        if abs(actual_diff - expected_diff) > 2:
                            continue  # Skip this match, try next

                    found_idx = oi
                    orig_cursor = oi + 1
                    break

        if found_idx is not None:
            classified.append(("context", si, found_idx, sl))
        else:
            classified.append(("new", si, None, sl))

    # Need at least 2 context anchors for confident matching
    context_entries = [c for c in classified if c[0] == "context"]
    if len(context_entries) < 2:
        _log.info(
            "Text-match: only %d context anchor(s), need ≥2 — falling back to model",
            len(context_entries),
        )
        return None

    # Safety: reject if a large gap would be dropped without a marker
    for ci in range(len(context_entries) - 1):
        curr = context_entries[ci]
        next_ = context_entries[ci + 1]
        gap_size = next_[2] - curr[2] - 1
        if gap_size <= 0:
            continue

        # Check for marker between these two context anchors
        has_marker = any(
            c[0] == "marker"
            for c in classified
            if curr[1] < c[1] < next_[1]
        )

        if not has_marker and gap_size > max_drop_gap:
            _log.info(
                "Text-match: %d-line gap without marker between orig L%d and L%d "
                "(limit %d) — falling back to model",
                gap_size, curr[2], next_[2], max_drop_gap,
            )
            return None

    first_orig = context_entries[0][2]
    last_orig = context_entries[-1][2]

    # Step 2: Build result using section-based processing
    result: list[str] = []

    # ── Fix B: Handle modified first line (signature replacement) ──
    # If the snippet has "new" lines before the first context anchor,
    # check whether they are replacing the original prefix (e.g., modified
    # function signature). If the prefix and leading new lines overlap in
    # structure, treat the new lines as a replacement, not an addition.
    first_ctx_si = context_entries[0][1]
    leading_new = [e for e in classified if e[1] < first_ctx_si and e[0] == "new"]

    if leading_new and first_orig > 0:
        # The snippet has new lines before its first anchor AND the original
        # has lines before that anchor (prefix). This typically means the
        # snippet is replacing the prefix (e.g., modified signature).
        # Emit the leading new lines AS the prefix, not in addition to it.
        for entry in leading_new:
            adjusted = _adjust_indent(
                entry[3], context_entries[0][2], context_entries[0][1],
                snip_raw, orig_lines,
            )
            result.append(adjusted)
    elif leading_new:
        # No prefix to replace — just insert leading new lines
        for entry in leading_new:
            adjusted = _adjust_indent(
                entry[3], context_entries[0][2], context_entries[0][1],
                snip_raw, orig_lines,
            )
            result.append(adjusted)
    else:
        # No leading new lines — emit original prefix as-is
        result.extend(orig_lines[:first_orig])

    # Process sections between consecutive context anchors
    for ci in range(len(context_entries)):
        ctx = context_entries[ci]
        ctx_orig = ctx[2]
        ctx_si = ctx[1]

        # Emit context line (from original, preserving its indent)
        result.append(orig_lines[ctx_orig])

        if ci == len(context_entries) - 1:
            break  # trailing section handled below

        next_ctx = context_entries[ci + 1]
        next_ctx_orig = next_ctx[2]
        next_ctx_si = next_ctx[1]

        # Collect snippet entries in this section
        section = [
            c for c in classified
            if ctx_si < c[1] < next_ctx_si
        ]

        has_marker = any(e[0] == "marker" for e in section)

        if has_marker:
            # Marker mode: keep original gap.
            # New lines go at their position relative to the marker:
            #   before marker → before gap, after marker → after gap
            # Don't emit blanks here — the marker preserves original blanks.
            gap_emitted = False
            for entry in section:
                if entry[0] == "marker" and not gap_emitted:
                    for i in range(ctx_orig + 1, next_ctx_orig):
                        result.append(orig_lines[i])
                    gap_emitted = True
                elif entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                    )
                    result.append(adjusted)
        else:
            # No marker: drop original gap, emit new lines and blanks
            for entry in section:
                if entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                    )
                    result.append(adjusted)
                elif entry[0] == "blank":
                    # ── Fix A: Emit blank lines in sections ──
                    result.append("")

    # Trailing section: entries after last context anchor
    last_ctx_si = context_entries[-1][1]
    trailing = [c for c in classified if c[1] > last_ctx_si]
    suffix_emitted = False

    for entry in trailing:
        if entry[0] == "marker" and not suffix_emitted:
            result.extend(orig_lines[last_orig + 1:])
            suffix_emitted = True
        elif entry[0] == "new":
            adjusted = _adjust_indent(
                entry[3], last_orig, last_ctx_si, snip_raw, orig_lines,
            )
            result.append(adjusted)
        elif entry[0] == "blank" and not suffix_emitted:
            # Emit blanks only before suffix (after suffix, originals have blanks)
            result.append("")

    if not suffix_emitted:
        result.extend(orig_lines[last_orig + 1:])

    merged = "\n".join(result)
    # Preserve trailing newline
    if original_func.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"

    n_new = sum(1 for c in classified if c[0] == "new")
    _log.info(
        "Text-match succeeded: %d context anchors, %d new lines, %d orig lines",
        len(context_entries), n_new, len(orig_lines),
    )
    return merged
