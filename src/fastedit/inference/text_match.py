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
import re

_log = logging.getLogger("fastedit.text_match")

_MARKER_PHRASES = ("... existing code ...", "// ...", "# ...")

# Canonical long-form markers the rest of the pipeline understands.
_CANONICAL_HASH_MARKER = "# ... existing code ..."
_CANONICAL_SLASH_MARKER = "// ... existing code ..."

# Short-form markers (v0.2.4): recognized by ``_normalize_markers`` and
# rewritten to the canonical long form before any downstream processing.
#
# Detection (stripped line):
#   - Exact: ``#...``, ``//...``, ``…`` (Unicode ellipsis U+2026)
#   - Legacy long forms still match via substring in ``_is_marker``.
#   - Generic regex catches spacing variants (e.g. ``# ...``, ``// ..``).
#
# The rule is intentionally permissive on the short side — a lone
# ``#...`` on a line is vanishingly unlikely to be real code.
_SHORT_HASH_RE = re.compile(r"^\s*#\s*\.\.\.\s*$")
_SHORT_SLASH_RE = re.compile(r"^\s*//\s*\.\.\.\s*$")
_UNICODE_ELLIPSIS_RE = re.compile(r"^\s*…\s*$")


def _normalize_markers(snippet: str) -> str:
    """Rewrite short/Unicode marker forms to the canonical long form.

    Accepts (per line, after leading-whitespace strip):

      * ``#...``              → ``# ... existing code ...``
      * ``//...``             → ``// ... existing code ...``
      * ``…`` (U+2026)    → ``# ... existing code ...`` (hash form;
        the substring is recognized by ``_is_marker`` regardless of
        language, so one canonical form suffices)

    Legacy long-form markers (``# ... existing code ...``,
    ``// ... existing code ...``) are passed through unchanged.

    Indentation is preserved on the rewritten line so downstream indent
    arithmetic in ``deterministic_edit`` (which uses marker snippet
    indent to infer gap-body indent deltas) continues to match the
    surrounding snippet lines.

    This is a pure string transform — no parsing, no I/O, no side
    effects. Safe to call on any input.
    """
    lines = snippet.splitlines(keepends=True)
    out: list[str] = []
    for raw in lines:
        # Separate the line body from its terminator so we can rewrite
        # the body without losing ``\n`` / ``\r\n``.
        if raw.endswith("\r\n"):
            body, term = raw[:-2], "\r\n"
        elif raw.endswith("\n"):
            body, term = raw[:-1], "\n"
        else:
            body, term = raw, ""

        indent_len = len(body) - len(body.lstrip())
        indent = body[:indent_len]
        stripped = body[indent_len:].rstrip()

        if stripped == "#...":
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif stripped == "//...":
            out.append(indent + _CANONICAL_SLASH_MARKER + term)
        elif stripped == "…":
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif _SHORT_HASH_RE.match(body) and "existing" not in body:
            # Covers spacing variants like ``# ...`` / ``#  ...`` that
            # aren't full legacy long-form markers. ``_is_marker``
            # already accepts these via substring match, but normalizing
            # here keeps the two code paths consistent and makes the
            # position-semantics check below simpler.
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        elif _SHORT_SLASH_RE.match(body) and "existing" not in body:
            out.append(indent + _CANONICAL_SLASH_MARKER + term)
        elif _UNICODE_ELLIPSIS_RE.match(body):
            out.append(indent + _CANONICAL_HASH_MARKER + term)
        else:
            out.append(raw)
    return "".join(out)

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


def _replacement_key(line: str) -> str | None:
    """Extract the LHS of an assignment-like line, for replacement matching.

    Returns a normalized key when the line is an assignment/binding whose
    LHS can be used to identify it as a potential replacement for a line in
    a marker-preserved gap. Returns ``None`` for lines that are not
    assignment-like (insertion semantics, not replacement).

    Handles common assignment forms across Python, JS/TS, Rust, Go, etc.:

      - ``self._data = {}``         → ``self._data``
      - ``x = 1``                   → ``x``
      - ``let x = 5``               → ``let x``
      - ``const x: number = 5``     → ``const x: number``
      - ``x += 1``                  → ``x`` (compound assignment)
      - ``x: int = 5``              → ``x: int``
      - ``name := "foo"``           → ``name`` (Go short-decl)

    Comparison operators (``==``, ``!=``, ``<=``, ``>=``) are NOT treated
    as assignments — they indicate the line is a condition, not a binding,
    and should never trigger replacement matching.
    """
    stripped = line.strip()
    if not stripped:
        return None
    # Split on the first assignment-like operator. Exclude comparison ops.
    # Simple scan: find first '=' that isn't part of '==' / '!=' / '<=' / '>='.
    eq_idx = -1
    i = 0
    while i < len(stripped):
        c = stripped[i]
        if c == "=":
            prev = stripped[i - 1] if i > 0 else ""
            nxt = stripped[i + 1] if i + 1 < len(stripped) else ""
            # Skip comparison ops: ==, !=, <=, >=, =>
            if prev in ("=", "!", "<", ">") or nxt == "=" or prev == "=" or nxt == ">":
                i += 1
                continue
            # Strip trailing compound-assignment char from LHS (+=, -=, *=, /=, %=, |=, &=, ^=, :=)
            lhs_end = i
            if lhs_end > 0 and stripped[lhs_end - 1] in "+-*/%|&^:":
                lhs_end -= 1
            eq_idx = lhs_end
            break
        i += 1
    if eq_idx <= 0:
        return None
    lhs = stripped[:eq_idx].strip()
    if not lhs:
        return None
    return lhs


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
    ref_shifted_right: bool = False,
) -> str:
    """Adjust indentation of a new line relative to the nearest context anchor.

    Args:
        new_line: The snippet-indent line to re-indent for the output.
        ref_orig_idx: Index in ``orig_lines`` of the reference context anchor.
        ref_snip_idx: Index in ``snip_raw`` of the reference context anchor.
        snip_raw: The full snippet, split into lines.
        orig_lines: The full original, split into lines.
        ref_shifted_right: When True, the reference anchor was re-indented to
            match its DEEPER snippet position (FASTEDIT-M13 context-anchor
            indent shift). The anchor's effective output indent is the snippet
            indent, so ``indent_diff`` becomes 0 and the new line emits at its
            snippet indent unchanged.
    """
    ref_orig = orig_lines[ref_orig_idx]
    ref_snip = snip_raw[ref_snip_idx]

    orig_indent = len(ref_orig) - len(ref_orig.lstrip())
    snip_indent = len(ref_snip) - len(ref_snip.lstrip())
    if ref_shifted_right:
        # Anchor was emitted at snip_indent (deeper than orig) — treat the
        # effective output indent of the anchor as the snippet indent.
        effective_orig_indent = snip_indent
    else:
        effective_orig_indent = orig_indent
    indent_diff = effective_orig_indent - snip_indent

    curr_indent = len(new_line) - len(new_line.lstrip())
    target_indent = max(0, curr_indent + indent_diff)

    # Preserve tab vs space indentation: prefer the new line's own indent
    # character when it's already indented (most snippets use consistent
    # whitespace), falling back to the original reference anchor's style.
    if new_line and new_line[0] == "\t":
        indent_char = "\t"
    elif ref_orig and ref_orig[0] == "\t":
        indent_char = "\t"
    else:
        indent_char = " "
    return indent_char * target_indent + new_line.lstrip()


def _infer_body_indent(orig_lines: list[str]) -> tuple[int, str]:
    """Infer body indentation of a function from its original lines.

    Returns ``(indent_count, indent_char)``. The indent char is ``\\t``
    when the first indented non-empty line starts with a tab, else a
    single space. Falls back to 4 spaces when the original has no
    indented lines (unusual — e.g. single-line body).
    """
    for ln in orig_lines[1:]:
        if ln.strip():
            stripped_len = len(ln) - len(ln.lstrip())
            if stripped_len > 0:
                indent_char = "\t" if ln[0] == "\t" else " "
                return stripped_len, indent_char
            # Non-indented non-blank line at body level is unusual but
            # possible (e.g. top-level free-standing snippet). Keep
            # looking for an indented anchor.
    return 4, " "


def _reindent_new_lines(
    new_entries: list[tuple[str, int, int | None, str]],
    target_indent: int,
    indent_char: str,
) -> list[str]:
    """Re-indent snippet "new" lines relative to a shared base indent.

    The smallest indent among non-blank new lines becomes the
    ``target_indent``; other lines preserve their relative offset. This
    preserves the internal structure of multi-line snippets (nested
    blocks, continuations) while aligning to the body's indent depth.
    """
    texts = [e[3] for e in new_entries]
    non_blank = [t for t in texts if t.strip()]
    if not non_blank:
        return list(texts)
    base = min(len(t) - len(t.lstrip()) for t in non_blank)
    out: list[str] = []
    for t in texts:
        if not t.strip():
            out.append("")
            continue
        curr = len(t) - len(t.lstrip())
        rel = curr - base
        out.append(indent_char * (target_indent + rel) + t.lstrip())
    return out


def _emit_position_top(
    orig_lines: list[str],
    snip_raw: list[str],
    new_before: list[tuple[str, int, int | None, str]],
    signature_anchor: tuple[str, int, int | None, str] | None,
    original_func: str,
) -> str:
    """Emit body with new lines at the TOP.

    Structure:
      * Signature line(s) from the original (preserves decorators,
        multi-line defs, etc. — we take everything up through the
        signature anchor's orig index, defaulting to line 0 if no
        anchor is present).
      * Re-indented new lines at body indent.
      * Remaining body lines verbatim.
    """
    sig_end = (signature_anchor[2] + 1) if signature_anchor else 1
    sig_end = max(sig_end, 1)
    prefix = orig_lines[:sig_end]
    rest = orig_lines[sig_end:]

    target_indent, indent_char = _infer_body_indent(orig_lines)
    re_indented = _reindent_new_lines(new_before, target_indent, indent_char)

    result = list(prefix) + re_indented + list(rest)
    merged = "\n".join(result)
    if original_func.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"
    _log.info(
        "Text-match: marker-position TOP insertion — %d new lines above body",
        len(re_indented),
    )
    return merged


def _emit_position_bottom(
    orig_lines: list[str],
    snip_raw: list[str],
    new_after: list[tuple[str, int, int | None, str]],
    signature_anchor: tuple[str, int, int | None, str] | None,
    original_func: str,
) -> str:
    """Emit body with new lines at the BOTTOM.

    Structure:
      * All original lines verbatim.
      * Re-indented new lines appended at body indent.
    """
    target_indent, indent_char = _infer_body_indent(orig_lines)
    re_indented = _reindent_new_lines(new_after, target_indent, indent_char)

    result = list(orig_lines) + re_indented
    merged = "\n".join(result)
    if original_func.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"
    _log.info(
        "Text-match: marker-position BOTTOM insertion — %d new lines below body",
        len(re_indented),
    )
    return merged


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

    # ── Marker-position semantics (v0.2.4) ──
    # When the snippet has a marker but ZERO body anchors (only the
    # signature line is matched, if at all), infer position from marker
    # placement:
    #   * ``<new_lines> + marker`` (all new lines BEFORE the marker)
    #         → insert new_lines at the TOP of the function body.
    #   * ``marker + <new_lines>`` (all new lines AFTER the marker)
    #         → insert new_lines at the BOTTOM of the function body.
    # Abuse-resistance: requires ``body_anchors`` (context matches beyond
    # line 0) to be empty. If the snippet has ANY overlapping context
    # with the body, the standard anchor-based path runs — guarding
    # against models accidentally dropping into position mode when they
    # meant something else. See module docstring / CHANGELOG 0.2.4.
    body_anchors = [c for c in context_entries if c[2] > 0]
    marker_entries = [c for c in classified if c[0] == "marker"]
    new_entries = [c for c in classified if c[0] == "new"]

    # Abuse-resistance helper: extract the leading significant token
    # of a line (the part before any whitespace / punctuation). Used to
    # detect "structural overlap" with the original body even when no
    # literal line matches — e.g. a snippet ``return {...modified}``
    # doesn't match any body line verbatim, but its ``return`` leading
    # token clearly signals a MODIFICATION of the existing return
    # statement rather than an ADDITION of a new peer line. In that
    # case position semantics would be wrong; fall through to the model.
    def _leading_token(line: str) -> str:
        s = line.strip()
        if not s:
            return ""
        # Split on whitespace or opening brackets/parens
        i = 0
        while i < len(s) and s[i] not in " \t(){}[]<>=,;:":
            i += 1
        return s[:i]

    if (
        len(body_anchors) == 0
        and len(marker_entries) == 1
        and len(new_entries) >= 1
    ):
        marker_si = marker_entries[0][1]
        new_before = [e for e in new_entries if e[1] < marker_si]
        new_after = [e for e in new_entries if e[1] > marker_si]

        # Structural-overlap guard: if any "new" line's leading token
        # (``return``, ``raise``, ``yield``, etc.) matches the leading
        # token of some ORIGINAL body line, the author likely means to
        # modify that line in place — defer to the model. Plain
        # identifiers and assignment targets are excluded from this
        # check (they're too common and genuinely new peer statements
        # often share identifiers with the body).
        #
        # Add-guard refinement (v0.2.5): flow tokens that sit INSIDE a
        # new block opener (``if X:``, ``if (y) {``) are part of the
        # new guard body, not modifications of the existing flow. We
        # detect this by indent: if a flow-token new line is indented
        # deeper than some earlier new line ending in ``:`` / ``{``,
        # it's nested inside a new block and doesn't count toward
        # overlap. Only flow tokens at the OUTER-most new indent level
        # can plausibly modify the original body in place.
        _FLOW_TOKENS = frozenset({
            "return", "raise", "yield", "throw", "panic!",
            "break", "continue", "goto",
        })
        body_leading_tokens = {
            _leading_token(orig_lines[i])
            for i in range(1, len(orig_lines))
            if orig_lines[i].strip()
        }

        def _is_nested_in_new_opener(
            entry: tuple[str, int, int | None, str],
        ) -> bool:
            """True if this new-line is indented inside an earlier new
            opener (``... :`` or ``... {``) within ``new_entries``."""
            si_e, line = entry[1], entry[3]
            line_indent = len(line) - len(line.lstrip())
            for other in new_entries:
                if other[1] >= si_e:
                    break
                other_line = other[3]
                if not other_line.strip():
                    continue
                other_stripped = other_line.rstrip()
                if not other_stripped.endswith((":", "{")):
                    continue
                other_indent = len(other_line) - len(other_line.lstrip())
                if line_indent > other_indent:
                    return True
            return False

        new_leading_tokens = {
            _leading_token(e[3])
            for e in new_entries
            if e[3].strip() and not _is_nested_in_new_opener(e)
        }
        overlap = (
            (new_leading_tokens & body_leading_tokens) & _FLOW_TOKENS
        )
        if overlap:
            _log.info(
                "Text-match: position-mode declined — new-line leading "
                "token(s) %s overlap body flow tokens; falling through",
                overlap,
            )
        elif new_before and new_after:
            # Ambiguous: new lines flank the marker without any body
            # anchors to pin the position. Don't guess — fall through to
            # the standard "< 2 anchors" rejection so the model can
            # decide semantically.
            _log.info(
                "Text-match: marker-position mode ambiguous "
                "(new lines on both sides of marker, no body anchors) "
                "— falling back to model",
            )
        elif new_before and not new_after:
            # Pattern: <new_lines> + marker → insert at TOP of body.
            #
            # Guard against ``wrap_block`` false positives. If the LAST
            # new-line before the marker ends with a block opener
            # (``:`` in Python, ``{`` in C-family — where ``{`` is the
            # final non-whitespace token, i.e. actually opening a block
            # rather than closing one), we *might* be wrapping the
            # preserved body in a new scope. But add-guard patterns —
            # inserting an early-return or validation block at the top
            # of a function — also begin with a ``:``/``{`` opener and
            # are NOT wrap_block. Distinguishing signal: indent
            # alignment between the block-opener and the marker.
            #
            #   * marker_indent  >  opener_indent → genuine wrap_block
            #     (marker sits INSIDE the opened scope). Fall through
            #     to the model; deterministic path can't emit correctly.
            #
            #   * marker_indent <= opener_indent → add-guard pattern
            #     (opener's body lives entirely within ``new_before``;
            #     marker is a parallel peer, not wrapped). Proceed with
            #     top-insertion semantics — this is the common case
            #     (early-return guards, input validation).
            last_new_line = new_before[-1][3]
            last_new_stripped = last_new_line.rstrip()
            looks_like_opener = last_new_stripped.endswith((":", "{"))
            if looks_like_opener:
                opener_indent = len(last_new_line) - len(last_new_line.lstrip())
                marker_line = snip_raw[marker_si]
                marker_indent = len(marker_line) - len(marker_line.lstrip())
                if marker_indent > opener_indent:
                    _log.info(
                        "Text-match: position-TOP declined — trailing "
                        "new-line %r is a block opener and marker is "
                        "nested deeper (marker_indent=%d > "
                        "opener_indent=%d); genuine wrap_block, "
                        "falling through",
                        last_new_stripped, marker_indent, opener_indent,
                    )
                    # fall through to < 2 anchors rejection
                else:
                    _log.info(
                        "Text-match: position-TOP add-guard detected — "
                        "opener %r at indent %d, marker at indent %d "
                        "(parallel); proceeding with top insertion",
                        last_new_stripped, opener_indent, marker_indent,
                    )
                    return _emit_position_top(
                        orig_lines, snip_raw, new_before,
                        signature_anchor=(
                            context_entries[0] if context_entries else None
                        ),
                        original_func=original_func,
                    )
            else:
                # Preserve signature (line 0 of original) if present,
                # then emit the new lines adjusted to body indent, then
                # the rest of the original body verbatim.
                return _emit_position_top(
                    orig_lines, snip_raw, new_before,
                    signature_anchor=(
                        context_entries[0] if context_entries else None
                    ),
                    original_func=original_func,
                )
        elif new_after and not new_before:
            # Pattern: marker + <new_lines> → insert at BOTTOM of body.
            return _emit_position_bottom(
                orig_lines, snip_raw, new_after,
                signature_anchor=(
                    context_entries[0] if context_entries else None
                ),
                original_func=original_func,
            )
        # else: marker with no new lines → no-op edit; fall through.

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

    # Compute whether the FIRST context anchor is shifted right so that
    # leading new-line adjustments match the anchor's effective output
    # indent (FASTEDIT-M13).
    first_ctx_orig_idx = context_entries[0][2]
    first_ctx_si_idx = context_entries[0][1]
    first_anchor_orig_indent = (
        len(orig_lines[first_ctx_orig_idx])
        - len(orig_lines[first_ctx_orig_idx].lstrip())
    )
    first_anchor_snip_indent = (
        len(snip_raw[first_ctx_si_idx])
        - len(snip_raw[first_ctx_si_idx].lstrip())
    )
    first_anchor_shifted_right = (
        first_anchor_snip_indent > first_anchor_orig_indent
    )

    if leading_new and first_orig > 0:
        # The snippet has new lines before its first anchor AND the original
        # has lines before that anchor (prefix). This typically means the
        # snippet is replacing the prefix (e.g., modified signature).
        # Emit the leading new lines AS the prefix, not in addition to it.
        for entry in leading_new:
            adjusted = _adjust_indent(
                entry[3], context_entries[0][2], context_entries[0][1],
                snip_raw, orig_lines,
                ref_shifted_right=first_anchor_shifted_right,
            )
            result.append(adjusted)
    elif leading_new:
        # No prefix to replace — just insert leading new lines
        for entry in leading_new:
            adjusted = _adjust_indent(
                entry[3], context_entries[0][2], context_entries[0][1],
                snip_raw, orig_lines,
                ref_shifted_right=first_anchor_shifted_right,
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

        # Emit context line. In most cases this is `orig_lines[ctx_orig]`
        # verbatim, but when the snippet places this shared line at a
        # deeper indent than the original (e.g. wrap_block wraps a body
        # line in a new scope), re-indent by the per-anchor snip↔orig
        # delta. This is the context-anchor analogue of the M7 preserved-
        # gap indent shift (FASTEDIT-M13).
        #
        # We only apply POSITIVE deltas (shift right). Shifting left is
        # intentionally NOT performed: a snippet at a SHALLOWER indent is
        # typically a "view" of the code (e.g. a method extracted from a
        # class), where the user's intent is to merge back at the deeper
        # original indent — not to flatten the enclosing scope. This
        # preserves the semantics captured by
        # test_snippet_at_different_indent_adjusts_new_lines.
        orig_anchor_line = orig_lines[ctx_orig]
        snip_anchor_line = snip_raw[ctx_si]
        orig_anchor_indent = (
            len(orig_anchor_line) - len(orig_anchor_line.lstrip())
        )
        snip_anchor_indent = (
            len(snip_anchor_line) - len(snip_anchor_line.lstrip())
        )
        anchor_indent_delta = snip_anchor_indent - orig_anchor_indent

        anchor_shifted_right = anchor_indent_delta > 0
        if anchor_shifted_right:
            indent_char = (
                "\t" if orig_anchor_line.startswith("\t") else " "
            )
            result.append(
                indent_char * anchor_indent_delta + orig_anchor_line
            )
        else:
            # Zero or negative delta — preserve original indent.
            result.append(orig_anchor_line)

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
        marker_count = sum(1 for e in section if e[0] == "marker")

        if marker_count >= 2:
            # Bug 2 fix — Two or more markers in one section: the position of
            # any new line between them is genuinely ambiguous (which gap does
            # it precede?). Fall back to the model — it can infer semantic
            # placement from context.
            _log.info(
                "Text-match: %d markers in section — falling back to model",
                marker_count,
            )
            return None

        if has_marker:
            # Marker mode: keep original gap.
            # New lines go at their position relative to the marker:
            #   before marker → before gap, after marker → after gap
            # Don't emit blanks here — the marker preserves original blanks.
            #
            # Bug 1 fix — preserved-gap indent adjust: when a wrapper block
            # (try/except, if-guard, with-block) adds indent around existing
            # code, the marker in the snippet sits at a deeper indent than
            # the preserved body is at in the original. Shift gap lines by
            # the indent delta so they sit correctly inside the wrapper.
            #
            # Bug 3 fix (v0.2.3) — new-line-replaces-gap-line: when a "new"
            # line in the same section as a marker shares its LHS with a
            # line in the preserved gap (e.g. ``self._data = OrderedDict()``
            # vs ``self._data = {}``), the author's intent is to REPLACE
            # that gap line, not to insert a duplicate alongside it. Without
            # this, ``replace=<method>`` with a minimal partial snippet
            # silently emits BOTH the old and new assignment. We detect
            # these by comparing the ``_replacement_key`` of each new line
            # against each gap line and skipping matches when emitting the
            # gap. See regression tests in
            # ``tests/test_class_method_partial_replace.py``.
            marker_entry = next(e for e in section if e[0] == "marker")
            marker_snip_indent = (
                len(snip_raw[marker_entry[1]])
                - len(snip_raw[marker_entry[1]].lstrip())
            )
            # Find the first non-blank line in the original gap to measure
            # the body's current indent.
            first_gap_orig_indent = None
            for i in range(ctx_orig + 1, next_ctx_orig):
                if orig_lines[i].strip():
                    first_gap_orig_indent = (
                        len(orig_lines[i]) - len(orig_lines[i].lstrip())
                    )
                    break
            # Baseline offset between snippet and original at the context
            # anchor (to cancel any uniform shift already present).
            ctx_orig_indent = (
                len(orig_lines[ctx_orig]) - len(orig_lines[ctx_orig].lstrip())
            )
            ctx_snip_indent = (
                len(snip_raw[ctx_si]) - len(snip_raw[ctx_si].lstrip())
            )
            if first_gap_orig_indent is not None:
                indent_delta = (
                    (marker_snip_indent - ctx_snip_indent)
                    - (first_gap_orig_indent - ctx_orig_indent)
                )
            else:
                indent_delta = 0
            # Preserve tab vs space indent char based on orig context anchor.
            indent_char = (
                "\t" if orig_lines[ctx_orig].startswith("\t") else " "
            )

            # Build a map from replacement key → snippet indent for "new"
            # lines in this section. A gap line is considered REPLACED by
            # a new line only when all of:
            #   (1) the LHS matches,
            #   (2) the original gap indent matches the new-line indent, AND
            #   (3) exactly ONE gap line matches that (key, indent) pair.
            #
            # Without (3) a snippet like ``data = normalize(data)`` after
            # a marker would wrongly delete both ``data = validate(data)``
            # and ``data = transform(data)`` — the author's intent there is
            # to APPEND, not replace, because the LHS alone doesn't pin
            # down which line they meant. Without (2), wrap_block snippets
            # that rebind the same name inside a new scope (``cleaned =
            # None`` inside an ``except:`` at indent 8, vs ``cleaned =
            # clean(data)`` at indent 4 inside the preserved ``try:``
            # body) would wrongly drop the preserved line. Both guards
            # are load-bearing — see regression tests.
            new_line_key_indents: dict[str, set[int]] = {}
            for entry in section:
                if entry[0] == "new":
                    sl = entry[3]
                    key = _replacement_key(sl)
                    if key is not None:
                        new_line_key_indents.setdefault(key, set()).add(
                            len(sl) - len(sl.lstrip())
                        )

            # Count gap-line matches per (key, indent) so we can enforce
            # the exactly-one rule.
            gap_match_counts: dict[tuple[str, int], int] = {}
            for i in range(ctx_orig + 1, next_ctx_orig):
                gap_line = orig_lines[i]
                if not gap_line.strip():
                    continue
                gap_key = _replacement_key(gap_line)
                if gap_key is None or gap_key not in new_line_key_indents:
                    continue
                gap_indent = len(gap_line) - len(gap_line.lstrip())
                if gap_indent not in new_line_key_indents[gap_key]:
                    continue
                k = (gap_key, gap_indent)
                gap_match_counts[k] = gap_match_counts.get(k, 0) + 1

            gap_emitted = False
            for entry in section:
                if entry[0] == "marker" and not gap_emitted:
                    for i in range(ctx_orig + 1, next_ctx_orig):
                        gap_line = orig_lines[i]
                        if not gap_line.strip():
                            # Preserve blank lines as-is (no indent added).
                            result.append(gap_line)
                            continue
                        # Skip gap lines that are uniquely identified as
                        # being replaced by a "new" line in this section.
                        gap_key = _replacement_key(gap_line)
                        if gap_key is not None and gap_key in new_line_key_indents:
                            gap_indent = (
                                len(gap_line) - len(gap_line.lstrip())
                            )
                            if (
                                gap_indent in new_line_key_indents[gap_key]
                                and gap_match_counts.get(
                                    (gap_key, gap_indent), 0
                                ) == 1
                            ):
                                continue
                        if indent_delta > 0:
                            result.append(indent_char * indent_delta + gap_line)
                        elif indent_delta < 0:
                            # Strip up to |indent_delta| leading indent chars,
                            # capped at the line's existing leading indent.
                            strip = min(
                                -indent_delta,
                                len(gap_line) - len(gap_line.lstrip()),
                            )
                            result.append(gap_line[strip:])
                        else:
                            result.append(gap_line)
                    gap_emitted = True
                elif entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                        ref_shifted_right=anchor_shifted_right,
                    )
                    result.append(adjusted)
        else:
            # No marker: drop original gap, emit new lines and blanks
            for entry in section:
                if entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                        ref_shifted_right=anchor_shifted_right,
                    )
                    result.append(adjusted)
                elif entry[0] == "blank":
                    # ── Fix A: Emit blank lines in sections ──
                    result.append("")

    # Trailing section: entries after last context anchor
    last_ctx_si = context_entries[-1][1]
    trailing = [c for c in classified if c[1] > last_ctx_si]
    suffix_emitted = False

    # Bug 1 fix — compute indent delta for trailing-marker suffix (same logic
    # as the in-section marker branch). Applies when a wrapper shifts suffix
    # lines deeper, e.g. `try: # ... existing code ...` after the last anchor.
    trailing_marker = next(
        (e for e in trailing if e[0] == "marker"), None
    )
    if trailing_marker is not None:
        marker_snip_indent = (
            len(snip_raw[trailing_marker[1]])
            - len(snip_raw[trailing_marker[1]].lstrip())
        )
        first_suffix_orig_indent = None
        for i in range(last_orig + 1, len(orig_lines)):
            if orig_lines[i].strip():
                first_suffix_orig_indent = (
                    len(orig_lines[i]) - len(orig_lines[i].lstrip())
                )
                break
        last_ctx_orig = context_entries[-1][2]
        ctx_orig_indent_t = (
            len(orig_lines[last_ctx_orig])
            - len(orig_lines[last_ctx_orig].lstrip())
        )
        ctx_snip_indent_t = (
            len(snip_raw[last_ctx_si]) - len(snip_raw[last_ctx_si].lstrip())
        )
        if first_suffix_orig_indent is not None:
            trailing_indent_delta = (
                (marker_snip_indent - ctx_snip_indent_t)
                - (first_suffix_orig_indent - ctx_orig_indent_t)
            )
        else:
            trailing_indent_delta = 0
        trailing_indent_char = (
            "\t" if orig_lines[last_ctx_orig].startswith("\t") else " "
        )
    else:
        trailing_indent_delta = 0
        trailing_indent_char = " "

    # Replacement keys+indents for "new" lines in the trailing section —
    # any suffix line with a matching (LHS, indent) that pins down
    # EXACTLY ONE candidate is skipped (bug 3 fix, same logic as the
    # in-section marker branch).
    trailing_new_key_indents: dict[str, set[int]] = {}
    for entry in trailing:
        if entry[0] == "new":
            sl = entry[3]
            key = _replacement_key(sl)
            if key is not None:
                trailing_new_key_indents.setdefault(key, set()).add(
                    len(sl) - len(sl.lstrip())
                )

    trailing_match_counts: dict[tuple[str, int], int] = {}
    for i in range(last_orig + 1, len(orig_lines)):
        line = orig_lines[i]
        if not line.strip():
            continue
        suffix_key = _replacement_key(line)
        if suffix_key is None or suffix_key not in trailing_new_key_indents:
            continue
        suffix_indent = len(line) - len(line.lstrip())
        if suffix_indent not in trailing_new_key_indents[suffix_key]:
            continue
        k = (suffix_key, suffix_indent)
        trailing_match_counts[k] = trailing_match_counts.get(k, 0) + 1

    def _emit_suffix_with_indent() -> None:
        for i in range(last_orig + 1, len(orig_lines)):
            line = orig_lines[i]
            if not line.strip():
                result.append(line)
                continue
            # Skip suffix lines uniquely identified as being replaced by
            # a "new" line in the trailing section.
            suffix_key = _replacement_key(line)
            if (
                suffix_key is not None
                and suffix_key in trailing_new_key_indents
            ):
                suffix_indent = len(line) - len(line.lstrip())
                if (
                    suffix_indent in trailing_new_key_indents[suffix_key]
                    and trailing_match_counts.get(
                        (suffix_key, suffix_indent), 0
                    ) == 1
                ):
                    continue
            if trailing_indent_delta > 0:
                result.append(
                    trailing_indent_char * trailing_indent_delta + line
                )
            elif trailing_indent_delta < 0:
                strip = min(
                    -trailing_indent_delta,
                    len(line) - len(line.lstrip()),
                )
                result.append(line[strip:])
            else:
                result.append(line)

    # Compute whether the LAST context anchor was shifted right so that
    # trailing new-line adjustments match the anchor's effective output
    # indent (FASTEDIT-M13).
    last_ctx_orig_idx = context_entries[-1][2]
    last_ctx_si_idx = context_entries[-1][1]
    last_anchor_orig_indent = (
        len(orig_lines[last_ctx_orig_idx])
        - len(orig_lines[last_ctx_orig_idx].lstrip())
    )
    last_anchor_snip_indent = (
        len(snip_raw[last_ctx_si_idx])
        - len(snip_raw[last_ctx_si_idx].lstrip())
    )
    last_anchor_shifted_right = (
        last_anchor_snip_indent > last_anchor_orig_indent
    )

    for entry in trailing:
        if entry[0] == "marker" and not suffix_emitted:
            _emit_suffix_with_indent()
            suffix_emitted = True
        elif entry[0] == "new":
            adjusted = _adjust_indent(
                entry[3], last_orig, last_ctx_si, snip_raw, orig_lines,
                ref_shifted_right=last_anchor_shifted_right,
            )
            result.append(adjusted)
        elif entry[0] == "blank" and not suffix_emitted:
            # Emit blanks only before suffix (after suffix, originals have blanks)
            result.append("")

    if not suffix_emitted:
        # If the snippet had trailing NEW lines (no marker), treat them as
        # REPLACING the orig suffix (analogous to the no-marker gap-drop
        # rule for mid-sections). Without this, a snippet that rewrites
        # the tail of a block (e.g. wrap_block with closing braces for a
        # new scope) would emit BOTH the new tail and the original tail —
        # duplicating closers. (FASTEDIT-M13.)
        has_trailing_new = any(e[0] == "new" for e in trailing)
        if not has_trailing_new:
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
