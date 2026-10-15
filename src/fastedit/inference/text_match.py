"""Deterministic text-match editing: zero model tokens.

Classifies snippet lines as context (matches original) or new (the edit).
Forward-scans through the original to find context positions, then splices
new lines between them.

PRESERVE-BY-DEFAULT semantics: a snippet declares an edit, it does not
license deletion. Unmentioned original lines always survive.

  - Leading section (before the first anchor): originals are preserved;
    leading new lines are inserted before the first anchor — except for
    the explicit signature/prefix-replacement case (below), where the
    leading new lines replace the original prefix in place.
  - Mid sections (between two anchors): the original gap is preserved
    verbatim, whitespace-only lines included. With a marker, new lines
    declared before the marker are inserted immediately after the
    preceding anchor (top of gap) and lines declared after the marker
    follow the gap; the marker itself never emits content. Without a
    marker the section is a protected insertion zone: new lines are
    emitted after the preserved gap, immediately before the next anchor.
  - Trailing section: the original suffix is ALWAYS preserved; trailing
    new lines are inserted immediately after the last anchor, before the
    preserved tail. A trailing marker is the coarse "keep everything
    after this point" idiom and is insertions-only.
  - In-place replacement is the ONLY sanctioned deletion: a new line
    replaces an original line when it restates it exactly (same
    normalized content AND indent, exactly one candidate), or — inside a
    mid marker-bearing section only — when it is uniquely identified by
    ``(replacement_key, indent)`` (the v0.2.3 contract).
  - AMBIGUOUS-REWRITE DECLINE: in a marker-free section a new line that
    looks like a rewrite of a preserved line (same assignment LHS key or
    same leading token, without restating it exactly) is ambiguous — the
    editor declines (returns None) rather than emitting either a
    duplicate or a deletion.
  - STRUCTURAL-BALANCE DECLINE: the merged span must not change the
    bracket balance of the original span; an imbalance caused by inserted
    lines (e.g. a duplicated closer) declines to the model path.

Falls back to None (model required) when <2 context anchors are found, when
any decline above fires, or when two candidate anchor bindings remain equally
valid after disambiguation (B24): each context line binds to one of ALL its
forward original occurrences — the one keeping the anchor sequence contiguous
with its neighbors and matching the snippet line's indent — and a tie between
bindings is a guess about where the edit belongs, so it is never resolved
silently.
"""

from __future__ import annotations

import logging

_log = logging.getLogger("fastedit.text_match")

# Marker detection lives in ONE module (B15 unification): the phrases,
# the short-form regexes, the canonical constants, the line-anchored
# predicate (``is_marker_line``) and the normalizer
# (``normalize_markers``) are defined in
# :mod:`fastedit.inference.markers` and imported here.
from .markers import (
    _CANONICAL_HASH_MARKER,
    _CANONICAL_SLASH_MARKER,
    _EXACT_SHORT_MARKERS,
    is_marker_line,
    normalize_markers,
)

# Backward-compatible module-level names (tests and external callers
# import these from ``text_match``) — thin aliases to the shared
# implementation, not second definitions.
_is_marker = is_marker_line
_normalize_markers = normalize_markers

# Lines that are too ambiguous to use as context anchors — they match
# too many positions and cause false anchors (e.g., closing braces).
_AMBIGUOUS_LINES = frozenset([
    "}", "{", "end", "]", ")", "];", "});", "});",
    "else:", "else {", "else", "pass", "break", "continue",
    "return", "return;", "return None", "return nil",
])

# Minimum non-whitespace characters for a line to be a context anchor.
_MIN_ANCHOR_LENGTH = 4


def _marker_snippet_placement_refusal(original_func: str, snippet: str) -> str:
    """Explain WHY a marker-bearing snippet cannot be placed (issue #12).

    ``deterministic_edit`` declined a snippet that carries a keep-marker,
    and the caller's refusal used to claim "no anchor line matched the
    original body" even when the anchor line EXISTS but is REPEATED across
    the symbol (a ``} catch {`` line twenty occurrences deep, say) — the
    one case where the fix is obvious from the data but the message said
    the opposite of what was true.

    This helper re-reads the same data the editor used and names the first
    actionable fact, with the remedy spelled out:

      * enough unique anchors matched but the editor still declined → the
        snippet is an ambiguous rewrite or changes the structural balance;
        the full replacement body is the way through;
      * a snippet line matches N > 1 original lines → that anchor is
        ambiguous; include more surrounding unique lines;
      * nothing matched → the classic no-anchor message (unchanged);
      * fewer than two unique anchors matched → say so (the editor's own
        floor).

    Diagnostic only — it never decides the edit; ``deterministic_edit``
    already declined, and this only makes the decline actionable.
    """
    original_counts: dict[str, int] = {}
    for line in original_func.splitlines():
        stripped = line.strip()
        if stripped:
            original_counts[stripped] = original_counts.get(stripped, 0) + 1

    unique_matches = 0
    matched_any = False
    repeated_line: str | None = None
    repeated_count = 0
    for line in snippet.splitlines():
        stripped = line.strip()
        if not stripped or is_marker_line(line):
            continue
        count = original_counts.get(stripped, 0)
        if count == 0:
            continue
        matched_any = True
        if count == 1:
            unique_matches += 1
        elif repeated_line is None:
            repeated_line, repeated_count = stripped, count

    if unique_matches >= 2:
        return (
            "the snippet matched the original body, but the deterministic "
            "editor still declined the merge — pass the full replacement "
            "body instead of a marker"
        )
    if repeated_line is not None:
        return (
            f"the anchor line {repeated_line!r} matches {repeated_count} "
            f"original lines, so its position is ambiguous — include more "
            f"surrounding unique lines in the snippet (or pass the full "
            f"replacement body instead of a marker)"
        )
    if not matched_any:
        return (
            "no anchor line matched the original body. Pass the full "
            "replacement body instead of a marker."
        )
    return (
        f"only {unique_matches} unique anchor line(s) matched the original "
        f"body — at least two are needed to place the edit; include more "
        f"surrounding unique lines in the snippet (or pass the full "
        f"replacement body instead of a marker)"
    )


def snippet_has_keep_marker(snippet: str) -> bool:
    """True when a snippet contains a line that IS a keep-marker.

    This predicate REFUSES to write a file, so its two failure directions are
    NOT symmetric and the design follows that asymmetry:

      * false POSITIVE -- refuses a valid snippet. Loud, recoverable: the user
        passes the full replacement body instead.
      * false NEGATIVE -- lets a partial snippet reach the direct-replacement
        splice, which deletes every original line the snippet did not restate.
        SILENT, exit 0, code lost.

    Where the two cannot both be satisfied, err toward refusing.

    THREE earlier versions were each MEASURED wrong, and every failure but the
    first was in the silent direction:

      * substring containment refused a valid body whose DOCSTRING merely
        mentioned the phrase -- proving it by refusing the very edit that fixed
        it, since this project documents markers in its own docstrings.
      * `startswith` on the stripped line missed a marker sitting after code;
        end-to-end that truncated a four-line body to one, exit 0.
      * a quote-in-prefix test missed BOTH a marker after an ordinary comment
        containing an apostrophe (`don't`, `won't`, `it's` -- routine English)
        and a marker after any earlier string literal on the same line.

    Each of those was a threshold guess. This is not: a marker only means
    anything inside a COMMENT, so find where the comment actually starts by
    scanning the line with quote state, then look for the marker after it.
    That is the property itself rather than a proxy for it.

    KNOWN LIMIT, accepted and deliberate: a marker alone on its own line inside
    a multi-line string is still treated as a marker, because seeing that needs
    a real parser. That is the LOUD direction -- it refuses a valid snippet
    rather than losing code -- and it is far rarer than the cases above.
    """

    def apostrophe_partner(line: str, start: int) -> int:
        """Index of the partner apostrophe for the one at ``start``, or -1.

        A partner counts only when it appears BEFORE any comment opener
        (``//`` or ``#``) on the line: a pairing that SPANS a comment
        opener is lexical noise — a Rust lifetime (``&'static``)
        accidentally paired with prose (``don't``) after the comment —
        not a string. Accepting such a pairing entered string mode,
        skipped the real ``//`` comment, and MISSED a genuine
        keep-marker, letting a partial snippet splice and drop code
        silently. (TRDD-CMRMA2YG, B42b)
        """
        j = start + 1
        while j < len(line):
            ch = line[j]
            if ch == "'":
                return j
            if ch == "#" or (
                ch == "/" and j + 1 < len(line) and line[j + 1] == "/"
            ):
                return -1
            j += 1
        return -1

    def comment_start(line: str) -> int:
        """Index where a comment opens on this line, ignoring quoted text."""
        quote = None
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch == '"' or (
                ch == "'" and apostrophe_partner(line, i) != -1
            ):
                quote = ch
                i += 1
                continue
            if ch == "#":
                return i
            if ch == "/" and i + 1 < n and line[i + 1] == "/":
                return i
            i += 1
        return -1

    for line in snippet.splitlines():
        if line.strip() in _EXACT_SHORT_MARKERS:
            return True
        start = comment_start(line)
        if start == -1:
            continue
        comment = line[start:]
        if _CANONICAL_HASH_MARKER in comment or _CANONICAL_SLASH_MARKER in comment:
            return True
    return False


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
    return len(stripped) < _MIN_ANCHOR_LENGTH


def _leading_token(line: str) -> str:
    """Extract the leading significant token of a line.

    The part before any whitespace / punctuation — ``return self.db.get()``
    → ``return``, ``if x > 1:`` → ``if``, ``data = clean(data)`` → ``data``.
    Used to detect a rewrite-shaped new line: one whose statement head
    matches a preserved original line, signaling a modification of that
    statement rather than an addition of a new peer line.
    """
    s = line.strip()
    if not s:
        return ""
    # Split on whitespace or opening brackets/parens
    i = 0
    while i < len(s) and s[i] not in " \t(){}[]<>=,;:":
        i += 1
    return s[:i]


def _indent_of(line: str) -> int:
    """Width of a line's leading whitespace, in characters."""
    return len(line) - len(line.lstrip())


def _indent_width(line: str) -> int:
    """Column width of a line's leading whitespace (tabs expand to 4).

    Unlike ``_indent_of`` — a raw character count — this measures indent
    the way editors render it, so tab-indented and space-indented lines
    can be compared in one unit when computing indent deltas.
    """
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip())


def _dominant_indent_char(lines: list[str], ref_line: str = "") -> str:
    """The span's dominant indent character: tab vs space.

    Counts non-blank lines whose leading whitespace STARTS with a tab vs
    a space; the majority wins. A tie — or a span with no indented lines
    at all — falls back to the reference anchor line's own style, and to
    spaces when that line has no leading whitespace either. (B8: the old
    first-character sniff of the *new* line salted space-indented
    fragments into tab-indented files and vice versa.)
    """
    tabs = sum(1 for ln in lines if ln[:1] == "\t")
    spaces = sum(1 for ln in lines if ln[:1] == " ")
    if tabs != spaces:
        return "\t" if tabs > spaces else " "
    return "\t" if ref_line[:1] == "\t" else " "


def _indent_prefix(width: int, indent_char: str) -> str:
    """Render ``width`` columns of indentation in the given style.

    Spaces render 1:1. Tabs render as whole LEVELS — one tab per 4-column
    band under ``expandtabs(4)`` — never as ``width`` literal tabs, which
    would treat a column count as a tab count and explode a 4-space body
    into four tabs (B8).
    """
    if indent_char == "\t":
        return "\t" * (width // 4)
    return " " * width


def _exact_restatement(line: str, preserved_lines: list[str]) -> list[int]:
    """Indices of ``preserved_lines`` that ``line`` restates exactly.

    A restatement matches on BOTH the normalized (stripped) content — the
    same comparison the classifier uses for context anchors — and the
    indent width. Such a line is an in-place replacement candidate, not a
    rewrite: the original already carries the content, so emitting the
    snippet line too would duplicate it.
    """
    target = line.strip()
    if not target:
        return []
    width = _indent_of(line)
    return [
        i for i, gl in enumerate(preserved_lines)
        if gl.strip() == target and _indent_of(gl) == width
    ]


def _has_rewrite_conflict(
    new_entries: list[tuple[str, int, int | None, str]],
    preserved_lines: list[str],
) -> bool:
    """AMBIGUOUS-REWRITE DECLINE predicate for marker-free sections.

    True when any section new line looks like a REWRITE of a preserved
    line rather than an addition: it shares the preserved line's
    assignment identity (``_replacement_key``) or its leading token
    (``return ...`` vs ``return ...``, ``if ...:`` vs ``if ...:``), and it
    does not restate the line exactly. Emitting both would duplicate the
    statement; dropping the original would be an unjustified deletion —
    both are guesses, so the caller declines and lets the model decide.

    Exact restatements are exempt: they are in-place replacements, not
    rewrites.
    """
    for entry in new_entries:
        line = entry[3]
        if not line.strip():
            continue
        if _exact_restatement(line, preserved_lines):
            continue
        key = _replacement_key(line)
        if key is not None and any(
            _replacement_key(gl) == key for gl in preserved_lines
        ):
            return True
        token = _leading_token(line)
        if token and any(
            _leading_token(gl) == token for gl in preserved_lines
        ):
            return True
    return False


def _bracket_balance(text: str) -> int:
    """Net ``()``, ``{}``, ``[]`` balance of ``text``, ignoring content
    inside string literals and comments.

    Line-local scanner: quote state resets per line, ``#`` and ``//``
    start a comment outside a string. This is deliberately simple — a
    safety net against inserted lines that duplicate or unbalance
    structural closers, not a parser. Keyword block closers (``end``,
    ``end function``) are intentionally NOT counted; grammars that need
    them fall through to the model path instead, which is the safe
    direction.
    """
    balance = 0
    for line in text.splitlines():
        quote: str | None = None
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in "\"'`":
                quote = ch
                i += 1
                continue
            if ch == "#" or (ch == "/" and i + 1 < n and line[i + 1] == "/"):
                break  # comment runs to end of line
            if ch in "([{":
                balance += 1
            elif ch in ")]}":
                balance -= 1
            i += 1
    return balance


def _adjust_indent(
    new_line: str,
    ref_orig_idx: int,
    ref_snip_idx: int,
    snip_raw: list[str],
    orig_lines: list[str],
    ref_shifted_right: bool = False,
) -> str:
    """Adjust indentation of a new line relative to the nearest context anchor.

    The line keeps its offset relative to the reference anchor as written
    in the snippet, re-based onto the anchor's effective output indent,
    and is rendered in the ORIGINAL span's dominant indent character
    (B8): a snippet written in the other whitespace style is converted,
    never mixed in, and widths are measured with ``expandtabs(4)``.

    KNOWN LIMITATION (scalar delta): one anchor-relative delta is applied
    per line, but the line's own indent is taken at face value — a
    snippet whose internal structure is itself inconsistent (mixed tab
    and space levels line-by-line) is converted per line, which is
    correct per line but cannot repair the snippet's structure.

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

    orig_width = _indent_width(ref_orig)
    snip_width = _indent_width(ref_snip)
    if ref_shifted_right:
        # Anchor was emitted at snip_width (deeper than orig) — treat the
        # effective output indent of the anchor as the snippet indent.
        effective_orig_width = snip_width
    else:
        effective_orig_width = orig_width
    indent_diff = effective_orig_width - snip_width

    curr_width = _indent_width(new_line)
    raw_target = curr_width + indent_diff
    if raw_target < 0:
        # Never emit a negative indent — floor at column 0 — but say so:
        # a silent clamp hides a snippet/anchor indent mismatch (B8).
        _log.warning(
            "Indent clamp: new line %r would take %d columns relative to "
            "its anchor; flooring at column 0",
            new_line.strip()[:40],
            raw_target,
        )
    target_width = max(0, raw_target)

    # Emit in the span's dominant indent character (B8). A tab-indented
    # file gets tab indentation — widths convert to whole tab levels, not
    # absolute column counts — and a snippet written in the other style
    # is converted rather than mixed in.
    indent_char = _dominant_indent_char(orig_lines, ref_orig)
    return _indent_prefix(target_width, indent_char) + new_line.lstrip()


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


def _position_mode_reference(
    orig_lines: list[str],
    snip_raw: list[str],
    new_entries: list[tuple[str, int, int | None, str]],
    signature_anchor: tuple[str, int, int | None, str] | None,
) -> tuple[int, int, str]:
    """Resolve the anchor reference for marker-position insertion.

    Returns ``(anchor_out_indent, anchor_snip_indent, indent_char)``:

    * With a context anchor (the signature line matched at the top of
      the span), the anchor IS the reference: each new line keeps its
      snippet offset from it, re-based onto the anchor's ORIGINAL indent
      — which is the indent the verbatim prefix actually emits at (B6).
    * With no context anchor at all there is nothing to be relative to;
      fall back to the previous convention — the snippet's own base (its
      shallowest new-line indent, the flush-left idiom) re-based onto the
      inferred body indent.
    """
    if signature_anchor is not None:
        anchor_orig = orig_lines[signature_anchor[2]]
        anchor_snip = snip_raw[signature_anchor[1]]
        return (
            _indent_width(anchor_orig),
            _indent_width(anchor_snip),
            _dominant_indent_char(orig_lines, anchor_orig),
        )
    non_blank = [e[3] for e in new_entries if e[3].strip()]
    body_indent, _ = _infer_body_indent(orig_lines)
    base = min((_indent_width(t) for t in non_blank), default=0)
    ref_line = orig_lines[0] if orig_lines else ""
    return (body_indent, base, _dominant_indent_char(orig_lines, ref_line))


def _reindent_new_lines(
    new_entries: list[tuple[str, int, int | None, str]],
    anchor_out_indent: int,
    anchor_snip_indent: int,
    indent_char: str,
) -> list[str]:
    """Re-indent snippet "new" lines relative to the reference anchor.

    Each new line keeps its offset RELATIVE to the reference anchor as
    written in the snippet (``line indent - anchor snippet indent``),
    re-based onto the anchor's effective output indent — the indent the
    anchor actually carries in the emitted span — and floored at column
    0. (B6: this used to base the group on the MINIMUM indent across the
    new lines, so one low-indent line — a flush-left comment, or content
    inside a multi-line string — dragged the base to 0 and over-indented
    every other new line relative to the anchor.)
    """
    out: list[str] = []
    for entry in new_entries:
        text = entry[3]
        if not text.strip():
            out.append("")
            continue
        rel = _indent_width(text) - anchor_snip_indent
        target = anchor_out_indent + rel
        if target < 0:
            _log.warning(
                "Indent clamp: new line %r would take %d columns relative "
                "to its anchor; flooring at column 0",
                text.strip()[:40],
                target,
            )
            target = 0
        out.append(_indent_prefix(target, indent_char) + text.lstrip())
    return out


def _emit_position_top(
    orig_lines: list[str],
    snip_raw: list[str],
    new_before: list[tuple[str, int, int | None, str]],
    signature_anchor: tuple[str, int, int | None, str] | None,
    original_func: str,
    prepend_signature_lines: int = 0,
) -> str:
    """Emit body with new lines at the TOP.

    Structure:
      * Signature span from the original (preserves decorators,
        multi-line defs and the body-opening delimiter — the pinned
        prepended-signature span covers original lines 0..N-1 (B9);
        without pinning we take everything up through the signature
        anchor's orig index, defaulting to line 0 if no anchor is
        present).
      * Re-indented new lines at body indent.
      * Remaining body lines verbatim.
    """
    sig_end = (signature_anchor[2] + 1) if signature_anchor else 1
    # B9: the pinned signature span must be emitted WHOLE — stopping at
    # the anchor's line would drop the continuation lines and the
    # body-opening delimiter from the output.
    sig_end = max(sig_end, 1, prepend_signature_lines)
    prefix = orig_lines[:sig_end]
    rest = orig_lines[sig_end:]

    anchor_out, anchor_snip, indent_char = _position_mode_reference(
        orig_lines, snip_raw, new_before, signature_anchor,
    )
    re_indented = _reindent_new_lines(
        new_before, anchor_out, anchor_snip, indent_char,
    )

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
    anchor_out, anchor_snip, indent_char = _position_mode_reference(
        orig_lines, snip_raw, new_after, signature_anchor,
    )
    re_indented = _reindent_new_lines(
        new_after, anchor_out, anchor_snip, indent_char,
    )

    result = list(orig_lines) + re_indented
    merged = "\n".join(result)
    if original_func.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"
    _log.info(
        "Text-match: marker-position BOTTOM insertion — %d new lines below body",
        len(re_indented),
    )
    return merged


def _pinned_signature_count(
    orig_lines: list[str],
    snip_raw: list[str],
    prepend_signature_lines: int,
) -> int:
    """Number of leading snippet lines that ARE the original's signature span.

    B9: when ``chunked_merge`` auto-prepends a multi-line signature
    (``def foo(\\n    a,\\n):``), the prepended continuation lines used to go
    through the ordinary forward scan, where they did real damage: short
    closers (``):``) fail the ambiguous-anchor rules and are classified
    "new" — emitted at column 0 alongside the marker-preserved original
    closer — and the longer continuation lines become body anchors that
    silently disable the marker-position insert path. When the caller
    passes ``prepend_signature_lines=N``, the first N snippet lines are
    PINNED context: matched verbatim to original lines 0..N-1 regardless
    of the ambiguous-anchor and minimum-length rules, never classified
    "new", never re-indented, never dropped. Body classification starts
    after the pinned span.

    The pin is VERIFIED against the original prefix before it is trusted:
    if any of the first N snippet lines does not match the original line
    at the same index (a fallback single-line signature for a grammar
    whose body opener sits mid-line, say), the prepend is not a
    line-aligned span of the original, so pinning is disabled entirely and
    the merge falls back to the un-pinned classification — which declines
    to the model instead of guessing.
    """
    n = min(prepend_signature_lines, len(snip_raw), len(orig_lines))
    if n <= 0:
        return 0
    for i in range(n):
        if snip_raw[i].strip() != orig_lines[i].strip():
            _log.info(
                "Prepended signature line %d does not match the original "
                "prefix; signature pinning disabled — falling back to "
                "ordinary classification",
                i + 1,
            )
            return 0
    return n


def _classify_forward_greedy(
    orig_stripped: list[str],
    orig_lines: list[str],
    snip_raw: list[str],
    pinned_count: int,
) -> list[tuple[str, int, int | None, str]]:
    """Legacy first-forward anchor binding (B24 fallback path).

    Used only when NO monotonic binding of the snippet's anchors exists at
    all — the snippet's anchor order contradicts the original, so the
    candidate-set disambiguation has nothing to choose between. This is
    the pre-B24 scan, preserved verbatim: first forward match wins,
    subject to the Fix-D indent-delta consistency check against the
    previous anchor; lines with no forward match classify as new.
    """
    classified: list[tuple[str, int, int | None, str]] = []
    orig_cursor = pinned_count

    for si, sl in enumerate(snip_raw):
        stripped = sl.strip()
        if si < pinned_count:
            # Pinned prepended-signature line (B9): context at the SAME
            # original index, bypassing the ambiguous-anchor and
            # minimum-length rules that would misclassify short closers
            # (``):``) as "new" lines and re-emit them at column 0.
            classified.append(("context", si, si, sl))
            continue
        if not stripped:
            classified.append(("blank", si, None, sl))
            continue
        if is_marker_line(sl):
            classified.append(("marker", si, None, sl))
            continue

        # Forward-scan: find this line in the original (order-preserving)
        # Skip ambiguous lines (e.g., lone `}`) as anchors ONLY when they
        # appear in the middle of the snippet (more lines after them).
        # At the end of the snippet, they're valid anchors (closing brace).
        found_idx = None
        remaining_significant = any(
            snip_raw[j].strip() and not is_marker_line(snip_raw[j])
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
                        ref_orig_indent = _indent_of(orig_lines[ref_ctx[2]])
                        ref_snip_indent = _indent_of(snip_raw[ref_ctx[1]])
                        expected_diff = ref_orig_indent - ref_snip_indent

                        orig_indent = _indent_of(orig_lines[oi])
                        snip_indent = _indent_of(sl)
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
    return classified


class _AmbiguousAnchorBinding(Exception):
    """Snippet anchors cannot be bound unambiguously (B24).

    ``_classify_edit_lines`` collects every forward original position a
    snippet context line could bind to; when more than one monotonic
    binding survives the contiguity and indent preferences, choosing one
    is a guess about WHERE the edit belongs. The classifier raises instead
    of guessing, and both consumers — the deterministic editor and the
    Step-6 faithfulness gate — treat the raise as a decline to the model
    path.
    """


def _lex_add(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Componentwise addition for the lexicographic binding-cost triple."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _resolve_anchor_binding(
    orig_lines: list[str],
    snip_raw: list[str],
    anchors: list[tuple[int, list[int]]],
) -> tuple[str, list[int] | None]:
    """Pick the snippet's monotonic anchor binding over ALL candidates (B24).

    Anchors are the snippet's context-eligible lines in order, each with
    every original position its content matches. A binding assigns one
    candidate per anchor, strictly increasing (the forward scan's
    order-preserving guarantee), and every consecutive pair must pass the
    Fix-D indent-delta consistency check that the old sequential scan
    applied against the previous anchor.

    Among the feasible bindings the optimum minimizes, lexicographically:

      1. lost adjacencies — transitions where the next anchor is not the
         very next original line, negated: the binding that keeps the
         snippet's anchor sequence contiguous (anchors landing together
         in one region) wins;
      2. total span — the distance from the first to the last bound line;
      3. indent mismatches — anchors whose original indent differs from
         the snippet line's (column width, tabs expanded): the occurrence
         written at the snippet's own indent wins.

    Ties are resolved by REGION, not silently:

      * Optimal bindings sharing the same first AND last anchor positions
        describe the same region — the residual ambiguity is only which
        duplicate line inside that region an anchor binds to, which
        cannot move the edit across regions. Those resolve to the
        earliest candidate per anchor (the legacy first-forward choice
        within the region), so byte-identical merges (repeated adjacent
        lines) keep working.
      * Optimal bindings with different region endpoints would place the
        edit in different parts of the original — choosing one is a guess,
        and the caller declines.

    Returns:
        ("ok", binding)      — a confident binding; ``binding[i]`` is the
                               original index for ``anchors[i]``.
        ("ambiguous", None)  — optimal bindings disagree on the region.
        ("infeasible", None) — no monotonic binding exists.
    """
    n = len(anchors)
    if n == 0:
        return "ok", []

    def _node_cost(i: int, o: int) -> tuple[int, int, int]:
        si = anchors[i][0]
        mismatch = (
            0
            if _indent_width(orig_lines[o]) == _indent_width(snip_raw[si])
            else 1
        )
        return (0, 0, mismatch)

    def _edge_cost(o: int, o2: int) -> tuple[int, int, int]:
        return (-1 if o2 == o + 1 else 0, o2 - o, 0)

    def _fix_d_ok(i: int, o: int, o2: int) -> bool:
        prev_si = anchors[i][0]
        si = anchors[i + 1][0]
        expected = _indent_of(orig_lines[o]) - _indent_of(snip_raw[prev_si])
        actual = _indent_of(orig_lines[o2]) - _indent_of(snip_raw[si])
        return abs(actual - expected) <= 2

    # suffix[i][o]: best cost of binding anchors i..n-1 with anchor i at o.
    suffix: list[dict[int, tuple[int, int, int]]] = [{} for _ in range(n)]
    for i in range(n - 1, -1, -1):
        cands = anchors[i][1]
        if i == n - 1:
            for o in cands:
                suffix[i][o] = _node_cost(i, o)
            continue
        next_suffix = suffix[i + 1]
        next_cands = anchors[i + 1][1]
        for o in cands:
            best: tuple[int, int, int] | None = None
            for o2 in next_cands:
                if o2 <= o or not _fix_d_ok(i, o, o2):
                    continue
                sub = next_suffix.get(o2)
                if sub is None:
                    continue
                cost = _lex_add(_edge_cost(o, o2), sub)
                if best is None or cost < best:
                    best = cost
            if best is not None:
                suffix[i][o] = _lex_add(_node_cost(i, o), best)

    # forward[i][o]: best cost of binding anchors 0..i with anchor i at o.
    forward: list[dict[int, tuple[int, int, int]]] = [{} for _ in range(n)]
    for i in range(n):
        cands = anchors[i][1]
        if i == 0:
            for o in cands:
                forward[i][o] = _node_cost(i, o)
            continue
        prev_forward = forward[i - 1]
        prev_cands = anchors[i - 1][1]
        for o in cands:
            best: tuple[int, int, int] | None = None
            for op in prev_cands:
                if o <= op or not _fix_d_ok(i - 1, op, o):
                    continue
                sub = prev_forward.get(op)
                if sub is None:
                    continue
                cost = _lex_add(
                    sub, _lex_add(_edge_cost(op, o), _node_cost(i, o)),
                )
                if best is None or cost < best:
                    best = cost
            if best is not None:
                forward[i][o] = best

    total = min(forward[n - 1].values(), default=None)
    if total is None:
        return "infeasible", None

    # A full binding through first-anchor o0 costs exactly suffix[0][o0];
    # one through last-anchor oN costs exactly forward[n-1][oN]. The sets
    # of optimal endpoints decide whether a tie crosses regions.
    first_options = sorted(
        o for o in anchors[0][1] if suffix[0].get(o) == total
    )
    last_options = sorted(
        o for o in anchors[n - 1][1] if forward[n - 1].get(o) == total
    )
    if not first_options or not last_options:  # defensive; unreachable
        return "infeasible", None
    if len(first_options) > 1 or len(last_options) > 1:
        return "ambiguous", None

    # Confident region — reconstruct the earliest (lexicographically
    # smallest) optimal binding: each anchor takes the smallest candidate
    # that still completes an optimal binding.
    binding: list[int] = [first_options[0]]
    o = first_options[0]
    for i in range(n - 1):
        node = _node_cost(i, o)
        total_i = suffix[i][o]
        rest = (total_i[0] - node[0], total_i[1] - node[1], total_i[2] - node[2])
        nxt = None
        for o2 in anchors[i + 1][1]:  # ascending → earliest candidate wins
            if o2 <= o or not _fix_d_ok(i, o, o2):
                continue
            sub = suffix[i + 1].get(o2)
            if sub is None:
                continue
            if _lex_add(_edge_cost(o, o2), sub) == rest:
                nxt = o2
                break
        if nxt is None:  # defensive: an optimal prefix cannot dead-end
            return "infeasible", None
        binding.append(nxt)
        o = nxt
    return "ok", binding


def _classify_edit_lines(
    orig_lines: list[str],
    snip_raw: list[str],
    pinned_count: int = 0,
) -> list[tuple[str, int, int | None, str]]:
    """Classify each snippet line for the deterministic editor (Step 1).

    Types: "context" (matches orig), "new" (the edit), "marker", "blank".
    Entries are ``(kind, snippet_index, original_index_or_None, raw_line)``.

    This is the SINGLE classifier for snippet binding. It is consumed by
    :func:`deterministic_edit` to synthesize the merge AND — via
    ``chunked_merge._deterministic_result_is_faithful`` — by the
    deterministic-path content gate, so the validator evaluates the editor's
    output against the same anchor/new/marker partition the editor used.
    Re-deriving a second, naive forward-scan binding inside the validator
    bound ambiguous structural lines (a mid-snippet lone ``}``) at a
    different original depth than the editor's ambiguous-anchor rule,
    producing phantom anchors that rejected faithful editor output (B3
    gate, Step 6). One classifier, two consumers — no drift.

    B24 anchor disambiguation: for every anchor-eligible snippet line the
    classifier collects ALL original positions the line could bind to —
    the old scan took the first forward match, and its Fix-D check only
    rejected indent deltas INCONSISTENT with the previous anchor, so a
    confidently-wrong-but-consistent repeat (``x = 1`` in two functions)
    dragged the edit to the wrong region. The binding is now chosen over
    the whole anchor set at once (see :func:`_resolve_anchor_binding`);
    when optimal bindings disagree on the region (different first/last
    anchor positions) the classifier raises :class:`_AmbiguousAnchorBinding`
    and both consumers decline to the model path. Ties INSIDE one region
    (repeated adjacent lines) resolve to the earliest candidates, and
    when no monotonic binding exists at all the legacy first-forward
    scan (:func:`_classify_forward_greedy`) applies unchanged.

    Moved verbatim from ``deterministic_edit`` (Step 1); binding upgraded
    in Step 17.
    """
    orig_stripped = [ln.strip() for ln in orig_lines]

    # Lines whose classification does not depend on the binding, plus the
    # anchor candidate sets. Pinned prepended-signature lines are anchors
    # with a single candidate (their own index, B9).
    decided: dict[int, tuple[str, int | None]] = {}
    anchors: list[tuple[int, list[int]]] = []

    for si, sl in enumerate(snip_raw):
        stripped = sl.strip()
        if si < pinned_count:
            anchors.append((si, [si]))
            continue
        if not stripped:
            decided[si] = ("blank", None)
            continue
        if is_marker_line(sl):
            decided[si] = ("marker", None)
            continue
        remaining_significant = any(
            snip_raw[j].strip() and not is_marker_line(snip_raw[j])
            for j in range(si + 1, len(snip_raw))
        )
        # Skip ambiguous lines (e.g., lone `}`) as anchors ONLY when they
        # appear in the middle of the snippet (more lines after them).
        # At the end of the snippet, they're valid anchors (closing brace).
        if _is_ambiguous_anchor(stripped) and remaining_significant:
            decided[si] = ("new", None)
            continue
        candidates = [
            oi for oi in range(pinned_count, len(orig_stripped))
            if orig_stripped[oi] == stripped
        ]
        if not candidates:
            decided[si] = ("new", None)
            continue
        anchors.append((si, candidates))

    status, binding = _resolve_anchor_binding(orig_lines, snip_raw, anchors)

    if status == "ambiguous":
        involved = ", ".join(
            f"line {si + 1} ({snip_raw[si].strip()!r})"
            for si, cands in anchors
            if len(cands) > 1
        ) or "the anchor sequence"
        raise _AmbiguousAnchorBinding(
            f"multiple equally valid original regions for {involved}"
        )
    if status == "infeasible":
        # No monotonic binding exists (the snippet's anchor order
        # contradicts the original); nothing was disambiguated, so keep
        # the legacy first-forward behavior for this snippet.
        return _classify_forward_greedy(
            orig_stripped, orig_lines, snip_raw, pinned_count,
        )

    bound = {
        si: oi
        for (si, _cands), oi in zip(anchors, binding, strict=True)
    }
    classified: list[tuple[str, int, int | None, str]] = []
    for si, sl in enumerate(snip_raw):
        if si in bound:
            classified.append(("context", si, bound[si], sl))
            continue
        kind, oi = decided[si]
        classified.append((kind, si, oi, sl))
    return classified


def deterministic_edit(
    original_func: str,
    snippet: str,
    max_drop_gap: int = 20,
    prepend_signature_lines: int = 0,
) -> str | None:
    """Apply an edit via pure text matching — no model needed.

    Preserve-by-default semantics (see the module docstring): unmentioned
    original lines always survive; deletions happen only through in-place
    replacement (exact restatement, or a unique ``(replacement_key,
    indent)`` match inside a mid marker-bearing section). When the merge
    would be a guess — an ambiguous rewrite of a preserved line, or a
    bracket-balance change caused by inserted lines — this returns None
    so the caller falls through to the model path.

    Args:
        original_func: The original function/symbol code.
        snippet: The edit snippet (context lines + new lines + optional markers).
        max_drop_gap: Deprecated and retained only for API compatibility.
            The old "drop gaps up to N lines" behavior is retired: gaps are
            preserved verbatim, and ambiguity declines to the model instead
            of dropping.
        prepend_signature_lines: When the caller auto-prepended the
            target's signature span (B9), the number of leading snippet
            lines that span occupies. Those lines are pinned to original
            lines 0..N-1 (see :func:`_pinned_signature_count`): matched
            verbatim regardless of anchor-ambiguity rules, never classified
            "new", never re-indented, never dropped; body classification
            and the body-anchor count start after the pinned span so the
            marker-position insert path stays reachable. 0 (default) means
            no prepend happened and classification runs unchanged.

    Returns:
        The edited function text, or None if text matching can't confidently
        apply the edit (not enough context anchors, ambiguous rewrite, or
        structural-balance change).
    """
    orig_lines = original_func.splitlines()
    snip_raw = snippet.splitlines()

    # ── B9: pinned prepended-signature span ──
    pinned_count = _pinned_signature_count(
        orig_lines, snip_raw, prepend_signature_lines,
    )

    # Step 1: Classify each snippet line (single classifier, shared with
    # the deterministic-path content gate — see _classify_edit_lines).
    try:
        classified = _classify_edit_lines(orig_lines, snip_raw, pinned_count)
    except _AmbiguousAnchorBinding as exc:
        # B24: two candidate anchor bindings remained equally valid after
        # sequence+indent disambiguation. Applying either would be a guess
        # about WHERE the edit belongs — decline to the model path.
        _log.info(
            "Text-match declined: ambiguous anchor binding (%s) — "
            "falling back to model",
            exc,
        )
        return None

    # Need at least 2 context anchors for confident matching
    context_entries = [c for c in classified if c[0] == "context"]

    # B9: with a pinned prepended signature, the span occupies original
    # lines 0..pinned_count-1 — those are signature, not body. Body anchors
    # are matches at or beyond the pinned span, so the prepended
    # continuation lines no longer disable the marker-position insert
    # path. Without pinning the legacy "anything beyond line 0" rule
    # applies.
    body_anchor_floor = max(pinned_count, 1)
    body_anchors = [
        c for c in context_entries if c[2] >= body_anchor_floor
    ]
    marker_entries = [c for c in classified if c[0] == "marker"]
    new_entries = [c for c in classified if c[0] == "new"]

    # ── Marker-position semantics (v0.2.4) ──
    # When the snippet has a marker but ZERO body anchors (only the
    # signature line is matched, if at all), infer position from marker
    # placement:
    #   * ``<new_lines> + marker`` (all new lines BEFORE the marker)
    #         → insert new_lines at the TOP of the function body.
    #   * ``marker + <new_lines>`` (all new lines AFTER the marker)
    #         → insert new_lines at the BOTTOM of the function body.
    # Abuse-resistance: requires ``body_anchors`` (context matches beyond
    # the signature span) to be empty. If the snippet has ANY overlapping
    # context with the body, the standard anchor-based path runs — guarding
    # against models accidentally dropping into position mode when they
    # meant something else. See module docstring / CHANGELOG 0.2.4.

    def _finish(merged: str) -> str | None:
        """Finalize a candidate merge — structural-balance gate + newline.

        STRUCTURAL-BALANCE DECLINE: verify the merged span does not change
        the bracket balance of the original span. Preserved lines are
        byte-identical and anchor re-emits are content-identical, so any
        difference is caused purely by inserted lines — e.g. a snippet
        whose trailing closers would duplicate the preserved ones, or a
        wrapper opener whose closer is missing. Both are rewrites the
        deterministic path cannot place safely, so decline and let the
        model path handle the wrap/rewrite. Bracket-only by design (see
        ``_bracket_balance``): exotic grammars decline here only when they
        also move brackets, and otherwise fall to the model via the other
        guards.
        """
        orig_balance = _bracket_balance(original_func)
        merged_balance = _bracket_balance(merged)
        if merged_balance != orig_balance:
            _log.info(
                "Text-match declined: inserted lines change bracket "
                "balance (%d -> %d) — falling back to model",
                orig_balance, merged_balance,
            )
            return None
        if original_func.endswith("\n") and not merged.endswith("\n"):
            merged += "\n"
        return merged

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
        # B9: the BODY starts after the pinned signature span — signature
        # lines are not body statements and must not feed the overlap
        # guard. Without pinning the legacy "skip line 0" range applies.
        body_leading_tokens = {
            _leading_token(orig_lines[i])
            for i in range(pinned_count or 1, len(orig_lines))
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
                    return _finish(_emit_position_top(
                        orig_lines, snip_raw, new_before,
                        signature_anchor=(
                            context_entries[0] if context_entries else None
                        ),
                        original_func=original_func,
                        prepend_signature_lines=pinned_count,
                    ))
            else:
                # Preserve signature (line 0 of original) if present,
                # then emit the new lines adjusted to body indent, then
                # the rest of the original body verbatim.
                return _finish(_emit_position_top(
                    orig_lines, snip_raw, new_before,
                    signature_anchor=(
                        context_entries[0] if context_entries else None
                    ),
                    original_func=original_func,
                    prepend_signature_lines=pinned_count,
                ))
        elif new_after and not new_before:
            # Pattern: marker + <new_lines> → insert at BOTTOM of body.
            return _finish(_emit_position_bottom(
                orig_lines, snip_raw, new_after,
                signature_anchor=(
                    context_entries[0] if context_entries else None
                ),
                original_func=original_func,
            ))
        # else: marker with no new lines → no-op edit; fall through.

    if len(context_entries) < 2:
        _log.info(
            "Text-match: only %d context anchor(s), need ≥2 — falling back to model",
            len(context_entries),
        )
        return None

    # NOTE: the old max_drop_gap safety loop is retired. Under
    # preserve-by-default a marker-free gap is never dropped — it is
    # preserved verbatim — so a large gap is no longer a decline signal
    # (and ``max_drop_gap`` is a deprecated no-op).

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

        # The original gap this section covers. Under preserve-by-default
        # it survives verbatim unless a line is explicitly replaced.
        gap_lines = orig_lines[ctx_orig + 1 : next_ctx_orig]
        section_new = [e for e in section if e[0] == "new"]

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
            marker_snip_indent = _indent_of(snip_raw[marker_entry[1]])
            # Find the first non-blank line in the original gap to measure
            # the body's current indent.
            first_gap_orig_indent = None
            for i in range(ctx_orig + 1, next_ctx_orig):
                if orig_lines[i].strip():
                    first_gap_orig_indent = _indent_of(orig_lines[i])
                    break
            # Baseline offset between snippet and original at the context
            # anchor (to cancel any uniform shift already present).
            ctx_orig_indent = _indent_of(orig_lines[ctx_orig])
            ctx_snip_indent = _indent_of(snip_raw[ctx_si])
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
            # Bug 1 fix (positive wrapper indent) — preserved-gap lines may
            # shift RIGHT only when the marker is indented deeper than the
            # preceding context anchor, i.e. the snippet explicitly opens a
            # wrapper block (try/except, if-guard, with-block) around the
            # preserved body. A marker at a SHALLOWER indent than the anchor
            # (e.g. the documented column-0 tail marker) is a view of the
            # code, never a license to move it.
            wrapper_shift = (
                indent_delta > 0 and marker_snip_indent > ctx_snip_indent
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
                        if wrapper_shift:
                            result.append(indent_char * indent_delta + gap_line)
                        else:
                            # Preserve the line at its ORIGINAL indent
                            # verbatim. The old negative-delta branch
                            # stripped leading indent here, dedenting
                            # preserved lines out of their block whenever
                            # the marker sat at a shallower indent than the
                            # body (B5) — a marker declares what to keep,
                            # never where to move it.
                            result.append(gap_line)
                    gap_emitted = True
                elif entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                        ref_shifted_right=anchor_shifted_right,
                    )
                    result.append(adjusted)
        else:
            # No marker: PROTECTED INSERTION ZONE (B2 fix). The gap is
            # preserved verbatim — a snippet without a marker never
            # licenses dropping it.
            if not section_new and any(gl.strip() for gl in gap_lines):
                # OMISSION DECLINE: the snippet restates both anchors but
                # says nothing about the gap content between them. That
                # shape implies a deletion the deterministic path will not
                # perform (preserve-by-default) and cannot safely ignore
                # (a silent no-op would report success without editing).
                # Decline so the model decides.
                _log.info(
                    "Text-match declined: marker-free section omits %d "
                    "preserved line(s) between anchors — falling back "
                    "to model",
                    sum(1 for gl in gap_lines if gl.strip()),
                )
                return None
            if _has_rewrite_conflict(section_new, gap_lines):
                _log.info(
                    "Text-match declined: marker-free section new line(s) "
                    "rewrite preserved line(s) (same assignment LHS or "
                    "leading token) — falling back to model",
                )
                return None
            # Alignment: a snippet line that EXACTLY restates a gap line
            # (same normalized content AND indent, exactly one candidate —
            # blanks align with blanks) pins that gap line to the snippet
            # position. The pinned original is emitted there in place of
            # the restatement; unpinned entries emit normally; gap lines
            # nobody restated keep their relative order around the pins.
            # This is the only sanctioned dedup: more than one candidate →
            # plain insertion/preservation, never a guess.
            pinned: dict[int, int] = {}  # gap idx -> entry idx
            used_entries: set[int] = set()
            for gi, gl in enumerate(gap_lines):
                candidates = [
                    ei for ei, e in enumerate(section)
                    if ei not in used_entries
                    and e[3].strip() == gl.strip()
                    and _indent_of(e[3]) == _indent_of(gl)
                ]
                if len(candidates) == 1:
                    pinned[gi] = candidates[0]
                    used_entries.add(candidates[0])
            first_pinned_gap = min(pinned, default=len(gap_lines))
            # Gap lines ahead of the first pin keep their original head
            # position (they precede everything the snippet declared).
            for gi in range(first_pinned_gap):
                result.append(gap_lines[gi])
            emitted_gap = set(range(first_pinned_gap))
            pinned_by_entry = {ei: gi for gi, ei in pinned.items()}
            for ei, entry in enumerate(section):
                gi = pinned_by_entry.get(ei)
                if gi is not None:
                    # In-place replacement: the original survives at the
                    # position the snippet gave it; the restatement is
                    # dropped (B32: byte-exact, whitespace-only included).
                    result.append(gap_lines[gi])
                    emitted_gap.add(gi)
                    continue
                if entry[0] == "blank":
                    # ── Fix A: Emit blank lines in sections ──
                    result.append("")
                elif entry[0] == "new":
                    adjusted = _adjust_indent(
                        entry[3], ctx_orig, ctx_si, snip_raw, orig_lines,
                        ref_shifted_right=anchor_shifted_right,
                    )
                    result.append(adjusted)
            for gi, gap_line in enumerate(gap_lines):
                if gi not in emitted_gap:
                    # Unrestated gap lines trail the additions (they were
                    # declared after the snippet's new content or never
                    # mentioned); byte-exact preservation (B32).
                    result.append(gap_line)
                    emitted_gap.add(gi)

    # Trailing section: entries after last context anchor. The original
    # suffix is ALWAYS preserved (B1 fix) — a snippet without a trailing
    # marker never licenses deleting it. Trailing new lines are inserted
    # immediately after the last anchor, before the preserved tail.
    last_ctx_si = context_entries[-1][1]
    trailing = [c for c in classified if c[1] > last_ctx_si]
    suffix_emitted = False
    suffix_lines = orig_lines[last_orig + 1:]
    trailing_new = [e for e in trailing if e[0] == "new"]

    # Bug 1 fix — compute indent delta for trailing-marker suffix (same logic
    # as the in-section marker branch). Applies when a wrapper shifts suffix
    # lines deeper, e.g. `try: # ... existing code ...` after the last anchor.
    trailing_marker = next(
        (e for e in trailing if e[0] == "marker"), None
    )
    if trailing_marker is not None:
        marker_snip_indent = _indent_of(snip_raw[trailing_marker[1]])
        first_suffix_orig_indent = None
        for i in range(last_orig + 1, len(orig_lines)):
            if orig_lines[i].strip():
                first_suffix_orig_indent = _indent_of(orig_lines[i])
                break
        last_ctx_orig = context_entries[-1][2]
        ctx_orig_indent_t = _indent_of(orig_lines[last_ctx_orig])
        ctx_snip_indent_t = _indent_of(snip_raw[last_ctx_si])
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
        # Bug 1 fix (positive wrapper indent) — same rule as the mid
        # section: preserved suffix lines shift RIGHT only when the marker
        # is indented deeper than the last context anchor (an explicit new
        # wrapper, e.g. `try:` before the marker). A marker at a shallower
        # indent (the documented column-0 tail marker) is a view of the
        # code, never a license to move it — and never a negative dedent.
        trailing_wrapper_shift = (
            trailing_indent_delta > 0
            and marker_snip_indent > ctx_snip_indent_t
        )
    else:
        trailing_indent_delta = 0
        trailing_indent_char = " "
        trailing_wrapper_shift = False

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

    if trailing_marker is not None:
        # A trailing marker preserves the suffix and is INSERTIONS-ONLY.
        # The marker is the coarse "everything after this point stays"
        # idiom (the documented middle-edit shape, tools_edit.py): its
        # author is not enumerating the suffix line-by-line, so a new line
        # sharing an LHS with a suffix line is an addition, not a rewrite
        # — key-based replacement never applies here (B2 fix). This is
        # what makes `data = validate / data = audit / #...` insert audit
        # while clean survives.
        for entry in trailing:
            if entry[0] == "marker" and not suffix_emitted:
                for line in suffix_lines:
                    if not line.strip():
                        # Byte-exact preservation of blank/whitespace-only
                        # suffix lines (B32).
                        result.append(line)
                        continue
                    if trailing_wrapper_shift:
                        result.append(
                            trailing_indent_char * trailing_indent_delta
                            + line
                        )
                    else:
                        # Original indent, verbatim — no negative-delta
                        # dedent (B5).
                        result.append(line)
                suffix_emitted = True
            elif entry[0] == "new":
                # Exact restatement of a preserved suffix line (same
                # normalized content AND indent, exactly one candidate) is
                # an in-place replacement: the original already carries
                # the content, so the restatement is skipped.
                if len(_exact_restatement(entry[3], suffix_lines)) == 1:
                    continue
                adjusted = _adjust_indent(
                    entry[3], last_orig, last_ctx_si, snip_raw, orig_lines,
                    ref_shifted_right=last_anchor_shifted_right,
                )
                result.append(adjusted)
            elif entry[0] == "blank" and not suffix_emitted:
                # Emit blanks only before suffix (after suffix, originals have blanks)
                result.append("")
    else:
        # No marker: preserve-by-default (B1 fix). The suffix ALWAYS
        # survives verbatim; trailing new lines are inserted immediately
        # after the last anchor, before the preserved tail. A new line
        # that looks like a rewrite of a suffix line (same assignment LHS
        # or leading token, without restating it) is an ambiguous rewrite:
        # decline rather than emit a duplicate or a deletion. (Duplicated
        # closers from a re-tail snippet are caught by the structural-
        # balance gate in ``_finish``.)
        if _has_rewrite_conflict(trailing_new, suffix_lines):
            _log.info(
                "Text-match declined: trailing new line(s) rewrite the "
                "preserved suffix (same assignment LHS or leading token) "
                "— falling back to model",
            )
            return None
        for entry in trailing:
            if entry[0] == "new":
                if len(_exact_restatement(entry[3], suffix_lines)) == 1:
                    continue
                adjusted = _adjust_indent(
                    entry[3], last_orig, last_ctx_si, snip_raw, orig_lines,
                    ref_shifted_right=last_anchor_shifted_right,
                )
                result.append(adjusted)
            elif entry[0] == "blank":
                result.append("")
        # Byte-exact suffix preservation, whitespace-only lines included.
        result.extend(suffix_lines)

    merged = "\n".join(result)
    n_new = sum(1 for c in classified if c[0] == "new")
    finished = _finish(merged)
    if finished is not None:
        _log.info(
            "Text-match succeeded: %d context anchors, %d new lines, %d orig lines",
            len(context_entries), n_new, len(orig_lines),
        )
    return finished
