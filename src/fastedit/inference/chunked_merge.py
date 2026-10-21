"""Chunked merge: AST-guided windowed editing for large files.

Instead of asking the model to rewrite an entire 1000+ line file,
we use tree-sitter AST analysis (via tldr) to locate the edit region,
extract a small chunk around it, merge just that chunk, and splice back.

The model only ever sees ~100-300 lines, which is its sweet spot.
Works across all 16 languages supported by tldr.
"""

from __future__ import annotations

import difflib
import itertools
import math
import os
import re
from collections import Counter

from ..lang_attributes import format_for_path
from ..split_join import (
    detect_line_ending,
    mask_string_spans,
    normalize_line_endings,
)
from ..text_heuristics import (
    TOLERANCE_MODEL_PROSE,
    TextOp,
    text_traits,
    validate_text_output,
)

# --- Re-export all public types and functions for backward compatibility ---
# All existing `from .inference.chunked_merge import X` imports continue to work.
from .ast_utils import (  # noqa: F401
    _CALL_SYMBOL_SPECS,
    _FORMAT_SYMBOL_SPECS,
    ASTNode,
    BatchEdit,
    ChunkedMergeResult,
    ChunkRegion,
    DeleteResult,
    MoveResult,
    _call_symbol_snippet_top_level_names,
    _qualified_symbol_names,
    _resolve_symbol,
    _snippet_head_is_same_kind_definition,
    get_ast_map,
    get_ast_map_from_source,
)
from .chunk_locator import (  # noqa: F401
    _MAX_BLOCK_LINES,
    _TEXT_ANCHOR_TAG,
    _find_enclosing_block,
    _find_enclosing_parent,
    _narrow_large_node,
    _narrow_will_engage,
    _text_window_snippets,
    locate_chunks,
)
from .indent import (
    _align_snippet_indent,
    _align_snippet_to_symbol_indent,
    _escape_tags,
    _new_tag_nonce,
    _realign_output,
    _unescape_tags,
)
from .markers import (  # noqa: F401  -- re-exported for backward compat
    _MARKER_PHRASES,
    is_marker_line,
    is_marker_text,
    normalize_markers,
)
from .snippet_analysis import (  # noqa: F401
    _DEFINITION_PATTERNS,
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
    _top_level_extras,
    _try_tldr_snippet_parse,
)
from .symbols import (  # noqa: F401
    batch_chunked_merge,
    delete_symbol,
    move_symbol,
)
from .text_match import (  # noqa: F401 -- deterministic_edit re-exported
    _bracket_balance,
    _indent_width,
    _replacement_key,
    deterministic_edit,
)

# Marker detection is unified in :mod:`fastedit.inference.markers`
# (single source of truth, B15). ``_MARKER_PHRASES`` is re-exported for
# backward compatibility; ``_is_marker_line``/``_normalize_markers`` alias
# the shared LINE-ANCHORED predicate/normalizer — marker phrases embedded
# mid-line are real content, not markers.
_is_marker_line = is_marker_line
_normalize_markers = normalize_markers


def _snippet_has_target_signature(snippet: str, target_name: str) -> bool:
    """Check if the snippet contains a definition line naming ``target_name``.

    A ``replace=<name>`` edit normally expects the snippet to start with the
    target's signature (so ``deterministic_edit`` can align lines). But the
    signature is redundant information — the ``replace=`` kwarg already
    identifies the target structurally. This helper detects the common case
    where the caller omitted the signature, so the caller can prepend it
    from the AST and keep the existing line-matching logic intact.

    Matches against all languages FastEdit supports via the shared
    ``_DEFINITION_PATTERNS`` regexes (def/fn/func/fun/class/struct/...).
    A match requires both the definition keyword AND the captured name
    equalling ``target_name`` — otherwise a snippet that references another
    `def` in a comment or docstring would spuriously match.

    Falls back to a broader class-keyword check that tolerates language
    modifiers the shared regex does not enumerate (Java/C#/TS ``public``,
    ``private``, ``final``, etc.). Comment-prefixed lines are skipped to
    avoid false positives.

    B3: a final parenthesized-name check recognizes the brace-style
    function-definition line shapes no keyword pattern covers — bash's
    ``run_build() {`` and C/C++'s ``int beta(int y) {``. Without it a
    snippet that DOES restate such a signature is judged signature-less,
    the AST prepend doubles the line, and the direct-swap gate then
    declines the unbalanced splice to the model. The check requires the
    name's parenthesized parameter list to close the line (optionally
    followed by ``{``), so a call like ``save(x);`` never matches.
    """
    for line in snippet.splitlines():
        stripped = line.lstrip()
        is_comment = stripped.startswith(("//", "#", "*", "--"))
        for pattern in _DEFINITION_PATTERNS:
            m = pattern.search(line)
            if m and m.group(1) == target_name:
                return True
        if not is_comment:
            m = re.search(
                r"\b(?:class|struct|enum|trait|interface|protocol|"
                r"module|object|impl)\s+(\w+)",
                line,
            )
            if m and m.group(1) == target_name:
                return True
            m = re.search(
                r"(?:^|\s)([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:\{\s*)?$",
                line.rstrip(),
            )
            if m and m.group(1) == target_name:
                return True
    return False


def _tokens_declare_insertion_only(
    tokens: list[tuple[str, int | None, str | None]],
) -> bool:
    """True when the snippet's own shape declares a scope-introducing edit.

    A NEW line before the FIRST marker that opens a block (ends with ``:``
    or ``{``) introduces a scope the preserved body must live inside — the
    canonical wrap_block, or an add-guard. Such a snippet preserves the
    WHOLE existing body; it is an INSERTION-ONLY op, so the positional
    deletion fallback must never justify a dropped original in it. (C2
    stress defect: a wrap snippet's ``with audit_lock:`` counted as front
    positional capacity, so the model could drop the target's docstring and
    the validator ratified the mutation — the wrap then "converged" to a
    body that had silently lost a line.)

    Judged on the CLASSIFIED tokens (not the raw snippet) so a context
    anchor that happens to end with a colon — a restated ``def foo():``
    signature line — never triggers it, and on any new line before the
    marker rather than only the last (an add-guard's ``if not ready:``
    precedes its ``return``) — both are insertions; neither licenses
    deletions.
    """
    seen_marker = False
    for kind, _index, value in tokens:
        if kind == "marker":
            seen_marker = True
        elif (
            kind == "new"
            and not seen_marker
            and value
            and value.endswith((":", "{"))
        ):
            return True
    return False


def _check_hallucinations(
    original_chunk: str,
    merged_chunk: str,
    snippet: str,
    tokens: list[tuple[str, int | None, str | None]] | None = None,
    allow_complete_replacement: bool = False,
) -> float:
    """Score merge quality: 1.0 = clean, 0.0 = hallucinated.

    The validator follows ``deterministic_edit``'s content-level marker
    semantics, but intentionally does not reproduce that editor's anchor
    selection heuristics. ``deterministic_edit`` uses ambiguous-line and raw
    indentation checks to synthesize one output; this validator operates on
    normalized line content and accepts any model output satisfying the
    invariants below. Marker syntax legitimately permits both preservation
    and a locally justified replacement.

    The snippet is forward-scanned into context anchors (lines matching an
    as-yet-unconsumed original line) and new lines, delimited by
    preservation markers. The anchors partition the original — and the
    anchors located in the merge partition the merge — into aligned
    segments; each segment is validated independently.

    ``tokens`` optionally supplies a PRE-COMPUTED snippet classification of
    the exact shape :func:`_classify_snippet` returns. The deterministic-
    path gate (:func:`_deterministic_result_is_faithful`) passes the
    editor's OWN binding here so the invariants are evaluated against the
    partition the editor actually used; a naive re-scan binds ambiguous
    structural lines (a mid-snippet lone ``}``) at a different original
    depth than the editor's ambiguous-anchor rule, producing phantom
    anchors that reject faithful editor output. ``None`` (the model path)
    classifies with :func:`_classify_snippet` exactly as before.

    ``allow_complete_replacement`` (issue #6) scopes the sanctioned
    single-line replacement to the caller that knows the file is
    STRUCTURELESS — only the battery (:func:`_merge_rejection_reason`) can
    grant it, because only there is a whole-span restatement the snippet
    genuinely declares (the text-window flow builds it from anchors over
    the whole window). Default ``False`` keeps the code-file contract
    strict (B4): a marker-free one-line transform of a code span stays a
    preserve-by-default refusal.

    A merge is rejected on any of:

      * **marker leakage** — a snippet placeholder echoed into the merge.
      * **anchor loss / reorder** — a declared context anchor missing from
        the merge or appearing out of order.
      * **invention / omission / duplication / new-line reorder** — the
        merge's new lines in a segment must equal the snippet's declared
        new lines for that segment exactly (order and multiplicity).
      * **unjustified deletion** — an original removed without a local
        one-to-one justification: a unique shared ``_replacement_key``
        identity, or — in a marker-bearing segment only — positional
        adjacency for an identity-free line on the marker-adjacent side.
        A keyed original may only be replaced by a same-key new line,
        never by positional proximity; ambiguous keys fail closed, and
        without a marker every identity-free deletion fails closed.
      * **preserved-line reorder / loss** — surviving originals must keep
        their relative order and multiplicity.
      * **indent unfaithfulness (B14)** — consecutive surviving originals
        must keep the indent delta they have in the original, measured on
        the raw indent-bearing lines the stripped comparison discards; a
        uniform shift of a whole group is legitimate, selective
        re-indentation of one survivor is not.
      * **marker side-order violation** — a snippet-new line declared
        before the first marker must precede every surviving original in
        its segment, and one declared after the last marker must follow
        every survivor. A new line teleported to the wrong side of the
        preserved gap is rejected even when nothing is deleted. New lines
        declared between two markers are position-ambiguous and impose no
        side constraint.

    ``_replacement_key`` (imported from :mod:`text_match`) is the single
    shared identity heuristic — the validator maintains no second
    language model. Returns ``1.0`` for a clean merge, ``0.0`` otherwise.

    Issue #12: the refusal must NAME the first failing line and the rule.
    The score alone cannot; use :func:`_hallucination_reason` for the
    diagnostic (this wrapper stays the score-level API).
    """
    return 1.0 if _hallucination_reason(
        original_chunk, merged_chunk, snippet, tokens,
        allow_complete_replacement=allow_complete_replacement,
    ) is None else 0.0


def _hallucination_reason(
    original_chunk: str,
    merged_chunk: str,
    snippet: str,
    tokens: list[tuple[str, int | None, str | None]] | None = None,
    allow_complete_replacement: bool = False,
) -> str | None:
    """First content-faithfulness violation with line + rule, or ``None``.

    Issue #12: "failed the content-faithfulness check" named nothing, so a
    refused deterministic splice could not be corrected. This is the same
    validator as :func:`_check_hallucinations` returning the FIRST
    violation as a human-readable reason instead of a 0.0 score — the
    retry loop and the CLI embed it verbatim:

      * ``marker leaked into the merge: '<line>'``;
      * everything :func:`_merge_unfaithful_reason` reports (dropped /
        invented / re-indented / reordered lines, each named).

    The marker leak is checked on the RAW merged chunk first so a leaked
    marker cannot be hidden by the stripped-line comparison below.

    ``allow_complete_replacement`` scopes the issue #6 single-line
    replacement to the structureless battery (see
    :func:`_check_hallucinations`); default ``False`` keeps code spans
    under the strict B4 preserve-by-default contract.
    """
    for raw_line in merged_chunk.splitlines():
        if raw_line.strip() and _is_marker_line(raw_line):
            return f"marker leaked into the merge: {raw_line.strip()!r}"

    orig, orig_raw = _raw_content_lines(original_chunk)
    merged, merged_raw = _raw_content_lines(merged_chunk)
    if tokens is None:
        tokens = _classify_snippet(snippet, orig)

    return _merge_unfaithful_reason(
        orig, merged, tokens,
        orig_raw, merged_raw,
        wrap_insertion_only=_tokens_declare_insertion_only(tokens),
        allow_complete_replacement=allow_complete_replacement,
    )


def _complete_restatement_removals(
    orig: list[str],
    tokens: list[tuple[str, int | None, str | None]],
) -> list[int]:
    """Original content lines a COMPLETE one-for-one restatement replaces.

    Issue #6's sanctioned single-line replacement shape on structureless
    files: the snippet restates the WHOLE span with the one line changed —
    every original content line except exactly ONE binds as a context
    anchor, and the snippet declares exactly one new line. The one
    uncovered line is then not an "unmentioned original": it is the line
    the snippet replaced, and the merge (which must equal the snippet's
    declared content) could not have kept it. Both consumers of this
    predicate agree by construction — and BOTH are reached only on the
    structureless flow, whose battery grants the rule explicitly
    (``allow_complete_replacement=structureless`` in
    :func:`_merge_rejection_reason`); a code span keeps the strict B4
    contract, so a code-file restatement shape is never credited here:

      * the content validator justifies that one deletion
        (:func:`_unjustified_deletion_reason`);
      * the op derivation credits it to ``removable_traits``
        (:func:`_snippet_justifiable_removals`), making the D1 trait
        arithmetic exact replace-one-line-with-one-line arithmetic.

    Anything else returns ``[]`` — a restatement that omits a line without
    a declared replacement, a two-lines-become-one rewrite, a zero-anchor
    snippet, and every partial (anchor-bracketed) shape all stay outside
    this rule and keep failing closed.

    ``orig`` is the content-line view (blank/marker lines already skipped —
    see :func:`_raw_content_lines`) and ``tokens`` the classified snippet
    tokens whose context indices index THAT view.
    """
    anchors = {t[1] for t in tokens if t[0] == "context"}
    new_count = sum(1 for t in tokens if t[0] == "new")
    if new_count != 1:
        return []
    uncovered = [i for i in range(len(orig)) if i not in anchors]
    if len(uncovered) != 1:
        return []
    return uncovered


def _leading_restatement_gap_unjustified(
    original_func: str,
    tokens: list[tuple[str, int | None, str | None]],
) -> str | None:
    """Decline a WHOLESALE-FRAGMENT restatement rewrite (issue #16, defect 4).

    The defect-4 shape: a ``replace=`` snippet that restates the target
    span's LEADING content (its anchors form a consecutive run starting at
    the span's first content line) and, within that restated run, declares
    a NEW line between two anchors — while the original line(s) between
    those anchors are left UNMENTIONED. The editor implements such a
    snippet as a conservative insertion (the unmentioned original survives
    beside the declared new line), silently turning a replacement intent
    into a duplication — and the deterministic path ratifies a wholesale
    rewrite with 0 model tokens while the same shape through the model
    path (a merge that drops the line) is rejected. Same input, different
    verdict: the exact nondeterminism the issue pins.

    The fix reads as one declarative rule over the editor's own token
    binding: with a leading restated run of >= 2 anchors and an embedded
    declared NEW line, the gap's dropped originals must be justified by
    the battery's own rules — a unique keyed identity or a high-similarity
    rename pairing (:func:`_find_rename_pairs`), computed with the SAME
    call the battery uses, so both gates can never disagree. A wholly
    different replacement line has neither → the deterministic path
    declines and the model path's merge is judged by the battery.

    Partial snippets anchored around an insertion point (no leading
    restatement), trailing-appends (no embedded new line between anchors),
    complete symbol re-definitions (direct-swap's structural proof) and
    anything marker-bearing keep their existing semantics.
    """
    orig, _orig_raw = _raw_content_lines(original_func)
    # Leading consecutive context-anchored run: snippet content anchored
    # to orig[0..idx-1] in order.
    idx = 0
    i = 0
    while i < len(tokens) and tokens[i][0] == "context" and tokens[i][1] == idx:
        idx += 1
        i += 1
    if idx < 2 or i >= len(tokens) or tokens[i][0] != "new":
        return None  # no leading restated run, or the next token is not new
    seg_tokens: list[tuple[str, int | None, str | None]] = []
    while i < len(tokens) and tokens[i][0] != "context":
        seg_tokens.append(tokens[i])
        i += 1
    seg_new = [t[2] for t in seg_tokens if t[0] == "new" and t[2]]
    if not seg_new:
        return None
    # The embedded new line must be BRACKETED on both sides by declared
    # anchors within the restated run — an unbracketed (trailing or
    # marker-terminated) declaration is an ordinary insertion, whose
    # preserve-the-remainder semantics are the editor's contract, never a
    # restatement gap.
    if i >= len(tokens) or tokens[i][0] != "context":
        return None
    next_anchor = tokens[i][1]
    if not isinstance(next_anchor, int) or next_anchor <= idx:
        return None  # nothing dropped inside the embedded gap
    if any(t[0] == "marker" for t in seg_tokens):
        return None  # a marker gap declares keep-the-rest, not a rewrite
    dropped_local = list(range(next_anchor - idx))
    return _unjustified_deletion_reason(
        orig[idx:next_anchor],
        dropped_local,
        seg_new,
        [], [],
        allow_positional=False,
        allow_complete_replacement=False,
    )


def _deterministic_result_is_faithful(
    original_func: str,
    edited: str,
    snippet: str,
    prepend_signature_lines: int = 0,
) -> bool:
    """Gate the deterministic splice on the shared content validator (B3).

    ``deterministic_edit`` output used to be spliced with ONLY a parse
    check, and a parse cannot see content corruption: a dropped original
    line (B1/B2), a leaked keep-marker or a selectively re-indented
    survivor all parse fine. This helper runs the same
    :func:`_check_hallucinations` validator the model path is held to and
    requires a fully clean score — the deterministic path must never
    bypass the validator.

    The inputs are span-local, exactly the values available at the splice
    site: the original target span (``original_func``), the editor's
    edited span (``edited``), the snippet that declared the edit
    (including any auto-prepended signature lines, so the validator sees
    the same snippet ``deterministic_edit`` consumed) and the number of
    prepended signature lines.

    The snippet is classified with the EDITOR'S OWN classifier
    (:func:`text_match._classify_edit_lines`) rather than the validator's
    simpler forward scan, and the resulting binding is handed to the
    validator via its ``tokens`` parameter. Classification asymmetry note
    (Step 6 triage): the editor's ambiguous-anchor rule treats a lone
    structural line mid-snippet (``}``, ``)``, ``end``, ...) as new
    content, while the validator's naive re-scan binds it to the first
    unconsumed original occurrence — often a deeper closer. That phantom
    anchor re-partitioned the original and made the preserved body lines
    look like unjustified deletions, rejecting FAITHFUL editor output
    (Rust guard-additions in tests/test_chained_edits_stale_ast.py and
    tests/test_replace_without_signature.py). Content faithfulness —
    deletions, inventions, reorders, marker leakage, indent deltas — is
    still enforced in full; only the ANCHOR SELECTION is taken from the
    editor, which is the component that owns that decision.

    Returns ``True`` only when the score is clean (``1.0``); anything else
    fails closed and the caller discards the deterministic result to the
    model path.

    Issue #12: a plain ``False`` names no line and no rule, so the refusal
    the CLI prints cannot be acted on. Use
    :func:`_deterministic_result_unfaithful_reason` for the diagnostic
    (this wrapper stays the boolean gate).
    """
    reason = _deterministic_result_unfaithful_reason(
        original_func, edited, snippet, prepend_signature_lines,
    )
    return reason is None


def _deterministic_result_unfaithful_reason(
    original_func: str,
    edited: str,
    snippet: str,
    prepend_signature_lines: int = 0,
) -> str | None:
    """First faithfulness violation of a deterministic splice (issue #12).

    Same gate as :func:`_deterministic_result_is_faithful`, returning the
    FIRST violation — rule + offending line content — instead of ``False``,
    e.g. ``"unmentioned original line dropped: 'ops.push(...)'"``. The
    CLI's deterministic-refusal print embeds this verbatim so the snippet
    can be corrected. Returns ``None`` when the splice is faithful.
    """
    from .text_match import _AmbiguousAnchorBinding, _classify_edit_lines

    raw_lines = original_func.splitlines()
    try:
        classified = _classify_edit_lines(
            raw_lines, snippet.splitlines(), prepend_signature_lines,
        )
    except _AmbiguousAnchorBinding:
        # B24: the editor's classifier raises when two candidate anchor
        # bindings remain equally valid after sequence+indent
        # disambiguation. The gate fails closed for the same inputs, so a
        # deterministic result built on a guessed binding never reaches
        # the file. (In practice unreachable — the editor already declined
        # on this classification — but the gate must never guess either.)
        return "ambiguous anchor binding: multiple equally valid original regions"

    # Translate the editor's raw-line indices into the validator's
    # content-line index space (blanks and marker lines are skipped
    # there). A context anchor bound to a literal marker line in the
    # original is invisible to the validator's content view, so it is
    # skipped here too — mirroring what _classify_snippet could ever
    # bind.
    content_idx: dict[int, int] = {}
    ci = 0
    for ri, ln in enumerate(raw_lines):
        if not ln.strip() or _is_marker_line(ln):
            continue
        content_idx[ri] = ci
        ci += 1

    tokens: list[tuple[str, int | None, str | None]] = []
    for kind, _si, orig_idx, line in classified:
        if kind == "blank":
            continue
        if kind == "marker":
            tokens.append(("marker", None, None))
        elif kind == "context" and orig_idx in content_idx:
            tokens.append(("context", content_idx[orig_idx], line.strip()))
        else:
            tokens.append(("new", None, line.strip()))

    # Issue #16 (defect 4), marker-free snippets only — a marker-bearing
    # snippet declares keep-the-rest semantics where completeness arithmetic
    # does not apply. A snippet that restates the WHOLE target span (every
    # content line except exactly one is a declared context anchor) and
    # declares exactly one new line has only ONE consistent reading — that
    # line was REPLACED. The editor implements it as a conservative
    # insertion (the unmentioned original survives beside the new line),
    # silently turning a replacement intent into a duplication — ratifying
    # a wholesale rewrite with 0 model tokens. The replacement is justified
    # ONLY by the battery's own declarative rules — a unique keyed identity
    # or a high-similarity rename pairing (:func:`_find_rename_pairs`) — so
    # anything else (the "Complete the onboarding survey..." shape) fails
    # closed and falls to the model, whose merge the battery then judges.
    if not any(t[0] == "marker" for t in tokens):
        orig, _orig_raw = _raw_content_lines(original_func)
        removals = _complete_restatement_removals(orig, tokens)
        if removals:
            new_lines = [t[2] for t in tokens if t[0] == "new" and t[2]]
            justification = _unjustified_deletion_reason(
                orig, removals, new_lines, [], [],
                allow_positional=False,
                allow_complete_replacement=False,
            )
            if justification is not None:
                return (
                    "whole-span restatement: the snippet restates the "
                    "entire target span except one line, so that line can "
                    f"only have been replaced — and {justification}. A "
                    "replacement is justified only by a uniquely keyed "
                    "identity with the declared new line or by a "
                    "high-similarity rename pairing"
                )
        fragment_reason = _leading_restatement_gap_unjustified(
            original_func, tokens,
        )
        if fragment_reason is not None:
            return fragment_reason

    return _hallucination_reason(
        original_func, edited, snippet, tokens=tokens,
    )


def _snippet_splice_covers_deleted_lines(
    original_func: str,
    snippet_text: str,
) -> bool:
    """Completeness check for a WHOLESALE symbol swap (issue #1, hole 1).

    A direct swap is legitimate exactly when it is a COMPLETE re-definition
    of the target symbol. For code symbols the grammar gives that proof
    structurally: the direct-swap branch's single-definition parse plus the
    bracket-balance equality (snippet balance == span balance) certify that
    the snippet closes the symbol the span opens. For FORMAT symbols
    (markdown sections, json keys, yaml keys, ...) there is no closing
    delimiter — nothing structural distinguishes a complete re-definition
    from a snippet that restates the definition line and silently drops the
    rest of the body.

    This predicate supplies the missing proof declaratively, in the
    vocabulary the content validator already uses: classify the snippet
    against the span with the editor's own classifier
    (:func:`text_match._classify_edit_lines`), then require that every
    non-blank original content line the swap deletes is covered by a
    declared new line (``len(deleted) <= len(new)``). A snippet that
    restates the definition line and one body line while the section has
    three fails; a snippet restating heading + full body passes. Blank
    lines and marker lines carry no comparable content and are skipped on
    both sides (same convention as :func:`_raw_content_lines`), so the
    count arithmetic is content-only.

    This is a NECESSARY condition, not the preserve-by-default validator:
    a swap may legitimately rewrite every body line, so per-line deletion
    justification (keyed identity, positional adjacency) does not apply
    here. The gate catches the dropped-body corruption shape; the parse
    gate and the CLI's caller-side checks keep their existing roles.
    """
    from .text_match import _AmbiguousAnchorBinding, _classify_edit_lines

    raw_lines = original_func.splitlines()
    try:
        classified = _classify_edit_lines(
            raw_lines, snippet_text.splitlines(), 0,
        )
    except _AmbiguousAnchorBinding:
        # The gate fails closed for the same inputs the editor declines.
        return False

    # Issue #24 (defect A): FULL-SECTION SWAP, TWICE-BRACKETED. A
    # format-symbol snippet (--replace over a definition_line_required
    # span) whose head restates the span's FIRST content line (the
    # definition line) AND whose last content line restates the span's
    # LAST content line is a deliberately bracketed whole-section
    # restatement: everything the snippet omits in between is an
    # INTENTIONAL deletion (the issue's section rewrite — one paragraph
    # deleted, two bullets reworded). The contract: a --replace snippet
    # restating the whole section IS the complete new text — deletions
    # included. Checked BEFORE the #8 truncated-tail refusal below: the
    # two shapes are disjoint (a REWORDED INTERIOR line of a full swap
    # can be a strict prefix of its original — the #24 rule requires the
    # tail restated EXACTLY, while #8 fires when the TAIL ITSELF is cut
    # mid-sentence), so ordering keeps #8 intact wherever the tail is
    # not restated.
    snippet_briquetted = [
        ln.strip() for ln in snippet_text.splitlines()
        if ln.strip() and not is_marker_line(ln)
    ]
    span_content = [
        ln.strip() for ln in raw_lines
        if ln.strip() and not is_marker_line(ln)
    ]
    if (
        snippet_briquetted
        and snippet_briquetted[0] == span_content[0]
        and snippet_briquetted[-1] == span_content[-1]
    ):
        return True

    content_idx: dict[int, int] = {}
    ci = 0
    for ri, ln in enumerate(raw_lines):
        if not ln.strip() or is_marker_line(ln):
            continue
        content_idx[ri] = ci
        ci += 1

    bound = {
        c[2] for c in classified
        if c[0] == "context" and c[2] in content_idx
    }
    new_lines = [c for c in classified if c[0] == "new"]

    # Issue #8: a snippet ending MID-SENTENCE — its final content line a
    # strict prefix of an original line the swap would delete — is a
    # TRUNCATED restatement, not a complete re-declaration. The count
    # arithmetic alone ratified the swap because the partial line counted
    # as a declared new line "covering" the deletion, so the tail after
    # the cut was dropped with exit 0 (the text-match editor declines the
    # same shape as a rewrite conflict, leaving the direct swap as the
    # only gate). Refuse when the snippet's last declared new line is a
    # strict prefix of any deleted original content line. A complete
    # rewrite is unaffected: its new lines are never strict prefixes of
    # the originals they replace.
    if new_lines:
        last_new = (new_lines[-1][3] or "").strip()
        if last_new:
            bound_raw = {
                c[2] for c in classified
                if c[0] == "context" and c[2] in content_idx
            }
            for ri in content_idx:
                if ri in bound_raw:
                    continue
                orig_line = raw_lines[ri].strip()
                if orig_line.startswith(last_new) and len(last_new) < len(orig_line):
                    return False

    # Issue #16 (defect 2): a deleted original the snippet restates with a
    # token RENAMED is covered by a declared new line through the SAME
    # declarative rename pairing the content validator consumes
    # (:func:`_find_rename_pairs`) — one-to-one, similarity-thresholded,
    # capped. This is the direct-swap analogue of the battery gate: the
    # swap deletes one original and adds one renamed replacement, so the
    # count arithmetic must credit the pair.
    seg_orig = [raw_lines[ri].strip() for ri in content_idx]
    deleted_idx = [ci for ri, ci in content_idx.items() if ri not in bound]
    seg_new = [
        c[3].strip() for c in classified
        if c[0] == "new" and (c[3] or "").strip()
    ]
    rename_pairs = _find_rename_pairs(seg_orig, deleted_idx, seg_new)
    covered = len(deleted_idx) - len(rename_pairs)
    return covered <= len(new_lines)


def _strip_leading_doc_duplicate(
    snippet: str,
    original_lines: list[str],
    span_start: int,
) -> tuple[str, int]:
    """Strip a snippet-restated doc comment above the replaced span (#3).

    A symbol's replaced span starts at its DEFINITION line (``def`` /
    ``function`` / ...); a doc comment above it is OUTSIDE the span. When
    the snippet restates that comment along with the signature and body,
    splicing it verbatim emitted the comment TWICE — the original above
    the span survived and the snippet's copy landed with the function
    (issue #3). The same restatement also broke the Python single-
    definition refinement (the docstring statement counted as a second
    top-level node), refusing a valid edit outright.

    Returns ``(stripped_snippet, lines_removed)``. The snippet's leading
    non-blank lines must restate a SUFFIX of the content block directly
    above the span (the k lines closest to the span, blank-skip
    comparison, stripped, in order — the most complete restatement wins);
    blank padding between the restated comment and the definition line is
    stripped with it. At least one snippet line always survives, nothing
    is stripped when the span starts at the top of the file, and a
    snippet whose head does not match the block above is returned
    untouched — a genuinely NEW doc comment added by the snippet is
    content, never stripped.

    KNOWN LIMIT, deliberate: the match is on stripped non-blank lines, so
    a snippet whose restatement differs from the file only in blank-line
    placement inside the comment is not stripped — the existing gates
    decide, never guess.
    """
    snippet_lines = snippet.splitlines(keepends=True)
    if not snippet_lines or span_start <= 0 or span_start > len(original_lines):
        return snippet, 0

    # The content lines directly above the span, top-down (blank lines
    # skipped; bounded scan).
    _DOC_SCAN_WINDOW = 20
    above: list[str] = []
    p = span_start - 1
    scanned = 0
    while p >= 0 and scanned < _DOC_SCAN_WINDOW:
        line = original_lines[p]
        if line.strip():
            above.append(line.strip())
            scanned += 1
        p -= 1
    above.reverse()  # the scan walks upward; the block reads top-down
    if not above:
        return snippet, 0

    # The snippet's leading non-blank lines restate a SUFFIX of the block
    # above — the k lines closest to the span. Largest k wins (the most
    # complete restatement); the snippet's last line never enters the
    # head, so at least one line always survives.
    snippet_nonblank = [
        (si, ln.strip()) for si, ln in enumerate(snippet_lines[:-1])
        if ln.strip()
    ]
    best_k = 0
    for k in range(min(len(above), len(snippet_nonblank)), 0, -1):
        if [s for _si, s in snippet_nonblank[:k]] == above[-k:]:
            best_k = k
            break
    if best_k == 0:
        return snippet, 0
    consumed = snippet_nonblank[best_k - 1][0] + 1
    # Blank padding between the restated comment and the definition line
    # belongs to the duplicate prefix too (the file already carries it).
    while consumed < len(snippet_lines) - 1 and not snippet_lines[consumed].strip():
        consumed += 1
    if consumed <= 0 or consumed >= len(snippet_lines):
        return snippet, 0
    return "".join(snippet_lines[consumed:]), consumed


_JSON_SNIPPET_UNSET = object()
"""Sentinel distinguishing "json.loads failed" from a parsed ``null``."""


def _try_json_key_replace(
    original_code: str,
    snippet: str,
    replace: str,
    target_node: ASTNode | None = None,
) -> tuple[str, bool] | None:
    """JSON-aware ``replace=<key>`` splice: change ONE key's value (#4).

    The generic ``replace=`` machinery is code-shaped: a bare value
    snippet (``"new"``, ``new``, ``true``) was refused (not valid JSON /
    no definition line), and a restated object snippet (``{"name":
    "new"}``) direct-swapped over the pair's span — which on a compact
    file (all pairs on one line) is the WHOLE document, wiping every
    sibling key. This helper intercepts before those gates when the
    target is a JSON ``pair`` symbol:

      * **Bare JSON value** — the snippet is validated by ``json.loads``
        alone (a document parse is never required) and ONLY the pair's
        value node is spliced, so sibling keys, the surrounding comma
        tokens, the pair's indentation, the file's trailing-newline state
        and its compact/pretty style are preserved by construction (the
        splice replaces exactly the value node's byte span, and a JSON
        serializer never emits a comma).
      * **Restated ``"key": value``** (the unbraced form, or a one-key
        object whose key IS the target) — the value is spliced the same
        way; the key node too when the restated key differs (a rename).
      * **One-key object whose key is NOT the target** — treated as a
        bare object VALUE for the target key (the key stays, the old
        value is replaced wholesale). Changing one key of a nested
        object is done by targeting the nested key (``config.debug``).
        A snippet restating sibling keys of the TARGET keeps the
        existing path, where ``definition_line_required`` and the
        splice-completeness gate guard against wipes.

    The merged document is validated strictly with ``json.loads``; a
    file with PRE-EXISTING defects (a dangling comma) that strict
    parsing rejects is validated relatively instead — the splice cannot
    add or remove a comma token (the value span excludes them; the
    serializer never emits one), so the edit is accepted only when the
    merged diagnostics carry no NEW error trait relative to the
    original's (the repo's req. 9 relative rule).

    Returns ``(merged_code, parse_valid)`` or ``None`` to decline (the
    caller then runs the existing gates unchanged).
    """
    import json as json_mod

    from ..data_gen.ast_analyzer import parse_code, parse_diagnostics
    from .ast_utils import _json_pair_name

    text = snippet.strip()
    if not text:
        return None

    try:
        tree = parse_code(original_code, "json")
    except Exception:  # noqa: BLE001 -- no json grammar: decline, never guess
        return None

    source_bytes = original_code.encode("utf-8")

    # Dotted targets (``config.debug``) resolve by their LEAF key: the
    # target node's span (matched below) disambiguates repeats. A target
    # that names a LITERAL dotted key (``"a.b": 1`` — the dot is part of
    # the key, resolved by _resolve_symbol's B3 literal pass) must match
    # its full name too: the leaf split alone never matches it, and the
    # edit would silently decline to the generic gates.
    leaf = replace.split(".")[-1] if replace else ""

    def _pair_end_line(node) -> int:
        end_row, end_col = node.end_point
        return end_row if end_col == 0 else end_row + 1

    matches: list = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "pair":
            pair_name = _json_pair_name(node, "json", source_bytes)
            if pair_name == leaf or (replace is not None and pair_name == replace):
                matches.append(node)
        stack.extend(node.children)
    if target_node is not None:
        matches = [
            n for n in matches
            if n.start_point[0] + 1 == target_node.line_start
            and _pair_end_line(n) == target_node.line_end
        ]
    if len(matches) != 1:
        # No such key, or an ambiguous repeat (two same-named keys in one
        # span): the caller's existing resolution/refusal paths decide.
        return None
    pair = matches[0]
    key_node = pair.child_by_field_name("key")
    value_node = pair.child_by_field_name("value")
    if key_node is None or value_node is None:
        return None
    old_key = _json_pair_name(pair, "json", source_bytes)
    old_value_text = source_bytes[
        value_node.start_byte:value_node.end_byte
    ].decode("utf-8", errors="replace")

    # (a) Classify the snippet WITHOUT parsing the document.
    kind: tuple[str, str | None, object] | None = None
    doc = _JSON_SNIPPET_UNSET
    try:
        doc = json_mod.loads(text)
    except ValueError:
        doc = _JSON_SNIPPET_UNSET
    if doc is not _JSON_SNIPPET_UNSET and isinstance(doc, dict) and len(doc) == 1:
        k, v = next(iter(doc.items()))
        if k == old_key:
            kind = ("pair", k, v)  # restated pair: value only (key identical)
        else:
            kind = ("value", None, doc)  # bare object value for the key
    elif doc is not _JSON_SNIPPET_UNSET:
        kind = ("value", None, doc)
    else:
        try:
            wrapped = json_mod.loads("{" + text + "}")
        except ValueError:
            wrapped = None
        if isinstance(wrapped, dict) and len(wrapped) == 1:
            kind = ("pair", *next(iter(wrapped.items())))
        elif not any(ch in text for ch in "{}[]\":,") and "\n" not in text:
            # A bare unquoted token ("new") is the caller's spelling of a
            # JSON string; validate it as one (still no document parse).
            try:
                kind = ("value", None, json_mod.loads(json_mod.dumps(text)))
            except ValueError:
                return None
        else:
            return None
    if kind is None:
        return None

    splice_key_rename = False
    if kind[0] == "pair":
        new_key, value = kind[1], kind[2]
        splice_key_rename = new_key != old_key
    else:
        value = kind[2]

    # (b) Serialize the new value in the file's own style. Pretty iff the
    # OLD value was pretty; the indent unit comes from the old value's
    # own first nested line; the compact separator (", " vs ",") from the
    # old value text.
    pretty = "\n" in old_value_text
    line_begin = source_bytes.rfind(b"\n", 0, pair.start_byte) + 1
    prefix = source_bytes[line_begin:pair.start_byte].decode(
        "utf-8", errors="replace",
    )
    pair_indent = prefix if not prefix.strip() else ""
    unit = "  "
    if pretty:
        for ln in old_value_text.split("\n")[1:]:
            if ln.strip():
                lead = ln[: len(ln) - len(ln.lstrip())]
                if lead.startswith(pair_indent) and len(lead) > len(pair_indent):
                    unit = lead[len(pair_indent):]
                break
    comma_sep = ", " if ", " in old_value_text else ","
    colon_sep = ": " if ": " in old_value_text else ":"
    compact_separators = (comma_sep, colon_sep)

    def _render(v, indent: str) -> str:
        if isinstance(v, dict):
            if not v:
                return "{}"
            pad = indent + unit
            items = [
                json_mod.dumps(k2, ensure_ascii=False) + ": " + _render(v2, pad)
                for k2, v2 in v.items()
            ]
            if pretty:
                return (
                    "{\n"
                    + ",\n".join(pad + item for item in items)
                    + "\n" + indent + "}"
                )
            return "{" + ", ".join(items) + "}"
        if isinstance(v, list):
            if not v:
                return "[]"
            pad = indent + unit
            items = [_render(v2, pad) for v2 in v]
            if pretty:
                return (
                    "[\n"
                    + ",\n".join(pad + item for item in items)
                    + "\n" + indent + "]"
                )
            return "[" + ", ".join(items) + "]"
        return json_mod.dumps(v, ensure_ascii=False)

    if pretty:
        new_value_text = _render(value, pair_indent)
    else:
        new_value_text = json_mod.dumps(
            value, ensure_ascii=False, separators=compact_separators,
        )

    data = bytearray(source_bytes)
    data[value_node.start_byte:value_node.end_byte] = new_value_text.encode("utf-8")
    # Key rename (restated `"key": value` with a different key): the key
    # node precedes the value node, so the splice above left its offsets
    # valid.
    if splice_key_rename:
        data[key_node.start_byte:key_node.end_byte] = json_mod.dumps(
            new_key, ensure_ascii=False,
        ).encode("utf-8")
    merged = bytes(data).decode("utf-8")

    # (Dangling-)comma safety belt: the splice cannot introduce one (the
    # value span excludes the comma tokens; the serializer never emits a
    # comma), so a NEW STRUCTURAL ",," can only mean a bug — refuse. The
    # check runs on the string-masked text: a value whose STRING CONTENT
    # contains ",," ({"a": "wait,, what"}) is content, never a splice bug,
    # and must not be refused.
    if (
        ",," in mask_string_spans(merged)
        and ",," not in mask_string_spans(original_code)
    ):
        return None

    # (Validation) Strict first; relative fallback for files whose
    # PRE-EXISTING defects strict parsing rejects (see docstring).
    try:
        json_mod.loads(merged)
        return merged, True
    except ValueError:
        pass
    ok, _reason = merged_is_acceptable(
        parse_diagnostics(original_code, "json"),
        parse_diagnostics(merged, "json"),
    )
    if ok:
        return merged, True
    return None


def _classify_snippet_raw(
    snippet: str,
    orig: list[str],
) -> list[tuple[str, int | None, str | None]]:
    """The raw-line form of :func:`_classify_snippet` — one binding logic.

    Identical classification (same marker predicate, same context binding,
    same closer rule) but every token's value is the snippet's RAW line,
    indent included. The validator consumes the stripped view; the realign
    protection (C3 seams stress) needs the raw view to recognize the model's
    verbatim echo of a declared new line.
    """
    tokens: list[tuple[str, int | None, str | None]] = []
    cursor = 0
    unclosed_opens = _bracket_balance(snippet) > 0
    for raw in snippet.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _is_marker_line(raw):
            tokens.append(("marker", None, None))
            continue
        match_idx = None
        for i in range(cursor, len(orig)):
            if orig[i] == s:
                match_idx = i
                break
        if match_idx is not None and not (
            unclosed_opens and _bracket_balance(raw) < 0
        ):
            tokens.append(("context", match_idx, raw))
            cursor = match_idx + 1
        else:
            tokens.append(("new", None, raw))
    return tokens


def _declared_new_raw_lines(snippet: str, original_chunk: str) -> list[str]:
    """The snippet's declared NEW lines, raw (indent included), in order.

    The C3 realign protection: these are the bytes the aligned snippet told
    the model to insert, so a model echo of them already carries the right
    indent and the chunk-repair shift must never touch them. Classified with
    the SAME binding logic the validator uses (:func:`_classify_snippet_raw`)
    against the chunk the snippet was aligned to.
    """
    return [
        value
        for kind, _idx, value in _classify_snippet_raw(snippet, _real_lines(original_chunk))
        if kind == "new" and value is not None
    ]


def _classify_snippet(
    snippet: str,
    orig: list[str],
) -> list[tuple[str, int | None, str | None]]:
    """Forward-scan the snippet into ordered tokens.

    Each token is ``("context", orig_idx, line)``, ``("new", None, line)``
    or ``("marker", None, None)``. A non-marker snippet line is a *context*
    anchor when its normalized content matches an as-yet-unconsumed original
    line (scanning forward, so anchors are strictly increasing in
    ``orig_idx``); otherwise it is a *new* line.

    This is deliberately simpler than ``deterministic_edit``'s classifier.
    The editor has synthesis-specific rules for ambiguous structural lines
    and indentation deltas; the validator receives stripped content after
    output realignment and must not copy those rules blindly. Its job is to
    enforce content, multiplicity, and ordering invariants, not reconstruct
    the editor's single chosen output.

    C2 closer rule (the one place the scan looks beyond line content): when
    the snippet's lines open more brackets than they close (``_bracket_balance
    > 0``), the snippet declares a NEW block it never closes — so a snippet
    line that is a pure closer AND matches an unconsumed original closer is
    the NEW block's closer, not a restatement of the original's. Binding it
    as context made the validator steal the anchor for the model's newly
    emitted closer and orphan the original's own surviving closer, which then
    read as an invention: EVERY faithful model merge of a brace-language
    wrap_block (``if x { ... } else { ... }``) was rejected on every retry,
    forever (found by the C2 100MB stress tier; the wrap could never converge
    — 9/9 attempts rejected — on any ``{``-body language). With the closer
    declared as a new line, the original closer survives through the shared
    survivor LCS exactly as the preserve-by-default contract intends.
    """
    return [
        (kind, idx, value.strip() if value is not None else None)
        for kind, idx, value in _classify_snippet_raw(snippet, orig)
    ]


def _merge_is_faithful(
    orig: list[str],
    merged: list[str],
    tokens: list[tuple[str, int | None, str | None]],
    orig_raw: list[str],
    merged_raw: list[str],
    wrap_insertion_only: bool = False,
) -> bool:
    """Check the merge against the per-segment invariants (issue #12 gate).

    Boolean convenience wrapper: faithful exactly when
    :func:`_merge_unfaithful_reason` finds no violation. See the reason
    variant for the invariant list and the failing-line diagnostics.
    """
    return _merge_unfaithful_reason(
        orig, merged, tokens, orig_raw, merged_raw,
        wrap_insertion_only=wrap_insertion_only,
    ) is None


def _merge_unfaithful_reason(
    orig: list[str],
    merged: list[str],
    tokens: list[tuple[str, int | None, str | None]],
    orig_raw: list[str],
    merged_raw: list[str],
    wrap_insertion_only: bool = False,
    allow_complete_replacement: bool = False,
) -> str | None:
    """First violation of the per-segment invariants, or ``None`` (issue #12).

    The boolean gate (:func:`_merge_is_faithful`) could only say
    "unfaithful" — the refusal then named no line and no rule, and a
    deterministic splice could not be corrected. This variant returns a
    human-readable description of the FIRST violation instead, naming the
    line content and the rule it broke; the retry loop and the CLI embed
    it verbatim in the refusal message. Checking order:

      * declared context anchor missing from the merge;
      * a context anchor re-indented without a declared scope;
      * per-segment violations (:func:`_segment_unfaithful_reason`), in
        segment order.

    ``wrap_insertion_only`` (C2): a snippet whose own shape declares a
    genuine wrap_block is an insertion-only op — the positional deletion
    fallback is disabled for every segment, so a model merge that "wraps"
    while dropping an original line is rejected instead of ratified.

    COMPLETE-RESTATEMENT REPLACEMENT (issue #6): when
    ``allow_complete_replacement`` is granted — ONLY by the battery for a
    STRUCTURELESS file (:func:`_merge_rejection_reason`) — and the snippet
    binds every original content line except exactly ONE as a context
    anchor AND declares exactly one new line
    (:func:`_complete_restatement_removals`), that one deletion is the
    snippet's declared single-line replacement — the merge is then
    byte-for-byte the snippet's declared content, nothing else could have
    survived, so the identity-free deletion is justified instead of
    failing closed. The structureless one-line replace flow
    (.gitignore-style files, no grammar, no markers) depends on this; a
    code span keeps the strict B4 contract (the flag defaults False — a
    marker-free one-line transform of code stays a refusal), and the
    per-segment one-for-one shape check in :func:`_deletions_justified`
    keeps every other deletion fail-closed.

    Anchors are located in the merge through the GLOBAL survivor alignment
    (:func:`_lcs_pair_map`, the same LCS the per-segment survivor check
    uses) rather than a first-occurrence scan. When an anchor's value
    occurs more than once on either side — a lone ``}`` is both the guard
    closer a deterministic edit emitted AND a preserved closer — the
    first-occurrence scan binds the anchor to the wrong occurrence and
    mis-partitions both sides, pushing real survivors into the wrong
    segment where they look like unjustified deletions. The maximal
    alignment pairs every surviving original with its true merge partner;
    an anchor the alignment did not pair (its line did not survive) falls
    back to the scan, which rejects when the line is genuinely missing.
    """
    anchors = [t[1] for t in tokens if t[0] == "context"]

    lcs_pairs = _lcs_pair_map(orig, merged)
    merged_pos: list[int] = []
    mcursor = 0
    for a in anchors:
        # Prefer the global survivor pairing; fall back to the first
        # occurrence at or after the previous anchor for a non-survivor.
        found = lcs_pairs.get(a)
        if found is None:
            val = orig[a]
            for j in range(mcursor, len(merged)):
                if merged[j] == val:
                    found = j
                    break
        if found is None:
            return (
                "declared context anchor missing from the merge: "
                f"{orig[a]!r}"
            )
        merged_pos.append(found)
        mcursor = found + 1

    # C3: the anchors are surviving originals TOO, and the segments below
    # only cover the gaps BETWEEN them — an anchor line's own indent was
    # never checked, so a model that re-indents the context lines themselves
    # passed the battery and silently corrupted the file (found by the C3
    # 100MB seams stress: the model shifted half a block's body one tab
    # deeper into an undeclared scope and the merge was ratified). An
    # anchor's indent may deviate from the original only when the snippet
    # DECLARES a scope-introducing line before it (the wrap-completion
    # tolerance B14 already encodes for survivors); without a declared
    # scope, an anchor indent shift is corruption and fails the battery.
    scope_declared = False
    anchor_rank = 0
    for t in tokens:
        if t[0] == "new":
            if t[2] and t[2].endswith((":", "{")):
                scope_declared = True
        elif t[0] == "context":
            orig_idx = t[1]
            merged_idx = merged_pos[anchor_rank]
            if (
                not scope_declared
                and _indent_width(orig_raw[orig_idx])
                != _indent_width(merged_raw[merged_idx])
            ):
                return (
                    "context anchor re-indented without a declared scope: "
                    f"{orig[orig_idx]!r}"
                )
            scope_declared = False
            anchor_rank += 1

    # Issue #6: the sanctioned single-line replacement engages only when the
    # battery granted it for a structureless file (see the docstring) — a
    # code span never gets the allowance (B4 preserve-by-default).
    span_replacement = (
        allow_complete_replacement
        and bool(_complete_restatement_removals(orig, tokens))
    )
    for (
        orig_seg, merged_seg, seg_tokens, orig_seg_raw, merged_seg_raw,
    ) in _build_segments(orig, merged, tokens, anchors, merged_pos, orig_raw, merged_raw):
        reason = _segment_unfaithful_reason(
            orig_seg, merged_seg, seg_tokens, orig_seg_raw, merged_seg_raw,
            wrap_insertion_only=wrap_insertion_only,
            allow_complete_replacement=span_replacement,
        )
        if reason is not None:
            return reason
    return None


def _build_segments(
    orig: list[str],
    merged: list[str],
    tokens: list[tuple[str, int | None, str | None]],
    anchors: list[int],
    merged_pos: list[int],
    orig_raw: list[str],
    merged_raw: list[str],
) -> list[
    tuple[
        list[str],
        list[str],
        list[tuple[str, int | None, str | None]],
        list[str],
        list[str],
    ]
]:
    """Slice orig, merged and the snippet tokens into aligned segments.

    Returns ``(orig_seg, merged_seg, seg_tokens, orig_seg_raw,
    merged_seg_raw)`` per segment, where ``seg_tokens`` are the non-context
    tokens (new lines and markers) declared in that segment and the raw
    lists carry the indent-bearing source lines behind the segment's
    stripped content, index-aligned with it (B14). The leading segment
    precedes the first anchor, internal segments run between consecutive
    anchors and the trailing segment follows the last anchor; with no
    anchors there is one whole-span segment.
    """
    # Group non-context tokens by the anchor they follow: rank -1 means
    # "before the first anchor", rank k means "after anchor k".
    groups: dict[int, list[tuple[str, int | None, str | None]]] = {}
    anchor_rank = -1
    for t in tokens:
        if t[0] == "context":
            anchor_rank += 1
        else:
            groups.setdefault(anchor_rank, []).append(t)

    if not anchors:
        return [(orig, merged, groups.get(-1, []), orig_raw, merged_raw)]

    segments: list[
        tuple[
            list[str],
            list[str],
            list[tuple[str, int | None, str | None]],
            list[str],
            list[str],
        ]
    ] = []
    segments.append((
        orig[:anchors[0]],
        merged[:merged_pos[0]],
        groups.get(-1, []),
        orig_raw[:anchors[0]],
        merged_raw[:merged_pos[0]],
    ))
    for k in range(len(anchors) - 1):
        segments.append((
            orig[anchors[k] + 1:anchors[k + 1]],
            merged[merged_pos[k] + 1:merged_pos[k + 1]],
            groups.get(k, []),
            orig_raw[anchors[k] + 1:anchors[k + 1]],
            merged_raw[merged_pos[k] + 1:merged_pos[k + 1]],
        ))
    segments.append((
        orig[anchors[-1] + 1:],
        merged[merged_pos[-1] + 1:],
        groups.get(len(anchors) - 1, []),
        orig_raw[anchors[-1] + 1:],
        merged_raw[merged_pos[-1] + 1:],
    ))
    return segments


def _segment_is_faithful(
    orig_seg: list[str],
    merged_seg: list[str],
    seg_tokens: list[tuple[str, int | None, str | None]],
    orig_seg_raw: list[str],
    merged_seg_raw: list[str],
    wrap_insertion_only: bool = False,
    allow_complete_replacement: bool = False,
) -> bool:
    """Validate one segment against the preserve-by-default invariants.

    Boolean convenience wrapper: faithful exactly when
    :func:`_segment_unfaithful_reason` finds no violation.
    """
    return _segment_unfaithful_reason(
        orig_seg, merged_seg, seg_tokens, orig_seg_raw, merged_seg_raw,
        wrap_insertion_only=wrap_insertion_only,
        allow_complete_replacement=allow_complete_replacement,
    ) is None


def _segment_unfaithful_reason(
    orig_seg: list[str],
    merged_seg: list[str],
    seg_tokens: list[tuple[str, int | None, str | None]],
    orig_seg_raw: list[str],
    merged_seg_raw: list[str],
    wrap_insertion_only: bool = False,
    allow_complete_replacement: bool = False,
) -> str | None:
    """First violation in one segment, naming the line and rule (issue #12).

    The boolean form of this gate could only say "unfaithful"; the refusal
    then carried no line and no rule. This variant describes the FIRST
    violation instead (the caller embeds it verbatim in the refusal):

      * ``declared new line '<x>' missing from the merge`` / ``merge line
        '<y>' is not a declared snippet line`` — the merge's new lines
        must equal the snippet's declared new lines exactly;
      * ``preserved line '<x>' re-indented`` — B14's indent-delta rule;
      * ``new line '<x>' on the wrong side of the preserved gap`` — the
        marker side-order rule;
      * ``unmentioned original line dropped: '<x>'`` (identity-free) or
        ``original line '<x>' dropped without a unique replacement``
        (keyed) — the deletion-justification rules.

    EVERY segment is protected (B4): a segment is an insertion zone, never
    a replacement zone. Whether it leads the span, trails it, fills a
    marker-less mid gap or a marker-bearing one, the originals it brackets
    survive by default and the snippet's declared new lines are INSERTIONS
    into them — the merge may not overwrite the segment with its new
    lines.

    ``allow_complete_replacement`` (issue #6) is the span-level
    complete-restatement property from
    :func:`_complete_restatement_removals`: when the snippet restates the
    whole span except one line, the single one-for-one deletion in a
    segment is the declared replacement instead of an unmentioned drop.
    """
    new_all = [t[2] for t in seg_tokens if t[0] == "new"]
    has_marker = any(t[0] == "marker" for t in seg_tokens)

    # Survivors are the originals kept (longest common subsequence on
    # stripped content); the remaining merged lines are new and must equal
    # the declared new lines exactly.
    kept_orig, kept_merged = _lcs_matched(orig_seg, merged_seg)
    new_in_merged = [
        merged_seg[j] for j in range(len(merged_seg)) if j not in kept_merged
    ]
    if new_in_merged != new_all:
        return _new_lines_mismatch_reason(new_all, new_in_merged)

    # B14: surviving originals keep their relative indentation. Checked on
    # the raw lines — the stripped comparison above cannot see indent
    # corruption.
    reason = _indent_delta_violation(
        orig_seg_raw, kept_orig, merged_seg_raw, kept_merged,
    )
    if reason is not None:
        return reason

    # Marker side-order invariant: a snippet-new line declared before the
    # first marker must precede every surviving original in the merge, and a
    # new line declared after the last marker must follow every survivor.
    # Checked before the deletion justification below because a wrong-side
    # new line can violate the invariant with no deletion at all (the
    # survivor multiset is intact — only the relative placement is corrupt).
    if has_marker:
        reason = _side_order_violation(seg_tokens, merged_seg, kept_merged)
        if reason is not None:
            return reason

    deleted = [i for i in range(len(orig_seg)) if i not in kept_orig]
    if not deleted:
        return None

    new_before: list[str] = []
    new_after: list[str] = []
    if has_marker:
        marker_positions = [i for i, t in enumerate(seg_tokens) if t[0] == "marker"]
        first_marker, last_marker = marker_positions[0], marker_positions[-1]
        new_before = [
            t[2] for i, t in enumerate(seg_tokens)
            if t[0] == "new" and i < first_marker
        ]
        new_after = [
            t[2] for i, t in enumerate(seg_tokens)
            if t[0] == "new" and i > last_marker
        ]
    return _unjustified_deletion_reason(
        orig_seg, deleted, new_all, new_before, new_after,
        allow_positional=has_marker and not wrap_insertion_only,
        allow_complete_replacement=allow_complete_replacement,
    )


def _new_lines_mismatch_reason(
    new_all: list[str | None],
    new_in_merged: list[str],
) -> str:
    """Name the first declared-new-line mismatch (issue #12 diagnostics)."""
    for declared, actual in zip(new_all, new_in_merged, strict=False):
        if declared != actual:
            return (
                f"declared new line {declared!r} is missing from the merge "
                f"(merge has {actual!r} there)"
            )
    if len(new_in_merged) > len(new_all):
        return (
            f"merge line {new_in_merged[len(new_all)]!r} is not a declared "
            f"snippet line"
        )
    if len(new_all) > len(new_in_merged):
        return (
            f"declared new line {new_all[len(new_in_merged)]!r} is missing "
            f"from the merge"
        )
    return "the merge's new lines do not match the snippet's declared new lines"





def _indent_deltas_preserved(
    orig_seg_raw: list[str],
    kept_orig: set[int],
    merged_seg_raw: list[str],
    kept_merged: set[int],
) -> bool:
    """B14 boolean wrapper: faithful exactly when no indent delta breaks.

    See :func:`_indent_delta_violation` for the rule and the diagnostic.
    """
    return _indent_delta_violation(
        orig_seg_raw, kept_orig, merged_seg_raw, kept_merged,
    ) is None


def _indent_delta_violation(
    orig_seg_raw: list[str],
    kept_orig: set[int],
    merged_seg_raw: list[str],
    kept_merged: set[int],
) -> str | None:
    """B14: name the first surviving original whose indent delta broke.

    The LCS pairs survivors on STRIPPED content, so a merge can keep every
    line's content while selectively re-indenting it — flattening one
    line's nesting under an ``if`` it no longer sits inside, say. For each
    pair of CONSECUTIVE surviving originals the indent delta between their
    raw lines must equal the delta between their merge counterparts.
    Widths come from ``_indent_width`` (tab-aware, ``expandtabs(4)``),
    reused from :mod:`text_match` — no second indent model.

    Blank/whitespace-only lines never enter the content view, so a
    survivor's neighbour here is its nearest non-blank line. A UNIFORM
    shift of a whole group changes every pairwise delta by the same
    amount and is accepted; moving a single line is not.

    The k-th original survivor pairs with the k-th merge survivor: the LCS
    traceback yields strictly increasing index pairs on both sides, so
    sorting each set and zipping reproduces the alignment.

    Returns ``None`` when every consecutive survivor pair keeps its delta,
    else a reason naming the second line of the failing pair — the line
    whose indent moved relative to its surviving neighbour.
    """
    pairs = zip(sorted(kept_orig), sorted(kept_merged))
    for (i1, j1), (i2, j2) in itertools.pairwise(pairs):
        orig_delta = (
            _indent_width(orig_seg_raw[i2]) - _indent_width(orig_seg_raw[i1])
        )
        merged_delta = (
            _indent_width(merged_seg_raw[j2]) - _indent_width(merged_seg_raw[j1])
        )
        if orig_delta != merged_delta:
            return (
                f"preserved line {orig_seg_raw[i2].strip()!r} re-indented "
                f"relative to its surviving neighbour"
            )
    return None


def _deletions_justified(
    orig_seg: list[str],
    deleted: list[int],
    new_all: list[str],
    new_before: list[str],
    new_after: list[str],
    allow_positional: bool = False,
    allow_complete_replacement: bool = False,
) -> bool:
    """Decide whether every deleted protected original is justified.

    Boolean convenience wrapper: justified exactly when
    :func:`_unjustified_deletion_reason` finds no violation.
    """
    return _unjustified_deletion_reason(
        orig_seg, deleted, new_all, new_before, new_after,
        allow_positional=allow_positional,
        allow_complete_replacement=allow_complete_replacement,
    ) is None


_RENAME_MIN_LINE_CHARS = 12
"""Minimum stripped-line length for a rename pairing (issue #16, defect 2).

Similarity on very short lines is unstable noise: ``return x`` vs
``return x + 1`` scores 0.80 and ``validate(x)`` vs ``audit(x)`` scores
0.79 — both below this line's 12-character floor together with the ratio
thresholds (which exist for the same reason). A line short enough to be
ambiguously similar is content the battery must not re-classify as a
rename; such edits keep their pre-#16 refusal semantics (a full
replacement body or a marker is the sanctioned way to express them).
"""

_RENAME_MAX_SEGMENT_FRACTION = 0.25
"""Cap on renamed lines per segment, as a fraction of its content lines
(issue #16, defect 2 constraint iii). Without the cap a snippet that
"renames" most of a segment is a rewrite wearing a rename's clothes — the
pairing must never become a fuzz pass that accepts rewrites. 25% keeps
the canonical rename shapes (one token in a prose section, a heading
token, a couple of URLs) far under the ceiling while a wholesale
restatement exceeds it and fails closed."""

_RENAME_QUICK_RATIO_MIN = 0.80
"""The quick-similarity pre-filter threshold (issue #16, defect 2). A
candidate new line whose ``quick_ratio`` against the dropped original is
below this can never reach the full-ratio threshold and is discarded
without the full match computation."""

_RENAME_RATIO_MIN = 0.92
"""The full-similarity threshold for a justified rename (issue #16,
defect 2): the snippet's new line must be a NEAR restatement of the
dropped original — the same line with a token renamed — not merely
similar. Calibrated against the pinned refusals AND acceptances
(monitor both sides): ``return x`` vs ``return x + 1`` scores 0.80 and
``validate(x)`` vs ``audit(x)`` 0.79 — far below, keep failing closed;
``step-1500`` vs ``step-1500-rewritten`` scores 0.8913 and the pinned
``midnight``/``dawn`` prose 0.9130 — below, so an undeclared mutation and
a prose-value swap keep failing closed (issue #12); while the issue's own
``appUrl`` → ``dashboardUrl`` shape scores 0.9242 — just above, so the
canonical rename lands."""


def _rename_similarity(original_line: str, new_line: str) -> float:
    """Character similarity of two stripped lines via difflib (issue #16).

    :class:`difflib.SequenceMatcher` is the stdlib's own restatement
    measure — no second similarity model, matching the repo doctrine that
    the validator maintains no language model. ``autojunk=False`` keeps
    the popular-element heuristic off: snippet lines are short enough that
    autojunk's frequency table can distort the ratio on prose.
    """
    return difflib.SequenceMatcher(
        None, original_line.strip(), new_line.strip(), autojunk=False,
    ).ratio()


def _rename_quick_similarity(original_line: str, new_line: str) -> float:
    """The cheap upper-bound similarity pre-filter (issue #16).

    ``SequenceMatcher.quick_ratio()`` is an upper bound on ``ratio()``; the
    pairing pass runs it FIRST so a candidate that cannot possibly reach
    the ratio threshold never pays the full match computation.
    """
    return difflib.SequenceMatcher(
        None, original_line.strip(), new_line.strip(), autojunk=False,
    ).quick_ratio()


def _find_rename_pairs(
    orig_seg: list[str],
    deleted: list[int],
    new_all: list[str],
) -> dict[int, int]:
    """The RENAME-AWARE pairing pass for one protected segment (issue #16).

    For each deleted identity-free original the exact-line battery treats
    as "unmentioned", look for a declared new line that is a near
    restatement of it — the snippet restate the line WITH a token renamed
    (the issue's ``${appUrl}/contact`` → ``${dashboardUrl}/contact``
    shape). A deleted original O and a declared new line N form a
    JUSTIFIED RENAME pair iff all of:

      * ``len(O.strip()) >= _RENAME_MIN_LINE_CHARS`` — short lines have
        unstable similarity (see the constant's rationale);
      * O has no exact survivor in the segment (the caller only routes
        identity-free drops here — an exact restatement is in-place
        replacement, never a rename);
      * ``quick_ratio(O, N) >= 0.80`` AND ``ratio(O, N) >= 0.85`` on the
        stripped lines (issue #16 thresholds; ``quick_ratio`` runs first
        as the cheap pre-filter);
      * the pairing is ONE-TO-ONE and UNAMBIGUOUS — each original pairs at
        most one new line and each new line at most one original; when two
        or more qualifying new lines compete for the same original the
        pairing fails closed instead of guessing which one replaced it;
      * the segment-wide rename cap holds: at most
        ``ceil(_RENAME_MAX_SEGMENT_FRACTION * len(orig_seg))`` pairs.

    Returns ``{deleted_original_index: new_line_index}`` for the accepted
    pairs only; candidates that fail any condition are simply not paired,
    and their deletions keep failing closed exactly as before. The pairing
    replaces the per-line justification ONLY — every other battery gate
    (relative parse, survivor order, indent deltas, marker side-order,
    declared-new equality) applies unchanged, and the merged output must
    still satisfy the relative parse gate + trait battery.
    """
    if not deleted or not new_all:
        return {}
    cap = max(1, math.ceil(_RENAME_MAX_SEGMENT_FRACTION * len(orig_seg)))
    pairs: dict[int, int] = {}
    used_new: set[int] = set()
    for i in deleted:  # in segment order — the pairing is deterministic
        original_line = orig_seg[i].strip()
        if len(original_line) < _RENAME_MIN_LINE_CHARS:
            continue
        candidates: list[int] = []
        for j, new_line in enumerate(new_all):
            if j in used_new or not new_line:
                continue
            if _rename_quick_similarity(original_line, new_line) < _RENAME_QUICK_RATIO_MIN:
                continue
            if _rename_similarity(original_line, new_line) < _RENAME_RATIO_MIN:
                continue
            candidates.append(j)
        # Ambiguity fails closed (issue #16 constraint i): two or more
        # near-restatements competing for one original cannot be told
        # apart — pairing either would be a guess about which one replaced
        # the line. Likewise a new line already claimed by an earlier
        # original is out of the pool, so many-to-one is impossible.
        if len(candidates) != 1:
            continue
        pairs[i] = candidates[0]
        used_new.add(candidates[0])
        if len(pairs) >= cap:
            break
    return pairs


def _unjustified_deletion_reason(
    orig_seg: list[str],
    deleted: list[int],
    new_all: list[str],
    new_before: list[str],
    new_after: list[str],
    allow_positional: bool = False,
    allow_complete_replacement: bool = False,
) -> str | None:
    """Name the FIRST unjustified deleted protected original (issue #12).

    The boolean form of this gate could only fail the merge; the refusal
    then read "preserve-by-default violation" with no line and no rule.
    This variant returns a reason naming the rule and the line content:

      * ``original line '<x>' dropped without a unique replacement`` — a
        keyed original whose ``_replacement_key`` is not shared by exactly
        one original and exactly one declared new line;
      * ``unmentioned original line dropped: '<x>'`` — an identity-free
        original with no local justification.

    A *keyed* original may be deleted only when exactly one original and
    exactly one declared new line share its key — a unique shared
    identity. Any other keyed deletion fails closed. An *identity-free*
    original may be deleted by positional fallback only, when it sits
    contiguously against a marker boundary that carries an identity-free
    new line (front for lines declared before the marker, back for lines
    declared after) — never an arbitrary bystander.

    The positional fallback requires ``allow_positional``: it exists only
    where a marker defines the boundary it leans on. A segment without a
    marker has no such boundary, so EVERY identity-free deletion in it
    fails closed regardless of the declared new lines (defaults to False
    — fail closed).

    ``allow_complete_replacement`` (issue #6) is the one sanctioned
    marker-free exception: when the snippet is a COMPLETE RESTATEMENT of
    the span — every original content line except exactly one is bound as
    a context anchor (:func:`_complete_restatement_removals`) — and this
    segment deletes exactly one identity-free original and declares
    exactly one new line, that deletion IS the declared single-line
    replacement (the merge is byte-for-byte the snippet's declared
    content; nothing else could have survived). Everything else keeps
    failing closed: a restatement that omits a line without a declared
    replacement, a two-lines-become-one rewrite, and the ambiguous
    anchor-bracketed shape where the old line simply never appears in the
    snippet.
    """
    n = len(orig_seg)
    deleted_set = set(deleted)

    orig_keys = [_replacement_key(line) for line in orig_seg]
    orig_key_count: dict[str, int] = {}
    for k in orig_keys:
        if k is not None:
            orig_key_count[k] = orig_key_count.get(k, 0) + 1
    new_key_count: dict[str, int] = {}
    for line in new_all:
        k = _replacement_key(line)
        if k is not None:
            new_key_count[k] = new_key_count.get(k, 0) + 1

    positional: list[int] = []
    for i in deleted:
        k = orig_keys[i]
        if k is not None:
            # Keyed deletion: require a unique shared identity on both sides.
            if orig_key_count.get(k, 0) == 1 and new_key_count.get(k, 0) == 1:
                continue
            return (
                f"original line {orig_seg[i]!r} dropped without a unique "
                f"replacement"
            )
        positional.append(i)

    if not positional:
        return None

    # Issue #16 (defect 2): a dropped identity-free original whose snippet
    # declares a NEAR-RESTATEMENT of it is a justified RENAME — the issue's
    # '${appUrl}/contact' → '${dashboardUrl}/contact' prose shape, which the
    # exact-line battery read as an unmentioned drop on every retry. The
    # pairing is declarative and bounded (see _find_rename_pairs): one-to-one
    # only, minimum line length, similarity thresholds, and a per-segment cap
    # so it can never become a rewrite-accepting fuzz pass. Unpaired drops
    # keep failing closed below, and every other gate still applies.
    rename_pairs = _find_rename_pairs(orig_seg, positional, new_all)
    if rename_pairs and len(rename_pairs) == len(positional):
        return None

    if not allow_positional:
        # No marker in this segment — no marker-adjacent boundary exists,
        # so an identity-free deletion has no local justification — unless
        # the span-level complete-restatement property holds and this
        # segment's shape is an exact one-for-one replacement (issue #6).
        if (
            allow_complete_replacement
            and len(deleted) == 1
            and len(new_all) == 1
        ):
            return None
        return (
            f"unmentioned original line dropped: {orig_seg[positional[0]]!r}"
        )

    # Positional fallback capacity comes from identity-free new lines on
    # each side; deletions must form a contiguous run against that boundary.
    front_cap = sum(1 for line in new_before if _replacement_key(line) is None)
    back_cap = sum(1 for line in new_after if _replacement_key(line) is None)

    front_run = 0
    while front_run < n and front_run in deleted_set:
        front_run += 1
    back_run = 0
    while back_run < n and (n - 1 - back_run) in deleted_set:
        back_run += 1
    front_cover = min(front_run, front_cap)
    back_cover = min(back_run, back_cap)

    for i in positional:
        if i < front_cover or i >= n - back_cover:
            continue
        return (
            f"unmentioned original line dropped: {orig_seg[i]!r}"
        )
    return None


def _new_side_order_ok(
    seg_tokens: list[tuple[str, int | None, str | None]],
    merged_seg: list[str],
    kept_merged: set[int],
) -> bool:
    """Check the marker side-order invariant for one protected segment.

    A snippet-new line declared *before the first marker* is a top-of-gap
    insertion — it must occur before every surviving original in the merge.
    A new line declared *after the last marker* is a bottom-of-gap
    insertion — it must occur after every survivor. A new line declared
    *between* two markers is position-ambiguous and imposes no constraint;
    the caller's invention/omission/deletion checks still apply to it.

    The k-th unmatched merged occurrence (in merge order) corresponds to the
    k-th declared new line, because the caller has already verified the
    merged new lines equal the declared new lines in order and multiplicity.
    Enforcing the constraint on the concrete merged indices from the LCS
    alignment — not on line values or global counts — keeps repeated
    survivors and repeated new lines correctly positioned, so an LCS tie
    cannot hide a wrong-side new occurrence.

    Returns ``True`` when the segment carries no marker or no surviving
    original (the constraint is then vacuous — deletion justification still
    governs acceptance upstream).
    """
    return _side_order_violation(seg_tokens, merged_seg, kept_merged) is None


def _side_order_violation(
    seg_tokens: list[tuple[str, int | None, str | None]],
    merged_seg: list[str],
    kept_merged: set[int],
) -> str | None:
    """Name the first marker side-order violation (issue #12 diagnostics).

    Boolean wrapper: :func:`_new_side_order_ok`. The rule is unchanged;
    the reason names the declared new line that landed on the wrong side
    of the preserved gap, so a rejected merge can be corrected.

    Returns ``None`` when the segment carries no marker or no surviving
    original (the constraint is then vacuous — deletion justification
    still governs acceptance upstream).
    """
    marker_positions = [i for i, t in enumerate(seg_tokens) if t[0] == "marker"]
    if not marker_positions:
        return None
    first_marker, last_marker = marker_positions[0], marker_positions[-1]

    # Side of each declared new line, in declared order. This list is
    # parallel to the unmatched merged occurrences (same length, same order).
    new_sides: list[str] = []
    for i, t in enumerate(seg_tokens):
        if t[0] != "new":
            continue
        if i < first_marker:
            new_sides.append("before")
        elif i > last_marker:
            new_sides.append("after")
        else:
            new_sides.append("between")

    survivors = sorted(kept_merged)
    if not survivors:
        return None
    first_survivor, last_survivor = survivors[0], survivors[-1]

    unmatched = [j for j in range(len(merged_seg)) if j not in kept_merged]
    for side, j in zip(new_sides, unmatched, strict=True):
        if side == "before" and j > first_survivor:
            return (
                f"new line {merged_seg[j]!r} declared before the marker "
                f"landed after a surviving original"
            )
        if side == "after" and j < last_survivor:
            return (
                f"new line {merged_seg[j]!r} declared after the marker "
                f"landed before a surviving original"
            )
    return None


_LCS_MAX_CELLS = 4_000_000
"""Cell budget for the survivor-alignment DP (issue #12).

The O(n·m) DP table holds n·m Python ints; on a very large symbol span
(2,000+ lines) with the model engine resident that table alone grew to
multi-GB and the process was OOM-killed (exit 137, no message) on EVERY
validation attempt. The cap bounds any single alignment to ~4M cells
(~10^2 MB): under the cap the exact DP runs, over it the banded fallback
below runs, and when even the band cannot fit the battery raises loudly.
"""

_LCS_BAND = 500
"""Off-diagonal band for the bounded fallback LCS (issue #12).

The survivor alignment of a faithful merge is near-diagonal: the merge
preserves the original's order, so a surviving original line lands within
a bounded displacement of its source. |i - j| <= :data:`_LCS_BAND`
restricts the DP to a band of ~2·500+1 cells per row (~10^2 MB total at
the admission cap) instead of the full n·m table.

SOUNDNESS of the approximation for the battery: the banded pairing is a
common subsequence (monotone, content-equal) and can only MISS pairs the
unbounded DP would find — a missed survivor pair makes the line read as
DELETED (needing justification) or as an UNDECLARED new line, both of
which the battery then rejects. Banded alignment is therefore strictly
STRONGER for validation (it may reject a wildly-reordered merge the exact
DP would pair up), never more permissive, and it cannot silently accept a
merge the exact alignment would have refused.
"""


def _lcs_pair_map_banded(
    a: list[str],
    b: list[str],
    band: int,
) -> dict[int, int]:
    """LCS pairing restricted to ``|i - j| <= band`` (issue #12).

    Same suffix-DP + greedy traceback as :func:`_lcs_pair_map`, computed
    only over cells within the band: row ``i`` covers ``j`` in
    ``[max(0, i - band), min(lb, i + band)]``. Cells outside the band are
    forbidden states of the constrained alignment problem (no valid
    pairing passes through them); they read as ``-1`` so the ``max``/
    traceback steps always prefer the in-band successor, and one is always
    available from any in-band cell. Forced matching of equal heads stays
    optimal under the band constraint: pairing ``(i, j)`` moves diagonally
    (never toward the band edge) and the exchange argument for "match
    equal heads first" is band-independent.

    Returns ``{a_index: b_index}`` — a maximal alignment WITHIN the band.
    """
    la, lb = len(a), len(b)
    los = [max(0, i - band) for i in range(la + 1)]
    his = [min(lb, i + band) for i in range(la + 1)]
    dp = [[0] * (his[i] - los[i] + 1) for i in range(la + 1)]

    def _val(i: int, j: int) -> int:
        if i > la or j > lb:
            return 0  # past the end: no further matches possible
        if j < los[i] or j > his[i]:
            return -1  # forbidden state: outside the declared band
        return dp[i][j - los[i]]

    for i in range(la - 1, -1, -1):
        row = dp[i]
        lo = los[i]
        ai = a[i]
        for k in range(len(row) - 1, -1, -1):
            j = lo + k
            if j == lb:
                # The row's boundary column is the empty-b suffix: its LCS
                # value is 0 and b[j] must not be read (out of range).
                row[k] = 0
            elif ai == b[j]:
                row[k] = _val(i + 1, j + 1) + 1
            else:
                down = _val(i + 1, j)
                right = _val(i, j + 1)
                row[k] = max(down, right)
    pairs: dict[int, int] = {}
    i = j = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            pairs[i] = j
            i += 1
            j += 1
        elif _val(i + 1, j) >= _val(i, j + 1):
            i += 1
        else:
            j += 1
    return pairs


def _lcs_pair_map(
    a: list[str],
    b: list[str],
) -> dict[int, int]:
    """Longest-common-subsequence pairing of two line lists (issue #12).

    Returns ``{a_index: b_index}`` for one maximal alignment — strictly
    increasing on both sides, so the k-th surviving original maps to the
    k-th surviving merge line. This is the canonical survivor alignment:
    :func:`_merge_is_faithful` uses it to locate context anchors in the
    merge, :func:`_lcs_matched` derives its survivor sets from it.

    BOUNDED MEMORY (issue #12): the DP is capped at
    :data:`_LCS_MAX_CELLS` cells. Inputs whose full table fits run the
    exact DP; larger inputs run :func:`_lcs_pair_map_banded` with the
    declared band — a documented approximation that can only be stricter
    for the battery (see its docstring); and when even the banded table
    cannot fit the budget, this raises ``ValueError`` naming the size and
    the remedy ("edit a sub-symbol or narrow the span") instead of
    silently building the multi-GB table that OOM-killed the process.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return {}
    if la * lb <= _LCS_MAX_CELLS:
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la - 1, -1, -1):
            for j in range(lb - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        pairs: dict[int, int] = {}
        i = j = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                pairs[i] = j
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1
        return pairs
    band_cells = (la + 1) * min(lb + 1, 2 * _LCS_BAND + 1)
    if band_cells > _LCS_MAX_CELLS:
        raise ValueError(
            f"symbol too large for exact line matching ({max(la, lb)} lines); "
            f"edit a sub-symbol or narrow the span"
        )
    return _lcs_pair_map_banded(a, b, _LCS_BAND)


def _lcs_matched(
    a: list[str],
    b: list[str],
) -> tuple[set[int], set[int]]:
    """Longest-common-subsequence alignment of two line lists.

    Returns ``(matched_a, matched_b)`` — the index sets paired by an LCS.
    ``matched_a`` identifies the originals that survived; the unmatched
    ``b`` indices are the merge's new lines. Order and multiplicity of
    repeated lines are respected (this is a subsequence match, not a set).
    """
    pairs = _lcs_pair_map(a, b)
    return set(pairs), set(pairs.values())


def _raw_content_lines(s: str) -> tuple[list[str], list[str]]:
    """Content lines paired with the raw, indent-bearing lines behind them.

    Returns ``(content, raw)`` with ``content[i]`` the stripped form of
    ``raw[i]``. Blank/whitespace-only lines and keep-marker lines carry no
    comparable content and are skipped from BOTH views, keeping the lists
    index-aligned. The raw view exists for the B14 indentation check,
    which must see the leading whitespace the stripped content view
    deliberately discards.

    PERFORMANCE (governor mission, PART 1): the classification runs per
    validation attempt over the whole chunk/file; this used to pay one
    ``strip`` per line plus up to five regex ``match`` calls per line for
    the marker predicate. The rewrite fast-paths the common content line:
    a line whose FIRST character is neither whitespace nor a marker head
    (``#/.<…`` — the exact first-character condition every marker phrase
    and marker regex shares) is provably content, decided without the
    exact predicate. Everything else — blank lines, whitespace-led and
    marker-head-led lines — takes the original exact path with the strip
    computed once and reused.
    """
    content: list[str] = []
    raw: list[str] = []
    for line in s.splitlines():
        c0 = line[:1]
        if c0 and not c0.isspace() and c0 not in "#/.<…":
            # Fast path: stripped[0] == line[0] (nothing to strip), the line
            # is non-blank, and no marker form can begin with this
            # character — exactly what the old body concluded, cheaper.
            content.append(line.strip())
            raw.append(line)
            continue
        stripped = line.strip()
        if not stripped or is_marker_text(stripped):
            continue
        content.append(stripped)
        raw.append(line)
    return content, raw


def _real_lines(s: str) -> list[str]:
    """Stripped, non-blank, non-marker lines — the lines that carry
    real content for the diff."""
    return _raw_content_lines(s)[0]


def _error_line_key(
    span: tuple[int, int, str],
    text: str,
) -> tuple[str, str]:
    """Defect identity key for one parse-error trait: (kind, its line).

    Tree-sitter error recovery can shift an error span's EXTENT when
    surrounding context changes, but an INHERITED defect sits on a
    byte-identical line (the preserve-by-default contract keeps untouched
    regions byte-exact), so "kind + the line the error starts on" is a
    stable trait identity between the original and the merged texts even
    when byte offsets shift with the edit's length delta.

    ``text`` MUST be the ``.source`` of the diagnostics the span came from
    (see :func:`parse_diagnostics`' normalization contract — spans index
    that copy, and a bare CR would change where lines break).
    """
    start = span[0]
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    return (span[2], text[line_start:line_end])


def _format_error_spans(
    spans: list[tuple[int, int, str]],
    text: str,
) -> str:
    """Human-readable one-line rendering of error traits (retry notes)."""
    parts: list[str] = []
    for span in spans[:4]:
        shown = _error_line_key(span, text)[1]
        if len(shown) > 60:
            shown = shown[:57] + "..."
        parts.append(
            f"{span[2]} at bytes {span[0]}-{span[1]} on line {shown!r}",
        )
    if len(spans) > 4:
        parts.append(f"... and {len(spans) - 4} more")
    return "; ".join(parts)


def merged_is_acceptable(
    original_diags,
    merged_diags,
    edited_spans: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> tuple[bool, str]:
    """The RELATIVE parse rule (Step A2, req. 9 — EDIT-NOT-CORRECT).

    Grammar/structure detection exists to IDENTIFY traits of the input and
    compare them to the output — never to correct the source. Acceptance
    is therefore trait-based, never an absolute well-formedness ideal:

      * **original parses cleanly** → the merged output must parse cleanly
        too (unchanged pre-A2 behavior for valid inputs). ``edited_spans``
        does NOT excuse breakage here: an edit that breaks a previously
        valid file is a regression wherever it lands (req. 9: when the
        command explicitly replaces malformed text the output must be
        well-formed — the mirror case below covers broken originals).

      * **original is broken** → the merged output is acceptable iff every
        error trait it contains is either (a) an INHERITED trait — matched,
        as a multiset, against the original's own error traits by
        (kind, containing line) — or (b) inside one of ``edited_spans``,
        where the op spec governs. Defect REMOVAL is always acceptable
        (the edit may target the broken text — req. 9's "replace malformed
        with well-formed" case); defect ADDITION outside the edited spans
        is not. The battery (:func:`_merge_rejection_reason`) passes NO
        edited spans: a replacement region is expected to be well-formed,
        so a new error inside it is retried like any other.

    Defect-identity caveat, by design: the multiset matching cannot tell
    "the op fixed the defect" from "the model helpfully fixed it" — both
    REMOVE a trait. Removing an error trait is therefore never a parse-
    rule violation; a model that mutates untouched content is caught by
    the CONTENT faithfulness validator (:func:`_check_hallucinations`),
    which is the mechanism that enforces EDIT-NOT-CORRECT for untouched
    lines. Conversely the parse rule uniquely catches CONTENT-CLEAN
    breakage (declared lines interacting with kept lines into invalid
    syntax), which the line-level validator cannot see.

    Args:
        original_diags: :class:`ParseDiagnostics` of the original text.
        merged_diags: :class:`ParseDiagnostics` of the merged text.
        edited_spans: ``(start, end)`` byte spans (in the MERGED text's
            coordinates) whose errors the op spec governs. Empty by
            default.

    Returns:
        ``(True, "")`` when acceptable, else ``(False, reason)`` with a
        human-readable reason for the retry note.
    """
    if original_diags.is_valid:
        if merged_diags.is_valid:
            return True, ""
        return False, (
            f"merged output has {len(merged_diags.errors)} parse error(s) "
            f"the original does not have: "
            f"{_format_error_spans(merged_diags.errors, merged_diags.source)}"
        )

    # Original broken: account for every merged error trait.
    remaining: Counter[tuple[str, str]] = Counter(
        _error_line_key(span, original_diags.source)
        for span in original_diags.errors
    )
    unexplained: list[tuple[int, int, str]] = []
    for span in merged_diags.errors:
        if any(span[0] < end and start < span[1] for start, end in edited_spans):
            continue  # inside an edited span — the op spec governs there
        key = _error_line_key(span, merged_diags.source)
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
        else:
            unexplained.append(span)
    if not unexplained:
        return True, ""
    return False, (
        f"merged output introduces {len(unexplained)} new parse error(s) "
        f"absent from the original: "
        f"{_format_error_spans(unexplained, merged_diags.source)}"
    )


_STRUCTURELESS_MISS_CACHE: set[str] = set()
"""Languages the resolver already answered "no grammar" for (memo).

The resolver caches successful resolutions internally but re-raises on
every failed probe; the battery runs per attempt, so the negative answer
is memoized here. Purely a cost memo — the honest answer is unchanged.
"""


def _is_structureless_language(language: str | None) -> bool:
    """True when the file has NO grammar to parse — the D1 trait gate's scope.

    The resolver's honest answer, never a hardcoded language list
    (CLAUDE.md): ``language=None`` is structureless by definition (the
    CLI/MCP detect no language for ``.txt``/``.log``/...), and a named
    language is structureless exactly when :func:`get_language` cannot
    resolve it (the all-grammars extra absent, an unknown name). For such
    files the relative parse gate cannot run — the text-trait branch
    validates instead of the battery crashing on the unresolved grammar.
    """
    if not language:
        return True
    if language in _STRUCTURELESS_MISS_CACHE:
        return True
    try:
        from ..data_gen.ast_analyzer import get_language

        get_language(language)
    except Exception:  # noqa: BLE001 -- ANY resolution failure IS the resolver's honest "no grammar" answer; the battery must degrade to the trait gate, never propagate
        _STRUCTURELESS_MISS_CACHE.add(language)
        return True
    return False


def _snippet_justifiable_removals(
    orig: list[str],
    orig_raw: list[str],
    raw_tokens: list[tuple[str, int | None, str | None]],
) -> list[str]:
    """The original lines this snippet's shape could justify removing.

    The OP-SIDE mirror of :func:`_deletions_justified` — computed from the
    snippet alone (never from the merge), so the trait oracle can widen its
    expected floor by exactly the removal capacity the content validator
    would grant, and by nothing more:

      * **keyed capacity** — an original whose ``_replacement_key`` is
        shared by exactly one original and exactly one declared new line
        of the same segment (the unique-identity replacement rule);
      * **complete-restatement replacement** (issue #6) — the one
        uncovered original of a snippet that restates the whole span and
        declares exactly one new line
        (:func:`_complete_restatement_removals`), the same predicate the
        content validator uses;
      * **positional capacity** — in a marker-bearing segment only, the
        marker-adjacent identity-free originals covered by the declared
        new lines: the first ``front_cap`` / last ``back_cap`` segment
        lines, where the caps count the identity-free new lines declared
        before the first / after the last marker. Disabled for a snippet
        whose own shape declares an insertion-only wrap
        (:func:`_tokens_declare_insertion_only`) — the same switch the
        content validator uses.
      * **rename capacity** (issue #16, defect 2) — identity-free
        originals paired with a near-restatement declared new line by
        :func:`_find_rename_pairs`, the SAME declarative pairing the
        content validator consumes (one-to-one, similarity thresholds,
        per-segment cap), so the trait gate and the content gate can
        never disagree about which removal a rename justifies.

    Segments are sliced exactly like :func:`_build_segments` (leading,
    between consecutive anchors, trailing), so per-segment caps match the
    validator's per-segment justification. The result deliberately
    over-approximates in one documented way: a positional-capacity line is
    taken even when it is itself keyed (the validator would route it
    through the keyed rule) — this only widens the trait floor, never
    rejects a faithful merge.
    """
    anchors = [t[1] for t in raw_tokens if t[0] == "context"]

    # Group the non-context tokens by the anchor they follow (rank -1 =
    # before the first anchor) — the same partition _build_segments uses.
    groups: dict[int, list[tuple[str, int | None, str | None]]] = {}
    rank = -1
    for t in raw_tokens:
        if t[0] == "context":
            rank += 1
        else:
            groups.setdefault(rank, []).append(t)

    # The wrap/insertion-only switch is judged on the CLASSIFIED (stripped)
    # tokens, exactly like the battery's content validator judges it.
    wrap_only = _tokens_declare_insertion_only([
        (kind, idx, value.strip() if value is not None else None)
        for kind, idx, value in raw_tokens
    ])

    if not anchors:
        segments: list[tuple[int, int, int]] = [(-1, 0, len(orig))]
    else:
        segments = [(-1, 0, anchors[0])]
        for k in range(len(anchors) - 1):
            segments.append((k, anchors[k] + 1, anchors[k + 1]))
        segments.append((len(anchors) - 1, anchors[-1] + 1, len(orig)))

    removable: set[int] = set()
    for seg_rank, lo, hi in segments:
        if lo >= hi:
            continue
        seg_tokens = groups.get(seg_rank, [])
        seg_new = [
            t[2] for t in seg_tokens if t[0] == "new" and t[2] is not None
        ]

        # Keyed capacity: unique shared identity within the segment.
        orig_keys = [_replacement_key(orig[i]) for i in range(lo, hi)]
        orig_key_count: dict[str, int] = {}
        for key in orig_keys:
            if key is not None:
                orig_key_count[key] = orig_key_count.get(key, 0) + 1
        new_key_count: dict[str, int] = {}
        for line in seg_new:
            key = _replacement_key(line)
            if key is not None:
                new_key_count[key] = new_key_count.get(key, 0) + 1
        for offset, key in enumerate(orig_keys):
            if (
                key is not None
                and orig_key_count[key] == 1
                and new_key_count.get(key, 0) == 1
            ):
                removable.add(lo + offset)

        # Positional capacity: marker-adjacent identity-free originals,
        # capped by the identity-free new lines declared on each side.
        has_marker = any(t[0] == "marker" for t in seg_tokens)
        if has_marker and seg_new and not wrap_only:
            marker_positions = [
                i for i, t in enumerate(seg_tokens) if t[0] == "marker"
            ]
            first_marker, last_marker = (
                marker_positions[0], marker_positions[-1],
            )
            front_cap = sum(
                1 for t in seg_tokens[:first_marker]
                if t[0] == "new"
                and _replacement_key(t[2] or "") is None
            )
            back_cap = sum(
                1 for t in seg_tokens[last_marker + 1:]
                if t[0] == "new"
                and _replacement_key(t[2] or "") is None
            )
            n = hi - lo
            for j in range(min(front_cap, n)):
                removable.add(lo + j)
            for j in range(min(back_cap, n)):
                removable.add(hi - 1 - j)

    # Complete-restatement replacement (issue #6): when the snippet
    # restates every original content line except exactly one and declares
    # exactly one new line, that uncovered line is the snippet's declared
    # single-line replacement — the SAME predicate the content validator
    # uses (:func:`_complete_restatement_removals`), so the trait gate and
    # the content gate can never disagree about which removal the op
    # declares. Crediting it here makes the trait arithmetic the exact
    # replace-one-line-with-one-line arithmetic on structureless files.
    removable.update(_complete_restatement_removals(orig, raw_tokens))

    # Rename capacity (issue #16, defect 2): the SAME declarative pairing
    # the content validator's deletion gate consumes
    # (:func:`_find_rename_pairs`), applied per segment so the caps and the
    # one-to-one constraint match exactly. Without this the D1 trait gate
    # would count the dropped original's traits as unexplained while the
    # content gate had (correctly) accepted the rename — two gates
    # disagreeing about the same snippet.
    for seg_rank, lo, hi in segments:
        if lo >= hi:
            continue
        seg_tokens = groups.get(seg_rank, [])
        seg_new = [
            t[2] for t in seg_tokens if t[0] == "new" and t[2] is not None
        ]
        seg_orig = orig[lo:hi]
        deleted = list(range(hi - lo))
        pairs = _find_rename_pairs(seg_orig, deleted, seg_new)
        for i in pairs:
            removable.add(lo + i)
    return [orig_raw[i] for i in sorted(removable)]


def _derived_text_op(
    original_code: str,
    snippet: str,
) -> tuple[TextOp, int, dict[str, int]]:
    """Derive the op spec a merge snippet declares, for the D1 trait gate.

    The battery knows the original (chunk or whole file), the merged output
    and the SNIPPET — the snippet is the op spec, so this helper turns it
    into a :class:`TextOp` plus the two uncertainty terms the trait
    comparison needs. Returns ``(op, layout_slack, removable_traits)``:

      * ``op`` — an insertion-shaped op whose payload is the snippet's
        DECLARED new lines, raw and indent included (the same classification
        the content validator uses, via :func:`_classify_snippet_raw`). A
        snippet never names the span it replaces, so ``op.removed`` stays
        empty; the removal capacity is ``removable_traits`` instead.
      * ``layout_slack`` — the snippet's own blank-line count. The
        classifier skips blank lines (they carry no comparable content), so
        a snippet blank is either echoed as new layout or already present
        among the preserved originals and nothing can tell which; the trait
        comparison tolerates exactly that much ``lines``/``blank_lines``
        drift in both directions.
      * ``removable_traits`` — per-trait size of the span(s) the snippet's
        shape could justify removing (see
        :func:`_snippet_justifiable_removals`).

    GRANULARITY (documented approximation): at the whole-file call site the
    inputs are the FULL original and the FULL snippet, so the trait
    arithmetic is exact whole-file arithmetic. At the per-chunk call site
    it is SPAN-LOCAL: the chunk's traits plus the chunk snippet's declared
    delta — the untouched rest of the file never passes through the battery
    there, and is protected mechanically by the byte-exact splice. (Step
    D2's AST-less chunking inherits exactly this span-local contract.)
    """
    orig, orig_raw = _raw_content_lines(original_code)
    raw_tokens = _classify_snippet_raw(snippet, orig)
    payload_lines = [
        value for kind, _idx, value in raw_tokens
        if kind == "new" and value is not None
    ]
    payload = "".join(line + "\n" for line in payload_lines)
    layout_slack = sum(
        1 for line in snippet.splitlines() if not line.strip()
    )
    removable_lines = _snippet_justifiable_removals(orig, orig_raw, raw_tokens)
    removable_traits = text_traits(
        "".join(line + "\n" for line in removable_lines),
    )
    kind = "insert" if payload else "none"
    return TextOp(kind=kind, payload=payload), layout_slack, removable_traits


def _derived_attribute_changes(original_code: str, snippet: str, fmt: str | None):
    """Derive the attribute/structure changes a merge snippet declares (D3).

    The D3 mirror of :func:`_derived_text_op`: the SNIPPET is the op spec,
    so the declared changes are read from it — every attribute-bearing
    line the snippet states (context anchor or declared new line alike) is
    a declared trait that must appear in the output, and the lines the
    snippet's shape could justify removing
    (:func:`_snippet_justifiable_removals`) excuse the traits they carry.
    The snippet classification is the SAME one the content validator uses
    (:func:`_classify_snippet_raw`), so the two gates can never disagree
    about which lines the op declares.

    Issue #19 (defect 1): the ORIGINAL bytes are visible here, so the
    markdown frontmatter-key declaration of a fragment is vouched — a
    ``key:``-shaped prose line (a wrapped paragraph's second line) in a
    document WITHOUT frontmatter is not a frontmatter key, and declaring
    it as one refused every otherwise-correct merge.
    """
    from ..lang_attributes import declared_changes_from_text

    orig, orig_raw = _raw_content_lines(original_code)
    raw_tokens = _classify_snippet_raw(snippet, orig)
    removable_lines = _snippet_justifiable_removals(orig, orig_raw, raw_tokens)
    return declared_changes_from_text(
        snippet,
        "".join(line + "\n" for line in removable_lines),
        fmt,
        target_starts_with_frontmatter=original_code.lstrip("\ufeff").startswith(
            "---",
        ),
    )


def _attribute_rejection_reason(
    original_code: str,
    merged_code: str,
    snippet: str,
    fmt: str | None,
) -> str | None:
    """The D3 language-attribute/structure gate (req. 7 + 9).

    Runs :func:`lang_attributes.attributes_match_expectation` — every
    original trait must appear in the output verbatim at the same
    relative position unless the snippet declares its replacement (then
    the declared value must land exactly), and no undeclared trait may
    appear. Used BOTH as the battery gate below (span-local inputs on
    every attempt) and as the final-assembly exact check (whole-file
    inputs after all chunks splice) — the only assembly-level exact check
    a multi-window md/html/xml/docx edit has.

    Inert for formats without a spec row (``fmt=None``): a .txt file that
    happens to contain ``lang=`` text is not judged.
    """
    if fmt is None:
        return None
    from ..lang_attributes import attributes_match_expectation

    declared = _derived_attribute_changes(original_code, snippet, fmt)
    ok, reason = attributes_match_expectation(
        original_code, merged_code, fmt, declared,
    )
    if ok:
        return None
    return f"merged output failed the language-attribute/structure check ({reason})"


def _merge_rejection_reason(
    original_code: str,
    merged_code: str,
    snippet: str,
    language: str | None,
    result_truncated: bool,
    *,
    fmt: str | None = None,
) -> str | None:
    """The Step A2 validation battery — one gate order for every loop.

    Replaces the old fixed-structure gates (whole-file B13/B29 check plus
    the per-chunk truncation/parse/anchor-score ladder) with ONE battery
    consumed by the unified retry-until-valid loop. Returns ``None`` when
    the attempt may be spliced/returned, else a human-readable reason
    string that the loop feeds back to the model via
    :func:`_append_corrective_note`. Gate order (each short-circuits):

      1. the engine's truncation flag (B12) — a length-capped response is
         a partial file even when its payload looks complete;
      2. the RELATIVE parse gate (req. 9) — :func:`merged_is_acceptable`
         compares the merged diagnostics against the ORIGINAL's: a clean
         original must stay clean; a broken original may only keep its own
         defects (a preserved pre-existing error is a trait; a new error
         anywhere is a regression). Never an absolute well-formedness
         ideal — fastedit validates, it does not correct;
      3. the shared content validator (:func:`_check_hallucinations`) —
         preserve-by-default content traits: dropped unmentioned lines,
         inventions, reorders, unjustified deletions and marker leaks all
         fail. This is the gate that catches a model "helpfully fixing"
         untouched content (req. 9's EDIT-NOT-CORRECT corollary);
      4. the D1 text-trait gate (req. 6, STRUCTURELESS files only) —
         :func:`validate_text_output` compares the output's CJK-aware
         count traits against the original's transformed by the op the
         snippet declares (:func:`_derived_text_op`). Counts come from
         INPUT + op, never from a "correct text" ideal: preserved garbage
         (a duplicated paragraph, an unbalanced quote count) passes and an
         uncommanded change fails. This is the only gate that sees the
         content view's blind spot — blank/whitespace-only lines — and
         the byte volume behind its stripped comparisons;
      5. the D3 language-attribute/structure gate (req. 7 + 9, formats
         with a declared spec row only — ``fmt`` is derived from the
         file's suffix) — :func:`_attribute_rejection_reason` requires
         every original attribute/structure trait (html ``lang``, xml
         ``xml:lang``, latex babel, rtf ``\\langNNNN``, md fence info +
         fence/frontmatter state, docx ``w:lang``) to appear in the
         output verbatim at the same relative position, unless the
         snippet declares its replacement (then the declared value must
         land exactly). MALFORMED traits are traits too (req. 9): a
         broken fence or malformed frontmatter is preserved as-is, and a
         model that "helpfully" repairs it without declaring the repair
         fails here like any other unfaithful merge. The same helper is
         the ASSEMBLY-level exact check for multi-window edits (the
         final-assembly block calls it with whole-file inputs).

    EXTENSION POINT (Step A2): later steps bolt additional validators on
    here — a new validator is one more branch returning a reason string;
    the retry loops themselves never change. (D1 shipped as gate 4 below;
    D2's AST-less chunking reuses it span-locally per chunk; D3 shipped
    as gate 5.)

    ``original_code``/``merged_code`` are span-local inputs: the whole-file
    path passes the FULL file on both sides (the same values the Step 11
    whole-file gate used — the D1 trait arithmetic is exact there), the
    per-chunk path passes the chunk text and the merged chunk (the D1
    arithmetic is span-local — see :func:`_derived_text_op`).
    """
    if result_truncated:
        return "truncated (model hit the token cap)"
    structureless = _is_structureless_language(language)
    if not structureless:
        from ..data_gen.ast_analyzer import parse_diagnostics
        ok, reason = merged_is_acceptable(
            parse_diagnostics(original_code, language),
            parse_diagnostics(merged_code, language),
        )
        if not ok:
            return f"merged output does not parse as {language} ({reason})"
    # Issue #6: the sanctioned single-line replacement is granted ONLY here,
    # for a STRUCTURELESS file — there the text-window snippet IS the op spec
    # (a whole-window restatement the flow itself built from the window's
    # anchors), so a restatement-minus-one-line plus one new line is the one
    # shape whose only consistent reading is "that line was replaced". A code
    # span keeps the strict B4 preserve-by-default contract: without a
    # marker, a one-line transform stays a refusal.
    reason = _hallucination_reason(
        original_code, merged_code, snippet,
        allow_complete_replacement=structureless,
    )
    if reason is not None:
        # Issue #12: name the first failing line and the rule instead of a
        # bare "preserve-by-default violation" — a refused merge must be
        # correctable from the message alone.
        return f"merged output failed the content-faithfulness check ({reason})"
    if structureless:
        # D1 gate (req. 6): no grammar → the parse gate cannot run, so the
        # text-trait oracle validates instead. The op spec is the snippet
        # (what the model was told to insert), the tolerance is the
        # model-prose preset, and the snippet's own blank lines plus the
        # snippet-justifiable removal capacity carry the op-side
        # uncertainty into the comparison.
        op, layout_slack, removable = _derived_text_op(
            original_code, snippet,
        )
        ok, reason = validate_text_output(
            original_code, op, merged_code,
            tolerance=TOLERANCE_MODEL_PROSE,
            layout_slack=layout_slack,
            removable_traits=removable,
        )
        if not ok:
            return f"merged output failed the text-trait check ({reason})"
    # D3 gate (req. 7 + 9): language attributes and structure traits, for
    # formats with a declared spec row (suffix-derived). Inert otherwise.
    attribute_reason = _attribute_rejection_reason(
        original_code, merged_code, snippet, fmt,
    )
    if attribute_reason is not None:
        return attribute_reason
    return None


# Backward-compatible alias: the battery used to be the whole-file gate
# only (Step 11). The per-chunk loop now runs the same battery.
_whole_file_rejection_reason = _merge_rejection_reason


def _rejection_category(reason: str) -> str:
    """Coarsen a battery rejection reason into its category (issue #24, B).

    The reason strings insist on the specific violation, but the retry
    loop's hopeless-shape check only needs to know the category the
    validator lives at — a hopeless merge can still OSCILLATE between two
    strings of one category ("declared new line '…' is missing from the
    merge" ↔ "unmentioned original line dropped: '…'"), which defeated the
    issue-#19 exact-string early stop and burned the whole 9-attempt
    budget (54 s on the issue's 10-line section) for zero chance of
    convergence. Categories:

      * ``truncation`` — the engine hit its token cap (B12);
      * ``parse`` — the relative parse rule (req. 9);
      * ``faithfulness`` — the content validator (dropped / invented /
        reindented / reordered lines, marker leaks);
      * ``trait`` — the D1 text-trait gate (structureless files);
      * ``attribute`` — the D3 language-attribute/structure gate;
      * ``other`` — anything unclassified (conservative: behaves like the
        old exact-string rule, since categories only ever coarsen).

    Consumers: both retry loops' early stop (two consecutive = stop) and
    the rejection bookkeeping threaded to the write-gate refusal (#24 C).
    """
    if reason.startswith("truncated"):
        return "truncation"
    if "does not parse" in reason or "parse error" in reason:
        return "parse"
    if "content-faithfulness" in reason:
        return "faithfulness"
    if "text-trait" in reason:
        return "trait"
    if "language-attribute" in reason:
        return "attribute"
    return "other"

_DEFAULT_MAX_VALIDATION_RETRIES = 8
"""Retry-until-valid budget (req. 5): bounded, loud failure on exhaustion."""


def _max_validation_retries(explicit: int | None = None) -> int:
    """Resolve the unified validation-retry budget (Step A2).

    Precedence: an explicit ``chunked_merge(max_validation_retries=...)``
    argument wins; otherwise the ``FASTEDIT_MAX_RETRIES`` environment
    variable (when set to a non-empty integer) overrides the default of
    :data:`_DEFAULT_MAX_VALIDATION_RETRIES`. Malformed or negative values
    fail loudly (repo convention — cf. ``FASTEDIT_POOL_SIZE`` being parsed
    with a bare ``int()``), never silently fall back.
    """
    if explicit is not None:
        if explicit < 0:
            raise ValueError(
                "max_validation_retries must be a non-negative integer, "
                f"got {explicit}",
            )
        return explicit
    raw = os.environ.get("FASTEDIT_MAX_RETRIES")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_VALIDATION_RETRIES
    value = int(raw)
    if value < 0:
        raise ValueError(
            "FASTEDIT_MAX_RETRIES must be a non-negative integer, "
            f"got {raw!r}",
        )
    return value


def _append_corrective_note(snippet: str, reason: str) -> str:
    """Append a corrective note to the whole-file retry prompt (Step 11).

    ``merge_fn`` takes only ``(original, snippet, language)``, so the
    snippet is the one channel a corrective note can ride on; the prompt
    template embeds it inside ``<update>``, where the model reads it as
    instruction. The note is plain prose — no marker syntax (a marker line
    would change the snippet's segment structure and could itself be
    echoed into a leak) and no language-specific comment prefix (no
    per-language branches). The retry is validated against the ORIGINAL
    snippet, so the note can never contaminate the gate: an attempt that
    echoes the note into code simply fails the validator.
    """
    note = (
        "NOTE: the previous merge attempt was rejected — "
        f"{reason}. Return the COMPLETE file: preserve every original "
        "line the snippet does not explicitly change, in order, and do "
        "not drop, reorder, summarize or invent code."
    )
    return f"{snippet}\n\n{note}"


def _validation_retries_metric(retries: int) -> str:
    """Render ``ChunkedMergeResult.retries`` for a result message's metrics
    segment (Step A3).

    Returns ``", N validation retry"/"retries"`` when *retries* is positive
    and ``""`` otherwise, so callers append it unconditionally and existing
    message shapes stay byte-stable whenever no retry was consumed. Lives
    here because this module owns the retry-until-valid loop and its
    accounting; the MCP tools and the CLI only surface the count — the
    budget itself (``FASTEDIT_MAX_RETRIES`` / ``max_validation_retries``)
    is resolved inside :func:`chunked_merge` and needs no caller plumbing.
    """
    if retries <= 0:
        return ""
    unit = "retry" if retries == 1 else "retries"
    return f", {retries} validation {unit}"


_MAX_MERGE_PROMPT_CHARS = 200_000
"""Character budget for any single text handed to the merge model (issue #13).

Measured envelopes of the fastedit mlx-8bit model (Step A1,
tests/test_real_llm_text.py): the context is 40,960 tokens, the generation
cap 16,384 tokens, and a ~150-line prose chunk measured ~1,713 prompt
tokens — about 11 characters per token. A span the model must re-emit
WHOLE (the whole-file path's output is the whole file; a chunk's output is
the chunk) therefore converges only under ~180,000 characters; anything
larger is guaranteed truncation (B12) followed by 8 wasted retries — and,
the reason this is a HARD gate, a prompt whose engine-side memory scales
with the file: the issue #13 report measured the process climbing to
80–97 GB on a 100 MB single-line file (``total_lines == 1`` sails past the
150-LINE whole-file gate) because tokenize + prefill + the prompt KV cache
all scale with ~25M tokens, on every one of the 8 retry attempts.

Every ``merge_fn`` call site therefore checks the prompt it is about to
send and raises a loud ``ValueError`` naming the size, the budget and the
remedy BEFORE the model is invoked — never a silent multi-GB climb and
never an OOM kill (exit 137). The issue #12 ``_lcs_pair_map`` cap remains
as the last-resort guard on the VALIDATION side; this budget covers the
MODEL side the LCS cap cannot see.
"""

_MAX_SNIPPET_CHARS = 8_000_000
"""Character budget for a merge snippet (issue #13).

A snippet is embedded in every prompt and classified line-by-line by the
marker normalizer, whose string-span masker builds a per-character mutable
copy (:func:`fastedit.split_join.mask_string_spans` — an O(n) list of char
objects, ~8 bytes per ASCII char of pointer plus full string objects for
non-latin-1 content), so an unbounded snippet turns that constant into a
multi-GB spike before any other gate runs. 8 MB bounds that worst case to
a few hundred MB while staying far above any plausible edit (a snippet
larger than the model's 16,384-token output envelope cannot converge
anyway). The prompt budget above bounds the snippet against the model
independently of this sanity cap.
"""


# ---------------------------------------------------------------------------
# Core merge function — the only logic that remains in this file
# ---------------------------------------------------------------------------

def _original_line_endings_are_uniform(original_code: str) -> bool:
    """True when every line ending in ``original_code`` is the same style.

    Counted directly (never via ``splitlines``, which also breaks on other
    Unicode characters). A file is uniform when:

      * it carries CRLF pairs and NOTHING else — every ``\\r`` and every
        ``\\n`` belongs to a pair; or
      * it carries exactly one of lone-CR or bare-LF (or no endings at all).

    A MIXED file (``\\r\\n`` alongside bare ``\\n`` or lone ``\\r``) has no
    single convention to enforce, so the central normalizer must NOT rewrite
    its endings wholesale: the untouched regions' endings are the user's
    bytes, and converting them to the dominant style would corrupt content
    the edit never came near. For such files only the trailing-newline state
    is enforced (see :func:`_normalize_merged_eol`).
    """
    crlf = original_code.count("\r\n")
    cr = original_code.count("\r")
    lf = original_code.count("\n")
    if crlf:
        return crlf == cr and crlf == lf
    return not (cr and lf)


def _normalize_merged_eol(merged_code: str, original_code: str) -> str:
    """Central EOL + trailing-newline funnel for every merge return path.

    Step 14 (B19, B20 remainder, B31, B41). Reuses the shared
    :func:`split_join.detect_line_ending` / ``normalize_line_endings``
    machinery — this helper owns the POLICY, not a second EOL model:

      * **Line-ending convention (B19/B20).** When the original has one
        line-ending convention (:func:`_original_line_endings_are_uniform`),
        every bare-LF piece a splice or model path produced is converted to
        the original's ending — untouched regions already carry that
        ending, so the rewrite is a no-op on them and a repair on the
        produced pieces. A produced CRLF piece inside an LF file is
        converted back the same way (the mirror direction of the same bug).
      * **Mixed-ending originals.** No convention to enforce: endings pass
        through untouched (byte-exactness beats guessing). Only the trailing
        rule below applies.
      * **Trailing-newline state (B31).** Taken from the ORIGINAL, never
        from a ``"\\n"`` constant at a splice site: an original without a
        trailing terminator yields a merged file without one (any appended
        terminator is stripped), and an original with one yields a merged
        file that ends with exactly one (a produced span that lost the
        file's terminator at EOF gets it restored, in the original's own
        ending).

    ``parse_valid`` MUST be computed on this function's output — callers
    funnel BEFORE validating so every parse/validate gate sees the final
    bytes that would be written.

    Passing ``merged_code == original_code`` (the rejection convention:
    the original file is kept as merged_code) returns the original
    verbatim — the funnel is an exact no-op on its own input.
    """
    if merged_code == original_code:
        return merged_code
    line_ending = detect_line_ending(original_code)
    if _original_line_endings_are_uniform(original_code):
        merged_code = normalize_line_endings(merged_code, line_ending)
    if original_code.endswith(("\n", "\r")):
        if merged_code and not merged_code.endswith(("\n", "\r")):
            merged_code += line_ending
    elif merged_code.endswith(("\n", "\r")):
        merged_code = merged_code.rstrip("\r\n")
    return merged_code


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

    from ..split_join import detect_line_ending, normalize_line_endings

    _log = logging.getLogger("fastedit.chunked_merge")

    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    # The file's own prevailing ending must win over the snippet's --
    # inserted/edited spans are normalized to it, untouched original
    # lines are never touched (TRDD-CMRMA2YG line-ending fix).
    line_ending = detect_line_ending(original_code)

    # Race-free in-memory AST parse. `original_code` may differ from what
    # the tldr daemon has cached on disk; parsing in-memory is the only
    # way to guarantee correct line coordinates under chained edits.
    # B3: the caller's `language` rides along as an explicit hint, exactly
    # like the after=/replace= fast paths — an extension-unwired language
    # (all-grammars extra) or a suffix-less path must still resolve.
    ast_nodes = get_ast_map_from_source(original_code, file_path, language) or []
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

    snippet_lines = normalize_line_endings(snippet, line_ending).splitlines(keepends=True)
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

    # Strip marker lines from the snippet — preserve_siblings subsumes
    # their role. Judged with the shared LINE-ANCHORED predicate
    # (markers.is_marker_line, B15 single source of truth): this IS a
    # merge-semantic decision, and the loose legacy ``_MARKER_RE`` regex
    # (kept importable for backward compatibility only) also swallowed
    # real comment lines like ``// ... rest of the class stays ...`` that
    # are content per the unified contract.
    snippet_lines = [ln for ln in snippet_lines if not _is_marker_line(ln)]

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
    # already end with a blank line. Uses the file's own line ending so the
    # splice does not leave a mixed-ending seam (TRDD-CMRMA2YG).
    if preserved_blocks and assembled and assembled[-1].strip() != "":
        assembled.append(line_ending)
    for i, block in enumerate(preserved_blocks):
        assembled.append(block)
        # Blank-line separator between preserved siblings (not after last).
        if i < len(preserved_blocks) - 1 and not block.endswith(("\n\n", "\r\n\r\n")):
            assembled.append(line_ending)
    assembled.extend(snippet_lines[close_idx:])

    # Splice the assembled class body back into the original file.
    result_lines = list(original_lines)
    result_lines[class_start - 1:class_end] = assembled
    merged = "".join(result_lines)
    # Step 14 (B19/B31): the snippet's re-indented lines and the preserved
    # blocks are funnelled through the central normalizer so the emitted
    # endings follow the original's convention and the file's trailing-
    # newline state is the original's, not the snippet's.
    merged = _normalize_merged_eol(merged, original_code)

    # Step A3: RELATIVE parse gate (req. 9) — the last of the absolute
    # ``validate_parse`` gates in this module. The edited span is the
    # ORIGINAL class span in byte offsets: the splice starts at the same
    # byte (everything before the class is untouched), so the span is valid
    # in the merged text's coordinates for the region it covers, and the
    # class body is the op's governed region — the op spec (snippet shell +
    # verbatim preserved siblings) is what governs error traits inside it.
    # Outside that span the relative rule still applies unchanged: a clean
    # original must stay clean, and an error the splice introduces elsewhere
    # is a regression. A preserved sibling's pre-existing defect is a trait:
    # inherited by its (kind, line) identity — the class lands where the
    # absolute gate used to refuse the identical, faithful splice.
    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import parse_diagnostics
        class_span_start = len(
            "".join(original_lines[:class_start - 1]).encode("utf-8"),
        )
        class_span_end = class_span_start + len(
            "".join(original_lines[class_start - 1:class_end]).encode("utf-8"),
        )
        parse_valid, _rel_reason = merged_is_acceptable(
            parse_diagnostics(original_code, language),
            parse_diagnostics(merged, language),
            edited_spans=[(class_span_start, class_span_end)],
        )
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



def _extract_signature_via_ast(
    source: str,
    language: str | None,
    target_start_line: int,
    target_end_line: int,
    fallback_line: str,
) -> str:
    """Return the full signature of a function/method/class, handling
    multi-line parameter lists.

    Uses tree-sitter's ``body`` field to find where the body begins; the
    signature is everything from the node's start byte up to (not including)
    the body's start byte. This correctly captures multi-line ``def foo(\n    a,\n    b,\n):``
    shapes that the prior heuristic (``original_lines[func_start]``) truncated to
    ``def foo(``, silently producing unclosed parens in the merged output.

    When the grammar places the body-opening delimiter at the start of the
    body node, the span is extended past it so the extracted signature ends
    AFTER the opener (B9). tree-sitter materializes such an opener as an
    anonymous child token at the body's own start byte — ``{`` for the
    C-family/Rust/TS/Go/Swift/Java class of grammars — so detecting it needs
    no language keyword list and no per-language branch: the check is purely
    structural (first body child, anonymous, starts exactly at the body,
    text is a single opening delimiter). Grammars whose opener is part of
    the signature line itself (Python's ``:``, a direct child of the
    function node) or that have no opener token (Ruby) need no extension:
    the colon is already inside the byte span, and prepending Ruby's
    brace-less signature is correct. A grammar that wraps the opener in an
    extra named node (Kotlin's ``function_body``) is a known limit — its
    span still ends before the brace, exactly as before this fix.

    Falls back to ``fallback_line`` when:
    - language is None or unsupported
    - tree-sitter parsing fails
    - no function-like node matches the target byte range
    - the matching node has no ``body`` field (unusual grammar)
    """
    if not language:
        return fallback_line
    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: docstring lists "tree-sitter parsing fails" as a fallback trigger; degrade to fallback_line, never propagate
        return fallback_line

    src_bytes = source.encode("utf-8")
    # tree-sitter rows are 0-indexed; ASTNode.line_start is 1-indexed.
    target_row = target_start_line - 1

    def walk(node):
        # Match any function/method/class-like node that starts on the
        # target row AND has a body field with a later start byte.
        if node.start_point[0] == target_row:
            # Most grammars expose the body via a named field called "body"
            # (Python, Rust, Go, JS/TS, Java, C/C++, Ruby, Swift, PHP, C#).
            body = node.child_by_field_name("body")
            if body is None:
                # Kotlin uses an unnamed "function_body" child; Elixir parses
                # `def foo(a) do ... end` as a call with a "do_block" child.
                for child in node.children:
                    if child.type in ("function_body", "do_block"):
                        body = child
                        break
            if body and body.start_byte > node.start_byte:
                end_byte = body.start_byte
                # B9: include the body-OPENING delimiter so the prepended
                # signature never yields a brace-less splice. The grammar
                # places the opener as an anonymous child token starting
                # exactly at the body's start byte (``{`` for C-family/
                # Rust/TS/Go/Swift/Java); structural detection only — no
                # language keyword lists. A named first child at the same
                # byte (Python's block starts at the first statement) or an
                # opener the grammar folds into the signature line (Python's
                # ``:``, already inside the span) needs no extension.
                if body.child_count:
                    first = body.children[0]
                    if (
                        first.start_byte == body.start_byte
                        and not first.is_named
                        and src_bytes[first.start_byte:first.end_byte]
                        in (b"{", b"(", b"[")
                    ):
                        end_byte = first.end_byte
                # B3: the AST node starts at its first TOKEN — the line's
                # leading whitespace is not part of the node. Extend the
                # span back over that whitespace so the prepended signature
                # keeps the target's own indentation: a class member is
                # indented, and without this the direct-swap alignment
                # compensates by shifting the WHOLE snippet (signature plus
                # body) by the missing columns, mis-indenting every line.
                start_byte = node.start_byte
                line_start = source.rfind("\n", 0, start_byte) + 1
                prefix = source[line_start:start_byte]
                if prefix and not prefix.strip():
                    start_byte = line_start
                return src_bytes[start_byte:end_byte].decode(
                    "utf-8", errors="replace",
                )
        for child in node.children:
            # Skip branches that can't contain the target
            if child.start_point[0] > target_end_line - 1:
                break
            if child.end_point[0] < target_row:
                continue
            result = walk(child)
            if result is not None:
                return result
        return None

    sig = walk(tree.root_node)
    if sig is None:
        return fallback_line
    # Trim trailing whitespace/newlines, then re-terminate with a single \n
    # so the prepended signature forms exactly one well-formed line prefix.
    return sig.rstrip() + "\n"


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
    max_validation_retries: int | None = None,
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
        max_validation_retries: Step A2 retry-until-valid budget (req. 5):
            how many RETRY attempts the unified validation loop may spend
            per merge site (whole-file or per chunk) after the first. None
            (default) resolves to ``FASTEDIT_MAX_RETRIES`` when set, else
            8. Every attempt runs the shared battery
            (:func:`_merge_rejection_reason`: relative parse + content
            faithfulness); a rejected attempt retries with its failure
            reason appended to the prompt; exhaustion rejects the site
            with the existing bookkeeping. Parse validation is RELATIVE
            (req. 9): a preserved pre-existing defect is a trait, new
            breakage is not tolerated.

    Returns:
        ChunkedMergeResult with the fully merged file. ``retries`` counts
        the validation-retry attempts consumed across all sites.
    """
    # preserve_siblings is only meaningful on a `replace=` edit. Fail
    # early and loudly so callers don't silently no-op.
    if preserve_siblings and not replace:
        raise ValueError(
            "preserve_siblings=True requires replace=ClassName. "
            "The flag controls how `replace=` behaves when the snippet "
            "describes only a subset of the class's members."
        )

    # Issue #13: bound the SNIPPET before the marker normalizer runs — its
    # string-span masker builds a per-character mutable copy, so an unbounded
    # snippet is a multi-GB spike before any other gate executes. Loud
    # refusal with the remedy, never a silent climb.
    if len(snippet) > _MAX_SNIPPET_CHARS:
        raise ValueError(
            f"snippet too large: {len(snippet):,} characters exceeds the "
            f"{_MAX_SNIPPET_CHARS:,}-character limit. Split the edit into "
            f"smaller snippets (one target per call)."
        )

    # Normalize short / Unicode marker forms (v0.2.4) → canonical long
    # form. All downstream code (chunk_locator, text_match, model paths,
    # snippet_analysis) continues to see ``# ... existing code ...`` /
    # ``// ... existing code ...``; no other module needs to know about
    # the short forms. See ``normalize_markers`` docstring. Imported from
    # the shared markers module (single source of truth, B15).
    snippet = _normalize_markers(snippet)

    # Step 14 (B19/B20/B31): the file's own line-ending convention is
    # detected ONCE, here, and reused by every splice site below; the file's
    # trailing-newline state is enforced ONCE per return path by
    # _normalize_merged_eol, so no splice site decides it with a hardcoded
    # "\n".
    line_ending = detect_line_ending(original_code)

    # Step D3: the language-attribute/structure spec format is derived ONCE
    # from the file's suffix and threaded through every battery call (the
    # whole-file gate, the per-chunk gates and the final-assembly exact
    # check). ``None`` for formats without a spec row — the gate is inert.
    attribute_format = format_for_path(file_path)

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
        # B16: this path splices the snippet VERBATIM, so a preservation
        # marker inside it would be written into the file as a literal
        # comment line — silent corruption (it parses as a comment, so no
        # downstream check ever objects). Markers are merge directives,
        # never content: rewrite short/Unicode forms to the canonical long
        # form (idempotent — chunked_merge() already normalized the snippet
        # above; repeated here so this fast path stays correct on its own)
        # and DROP the marker-only lines BEFORE any indent arithmetic, so a
        # leading marker line cannot become the alignment base.
        snippet_text = _normalize_markers(snippet.rstrip("\n") + "\n")
        raw_snippet_lines = snippet_text.splitlines(keepends=True)
        kept_snippet_lines = [
            ln for ln in raw_snippet_lines if not _is_marker_line(ln)
        ]
        dropped_markers = len(raw_snippet_lines) - len(kept_snippet_lines)
        if dropped_markers:
            _log.warning(
                "after='%s': dropped %d preservation marker line(s) from the "
                "snippet — markers are directives to the merge pipeline, "
                "never content",
                after, dropped_markers,
            )
        if not any(ln.strip() for ln in kept_snippet_lines):
            # Fail loudly (repo convention): silently splicing an empty
            # piece would return a parse-valid, zero-token result
            # indistinguishable from a successful insert.
            raise ValueError(
                f"after='{after}' snippet carries no insertable code: every "
                f"line is a preservation marker or blank. Pass the new code "
                f"to insert after '{after}'."
            )
        snippet_text = "".join(kept_snippet_lines)

        # Parse the in-memory `original_code` (authoritative) instead of
        # shelling out to `tldr structure`, which consults a daemon cache
        # that can return stale line numbers after a recent write. See
        # `get_ast_map_from_source` for rationale. B3: the caller's
        # `language` rides along as an explicit hint so extension-unwired
        # languages (all-grammars extras) still get symbol anchoring.
        ast_nodes = get_ast_map_from_source(original_code, file_path, language)
        anchor_node = _resolve_symbol(after, ast_nodes or [])
        if anchor_node is None:
            available = _qualified_symbol_names(ast_nodes or [])
            raise ValueError(
                f"Symbol '{after}' not found in {file_path}. "
                f"Available: {available}"
            )

        # Insert after the anchor's last line — UNLESS the anchor is a
        # format symbol whose FIRST line is its naming line
        # (``definition_line_required``) and the snippet's first line is
        # itself a definition line of the same kind naming the same
        # symbol (issue #16, defect 3): the reported duplication shape,
        # ``--snippet '#### S001: ...' --after 'Scenario: ...'`` landed
        # the new heading at the END of the section span (60 lines away),
        # duplicating the S001 heading. "After the symbol" for such
        # snippets reads as "after the symbol's DEFINITION LINE". The
        # decision is declarative (the format spec table + the same
        # extractor that names the symbol), never per-format branches.
        insert_after_line = anchor_node.line_end
        head_at_definition_line = _snippet_head_is_same_kind_definition(
            snippet_text, anchor_node, language or "",
            anchor_line=(
                original_lines[anchor_node.line_start - 1]
                if anchor_node.line_start - 1 < len(original_lines) else ""
            ),
        )
        if head_at_definition_line:
            insert_after_line = anchor_node.line_start
            _log.info(
                "Fast-path insert after '%s': snippet head restates the "
                "symbol's definition line — inserting after L%d (the "
                "definition line) instead of the span end L%d",
                after, insert_after_line, anchor_node.line_end,
            )
        before = original_lines[:insert_after_line]
        after_lines = original_lines[insert_after_line:]

        # Align snippet indent to match the anchor's indent level.
        anchor_start_idx = anchor_node.line_start - 1
        anchor_first_line = original_lines[anchor_start_idx] if anchor_start_idx < total_lines else ""
        snippet_text = _align_snippet_indent(snippet_text, anchor_first_line)

        # B20 (splice half): the inserted piece must carry the ORIGINAL's
        # line-ending convention — a bare-LF piece inside a CRLF file is a
        # mixed-ending seam. Step 14 funnels the assembled merge through
        # _normalize_merged_eol below; normalizing the piece here as well
        # keeps the piece and its blank-line separators in one convention
        # before the (idempotent) funnel re-check.
        snippet_text = normalize_line_endings(snippet_text, line_ending)
        snippet_parts = snippet_text.splitlines(keepends=True)

        # Ensure blank line separator before and after the new code —
        # spelled with the file's own ending, never a hardcoded "\n".
        # Definition-line landing (issue #16 defect 3): the new content
        # belongs DIRECTLY after the anchor's naming line (a subsection
        # heading sits with its parent heading, no blank between), so the
        # leading separator is suppressed there.
        at_definition_line = head_at_definition_line
        if at_definition_line:
            # No-duplicate invariant (issue #16 defect 3): when the snippet's
            # head definition line ALREADY exists inside the anchor span
            # (the reported shape — the '#### S001' heading re-inserted after
            # its own parent section), the splice must not create a SECOND
            # copy. The pre-existing identical line is dropped so the head
            # lands exactly once, at the definition line. Only the head line
            # participates — the snippet's remaining lines are new content.
            snippet_head_line = next(
                (ln.rstrip("\r\n") for ln in snippet_parts if ln.strip()), None,
            )
            if snippet_head_line is not None:
                # The pre-existing copy lies INSIDE the anchor span; given
                # the definition-line insertion point it is AFTER the splice
                # slot — drop it from `after_lines` (never from `before`,
                # which ends at the naming line by construction).
                for i, ln in enumerate(after_lines):
                    if ln.strip() == snippet_head_line.strip():
                        after_lines = [
                            l2 for j, l2 in enumerate(after_lines) if j != i
                        ]
                        break
        separator = [] if at_definition_line else (
            [line_ending] if before and before[-1].strip() != "" else []
        )
        trailing = [line_ending] if after_lines and after_lines[0].strip() != "" else []

        result_lines = before + separator + snippet_parts + trailing + after_lines
        merged = "".join(result_lines)
        # Step 14: funnel — the after= path already normalizes its own piece,
        # so this is a no-op here; every return path routes through it so the
        # trailing-newline state is enforced in exactly one place.
        merged = _normalize_merged_eol(merged, original_code)

        # Step A3: RELATIVE parse gate (req. 9) — the last of the absolute
        # ``validate_parse`` gates in this module. This path INSERTS the
        # snippet and never touches an original byte, so the relative rule
        # applies directly with NO excused edited span: an insertion cannot
        # fix a pre-existing error elsewhere, so every original error trait
        # survives byte-exact and is inherited as a trait, while a malformed
        # snippet (or its interaction with the kept lines) shows up as a NEW
        # trait — a regression the insertion zone must answer for, whether
        # the file was clean or already broken. Refusal style is unchanged:
        # the result is returned with parse_valid=False (zero-model, no
        # retry, no exception) and the MCP/CLI write gates refuse it.
        parse_valid = True
        if language:
            from ..data_gen.ast_analyzer import parse_diagnostics
            parse_valid, _rel_reason = merged_is_acceptable(
                parse_diagnostics(original_code, language),
                parse_diagnostics(merged, language),
            )

        _log.info(
            "Fast-path insert after '%s' (L%d): %d snippet lines, 0 model tokens",
            after, insert_after_line, len(snippet_parts),
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
        extras = _top_level_extras(snippet, language, replace)
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
        from .text_match import _bracket_balance, deterministic_edit

        # In-memory parse (race-free). See comment above the `after:` fast
        # path for why we do not consult the tldr daemon here. B3: the
        # caller's `language` rides along as an explicit hint.
        ast_nodes = get_ast_map_from_source(original_code, file_path, language)
        target_node = _resolve_symbol(replace, ast_nodes or [])
        if target_node:
            func_start = target_node.line_start - 1  # 0-indexed
            func_end = target_node.line_end  # 1-indexed inclusive
            original_func = "".join(original_lines[func_start:func_end])

            # Issue #4: JSON key-level edits — a bare JSON VALUE snippet or
            # a restated `"key": value` pair splices ONLY the pair's value
            # (and the key on rename) instead of refusing, falling to the
            # model, or wiping sibling keys. Anything else declines here
            # and keeps the existing gates.
            if language == "json" and target_node.kind == "key":
                json_fast = _try_json_key_replace(
                    original_code, snippet, replace, target_node,
                )
                if json_fast is not None:
                    merged = _normalize_merged_eol(json_fast[0], original_code)
                    _log.info(
                        "JSON key replace for replace='%s': 0 model tokens "
                        "(value-only splice)", replace,
                    )
                    return ChunkedMergeResult(
                        merged_code=merged,
                        parse_valid=json_fast[1],
                        chunks_used=0,
                        chunk_regions=[],
                        model_tokens=0,
                        latency_ms=0.0,
                    )

            # Issue #3: a snippet restating the doc comment ABOVE the
            # symbol used to duplicate it (the span starts at the
            # definition line; the comment above stayed AND the snippet's
            # copy was spliced in). Strip the restated head before the
            # classifier or the signature prepend sees it.
            snippet, _stripped_doc_lines = _strip_leading_doc_duplicate(
                snippet, original_lines, func_start,
            )

            # B9: how many leading snippet lines ARE the auto-prepended
            # signature span (0 when no prepend happens). Threaded into
            # deterministic_edit so the span can be pinned as fixed context.
            prepend_signature_lines = 0

            # Auto-preserve signature: when the caller passes replace=<name>
            # and the snippet doesn't contain the target's def/fn/func line,
            # prepend it from the AST. The replace= kwarg already identifies
            # the target structurally — requiring the signature in the snippet
            # is redundant. Without this guard the deterministic_edit path
            # (and the direct-swap fallback) treat the snippet as the full
            # new body and silently strip the signature. See regression test
            # ``test_replace_with_body_only_snippet_preserves_signature``.
            #
            # Scope: function/method/class-like symbols only. Constants,
            # fields, and other value-like targets are intentionally
            # excluded — their "body" is a single expression and callers
            # always include the full declaration. Auto-prepending on a
            # const turns a valid full-replacement snippet into a
            # two-declaration blob that breaks deterministic_edit (see
            # ``test_direct_swap_extend_literal_rust``).
            _SIGNATURE_KINDS = {
                "function", "method", "class", "interface",
                "struct", "enum", "trait", "impl", "module",
                "object", "protocol",
            }
            node_size = target_node.line_end - target_node.line_start + 1
            if (
                target_node.kind in _SIGNATURE_KINDS
                and func_start < len(original_lines)
                # C3: when the locator will narrow this target to a
                # sub-block, the signature is OUTSIDE the chunk the model
                # will see — prepending it makes the snippet declare a line
                # the merge must not contain (the model echoes the snippet's
                # first line into its output, the validator rejects, and a
                # narrowed edit can never converge — found by the C3 100MB
                # seams stress on every language). The narrow only engages
                # on marker-bearing snippets, which the deterministic paths
                # decline anyway, so skipping the prepend here changes no
                # deterministic behavior.
                and not _narrow_will_engage(snippet, node_size)
                and not _snippet_has_target_signature(snippet, replace)
            ):
                # Multi-line signatures (def foo(\n    a,\n    b,\n):) require
                # walking the AST to find where the body begins. The old
                # approach grabbed only line[func_start] — the ``def foo(``
                # line — and dropped the continuation, producing unclosed
                # parens. _extract_signature_via_ast returns the full span
                # up to and including the body-opening delimiter (B9) and
                # falls back to the single-line behavior for unusual
                # grammars.
                fallback = original_lines[func_start]
                if not fallback.endswith("\n"):
                    fallback += "\n"
                target_signature_line = _extract_signature_via_ast(
                    original_code, language,
                    target_node.line_start, target_node.line_end,
                    fallback,
                )
                snippet = target_signature_line + snippet
                prepend_signature_lines = target_signature_line.count("\n")
                _log.info(
                    "replace='%s': snippet missing signature, auto-prepended "
                    "from AST (line %d, %d lines)",
                    replace, func_start + 1, prepend_signature_lines,
                )

            edited = deterministic_edit(
                original_func, snippet,
                prepend_signature_lines=prepend_signature_lines,
            )
            if edited is not None and _deterministic_result_is_faithful(
                original_func, edited, snippet, prepend_signature_lines,
            ):
                edited_lines = edited.splitlines(keepends=True)
                result_lines = list(original_lines)
                result_lines[func_start:func_end] = edited_lines
                merged = "".join(result_lines)
                # B19/B31 (Step 14): the editor re-emits the replaced span
                # LF-only and nothing here may decide the file's trailing-
                # newline state with a constant. The central funnel converts
                # the span to the original's ending (a mid-file span keeps
                # the terminator deterministic_edit's own finisher gave it)
                # and enforces the original's trailing state at EOF.
                merged = _normalize_merged_eol(merged, original_code)

                # Step A2: RELATIVE parse gate (req. 9). The original's own
                # diagnostics are the baseline: a clean original must stay
                # clean; a broken original accepts its own preserved
                # defects (EDIT-NOT-CORRECT — the editor must not decline
                # an unrelated edit, nor may it silently "repair" the
                # broken region the snippet never mentioned).
                parse_valid = True
                if language:
                    from ..data_gen.ast_analyzer import parse_diagnostics
                    parse_valid, _rel_reason = merged_is_acceptable(
                        parse_diagnostics(original_code, language),
                        parse_diagnostics(merged, language),
                    )

                if not parse_valid:
                    # The splice is content-faithful but the RELATIVE parse
                    # rule rejects it — e.g. a brace-language full-function
                    # snippet whose new body line cannot coexist with the
                    # kept original body line, or (on an already-broken
                    # file) a splice introducing NEW error traits. Hold the
                    # editor's output to the same structural standard it
                    # applies to partial snippets (see the direct-swap gate
                    # below): discard it and let the qualified direct-swap
                    # — or the validated model path — produce a writable
                    # merge, instead of returning output the tool gates
                    # would only refuse.
                    _log.warning(
                        "Deterministic text-match for replace='%s' produced "
                        "a relatively parse-invalid merge; discarding it "
                        "and falling through to direct-swap/model",
                        replace,
                    )
                else:
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
            elif edited is not None:
                # B3: the editor's own decline rules cover structure, but
                # a content-corrupt result (dropped original line, leaked
                # marker, selective re-indent) parses fine. Defense in
                # depth: discard it and take the ordinary direct-swap /
                # model route below — the deterministic path never
                # bypasses the validator.
                _log.warning(
                    "Deterministic result for replace='%s' failed the "
                    "content-faithfulness check (%s); discarding it and "
                    "falling through to the model path",
                    replace,
                    _deterministic_result_unfaithful_reason(
                        original_func, edited, snippet, prepend_signature_lines,
                    ),
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
            #   4. The snippet is a COMPLETE re-definition: its bracket
            #      balance equals the replaced span's. A partial snippet
            #      (e.g. signature prepended, body given, closer still
            #      unspoken) would splice into an unbalanced, parse-invalid
            #      file — the same structural gate the text-match editor
            #      applies to its own output. Decline to the model instead.
            if not any(_is_marker_line(ln) for ln in snippet.splitlines()):
                from pathlib import Path as _Path
                snippet_parse = _try_tldr_snippet_parse(
                    snippet, _Path(file_path).suffix,
                )
                snippet_balance = _bracket_balance(snippet)
                span_balance = _bracket_balance(original_func)
                snippet_is_complete = snippet_balance == span_balance
                # Issue #5: for a call-symbol target (jest it()/test()/
                # describe() blocks) the flat name list overcounts — the
                # in-memory walker recurses INTO call bodies, so the
                # snippet's inner constants surface as extra "definitions"
                # even though they are members of the call being swapped.
                # The completeness proof for a call target reads the
                # snippet's PARENTLESS names instead (exactly one, named
                # `replace`). The declarative spec table decides; code
                # symbols keep the flat single-name check.
                _call_spec = _CALL_SYMBOL_SPECS.get(language or "")
                _call_snippet_swap = (
                    _call_spec is not None
                    and target_node.kind == _call_spec.kind
                    and _call_symbol_snippet_top_level_names(
                        snippet, file_path, language,
                    ) == [replace]
                )
                if snippet_parse and not snippet_is_complete:
                    _log.info(
                        "Direct-swap declined for replace='%s': snippet "
                        "parses but is not a complete re-definition "
                        "(bracket balance %d vs %d) — falling through",
                        replace, snippet_balance, span_balance,
                    )
                if (
                    snippet_parse
                    and snippet_is_complete
                    and (
                        (len(snippet_parse) == 1 and snippet_parse[0] == replace)
                        or _call_snippet_swap
                    )
                ):
                    # Align snippet indent to match the target function's
                    # indent in the original.
                    anchor_first_line = (
                        original_lines[func_start]
                        if func_start < total_lines
                        else ""
                    )
                    # B31 (Step 14): the terminator appended here is only the
                    # SPLICE SEPARATOR a mid-file replacement needs (an
                    # unterminated last line would concatenate onto the next
                    # original line). It is spelled with the file's own
                    # ending; whether the FILE ends with a terminator is
                    # decided by _normalize_merged_eol below, from the
                    # original — never by this constant.
                    snippet_text = snippet.rstrip("\r\n") + line_ending
                    # Issue #23: same alignment as the CLI direct-swap
                    # splice — the snippet is re-indented to the TARGET
                    # symbol's own level (anchor_first_line IS the target's
                    # head line here; the replaced span rides along as the
                    # indent-char vote context), never spliced at its
                    # caller-chosen indent.
                    snippet_text = _align_snippet_to_symbol_indent(
                        snippet_text,
                        anchor_first_line,
                        "".join(original_lines[func_start:func_end]),
                    )
                    snippet_lines = snippet_text.splitlines(keepends=True)
                    result_lines = list(original_lines)
                    result_lines[func_start:func_end] = snippet_lines
                    merged = "".join(result_lines)
                    # Step 14 (B19): a bare-LF snippet swapped into a CRLF
                    # file is converted to the file's convention here — the
                    # splice that leaked CR bytes on the batch/multi paths.
                    merged = _normalize_merged_eol(merged, original_code)

                    # Step A2: RELATIVE parse gate (req. 9) — the same
                    # trait-based standard as the text-match gate above: a
                    # preserved pre-existing defect is acceptable, new
                    # breakage is not.
                    parse_valid = True
                    if language:
                        from ..data_gen.ast_analyzer import parse_diagnostics
                        parse_valid, _rel_reason = merged_is_acceptable(
                            parse_diagnostics(original_code, language),
                            parse_diagnostics(merged, language),
                        )

                    # Issue #1 (hole 1): parse validity cannot see content
                    # corruption in a wholesale swap. For a CODE symbol the
                    # branch's structural proof stands: the snippet parsed
                    # as exactly one definition and the bracket balance
                    # matches the span's, so it IS the complete new symbol.
                    # A FORMAT symbol (definition_line_required spec row)
                    # has no closing delimiter — nothing structural
                    # distinguishes a complete re-definition from a snippet
                    # restating the definition line while dropping body
                    # lines — so the splice-completeness check supplies the
                    # missing proof. Failing it → decline to the validated
                    # model path, never write the mangled swap.
                    format_spec = _FORMAT_SYMBOL_SPECS.get(language or "")
                    # Issue #5: a call-symbol swap carries the same
                    # completeness burden as a format-symbol swap — the
                    # spec row declares the naming line required, and the
                    # splice-completeness check below proves no body line
                    # was silently dropped.
                    call_spec = _CALL_SYMBOL_SPECS.get(language or "")
                    completeness_unproven = (
                        format_spec is not None and format_spec.definition_line_required
                    ) or (
                        call_spec is not None
                        and target_node.kind == call_spec.kind
                    )
                    if (
                        completeness_unproven
                        and not _snippet_splice_covers_deleted_lines(
                            original_func, snippet_text,
                        )
                    ):
                        _log.warning(
                            "Direct-swap for replace='%s' would delete "
                            "original lines the snippet does not restate; "
                            "discarding it and falling through to the "
                            "validated model path",
                            replace,
                        )
                    else:
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

    # Step D2: text-anchor windows are WINDOW chunks even when a window
    # happens to span the whole file (small file, or anchors closer than
    # the window budget). The snippet's anchor declared where the edit
    # goes — exactly the information the whole-file gate below exists to
    # demand — so such an edit bypasses that gate and runs through the
    # per-chunk loop like any other chunked edit. Each window gets its
    # own snippet portion (``_text_window_snippets``): a multi-anchor
    # snippet must not hand every window the anchors of the others, or
    # the span-local battery would bind a foreign anchor as a new line
    # and demand the model duplicate it.
    text_window_snippets: list[str] | None = None
    if chunks and all(_TEXT_ANCHOR_TAG in c.matched_nodes for c in chunks):
        text_window_snippets = _text_window_snippets(
            snippet, [(c.start_line, c.end_line) for c in chunks],
            original_code.splitlines(),
        )
    is_text_anchor_edit = text_window_snippets is not None

    # Reject whole-file merge on large files — model will truncate
    _MAX_WHOLE_FILE_LINES = 150
    is_whole_file = (
        len(chunks) == 1
        and chunks[0].start_line == 1
        and chunks[0].end_line == total_lines
        and not is_text_anchor_edit
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

    # Step A2: resolve the unified retry-until-valid budget once — both the
    # whole-file path and the per-chunk loop below consume it.
    max_retries = _max_validation_retries(max_validation_retries)

    # If only one chunk covering the whole file, just do a normal merge
    if is_whole_file:
        # B33: one random nonce per merge attempt — the placeholders sent to
        # the model cannot collide with user text (e.g. a file that literally
        # contains the old fixed placeholder string).
        tag_nonce = _new_tag_nonce()
        # Issue #13 (gate placement): bound the prompt BEFORE the tag-escape
        # transform. Escape never shortens (placeholders are longer than the
        # tags, lengths identical otherwise), so the ORIGINAL length is an
        # exact lower bound of the escaped prompt's — this check refuses
        # oversized input without paying the full-text escape scan/copy.
        if len(original_code) > _MAX_MERGE_PROMPT_CHARS:
            raise ValueError(
                f"file too large for a whole-file merge: {len(original_code):,} "
                f"characters exceeds the {_MAX_MERGE_PROMPT_CHARS:,}-character "
                f"model budget (the model cannot re-emit a file this large; "
                f"it would be truncated and rejected). Edit a smaller target "
                f"instead: pass a snippet whose unique context lines anchor "
                f"the edited window, use `after=`/`replace=` to scope the "
                f"edit to one symbol, or split the file first "
                f"(`fastedit split --lines N`)."
            )
        safe_code = _escape_tags(original_code, tag_nonce)
        safe_snippet = _escape_tags(snippet, tag_nonce)

        # Issue #13: the whole-file prompt must fit the model's re-emit
        # envelope. The 150-LINE gate above cannot see a 100 MB SINGLE-LINE
        # file (total_lines == 1) or a few-line file of megabyte-long lines —
        # exactly the reported 80–97 GB shape: the whole file went to
        # merge_fn on every retry attempt and the engine's tokenize/prefill/
        # KV memory scaled with it. Fail loud with the remedy BEFORE any
        # model call; the model path is never entered with a prompt the
        # model cannot re-emit.
        if len(safe_code) > _MAX_MERGE_PROMPT_CHARS:
            raise ValueError(
                f"file too large for a whole-file merge: {len(safe_code):,} "
                f"characters exceeds the {_MAX_MERGE_PROMPT_CHARS:,}-character "
                f"model budget (the model cannot re-emit a file this large; "
                f"it would be truncated and rejected). Edit a smaller target "
                f"instead: pass a snippet whose unique context lines anchor "
                f"the edited window, use `after=`/`replace=` to scope the "
                f"edit to one symbol, or split the file first "
                f"(`fastedit split --lines N`)."
            )

        # Step A2 unified retry-until-valid loop (req. 5 + req. 9). Replaces
        # the fixed initial-call + single corrective retry: EVERY attempt
        # runs the same battery (relative parse + content faithfulness; see
        # _merge_rejection_reason) and a rejected attempt retries with its
        # failure reason appended to the prompt. The note is rebuilt from
        # the ORIGINAL snippet each time (one note, latest reason), and the
        # gate always validates against the ORIGINAL snippet, so the note
        # itself can never contaminate the gate.
        attempt_snippet = safe_snippet
        result = None
        merged = None
        reason: str | None = None
        attempts = 0
        whole_tokens = 0
        whole_latency = 0.0
        prev_reason: str | None = None
        rejection_categories: set[str] = set()
        for attempt in range(max_retries + 1):
            result = merge_fn(safe_code, attempt_snippet, language)
            attempts += 1
            whole_tokens += result.tokens_generated
            whole_latency += result.latency_ms
            merged = _unescape_tags(result.merged_code, tag_nonce)
            # Step 14 (B19/B31): the model's payload is funnelled through the
            # central normalizer BEFORE the battery, so the parse and
            # content validators see the exact bytes that would be written.
            merged = _normalize_merged_eol(merged, original_code)
            # Step 11 (B13/B29) battery: the whole-file path hands the ENTIRE
            # file to the model, so its output is validated before it may be
            # returned — truncation flag (B12), RELATIVE parse rule (req. 9),
            # and the shared content validator over the FULL original vs the
            # FULL merge output.
            reason = _merge_rejection_reason(
                original_code, merged, snippet, language,
                getattr(result, "truncated", False),
                fmt=attribute_format,
            )
            if reason is None:
                break
            _log.warning(
                "Whole-file merge attempt %d/%d rejected: %s",
                attempt + 1, max_retries + 1, reason,
            )
            # Issue #19 (defect 2), issue #24 (defect B): when the battery
            # rejects with the same REJECTION CATEGORY on consecutive
            # attempts (parse / faithfulness / truncation / ...), the retry
            # budget cannot change the verdict. This OBSERVED hopeless
            # shape oscillates two faithfulness strings
            # ("declared new line missing" ↔ "unmentioned original line
            # dropped") — comparing EXACT strings (the #19 rule) was
            # defeated by the oscillation, 9 attempts on a 10-line file;
            # the category rule stops it after 2. A converging merge never
            # hits this: its rejection category changes between attempts
            # (cycling categories still exhaust the budget) or the
            # rejection disappears.
            final_category = _rejection_category(reason)
            rejection_categories.add(final_category)
            if final_category == prev_reason:
                _log.warning(
                    "Whole-file merge rejected with the same category %d "
                    "attempt(s) in a row (%s: %s) — stopping retries early",
                    2, final_category, reason,
                )
                break
            prev_reason = final_category
            attempt_snippet = _append_corrective_note(safe_snippet, reason)

        if reason is not None:
            # Budget exhausted: reject the whole-file merge — no write.
            # Rejection convention = the chunk loop's: the original file is
            # kept as merged_code (never the corrupted payload), parse_valid
            # is forced False (the universal "do not persist this" signal),
            # and the chunks_rejected/chunks_used accounting makes the
            # existing MCP/CLI gates refuse the write naturally.
            _log.error(
                "Whole-file merge rejected after %d attempt(s) (%s) — "
                "keeping the original file",
                attempts, reason,
            )
            return ChunkedMergeResult(
                merged_code=original_code,
                parse_valid=False,
                chunks_used=1,
                chunk_regions=[(1, total_lines)],
                model_tokens=whole_tokens,
                latency_ms=whole_latency,
                chunks_rejected=1,
                retries=attempts - 1,
                rejection_categories=tuple(sorted(rejection_categories)),
            )

        # parse_valid is the pipeline's RELATIVE verdict (Step A2): the
        # battery accepted this output, so it introduces no new structural
        # breakage — a preserved pre-existing defect does not make it
        # parse-invalid. Tokens/latency sum EVERY consumed attempt.
        return ChunkedMergeResult(
            merged_code=merged,
            parse_valid=True,
            chunks_used=1,
            chunk_regions=[(1, total_lines)],
            model_tokens=whole_tokens,
            latency_ms=whole_latency,
            chunks_rejected=0,
            retries=attempts - 1,
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
    total_retries = 0
    rejected_chunks = 0
    chunk_regions = []
    all_rejection_categories: set[str] = set()

    for rev_pos, chunk in enumerate(reversed(chunks)):
        chunk_pos = len(chunks) - 1 - rev_pos  # index into `chunks` order
        start_idx = chunk.start_line - 1  # 0-indexed
        end_idx = chunk.end_line           # exclusive

        # B33: a fresh nonce per chunk attempt — the chunk's escaped text is
        # the ONLY text carrying this nonce, so even a user file containing
        # placeholder-looking strings cannot collide with it.
        tag_nonce = _new_tag_nonce()
        raw_chunk = "".join(original_lines[start_idx:end_idx])
        # Issue #13 (gate placement): the same pre-escape bound as the
        # whole-file path — the original chunk's length lower-bounds the
        # escaped prompt's, so refuse before paying the escape scan/copy.
        if len(raw_chunk) > _MAX_MERGE_PROMPT_CHARS:
            raise ValueError(
                f"chunk {chunk.start_line}-{chunk.end_line} too large for a "
                f"merge: {len(raw_chunk):,} characters exceeds the "
                f"{_MAX_MERGE_PROMPT_CHARS:,}-character model budget (the "
                f"model would truncate it and every attempt would be "
                f"rejected). Narrow the target: `replace=`/`after=` a "
                f"sub-symbol, or anchor the snippet closer to the edited "
                f"lines."
            )
        escaped_chunk = _escape_tags(raw_chunk, tag_nonce)

        # Issue #13: the chunk prompt must fit the model's re-emit envelope.
        # Window chunks are line-bounded but not BYTE-bounded (megabyte-long
        # lines), and a replace= chunk is a whole symbol. The size problem
        # used to surface only AFTER the model had been called — at the
        # issue #12 LCS DP gate — so the model burned its attempts first.
        # Fail loud here, before merge_fn, with the remedy.
        if len(escaped_chunk) > _MAX_MERGE_PROMPT_CHARS:
            raise ValueError(
                f"chunk {chunk.start_line}-{chunk.end_line} too large for a "
                f"merge: {len(escaped_chunk):,} characters exceeds the "
                f"{_MAX_MERGE_PROMPT_CHARS:,}-character model budget (the "
                f"model would truncate it and every attempt would be "
                f"rejected). Narrow the target: `replace=`/`after=` a "
                f"sub-symbol, or anchor the snippet closer to the edited "
                f"lines."
            )

        # Use the appropriate snippet portion for this chunk. Step D2:
        # text-anchor windows each get the snippet PORTION scoped to their
        # anchors (computed once, above, in `chunks` order).
        if text_window_snippets is not None:
            chunk_snippet = text_window_snippets[chunk_pos]
        elif len(chunks) > 1 and "<imports>" in chunk.matched_nodes:
            chunk_snippet = import_snippet
        elif len(chunks) > 1:
            chunk_snippet = code_snippet
        else:
            chunk_snippet = snippet

        # C3 seams stress: the declared NEW lines are computed from the
        # ALIGNED snippet — the exact bytes the model is told to insert —
        # and handed to _realign_output as protected lines, so the chunk
        # repair's uniform shift restores the model's drifted copy of the
        # CHUNK without re-indenting the model's verbatim echo of the
        # declared lines (aligning before escaping is byte-identical: the
        # tag placeholders never change a line's leading whitespace).
        #
        # Step D4: text-anchor window snippets are NOT re-aligned. Their
        # anchors were matched against the window's own lines and their
        # declared payload lines carry the file's true indentation — a
        # uniform shift to the window's base indent (a prose paragraph at
        # column 0) would distort a fenced code block's deeper indentation
        # and move the declared payload off its golden bytes.
        if is_text_anchor_edit:
            aligned_snippet = chunk_snippet
        else:
            aligned_snippet = _align_snippet_indent(chunk_snippet, raw_chunk)
        protected_new_lines = _declared_new_raw_lines(aligned_snippet, raw_chunk)

        safe_chunk_snippet = _escape_tags(aligned_snippet, tag_nonce)

        # Step A2 unified retry-until-valid loop (req. 5 + req. 9) — the
        # SAME loop the whole-file path runs, replacing the old fixed
        # structure (initial call + one truncation/parse retry + one
        # anchor-score retry + the <0.5 rejection): every attempt runs the
        # shared battery (relative parse + content faithfulness; see
        # _merge_rejection_reason), a rejected attempt retries with its
        # failure reason appended to the prompt, and only exhaustion
        # rejects the chunk (existing bookkeeping, original kept). A
        # truncated result is error-shaped (B12): it can never pass the
        # battery, so it can never be spliced.
        prompt_snippet = safe_chunk_snippet
        result = None
        merged_chunk_code: str | None = None
        attempts = 0
        chunk_tokens = 0
        chunk_latency = 0.0
        reason: str | None = None
        prev_reason: str | None = None
        chunk_rejection_categories: set[str] = set()
        for attempt in range(max_retries + 1):
            result = merge_fn(escaped_chunk, prompt_snippet, language)
            attempts += 1
            chunk_tokens += result.tokens_generated
            chunk_latency += result.latency_ms
            candidate = _unescape_tags(result.merged_code, tag_nonce)
            if not getattr(result, "truncated", False):
                # Re-align BEFORE validating: the battery must judge the
                # exact bytes a splice would write (Step 14 doctrine). The
                # snippet's declared new lines are protected (C3 seams
                # stress): the repair restores the model's drifted copy of
                # the CHUNK, never the model's verbatim echo of the bytes
                # the aligned snippet told it to insert.
                candidate = _realign_output(
                    candidate, raw_chunk, protected_lines=protected_new_lines,
                )
            # The OP SPEC is the ALIGNED snippet — the exact bytes the
            # model was told to insert (C3's protected lines come from the
            # same alignment). Classification is indent-invariant, so the
            # content gate sees identical tokens either way; the D1 trait
            # derivation, however, must measure the payload the model
            # actually echoes, indent included.
            reason = _merge_rejection_reason(
                raw_chunk, candidate, aligned_snippet, language,
                getattr(result, "truncated", False),
                fmt=attribute_format,
            )
            if reason is None:
                merged_chunk_code = candidate
                break
            _log.warning(
                "Chunk %d-%d attempt %d/%d rejected: %s",
                chunk.start_line, chunk.end_line,
                attempt + 1, max_retries + 1, reason,
            )
            # Issue #19 (defect 2) + issue #24 (defect B): same CATEGORY
            # twice in a row = a hopeless shape — the issue's oscillation
            # alternates the faithfulness STRING but stays one category, so
            # the string rule was defeated by the oscillation (the
            # whole-file twin above carries the rationale). Cycling
            # categories (parse ↔ faithfulness) still exhaust the budget.
            chunk_category = _rejection_category(reason)
            chunk_rejection_categories.add(chunk_category)
            if chunk_category == prev_reason:
                _log.warning(
                    "Chunk %d-%d rejected with the same category %d "
                    "attempt(s) in a row (%s: %s) — stopping retries early",
                    chunk.start_line, chunk.end_line, 2,
                    chunk_category, reason,
                )
                break
            prev_reason = chunk_category
            prompt_snippet = _append_corrective_note(safe_chunk_snippet, reason)

        total_tokens += chunk_tokens
        total_latency += chunk_latency
        total_retries += attempts - 1
        all_rejection_categories.update(chunk_rejection_categories)

        if merged_chunk_code is None:
            # Budget exhausted: keep the original chunk unchanged rather
            # than writing corrupted code to disk (rejection bookkeeping
            # unchanged; every consumed attempt's tokens are accounted).
            _log.error(
                "Chunk %d-%d rejected after %d attempt(s) (%s) — keeping "
                "the original chunk",
                chunk.start_line, chunk.end_line, attempts, reason,
            )
            rejected_chunks += 1
            chunk_regions.append((chunk.start_line, chunk.end_line))
            continue

        # A mid-file chunk whose last line lacks a terminator would
        # concatenate onto the following original line at splice time, so
        # the chunk is ALWAYS well-formed here; WHICH ending it carries —
        # and whether the FILE ends with a terminator at all — is decided by
        # _normalize_merged_eol at the return below (B31: the file's
        # trailing-newline state comes from the original, never from this
        # separator).
        merged_chunk_lines = merged_chunk_code.splitlines(keepends=True)
        if merged_chunk_lines and not merged_chunk_lines[-1].endswith(("\n", "\r")):
            merged_chunk_lines[-1] += line_ending

        result_lines[start_idx:end_idx] = merged_chunk_lines
        chunk_regions.append((chunk.start_line, chunk.end_line))

    merged_code = "".join(result_lines)
    # Step 14 (B19/B31): the assembled file is funnelled through the central
    # normalizer — produced chunks carrying the model's endings are converted
    # to the original's convention, and the file's trailing-newline state is
    # the original's.
    merged_code = _normalize_merged_eol(merged_code, original_code)

    # Step A2: final-assembly gate — RELATIVE (req. 9). Each chunk already
    # passed the battery against its own text; this gate checks the ASSEMBLED
    # file against the ORIGINAL file's diagnostics: an untouched pre-existing
    # defect is a preserved trait (parse_valid stays True), while new
    # breakage introduced anywhere by the assembly is a regression.
    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import parse_diagnostics
        parse_valid, _rel_reason = merged_is_acceptable(
            parse_diagnostics(original_code, language),
            parse_diagnostics(merged_code, language),
        )

    # Step D2: structureless final-assembly TRAIT gate (req. 6 + 9). The
    # parse rule above cannot run without a grammar, and the span-local
    # batteries each saw only their own window — so the ASSEMBLED file is
    # checked here against the op the FULL snippet declares, with the
    # EXACT whole-file arithmetic the D1 gate documented
    # (_derived_text_op on the full original + full snippet: expected =
    # original traits transformed by the declared op, the tolerance the
    # model-prose preset, the op-side layout/removal uncertainty the
    # snippet's own shape carries). This is the gate that catches
    # assembly-level corruption the per-chunk batteries cannot see — a
    # double-spliced chunk, a dropped or duplicated window — while a
    # faithful assembly (byte-exact reverse-order splices of validated
    # windows) passes by construction.
    #
    # A merge with ANY rejected window is refused outright, before the
    # trait arithmetic even runs: the trait view is the only assembly-
    # level view a structureless file has, and its tolerance policy
    # (rightly — it must absorb model seam noise) cannot certify that a
    # missing window's declared payload is the ONLY thing missing. The
    # D1 whole-file convention is therefore extended here: the original
    # file is kept as merged_code (never a silently partial text edit),
    # parse_valid is forced False (the universal "do not persist this"
    # signal), and every chunk is reported rejected so the existing
    # MCP/CLI gates refuse the write naturally.
    if _is_structureless_language(language):
        if rejected_chunks:
            _log.error(
                "%d/%d window(s) rejected — the structureless assembly gate "
                "refuses a partial text edit; keeping the original file",
                rejected_chunks, len(chunks),
            )
            return ChunkedMergeResult(
                merged_code=original_code,
                parse_valid=False,
                chunks_used=len(chunks),
                chunk_regions=list(reversed(chunk_regions)),
                model_tokens=total_tokens,
                latency_ms=total_latency,
                chunks_rejected=len(chunks),
                retries=total_retries,
                rejection_categories=tuple(sorted(all_rejection_categories)),
            )
        op, layout_slack, removable = _derived_text_op(original_code, snippet)
        ok, trait_reason = validate_text_output(
            original_code, op, merged_code,
            tolerance=TOLERANCE_MODEL_PROSE,
            layout_slack=layout_slack,
            removable_traits=removable,
        )
        if not ok:
            _log.error(
                "Final-assembly text-trait check failed (%s) — keeping the "
                "original file", trait_reason,
            )
            return ChunkedMergeResult(
                merged_code=original_code,
                parse_valid=False,
                chunks_used=len(chunks),
                chunk_regions=list(reversed(chunk_regions)),
                model_tokens=total_tokens,
                latency_ms=total_latency,
                chunks_rejected=len(chunks),
                retries=total_retries,
                rejection_categories=tuple(sorted(
                    all_rejection_categories | {"trait"},
                )),
            )

    # Step D3: final-assembly ATTRIBUTE/STRUCTURE gate (req. 7 + 9) — the
    # EXACT whole-file check the span-local batteries cannot perform: each
    # window was validated against its own text (whose scans cut mid-
    # structure by design), so only the assembled file can verify that the
    # fence/frontmatter STATE and every language attribute survived the
    # splices exactly as the FULL snippet declared. The refusal convention
    # mirrors the structureless assembly gate above: the original file is
    # kept, parse_valid is forced False, and every chunk is reported
    # rejected so the existing MCP/CLI gates refuse the write naturally.
    if attribute_format is not None:
        assembly_reason = _attribute_rejection_reason(
            original_code, merged_code, snippet, attribute_format,
        )
        if assembly_reason is not None:
            _log.error(
                "Final-assembly %s — keeping the original file",
                assembly_reason,
            )
            return ChunkedMergeResult(
                merged_code=original_code,
                parse_valid=False,
                chunks_used=len(chunks),
                chunk_regions=list(reversed(chunk_regions)),
                model_tokens=total_tokens,
                latency_ms=total_latency,
                chunks_rejected=len(chunks),
                retries=total_retries,
                rejection_categories=tuple(sorted(
                    all_rejection_categories | {"attribute"},
                )),
            )

    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=len(chunks),
        chunk_regions=list(reversed(chunk_regions)),
        model_tokens=total_tokens,
        latency_ms=total_latency,
        chunks_rejected=rejected_chunks,
        retries=total_retries,
        # Issue #24 (defect C): when the run fully lands, rejected_pieces
        # carry their own categories; an all-rejected run surfaces them so
        # the refusal can branch its advice. A clean run stays empty.
        rejection_categories=(
            tuple(sorted(all_rejection_categories))
            if rejected_chunks else ()
        ),
    )
