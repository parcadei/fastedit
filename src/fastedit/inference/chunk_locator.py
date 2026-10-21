"""Chunk location: finds which regions of a file to extract and merge.

Given a snippet and original file, uses AST analysis to determine the
minimal chunk(s) that need to be sent to the model.

Step D2 adds the AST-less branch: when no grammar resolves (``.txt``,
``.log``, any language the resolver honestly cannot parse), the snippet's
UNIQUE context lines act as anchors and each anchor yields a chunk window
sized to the measured model context budget (:data:`_MAX_TEXT_CHUNK_LINES`).
A structureless file of ANY size becomes editable; with no anchor the
whole-file chunk (and its fail-loud >150-line gate) is kept — nothing
declares where the edit goes, so the only safe behavior is to refuse.
"""

from __future__ import annotations

from ..split_join import normalize_bare_cr_for_ast
from .ast_utils import (
    ASTNode,
    ChunkRegion,
    _resolve_symbol,
    get_ast_map_from_source,
)
from .markers import is_marker_line
from .snippet_analysis import (
    _find_import_region,
    _find_insertion_region,
    _find_matching_nodes,
    _has_import_changes,
    _merge_overlapping_regions,
    _snippet_defines_new_symbols,
)

_MAX_BLOCK_LINES = 100  # Narrow functions larger than this to sub-blocks

# ---------------------------------------------------------------------------
# Step D2: AST-less text anchor chunking
# ---------------------------------------------------------------------------

_MAX_TEXT_CHUNK_LINES = 40
"""The per-chunk line budget for AST-less text windows.

Token-budget derivation (NO tokenizer dependency — a MEASURED estimate
from the real fastedit mlx-8bit model, tests/test_real_llm_text_chunks.py
and the Step A1 probe): the model has a 40 960-token context and A1
measured a ~150-line prose chunk at ~1 713 prompt tokens, so context
headroom is not the binding constraint — the model's GENERATION envelope
is. Measured: a ~400-line window fills the 16 384-token output cap
without converging (truncated on every attempt); ~120-line windows apply
the edit but lossily drop or reinvent untouched lines (content-
faithfulness rejections on every attempt); the measured converging
envelope for prose merges is the D1 whole-file shape (tests/
test_real_llm_text.py: a 37-line natural-prose file merges byte-exact).
40 lines keeps an entry's paragraph plus its neighbours inside that
envelope (~460 prompt tokens — ~90x context headroom), while larger
edits simply get more windows. A tokenizer-dependent budget would add a
model-loading cost to every locate for no measured benefit; the window's
bytes are re-checked span-locally by the validation battery anyway.

The budget bounds the CONTEXT around an anchor cluster, not the cluster
itself: a merged window's own anchor span (the lines between the
snippet's anchors, which marker-bearing gaps may hide content in) is the
floor — see :func:`_text_anchor_windows`. Two further measured shape
constraints surfaced by the real-model probes (documented in
tests/test_real_llm_text_chunks.py): the window the model re-emits must
not repeat a line within itself (its edges are therefore trimmed to
content lines — :func:`_trim_window_to_content` — and the corpus's text
dialect draws stride-disjoint body sentences), and prose merges converge
for the marker-free insertion shapes while the marker-bearing replace
idiom echoes the marker at every window size (a model limitation the
battery rightly rejects — the tests pin the insertion shapes).
"""

_TEXT_WINDOW_CONTEXT = (_MAX_TEXT_CHUNK_LINES - 1) // 2
"""Context lines on each side of a single anchor: 199, so one anchor's
window is exactly :data:`_MAX_TEXT_CHUNK_LINES` - 1 lines before
clamping to the file bounds."""

_MIN_TEXT_ANCHOR_LENGTH = 4
"""Minimum stripped length of a snippet line that may anchor a text
window — mirrors ``text_match._MIN_ANCHOR_LENGTH`` (the existing
anchor-quality doctrine) so the two matchers cannot disagree about what
makes a line too short/common to locate anything."""

_TEXT_ANCHOR_TAG = "<text anchor>"
"""The ``ChunkRegion.matched_nodes`` sentinel marking a D2 text-anchor
window. ``chunked_merge`` reads it to treat such regions as WINDOW
chunks even when a window happens to span the whole file: the snippet's
anchor declared where the edit goes, which is exactly the information
the >150-line whole-file gate exists to demand. The tag joins the
existing declarative region vocabulary (``<whole file>``,
``<unmatched>``, ``<imports>``)."""


def _snippet_has_ellipsis(snippet: str) -> bool:
    """True when the snippet carries a preservation-marker ellipsis.

    The exact predicate :func:`locate_chunks` uses to decide the
    ``replace=`` narrow; shared with chunked_merge's auto-prepend guard so
    the two sites can never disagree about when a narrow will engage.
    """
    return "... existing code ..." in snippet or "# ..." in snippet


def _text_snippet_anchor_lines(
    snippet: str,
    original_lines: list[str],
) -> list[tuple[int, int]]:
    """Match the snippet's context lines to UNIQUE original lines.

    Returns ``(snippet_line_index, original_line_number)`` pairs (the line
    number 1-indexed) for every snippet line that names exactly one
    original line — the anchors a text window can be built around.

    This is deliberately a SIMPLER exact/normalized matcher than the
    battery's snippet classifier (``chunked_merge._classify_snippet``),
    not a second copy of it: the classifier decides context-vs-new for
    MERGE VALIDATION (forward-scanning, cursor-consuming, indent-aware);
    the matcher here only answers "which snippet lines pinpoint a
    location", and it does so with the two properties windowing needs:

      * **uniqueness** — the stripped line occurs exactly once in the
        original. A line with several occurrences cannot locate a window
        (which occurrence would the edit target?), and picking one would
        be the duplicated-paragraph trap (the GIGO corpus keeps
        pre-existing duplicates; a wrong-occurrence window would hand the
        model the wrong copy). Uniqueness makes the whole class of
        wrong-occurrence cuts impossible instead of heuristic.
      * **forward-scan order** — the anchors' original positions are
        strictly increasing, the SAME binding rule the battery's
        classifier uses (``_classify_snippet_raw``'s cursor). An anchor
        the battery could never bind as context would classify as a NEW
        line and make every merge of the window unfaithful, so it must
        not anchor a window either: one binding logic, two consumers.

    Blank lines, marker lines (``is_marker_line`` — the shared B15
    predicate) and lines shorter than :data:`_MIN_TEXT_ANCHOR_LENGTH` are
    skipped: they carry no locatable content.

    Cost: one O(lines) pass to index the original (first occurrence +
    duplicate set), then O(1) per snippet line — linear in the file, no
    parsing, no grammar.
    """
    first_index: dict[str, int] = {}
    duplicated: set[str] = set()
    for idx, line in enumerate(original_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in first_index:
            duplicated.add(stripped)
        else:
            first_index[stripped] = idx

    anchors: list[tuple[int, int]] = []
    cursor = 0
    for si, raw in enumerate(snippet.splitlines()):
        stripped = raw.strip()
        if not stripped or is_marker_line(raw):
            continue
        if len(stripped) < _MIN_TEXT_ANCHOR_LENGTH:
            continue
        if stripped in duplicated:
            continue
        idx = first_index.get(stripped)
        if idx is None or idx < cursor:
            continue
        anchors.append((si, idx + 1))  # 1-indexed original line
        cursor = idx + 1
    return anchors


def _text_anchor_windows(
    snippet: str,
    original_lines: list[str],
) -> list[tuple[int, int]]:
    """Window regions around the snippet's unique text anchors (Step D2).

    Each anchor yields ``anchor ± :data:`_TEXT_WINDOW_CONTEXT`` clamped to
    the file; the windows are then merged with the shared
    :func:`_merge_overlapping_regions` (multiple anchors referencing
    several regions become several chunks), and every merged region is
    re-fitted around ITS anchors' span so the
    :data:`_MAX_TEXT_CHUNK_LINES` budget bounds the context, not the
    anchors themselves: a cluster of nearby anchors merges into one
    window that still covers every anchor it owns (and any marker-hidden
    gap between them) while trimming the surrounding context to the
    budget. The cluster's own span is the floor — when the snippet's
    anchors are farther apart than the budget, the window is that span
    plus whatever context fits, never a cut through the declared edit.

    Each fitted window's edges are then pulled in onto CONTENT lines
    (:func:`_trim_window_to_content` — a blank paragraph separator at a
    window edge is the one line the model measurably trims from its
    re-emitted chunk, corrupting the seam).

    Returns a list of ``(start_line, end_line)`` 1-indexed inclusive
    regions, empty when the snippet provides no anchor.
    """
    anchors = _text_snippet_anchor_lines(snippet, original_lines)
    if not anchors:
        return []
    total_lines = len(original_lines)
    windows = [
        (max(1, anchor - _TEXT_WINDOW_CONTEXT),
         min(total_lines, anchor + _TEXT_WINDOW_CONTEXT))
        for _si, anchor in anchors
    ]
    merged = _merge_overlapping_regions(windows)
    fitted: list[tuple[int, int]] = []
    for start, end in merged:
        cluster = [a for _si, a in anchors if start <= a <= end]
        lo, hi = min(cluster), max(cluster)
        fit_ctx = max(0, (_MAX_TEXT_CHUNK_LINES - (hi - lo + 1)) // 2)
        fitted.append(_trim_window_to_content(
            max(1, lo - fit_ctx), min(total_lines, hi + fit_ctx),
            original_lines,
        ))
    return fitted


def _trim_window_to_content(
    start: int,
    end: int,
    original_lines: list[str],
) -> tuple[int, int]:
    """Pull a fitted window's edges in onto CONTENT lines (Step D2 fix).

    Measured (tests/test_real_llm_text_chunks.py, deterministic greedy
    probes): when the fitted window's first or last line is a BLANK
    paragraph separator, the model trims that edge blank from its re-emitted
    chunk — the battery's content view cannot see blanks and its ±1 layout
    floor absorbs the drift, so the seam corruption flowed through to the
    file and broke byte-exactness (a paragraph separator vanished at the
    window edge on every attempt). A window that STARTS and ENDS on content
    lines removes the edge the model was trimming; the anchors are content
    lines, so the trim can never cut one.

    The window can only shrink, and only past blanks at its edges; a fully
    blank window is impossible (every window contains an anchor).
    """
    while start <= end and not original_lines[start - 1].strip():
        start += 1
    while end >= start and not original_lines[end - 1].strip():
        end -= 1
    return (start, end)


def _text_window_snippets(
    snippet: str,
    windows: list[tuple[int, int]],
    original_lines: list[str],
) -> list[str]:
    """Scope the snippet to each window (one snippet portion per chunk).

    A multi-anchor snippet must not be handed to every window whole: an
    anchor living in ANOTHER window is not in this chunk, so the battery's
    classifier would bind it as a NEW line and demand the model insert a
    duplicate of it. Each snippet line therefore rides with the window of
    the nearest PRECEDING anchor (markers, blanks and declared new lines
    belong to the anchor they follow — the same per-anchor grouping the
    battery's segment builder uses); lines before the first anchor ride
    with the first anchor's window (a top-of-file insertion declared
    before the first anchor is that window's leading segment). Windows
    keep the region order :func:`_text_anchor_windows` produced.
    """
    anchors = _text_snippet_anchor_lines(snippet, original_lines)
    anchor_window: dict[int, int] = {}
    for si, anchor in anchors:
        for w_idx, (start, end) in enumerate(windows):
            if start <= anchor <= end:
                anchor_window[si] = w_idx
                break
    current = anchor_window.get(anchors[0][0], 0) if anchors else 0
    portions: list[list[str]] = [[] for _ in windows]
    for si, raw in enumerate(snippet.splitlines(keepends=True)):
        if si in anchor_window:
            current = anchor_window[si]
        portions[min(current, len(windows) - 1)].append(raw)
    return ["".join(part) for part in portions]


def _narrow_will_engage(snippet: str, node_line_count: int) -> bool:
    """True when ``locate_chunks`` will narrow this ``replace=`` target.

    Mirrors the locator's condition (node larger than ``_MAX_BLOCK_LINES``
    plus an ellipsis marker): the chunk the model will see is a SUB-BLOCK
    of the symbol, not the symbol itself.
    """
    return node_line_count > _MAX_BLOCK_LINES and _snippet_has_ellipsis(snippet)

# B27: minimum sliding-window content score for trusting a narrow. The old
# 0.2 accepted windows where four of five lines were NOT in the original —
# the window locked onto spurious stripped-line matches and was then cut
# ±padding RAW lines around it, handing the model a mid-block fragment to
# "repair". 0.6 means the snippet's leading window lines must mostly exist,
# contiguously, in the node before the narrow is trusted. Tunable: lower it
# and sloppy snippets narrow again (mid-block cuts return); raise it and
# large-node edits fall through to the model path more often.
_MIN_NARROW_SCORE = 0.6

# tree-sitter node types that represent block structures
_BLOCK_TYPES = {
    "for_statement", "while_statement", "if_statement",
    "with_statement", "try_statement", "match_statement",
    # JS/TS/Go/Rust/Java/etc.
    "for_in_statement", "for_of_statement", "switch_statement",
    "do_statement", "loop_expression", "match_expression",
}


def locate_chunks(
    snippet: str,
    original_code: str,
    file_path: str,
    padding: int = 30,
    language: str | None = None,
    after: str | None = None,
    replace: str | None = None,
) -> list[ChunkRegion]:
    """Locate the chunk region(s) in the original file that the snippet edits.

    Language-agnostic: uses tldr AST analysis (16 languages) with regex fallback.
    Handles single-site edits, multi-site edits, import changes, and
    targeted insertion via the ``after`` parameter. When NO grammar
    resolves (Step D2), the snippet's unique context lines anchor window
    chunks sized to the measured model context budget
    (:data:`_MAX_TEXT_CHUNK_LINES`) — see :func:`_text_anchor_windows`;
    with no anchor the whole file is one chunk and the caller's
    whole-file gate applies.

    Args:
        snippet: The edit snippet (with or without markers).
        original_code: The full original file content.
        file_path: Path to the file (for tldr AST extraction).
        padding: Lines of context to include above and below the edit region.
        language: Optional language hint (e.g. "python", "typescript").
        after: Optional symbol name — insert new code after this function/class.
            Uses tldr to find the symbol's line range and creates a tight chunk.
        replace: Optional symbol name — replace this function/class/method entirely
            with the snippet. Uses AST to find the symbol's exact line range.

    Returns:
        List of ChunkRegion(s) to extract and merge.
    """
    original_lines = original_code.splitlines()
    total_lines = len(original_lines)

    # B26: parse the IN-MEMORY `original_code`, not the disk-cached tldr map.
    # get_ast_map consults the tldr daemon, whose salsa cache can hold
    # pre-write line numbers for a file that was just rewritten — chunks
    # spliced from stale coordinates land in the wrong region. The fast
    # paths in chunked_merge already parse `original_code` in-memory via
    # get_ast_map_from_source; locate_chunks must agree with them. A bare CR
    # is swapped for LF by a same-length, same-position substitution first
    # (tree-sitter counts rows by scanning for "\n"), keeping every returned
    # line number valid against original_lines — the same protection
    # get_ast_map's LF-normalized temp-file path used to provide. B3: the
    # caller's `language` hint rides along for extension-unwired languages.
    ast_nodes = get_ast_map_from_source(
        normalize_bare_cr_for_ast(original_code), file_path, language,
    )

    if not ast_nodes:
        # Step D2: no grammar resolved — the file is AST-less (``.txt``,
        # ``.log``, any unresolvable language). The snippet's UNIQUE
        # context lines are anchors: each yields a window chunk sized to
        # the measured model context budget, so a structureless file of
        # ANY size is editable and the model still sees only a slice.
        # Multiple anchors produce multiple windows (merged per the
        # shared ``_merge_overlapping_regions``). With NO anchor nothing
        # declares where the edit goes — no window can be trusted and
        # only the whole-file chunk remains, whose >150-line gate
        # (chunked_merge) keeps failing loud for exactly this case.
        windows = _text_anchor_windows(snippet, original_lines)
        if windows:
            return [
                ChunkRegion(start, end, [_TEXT_ANCHOR_TAG])
                for start, end in windows
            ]
        return [ChunkRegion(1, total_lines, ["<whole file>"])]

    # If `replace` is specified, find the named symbol's exact line range
    # and use that as the chunk region.  Supports 'Class.method' to
    # disambiguate duplicate names (e.g. two __init__ methods).
    if replace:
        target_node = _resolve_symbol(replace, ast_nodes)
        if target_node:
            # If the function is large and the snippet uses ellipsis markers,
            # narrow to just the edited sub-region within the function.
            node_size = target_node.line_end - target_node.line_start + 1
            has_ellipsis = _snippet_has_ellipsis(snippet)
            if node_size > _MAX_BLOCK_LINES and has_ellipsis:
                narrowed = _narrow_large_node(
                    target_node, snippet, original_lines,
                    original_code=original_code, language=language,
                    max_lines=_MAX_BLOCK_LINES, padding=padding,
                )
                return [ChunkRegion(
                    narrowed[0], narrowed[1],
                    [replace],
                )]
            return [ChunkRegion(
                target_node.line_start, target_node.line_end,
                [replace],
            )]

    # If `after` is specified, find the named symbol and the next one,
    # create a chunk spanning the gap between them.  Supports 'Class.method'.
    if after:
        anchor_node = _resolve_symbol(after, ast_nodes)
        if anchor_node:
            # Find the next AST node after the anchor
            next_node = None
            for node in ast_nodes:
                if node.line_start > anchor_node.line_end and (next_node is None or node.line_start < next_node.line_start):
                    next_node = node
            if next_node:
                return [ChunkRegion(
                    anchor_node.line_start, next_node.line_end,
                    [after, next_node.name],
                )]
            else:
                # anchor is the last symbol — chunk from it to EOF
                return [ChunkRegion(
                    anchor_node.line_start, total_lines,
                    [after],
                )]

    # Find which AST nodes the snippet modifies
    matched_nodes = _find_matching_nodes(snippet, original_lines, ast_nodes, language)

    # Check for import changes
    import_region = None
    if _has_import_changes(snippet, original_code, language):
        import_region = _find_import_region(original_code, language, ast_nodes)

    # Check for new code insertion (snippet has definitions not in file).
    # Step D4 defect fix: the decision is made in the FILE's own symbol
    # vocabulary (:func:`_snippet_defines_new_symbols`) — the old
    # language-blind regex names fabricated phantom "new definitions" for
    # document formats (a def line inside a markdown fenced block) and
    # routed the edit to a tail insertion region that does not contain
    # the edit target.
    has_new_defs = _snippet_defines_new_symbols(
        snippet, language, {n.name for n in ast_nodes},
    )

    insertion = None
    if has_new_defs:
        insertion = _find_insertion_region(
            snippet, original_lines, ast_nodes, total_lines, language,
        )

    # If nothing matched at all, fall back to whole file
    if not matched_nodes and not import_region and not insertion:
        return [ChunkRegion(1, total_lines, ["<unmatched>"])]

    # Build regions from matches
    raw_regions: list[tuple[int, int]] = []
    region_names: dict[tuple[int, int], list[str]] = {}

    # Import region (if detected)
    if import_region:
        raw_regions.append(import_region)
        region_names[import_region] = ["<imports>"]

    # Insertion region for new definitions
    if insertion:
        ins_region = (insertion.start_line, insertion.end_line)
        raw_regions.append(ins_region)
        region_names[ins_region] = insertion.matched_nodes

    # Function/class regions — snap to enclosing class boundaries,
    # then narrow large nodes to sub-blocks
    for node in matched_nodes:
        parent = _find_enclosing_parent(node, ast_nodes)
        if parent:
            region = (parent.line_start, parent.line_end)
        else:
            region = _narrow_large_node(
                node, snippet, original_lines,
                original_code=original_code, language=language,
            )

        raw_regions.append(region)
        region_names.setdefault(region, []).append(node.name)

    # Merge overlapping/close regions
    merged_regions = _merge_overlapping_regions(raw_regions)

    chunks = []
    for start, end in merged_regions:
        # Collect names from all raw regions within this merged region
        names: list[str] = []
        for raw_region, raw_names in region_names.items():
            if raw_region[0] >= start and raw_region[1] <= end:
                names.extend(raw_names)
        chunks.append(ChunkRegion(start, end, names))

    return chunks


def _find_enclosing_block(
    source: str,
    language: str,
    target_line: int,
    function_start: int,
    function_end: int,
) -> tuple[int, int] | None:
    """Use tree-sitter to find the smallest block containing target_line.

    Walks the AST to find the tightest for/if/while/try/with block that
    encloses the target line within the given function range.

    Returns (start_line, end_line) 1-indexed, or None if no block found.
    """
    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: parse of arbitrary source (unknown language/malformed) means "no enclosing block", never propagates
        return None

    # Collect ALL enclosing blocks, pick the best-sized one
    candidates: list[tuple[int, int]] = []

    def walk(node):
        node_start = node.start_point[0] + 1  # 1-indexed
        node_end = node.end_point[0] + 1

        # Skip nodes outside the function
        if node_end < function_start or node_start > function_end:
            return
        # Skip nodes that don't contain the target line
        if node_start > target_line or node_end < target_line:
            return

        if node.type in _BLOCK_TYPES:
            candidates.append((node_start, node_end))

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not candidates:
        return None

    # Prefer blocks that are 30-100 lines. If none in that range,
    # pick the smallest block that's >= 20 lines.
    _MIN_BLOCK = 20
    _IDEAL_MIN = 30
    _IDEAL_MAX = 100

    ideal = [c for c in candidates if _IDEAL_MIN <= (c[1] - c[0]) <= _IDEAL_MAX]
    if ideal:
        return min(ideal, key=lambda c: c[1] - c[0])

    adequate = [c for c in candidates if (c[1] - c[0]) >= _MIN_BLOCK]
    if adequate:
        return min(adequate, key=lambda c: c[1] - c[0])

    # All blocks are tiny — return the largest one
    return max(candidates, key=lambda c: c[1] - c[0])


def _narrow_large_node(
    node: ASTNode,
    snippet: str,
    original_lines: list[str],
    original_code: str = "",
    language: str | None = None,
    max_lines: int = _MAX_BLOCK_LINES,
    padding: int = 10,
) -> tuple[int, int]:
    """Narrow a large function to the sub-block relevant to the snippet.

    Returns ``(start_line, end_line)`` (1-indexed, inclusive), always
    constrained to the node's own span: either the sub-block the snippet
    clearly targets, or the FULL node range when the narrow cannot be
    trusted (callers treat the full range as "send the whole node").

    B27 contract (Step 17): the narrow is trusted only when the snippet's
    non-marker lines match a contiguous window of the node with score
    ≥ ``_MIN_NARROW_SCORE`` AND tree-sitter can name the enclosing block
    around that window. The cut boundaries are then that block's AST node
    edges — never a ±``padding`` raw-line window, which cut mid-block and
    handed the model a broken fragment to "repair". Without a confident
    window match, without ``original_code``/``language`` to parse, or
    without an enclosing block, the narrow is REJECTED and the full node
    range is returned.

    ``padding`` is retained for API compatibility and is now a no-op:
    raw line-count padding was exactly the mid-block cut this function no
    longer performs (same deprecation pattern as ``deterministic_edit``'s
    ``max_drop_gap``).

    C3 fix (wrong-sub-block cut): the cut anchors on the FIRST MATCHED
    line of the best-scoring window, not on the window's first line. A
    snippet whose leading lines match nothing at the window position —
    concretely: ``replace=`` on a >100-line symbol where the auto-prepended
    signature (see ``chunked_merge``) occupies the window's first slot, or
    any snippet with leading declared-new lines — used to slide the window
    one-or-more lines past its true alignment and then anchor the
    enclosing-block lookup on a bystander line, cutting the WRONG
    sub-block of the large node (found by the C3 deep-nesting stress: the
    model was handed block A while the snippet edited block B, and every
    attempt was rejected to exhaustion). Anchoring on the first matched
    line is a no-op for snippets whose first line is a real context match
    (the whole-symbol wrap shapes).

    Issue #12 e2e follow-up (first-matched-line-has-no-block walk + matched
    span extension): the first matched line is not always INSIDE a block —
    the issue's real shape had it as a top-level statement of the method
    body (``ops.push('step-2489');`` at body level), with the snippet's
    later matched lines inside a small ``try`` block. Two defects followed:
    the old single-anchor lookup returned None and the narrow fell back to
    the FULL node — a 2,499-line / 66 KB chunk handed to a 1.7B model,
    whose speculative verification memory then scaled with the chunk and
    got the process OOM-killed mid-merge (the issue's exit-137,
    reproduced). Fix 1: walk the best window's matched lines FIRST onward
    and cut at the first one that sits inside a block; only when NO
    matched line has a block does the narrow stay rejected (full node,
    unchanged contract). Fix 2: the cut extends to cover every matched
    context line of the window (snippet-anchored, never raw ±padding), so
    a declared anchor above the block is inside the chunk instead of being
    misread by the battery as a declared-new line missing from the merge.
    """
    node_size = node.line_end - node.line_start + 1
    if node_size <= max_lines:
        return (node.line_start, node.line_end)

    # Step 1: Find which line the snippet targets via line matching
    node_lines = original_lines[node.line_start - 1:node.line_end]
    # Filter out ellipsis markers — they don't represent real code and
    # would tank the match score when most snippet lines are markers.
    # Uses the shared LINE-ANCHORED predicate (single source of truth,
    # B15): a marker phrase embedded mid-line is real code, not a marker.
    snippet_lines = [
        line.rstrip() for line in snippet.splitlines()
        if line.strip() and not is_marker_line(line)
    ]

    if not snippet_lines:
        return (node.line_start, node.line_end)

    # Sliding window to find best match position. The whole window's
    # matched-line set is kept (not just its first hit): the enclosing-block
    # lookup walks forward through these candidates (issue #12 follow-up).
    best_score = 0.0
    best_offset = 0
    window = min(len(snippet_lines), 10)

    for offset in range(len(node_lines) - window + 1):
        region = [line.rstrip() for line in node_lines[offset:offset + window]]
        hit_flags = [
            s.strip() == r.strip()
            for s, r in zip(snippet_lines[:window], region, strict=False)
        ]
        score = sum(hit_flags) / window
        if score > best_score:
            best_score = score
            best_offset = offset

    if best_score < _MIN_NARROW_SCORE:
        # B27: a window the snippet barely overlaps is not evidence for
        # ANY location — no confident narrow, keep the whole node.
        return (node.line_start, node.line_end)

    # Step 2: cut at the enclosing block's AST node edges (B27). Without
    # something parsed there is no node edge to cut on — a raw-line window
    # here is precisely the mid-block cut B27 removes.
    if not (original_code and language):
        return (node.line_start, node.line_end)

    # Collect the matched lines IN the best-aligned window, first onward:
    # (original line number, stripped content). The first is the C3 anchor;
    # the rest are the walk's candidates.
    window_region = [
        line.rstrip() for line in node_lines[best_offset:best_offset + window]
    ]
    matched_positions = [
        (node.line_start + best_offset + i, s.strip())
        for i, (s, r) in enumerate(zip(snippet_lines[:window], window_region, strict=False))
        if s.strip() == r.strip()
    ]
    if not matched_positions:
        return (node.line_start, node.line_end)

    block = None
    for candidate_line, _content in matched_positions:
        block = _find_enclosing_block(
            original_code, language, candidate_line,
            node.line_start, node.line_end,
        )
        if block:
            break
    if not block:
        # No matched line sits inside any block; no AST edge bounds a
        # smaller cut, so keep the whole node (the B27 contract).
        return (node.line_start, node.line_end)

    # Issue #12 follow-up, span extension: the cut must cover EVERY
    # matched context line of the best window, not just the block edges —
    # a context line above the block (the statement before the edited
    # try, say) is a declared snippet anchor; leaving it outside the
    # chunk makes the battery read it as a declared-new line missing from
    # the merge and rejects every attempt. Extending to the matched span
    # keeps the cut anchored on snippet-declared lines (never raw
    # ±padding), so the B27 mid-block-cut hazard does not return.
    start = max(node.line_start, min(matched_positions[0][0], block[0]))
    end = min(node.line_end, max(matched_positions[-1][0], block[1]))
    return (start, end)


def _find_enclosing_parent(
    node: ASTNode,
    ast_nodes: list[ASTNode],
) -> ASTNode | None:
    """Find the smallest enclosing class/interface that contains the node.

    For large classes (>150 lines), returns None to keep method edits at
    method granularity instead of snapping to the whole class.
    """
    best = None
    for candidate in ast_nodes:
        if candidate is node:
            continue
        if candidate.kind not in ("class", "interface"):
            continue
        if (candidate.line_start <= node.line_start
                and candidate.line_end >= node.line_end
                and (best is None or (candidate.line_end - candidate.line_start <
                                      best.line_end - best.line_start))):
            # Pick the smallest enclosing parent
            best = candidate
    # Don't snap to large classes — keep method edits at method granularity
    if best is not None and (best.line_end - best.line_start) > 150:
        return None
    return best
