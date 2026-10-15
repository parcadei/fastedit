"""Snippet parsing, matching, and import detection.

Analyzes edit snippets to find definition names, match them against AST nodes,
detect import changes, and locate insertion regions.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile

from ..io_utils import write_all
from .ast_utils import (
    ASTNode,
    ChunkRegion,
    _get_ast_via_extract,
    _get_ast_via_structure,
)

# Marker constants live in ONE module (B15 unification). Marker detection
# goes through ``markers.is_marker_line`` — the shared line-anchored
# predicate covering the legacy long forms AND the short forms (``#...``,
# ``//...``, ``…``) — so this module can never disagree with the rest of the
# pipeline about what a marker is (B17 remainder). ``_MARKER_RE`` itself is
# kept importable from here purely for backward compatibility
# (``chunked_merge`` re-exports it; tests/test_snippet_analysis.py imports
# it from this module); no merge-semantic decision may use it directly.
from .markers import _MARKER_RE, is_marker_line  # noqa: F401

# --- Language extension mapping for temp file parsing ---
_LANG_EXT = {
    "python": ".py", "typescript": ".ts", "javascript": ".js",
    "go": ".go", "rust": ".rs", "java": ".java", "c": ".c",
    "cpp": ".cpp", "ruby": ".rb", "php": ".php", "kotlin": ".kt",
    "swift": ".swift", "csharp": ".cs", "scala": ".scala",
    "elixir": ".ex", "lua": ".lua",
}

# --- Multi-language definition patterns (regex fallback) ---
_DEFINITION_PATTERNS = [
    # Python/Ruby/Scala/Elixir: def/defp/defmodule (Ruby: def self.method)
    re.compile(r"^\s*(?:async\s+)?(?:defp?\s+|defmodule\s+)(?:self\.)?(\w+)"),
    # JS/TS/PHP/Lua: function keyword
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)"),
    # Go/Swift/Kotlin: func/fun keyword (Go receiver methods too)
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?(?:func|fun)\s+(?:\([^)]*\)\s+)?(\w+)"),
    # Rust: fn keyword
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"),
    # Class-like: class/struct/enum/trait/interface/protocol/object/impl
    re.compile(
        r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:abstract\s+)?"
        r"(?:class|struct|enum|trait|interface|protocol|module|object|impl)\s+(\w+)"
    ),
]

# --- tree-sitter import node types per language ---
_TS_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement", "future_import_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "rust": {"use_declaration"},
    "go": {"import_declaration"},
    "java": {"import_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include", "using_declaration"},
    "swift": {"import_declaration"},
    "kotlin": {"import_header"},
    "c_sharp": {"using_directive"},
    "php": {"namespace_use_declaration"},
    "ruby": {"call"},       # require/require_relative — filtered by name
    "scala": {"import_declaration"},
    "elixir": {"call"},     # alias/import/use/require — filtered by name
    "lua": {"function_call"},  # require() — filtered by name
}
# Languages where "call" nodes need function-name filtering
_IMPORT_CALL_NAMES: dict[str, set[str]] = {
    "ruby": {"require", "require_relative"},
    "elixir": {"alias", "import", "use", "require"},
    "lua": {"require"},
}

# Marker line detection is defined in ``fastedit.inference.markers`` (single
# source of truth, B15) and imported at the top: ``is_marker_line`` drives
# every marker decision here; ``_MARKER_RE`` (long forms only) is kept
# importable for backward compatibility only.


# ---------------------------------------------------------------------------
# Snippet name extraction (language-agnostic)
# ---------------------------------------------------------------------------

def _extract_snippet_names(snippet: str, language: str | None = None) -> list[str]:
    """Extract function/class/method names from a snippet.

    Strategy:
    1. Try tldr structure on a temp file (most accurate, handles complex syntax)
    2. Fall back to multi-language regex patterns
    """
    if language:
        ext = _LANG_EXT.get(language)
        if ext:
            names = _try_tldr_snippet_parse(snippet, ext)
            if names:
                return names

    return _regex_extract_names(snippet)


def _strip_marker_lines(snippet: str) -> str:
    """The snippet without its preservation-marker lines (Step D4).

    B15: a marker line is a merge DIRECTIVE, never content — every other
    consumer in the pipeline excludes it from content decisions, and any
    new-definition judgment must too (in markdown the canonical
    ``# ... existing code ...`` marker even parses as a heading, so
    parsing a snippet with its marker lines still inside fabricates a
    section nobody declared).
    """
    return "".join(
        line for line in snippet.splitlines(keepends=True)
        if not is_marker_line(line)
    )


def _snippet_definition_names(
    snippet: str,
    language: str,
) -> list[str] | None:
    """The snippet's own definition names, in the FILE's symbol vocabulary.

    The file's grammar resolves the snippet (B1) and the SAME declarative
    B3 format spec that names the file's symbols names the snippet's
    (:func:`get_ast_map_from_source`) — one vocabulary on both sides of
    every name comparison, in process, no daemon, no temp file. Marker
    lines are excluded first (:func:`_strip_marker_lines`).

    Returns ``None`` when the file has NO grammar (the B1 resolver's
    honest answer): the caller keeps the legacy language-blind regex
    answer, which is all it ever had.
    """
    try:
        from ..data_gen.ast_analyzer import get_language

        get_language(language)  # the resolver's honest "no grammar" answer raises
    except Exception:  # noqa: BLE001 -- ANY resolution failure IS the "no grammar" answer
        return None
    try:
        from .ast_utils import get_ast_map_from_source

        nodes = get_ast_map_from_source(
            _strip_marker_lines(snippet),
            f"snippet{_LANG_EXT.get(language, '.txt')}",
            language,
        )
    except Exception:  # noqa: BLE001 -- a snippet the grammar cannot parse defines nothing
        return []
    return [n.name for n in nodes if n.name]


def _snippet_defines_new_symbols(
    snippet: str,
    language: str | None,
    existing_names: set[str],
) -> bool:
    """True when the snippet defines a symbol the file does not have —
    judged in the FILE's own symbol vocabulary (Step D4 defect fix).

    The decision this helper replaces compared two DIFFERENT vocabularies:
    the snippet's names came from the language-blind definition regex
    (:func:`_regex_extract_names` — def/fn/func/class shapes in ANY
    syntax) while ``existing_names`` are the file's own B3 format-spec
    symbols. For a document format whose symbols are NOT code definitions
    (a markdown file's symbols are its SECTIONS), a code-shaped line
    inside a fenced code block fabricated a phantom "new definition" —
    ``def summarize(users):`` inside a ```python fence became a "new
    function" ``summarize`` — the edit was routed to a tail "insertion
    region" that does not contain the edit target, and the model was
    handed the wrong bytes: the battery rejected every attempt (measured
    9/9 against the real model) or, for a compliant model echo, would
    have ratified an insertion into the wrong region.

    The fix is the one-vocabulary rule (CLAUDE.md — declarative, no
    per-format branches): when the file's grammar resolves, the snippet's
    definitions are read with the file's own format spec
    (:func:`_snippet_definition_names`); only grammar-less files keep the
    legacy regex answer (nothing better exists there, and the AST-less
    branch never reaches the insertion decision anyway).

    Accepted asymmetry, documented: a document-format snippet that
    genuinely adds a new section/element may now still fall through to
    the conservative whole-file region, because the legacy temp-file
    parser behind :func:`_find_insertion_region` only knows code
    definitions. Missing a tight chunk is safe; inventing a wrong-region
    one was not.
    """
    if language:
        names = _snippet_definition_names(snippet, language)
        if names is not None:
            return any(name not in existing_names for name in names)
    return any(
        name not in existing_names for name in _regex_extract_names(snippet)
    )


def _top_level_extras(
    snippet: str, language: str | None, target: str
) -> list[str]:
    """Return function/method names in the snippet other than `target`.

    Used by the `replace=X` guard to detect silent-deletion risk: nested
    methods inside a class snippet are extras (the original class may have
    other methods that would be silently dropped), but field/variable
    declarations are not — a field-only class snippet is a safe full-class
    swap with no deletion risk.

    Strategy: shell out to `tldr structure` on a temp file and filter
    `definitions` to kind ∈ {method, function}. tldr is already a fastedit
    prerequisite (used by fast_read, fast_search, etc.), so this introduces
    no new dep. Latency is ~10ms per call — negligible for a one-shot guard.

    Falls back to fastedit's in-process `analyze_file` if tldr is missing
    or misbehaves.
    """
    if not language:
        return [n for n in _extract_snippet_names(snippet, language) if n != target]

    ext = _LANG_EXT.get(language)
    if ext:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            write_all(fd, snippet.encode())
            os.close(fd)
            result = subprocess.run(
                ["tldr", "structure", tmp_path, "--format", "compact"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                files = data.get("files", [])
                if files:
                    defs = files[0].get("definitions", [])
                    return [
                        d["name"] for d in defs
                        if d.get("name") and d.get("name") != target
                        and d.get("kind") in ("method", "function")
                    ]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, FileNotFoundError):
            pass
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    try:
        from ..data_gen.ast_analyzer import analyze_file
        fs = analyze_file(snippet, language)
    except Exception:  # noqa: BLE001 -- deliberate: tree-sitter parse of arbitrary snippet input degrades to the regex extractor on any failure
        return [n for n in _extract_snippet_names(snippet, language) if n != target]
    extras: list[str] = []
    for fn in fs.functions:
        if fn.name and fn.name != target:
            extras.append(fn.name)
    for cls in fs.classes:
        if cls.name and cls.name != target:
            extras.append(cls.name)
    return extras


def _member_snippet_parse_names(snippet: str, file_name: str) -> list[str]:
    """The definition names ``snippet`` defines when read as a MEMBER.

    Issue #5: a member snippet (a class constructor, a method shorthand) is
    only grammatical INSIDE its container — the compilation-unit parse the
    bare helper runs sees an error node where the definition should be.
    Wrapping it in the language's minimal container (declared in
    ``_MEMBER_SNIPPET_CONTAINERS``) asks the grammar the question the
    snippet actually answers. Returns [] for languages without a declared
    container and for snippets that do not parse as a member either — the
    synthetic container itself is never reported.
    """
    try:
        from ..data_gen.ast_analyzer import detect_language
        from .ast_utils import (
            _MEMBER_SNIPPET_CONTAINER_NAME,
            _MEMBER_SNIPPET_CONTAINERS,
            get_ast_map_from_source,
        )
        language = detect_language(file_name)
        container = _MEMBER_SNIPPET_CONTAINERS.get(language or "")
        if container is None:
            return []
        prefix, suffix = container
        nodes = get_ast_map_from_source(prefix + snippet + suffix, file_name)
    except Exception:  # noqa: BLE001 -- arbitrary snippet text degrades to "defines nothing"
        return []
    return [
        n.name for n in nodes
        if n.name and n.name != _MEMBER_SNIPPET_CONTAINER_NAME
    ]


def _snippet_parses_as_member(snippet: str, language: str) -> bool:
    """True when ``snippet`` parses as a member definition of ``language``.

    Issue #5: the standalone validate-parse gate asks a member snippet the
    wrong question — ``constructor(...) {...}`` is ungrammatical at
    compilation-unit level BY DESIGN. Re-asking inside the language's
    minimal container accepts exactly the well-formed member replacements.
    """
    from ..data_gen.ast_analyzer import validate_parse
    from .ast_utils import _MEMBER_SNIPPET_CONTAINERS

    container = _MEMBER_SNIPPET_CONTAINERS.get(language)
    if container is None:
        return False
    prefix, suffix = container
    try:
        return validate_parse(prefix + snippet + suffix, language)
    except Exception:  # noqa: BLE001 -- a wedged/unknown grammar cannot validate the snippet
        return False


def _try_tldr_snippet_parse(snippet: str, ext: str) -> list[str]:
    """Return the definition names a parser sees in ``snippet``.

    B3: the primary parser is fastedit's own in-memory tree-sitter
    resolver (:func:`get_ast_map_from_source`) — the same authoritative
    source every AST map in the pipeline uses (B26 rationale: no daemon,
    no salsa-cache staleness, no temp file), and the only one that knows
    the data formats (html/xml/json/yaml/css/toml/sql/dockerfile/...)
    whose symbols B3 wired. The ``tldr structure`` daemon stays as a
    fallback for the historical surface in case an in-memory map comes
    back empty for a language the daemon knows.

    Issue #5: when neither the bare in-memory parse nor the daemon sees a
    definition, the snippet is re-read as a MEMBER of its language's
    minimal container (:func:`_member_snippet_parse_names`) — a
    ``constructor(...) {...}`` replacement defines ``constructor``, it is
    not "defines nothing".
    """
    names: list[str] = []
    try:
        from .ast_utils import get_ast_map_from_source
        nodes = get_ast_map_from_source(snippet, f"snippet{ext}")
        names = [n.name for n in nodes if n.name]
    except Exception:  # noqa: BLE001 -- deliberate: parsing arbitrary snippet text must degrade to the next resolver, never raise
        names = []
    if names:
        return names

    member_names = _member_snippet_parse_names(snippet, f"snippet{ext}")
    if member_names:
        return member_names

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        write_all(fd, snippet.encode())
        os.close(fd)

        result = subprocess.run(
            ["tldr", "structure", tmp_path, "--format", "compact"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        files = data.get("files", [])
        if not files:
            return []

        return [d["name"] for d in files[0].get("definitions", []) if d.get("name")]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def _regex_extract_names(snippet: str) -> list[str]:
    """Extract definition names using multi-language regex patterns."""
    names = []
    seen: set[str] = set()
    for line in snippet.splitlines():
        for pattern in _DEFINITION_PATTERNS:
            m = pattern.search(line)
            if m:
                name = m.group(1)
                if name not in seen:
                    names.append(name)
                    seen.add(name)
                break  # one match per line
    return names


def _get_snippet_definitions(snippet: str, language: str | None) -> list[ASTNode]:
    """Parse a snippet with tldr structure to get definitions with line ranges."""
    ext = _LANG_EXT.get(language, ".py") if language else ".py"
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        write_all(fd, snippet.encode())
        os.close(fd)
        nodes = _get_ast_via_structure(tmp_path)
        if not nodes:
            nodes = _get_ast_via_extract(tmp_path, len(snippet.splitlines()))
        return nodes
    except OSError:
        return []
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Import detection and snippet splitting (tree-sitter AST)
# ---------------------------------------------------------------------------

def _get_import_line_set(source: str, language: str) -> set[int]:
    """Get 1-indexed line numbers of import statements using tree-sitter.

    Walks the AST to find import nodes for the given language.
    Multi-line imports (Python ``from X import (...)`` , Go ``import (...)``)
    are handled natively — all lines within the node are included.

    For Ruby/Elixir/Lua, filters ``call`` nodes by function name to avoid
    matching non-import calls.
    """
    import_types = _TS_IMPORT_TYPES.get(language, set())
    if not import_types:
        return set()

    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: parse of arbitrary source (unknown language/malformed) yields no import lines, never propagates
        return set()

    call_names = _IMPORT_CALL_NAMES.get(language)
    import_lines: set[int] = set()

    def is_import(node) -> bool:
        if node.type not in import_types:
            return False
        if call_names and node.children:
            name = source[node.children[0].start_byte:node.children[0].end_byte]
            return name in call_names
        return not call_names

    def walk(node):
        if is_import(node):
            for line in range(node.start_point[0] + 1, node.end_point[0] + 2):
                import_lines.add(line)
            return  # don't recurse into import nodes
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return import_lines


def _has_import_changes(
    snippet: str, original_code: str, language: str | None = None,
) -> bool:
    """Check if the snippet contains new import statements.

    Parses the snippet with tree-sitter to find import nodes, then checks
    whether each import line already exists in the original file.
    """
    if not language:
        return False

    snippet_import_lines = _get_import_line_set(snippet, language)
    if not snippet_import_lines:
        return False

    snippet_lines = snippet.splitlines()
    for line_num in snippet_import_lines:
        if 0 < line_num <= len(snippet_lines):
            import_text = snippet_lines[line_num - 1].strip()
            if import_text and import_text not in original_code:
                return True
    return False


def _split_snippet(
    snippet: str, language: str | None = None,
) -> tuple[str, str]:
    """Split a multi-site snippet into import part and code part.

    Uses tree-sitter to identify import lines.  Everything before the first
    non-import, non-blank line goes into the import snippet; the rest is code.

    Returns:
        (import_snippet, code_snippet) — either may be empty.
    """
    if not language:
        return "", snippet

    import_line_nums = _get_import_line_set(snippet, language)
    if not import_line_nums:
        return "", snippet

    lines = snippet.splitlines(keepends=True)
    import_lines: list[str] = []
    code_lines: list[str] = []
    past_imports = False

    for i, line in enumerate(lines, 1):
        if past_imports:
            code_lines.append(line)
            continue

        if i in import_line_nums:
            import_lines.append(line)
            continue

        # Blank line right after imports — include in import part as separator
        if import_lines and line.strip() == "":
            import_lines.append(line)
            continue

        # First non-import, non-blank line → switch to code
        past_imports = True
        code_lines.append(line)

    return "".join(import_lines), "".join(code_lines)


def _find_import_region(
    original_code: str,
    language: str,
    ast_nodes: list[ASTNode],
) -> tuple[int, int] | None:
    """Find the import region in the original file using tree-sitter.

    Finds all import nodes before the first AST definition and returns
    their combined line range.  Multi-line imports are handled natively
    by tree-sitter (the entire node spans all lines).
    """
    import_lines = _get_import_line_set(original_code, language)
    if not import_lines:
        return None

    # Restrict to lines before the first AST definition. The `imports`
    # pseudo-symbol (issue #5) IS the import block, not a definition —
    # counting it as the first definition would push the boundary above
    # the imports and filter the whole region away.
    definition_starts = [
        n.line_start for n in ast_nodes if n.kind != "imports"
    ]
    if definition_starts:
        first_def = min(definition_starts)
        import_lines = {ln for ln in import_lines if ln < first_def}

    if not import_lines:
        return None

    return (min(import_lines), max(import_lines))


# ---------------------------------------------------------------------------
# Node matching
# ---------------------------------------------------------------------------

def _extract_identifiers(source: str, language: str) -> set[str]:
    """Extract all identifier tokens from source code using tree-sitter."""
    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: parse of arbitrary source (unknown language/malformed) yields no identifiers, never propagates
        return set()

    identifiers: set[str] = set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "identifier":
            identifiers.add(source[node.start_byte:node.end_byte])
        stack.extend(node.children)
    return identifiers


def _find_matching_nodes(
    snippet: str,
    original_lines: list[str],
    ast_nodes: list[ASTNode],
    language: str | None = None,
) -> list[ASTNode]:
    """Find AST nodes that the snippet is editing.

    Strategies (in order):
    1. Extract names from snippet via tldr → match to AST nodes by name
    2. AST identifier scoring — parse snippet with tree-sitter, extract
       identifiers, filter to discriminative ones (appear in ≤30% of nodes),
       score each node by how many it contains
    3. Signature substring matching (fallback for languages without tree-sitter)
    """
    matched: list[ASTNode] = []

    # Strategy 1: name-based matching (uses tldr structure on snippet)
    snippet_names = _extract_snippet_names(snippet, language)
    for name in snippet_names:
        for node in ast_nodes:
            if node.name == name and node not in matched:
                matched.append(node)

    if matched:
        return matched

    # Strategy 2: AST identifier scoring via tree-sitter
    if language:
        snippet_idents = _extract_identifiers(snippet, language)
        if snippet_idents and len(ast_nodes) >= 2:
            # Build per-node text cache
            node_texts = {}
            for node in ast_nodes:
                node_texts[node.name] = "\n".join(
                    original_lines[node.line_start - 1:node.line_end]
                )

            # Filter to discriminative identifiers (appear in ≤30% of nodes)
            threshold = max(1, len(ast_nodes) * 0.3)
            discriminative = set()
            for ident in snippet_idents:
                count = sum(1 for text in node_texts.values() if ident in text)
                if count <= threshold:
                    discriminative.add(ident)

            if discriminative:
                best_hits = 0
                best_node = None
                for node in ast_nodes:
                    hits = sum(
                        1 for ident in discriminative
                        if ident in node_texts[node.name]
                    )
                    if hits > best_hits:
                        best_hits = hits
                        best_node = node

                if best_node and best_hits >= min(3, len(discriminative)):
                    return [best_node]

    # Strategy 3: signature substring matching (no tree-sitter needed)
    for node in ast_nodes:
        sig = node.signature.strip()
        if sig and sig in snippet and node not in matched:
                matched.append(node)

    if matched:
        return matched

    return []


def _region_bracketing_anchor(
    anchor: int,
    ast_nodes: list[ASTNode],
    total_lines: int,
) -> tuple[int, int] | None:
    """Span the AST nodes bracketing ``anchor`` (1-indexed file line).

    Exact port of the pre-B28 bracketing: the closest node ending at-or-
    before the anchor (+2 slack) opens the region; the closest node
    starting at-or-after it closes it. Returns ``None`` only when no nodes
    exist on either side.
    """
    before_node = None
    after_node = None
    for node in ast_nodes:
        if node.line_end <= anchor + 2 and (
            before_node is None or node.line_end > before_node.line_end
        ):
            before_node = node
        if node.line_start >= anchor and (
            after_node is None or node.line_start < after_node.line_start
        ):
            after_node = node

    if before_node and after_node:
        return (before_node.line_start, after_node.line_end)
    if before_node:
        return (
            before_node.line_start,
            min(total_lines, before_node.line_end + 30),
        )
    if after_node:
        return (max(1, after_node.line_start - 30), after_node.line_end)
    return None


def _consistent_insertion_region(
    context_occurrences: list[tuple[str, list[int]]],
    ast_nodes: list[ASTNode],
    total_lines: int,
) -> tuple[int, int] | None:
    """Bracket one insertion region that ALL context lines agree on (B28).

    The anchor is the RAREST context line — least frequent normalized
    content across the original, ties broken by the longer (more
    specific) line. Pre-fix every context line matched its FIRST
    occurrence and ``max()`` picked the anchor, so one common line steered
    the insertion anywhere. Every candidate anchor occurrence brackets a
    region; the region wins only when it contains at least one occurrence
    of EVERY context line, and only when exactly one distinct region does.
    ``None`` means conflicting (or ambiguous) context — the caller must
    fall back rather than guess a neighborhood.
    """
    if not ast_nodes:
        return None

    anchor_positions = min(
        context_occurrences,
        key=lambda item: (len(item[1]), -len(item[0])),
    )[1]

    regions: list[tuple[int, int]] = []
    for anchor in anchor_positions:
        region = _region_bracketing_anchor(anchor, ast_nodes, total_lines)
        if region is None:
            continue
        start, end = region
        consistent = all(
            any(start <= pos <= end for pos in positions)
            for _line, positions in context_occurrences
        )
        if consistent and region not in regions:
            regions.append(region)

    if len(regions) == 1:
        return regions[0]
    # No consistent region, or several distinct ones: a guess either way.
    return None


def _find_insertion_region(
    snippet: str,
    original_lines: list[str],
    ast_nodes: list[ASTNode],
    total_lines: int,
    language: str | None = None,
) -> ChunkRegion | None:
    """Find insertion point for new code using tldr AST analysis.

    When a snippet contains definitions not present in the file's AST,
    locates the insertion point by:
    1. Parsing the snippet with tldr to get new definitions and their line ranges
    2. Identifying context lines (snippet lines outside new definitions)
    3. Matching context lines against the file to find the insertion neighborhood
    4. Creating a chunk spanning the neighboring AST nodes

    B28 (Step 17): the anchor is the RAREST context line and the bracketed
    region must contain at least one occurrence of EVERY context line;
    conflicting context returns ``None`` so the caller's fallback runs
    (for :func:`locate_chunks` that is the conservative whole-file
    region). Context lines that match nothing in the original still
    cannot vote — unchanged.
    """
    # Use tldr to parse both snippet and file
    snippet_defs = _get_snippet_definitions(snippet, language)
    existing_names = {n.name for n in ast_nodes}
    new_defs = [d for d in snippet_defs if d.name not in existing_names]

    if not new_defs:
        return None

    # Build set of snippet lines that belong to new definitions (1-indexed)
    new_def_lines: set[int] = set()
    for d in new_defs:
        for ln in range(d.line_start, d.line_end + 1):
            new_def_lines.add(ln)

    # Context lines: snippet lines NOT in new definitions, not blank, not
    # markers. B28: collect ALL original occurrences per context line
    # (1-indexed) — both the rarity ranking and the consistency check need
    # the full picture, not the first match.
    snippet_lines = snippet.splitlines()
    context_occurrences: list[tuple[str, list[int]]] = []

    for si, sline in enumerate(snippet_lines, 1):
        if si in new_def_lines:
            continue
        stripped = sline.strip()
        if not stripped or is_marker_line(sline):
            continue
        positions = [
            fi + 1 for fi, fline in enumerate(original_lines)
            if fline.strip() == stripped
        ]
        if positions:
            context_occurrences.append((stripped, positions))

    if context_occurrences and ast_nodes:
        region = _consistent_insertion_region(
            context_occurrences, ast_nodes, total_lines,
        )
        if region is not None:
            return ChunkRegion(
                region[0], region[1], [d.name for d in new_defs],
            )
        # B28: context existed but cannot agree on one region — return no
        # region so the caller's fallback runs, instead of letting a
        # common line's first occurrence steer the insertion.
        return None

    # No context matched — default to tail of file
    if len(ast_nodes) >= 2:
        start = ast_nodes[-2].line_start
    elif ast_nodes:
        start = max(1, ast_nodes[-1].line_start - 10)
    else:
        start = max(1, total_lines - 50)

    return ChunkRegion(start, total_lines, [d.name for d in new_defs])


# ---------------------------------------------------------------------------
# Region merging
# ---------------------------------------------------------------------------

def _merge_overlapping_regions(
    regions: list[tuple[int, int]],
    gap: int = 20,
) -> list[tuple[int, int]]:
    """Merge regions that are close together (within `gap` lines)."""
    if not regions:
        return []
    sorted_regions = sorted(regions)
    merged = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
