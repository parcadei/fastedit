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

from .ast_utils import (
    ASTNode,
    ChunkRegion,
    _get_ast_via_extract,
    _get_ast_via_structure,
)

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

_MARKER_RE = re.compile(r'^\s*(?:#|//|/\*)\s*\.\.\..*(?:existing|rest).*\.\.\.')


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


def _try_tldr_snippet_parse(snippet: str, ext: str) -> list[str]:
    """Try to parse a snippet with tldr structure via a temp file."""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.write(fd, snippet.encode())
        os.close(fd)

        result = subprocess.run(
            ["tldr", "structure", tmp_path, "--format", "compact"],
            capture_output=True, text=True, timeout=5,
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
        os.write(fd, snippet.encode())
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
    except Exception:
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

    # Restrict to lines before the first AST definition
    if ast_nodes:
        first_def = min(n.line_start for n in ast_nodes)
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
    except Exception:
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

    # Context lines: snippet lines NOT in new definitions, not blank, not markers
    snippet_lines = snippet.splitlines()
    context_file_lines: list[int] = []  # 1-indexed in original file

    for si, sline in enumerate(snippet_lines, 1):
        if si in new_def_lines:
            continue
        stripped = sline.strip()
        if not stripped or _MARKER_RE.match(sline):
            continue
        # Find this line in the original file
        for fi, fline in enumerate(original_lines):
            if fline.strip() == stripped:
                context_file_lines.append(fi + 1)
                break

    if context_file_lines:
        # Use the latest context line as anchor for insertion point
        anchor = max(context_file_lines)

        # Find AST nodes that bracket the anchor
        before_node = None
        after_node = None
        for node in ast_nodes:
            if node.line_end <= anchor + 2 and (before_node is None or node.line_end > before_node.line_end):
                before_node = node
            if node.line_start >= anchor and (after_node is None or node.line_start < after_node.line_start):
                after_node = node

        # Chunk spans from before_node through after_node
        if before_node and after_node:
            return ChunkRegion(
                before_node.line_start, after_node.line_end,
                [d.name for d in new_defs],
            )
        elif before_node:
            end = min(total_lines, before_node.line_end + 30)
            return ChunkRegion(
                before_node.line_start, end,
                [d.name for d in new_defs],
            )
        elif after_node:
            start = max(1, after_node.line_start - 30)
            return ChunkRegion(
                start, after_node.line_end,
                [d.name for d in new_defs],
            )

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
