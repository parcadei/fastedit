"""AST data types and retrieval via tldr structure/extract.

Provides the ASTNode dataclass and functions to get AST maps from files
using tldr's structure and extract commands.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class ASTNode:
    """A function, method, or class from the AST map."""
    name: str
    kind: str  # "function", "method", "class", "interface", etc.
    line_start: int
    line_end: int
    signature: str
    parent: str | None = None  # parent class name (from tldr extract)


@dataclass
class ChunkRegion:
    """A region of the file to extract and merge."""
    start_line: int  # 1-indexed, inclusive
    end_line: int     # 1-indexed, inclusive
    matched_nodes: list[str]  # names of AST nodes in this region


@dataclass
class ChunkedMergeResult:
    """Result of a chunked merge operation."""
    merged_code: str
    parse_valid: bool
    chunks_used: int
    chunk_regions: list[tuple[int, int]]  # (start, end) of each chunk
    model_tokens: int
    latency_ms: float
    chunks_rejected: int = 0  # chunks rejected due to hallucination


@dataclass
class BatchEdit:
    """A single edit operation in a batch. Extensible — add any number to a list."""
    snippet: str
    after: str | None = None
    replace: str | None = None


@dataclass
class DeleteResult:
    """Result of a deterministic symbol deletion."""
    merged_code: str
    parse_valid: bool
    deleted_symbol: str
    deleted_kind: str  # "function", "method", "class", etc.
    deleted_lines: tuple[int, int]  # (start, end) that were removed
    lines_removed: int


@dataclass
class MoveResult:
    """Result of a deterministic symbol move."""
    merged_code: str
    parse_valid: bool
    moved_symbol: str
    moved_kind: str
    from_lines: tuple[int, int]  # original position
    after_symbol: str
    new_lines: tuple[int, int]   # new position after move


def get_ast_map(file_path: str, total_lines: int = 0) -> list[ASTNode]:
    """Get AST definitions with line ranges and parent class info.

    Primary: tldr structure (has line_start + line_end for most languages),
    enriched with parent info from tldr extract (class→method hierarchy).
    Fallback: tldr extract only (has line_number + parent, end lines computed).
    """
    import logging
    _log = logging.getLogger("fastedit.chunked_merge")
    nodes = _get_ast_via_structure(file_path)
    if nodes:
        _enrich_parents_from_extract(nodes, file_path)
        _log.info("get_ast_map: %d nodes via structure for %s", len(nodes), file_path)
        return nodes
    _log.warning("get_ast_map: structure returned 0 nodes, trying extract for %s", file_path)
    nodes = _get_ast_via_extract(file_path, total_lines)
    _log.info("get_ast_map: %d nodes via extract for %s", len(nodes), file_path)
    return nodes


def _enrich_parents_from_extract(nodes: list[ASTNode], file_path: str) -> None:
    """Set parent class names on method nodes using tldr extract hierarchy.

    tldr structure returns a flat list (no parent info). tldr extract nests
    methods under their class. We call extract, build a line→parent map,
    and apply it to the structure nodes.
    """
    try:
        result = subprocess.run(
            ["tldr", "extract", file_path, "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return

    # Build line_number → parent_class_name map from extract hierarchy
    line_to_parent: dict[int, str] = {}
    for cls in data.get("classes", []):
        cls_name = cls.get("name", "")
        if not cls_name:
            continue
        for method in cls.get("methods", []):
            line = method.get("line_number")
            if line:
                line_to_parent[line] = cls_name

    # Apply to nodes: match by line_start
    for node in nodes:
        if node.kind in ("method", "function") and node.line_start in line_to_parent:
            node.parent = line_to_parent[node.line_start]


def _get_ast_via_structure(file_path: str) -> list[ASTNode]:
    """Primary: tldr structure --format compact → definitions with line ranges."""
    try:
        result = subprocess.run(
            ["tldr", "structure", file_path, "--format", "compact"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        files = data.get("files", [])
        if not files:
            return []
        definitions = files[0].get("definitions", [])

        nodes = []
        for defn in definitions:
            if defn.get("line_start") and defn.get("line_end"):
                nodes.append(ASTNode(
                    name=defn["name"],
                    kind=defn.get("kind", "unknown"),
                    line_start=defn["line_start"],
                    line_end=defn["line_end"],
                    signature=defn.get("signature", ""),
                ))
        return nodes
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


def _get_ast_via_extract(file_path: str, total_lines: int = 0) -> list[ASTNode]:
    """Fallback: tldr extract → line_number only, compute end lines."""
    try:
        result = subprocess.run(
            ["tldr", "extract", file_path, "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    # Collect all entries with line numbers
    # (name, kind, line, signature, parent_class_name_or_None)
    raw: list[tuple[str, str, int, str, str | None]] = []

    for fn in data.get("functions", []):
        if fn.get("name") and fn.get("line_number"):
            raw.append((fn["name"], "function", fn["line_number"], "", None))

    for cls in data.get("classes", []):
        cls_name = cls.get("name", "")
        cls_line = cls.get("line_number", 0)
        if cls_name and cls_line:
            raw.append((cls_name, "class", cls_line, "", None))
        for method in cls.get("methods", []):
            if method.get("name") and method.get("line_number"):
                raw.append((method["name"], "method", method["line_number"], "", cls_name))

    if not raw:
        return []

    raw.sort(key=lambda x: x[2])

    # Compute end lines from consecutive entries
    if not total_lines:
        try:
            with open(file_path) as f:
                total_lines = sum(1 for _ in f)
        except OSError:
            total_lines = raw[-1][2] + 50  # rough estimate

    nodes = []
    for i, (name, kind, line_start, sig, parent) in enumerate(raw):
        line_end = raw[i + 1][2] - 1 if i + 1 < len(raw) else total_lines
        nodes.append(ASTNode(
            name=name, kind=kind,
            line_start=line_start, line_end=line_end,
            signature=sig, parent=parent,
        ))
    return nodes


def _resolve_symbol(name: str, ast_nodes: list[ASTNode]) -> ASTNode | None:
    """Find an AST node by name, supporting 'Class.method' qualification.

    If `name` contains a dot (e.g. 'MyClass.__init__'), uses the node.parent
    field (populated from tldr extract hierarchy) to find the correct method.

    If `name` is a simple identifier, returns the first match (existing behavior).
    """
    if "." in name:
        class_name, method_name = name.split(".", 1)
        for node in ast_nodes:
            if node.name == method_name and node.parent == class_name:
                return node
        return None
    else:
        for node in ast_nodes:
            if node.name == name:
                return node
        return None


def _qualified_symbol_names(ast_nodes: list[ASTNode]) -> list[str]:
    """Build qualified symbol names for error messages.

    Methods with a parent class become 'Class.method'.
    If a name appears only once, use the bare name.
    Duplicate names get qualified to help Claude disambiguate.
    """
    # Count occurrences of each name
    name_counts: dict[str, int] = {}
    for node in ast_nodes:
        name_counts[node.name] = name_counts.get(node.name, 0) + 1

    result = []
    for node in ast_nodes:
        if node.name == "__init__":
            # Always qualify __init__ — it's always a duplicate
            if node.parent:
                result.append(f"{node.parent}.{node.name}")
            continue
        if name_counts[node.name] > 1 and node.parent:
            result.append(f"{node.parent}.{node.name}")
        else:
            result.append(node.name)
    return result
