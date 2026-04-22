"""AST analysis using tree-sitter for language-agnostic code understanding.

Provides structural metadata extraction for any supported language:
function boundaries, class hierarchies, import blocks, scope nesting.
This powers the AST-aware data generation pipeline.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter

# Language module mapping: language name -> tree-sitter package name
_LANGUAGE_MODULES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "swift": "tree_sitter_swift",
    "kotlin": "tree_sitter_kotlin",
    "c_sharp": "tree_sitter_c_sharp",
    "php": "tree_sitter_php",
    "elixir": "tree_sitter_elixir",
}

# File extension -> language mapping
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "c_sharp",
    ".php": "php",
    ".ex": "elixir",
    ".exs": "elixir",
}

# AST node types that represent function-like constructs per language
_FUNCTION_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function",
                    "function_expression", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "arrow_function",
                    "function_expression", "method_signature"},
    "tsx": {"function_declaration", "method_definition", "arrow_function",
            "function_expression", "method_signature"},
    "rust": {"function_item", "impl_item"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "c": {"function_definition"},
    "cpp": {"function_definition", "template_declaration"},
    "ruby": {"method", "singleton_method"},
    "swift": {"function_declaration", "initializer_declaration"},
    "kotlin": {"function_declaration"},
    "c_sharp": {"method_declaration", "constructor_declaration"},
    "php": {"function_definition", "method_declaration"},
    # Elixir: `def`, `defp`, `defmacro`, `defmacrop` all parse as `call`
    # nodes (the macros look like function invocations syntactically).
    # We disambiguate function vs module via the call-target identifier
    # text (see _is_elixir_function / _is_elixir_module).
    "elixir": {"call"},
}

# AST node types that represent class-like constructs
_CLASS_NODE_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "class"},
    "typescript": {"class_declaration", "interface_declaration", "type_alias_declaration"},
    "tsx": {"class_declaration", "interface_declaration", "type_alias_declaration"},
    "rust": {"struct_item", "enum_item", "trait_item"},
    "go": {"type_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "c": {"struct_specifier", "enum_specifier"},
    "cpp": {"class_specifier", "struct_specifier"},
    "ruby": {"class", "module"},
    "swift": {"class_declaration", "struct_declaration", "protocol_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "interface_declaration"},
    "c_sharp": {"class_declaration", "interface_declaration", "struct_declaration"},
    "php": {"class_declaration", "interface_declaration", "trait_declaration"},
    # Elixir: `defmodule` also parses as a `call` node. Distinguished
    # from `def`/`defp` by the target-identifier text.
    "elixir": {"call"},
}

# AST node types for import statements
_IMPORT_NODE_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "import_declaration"},
    "typescript": {"import_statement", "import_declaration"},
    "tsx": {"import_statement", "import_declaration"},
    "rust": {"use_declaration"},
    "go": {"import_declaration"},
    "java": {"import_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include", "using_declaration"},
    "ruby": {"call"},  # require/require_relative
    "swift": {"import_declaration"},
    "kotlin": {"import_header"},
    "c_sharp": {"using_directive"},
    "php": {"namespace_use_declaration"},
    # Elixir: `import`, `alias`, `require`, `use` all appear as `call`
    # nodes with the matching target identifier. Filtered by target text.
    "elixir": {"call"},
}

# Elixir macro-call target identifiers that mark function-like definitions.
# `def` = public function, `defp` = private, `defmacro(p)` = macro.
_ELIXIR_FUNCTION_TARGETS: frozenset[str] = frozenset(
    {"def", "defp", "defmacro", "defmacrop"}
)

# Elixir macro-call target identifiers that mark module-like / protocol
# definitions. Treated as "class-like" for the purposes of this analyzer.
_ELIXIR_MODULE_TARGETS: frozenset[str] = frozenset(
    {"defmodule", "defprotocol", "defimpl"}
)

# Elixir macro-call target identifiers that act as imports.
_ELIXIR_IMPORT_TARGETS: frozenset[str] = frozenset(
    {"import", "alias", "require", "use"}
)


def _elixir_call_target_text(node: tree_sitter.Node, source_bytes: bytes) -> str | None:
    """Return the text of an Elixir `call` node's target identifier, if any.

    Elixir's tree-sitter grammar models every macro invocation as a ``call``
    node with a ``target`` field that is typically an ``identifier``. This
    helper returns that identifier's source text, or ``None`` when the
    node is not a ``call`` or its target is something other than a simple
    identifier (e.g. a ``dot`` for ``IO.puts``).
    """
    if node.type != "call":
        return None
    target = node.child_by_field_name("target")
    if target is None:
        # Field-name may be missing in some grammar builds; the target
        # is always the first named child of a `call` node.
        for child in node.children:
            if child.is_named:
                target = child
                break
    if target is None or target.type != "identifier":
        return None
    return source_bytes[target.start_byte : target.end_byte].decode(
        "utf-8", errors="replace"
    )


def _is_elixir_function_node(node: tree_sitter.Node, source_bytes: bytes) -> bool:
    """True iff *node* is an Elixir `call` introducing a function/macro def."""
    t = _elixir_call_target_text(node, source_bytes)
    return t is not None and t in _ELIXIR_FUNCTION_TARGETS


def _is_elixir_module_node(node: tree_sitter.Node, source_bytes: bytes) -> bool:
    """True iff *node* is an Elixir `call` introducing a module/protocol."""
    t = _elixir_call_target_text(node, source_bytes)
    return t is not None and t in _ELIXIR_MODULE_TARGETS


def _is_elixir_import_node(node: tree_sitter.Node, source_bytes: bytes) -> bool:
    """True iff *node* is an Elixir `call` for import/alias/require/use."""
    t = _elixir_call_target_text(node, source_bytes)
    return t is not None and t in _ELIXIR_IMPORT_TARGETS


def _elixir_definition_name(node: tree_sitter.Node, source_bytes: bytes) -> str:
    """Extract the name of an Elixir `def*`/`defmodule` call.

    Handles the two shapes we see in practice:

    - ``def hello(name) do ... end``  → the first child of ``arguments``
      is a nested ``call`` whose own target identifier holds the name.
    - ``defp helper, do: :ok``        → the first child of ``arguments``
      is a bare ``identifier`` holding the name.
    - ``defmodule Foo do ... end``    → the first child of ``arguments``
      is an ``alias`` node whose text is the module name.
    - ``defmodule Foo.Bar do ... end``→ the ``alias`` text is dotted and
      returned verbatim.

    Returns ``"<anonymous>"`` when the shape doesn't match (e.g. a
    parse-error tree).
    """
    # In tree-sitter-elixir the ``arguments`` child is not tagged with a
    # field name in every grammar build — fall back to positional scan
    # (it's always the first named sibling of the ``target`` identifier).
    args = node.child_by_field_name("arguments")
    if args is None:
        for child in node.children:
            if child.type == "arguments":
                args = child
                break
    if args is None:
        return "<anonymous>"
    # First named child of `arguments` is the head of the definition.
    for child in args.children:
        if not child.is_named:
            continue
        if child.type == "call":
            # def hello(name) — recurse one level to the inner call's target
            inner_target = child.child_by_field_name("target")
            if inner_target is not None:
                return source_bytes[
                    inner_target.start_byte : inner_target.end_byte
                ].decode("utf-8", errors="replace")
        if child.type in ("identifier", "alias"):
            return source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
        # Operator definitions e.g. ``def a + b, do: ...`` — use the
        # operator text as the name so downstream resolvers at least see
        # a stable key.
        return source_bytes[child.start_byte : child.end_byte].decode(
            "utf-8", errors="replace"
        )
    return "<anonymous>"


@dataclass
class ASTNode:
    """Represents a significant AST node with its metadata."""
    node_type: str
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    children: list[ASTNode] = field(default_factory=list)
    parent_name: str | None = None


@dataclass
class FileStructure:
    """Complete structural analysis of a source file."""
    language: str
    file_path: str
    total_lines: int
    functions: list[ASTNode]
    classes: list[ASTNode]
    imports: list[ASTNode]
    top_level_nodes: list[ASTNode]
    has_parse_errors: bool
    nesting_depth: int

    @property
    def complexity_bucket(self) -> str:
        n_funcs = len(self.functions)
        n_classes = len(self.classes)
        if n_funcs <= 3 and n_classes == 0:
            return "simple"
        elif n_funcs <= 10 and n_classes <= 2:
            return "moderate"
        elif n_funcs <= 20:
            return "complex"
        else:
            return "very_complex"


# Cache loaded languages
_language_cache: dict[str, tree_sitter.Language] = {}
_parser_cache: dict[str, tree_sitter.Parser] = {}


def get_language(lang: str) -> tree_sitter.Language:
    """Load a tree-sitter language, caching for reuse."""
    if lang in _language_cache:
        return _language_cache[lang]

    module_name = _LANGUAGE_MODULES.get(lang)
    if not module_name:
        raise ValueError(f"Unsupported language: {lang}")

    mod = importlib.import_module(module_name)

    # Some grammars use non-standard function names
    if lang == "tsx":
        ts_lang = tree_sitter.Language(mod.language_tsx())
    elif lang == "typescript":
        ts_lang = tree_sitter.Language(mod.language_typescript())
    elif lang == "php":
        ts_lang = tree_sitter.Language(mod.language_php())
    else:
        ts_lang = tree_sitter.Language(mod.language())

    _language_cache[lang] = ts_lang
    return ts_lang


def get_parser(lang: str) -> tree_sitter.Parser:
    """Get a parser for the given language, caching for reuse."""
    if lang in _parser_cache:
        return _parser_cache[lang]

    parser = tree_sitter.Parser(get_language(lang))
    _parser_cache[lang] = parser
    return parser


def detect_language(file_path: str | Path) -> str | None:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def parse_code(source: str, language: str) -> tree_sitter.Tree:
    """Parse source code into a tree-sitter AST."""
    parser = get_parser(language)
    return parser.parse(source.encode("utf-8"))


def _get_node_name(
    node: tree_sitter.Node,
    source_bytes: bytes,
    language: str | None = None,
) -> str:
    """Extract the name of a function/class/import node."""
    # Elixir: every defining construct is a `call` — use the dedicated extractor.
    if language == "elixir" and node.type == "call":
        return _elixir_definition_name(node, source_bytes)

    # Look for an identifier child node
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier",
                          "type_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
        # Python decorated_definition: dig into the inner definition
        if child.type == "function_definition" or child.type == "class_definition":
            return _get_node_name(child, source_bytes, language)
    # For imports, return the full text
    if node.type in _IMPORT_NODE_TYPES.get("python", set()) | \
                     _IMPORT_NODE_TYPES.get("javascript", set()):
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        return text[:80]  # truncate long imports
    return "<anonymous>"


# Kind labels for the Elixir `call`-node filter. Keep in sync with the
# _ELIXIR_*_TARGETS frozensets above.
_ELIXIR_KIND_PREDICATES = {
    "function": _is_elixir_function_node,
    "class": _is_elixir_module_node,
    "import": _is_elixir_import_node,
}


def _collect_nodes(
    node: tree_sitter.Node,
    source_bytes: bytes,
    language: str,
    target_types: set[str],
    parent_name: str | None = None,
    elixir_kind: str | None = None,
) -> list[ASTNode]:
    """Recursively collect AST nodes matching target types.

    ``elixir_kind``: one of ``"function" | "class" | "import"`` when
    ``language == "elixir"``. Because every Elixir define is a ``call``
    node, the caller tells us which macro family we want to harvest.
    Ignored for every other language.
    """
    results = []
    type_match = node.type in target_types
    # Elixir: narrow the `call` match to the requested macro family.
    elixir_match = True
    if type_match and language == "elixir" and node.type == "call":
        pred = _ELIXIR_KIND_PREDICATES.get(elixir_kind or "")
        elixir_match = bool(pred and pred(node, source_bytes))

    if type_match and elixir_match:
        name = _get_node_name(node, source_bytes, language)
        ast_node = ASTNode(
            node_type=node.type,
            name=name,
            start_line=node.start_point[0] + 1,  # 1-indexed
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            parent_name=parent_name,
        )
        # Collect nested functions/classes within this node
        for child in node.children:
            ast_node.children.extend(
                _collect_nodes(
                    child, source_bytes, language, target_types, name, elixir_kind
                )
            )
        results.append(ast_node)
    else:
        for child in node.children:
            results.extend(
                _collect_nodes(
                    child,
                    source_bytes,
                    language,
                    target_types,
                    parent_name,
                    elixir_kind,
                )
            )
    return results


def _max_nesting_depth(node: tree_sitter.Node, current: int = 0) -> int:
    """Calculate maximum nesting depth of block-like nodes."""
    block_types = {"block", "statement_block", "compound_statement",
                   "if_statement", "for_statement", "while_statement",
                   "match_statement", "switch_statement", "try_statement"}
    depth = current + 1 if node.type in block_types else current
    max_child = depth
    for child in node.children:
        max_child = max(max_child, _max_nesting_depth(child, depth))
    return max_child


def analyze_file(source: str, language: str, file_path: str = "<unknown>") -> FileStructure:
    """Perform complete structural analysis of a source file.

    Returns a FileStructure with all functions, classes, imports,
    and structural metadata extracted via tree-sitter AST parsing.
    """
    tree = parse_code(source, language)
    root = tree.root_node
    source_bytes = source.encode("utf-8")

    func_types = _FUNCTION_NODE_TYPES.get(language, set())
    class_types = _CLASS_NODE_TYPES.get(language, set())
    import_types = _IMPORT_NODE_TYPES.get(language, set())

    functions = _collect_nodes(
        root, source_bytes, language, func_types, elixir_kind="function"
    )
    classes = _collect_nodes(
        root, source_bytes, language, class_types, elixir_kind="class"
    )
    imports = _collect_nodes(
        root, source_bytes, language, import_types, elixir_kind="import"
    )

    # Top-level nodes (direct children of root)
    top_level = []
    for child in root.children:
        name = _get_node_name(child, source_bytes, language)
        top_level.append(ASTNode(
            node_type=child.type,
            name=name,
            start_line=child.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            start_byte=child.start_byte,
            end_byte=child.end_byte,
        ))

    nesting = _max_nesting_depth(root)
    has_errors = root.has_error

    return FileStructure(
        language=language,
        file_path=file_path,
        total_lines=source.count("\n") + 1,
        functions=functions,
        classes=classes,
        imports=imports,
        top_level_nodes=top_level,
        has_parse_errors=has_errors,
        nesting_depth=nesting,
    )


def analyze_file_from_path(file_path: str | Path) -> FileStructure | None:
    """Analyze a file from disk, auto-detecting language."""
    path = Path(file_path)
    language = detect_language(path)
    if language is None:
        return None
    source = path.read_text(encoding="utf-8", errors="ignore")
    return analyze_file(source, language, str(path))


def validate_parse(source: str, language: str) -> bool:
    """Check if source code parses without errors."""
    tree = parse_code(source, language)
    return not tree.root_node.has_error


def count_ast_nodes(tree: tree_sitter.Tree) -> int:
    """Count total AST nodes in a tree."""
    count = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children)
    return count


def get_node_at_lines(
    source: str, language: str, start_line: int, end_line: int
) -> list[ASTNode]:
    """Find AST nodes that overlap with the given line range."""
    tree = parse_code(source, language)
    source_bytes = source.encode("utf-8")
    results = []

    def walk(node: tree_sitter.Node) -> None:
        node_start = node.start_point[0] + 1
        node_end = node.end_point[0] + 1
        if node_end < start_line or node_start > end_line:
            return
        if node_start >= start_line and node_end <= end_line:
            results.append(ASTNode(
                node_type=node.type,
                name=_get_node_name(node, source_bytes, language),
                start_line=node_start,
                end_line=node_end,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            ))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return results
