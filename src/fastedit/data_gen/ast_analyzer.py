"""AST analysis using tree-sitter for language-agnostic code understanding.

Provides structural metadata extraction for any supported language:
function boundaries, class hierarchies, import blocks, scope nesting.
This powers the AST-aware data generation pipeline.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import tree_sitter


class GrammarUnavailableError(ValueError):
    """Raised when a language cannot be resolved to a tree-sitter grammar.

    Typed subclass of :class:`ValueError` so existing
    ``except (ValueError, ...)`` consumers keep working, but callers that
    want to distinguish "grammar missing" from "bad input" can catch this
    specifically. The resolver reports truthfully — it NEVER falls back to
    "no validation"; whether an unresolvable language aborts, degrades, or
    skips validation is the caller's policy, not the resolver's.

    Attributes:
        language: The (canonicalized) language name that could not be
            resolved.
    """

    def __init__(self, language: str, message: str) -> None:
        super().__init__(message)
        self.language = language


@dataclass(frozen=True)
class GrammarSpec:
    """Declarative resolution recipe for one canonical fastedit language.

    Adding a language whose wheel follows the standard convention requires
    NO code change at all (see :func:`get_language`'s generic probe); a
    ``GrammarSpec`` row exists only where the wheel deviates — multi-grammar
    packages (typescript/javascript/tsx in one wheel, php variants) or
    unusual entry-point names. For those, ``entries`` are EXACT candidates:
    the resolver never substitutes a different grammar as a fallback, since
    e.g. parsing ``.tsx`` with the typescript grammar would mis-parse JSX.

    Attributes:
        modules: Candidate pip module names, tried in order (imported with
            :func:`importlib.import_module`).
        entries: Candidate attribute names inside the first module that
            yields the grammar — a zero-arg function returning the language
            capsule (``language()`` / ``language_typescript()``) or a bare
            capsule constant.
    """

    modules: tuple[str, ...]
    entries: tuple[str, ...]


# Canonical fastedit language -> how to obtain its tree-sitter grammar.
# Languages NOT listed here still resolve via the generic convention probe
# (module ``tree_sitter_<canonical>``, entry ``language()`` /
# ``language_<canonical>()``), so freshly installed wheels need no edit.
LANGUAGE_ALIASES: dict[str, GrammarSpec] = {
    "python": GrammarSpec(
        modules=("tree_sitter_python",), entries=("language",),
    ),
    "javascript": GrammarSpec(
        modules=("tree_sitter_javascript",), entries=("language",),
    ),
    # tree_sitter_typescript ships TWO grammars in one wheel (plus the JS
    # grammar re-exported by its own wheel): exact entries, no fallback.
    "typescript": GrammarSpec(
        modules=("tree_sitter_typescript",), entries=("language_typescript",),
    ),
    "tsx": GrammarSpec(
        modules=("tree_sitter_typescript",), entries=("language_tsx",),
    ),
    "rust": GrammarSpec(modules=("tree_sitter_rust",), entries=("language",)),
    "go": GrammarSpec(modules=("tree_sitter_go",), entries=("language",)),
    "java": GrammarSpec(modules=("tree_sitter_java",), entries=("language",)),
    "c": GrammarSpec(modules=("tree_sitter_c",), entries=("language",)),
    "cpp": GrammarSpec(modules=("tree_sitter_cpp",), entries=("language",)),
    "ruby": GrammarSpec(modules=("tree_sitter_ruby",), entries=("language",)),
    "swift": GrammarSpec(modules=("tree_sitter_swift",), entries=("language",)),
    "kotlin": GrammarSpec(modules=("tree_sitter_kotlin",), entries=("language",)),
    "c_sharp": GrammarSpec(
        modules=("tree_sitter_c_sharp",), entries=("language",),
    ),
    # tree_sitter_php exposes variant entry points (full PHP with HTML
    # interleaving vs php_only); ``language`` is the historical generic
    # name some wheel revisions provide — same full grammar.
    "php": GrammarSpec(
        modules=("tree_sitter_php",), entries=("language_php", "language"),
    ),
    "elixir": GrammarSpec(
        modules=("tree_sitter_elixir",), entries=("language",),
    ),
    # --- B2 broad grammar set (Step B2): every row below was verified by
    # importing the wheel and parsing a representative snippet against
    # tree-sitter 0.25.x before being declared. Rows exist for ALL of them
    # (not only the multi-grammar wheels) so detect_language()'s promise
    # keeps a declarative backer and error hints name the right wheel.
    "html": GrammarSpec(modules=("tree_sitter_html",), entries=("language",)),
    # tree_sitter_xml ships TWO grammars: the XML document grammar and a
    # separate DTD grammar. Exact entries — the generic `language` probe
    # would find neither.
    "xml": GrammarSpec(
        modules=("tree_sitter_xml",), entries=("language_xml",),
    ),
    "xml_dtd": GrammarSpec(
        modules=("tree_sitter_xml",), entries=("language_dtd",),
    ),
    # tree_sitter_markdown ships the block grammar plus the markdown_inline
    # sub-grammar used for inline constructs inside block-level nodes. Both
    # live in the SAME wheel (there is no separate markdown-inline package on
    # PyPI); the inline entry point is named `inline_language`.
    "markdown": GrammarSpec(
        modules=("tree_sitter_markdown",), entries=("language",),
    ),
    "markdown_inline": GrammarSpec(
        modules=("tree_sitter_markdown",), entries=("inline_language",),
    ),
    "json": GrammarSpec(modules=("tree_sitter_json",), entries=("language",)),
    "yaml": GrammarSpec(modules=("tree_sitter_yaml",), entries=("language",)),
    "css": GrammarSpec(modules=("tree_sitter_css",), entries=("language",)),
    "bash": GrammarSpec(modules=("tree_sitter_bash",), entries=("language",)),
    "toml": GrammarSpec(modules=("tree_sitter_toml",), entries=("language",)),
    "sql": GrammarSpec(modules=("tree_sitter_sql",), entries=("language",)),
    "dockerfile": GrammarSpec(
        modules=("tree_sitter_dockerfile",), entries=("language",),
    ),
    # --- B2 all-grammars extra: verified wheels that are NOT hard
    # dependencies. `language="<name>"` resolves when the extra (or the bare
    # wheel) is installed and raises the typed GrammarUnavailableError with a
    # pip hint otherwise. No extensions map to these by default: the
    # EXTENSION_TO_LANGUAGE contract is "resolvable on a DEFAULT install".
    "scala": GrammarSpec(modules=("tree_sitter_scala",), entries=("language",)),
    "lua": GrammarSpec(modules=("tree_sitter_lua",), entries=("language",)),
    "perl": GrammarSpec(modules=("tree_sitter_perl",), entries=("language",)),
    "julia": GrammarSpec(modules=("tree_sitter_julia",), entries=("language",)),
    "zig": GrammarSpec(modules=("tree_sitter_zig",), entries=("language",)),
    "svelte": GrammarSpec(modules=("tree_sitter_svelte",), entries=("language",)),
    "graphql": GrammarSpec(modules=("tree_sitter_graphql",), entries=("language",)),
    # tree-sitter-hcl also parses Terraform configuration (HCL is Terraform's
    # syntax; a `variable`/`resource` block verified parse-clean).
    "hcl": GrammarSpec(modules=("tree_sitter_hcl",), entries=("language",)),
    "make": GrammarSpec(modules=("tree_sitter_make",), entries=("language",)),
    "nix": GrammarSpec(modules=("tree_sitter_nix",), entries=("language",)),
}

# Alternate spellings (user input, other tools' naming) -> canonical
# fastedit name. Canonicalization happens BEFORE the parser cache key, so
# ``csharp`` and ``c_sharp`` share one parser object. Targets must be
# canonical names present in LANGUAGE_ALIASES.
LANGUAGE_NAME_ALIASES: dict[str, str] = {
    "csharp": "c_sharp",
    "c#": "c_sharp",
    "golang": "go",
    "c++": "cpp",
    # B2: common alternate spellings for the broad grammar set. All targets
    # are canonical B2 names whose wheels are verified (bash/hcl/dockerfile/
    # make/graphql) — "terraform"→hcl is honest because the HCL grammar
    # parses Terraform configuration (probe-verified on a `variable` block).
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "docker": "dockerfile",
    "makefile": "make",
    "gql": "graphql",
    "terraform": "hcl",
    "yml": "yaml",
}

# Optional aggregate packs, tried as a FALLBACK after the per-language
# wheels. Never a required dependency: when absent the resolver simply
# reports :class:`GrammarUnavailableError` instead.
_LANGUAGE_PACK_MODULES: tuple[str, ...] = (
    "tree_sitter_languages",
    "tree_sitter_language_pack",
)

# File extension -> language mapping. Every value MUST be a canonical name
# resolvable by :func:`get_language` whose grammar is a HARD dependency —
# detect_language() promises resolution on a DEFAULT install, so B2 wires
# only the core-format extensions here. Languages served by the optional
# ``all-grammars`` extra (scala, lua, perl, julia, zig, svelte, graphql,
# hcl, make, nix, and every pack-only language) deliberately have NO
# extension mapping: they resolve when requested explicitly
# (``language="scala"``) and raise the typed GrammarUnavailableError with a
# pip hint when the extra is absent — never a silent plain-text fallback
# for a file fastedit claims to understand.
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
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
    ".hxx": "cpp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "c_sharp",
    ".php": "php",
    ".phtml": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".ex": "elixir",
    ".exs": "elixir",
    # --- B2 core formats (hard dependencies; wheel-verified) ---
    ".html": "html",
    ".htm": "html",
    # SVG is XML with namespaced attributes — the XML grammar parses a
    # representative SVG document with zero error traits (probe-verified).
    # Same for DTD files via the xml wheel's dedicated DTD sub-grammar.
    ".xml": "xml",
    ".svg": "xml",
    ".dtd": "xml_dtd",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".css": "css",
    ".sh": "bash",
    ".bash": "bash",
    ".toml": "toml",
    ".sql": "sql",
    # Extension-only form: bare "Dockerfile" / "Containerfile" carry no
    # suffix, so detect_language() (suffix-based) cannot see them — the
    # .dockerfile convention still reaches the grammar.
    ".dockerfile": "dockerfile",
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


# Caches — parse calls happen on multi-MB files, so a language and its
# parser are resolved once per process and reused. Keyed by CANONICAL
# language name (alias spellings share one entry). Failed resolutions are
# never cached negatively: installing a grammar mid-process is picked up
# on the next call.
_language_cache: dict[str, tree_sitter.Language] = {}
_parser_cache: dict[str, tree_sitter.Parser] = {}


def clear_grammar_caches() -> None:
    """Drop every cached Language/Parser (used by tests and B2 verification)."""
    _language_cache.clear()
    _parser_cache.clear()


def canonical_language_name(lang: str) -> str:
    """Map any accepted spelling to its canonical fastedit language name.

    Handles casing/whitespace and the declared :data:`LANGUAGE_NAME_ALIASES`
    (``csharp`` → ``c_sharp``, ``golang`` → ``go``, ``c++`` → ``cpp``).
    Purely lexical — does NOT verify a grammar is actually resolvable.
    """
    name = str(lang).strip().lower()
    return LANGUAGE_NAME_ALIASES.get(name, name)


def _grammar_module_candidates(canonical: str) -> tuple[str, ...]:
    """Candidate pip module names for a canonical language, in try order."""
    spec = LANGUAGE_ALIASES.get(canonical)
    if spec is not None:
        return spec.modules
    # Generic convention: a wheel named after the language. The no-underscore
    # variant covers packages that flatten multi-word names
    # (tree_sitter_<x> where <x> keeps underscores is the dominant form).
    return (
        f"tree_sitter_{canonical}",
        f"tree_sitter_{canonical.replace('_', '')}",
    )


def _grammar_entry_candidates(canonical: str) -> tuple[str, ...]:
    """Candidate grammar entry-point names, in try order.

    Table'd languages use their declared EXACT entries (never a generic
    fallback that could silently substitute a different grammar, as with
    the typescript/tsx split wheel). Table-less languages get the standard
    wheel conventions.
    """
    spec = LANGUAGE_ALIASES.get(canonical)
    if spec is not None:
        return spec.entries
    return ("language", f"language_{canonical}")


def _language_from_module(module: object, entry: str) -> tree_sitter.Language | None:
    """Build a Language from one module attribute, or None if absent.

    Handles the known wheel conventions generically: a zero-arg function
    returning the language capsule (``language()``, ``language_tsx()``, ...)
    or a bare capsule/pointer attribute.
    """
    attribute = getattr(module, entry, None)
    if attribute is None:
        return None
    capsule = attribute() if callable(attribute) else attribute
    return tree_sitter.Language(capsule)


def _resolve_from_language_pack(canonical: str) -> tree_sitter.Language | None:
    """Fallback resolver over the optional aggregate grammar packs.

    Tries each declared pack module (``tree_sitter_languages`` or the newer
    ``tree_sitter_language_pack``) via its ``get_language(name)`` — or
    ``get_parser(name).language`` when only the parser getter exists.
    Returns None when no pack is importable or the pack doesn't know the
    language; never raises for absence.
    """
    for pack_name in _LANGUAGE_PACK_MODULES:
        try:
            pack = importlib.import_module(pack_name)
        except ImportError:
            continue
        getter = getattr(pack, "get_language", None)
        if getter is not None:
            try:
                obj = getter(canonical)
            except Exception:  # noqa: BLE001 — pack lookup failure = "not here"
                obj = None
            if obj is not None:
                language_obj = (
                    obj if isinstance(obj, tree_sitter.Language)
                    else tree_sitter.Language(obj)
                )
                return language_obj
        parser_getter = getattr(pack, "get_parser", None)
        if parser_getter is not None:
            try:
                pack_parser = parser_getter(canonical)
            except Exception:  # noqa: BLE001 — pack lookup failure = "not here"
                pack_parser = None
            if pack_parser is not None and hasattr(pack_parser, "parse"):
                language_obj = getattr(pack_parser, "language", None)
                if isinstance(language_obj, tree_sitter.Language):
                    return language_obj
    return None


def _grammar_unavailable(
    canonical: str,
    modules: tuple[str, ...],
    failures: list[str],
) -> GrammarUnavailableError:
    """Build the typed error for an unresolvable language.

    The message names the language, what was tried, and the pip install
    hint — the caller decides policy, the resolver stays truthful.
    """
    primary = modules[0].replace("_", "-")  # tree_sitter_zig → tree-sitter-zig
    detail = ""
    if failures:
        detail = " Underlying errors: " + "; ".join(failures[-3:])
    return GrammarUnavailableError(
        canonical,
        f"no tree-sitter grammar available for language '{canonical}'. "
        f"Tried modules: {', '.join(modules)}. Install a grammar wheel "
        f"(`pip install {primary}`) or an aggregate language pack "
        f"(`pip install tree-sitter-languages`), then retry.{detail}",
    )


def get_language(lang: str) -> tree_sitter.Language:
    """Resolve a canonical language name to a tree-sitter Language.

    Resolution order per language:

    1. Declared :data:`LANGUAGE_ALIASES` spec — import each candidate
       ``tree_sitter_<x>`` module, try each declared (exact) entry point.
    2. Generic convention probe — module ``tree_sitter_<canonical>``, entry
       ``language()`` / ``language_<canonical>()``. This is what makes a
       freshly ``pip install``ed wheel work with zero code changes.
    3. Optional aggregate pack (:func:`_resolve_from_language_pack`) —
       ``tree_sitter_languages``-style ``get_language``/``get_parser``.

    Exhausting all three raises :class:`GrammarUnavailableError` naming the
    language and the ``pip install`` hint. Results are cached per canonical
    name; failures are not cached.
    """
    canonical = canonical_language_name(lang)
    cached = _language_cache.get(canonical)
    if cached is not None:
        return cached

    modules = _grammar_module_candidates(canonical)
    entries = _grammar_entry_candidates(canonical)
    failures: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        for entry in entries:
            try:
                language_obj = _language_from_module(module, entry)
            except Exception as exc:  # noqa: BLE001 — wheel ABI errors are "not resolvable here"
                failures.append(f"{module_name}.{entry}: {exc}")
                continue
            if language_obj is not None:
                _language_cache[canonical] = language_obj
                return language_obj

    pack_language = _resolve_from_language_pack(canonical)
    if pack_language is not None:
        _language_cache[canonical] = pack_language
        return pack_language

    raise _grammar_unavailable(canonical, modules, failures)


def get_parser(lang: str) -> tree_sitter.Parser:
    """Get a parser for the given language, caching for reuse.

    Alias spellings canonicalize to the same key, so ``get_parser("csharp")``
    and ``get_parser("c_sharp")`` return the identical parser object.
    """
    canonical = canonical_language_name(lang)
    cached = _parser_cache.get(canonical)
    if cached is not None:
        return cached

    parser = tree_sitter.Parser(get_language(canonical))
    _parser_cache[canonical] = parser
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
    """Check if source code parses without errors.

    Delegates to :func:`parse_diagnostics` so every absolute check sees the
    same defect set the relative rule sees — including the C2 suite-opener
    scan for colon-headed indentation languages, which tree-sitter's
    error recovery silently waves through.

    Guarded grammars (:data:`PATHOLOGICAL_RECOVERY_GRAMMARS`) route through
    the subprocess watchdog with the same routing: healthy input returns
    the ordinary verdict; a wedged parse raises the typed
    :class:`ParseWatchdogTimeout` instead of hanging.
    """
    return not parse_diagnostics(source, language).errors


class ParseDiagnostics(NamedTuple):
    """Ordered tree-sitter error traits for one parse (Step A2, req. 9).

    ``errors`` is the document-ordered list of ``(start_byte, end_byte,
    kind)`` spans for the parse's ERROR and MISSING nodes (``kind`` is
    ``"ERROR"`` or ``"MISSING"``). ``is_valid`` is True iff the parse
    produced no (unfiltered) error trait — identical to
    :func:`validate_parse`. ``source`` is the exact text the byte spans
    index — see :func:`parse_diagnostics` for the normalization contract.

    ``grammar_artifacts`` (Step G1a) holds the traits that were removed
    from ``errors`` because they match a declared KNOWN grammar artifact
    (:data:`_GRAMMAR_ARTIFACT_FILTERS`) — an error trait the grammar
    itself emits systematically for well-formed input, not a property of
    the text. It is empty for every language without a declared filter.
    """

    errors: list[tuple[int, int, str]]
    is_valid: bool
    source: str
    grammar_artifacts: tuple[tuple[int, int, str], ...] = ()


# ---------------------------------------------------------------------------
# Step G1a — bounded parse watchdog for pathological-recovery grammars
# ---------------------------------------------------------------------------

PATHOLOGICAL_RECOVERY_GRAMMARS = frozenset({"cobol"})
"""Grammars whose error recovery can wedge FOREVER on unrecoverable input.

Declarative extension point (CLAUDE.md): a language whose tree-sitter
grammar's error recovery loops instead of terminating joins by adding its
name here — every parse of it is then routed through the bounded
subprocess watchdog (:func:`bounded_parse_diagnostics`) instead of the
in-process parse. Healthy input is unaffected (the grammar parses it
normally); pathological input gets a typed :class:`ParseWatchdogTimeout`
instead of hanging the whole edit pipeline.

Measured behavior behind the founding member (Phase F census, where the
``cobol`` probe hung attempt #1 and is classified unresolvable only
because the census runs hang-proof per-language subprocesses):

* ``get_language("cobol")`` resolves instantly and parsing a REAL cobol
  program completes in 0.0s with zero error traits — the grammar is
  healthy on well-formed input;
* parsing INVALID cobol (e.g. the single line ``x``) NEVER returns: the
  grammar's error recovery wedges on unrecoverable input, hanging any
  in-process parse.

Members are canonical language names (:func:`canonical_language_name`).
"""

PARSE_WATCHDOG_TIMEOUT_SECONDS = 15.0
"""Hard deadline for one bounded subprocess parse.

Generous over the census's 10 s probe cap so a slow-but-terminating
recovery on a large healthy file is never misclassified as a wedge —
while still bounded, which is the point. Bounded is the contract;
precision is not.
"""


class ParseWatchdogTimeout(RuntimeError):
    """A :data:`PATHOLOGICAL_RECOVERY_GRAMMARS` parse exceeded its deadline.

    Raised by :func:`bounded_parse_diagnostics` when the subprocess parse
    did not terminate within the watchdog budget — the measured signature
    of the grammar's error recovery wedging on unrecoverable input. No
    honest diagnostics exist for such text, so the typed error is the
    verdict.

    A :class:`RuntimeError` subclass on purpose: the existing
    ``except (ValueError, RuntimeError, ImportError)`` consumers (e.g.
    :func:`fastedit.inference.ast_utils.get_ast_map_from_source`) treat a
    wedge exactly like an unresolvable grammar (empty AST map) with zero
    changes, while callers that want to distinguish "wedged parse" can
    catch this specifically.

    Attributes:
        language: The canonical language whose parse was bounded.
        timeout_seconds: The deadline that was exceeded.
    """

    def __init__(self, language: str, timeout_seconds: float) -> None:
        super().__init__(
            f"parse of '{language}' exceeded the {timeout_seconds}s parse "
            f"watchdog — the grammar's error recovery wedged on "
            f"unrecoverable input (see PATHOLOGICAL_RECOVERY_GRAMMARS); "
            f"no honest diagnostics exist for this text"
        )
        self.language = language
        self.timeout_seconds = timeout_seconds


# The watchdog worker: a tiny `python -c` program executed in a FRESH
# interpreter. It reads the source from a temp file (large texts never
# travel through argv or a pipe), parses with the UNGUARDED in-process
# path — never the routing wrapper, so a guarded grammar cannot spawn
# watchdogs recursively — and writes one JSON document (UTF-8 bytes) with
# the parse diagnostics. The parent imposes the hard deadline via
# subprocess.run(timeout=...) and kills the wedged child on expiry.
_WATCHDOG_WORKER_PROGRAM = """\
import json
import sys

import fastedit.data_gen.ast_analyzer as analyzer

with open(sys.argv[1], "rb") as fh:
    source = fh.read().decode("utf-8")
diagnostics = analyzer._parse_diagnostics_unguarded(source, sys.argv[2])
payload = {
    "errors": [list(span) for span in diagnostics.errors],
    "grammar_artifacts": [
        list(span) for span in diagnostics.grammar_artifacts
    ],
    "source": diagnostics.source,
}
sys.stdout.buffer.write(json.dumps(payload).encode("utf-8"))
"""


def bounded_parse_diagnostics(
    source: str,
    language: str,
    timeout_seconds: float = PARSE_WATCHDOG_TIMEOUT_SECONDS,
) -> ParseDiagnostics:
    """Parse *source* under a hard deadline in a killable subprocess.

    The bounded path for :data:`PATHOLOGICAL_RECOVERY_GRAMMARS` members —
    callers route ONLY listed grammars here (:func:`parse_diagnostics`
    and :func:`validate_parse` do exactly that). The parse runs in a
    fresh ``python -c`` worker that reads the text from a temp file and
    writes JSON diagnostics, so a grammar whose error recovery wedges on
    unrecoverable input is killed at the deadline instead of hanging the
    pipeline forever. The worker reuses
    :func:`_parse_diagnostics_unguarded`, so the subprocess verdict is
    byte-identical to the in-process one for healthy input — one code
    path, no drift.

    Returns:
        Real :class:`ParseDiagnostics` for input that parses within the
        deadline.

    Raises:
        ParseWatchdogTimeout: the deadline expired — the measured
            "error recovery wedged" verdict; no honest diagnostics exist.
        GrammarUnavailableError: the grammar itself cannot be resolved
            (checked in THIS process first, so the typed error survives).
        RuntimeError: the worker crashed or produced malformed output
            (its stderr tail is included) — a loud failure, never silence.
    """
    import os
    import subprocess
    import sys
    import tempfile

    # Resolve the grammar HERE first: an unresolvable guarded grammar
    # raises the typed GrammarUnavailableError from this process instead
    # of degrading into a generic worker-crash report.
    get_parser(language)

    fd, temp_path = tempfile.mkstemp(prefix="fastedit-watchdog-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(source.encode("utf-8"))
        # Guarantee the worker can import fastedit the same way this
        # process does, even when the parent's sys.path was patched at
        # runtime (e.g. a checkout run without an installed wheel).
        env = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[2])
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{package_root}{os.pathsep}{python_path}"
            if python_path else package_root
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-c", _WATCHDOG_WORKER_PROGRAM,
                    temp_path, language,
                ],
                capture_output=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParseWatchdogTimeout(language, timeout_seconds) from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"parse watchdog worker for '{language}' failed "
            f"(exit {proc.returncode}): {' | '.join(tail)}"
        )
    try:
        payload = json.loads(proc.stdout)
        errors = [tuple(span) for span in payload["errors"]]
        artifacts = [
            tuple(span) for span in payload["grammar_artifacts"]
        ]
        return ParseDiagnostics(
            errors=errors,
            is_valid=not errors,
            source=payload["source"],
            grammar_artifacts=tuple(artifacts),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"parse watchdog worker for '{language}' produced malformed "
            f"output: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Step G1a — known grammar artifacts (systematic grammar-defect traits)
# ---------------------------------------------------------------------------

_GRAMMAR_ARTIFACT_FILTERS: dict[str, tuple[tuple[str, str], ...]] = {
    "test": (("MISSING", "eof"),),
}
"""Per-language declarative filters for KNOWN grammar artifacts (G1a).

A grammar artifact is an error trait a grammar emits SYSTEMATICALLY — for
every parse of its own well-formed input — because of a defect in the
grammar, not in the text. The founding member is the language-pack
``test`` grammar (tree-sitter-language-pack 0.13.0): its ``input`` region
greedily consumes the record separator, so EVERY complete test record
ends in one zero-width ``MISSING`` trait at EOF (recorded fail-loud with
exact errors in tests/golden/test/manifest.json, pinned by the census
test so an upstream fix lifts the exclusion loudly).

Each value is a tuple of ``(kind, locator)`` signatures; a trait is an
artifact when its kind matches AND the locator matches. Locators are the
declarative matchers in ``_GRAMMAR_ARTIFACT_LOCATORS`` — ``"eof"`` means
a zero-width trait pinned at the very end of the normalized source. A
signature must stay NARROW: it filters a systematic emission, never a
genuine defect (a header-only ``.test`` file is an outright ERROR node
and is deliberately NOT matched).

Why matching artifacts are REMOVED from the trait lists (visible only in
``ParseDiagnostics.grammar_artifacts``): the relative parse rule
(:func:`fastedit.inference.chunked_merge.merged_is_acceptable`) compares
merged vs original traits as a multiset keyed by (kind, containing line).
A systematic artifact is a constant of the GRAMMAR, not of the TEXT — it
carries no edit information and can never distinguish a faithful merge
from a corrupted one. Worse, its position tracks EOF, so any edit that
changes the file's length moves its containing line: left inside the
trait lists it would flip between "inherited" and "new" — phantom
rejections of faithful merges — or, when a genuinely NEW defect lands on
the file's last line, be satisfied by it and make the rule blind to real
breakage. Removing the artifact from BOTH the original's and the merged's
trait lists makes it vanish from the rule's accounting entirely — exactly
"treated as inherited (never new)" — while every REAL trait keeps its
normal behavior (a NEW real error in a .test file still rejects the
merge). An artifact-only parse is therefore VALID: the grammar, not the
text, is what fails to parse cleanly. The removed traits stay tagged in
``grammar_artifacts`` so what was filtered remains introspectable and
fail-loud rather than silent.
"""

# Declarative locators for ``_GRAMMAR_ARTIFACT_FILTERS`` signatures:
# (start_byte, end_byte, source_byte_length) -> bool. Adding a new artifact
# shape means adding a locator here and a signature row — no new branches.
_GRAMMAR_ARTIFACT_LOCATORS: dict[str, object] = {
    "eof": lambda start, end, source_len: start == end == source_len,
}


def _is_known_grammar_artifact(
    span: tuple[int, int, str],
    source_len: int,
    filters: tuple[tuple[str, str], ...],
) -> bool:
    """True iff *span* matches one of the declared artifact signatures.

    A signature is ``(kind, locator)``: the trait's kind must equal the
    signature's kind and the named locator must accept the span's
    position. Unknown locator names never match (a typo'd signature
    filters nothing — fail-silent-proof by construction, and pinned by
    the declarative-table test).
    """
    start, end, kind = span
    for want_kind, locator in filters:
        if kind != want_kind:
            continue
        predicate = _GRAMMAR_ARTIFACT_LOCATORS.get(locator)
        if predicate is not None and predicate(start, end, source_len):
            return True
    return False


def parse_diagnostics(source: str, language: str) -> ParseDiagnostics:
    """Tree-sitter parse producing ordered error traits (Step A2, req. 9).

    Where :func:`validate_parse` answers the ABSOLUTE question "does this
    text parse?", this function answers the trait question "WHAT does the
    parser object to, and where?" — the input's structural defects as
    first-class, comparable data. The relative validation rule
    (:func:`fastedit.inference.chunked_merge.merged_is_acceptable`)
    compares a merged output's traits against the original's so a
    pre-existing defect can be preserved (EDIT-NOT-CORRECT) while new
    breakage is rejected.

    Routing (Step G1a):

      * A language in :data:`PATHOLOGICAL_RECOVERY_GRAMMARS` parses under
        the subprocess watchdog (:func:`bounded_parse_diagnostics`) —
        same return value, bounded runtime; a wedged parse raises the
        typed :class:`ParseWatchdogTimeout` instead of hanging.
      * Traits matching a declared known grammar artifact
        (:data:`_GRAMMAR_ARTIFACT_FILTERS`) are removed from ``errors``
        and tagged in ``ParseDiagnostics.grammar_artifacts``.
      * Every other language keeps the in-process path unchanged.

    Trait collection rules:

      * **TOPMOST traits only** — an ERROR or MISSING node is recorded
        once and NOT descended into. Error recovery nests further ERROR /
        MISSING nodes inside an outer one; recording the outermost span
        keeps one broken construct = one trait, which is the granularity
        the relative rule (and its unit tests) reason about.
      * **Document order** — the pre-order walk yields spans ascending by
        ``start_byte``.

    Normalization contract (mind the pipeline's CR handling): the source
    is parsed exactly the way every other AST consumer in the pipeline
    sees it — after :func:`fastedit.split_join.normalize_bare_cr_for_ast`
    swaps each bare CR for LF. That substitution is 1 byte for 1 byte at
    identical offsets, so the returned spans index the SAME byte positions
    in the caller's raw text; but because a bare CR changes where lines
    break, consumers that need the text AROUND a span (line text for
    defect identity, human-readable messages) MUST read it from
    ``ParseDiagnostics.source`` — the normalized copy the spans refer to —
    never by re-slicing a differently normalized copy. No offset mapping
    is ever needed: positions are identical by construction.
    """
    canonical = canonical_language_name(language)
    if canonical in PATHOLOGICAL_RECOVERY_GRAMMARS:
        return bounded_parse_diagnostics(source, canonical)
    return _parse_diagnostics_unguarded(source, language)


def _parse_diagnostics_unguarded(
    source: str, language: str,
) -> ParseDiagnostics:
    """The in-process parse behind :func:`parse_diagnostics` — no watchdog.

    Contract (traits, normalization, artifact filtering) is documented on
    :func:`parse_diagnostics`. This is ALSO the worker-side entry point
    for :func:`bounded_parse_diagnostics`: the subprocess calls THIS, so
    the guarded path can never spawn watchdogs recursively.
    """
    from ..split_join import normalize_bare_cr_for_ast

    canonical = canonical_language_name(language)
    normalized = normalize_bare_cr_for_ast(source)
    tree = parse_code(normalized, language)
    root = tree.root_node
    errors: list[tuple[int, int, str]] = []

    def _walk(node: tree_sitter.Node) -> None:
        # MISSING before ERROR: a missing token is reported as MISSING even
        # when the parser also wrapped the region in an ERROR ancestor —
        # the topmost check below keeps one construct = one trait.
        if node.is_missing:
            errors.append((node.start_byte, node.end_byte, "MISSING"))
            return
        if node.is_error:
            errors.append((node.start_byte, node.end_byte, "ERROR"))
            return
        for child in node.children:
            _walk(child)

    _walk(root)
    if not errors and canonical in _COLON_SUITE_LANGUAGES:
        # tree-sitter-python silently recovers vanished suites (C2): its
        # external INDENT/DEDENT machinery re-anchors a compound statement
        # whose body is missing without emitting ERROR or MISSING nodes, so
        # the walk above reports a clean parse for a file CPython refuses
        # with ``SyntaxError: expected an indented block``. The stress tier
        # hit exactly that: a merge emitted the guard line without
        # re-indenting the body and the parse gate called the corrupted file
        # valid. The tokenizer-level scan below is the missing judge — it is
        # only consulted when tree-sitter found nothing (the expensive case
        # is the clean file; a file tree-sitter already rejects is invalid
        # regardless).
        errors = _colon_suite_opener_violations(normalized)

    # Step G1a: known grammar artifacts are removed from the trait list
    # (tagged in .grammar_artifacts) — see _GRAMMAR_ARTIFACT_FILTERS for
    # why removal-from-both-sides is the sound relative-rule integration.
    artifacts: tuple[tuple[int, int, str], ...] = ()
    filters = _GRAMMAR_ARTIFACT_FILTERS.get(canonical)
    if filters:
        kept: list[tuple[int, int, str]] = []
        matched: list[tuple[int, int, str]] = []
        source_len = len(normalized.encode("utf-8"))
        for span in errors:
            if _is_known_grammar_artifact(span, source_len, filters):
                matched.append(span)
            else:
                kept.append(span)
        errors = kept
        artifacts = tuple(matched)
    return ParseDiagnostics(
        errors=errors, is_valid=not errors, source=normalized,
        grammar_artifacts=artifacts,
    )


_COLON_SUITE_LANGUAGES = frozenset({"python"})
"""Languages whose compound statements are colon-headed indentation suites.

Declarative extension point (CLAUDE.md): a language joins the suite-opener
judgment by adding its name here. Ruby/Elixir style ``end``-blocked grammars
never qualify — their suites cannot vanish silently the way an indentation
suite's can.
"""


def _colon_suite_opener_violations(source: str) -> list[tuple[int, int, str]]:
    """Indentation-suite traits the tree-sitter grammar silently recovers.

    Walks the source with the stdlib :mod:`tokenize` stream (exact string,
    comment and indentation handling; O(1) memory) and flags every compound
    statement header — a logical line whose LAST significant token is a
    top-level ``:`` — whose suite never opens: the next token is not an
    INDENT (a peer or shallower statement follows, or the file ends). Each
    violation is reported as an ``INDENT`` trait span over the header's
    colon, so the relative parse rule treats it like any other defect
    trait: inherited when the original carried it, a regression when a
    merge introduces it.

    False-positive analysis: an empty suite is a SyntaxError in CPython
    without exception — ``pass`` (or any statement) is always required — so
    a flagged file is genuinely broken. Dict/annotation/slice/lambda colons
    never end a logical line at bracket depth 0 (a significant token always
    follows them), inline suites (``if x: return``) do not end in a colon,
    and multi-line strings/bracket continuations are handled by the
    tokenizer and the depth counter respectively.
    """
    import io
    import tokenize

    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def _byte_offset(row: int, col: int) -> int:
        """(1-indexed row, 0-indexed col) → byte offset in *source*."""
        return line_starts[row - 1] + col

    significant_types = (
        tokenize.OP, tokenize.NAME, tokenize.STRING, tokenize.NUMBER,
        tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END,
    )
    violations: list[tuple[int, int, str]] = []
    depth = 0
    # A top-level ":" seen in the current logical line that no significant
    # token has followed yet — the candidate compound-statement header.
    colon_candidate: tuple[int, int] | None = None
    # A CONFIRMED header (its NEWLINE arrived) still waiting for its suite.
    open_header: tuple[int, int] | None = None

    def _flag() -> list[tuple[int, int, str]]:
        assert open_header is not None
        return [(open_header[0], open_header[1], "INDENT")]

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.NEWLINE:
                # The logical line ended. A surviving candidate is a header.
                if colon_candidate is not None:
                    open_header = colon_candidate
                    colon_candidate = None
                continue
            if tok.type in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING):
                continue
            if tok.type == tokenize.INDENT:
                # The suite opened — any waiting header is satisfied.
                colon_candidate = None
                open_header = None
                continue
            if tok.type == tokenize.DEDENT:
                continue  # judged when the next significant token arrives
            if tok.type == tokenize.ENDMARKER:
                if open_header is not None:
                    violations.extend(_flag())
                continue
            if tok.type not in significant_types:
                continue
            # A significant token: it cancels a colon candidate (the colon
            # did not close the logical line — a dict pair, lambda, inline
            # suite, ...) and satisfies-or-condemns a waiting header.
            colon_candidate = None
            if open_header is not None:
                violations.extend(_flag())
                open_header = None
            if tok.type == tokenize.OP:
                if tok.string in "([{":
                    depth += 1
                elif tok.string in ")]}":
                    depth -= 1
                elif tok.string == ":" and depth == 0:
                    colon_candidate = (
                        _byte_offset(*tok.start), _byte_offset(*tok.end),
                    )
    except (SyntaxError, IndentationError, tokenize.TokenError) as exc:
        # The tokenizer itself refused the text — a real lexical defect the
        # tree-sitter walk did not see. Report it on the offending line.
        row = getattr(exc, "lineno", 1) or 1
        start = line_starts[min(row, len(line_starts) - 1)]
        return [(start, start, "INDENT")]
    if open_header is not None:
        violations.extend(_flag())
    return violations


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



# --- content-based text/binary detection (never trusts the extension) ---











