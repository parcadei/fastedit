"""Cross-file symbol relocation: ``move_to_file``.

Milestone 4 (VAL-M4-001..003). Relocates a function / class / method from
``from_file`` into ``to_file`` and rewrites every dependent file's import
statement so the symbol still resolves.

Design:
    * Source span detection reuses :mod:`.ast_utils` (``get_ast_map`` +
      ``_resolve_symbol``) — the same AST pass that backs ``delete_symbol``
      and ``move_symbol``. No duplication.
    * Consumer discovery runs through ``tldr references <symbol> <root>``
      at workspace scope. On tldr's AST-native langs (python, typescript,
      go, rust) we filter by ``kind == "import"``. On the other 9 langs
      tldr emits ``kind == "other"``, so we fall back to a text-level
      heuristic — a per-language regex that recognizes that language's
      import syntax — applied to the reference's ``context`` line.
    * Writes are staged and applied atomically at the end so a crash
      mid-plan leaves the tree coherent (``_atomic_write`` snapshots into
      ``BackupStore`` for ``fast_undo``).

Milestone 4.7: per-language import rewriters. Python + JS/TS still use
the hand-tuned rewriters below; java, kotlin, go, rust, swift, c#, ruby,
php, scala, elixir, lua, c, and cpp each get a thin rewriter that:

    (1) derives the new module/namespace path from ``to_file`` relative
        to ``project_root`` (dotted path for JVM-family langs, backslash
        path for PHP, filesystem path for ruby/lua/go/C-family, crate::
        path for rust), and
    (2) performs a substring replacement inside the already-identified
        import line, swapping the old module segment for the new one.

Rust's ``use`` trees add one extra case: a braced multi-import
``use crate::foo::{Bar, Baz};`` where only ``Bar`` is being moved — we
split into ``use crate::foo::{Baz};`` plus a fresh
``use crate::new::Bar;``. Nested braces / wildcard imports / renamed
items (``use foo::Bar as Renamed;``) are flagged as manual review
rather than attempting a half-handled rewrite that could break valid
code.

Same-file moves (``from_file == to_file``) are rejected with a hint to
use ``fast_move`` / the ``move`` CLI subcommand.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE, detect_language
from .ast_utils import (
    _qualified_symbol_names,
    _resolve_symbol,
    get_ast_map,
)

# ---------------------------------------------------------------------------
# Supported extensions and language families.
# ---------------------------------------------------------------------------
#
# Each language family has:
#   * A set of file extensions that dispatch to it.
#   * A ``_import_specifier_for_<family>`` function that converts a
#     ``to_file`` path into the string used in that language's import
#     statement (dotted path / backslash path / relative fs path / etc).
#   * A ``_rewrite_<family>_import_line`` function that takes an
#     already-identified import line and produces the rewritten text.
#   * A ``_looks_like_import_line_<family>`` text matcher used on
#     non-AST-native langs where tldr emits ``kind="other"`` for every
#     reference, including imports.
#
# Langs that share syntax (java/kotlin/scala — all dotted, with minor
# trailing-punctuation differences) share a rewriter; the dispatcher
# selects the right one per extension.

_PY_EXTS = frozenset({".py"})
_TS_EXTS = frozenset({".ts", ".tsx", ".js", ".jsx"})
_JAVA_EXTS = frozenset({".java"})
_KOTLIN_EXTS = frozenset({".kt", ".kts"})
_SCALA_EXTS = frozenset({".scala"})
_CSHARP_EXTS = frozenset({".cs"})
_PHP_EXTS = frozenset({".php"})
_GO_EXTS = frozenset({".go"})
_RUST_EXTS = frozenset({".rs"})
_SWIFT_EXTS = frozenset({".swift"})
_RUBY_EXTS = frozenset({".rb"})
_ELIXIR_EXTS = frozenset({".ex", ".exs"})
_LUA_EXTS = frozenset({".lua"})
_C_EXTS = frozenset({".c", ".h"})
_CPP_EXTS = frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh"})

_SUPPORTED_EXTS = (
    _PY_EXTS | _TS_EXTS | _JAVA_EXTS | _KOTLIN_EXTS | _SCALA_EXTS
    | _CSHARP_EXTS | _PHP_EXTS | _GO_EXTS | _RUST_EXTS | _SWIFT_EXTS
    | _RUBY_EXTS | _ELIXIR_EXTS | _LUA_EXTS | _C_EXTS | _CPP_EXTS
)


def _ext_family(ext: str) -> str | None:
    """Return the rewriter-family name for a file extension, or None."""
    if ext in _PY_EXTS:
        return "python"
    if ext in _TS_EXTS:
        return "ts"
    if ext in _JAVA_EXTS:
        return "java"
    if ext in _KOTLIN_EXTS:
        return "kotlin"
    if ext in _SCALA_EXTS:
        return "scala"
    if ext in _CSHARP_EXTS:
        return "csharp"
    if ext in _PHP_EXTS:
        return "php"
    if ext in _GO_EXTS:
        return "go"
    if ext in _RUST_EXTS:
        return "rust"
    if ext in _SWIFT_EXTS:
        return "swift"
    if ext in _RUBY_EXTS:
        return "ruby"
    if ext in _ELIXIR_EXTS:
        return "elixir"
    if ext in _LUA_EXTS:
        return "lua"
    if ext in _C_EXTS:
        return "c"
    if ext in _CPP_EXTS:
        return "cpp"
    return None


@dataclass
class ImportRewrite:
    """One file's worth of import changes for a move operation."""

    file: str
    line: int
    old_import: str
    new_import: str
    # When True, the old line had other symbols on it and we split the
    # moved name into a fresh line targeting the new module. The old
    # line is preserved with the moved name stripped.
    split: bool = False
    # True when the rewrite could not be applied automatically and is
    # surfaced as a manual-review hint.
    manual_review: bool = False
    reason: str = ""


@dataclass
class MoveToFilePlan:
    """Result of planning and/or executing a cross-file move."""

    symbol: str
    from_file: str
    to_file: str
    # 1-indexed inclusive span of the extracted symbol in the ORIGINAL
    # source file (pre-removal). Useful for CLI dry-run display.
    source_span: tuple[int, int]
    # 1-indexed line in ``to_file`` AFTER which the symbol will be
    # inserted (0 means "at top of file"; file-end insertion uses the
    # pre-insertion line count).
    insertion_line: int
    # Every caller-file rewrite we planned. Flagged entries with
    # ``manual_review=True`` were not applied automatically.
    import_rewrites: list[dict] = field(default_factory=list)
    # Files we identified as importers but chose not to rewrite (wildcard
    # imports, re-exports, etc.). Paired with a short reason string.
    skipped_importers: list[dict] = field(default_factory=list)
    # True when the plan was applied (files written), False on dry-run.
    applied: bool = False
    # Free-form human message suitable for CLI/MCP output.
    message: str = ""


# ---------------------------------------------------------------------------
# tldr helpers — local to this module to keep coupling minimal.
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` object in ``text``.

    Mirrors ``rename._extract_json_object`` — tldr may emit trailing
    diagnostic lines on stdout.
    """
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


# Per-language import-line recognizers. Used to filter tldr refs when the
# ``--kinds import`` upstream filter doesn't fire — i.e., on non-AST-
# native langs where every reference is ``kind="other"``. These regexes
# only have to match the *import line text*, not the full import
# grammar; they run against the line's stripped content.
_IMPORT_LINE_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^\s*(from\s+\S+\s+import\b|import\s+\S)"),
    "ts": re.compile(r"^\s*import\b"),
    "java": re.compile(r"^\s*import\s+[\w.]+;?\s*$"),
    "kotlin": re.compile(r"^\s*import\s+[\w.]+(?:\s+as\s+\w+)?\s*$"),
    "scala": re.compile(r"^\s*import\s+[\w.]+(?:\.\{[^}]+\})?\s*$"),
    "csharp": re.compile(r"^\s*using\s+[\w.]+\s*;"),
    "php": re.compile(r"^\s*use\s+[\w\\]+(?:\s+as\s+\w+)?\s*;"),
    "go": re.compile(r"^\s*(?:import\s+)?\"[^\"]+\"\s*$"),
    "rust": re.compile(r"^\s*(?:pub\s+)?use\s+[\w:{},\s\*]+;"),
    "swift": re.compile(r"^\s*import\s+[\w.]+\s*$"),
    "ruby": re.compile(r"^\s*(require|require_relative|load)\s+['\"][^'\"]+['\"]"),
    "elixir": re.compile(r"^\s*(alias|import|require|use)\s+[\w.]+"),
    "lua": re.compile(r"^\s*(?:local\s+\w+\s*=\s*)?require\s*[\(\"']"),
    "c": re.compile(r"^\s*#include\s+[<\"][^>\"]+[>\"]"),
    "cpp": re.compile(r"^\s*#include\s+[<\"][^>\"]+[>\"]"),
}


def _is_import_line_for(family: str, line: str) -> bool:
    """Return True when ``line`` looks like an import statement in ``family``.

    Used as a text-level fallback when tldr emits ``kind="other"`` for
    every reference (non-AST-native langs). The match is intentionally
    loose — we already know the ref points at the symbol in question,
    so we just need to confirm we're looking at an import statement
    and not a usage site.
    """
    pattern = _IMPORT_LINE_PATTERNS.get(family)
    if pattern is None:
        return False
    return bool(pattern.match(line))


def _run_tldr_references_all(symbol: str, root: Path) -> list[dict]:
    """Return every tldr reference at workspace scope.

    Skips the ``--kinds import`` filter because on non-AST-native langs
    tldr emits ``kind="other"`` for imports too; filtering upstream
    would drop real hits. Callers filter client-side via
    :func:`_is_import_line_for` or by checking ``kind == "import"`` on
    AST-native langs.
    """
    cmd = [
        "tldr", "references", symbol, str(root),
        "--format", "json",
        "--scope", "workspace",
        "--min-confidence", "0.9",
        "--limit", "10000",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    payload = _extract_json_object(proc.stdout or "")
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    refs = data.get("references") or []
    return [r for r in refs if isinstance(r, dict)]


def _filter_refs_to_imports(
    refs: list[dict], family: str,
) -> list[dict]:
    """Return only the refs whose line looks like an import in ``family``.

    AST-native langs (python/ts/go/rust) get tldr's semantic ``kind``
    checked first; falls back to text match for consistency.
    """
    out: list[dict] = []
    for r in refs:
        if r.get("kind") == "import":
            out.append(r)
            continue
        # Text-level match on the ref's context line.
        ctx = r.get("context")
        if isinstance(ctx, str) and _is_import_line_for(family, ctx):
            out.append(r)
    return out


# Languages where the import statement names the FILE, not the symbol:
#   * Ruby:  require_relative "a"  — path string
#   * Go:    import "./a"          — path string (when local)
#   * Lua:   require "a"           — path string
#   * C/C++: #include "a.h"        — header path
#
# tldr's ``references <symbol>`` can't find these lines because the
# symbol name doesn't appear in them. We instead scan project files
# for lines mentioning the source file's basename, within an import-
# shaped line.
_FILE_NAMED_IMPORT_FAMILIES = frozenset({"ruby", "go", "lua", "c", "cpp"})


def _scan_file_named_imports(
    family: str,
    from_file: Path,
    root: Path,
) -> list[dict]:
    """Scan ``root`` for import lines referencing ``from_file`` by path.

    Walks every file under ``root`` matching the family's extension set
    and returns synthetic ref dicts (``file``, ``line``, ``context``,
    ``kind="import"``) for each import line that names the source
    file's basename or stem.

    Returns an empty list on read failures. This is intentionally
    conservative — we only match the basename (not arbitrary path
    substrings), so a rename of ``helpers/a.c`` to ``b.c`` won't touch
    an unrelated ``#include "somewhere/a.c"`` in a different dir.
    """
    exts_by_family = {
        "ruby": _RUBY_EXTS,
        "go": _GO_EXTS,
        "lua": _LUA_EXTS,
        "c": _C_EXTS | _CPP_EXTS,   # headers often included cross-family
        "cpp": _C_EXTS | _CPP_EXTS,
    }
    exts = exts_by_family.get(family)
    if exts is None:
        return []

    stem = from_file.stem
    basename = from_file.name
    # Match any import line that contains the stem or basename as a
    # quoted / bracketed substring. The per-family regex below is
    # intentionally loose — we're already filtering by ``_is_import_
    # line_for``, which catches the import keyword.
    # Match quoted or bracketed path ending in the stem with optional
    # extension. We build the regex from three pieces to avoid mixing
    # ``"`` and ``\'`` inside a single raw string.
    _open = '[\"\'<]'
    _close = '[\"\'>]'
    _prefix = '(?:[^\"\'<>]*/)?'
    needle_re = re.compile(
        _open + _prefix + re.escape(stem) + r"(?:\.[A-Za-z0-9]+)?" + _close
    )

    out: list[dict] = []
    try:
        from_resolved = from_file.resolve()
    except OSError:
        from_resolved = from_file

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in exts:
            continue
        try:
            if path.resolve() == from_resolved:
                continue
        except OSError:
            pass
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not _is_import_line_for(family, stripped):
                continue
            if not needle_re.search(line):
                continue
            out.append({
                "file": str(path),
                "line": i,
                "context": line,
                "kind": "import",
            })
    return out


# ---------------------------------------------------------------------------
# Module-path conversion — per language.
# ---------------------------------------------------------------------------


def _dotted_module_for(file_path: Path, project_root: Path) -> str:
    """Return ``pkg.sub.mod`` for ``<root>/pkg/sub/mod.ext``.

    Common helper for python/java/kotlin/scala/c#. The caller may want
    to drop the module *or* retain it depending on the language's
    convention; callers handle trailing semantics.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.stem
    parts = list(rel.with_suffix("").parts)
    # Drop trailing ``__init__`` for Python packages.
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


def _python_module_for(file_path: Path, project_root: Path) -> str:
    """Python dotted module path. See :func:`_dotted_module_for`."""
    return _dotted_module_for(file_path, project_root)


def _java_module_for(file_path: Path, project_root: Path) -> str:
    """Java ``pkg.sub.Class`` path.

    Java imports point at a specific class, so the path is the package
    (directories) plus the class name (file stem). The file stem is
    preserved case-sensitively — Java conventions capitalize public
    class names, which must match the file name.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.stem
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts) if parts else file_path.stem


def _kotlin_module_for(file_path: Path, project_root: Path) -> str:
    """Kotlin ``pkg.sub.Symbol`` — same dotted form as Java, but Kotlin
    imports name the symbol (not the file), so we return the package
    path plus a trailing ``.<symbol>`` appended at the call site.

    This helper returns just the package prefix; the caller appends
    ``.<symbol>`` so the same helper works for moving a
    top-level-in-package function (dotted path = pkg + symbol) or a
    class (pkg + class-name).
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return ""
    # Kotlin imports use the directory path as the package, not the
    # file stem. Files can declare any package via ``package ...``
    # header; we default to the directory path which is the most
    # common convention.
    parts = list(rel.parent.parts)
    return ".".join(parts)


def _scala_module_for(file_path: Path, project_root: Path) -> str:
    """Scala ``pkg.sub`` — same as Kotlin: package path only, caller
    appends the symbol.
    """
    return _kotlin_module_for(file_path, project_root)


def _csharp_namespace_for(file_path: Path, project_root: Path) -> str:
    """C# ``Namespace.Sub`` — directory-path in PascalCase.

    Real C# projects derive namespaces from .csproj settings; we can't
    see those, so we use the directory path as-is. Callers can always
    edit the derived path after the rewrite — this is a best-effort
    scaffold.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return ""
    parts = list(rel.parent.parts)
    return ".".join(parts)


def _php_namespace_for(file_path: Path, project_root: Path) -> str:
    """PHP ``Foo\\Bar`` — backslash-separated namespace path.

    PSR-4 autoloading maps ``Foo\\Bar\\Baz`` to ``Foo/Bar/Baz.php``, so
    we use the directory path with backslash separators. The class
    name (file stem) is appended by the caller.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return ""
    parts = list(rel.parent.parts)
    return "\\".join(parts)


def _go_import_path_for(file_path: Path, project_root: Path) -> str:
    """Go ``module/path/subdir`` — POSIX-path of the containing directory.

    Real Go modules have an import prefix declared in go.mod. We can't
    see that, so we return the directory path relative to project root
    (a common case for local-only modules and the substring the user
    needs to replace in their existing import string).
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.parent.name
    parts = list(rel.parent.parts)
    return "/".join(parts) if parts else "."


def _rust_use_path_for(file_path: Path, project_root: Path) -> str:
    """Rust ``crate::pkg::sub`` — module path for a ``use`` statement.

    Rust modules nest via ``pub mod foo;`` declarations and
    ``foo/mod.rs`` files. The reliable approximation is a path-based
    one: the directory portion relative to ``src/`` becomes
    ``crate::<parts>`` and the file stem becomes the module name.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return "crate::" + file_path.stem
    parts = list(rel.with_suffix("").parts)
    # Drop a leading ``src`` directory — by convention that's the crate
    # root, not part of the module path.
    if parts and parts[0] == "src":
        parts = parts[1:]
    # ``mod.rs`` files don't contribute a segment of their own.
    if parts and parts[-1] == "mod":
        parts = parts[:-1]
    return "crate::" + "::".join(parts) if parts else "crate"


def _swift_module_for(file_path: Path, project_root: Path) -> str:
    """Swift ``import Foo`` — just the top-level module name.

    Swift modules map 1:1 to build targets, not filesystem paths. Best
    approximation: the top-level directory under ``project_root``.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.parent.name
    parts = list(rel.parent.parts)
    return parts[0] if parts else file_path.stem


def _ruby_require_path_for(
    from_file: Path, to_file: Path, project_root: Path,
) -> str:
    """Ruby ``require_relative`` path — relative path, no extension.

    Ruby's ``require`` uses $LOAD_PATH; ``require_relative`` uses a path
    relative to the importing file. We generate the require_relative
    form because it's deterministic from filesystem layout.
    """
    try:
        rel = os.path.relpath(
            str(to_file.resolve().with_suffix("")),
            start=str(from_file.resolve().parent),
        )
    except (ValueError, OSError):
        return to_file.stem
    return rel.replace(os.sep, "/")


def _elixir_module_for(file_path: Path, project_root: Path) -> str:
    """Elixir ``Foo.Bar`` — PascalCase directory path.

    Convention: ``lib/foo/bar.ex`` defines ``Foo.Bar``. We convert
    directory segments to PascalCase.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.stem.capitalize()
    parts = list(rel.with_suffix("").parts)
    # Drop leading ``lib`` directory by convention.
    if parts and parts[0] == "lib":
        parts = parts[1:]

    def _pascal(seg: str) -> str:
        return "".join(part.capitalize() for part in seg.split("_"))

    return ".".join(_pascal(p) for p in parts) or file_path.stem.capitalize()


def _lua_require_path_for(
    from_file: Path, to_file: Path, project_root: Path,
) -> str:
    """Lua ``require "foo.bar"`` path — dotted or slashed form.

    Lua's ``require`` uses package paths resolved via ``package.path``.
    We return the project-root-relative path with dots so it matches
    the default ``?.lua`` resolution.
    """
    try:
        rel = to_file.resolve().relative_to(project_root.resolve())
    except ValueError:
        return to_file.stem
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts) if parts else to_file.stem


def _c_include_path_for(
    from_file: Path, to_file: Path, project_root: Path,
) -> str:
    """C/C++ ``#include "path.h"`` path — relative to the caller.

    ``#include <...>`` uses system search paths we can't see. Local
    project includes use ``"..."`` with paths relative to the caller
    (or to a standard include dir). Moving a function normally moves
    its declaration across .h files — which we can't infer with just
    the .c/.cpp source. We return a caller-relative path for the
    target file and let the caller decide whether to apply or flag.
    """
    try:
        rel = os.path.relpath(
            str(to_file.resolve()),
            start=str(from_file.resolve().parent),
        )
    except (ValueError, OSError):
        return to_file.name
    return rel.replace(os.sep, "/")


def _ts_relative_path(from_file: Path, to_file: Path) -> str:
    """Return the TypeScript-style relative module specifier.

    ``import { x } from "<here>"`` — the returned string has no extension
    and is prefixed with ``./`` or ``../`` as needed. On any failure we
    return ``"./" + to_file.stem`` as a fallback (caller's same-directory
    case).
    """
    try:
        rel = os.path.relpath(
            str(to_file.resolve().with_suffix("")),
            start=str(from_file.resolve().parent),
        )
    except (ValueError, OSError):
        return "./" + to_file.stem
    # Normalize Windows-style separators; TS import specifiers are POSIX.
    rel = rel.replace(os.sep, "/")
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


# ---------------------------------------------------------------------------
# Import-line rewriters (per language family).
# ---------------------------------------------------------------------------


def _rewrite_python_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Rewrite a single Python ``from <mod> import ...`` line.

    Returns ``(new_text, split, reason)`` where:
        new_text   — replacement text for the line (including the newline
                     if the original had one). When multiple symbols share
                     the line we split into two lines. ``None`` means the
                     line could not be rewritten (wildcard / aliased /
                     malformed) and the caller should surface it as a
                     manual-review entry.
        split      — True when we emitted an additional line for the
                     moved symbol (multi-name import split).
        reason     — Explanation when ``new_text`` is ``None``.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"

    # Wildcard: nothing to rewrite, leave for manual review.
    if "import *" in stripped:
        return None, False, "wildcard import"

    # Only handle ``from X import Y[, Z]`` — not plain ``import X``,
    # which doesn't refer to the moved symbol anyway (tldr wouldn't
    # flag it with --kinds import pointing at the symbol).
    if not stripped.lstrip().startswith("from "):
        return None, False, "not a from-import"

    try:
        head, tail = stripped.split(" import ", 1)
    except ValueError:
        return None, False, "unparsable import"

    # Preserve the original leading whitespace (the import might be
    # nested inside a function — unusual but legal).
    indent_len = len(head) - len(head.lstrip())
    indent = head[:indent_len]

    # Multi-line parenthesized imports are out of scope for v1 — the
    # tldr reference points at the symbol's column but not the line
    # span, and splitting across lines correctly requires parsing the
    # paren group. Flag for manual review.
    if "(" in tail and ")" not in tail:
        return None, False, "multi-line parenthesized import"

    # Strip surrounding parens if present on a single line.
    inner = tail.strip()
    had_parens = False
    if inner.startswith("(") and inner.endswith(")"):
        had_parens = True
        inner = inner[1:-1].strip()

    # Split names. Each item may be ``name`` or ``name as alias``.
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if not parts:
        return None, False, "empty import list"

    def _base_name(item: str) -> str:
        return item.split(" as ", 1)[0].strip()

    matching = [p for p in parts if _base_name(p) == symbol]
    if not matching:
        # tldr pointed at this line but the symbol isn't in the import
        # list — probably an aliased rename we can't pattern-match. Let
        # the user handle it.
        return None, False, "symbol not directly named in import list"

    remaining = [p for p in parts if _base_name(p) != symbol]
    moved_piece = matching[0]  # preserve ``foo as bar`` if present

    new_line_for_symbol = f"{indent}from {new_module} import {moved_piece}{newline}"

    if not remaining:
        # The whole line was for this symbol — just rewrite in place.
        return new_line_for_symbol, False, ""

    # Multi-name import — emit both lines: the original with the moved
    # name stripped, plus a fresh line for the new module.
    joined = ", ".join(remaining)
    if had_parens:
        joined = f"({joined})"
    remaining_line = f"{indent}{head.lstrip()} import {joined}{newline}"
    # Re-attach indent properly (we stripped above for the lstrip call).
    remaining_line = f"{indent}from {head.lstrip()[5:]} import {joined}{newline}"
    return remaining_line + new_line_for_symbol, True, ""


def _rewrite_ts_import_line(
    line: str, symbol: str, new_specifier: str,
) -> tuple[str | None, bool, str]:
    """Rewrite a single TS/JS ``import { ... } from "..."`` line.

    Same return contract as :func:`_rewrite_python_import_line`.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"

    s = stripped.strip()

    # Out of scope: side-effect imports, default imports, namespace
    # imports, dynamic imports.
    if s.startswith("import *"):
        return None, False, "namespace import"
    if not s.startswith("import "):
        return None, False, "not a top-level import"
    if "{" not in s or "}" not in s:
        return None, False, "no named-import brace"

    brace_start = s.index("{")
    brace_end = s.index("}")
    if brace_end < brace_start:
        return None, False, "malformed braces"

    names_raw = s[brace_start + 1 : brace_end]
    names = [n.strip() for n in names_raw.split(",") if n.strip()]

    def _base_name(item: str) -> str:
        # ``foo as bar`` — rare in named imports but legal.
        return item.split(" as ", 1)[0].strip()

    matching = [n for n in names if _base_name(n) == symbol]
    if not matching:
        return None, False, "symbol not in named imports"

    remaining = [n for n in names if _base_name(n) != symbol]
    moved_piece = matching[0]

    # Preserve leading whitespace.
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    # Preserve the tail after the closing brace (``from "..."``) so we
    # can reuse it on the residual line when we split.
    tail_after_brace = s[brace_end + 1:]  # includes ``from "<spec>";``

    new_symbol_line = (
        f'{indent}import {{ {moved_piece} }} from "{new_specifier}";{newline}'
    )

    if not remaining:
        return new_symbol_line, False, ""

    # Split: keep the remaining named imports pointing at the original
    # source, and add a fresh line for the moved symbol.
    residual = (
        f"{indent}import {{ {', '.join(remaining)} }}{tail_after_brace}{newline}"
    )
    return residual + new_symbol_line, True, ""


def _rewrite_dotted_import_line(
    line: str,
    symbol: str,
    new_module: str,
    *,
    keyword: str,
    trailing_semi: bool,
    path_sep: str = ".",
) -> tuple[str | None, bool, str]:
    """Rewrite a Java/Kotlin/Scala/C#-style ``<keyword> pkg.sub.Sym;`` line.

    Java / Kotlin / Scala / C# all share the same basic shape:

        import pkg.sub.Symbol[;]        # java/kotlin/scala
        using Pkg.Sub.Symbol;           # c#

    The import points at the symbol directly (Java inner classes use
    ``pkg.sub.Outer.Inner`` — still a dotted path ending in Symbol).
    We match the trailing ``.<symbol>`` segment (or ``.<Symbol>`` with
    the exact case tldr gave us), swap the prefix for ``new_module``,
    and keep the trailing semicolon if the lang requires one.

    PHP's ``use Foo\\Bar\\Sym;`` reuses this function with
    ``path_sep='\\'``.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"

    s = stripped.strip()
    # Indent preserved for nested imports (rare but legal in some langs).
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    # Strip trailing semicolon for uniform matching; re-add per lang.
    had_semi = s.endswith(";")
    if had_semi:
        s = s[:-1].rstrip()

    if not s.startswith(keyword + " "):
        return None, False, f"not a {keyword} statement"

    path = s[len(keyword) + 1:].strip()

    # Wildcards: java `import pkg.*;`, c# `using X.*;` — moving a single
    # symbol doesn't require rewriting a wildcard because the whole
    # package is already in scope. Flag for manual review since the
    # semantics are: after the move, the symbol lives elsewhere and the
    # wildcard won't cover it.
    if path.endswith(path_sep + "*") or path.endswith(".*"):
        return None, False, "wildcard import"

    # Split on the path separator to find the symbol suffix.
    if path_sep == "\\":
        segments = path.split(path_sep)
    else:
        segments = path.split(path_sep)
    if not segments:
        return None, False, "empty import path"

    last = segments[-1]
    # Some langs allow ``import pkg.sub.{A, B}`` (scala). Flag and let
    # the user handle.
    if "{" in path or "}" in path:
        return None, False, "braced multi-import (manual review)"

    if last != symbol:
        return None, False, f"last segment {last!r} != symbol {symbol!r}"

    # Swap: new_module already includes the full prefix up to the
    # symbol for JVM-family, and up to the namespace for PHP. We append
    # the symbol with the right path separator.
    if path_sep == "\\":
        new_path = new_module + "\\" + symbol
    else:
        new_path = new_module + "." + symbol
    rebuilt = f"{indent}{keyword} {new_path}"
    if had_semi or trailing_semi:
        rebuilt += ";"
    return rebuilt + newline, False, ""


def _rewrite_java_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Java: ``import pkg.sub.Symbol;``."""
    return _rewrite_dotted_import_line(
        line, symbol, new_module, keyword="import", trailing_semi=True,
    )


def _rewrite_kotlin_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Kotlin: ``import pkg.sub.Symbol`` (no semicolon)."""
    return _rewrite_dotted_import_line(
        line, symbol, new_module, keyword="import", trailing_semi=False,
    )


def _rewrite_scala_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Scala: ``import pkg.sub.Symbol`` (no semicolon, Scala 3 / 2 both)."""
    return _rewrite_dotted_import_line(
        line, symbol, new_module, keyword="import", trailing_semi=False,
    )


def _rewrite_csharp_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """C#: ``using Pkg.Sub.Cls;``."""
    return _rewrite_dotted_import_line(
        line, symbol, new_module, keyword="using", trailing_semi=True,
    )


def _rewrite_php_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """PHP: ``use Foo\\Bar\\Symbol;``."""
    return _rewrite_dotted_import_line(
        line, symbol, new_module, keyword="use", trailing_semi=True,
        path_sep="\\",
    )


def _rewrite_go_import_line(
    line: str, symbol: str, new_specifier: str,
) -> tuple[str | None, bool, str]:
    """Go: ``import "module/path"`` (the symbol isn't named in the
    import — Go imports are package-path-only).

    Go imports don't carry symbol names — the symbol is accessed via
    ``<pkg>.<Symbol>`` at the call site. Moving ``Foo`` from
    ``pkg/a`` to ``pkg/b`` means callers' import lines must be
    switched from ``"pkg/a"`` to ``"pkg/b"``. We do a substring swap
    inside the quoted path.

    ``new_specifier`` is the new package path (no quotes).
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"

    s = stripped.strip()
    # Normalize: ``import "path"`` and bare ``"path"`` inside a grouped
    # import block both end up here; tldr flags either shape.
    if s.startswith("import "):
        prefix = s[:7]
        rest = s[7:].strip()
    else:
        prefix = ""
        rest = s
    if not (rest.startswith('"') and rest.endswith('"')):
        return None, False, "not a quoted import specifier"
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]
    rebuilt = f"{indent}{prefix}\"{new_specifier}\"{newline}"
    return rebuilt, False, ""


def _rewrite_rust_import_line(
    line: str, symbol: str, new_use_path: str,
) -> tuple[str | None, bool, str]:
    """Rust: rewrite a ``use`` statement so ``symbol`` now resolves via
    ``new_use_path``.

    Cases handled automatically:

    * Simple: ``use crate::foo::Bar;`` → ``use crate::new::Bar;``.
    * Aliased top-level: ``use crate::foo::Bar as R;`` →
      ``use crate::new::Bar as R;`` — the ``as R`` clause is preserved
      verbatim.
    * Braced (flat): ``use crate::foo::{Bar, Baz};`` moving ``Bar`` →
      ``use crate::foo::Baz;`` + ``use crate::new::Bar;``. Collapses the
      braces when one name remains, drops the line entirely when none do.
    * Braced with an aliased sibling: ``use crate::foo::{Bar as R, Baz};``
      — the sibling's ``as`` clause is kept verbatim.
    * Nested (arbitrary depth): ``use crate::foo::{sub::{Bar, Qux}};``
      moving ``Bar`` → a new top-level ``use crate::new::Bar;`` plus the
      original tree with ``Bar`` structurally removed (empty subgroups
      collapse cleanly).
    * Wildcard: ``use crate::foo::*;`` — the wildcard line is preserved
      unchanged and an explicit ``use <new_use_path>::<symbol>;`` is
      appended. A ``manual_review`` warning is emitted noting the glob
      may now be unused.

    ``new_use_path`` is the path prefix up to (but not including) the
    symbol — e.g. ``crate::new_module`` — with no trailing ``::``.

    Returns ``(new_text | None, split_flag, reason)``. ``new_text``
    contains the replacement line(s) (possibly multi-line, joined with
    the original newline style). ``split_flag`` is True when the
    rewrite produced a residual import in addition to the moved-symbol
    line. ``reason`` is non-empty iff new_text is None *or* the
    rewrite succeeded but carries a manual_review advisory (currently
    only the wildcard case). When ``reason`` is non-empty alongside a
    non-None new_text, callers should treat the rewrite as applied but
    surface the reason to the user.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    had_semi = s.endswith(";")
    if had_semi:
        s = s[:-1].rstrip()

    # Strip ``pub `` visibility qualifier — the rewrite preserves it.
    pub_prefix = ""
    if s.startswith("pub "):
        pub_prefix = "pub "
        s = s[4:]
    if not s.startswith("use "):
        return None, False, "not a use statement"
    path = s[4:].strip()

    semi = ";" if had_semi else ""

    # --- Wildcard: preserve line, append explicit import ---------------
    # Users write ``use foo::*;`` to get every public symbol of ``foo``.
    # We can't tell which symbols they actually use without a type
    # checker, so the safe bounded solution is: leave the wildcard alone
    # (other symbols may still need it), append an explicit import for
    # the moved symbol, and warn the caller.
    if path.endswith("::*") or path == "*":
        wildcard_line = f"{indent}{pub_prefix}use {path}{semi}{newline}"
        explicit_line = (
            f"{indent}{pub_prefix}use {new_use_path}::{symbol}{semi}{newline}"
        )
        return (
            wildcard_line + explicit_line,
            True,
            (
                "wildcard import may now be unused \u2014 consider "
                f"removing ``use {path};`` if ``{path[:-3] if path.endswith('::*') else path}::*`` "
                f"was only used for ``{symbol}``"
            ),
        )

    # --- Top-level aliased simple path: ``use a::b::Bar as R;`` --------
    # The ``as R`` clause lives on the whole path; preserve it verbatim.
    # We only match when the alias applies to the ENTIRE path (no
    # braces), because an alias inside ``{...}`` is a sibling item and
    # is handled by the braced/nested branch below.
    if " as " in path and "{" not in path:
        alias_idx = path.rfind(" as ")
        head = path[:alias_idx].strip()
        alias = path[alias_idx + 4 :].strip()
        segments = head.split("::")
        if not segments or segments[-1] != symbol:
            return (
                None,
                False,
                f"aliased use last segment {segments[-1]!r} != symbol",
            )
        rebuilt = (
            f"{indent}{pub_prefix}use {new_use_path}::{symbol} as {alias}"
            f"{semi}{newline}"
        )
        return rebuilt, False, ""

    # --- Braced group (possibly nested): use tree-sitter ---------------
    if "{" in path:
        result = _rewrite_rust_braced_use(
            stripped=stripped,
            pub_prefix=pub_prefix,
            indent=indent,
            newline=newline,
            symbol=symbol,
            new_use_path=new_use_path,
            had_semi=had_semi,
        )
        if result is not None:
            return result
        # Tree-sitter unavailable / unparseable — fall back to the flat
        # regex handler below for the simplest case, else flag.
        if path.count("{") > 1:
            return None, False, "nested use tree (tree-sitter unavailable)"
        brace_start = path.index("{")
        if not path.endswith("}"):
            return None, False, "malformed use-tree"
        prefix = path[:brace_start].rstrip(":")
        inner = path[brace_start + 1 : -1]
        names = [n.strip() for n in inner.split(",") if n.strip()]
        if any("::" in n or "{" in n for n in names):
            return None, False, "nested use-tree (tree-sitter unavailable)"
        if symbol not in names and not any(
            _alias_head(n) == symbol for n in names
        ):
            return None, False, "symbol not in use-tree"
        remaining = [n for n in names if _alias_head(n) != symbol]
        new_symbol_line = (
            f"{indent}{pub_prefix}use {new_use_path}::{symbol}{semi}{newline}"
        )
        if not remaining:
            return new_symbol_line, False, ""
        if len(remaining) == 1:
            residual = (
                f"{indent}{pub_prefix}use {prefix}::{remaining[0]}{semi}{newline}"
            )
        else:
            joined = ", ".join(remaining)
            residual = (
                f"{indent}{pub_prefix}use {prefix}::{{{joined}}}{semi}{newline}"
            )
        return residual + new_symbol_line, True, ""

    # --- Simple ``use foo::bar::Symbol;`` ------------------------------
    segments = path.split("::")
    if not segments or segments[-1] != symbol:
        return None, False, f"last segment {segments[-1]!r} != symbol"
    rebuilt = (
        f"{indent}{pub_prefix}use {new_use_path}::{symbol}{semi}{newline}"
    )
    return rebuilt, False, ""


def _alias_head(name: str) -> str:
    """Given a use-list item like ``Bar`` or ``Bar as R``, return the
    head identifier (``Bar`` in both cases). Used to match group items
    against the moved symbol without tripping on the alias.
    """
    if " as " in name:
        return name.split(" as ", 1)[0].strip()
    return name.strip()


def _rewrite_rust_braced_use(
    *,
    stripped: str,
    pub_prefix: str,
    indent: str,
    newline: str,
    symbol: str,
    new_use_path: str,
    had_semi: bool,
) -> tuple[str | None, bool, str] | None:
    """Rewrite a Rust braced ``use`` statement using tree-sitter-rust.

    Handles arbitrary nesting. Returns a ``(new_text, split, reason)``
    tuple on success, ``(None, False, reason)`` on a bounded failure
    (symbol not found, malformed tree), or ``None`` when tree-sitter is
    unavailable — caller then falls back to the flat regex path.
    """
    try:
        import tree_sitter_rust  # type: ignore[import-not-found]
        from tree_sitter import Language, Parser  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        lang = Language(tree_sitter_rust.language())
        parser = Parser(lang)
    except Exception:
        return None

    # Parse just the single stripped line so byte offsets are local.
    src_bytes = stripped.encode("utf-8")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    # Find the use_declaration child.
    use_decl = None
    for child in root.children:
        if child.type == "use_declaration":
            use_decl = child
            break
    if use_decl is None:
        return None, False, "could not parse use declaration"

    # The interesting payload is the child after ``use`` (and before ``;``).
    # For braced groups it's a ``scoped_use_list``.
    payload = None
    for child in use_decl.children:
        if child.type in {
            "scoped_use_list", "use_list", "scoped_identifier",
            "use_as_clause", "use_wildcard", "identifier",
        }:
            payload = child
            break
    if payload is None:
        return None, False, "use declaration has no payload"

    src_text = stripped

    def node_text(n) -> str:
        return src_text[n.start_byte : n.end_byte]

    def list_items(use_list_node):
        """Yield direct item children of a ``use_list`` (skip punctuation)."""
        items = []
        for c in use_list_node.children:
            if c.type in {"{", "}", ","}:
                continue
            items.append(c)
        return items

    def item_head(item_node) -> str:
        """Return the identifier that ``item_node`` imports at the leaf.

        * ``identifier`` → its text.
        * ``use_as_clause`` → the aliased target's head (``Bar`` in
          ``Bar as R``).
        * ``scoped_identifier`` / ``scoped_use_list`` → the last path
          segment's head (for scoped_identifier) or, for a scoped_use_list,
          its own name (the group as a whole has no single symbol).
        """
        t = item_node.type
        if t == "identifier":
            return node_text(item_node)
        if t == "use_as_clause":
            # First child is the aliased thing.
            for c in item_node.children:
                if c.type in {"identifier", "scoped_identifier"}:
                    return item_head(c)
            return ""
        if t == "scoped_identifier":
            last = None
            for c in item_node.children:
                if c.type == "identifier":
                    last = c
            return node_text(last) if last else ""
        if t == "scoped_use_list":
            # A group like ``a::{...}`` — no single leaf.
            return ""
        if t == "self":
            return "self"
        if t == "use_wildcard":
            return "*"
        return ""

    def contains_symbol(item_node) -> bool:
        """Recursively check whether ``item_node`` imports ``symbol``."""
        t = item_node.type
        if t in {"identifier", "use_as_clause", "scoped_identifier"}:
            return item_head(item_node) == symbol
        if t in {"scoped_use_list", "use_list"}:
            for sub in (
                list_items(item_node) if t == "use_list"
                else [
                    c for c in item_node.children
                    if c.type == "use_list"
                ]
            ):
                if t == "use_list":
                    if contains_symbol(sub):
                        return True
                else:
                    for inner in list_items(sub):
                        if contains_symbol(inner):
                            return True
        return False

    def find_symbol_alias(item_node) -> str | None:
        """If ``item_node`` imports ``symbol`` via a ``use_as_clause``,
        return the alias string (the ``R`` in ``Bar as R``). Walks into
        nested groups. Returns None when the symbol appears unaliased
        or not at all.
        """
        t = item_node.type
        if t == "use_as_clause":
            if item_head(item_node) == symbol:
                # Last identifier child is the alias.
                last_id = None
                for c in item_node.children:
                    if c.type == "identifier":
                        last_id = c
                if last_id is not None:
                    return node_text(last_id)
            return None
        if t in {"identifier", "scoped_identifier"}:
            return None
        if t == "use_list":
            for inner in list_items(item_node):
                alias = find_symbol_alias(inner)
                if alias is not None:
                    return alias
            return None
        if t == "scoped_use_list":
            for c in item_node.children:
                if c.type == "use_list":
                    alias = find_symbol_alias(c)
                    if alias is not None:
                        return alias
            return None
        return None

    def serialize_without_symbol(item_node) -> str | None:
        """Return the source for ``item_node`` with ``symbol`` removed.

        Returns ``None`` when the item itself is the symbol (the caller
        should then drop it from its parent list). The returned text
        preserves the surviving tree shape; an empty subgroup collapses
        by returning None at that level.
        """
        t = item_node.type
        if t in {"identifier", "use_as_clause", "scoped_identifier"}:
            if item_head(item_node) == symbol:
                return None
            return node_text(item_node)
        if t == "scoped_use_list":
            # scoped_use_list := <scoped_ident or ident> :: use_list
            head_prefix_parts: list[str] = []
            inner_list_node = None
            for c in item_node.children:
                if c.type == "use_list":
                    inner_list_node = c
                elif c.type in {"scoped_identifier", "identifier", "::"}:
                    head_prefix_parts.append(node_text(c))
            if inner_list_node is None:
                return node_text(item_node)  # shouldn't happen
            surviving = []
            for inner in list_items(inner_list_node):
                piece = serialize_without_symbol(inner)
                if piece is not None:
                    surviving.append(piece)
            if not surviving:
                return None
            head_prefix = "".join(head_prefix_parts).rstrip(":")
            if len(surviving) == 1:
                # Collapse single-element group.
                piece = surviving[0]
                # If piece is a scoped_use_list already serialized with
                # its own prefix, joining with ``::`` still works.
                return f"{head_prefix}::{piece}"
            return f"{head_prefix}::{{{', '.join(surviving)}}}"
        if t == "use_list":
            surviving = []
            for inner in list_items(item_node):
                piece = serialize_without_symbol(inner)
                if piece is not None:
                    surviving.append(piece)
            if not surviving:
                return None
            if len(surviving) == 1:
                return surviving[0]
            return f"{{{', '.join(surviving)}}}"
        if t in {"self", "use_wildcard"}:
            return node_text(item_node)
        return node_text(item_node)

    # Payload shape dispatch.
    if payload.type != "scoped_use_list":
        # We only handle braced groups here; the caller handles simple /
        # aliased / wildcard before delegating.
        return None, False, "braced helper invoked on non-braced payload"

    # Confirm the symbol is somewhere inside.
    inner_list = None
    head_parts: list[str] = []
    for c in payload.children:
        if c.type == "use_list":
            inner_list = c
        elif c.type in {"scoped_identifier", "identifier", "crate", "::"}:
            head_parts.append(node_text(c))
    if inner_list is None:
        return None, False, "scoped_use_list has no use_list child"

    found = False
    matched_alias: str | None = None
    for item in list_items(inner_list):
        if contains_symbol(item):
            found = True
            matched_alias = find_symbol_alias(item)
            break
    if not found:
        return None, False, "symbol not in use-tree"

    # Serialize surviving items.
    surviving: list[str] = []
    for item in list_items(inner_list):
        piece = serialize_without_symbol(item)
        if piece is not None:
            surviving.append(piece)

    semi = ";" if had_semi else ""
    alias_suffix = f" as {matched_alias}" if matched_alias else ""
    new_symbol_line = (
        f"{indent}{pub_prefix}use {new_use_path}::{symbol}{alias_suffix}"
        f"{semi}{newline}"
    )

    head_prefix = "".join(head_parts).rstrip(":")
    if not surviving:
        # Entire use-tree collapsed — just the new line.
        return new_symbol_line, False, ""
    if len(surviving) == 1:
        residual = (
            f"{indent}{pub_prefix}use {head_prefix}::{surviving[0]}{semi}{newline}"
        )
    else:
        residual = (
            f"{indent}{pub_prefix}use {head_prefix}::{{{', '.join(surviving)}}}{semi}{newline}"
        )
    return residual + new_symbol_line, True, ""


def _rewrite_swift_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Swift: ``import Foo`` (module name only).

    Swift imports name modules, not symbols. Moving a symbol between
    files in the same module requires no import rewrite; moving
    between modules requires swapping the module name.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    if not s.startswith("import "):
        return None, False, "not a Swift import"
    current = s[7:].strip()
    # Skip submodule imports like ``import Foo.Bar`` — we'd need to
    # know whether the user wants the whole module or a submodule.
    if "." in current:
        return None, False, "submodule import (manual review)"
    if current == new_module:
        # No-op: destination in same module.
        return stripped + newline, False, ""
    rebuilt = f"{indent}import {new_module}{newline}"
    return rebuilt, False, ""


def _rewrite_ruby_import_line(
    line: str, symbol: str, new_require_path: str,
) -> tuple[str | None, bool, str]:
    """Ruby: ``require_relative "foo/bar"`` or ``require "foo/bar"``.

    Ruby imports name the FILE, not the symbol. We do a path-string
    substring swap inside the existing quotes.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    m = re.match(
        r"^(require|require_relative|load)\s+(['\"])([^'\"]+)(['\"])\s*$",
        s,
    )
    if not m:
        return None, False, "unparsable require"
    keyword = m.group(1)
    quote = m.group(2)
    rebuilt = (
        f"{indent}{keyword} {quote}{new_require_path}{quote}{newline}"
    )
    return rebuilt, False, ""


def _rewrite_elixir_import_line(
    line: str, symbol: str, new_module: str,
) -> tuple[str | None, bool, str]:
    """Elixir: ``alias Foo.Bar`` or ``import Foo.Bar`` / ``use ...``.

    Handles ``alias`` and ``import`` with a single dotted-module
    reference. Braced aliases (``alias Foo.{Bar, Baz}``) are flagged
    for manual review — the same nested-group caveat as Rust applies.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    m = re.match(
        r"^(alias|import|require|use)\s+([\w.]+(?:\.\{[^}]+\})?)"
        r"(?:\s*,\s*as:\s*(\w+))?\s*$",
        s,
    )
    if not m:
        return None, False, "unparsable elixir alias/import"
    keyword = m.group(1)
    path = m.group(2)
    as_clause = m.group(3)

    if "{" in path:
        return None, False, "braced alias group (manual review)"

    segments = path.split(".")
    if not segments or segments[-1] != symbol:
        return None, False, (
            f"last segment {segments[-1]!r} != symbol {symbol!r}"
        )

    new_path = f"{new_module}.{symbol}" if new_module else symbol
    rebuilt = f"{indent}{keyword} {new_path}"
    if as_clause:
        rebuilt += f", as: {as_clause}"
    return rebuilt + newline, False, ""


def _rewrite_lua_import_line(
    line: str, symbol: str, new_require_path: str,
) -> tuple[str | None, bool, str]:
    """Lua: ``require "foo.bar"`` or ``require("foo.bar")`` or
    ``local m = require "foo.bar"``.

    Like Ruby, Lua imports name modules/files, not symbols. Swap the
    quoted path string.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    # Match ``require "x"``, ``require('x')``, ``local v = require "x"``.
    m = re.match(
        r"^(local\s+\w+\s*=\s*)?require\s*[\(\s]\s*(['\"])([^'\"]+)\2\s*\)?"
        r"\s*$",
        s,
    )
    if not m:
        return None, False, "unparsable lua require"
    prefix = m.group(1) or ""
    quote = m.group(2)
    # Rebuild with either call-style or bare-string-style based on the
    # original form. We check for a '(' after ``require`` to decide.
    uses_parens = "(" in s.split("require", 1)[1]
    if uses_parens:
        rebuilt = (
            f"{indent}{prefix}require({quote}{new_require_path}{quote})"
            f"{newline}"
        )
    else:
        rebuilt = (
            f"{indent}{prefix}require {quote}{new_require_path}{quote}"
            f"{newline}"
        )
    return rebuilt, False, ""


def _rewrite_c_import_line(
    line: str, symbol: str, new_header_path: str,
) -> tuple[str | None, bool, str]:
    """C/C++: ``#include "foo.h"`` or ``#include <foo.h>``.

    We only rewrite the ``"..."`` form (project-local includes).
    ``<...>`` is left for manual review — it implies a system or
    compiler-searched include whose path can't be inferred from
    filesystem layout alone.
    """
    stripped = line.rstrip("\n")
    newline = line[len(stripped):] or "\n"
    s = stripped.strip()
    indent_len = len(stripped) - len(stripped.lstrip())
    indent = stripped[:indent_len]

    m = re.match(r'^#include\s+"([^"]+)"\s*$', s)
    if m:
        rebuilt = f"{indent}#include \"{new_header_path}\"{newline}"
        return rebuilt, False, ""
    m = re.match(r"^#include\s+<[^>]+>\s*$", s)
    if m:
        return None, False, "system include (manual review)"
    return None, False, "unparsable include"


# ---------------------------------------------------------------------------
# Family dispatch — import specifier + rewriter.
# ---------------------------------------------------------------------------


def _compute_import_specifier(
    family: str,
    symbol: str,
    from_file: Path,
    to_file: Path,
    caller_file: Path,
    project_root: Path,
) -> str:
    """Compute the new module/namespace/path string for a rewrite.

    The returned value is what the family's rewriter expects as its
    ``new_module`` / ``new_specifier`` argument — NOT the full new
    import line. Each family has different conventions:

    * python: dotted module path (caller appends ``.<symbol>`` via
      ``from <this> import <symbol>``)
    * ts/js: relative path from caller to target, no extension
    * java/scala: ``pkg.sub`` (caller appends ``.<symbol>``)
    * kotlin: ``pkg.sub`` (caller appends ``.<symbol>``)
    * csharp: ``Pkg.Sub`` (caller appends ``.<symbol>``)
    * php: ``Foo\\Bar`` (caller appends ``\\<symbol>``)
    * go: package path in quotes (no symbol)
    * rust: ``crate::pkg::sub`` (caller appends ``::<symbol>``)
    * swift: module name only
    * ruby: relative require path, no quotes, no extension
    * elixir: ``Foo.Bar`` (caller appends ``.<symbol>``)
    * lua: dotted require path
    * c/cpp: caller-relative header path
    """
    if family == "python":
        return _python_module_for(to_file, project_root)
    if family == "ts":
        return _ts_relative_path(caller_file, to_file)
    if family == "java":
        # Java imports name the class; ``pkg.sub`` + class stem.
        try:
            rel = to_file.resolve().relative_to(project_root.resolve())
            parts = list(rel.parent.parts)
        except ValueError:
            parts = []
        return ".".join(parts)
    if family == "kotlin":
        return _kotlin_module_for(to_file, project_root)
    if family == "scala":
        return _scala_module_for(to_file, project_root)
    if family == "csharp":
        return _csharp_namespace_for(to_file, project_root)
    if family == "php":
        return _php_namespace_for(to_file, project_root)
    if family == "go":
        return _go_import_path_for(to_file, project_root)
    if family == "rust":
        return _rust_use_path_for(to_file, project_root)
    if family == "swift":
        return _swift_module_for(to_file, project_root)
    if family == "ruby":
        return _ruby_require_path_for(caller_file, to_file, project_root)
    if family == "elixir":
        return _elixir_module_for(to_file, project_root)
    if family == "lua":
        return _lua_require_path_for(caller_file, to_file, project_root)
    if family in {"c", "cpp"}:
        return _c_include_path_for(caller_file, to_file, project_root)
    return ""


def _rewrite_line_for_family(
    family: str, line: str, symbol: str, specifier: str,
) -> tuple[str | None, bool, str]:
    """Dispatch to the family's import-rewriter. See the individual
    ``_rewrite_<family>_import_line`` docstrings for semantics.
    """
    if family == "python":
        return _rewrite_python_import_line(line, symbol, specifier)
    if family == "ts":
        return _rewrite_ts_import_line(line, symbol, specifier)
    if family == "java":
        return _rewrite_java_import_line(line, symbol, specifier)
    if family == "kotlin":
        return _rewrite_kotlin_import_line(line, symbol, specifier)
    if family == "scala":
        return _rewrite_scala_import_line(line, symbol, specifier)
    if family == "csharp":
        return _rewrite_csharp_import_line(line, symbol, specifier)
    if family == "php":
        return _rewrite_php_import_line(line, symbol, specifier)
    if family == "go":
        return _rewrite_go_import_line(line, symbol, specifier)
    if family == "rust":
        return _rewrite_rust_import_line(line, symbol, specifier)
    if family == "swift":
        return _rewrite_swift_import_line(line, symbol, specifier)
    if family == "ruby":
        return _rewrite_ruby_import_line(line, symbol, specifier)
    if family == "elixir":
        return _rewrite_elixir_import_line(line, symbol, specifier)
    if family == "lua":
        return _rewrite_lua_import_line(line, symbol, specifier)
    if family in {"c", "cpp"}:
        return _rewrite_c_import_line(line, symbol, specifier)
    return None, False, f"no rewriter for {family}"


# ---------------------------------------------------------------------------
# Source-span extraction + destination insertion.
# ---------------------------------------------------------------------------


def _extract_span(
    file_path: str, symbol: str,
) -> tuple[list[str], tuple[int, int], list[str]]:
    """Extract ``symbol``'s full AST span from ``file_path``.

    Returns ``(remaining_lines, span, extracted_lines)``:
        remaining_lines — the source lines with the symbol removed
        span            — (start, end) 1-indexed inclusive (pre-removal)
        extracted_lines — the lines representing the symbol (incl. one
                          trailing blank separator when present)

    Raises ``ValueError`` if the symbol isn't found.
    """
    path = Path(file_path)
    original = path.read_text(encoding="utf-8", errors="replace")
    lines = original.splitlines(keepends=True)
    total = len(lines)

    ast_nodes = get_ast_map(file_path, total)
    target = _resolve_symbol(symbol, ast_nodes)
    if target is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Symbol '{symbol}' not found in {file_path}. "
            f"Available: {available}"
        )

    start_idx = target.line_start - 1  # 0-indexed
    end_idx = target.line_end           # exclusive

    # Include a single trailing blank line as separator (mirrors
    # ``move_symbol``'s behaviour).
    if end_idx < total and lines[end_idx].strip() == "":
        end_idx += 1

    extracted = lines[start_idx:end_idx]
    remaining = lines[:start_idx] + lines[end_idx:]

    return remaining, (target.line_start, target.line_end), extracted


def _insert_into_destination(
    to_path: Path, extracted: list[str], after: str | None,
) -> tuple[str, int]:
    """Return the new content of ``to_path`` and the insertion line.

    When ``after`` is None we append to end-of-file. When it's a name we
    locate the matching AST node and insert just after its last line.
    Raises ``ValueError`` when ``after`` is given but not found.
    """
    dst_text = to_path.read_text(encoding="utf-8", errors="replace")
    dst_lines = dst_text.splitlines(keepends=True)
    total = len(dst_lines)

    if after is None:
        # Append at EOF. Ensure a blank separator if the file doesn't
        # end with a newline or blank.
        prefix = dst_lines
        if prefix and not prefix[-1].endswith("\n"):
            prefix = list(prefix)
            prefix[-1] = prefix[-1] + "\n"
        # Separator blank line between existing content and the
        # extracted symbol (only if there's existing content and the
        # last line isn't already blank).
        if prefix and prefix[-1].strip() != "":
            prefix = list(prefix) + ["\n"]
        merged = prefix + list(extracted)
        # Ensure final newline.
        if merged and not merged[-1].endswith("\n"):
            merged[-1] = merged[-1] + "\n"
        return "".join(merged), len(prefix)

    # after is named — resolve via the destination's AST.
    ast_nodes = get_ast_map(str(to_path), total)
    anchor = _resolve_symbol(after, ast_nodes)
    if anchor is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Anchor '{after}' not found in {to_path}. "
            f"Available: {available}"
        )

    insert_at = anchor.line_end  # 0-indexed exclusive == 1-indexed end
    prefix = list(dst_lines[:insert_at])
    suffix = list(dst_lines[insert_at:])

    # Separator blank line before the extracted symbol if the anchor
    # didn't end with one.
    if prefix and prefix[-1].strip() != "":
        prefix.append("\n")

    injected = list(extracted)
    # Ensure a trailing blank between the extracted span and the
    # following content.
    if suffix and injected and injected[-1].strip() != "":
        injected.append("\n")

    merged = prefix + injected + suffix
    if merged and not merged[-1].endswith("\n"):
        merged[-1] = merged[-1] + "\n"
    return "".join(merged), len(prefix)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def move_to_file(
    symbol: str,
    from_file: str,
    to_file: str,
    after: str | None,
    project_root: Path | str,
    dry_run: bool = False,
) -> MoveToFilePlan:
    """Move ``symbol`` from ``from_file`` into ``to_file`` and rewrite imports.

    See module docstring for the full design. Raises ``ValueError`` for
    user-recoverable problems (same-file move, conflict, missing file,
    unsupported extension, symbol not found).
    """
    # --- Input validation --------------------------------------------------
    from_path = Path(from_file)
    to_path = Path(to_file)
    root = Path(project_root)

    if from_path.resolve() == to_path.resolve():
        raise ValueError(
            "from_file and to_file are the same — use fast_move "
            "(MCP) / `fastedit move` (CLI) for same-file moves."
        )

    if not from_path.exists():
        raise ValueError(f"Source file not found: {from_file}")
    if not to_path.exists():
        raise ValueError(f"Target file not found: {to_file}")

    from_ext = from_path.suffix
    to_ext = to_path.suffix
    if from_ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported source extension '{from_ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTS)}"
        )
    if to_ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported target extension '{to_ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTS)}"
        )

    # Cross-family moves (e.g. .py -> .ts) are nonsense.
    from_family = _ext_family(from_ext)
    to_family = _ext_family(to_ext)
    if from_family != to_family:
        raise ValueError(
            "Cross-language move not supported: "
            f"{from_ext} -> {to_ext}"
        )
    family = from_family or ""

    language = detect_language(from_path)

    # --- Conflict check: symbol already defined in destination -----------
    dst_lines_total = len(
        to_path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    dst_ast = get_ast_map(str(to_path), dst_lines_total)
    if _resolve_symbol(symbol, dst_ast) is not None:
        raise ValueError(
            f"Symbol '{symbol}' already exists in {to_file} — "
            f"conflict, refusing to move."
        )

    # --- Extract span from source (in-memory) ----------------------------
    remaining_src_lines, span, extracted = _extract_span(str(from_path), symbol)

    # --- Compute destination content (in-memory) -------------------------
    new_to_content, insertion_line = _insert_into_destination(
        to_path, extracted, after,
    )

    # --- Plan import rewrites -------------------------------------------
    # Query tldr for every reference (not just imports — on non-native
    # langs tldr emits kind="other" for imports too), then filter to
    # import-shaped lines with the per-family text matcher.
    raw_refs = _run_tldr_references_all(symbol, root)
    import_refs = _filter_refs_to_imports(raw_refs, family)

    # For langs where imports name the FILE (not the symbol) — ruby,
    # go, lua, c/cpp — tldr won't find the import line via the symbol
    # name. Scan consumer files directly for import-shaped lines that
    # mention the source file's basename.
    if family in _FILE_NAMED_IMPORT_FAMILIES:
        file_named = _scan_file_named_imports(family, from_path, root)
        # Dedupe against any hits tldr already produced.
        seen: set[tuple[str, int]] = {
            (r.get("file"), r.get("line"))
            for r in import_refs
        }
        for r in file_named:
            key = (r.get("file"), r.get("line"))
            if key in seen:
                continue
            seen.add(key)
            import_refs.append(r)

    # Group refs by file so we apply once per file (multi-name import
    # rewrites generate multiple output lines from a single input line).
    refs_by_file: dict[str, list[dict]] = {}
    for ref in import_refs:
        f = ref.get("file")
        if not f:
            continue
        # Drop refs in the source file itself — it's about to go away.
        try:
            if Path(f).resolve() == from_path.resolve():
                continue
        except OSError:
            continue
        # Drop refs in the destination file — tldr may have seen the
        # old import there too; after the move they'd be self-imports
        # which langs don't want.
        try:
            if Path(f).resolve() == to_path.resolve():
                continue
        except OSError:
            pass
        refs_by_file.setdefault(f, []).append(ref)

    rewrites: list[ImportRewrite] = []
    skipped: list[dict] = []
    # Staged writes for consumer files so a dry-run is a no-op.
    staged_writes: dict[Path, str] = {}

    for file_str, file_refs in refs_by_file.items():
        caller_path = Path(file_str)
        try:
            content = caller_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append({"file": file_str, "reason": "read error"})
            continue
        content_lines = content.splitlines(keepends=True)

        # Unique line numbers. A single import line may be emitted more
        # than once (e.g. definition + reference in older tldr versions).
        seen_lines: set[int] = set()
        lines_to_rewrite: list[int] = []
        for ref in file_refs:
            ln = ref.get("line", 0)
            if ln < 1 or ln > len(content_lines):
                continue
            if ln in seen_lines:
                continue
            seen_lines.add(ln)
            lines_to_rewrite.append(ln)

        # Per-file import specifier: some langs derive from caller (ts
        # relative, ruby require_relative, c/cpp include), others from
        # the destination alone (python dotted, java/kotlin, rust).
        specifier = _compute_import_specifier(
            family=family,
            symbol=symbol,
            from_file=from_path,
            to_file=to_path,
            caller_file=caller_path,
            project_root=root,
        )

        # Apply rewrites bottom-up to keep earlier line indices valid.
        new_content_lines = list(content_lines)
        lines_to_rewrite.sort(reverse=True)
        file_rewrites: list[ImportRewrite] = []
        for ln in lines_to_rewrite:
            old_line = new_content_lines[ln - 1]
            new_text, split, reason = _rewrite_line_for_family(
                family, old_line, symbol, specifier,
            )
            if new_text is None:
                file_rewrites.append(
                    ImportRewrite(
                        file=file_str,
                        line=ln,
                        old_import=old_line.rstrip("\n"),
                        new_import="",
                        split=False,
                        manual_review=True,
                        reason=reason,
                    )
                )
                continue
            file_rewrites.append(
                ImportRewrite(
                    file=file_str,
                    line=ln,
                    old_import=old_line.rstrip("\n"),
                    new_import=new_text.rstrip("\n"),
                    split=split,
                )
            )
            new_content_lines[ln - 1 : ln] = [new_text]

        if any(r for r in file_rewrites if not r.manual_review):
            staged_writes[caller_path] = "".join(new_content_lines)
        rewrites.extend(file_rewrites)

    # --- Build plan object -----------------------------------------------
    plan = MoveToFilePlan(
        symbol=symbol,
        from_file=str(from_path),
        to_file=str(to_path),
        source_span=span,
        insertion_line=insertion_line,
        import_rewrites=[
            {
                "file": r.file,
                "line": r.line,
                "old_import": r.old_import,
                "new_import": r.new_import,
                "split": r.split,
                "manual_review": r.manual_review,
                "reason": r.reason,
            }
            for r in rewrites
        ],
        skipped_importers=skipped,
        applied=False,
    )

    if dry_run:
        plan.message = _format_plan_message(plan, preview=True)
        return plan

    # --- Apply writes via atomic helper ---------------------------------
    from ..mcp.backup import BackupStore, _atomic_write

    backups = BackupStore()

    # Source: write the remaining lines (symbol stripped).
    _atomic_write(from_path, "".join(remaining_src_lines), backups=backups)

    # Destination: write the updated content.
    _atomic_write(to_path, new_to_content, backups=backups)

    # Consumers.
    for caller_path, new_content in staged_writes.items():
        _atomic_write(caller_path, new_content, backups=backups)

    plan.applied = True
    plan.message = _format_plan_message(plan, preview=False)
    return plan


def _format_plan_message(plan: MoveToFilePlan, preview: bool) -> str:
    """Render a human-friendly summary. Shared between dry-run + applied."""
    header = "Plan (dry-run):" if preview else "Moved:"
    lines = [
        f"{header} {plan.symbol}",
        f"  source: {plan.from_file}:L{plan.source_span[0]}-{plan.source_span[1]}",
        f"  target: {plan.to_file} (inserted after line {plan.insertion_line})",
    ]
    if plan.import_rewrites:
        lines.append(
            f"  imports rewritten: {len(plan.import_rewrites)} entr(y/ies) "
            f"across {len({r['file'] for r in plan.import_rewrites})} file(s)"
        )
        for r in plan.import_rewrites[:10]:
            tag = "" if not r["manual_review"] else " [manual review]"
            lines.append(
                f"    {r['file']}:{r['line']}{tag}\n"
                f"      - {r['old_import']}\n"
                f"      + {r['new_import'] or '<no change — review needed>'}"
            )
        if len(plan.import_rewrites) > 10:
            lines.append(f"    ... and {len(plan.import_rewrites) - 10} more")
    else:
        lines.append("  imports rewritten: none (no consumers found)")
    if plan.skipped_importers:
        lines.append(f"  skipped importers: {len(plan.skipped_importers)}")
    return "\n".join(lines)
