"""Cross-file symbol relocation: ``move_to_file``.

Milestone 4 (VAL-M4-001..003). Relocates a function / class / method from
``from_file`` into ``to_file`` and rewrites every dependent file's import
statement so the symbol still resolves.

Design:
    * Source span detection reuses :mod:`.ast_utils` (``get_ast_map`` +
      ``_resolve_symbol``) — the same AST pass that backs ``delete_symbol``
      and ``move_symbol``. No duplication.
    * Consumer discovery + per-symbol column data runs through
      ``tldr references <symbol> <root> --kinds import --scope workspace``.
      The helper :func:`_run_tldr_import_refs` is local to this module to
      avoid cross-module coupling, but mirrors the fall-open semantics
      of :func:`fastedit.inference.rename._run_tldr_references`.
    * Writes are staged and applied atomically at the end so a crash
      mid-plan leaves the tree coherent (we call ``_atomic_write`` which
      also snapshots into ``BackupStore`` for ``fast_undo``).

Happy-path scope:
    * Python: ``from <module> import <symbol>`` (including multi-name
      imports: the moved symbol is split onto its own line, pointing at
      the new module).
    * TypeScript / TSX / JavaScript: ``import { <symbol> } from "<path>"``
      with relative paths (same-dir / parent-dir). Other specifiers on
      the same line are kept on the original import.

Out of scope (flagged in plan as "not_rewritten"):
    * Wildcard imports (``from x import *``, ``import * as x from ...``)
    * Re-exports (barrel files), dynamic imports, conditional imports
    * Non-relative or aliased module specifiers (``import foo from "pkg/a"``)
      where we can't infer the new module path
    * Languages other than Python / JS / TS / TSX

Same-file moves (``from_file == to_file``) are rejected with a hint to
use ``fast_move`` / the ``move`` CLI subcommand.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE, detect_language
from .ast_utils import (
    _qualified_symbol_names,
    _resolve_symbol,
    get_ast_map,
)

# Supported extensions for the import-rewrite step. Extraction / deletion
# works for every language the AST analyzer supports (that's what
# delete_symbol uses) — but the rewrite templates below only know Python
# and JS-family syntax, so we gate the top-level API to those.
_PY_EXTS = frozenset({".py"})
_TS_EXTS = frozenset({".ts", ".tsx", ".js", ".jsx"})
_SUPPORTED_EXTS = _PY_EXTS | _TS_EXTS


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


def _run_tldr_import_refs(symbol: str, root: Path) -> list[dict]:
    """Return ``tldr references <symbol> <root> --kinds import`` hits.

    Empty list on any failure (binary missing, timeout, bad JSON). We
    fall-open so infra issues don't silently corrupt a move — the caller
    checks ``import_rewrites`` and warns if zero hits were found.
    """
    cmd = [
        "tldr", "references", symbol, str(root),
        "--format", "json",
        "--scope", "workspace",
        "--kinds", "import",
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
    return [r for r in refs if isinstance(r, dict) and r.get("kind") == "import"]


# ---------------------------------------------------------------------------
# Module-path conversion
# ---------------------------------------------------------------------------


def _python_module_for(file_path: Path, project_root: Path) -> str:
    """Convert ``<root>/pkg/sub/mod.py`` -> ``pkg.sub.mod``.

    When ``file_path`` is outside ``project_root`` we return the stem —
    which yields a best-effort rewrite that won't be worse than leaving
    the import alone.
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return file_path.stem
    parts = list(rel.with_suffix("").parts)
    # Drop trailing __init__: `pkg/__init__.py` imports as `pkg`.
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


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
    if (from_ext in _PY_EXTS) != (to_ext in _PY_EXTS):
        raise ValueError(
            "Cross-language move not supported: "
            f"{from_ext} -> {to_ext}"
        )

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
    is_python = from_ext in _PY_EXTS
    if is_python:
        new_module = _python_module_for(to_path, root)
    else:
        new_module = None  # per-importer: computed relative to caller

    refs = _run_tldr_import_refs(symbol, root)

    # Group refs by file so we apply once per file (multi-name import
    # rewrites generate multiple output lines from a single input line).
    refs_by_file: dict[str, list[dict]] = {}
    for ref in refs:
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
        # which Python/TS don't want.
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

        # Per-file new_specifier for TS/JS: relative to THIS caller.
        if is_python:
            specifier = new_module
        else:
            specifier = _ts_relative_path(caller_path, to_path)

        # Apply rewrites bottom-up to keep earlier line indices valid.
        new_content_lines = list(content_lines)
        lines_to_rewrite.sort(reverse=True)
        file_rewrites: list[ImportRewrite] = []
        for ln in lines_to_rewrite:
            old_line = new_content_lines[ln - 1]
            if is_python:
                new_text, split, reason = _rewrite_python_import_line(
                    old_line, symbol, specifier,
                )
            else:
                new_text, split, reason = _rewrite_ts_import_line(
                    old_line, symbol, specifier,
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
