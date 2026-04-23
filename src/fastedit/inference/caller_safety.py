"""Cross-file caller-safety checks for destructive operations (fast_delete).

When a caller asks to delete a symbol we first run `tldr references` at
project scope and refuse if other files still call / import the symbol.
The MCP tool and CLI both delegate here so the behavior stays consistent.

Fall-open policy (VAL-M2-002): if the `tldr` binary is missing or errors,
the check returns an empty list rather than failing closed. The caller
surfaces a "check skipped" note and proceeds. Deletion is a user-driven
action — we don't want infra issues to block it.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

# Project-root markers, in priority order. First hit wins when walking up
# from the file being edited.
_PROJECT_ROOT_MARKERS = (".git", "pyproject.toml", "package.json")

# Suffix per tldr language name. Used when materializing in-memory source
# to a temp file for `tldr structure`, which reads from disk. Keys match
# EXTENSION_TO_LANGUAGE values.
_LANGUAGE_TO_DEFAULT_SUFFIX: dict[str, str] = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "tsx": ".tsx",
    "rust": ".rs",
    "go": ".go",
    "java": ".java",
    "c": ".c",
    "cpp": ".cpp",
    "ruby": ".rb",
    "swift": ".swift",
    "kotlin": ".kt",
    "c_sharp": ".cs",
    "php": ".php",
    "elixir": ".ex",
    # Not in fastedit's EXTENSION_TO_LANGUAGE today but tldr knows them;
    # harmless to include for forward compat.
    "scala": ".scala",
    "lua": ".lua",
}


def _find_project_root(file_path: Path) -> Path:
    """Walk up from `file_path` looking for a project-root marker.

    Markers: ``.git`` directory, ``pyproject.toml``, or ``package.json``.
    If none is found we fall back to the file's parent directory so the
    safety check still has a sensible scope rather than erroring out.
    """
    file_path = Path(file_path)
    for parent in [file_path.parent, *file_path.parents]:
        for marker in _PROJECT_ROOT_MARKERS:
            if (parent / marker).exists():
                return parent
    return file_path.parent


def _run_tldr_references_workspace(symbol: str, root: Path) -> dict:
    """Invoke `tldr references <symbol> <root>` at workspace scope.

    Returns the parsed JSON payload, or an empty {"references": []} on any
    failure (binary missing, timeout, non-zero exit, bad JSON). We mirror
    the fall-open semantics of ``rename._run_tldr_references`` — infra
    issues must not fail-close a destructive operation.
    """
    cmd = [
        "tldr", "references", symbol, str(root),
        "--format", "json",
        "--scope", "workspace",
        "--min-confidence", "0.9",
        "--limit", "10000",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"references": []}

    stdout = result.stdout or ""
    if not stdout.strip():
        return {"references": []}

    # tldr sometimes emits diagnostic lines before the JSON object. Strip
    # to the first '{' to be lenient.
    start = stdout.find("{")
    if start == -1:
        return {"references": []}
    try:
        data = json.loads(stdout[start:])
    except (json.JSONDecodeError, ValueError):
        return {"references": []}
    if not isinstance(data, dict):
        return {"references": []}
    data.setdefault("references", [])
    return data


def check_cross_file_callers(
    file_path: Path,
    symbol: str,
    project_root: Path,
) -> list[dict]:
    """Return references to `symbol` that live OUTSIDE `file_path`.

    References inside the file being edited are filtered out — they're
    about to go away along with the symbol itself, so they don't
    represent a caller-safety concern. The definition entry is also
    dropped; tldr may include it alongside the usage list.

    On infra failure (tldr missing/erroring) returns an empty list
    (fall-open).

    Each element is the tldr reference dict, augmented with 'file',
    'line', 'kind' keys at minimum.
    """
    file_path = Path(file_path).resolve()
    data = _run_tldr_references_workspace(symbol, project_root)

    out: list[dict] = []
    for ref in data.get("references") or []:
        ref_file = ref.get("file")
        if not ref_file:
            continue
        # Drop self-references: same file as the one being edited.
        try:
            if Path(ref_file).resolve() == file_path:
                continue
        except OSError:
            # Path resolution on weird paths: skip conservatively.
            continue
        # Drop the definition entry explicitly (belt and suspenders —
        # tldr usually exposes it via data["definition"], not as a ref,
        # but some versions emit a "definition"-kind row too).
        if ref.get("kind") == "definition":
            continue
        out.append(ref)
    return out


def format_refusal_message(
    symbol: str,
    refs: list[dict],
    force_hint: str,
) -> str:
    """Build the structured refusal message shown to users.

    `force_hint` is the caller-specific override hint (e.g.
    "Pass force=True (MCP) / --force (CLI) to delete anyway").
    Truncates the reference list at 10 lines.
    """
    files = {r.get("file") for r in refs if r.get("file")}
    lines = [
        f"Refused: '{symbol}' has {len(refs)} reference(s) "
        f"in {len(files)} other file(s):",
    ]
    shown = refs[:10]
    for r in shown:
        fp = r.get("file", "?")
        line_no = r.get("line", "?")
        kind = r.get("kind", "ref")
        lines.append(f"  {fp}:{line_no} [{kind}]")
    remaining = len(refs) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    lines.append(force_hint)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signature detection — outsourced to ``tldr structure``.
#
# Previously this module walked a tree-sitter AST per language, covering
# 6 langs (python / ts / go / rust / java / kotlin). ``tldr structure``
# exposes per-definition (name, kind, line_start, line_end, signature)
# for every language fastedit supports (13/13 verified 2026-04-23).
#
# Hot-path discipline (VAL-M3-002): ``compute_signature_impact_note``
# is called on every fast_edit. A naive "call tldr twice" implementation
# would spawn 2 subprocesses per edit. We use a cheap in-process
# pre-check (:func:`_lines_touching_symbol_changed`) and only shell out
# to tldr when the pre-check suggests something relevant moved.
# ---------------------------------------------------------------------------


def _language_to_suffix(language: str | None) -> str:
    """Map a fastedit/tldr language name to a file suffix for tempfiles.

    Used to satisfy ``tldr structure``'s extension-based language
    detection when we only have in-memory source. Returns ``.txt`` when
    the language is unknown — tldr will then emit an empty structure,
    which :func:`_get_signature_text` treats as "no signature" (returns
    ``None``). Exotic langs therefore silently bypass the signature-
    change gate.
    """
    if not language:
        return ".txt"
    return _LANGUAGE_TO_DEFAULT_SUFFIX.get(language, ".txt")


def _extract_declaration_block(text: str, symbol: str) -> str | None:
    """Return the declaration-plus-signature block for ``symbol``.

    Finds the first line mentioning ``symbol`` at word boundary and
    extends it forward through multi-line parameter lists, stopping at
    whichever comes first:

    * A body-opener (``{``, ``:``, ``;``) at bracket depth 0 — captures
      brace-bodied langs (java/js/ts/go/rust/swift/c/cpp/c#/kotlin/php/
      scala), python/ruby-style ``:``, and c-family forward declarations.
    * End of a line at bracket depth 0 with no body-opener — captures
      Ruby's ``def foo(a)`` where the next line starts the body without
      any delimiter on the signature line.

    This correctly captures multi-line signatures like::

        def multi(
            a,
            b,
        ):

    where the symbol only appears on line 1 but the signature spans 4
    lines (paren depth stays > 0 across the first three newlines). For
    Ruby the same walk terminates at line 1 because paren depth returns
    to 0 before the newline, and there's no `{`/`:` on that line.

    Returns ``None`` when ``symbol`` doesn't appear in ``text`` (e.g.
    the edit removed the definition), which the caller treats as
    "declaration moved — must verify via tldr".
    """
    if not symbol:
        return None
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if pattern.search(ln):
            start = i
            break
    if start is None:
        return None

    paren = 0
    bracket = 0
    brace = 0
    collected: list[str] = []
    max_lines = 50

    for offset in range(max_lines):
        idx = start + offset
        if idx >= len(lines):
            break
        line = lines[idx]
        stopped_mid_line = False
        for j, ch in enumerate(line):
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)
            elif ch == "{":
                if paren == 0 and bracket == 0 and brace == 0:
                    collected.append(line[: j + 1])
                    return "\n".join(collected)
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == ":" and paren == 0 and bracket == 0 and brace == 0:
                collected.append(line[: j + 1])
                return "\n".join(collected)
            elif ch == ";" and paren == 0 and bracket == 0 and brace == 0:
                collected.append(line[: j + 1])
                return "\n".join(collected)
        if stopped_mid_line:
            break
        collected.append(line)
        # End-of-line stop: signature line has closed all brackets and
        # didn't hit a body-opener. Ruby's ``def foo(a)`` terminates
        # here. Python/Java multi-line sigs keep paren > 0 until a later
        # line, so this guard doesn't fire early.
        if paren == 0 and bracket == 0 and brace == 0:
            return "\n".join(collected)

    return "\n".join(collected) if collected else None


def _lines_touching_symbol_changed(
    old_code: str, new_code: str, symbol: str,
) -> bool:
    """Cheap in-process pre-check before shelling out to tldr structure.

    Compares the declaration block (first symbol occurrence extended
    through any multi-line parameter list to the body opener) between
    ``old_code`` and ``new_code``. Returns ``False`` iff the blocks are
    byte-identical — then the signature cannot have moved, and we skip
    the subprocess. Returns ``True`` when the block differs OR when one
    side has no occurrence (symbol removed / renamed) — in both those
    cases a tldr-level check is warranted.

    Correctness-preserving: false positives (subprocess spawned on a
    non-signature edit that happened to touch the declaration block)
    are tolerable overhead; false negatives (signature change missed)
    are impossible because any signature change must alter at least one
    character of the declaration block.
    """
    if old_code == new_code:
        return False
    if not symbol:
        return False
    old_block = _extract_declaration_block(old_code, symbol)
    new_block = _extract_declaration_block(new_code, symbol)
    if old_block is None or new_block is None:
        # Symbol absent on one side — treat as changed (caller will
        # call tldr and see whether the signature actually moved).
        return old_block != new_block
    return old_block != new_block


def _run_tldr_structure(source: str, suffix: str) -> dict | None:
    """Return parsed ``tldr structure --format json`` for ``source``.

    Writes ``source`` to a NamedTemporaryFile with ``suffix`` (so tldr
    picks the right grammar), invokes tldr, and returns the JSON.
    Returns ``None`` on any failure (binary missing, timeout, bad JSON).
    Callers treat ``None`` as "no signature available" and skip the
    signature-change gate — matching the previous fall-closed behaviour
    of the AST walker.
    """
    try:
        # delete=False + manual unlink: we need the file to exist across
        # the subprocess invocation, and tempfile's auto-cleanup on
        # close() races with subprocess on some platforms.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8",
        ) as fh:
            fh.write(source)
            tmp_path = Path(fh.name)
    except OSError:
        return None

    try:
        try:
            result = subprocess.run(
                ["tldr", "structure", str(tmp_path), "--format", "json"],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

        stdout = result.stdout or ""
        start = stdout.find("{")
        if start == -1:
            return None
        try:
            data = json.loads(stdout[start:])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# Characters that close a signature across languages. We scan from the
# declaration's ``line_start`` tracking paren/bracket/brace depth; the
# signature ends when depth returns to 0 AND we encounter one of these
# terminators. Captures python/ruby ``:`` / ``|`` , brace-bodied langs
# ``{`` , arrow-return ``=>`` , rust/swift where-clause ``{`` after any
# return type, and the ``;`` used by c-family forward declarations.
_SIG_TERMINATORS = frozenset({":", "{", ";"})


def _extract_signature_from_source(
    source: str, line_start_1: int,
) -> str | None:
    """Return the signature span starting at 1-indexed ``line_start_1``.

    Walks forward from the declaration line character by character,
    tracking ``()``, ``[]``, ``{}`` depth. Stops at whichever terminator
    comes first:

    * ``{`` at depth 0 — brace-bodied langs (java/js/ts/go/rust/swift/
      c/cpp/c#/kotlin/php/scala). Included in the signature (consistent
      with how ``tldr structure`` emits ``public class Foo {``).
    * ``:`` at depth 0 — python/ruby body opener. Included.
    * ``;`` at depth 0 — c-family forward declaration. Included.
    * End of line with all brackets closed AND no terminator hit yet —
      ruby's ``def foo(a)`` where the body begins on the next line
      without any delimiter on the signature line itself.

    Returns ``None`` when ``line_start_1`` is out of range. Multi-line
    parameter lists are handled naturally: paren depth stays > 0 across
    newlines, preventing the EOL stop from firing early.
    """
    lines = source.splitlines(keepends=True)
    start_idx = line_start_1 - 1
    if start_idx < 0 or start_idx >= len(lines):
        return None

    paren = 0
    bracket = 0
    brace = 0
    collected: list[str] = []
    max_lines = 50

    for offset in range(max_lines):
        idx = start_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx]
        for i, ch in enumerate(line):
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)
            elif ch == "{":
                if paren == 0 and bracket == 0 and brace == 0:
                    collected.append(line[: i + 1])
                    return "".join(collected).strip()
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == ":" and paren == 0 and bracket == 0 and brace == 0:
                collected.append(line[: i + 1])
                return "".join(collected).strip()
            elif ch == ";" and paren == 0 and bracket == 0 and brace == 0:
                collected.append(line[: i + 1])
                return "".join(collected).strip()
        collected.append(line)
        # End-of-line stop when all brackets are balanced: ruby-style
        # ``def foo(a)`` where body begins on the next line. Does not
        # fire on multi-line parameter lists (paren > 0).
        if paren == 0 and bracket == 0 and brace == 0:
            return "".join(collected).strip()

    text = "".join(collected).strip()
    return text or None


def _get_signature_text(
    source: str, symbol: str, language: str | None,
) -> str | None:
    """Return the canonical signature string of ``symbol`` in ``source``.

    Uses ``tldr structure`` to locate the definition (name + line_start),
    then re-extracts the signature text directly from ``source`` via
    :func:`_extract_signature_from_source`. This handles multi-line
    parameter lists that tldr's own ``signature`` field truncates to the
    first line — we need the full span so a param-addition edit that
    adds a new argument on a new line still compares as changed.

    Returns ``None`` when:
    - ``symbol`` is falsy
    - tldr is unavailable or errors
    - no definition with ``name == symbol`` is found
    - the signature span can't be extracted

    The returned string is trimmed of leading/trailing whitespace so
    string equality between two revisions is meaningful.
    """
    if not symbol:
        return None
    suffix = _language_to_suffix(language)
    data = _run_tldr_structure(source, suffix)
    if data is None:
        return None

    files = data.get("files") or []
    for file_entry in files:
        for defn in file_entry.get("definitions") or []:
            if defn.get("name") != symbol:
                continue
            line_start = defn.get("line_start")
            if not isinstance(line_start, int) or line_start < 1:
                continue
            sig = _extract_signature_from_source(source, line_start)
            if sig:
                return sig
            # Fall back to tldr's own signature field if source-level
            # extraction failed (e.g. truncated source).
            tldr_sig = defn.get("signature")
            if isinstance(tldr_sig, str) and tldr_sig.strip():
                return tldr_sig.strip()
    return None


def signature_changed(
    old_code: str, new_code: str, symbol: str, language: str | None,
) -> bool:
    """Return True iff the signature text of ``symbol`` differs between
    ``old_code`` and ``new_code``.

    "Signature" means the def/fn/class declaration span up to (and
    including) the body opener — ``:`` for python/ruby, ``{`` for
    brace-bodied langs, ``;`` for forward decls. Parameter changes,
    name changes, decorators on the same line, return-type annotations,
    and visibility modifiers all count. Body-only changes do not.

    Hot path (VAL-M3-002): we run a cheap in-process pre-check first
    (:func:`_lines_touching_symbol_changed`). When no line mentioning
    ``symbol`` has changed between old and new, the declaration can't
    have moved either, so we return ``False`` without spawning any
    subprocess. Only when the pre-check flags a potential change do we
    shell out to ``tldr structure``.

    When either side lacks the symbol after the pre-check passes
    (removed wholesale, renamed, or unsupported language) we return
    ``False`` — the caller uses this signal only to decide whether to
    spend the cross-file caller scan, so being conservative keeps the
    hot path free on exotic edges.
    """
    if not _lines_touching_symbol_changed(old_code, new_code, symbol):
        return False
    old_sig = _get_signature_text(old_code, symbol, language)
    new_sig = _get_signature_text(new_code, symbol, language)
    if old_sig is None or new_sig is None:
        return False
    return old_sig != new_sig


def compute_signature_impact_note(
    *,
    old_code: str,
    new_code: str,
    symbol: str | None,
    language: str | None,
    file_path: Path,
    project_root: Path,
) -> str | None:
    """Build the fast_edit pre-flight impact note, or return ``None``.

    Returns ``None`` when any of the following holds:
    - ``symbol`` is falsy (no ``replace=`` target)
    - the symbol's signature is unchanged (hot path — no callers scan)
    - zero cross-file callers exist for the symbol

    Otherwise returns a single-line advisory the caller appends to the
    normal edit-success message (VAL-M3-001). Cross-file callers are
    counted via :func:`check_cross_file_callers`, which filters out
    self-references (same file as the edit) and definition entries.

    The note format::

        note: signature of 'X' changed — X has N caller(s) in M
        file(s). Run `tldr impact X <project_root>` or `fast_search X`
        to review.
    """
    if not symbol:
        return None
    if not signature_changed(old_code, new_code, symbol, language):
        return None
    refs = check_cross_file_callers(
        file_path=Path(file_path),
        symbol=symbol,
        project_root=Path(project_root),
    )
    if not refs:
        return None
    files = {r.get("file") for r in refs if r.get("file")}
    n_refs = len(refs)
    n_files = len(files)
    return (
        f"note: signature of '{symbol}' changed — "
        f"{symbol} has {n_refs} caller(s) in {n_files} file(s). "
        f"Run `tldr impact {symbol} {project_root}` or "
        f"`fast_search {symbol}` to review."
    )
