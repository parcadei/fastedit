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
import subprocess
from pathlib import Path

# Project-root markers, in priority order. First hit wins when walking up
# from the file being edited.
_PROJECT_ROOT_MARKERS = (".git", "pyproject.toml", "package.json")


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



def _get_signature_text(
    source: str, symbol: str, language: str | None,
) -> str | None:
    """Return the signature span of ``symbol`` in ``source``.

    The signature runs from the function/class node's start byte up to
    (but not including) either its body child or the first comment
    sitting between the signature punctuation and the body. This
    correctly handles multi-line parameter lists::

        def foo(
            a,
            b,
        ):

    while excluding body-adjacent comments that tree-sitter parents
    under the function node. Returns ``None`` when the language is
    unsupported, parsing fails, no function/class named ``symbol``
    exists, or the matched node has no identifiable body child. The
    returned string is trimmed and newline-terminated so string equality
    between two revisions is meaningful.
    """
    if not language or not symbol:
        return None
    try:
        from ..data_gen.ast_analyzer import analyze_file, parse_code
    except Exception:
        return None

    try:
        structure = analyze_file(source, language)
    except Exception:
        return None

    candidates = list(structure.functions) + list(structure.classes)
    target = None
    for node in candidates:
        if node.name == symbol:
            target = node
            break
    if target is None:
        return None

    try:
        tree = parse_code(source, language)
    except Exception:
        return None

    src_bytes = source.encode("utf-8")
    target_row = target.start_line - 1  # 0-indexed rows

    def walk(node):
        if node.start_point[0] == target_row:
            body = node.child_by_field_name("body")
            if body is None:
                for child in node.children:
                    if child.type in (
                        "block",
                        "function_body",
                        "class_body",
                        "do_block",
                        "compound_statement",
                    ):
                        body = child
                        break
            if body is not None and body.start_byte > node.start_byte:
                end_byte = body.start_byte
                for child in node.children:
                    if child == body:
                        break
                    # A comment between signature punctuation and body
                    # belongs to the body, not the signature.
                    if child.type == "comment":
                        end_byte = child.start_byte
                        break
                    end_byte = child.end_byte
                return src_bytes[node.start_byte:end_byte].decode(
                    "utf-8", errors="replace",
                )
        for child in node.children:
            if child.start_point[0] > target.end_line - 1:
                break
            if child.end_point[0] < target_row:
                continue
            result = walk(child)
            if result is not None:
                return result
        return None

    sig = walk(tree.root_node)
    if sig is None:
        return None
    return sig.strip() + "\n"


def signature_changed(
    old_code: str, new_code: str, symbol: str, language: str | None,
) -> bool:
    """Return True iff the signature text of ``symbol`` differs between
    ``old_code`` and ``new_code``.

    "Signature" means the def/fn/class declaration span up to the body,
    so parameter changes, name changes, decorators on the same line,
    return-type annotations, and visibility modifiers all count.
    Body-only changes do not.

    When either side lacks the symbol (removed wholesale, renamed, or
    unsupported language) we return ``False`` — the caller uses this
    signal only to decide whether to spend the extra tldr subprocess,
    so being conservative keeps the hot path free on exotic edges.
    """
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
    - the symbol's signature is unchanged (hot path — no tldr call)
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
