"""Tests for the direct-symbol-swap fast path in chunked_merge.

When `replace=X` is used and the snippet is a complete new definition of X
(no markers, parses cleanly, exactly one top-level definition whose name
is X), chunked_merge should replace the resolved function region directly
with the snippet — zero model tokens, pure AST boundary splice.

This path runs AFTER deterministic_edit (so line-level edits with context
anchors still use the cheaper, more conservative path) but BEFORE any
model fallback.
"""

from __future__ import annotations

import logging

import pytest

from fastedit.inference.chunked_merge import chunked_merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_model(*_args, **_kwargs):
    """Merge fn that fails loudly if called — direct-swap must short-circuit."""
    raise AssertionError(
        "merge_fn should NOT be called — direct-swap must short-circuit "
        "before reaching the model"
    )


# ---------------------------------------------------------------------------
# 1. Direct swap happy path — change_signature (Python)
# ---------------------------------------------------------------------------

PY_ORIGINAL = """\
import os


def greet(name):
    return "hello " + name


def render_template(template, context=None):
    ctx = context or {}
    return env.get_template(template).render(**ctx)


def farewell(name):
    return "bye " + name
"""

PY_SNIPPET_FULL_REPLACEMENT = """\
def render_template(template: str, context: dict | None = None, **kwargs) -> str:
    ctx = {**context, **kwargs} if context else kwargs
    return env.get_template(template).render(**ctx)
"""

PY_EXPECTED = """\
import os


def greet(name):
    return "hello " + name


def render_template(template: str, context: dict | None = None, **kwargs) -> str:
    ctx = {**context, **kwargs} if context else kwargs
    return env.get_template(template).render(**ctx)


def farewell(name):
    return "bye " + name
"""


def test_direct_swap_change_signature_python(tmp_path):
    """replace='render_template' with a full new definition:
    every body line changed → deterministic_edit returns None →
    direct-swap fast-path fires → zero model tokens."""
    file_path = tmp_path / "templates.py"
    file_path.write_text(PY_ORIGINAL)

    result = chunked_merge(
        original_code=PY_ORIGINAL,
        snippet=PY_SNIPPET_FULL_REPLACEMENT,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="render_template",
    )

    assert result.model_tokens == 0, "direct-swap must not call the model"
    assert result.chunks_used == 0
    assert result.merged_code == PY_EXPECTED


# ---------------------------------------------------------------------------
# 2. Direct swap — extend_literal (Rust)
# ---------------------------------------------------------------------------

RS_ORIGINAL = """\
use std::path::Path;

pub const SUPPORTED_EXTS: &[&str] = &[".txt", ".md"];

pub fn pick(p: &Path) -> bool {
    SUPPORTED_EXTS.iter().any(|e| p.to_string_lossy().ends_with(e))
}
"""

RS_SNIPPET_FULL_REPLACEMENT = """\
pub const SUPPORTED_EXTS: &[&str] = &[".txt", ".md", ".rst", ".adoc"];
"""

RS_EXPECTED = """\
use std::path::Path;

pub const SUPPORTED_EXTS: &[&str] = &[".txt", ".md", ".rst", ".adoc"];

pub fn pick(p: &Path) -> bool {
    SUPPORTED_EXTS.iter().any(|e| p.to_string_lossy().ends_with(e))
}
"""


def test_direct_swap_extend_literal_rust(tmp_path):
    """replace='SUPPORTED_EXTS' with a full new const definition:
    single-line body fully replaced → direct-swap fires → zero model tokens."""
    file_path = tmp_path / "ext.rs"
    file_path.write_text(RS_ORIGINAL)

    result = chunked_merge(
        original_code=RS_ORIGINAL,
        snippet=RS_SNIPPET_FULL_REPLACEMENT,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="rust",
        replace="SUPPORTED_EXTS",
    )

    assert result.model_tokens == 0, "direct-swap must not call the model"
    assert result.chunks_used == 0
    assert result.merged_code == RS_EXPECTED


# ---------------------------------------------------------------------------
# 3. Regression guard — snippet with markers does NOT direct-swap
# ---------------------------------------------------------------------------

PY_ORIGINAL_FOR_MARKER = """\
def process_file(path, debug=False):
    if debug:
        print(f"starting: {path}")
    data = open(path).read()
    lines = data.splitlines()
    result = []
    for line in lines:
        if line.strip():
            result.append(line.upper())
    return result
"""

# Snippet with a marker — adds a new first line, preserves the rest via marker.
# This has 2+ context anchors (the lines around the marker) so
# deterministic_edit should handle it.
PY_MARKER_SNIPPET = """\
def process_file(path, debug=False):
    if debug:
        print(f"starting: {path}")
    open_time = time.time()
    data = open(path).read()
    # ... existing code ...
"""


def test_direct_swap_skipped_when_snippet_has_markers(tmp_path):
    """A snippet with '# ... existing code ...' implies 'preserve original chunks'
    which is incompatible with whole-symbol swap. The direct-swap fast-path
    must NOT fire; deterministic_edit should handle this (it has anchors)."""
    file_path = tmp_path / "proc.py"
    file_path.write_text(PY_ORIGINAL_FOR_MARKER)

    # Deterministic_edit should succeed on this (markers + anchors).
    # If direct-swap fired instead, the final code would have replaced the
    # whole function with just the snippet — losing the body after the marker.
    result = chunked_merge(
        original_code=PY_ORIGINAL_FOR_MARKER,
        snippet=PY_MARKER_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="process_file",
    )

    # Must have preserved the original body lines (the for-loop)
    assert "for line in lines:" in result.merged_code, (
        "Direct-swap must NOT have run on a marker-containing snippet — "
        "body lines lost"
    )
    assert "result.append(line.upper())" in result.merged_code
    # And added the new line
    assert "open_time = time.time()" in result.merged_code
    assert result.model_tokens == 0


# ---------------------------------------------------------------------------
# 4. Regression guard — snippet with wrong symbol name is NOT swapped
# ---------------------------------------------------------------------------

PY_WRONG_NAME_SNIPPET = """\
def totally_different_function(x):
    return x * 2
"""


def test_direct_swap_rejects_wrong_symbol_name(tmp_path):
    """replace='render_template' but snippet defines 'totally_different_function':
    existing extras-check must raise ValueError before direct-swap even runs."""
    file_path = tmp_path / "templates.py"
    file_path.write_text(PY_ORIGINAL)

    with pytest.raises(ValueError) as exc_info:
        chunked_merge(
            original_code=PY_ORIGINAL,
            snippet=PY_WRONG_NAME_SNIPPET,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="python",
            replace="render_template",
        )

    msg = str(exc_info.value)
    assert "totally_different_function" in msg, (
        f"error should mention the extra symbol: {msg}"
    )


# ---------------------------------------------------------------------------
# 5. Ordering — deterministic_edit wins when it can anchor
# ---------------------------------------------------------------------------

PY_ORIGINAL_FOR_GUARD = """\
def login(user, pw):
    token = auth(user, pw)
    session.store(token)
    return token
"""

# add_guard pattern: 2 anchors (def line + auth line), one new line between
PY_GUARD_SNIPPET = """\
def login(user, pw):
    if not user:
        raise ValueError("user required")
    token = auth(user, pw)
"""

PY_GUARD_EXPECTED_BODY_PRESERVED = "    session.store(token)"


def test_deterministic_edit_wins_when_it_can_anchor(tmp_path):
    """When the snippet has 2+ context anchors (add_guard pattern), the
    deterministic_edit path must fire BEFORE direct-swap. The proof:
    the unchanged body lines (session.store, return token) are preserved
    exactly — direct-swap would have dropped them because the snippet
    doesn't contain them.
    """
    file_path = tmp_path / "auth.py"
    file_path.write_text(PY_ORIGINAL_FOR_GUARD)

    result = chunked_merge(
        original_code=PY_ORIGINAL_FOR_GUARD,
        snippet=PY_GUARD_SNIPPET,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="python",
        replace="login",
    )

    assert result.model_tokens == 0
    assert result.chunks_used == 0
    # Proof deterministic_edit ran (not direct-swap):
    # the body lines not in the snippet were preserved.
    assert PY_GUARD_EXPECTED_BODY_PRESERVED in result.merged_code, (
        "deterministic_edit must have preserved the unchanged body lines — "
        "direct-swap would have dropped them"
    )
    assert "return token" in result.merged_code
    # And the guard was inserted
    assert 'raise ValueError("user required")' in result.merged_code


# ---------------------------------------------------------------------------
# 6. Log check: direct-swap emits a recognizable log message
# ---------------------------------------------------------------------------

def test_direct_swap_emits_log_message(tmp_path, caplog):
    """Direct-swap should log a message identifying the fast path."""
    file_path = tmp_path / "templates.py"
    file_path.write_text(PY_ORIGINAL)

    with caplog.at_level(logging.INFO, logger="fastedit.chunked_merge"):
        result = chunked_merge(
            original_code=PY_ORIGINAL,
            snippet=PY_SNIPPET_FULL_REPLACEMENT,
            file_path=str(file_path),
            merge_fn=_no_model,
            language="python",
            replace="render_template",
        )

    assert result.model_tokens == 0
    # A log line mentioning direct-swap
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Direct-swap" in joined or "direct-swap" in joined.lower()
