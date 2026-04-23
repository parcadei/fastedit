"""Tests for v0.2.4 marker improvements.

Covers:
  1. Marker-position semantics — snippet with a marker but zero body
     anchors uses marker placement to infer top vs bottom insertion.
  2. Short marker forms — ``#...``, ``//...``, ``…`` are normalized to
     the canonical long form before any downstream processing.
  3. The combination — a short marker with position semantics.
  4. Cross-language coverage via Rust.
"""

from __future__ import annotations

import textwrap

from fastedit.inference.text_match import (
    _normalize_markers,
    deterministic_edit,
)


# ---------------------------------------------------------------------------
# Marker-position semantics
# ---------------------------------------------------------------------------

class TestMarkerPositionSemantics:
    """Marker placement implies top vs bottom insertion when a snippet
    has no body anchors."""

    def test_marker_at_end_inserts_at_top(self):
        """Snippet is ``<new> + marker`` with no body anchors →
        new lines go at the TOP of the function body, rest preserved."""
        original = textwrap.dedent("""\
            def handle(request):
                validated = validate(request)
                processed = process(validated)
                return processed
            """)
        snippet = textwrap.dedent("""\
            def handle(request):
                log.info("handling request")
                audit.record(request)
                # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "def handle(request):"
        assert lines[1] == '    log.info("handling request")'
        assert lines[2] == "    audit.record(request)"
        assert lines[3] == "    validated = validate(request)"
        assert lines[4] == "    processed = process(validated)"
        assert lines[5] == "    return processed"

    def test_marker_at_start_inserts_at_bottom(self):
        """Snippet is ``marker + <new>`` with no body anchors →
        new lines go at the BOTTOM of the function body."""
        original = textwrap.dedent("""\
            def handle(request):
                validated = validate(request)
                processed = process(validated)
                return processed
            """)
        snippet = textwrap.dedent("""\
            def handle(request):
                # ... existing code ...
                metrics.increment("handle.count")
                telemetry.flush()
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "def handle(request):"
        assert lines[1] == "    validated = validate(request)"
        assert lines[2] == "    processed = process(validated)"
        assert lines[3] == "    return processed"
        assert lines[4] == '    metrics.increment("handle.count")'
        assert lines[5] == "    telemetry.flush()"

    def test_marker_with_anchors_still_uses_anchors(self):
        """Snippet with marker AND body anchors uses the existing
        anchor-based logic (no regression from the new path)."""
        original = textwrap.dedent("""\
            def compute(x):
                a = step_one(x)
                b = step_two(a)
                c = step_three(b)
                return c
            """)
        snippet = textwrap.dedent("""\
            def compute(x):
                a = step_one(x)
                extra = intermediate(a)
                # ... existing code ...
                return c
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        assert "a = step_one(x)" in result
        assert "extra = intermediate(a)" in result
        assert "b = step_two(a)" in result
        assert "c = step_three(b)" in result
        # New line sits between the first anchor and the preserved gap.
        lines = result.splitlines()
        idx_a = lines.index("    a = step_one(x)")
        idx_extra = lines.index("    extra = intermediate(a)")
        idx_b = lines.index("    b = step_two(a)")
        assert idx_a < idx_extra < idx_b

    def test_snippet_with_no_marker_still_full_replace(self):
        """Snippet with no marker and no body anchors → the existing
        ``< 2 anchors`` rejection still fires (falls back to model)."""
        original = textwrap.dedent("""\
            def foo():
                x = 1
                y = 2
                return x + y
            """)
        # No marker, no anchor overlap → should fall back to None
        # (original behavior preserved; only the marker-present case
        # engages position semantics).
        snippet = textwrap.dedent("""\
            def foo():
                completely_different()
                also_new()
            """)
        result = deterministic_edit(original, snippet)
        assert result is None


# ---------------------------------------------------------------------------
# Short marker normalization
# ---------------------------------------------------------------------------

class TestNormalizeMarkers:
    """Short / Unicode marker forms normalize to canonical long form."""

    def test_short_hash_dots_normalizes(self):
        out = _normalize_markers("    #...\n")
        assert out == "    # ... existing code ...\n"

    def test_short_slash_dots_normalizes(self):
        out = _normalize_markers("    //...\n")
        assert out == "    // ... existing code ...\n"

    def test_unicode_ellipsis_normalizes(self):
        out = _normalize_markers("    …\n")
        assert out == "    # ... existing code ...\n"

    def test_legacy_long_hash_passes_through(self):
        line = "    # ... existing code ...\n"
        assert _normalize_markers(line) == line

    def test_legacy_long_slash_passes_through(self):
        line = "    // ... existing code ...\n"
        assert _normalize_markers(line) == line

    def test_regular_code_untouched(self):
        line = "x = foo(bar)\n"
        assert _normalize_markers(line) == line

    def test_multiline_preserves_indentation(self):
        snippet = "def foo():\n    #...\n    return 1\n"
        out = _normalize_markers(snippet)
        assert out == (
            "def foo():\n"
            "    # ... existing code ...\n"
            "    return 1\n"
        )


# ---------------------------------------------------------------------------
# Short markers end-to-end with deterministic_edit
# ---------------------------------------------------------------------------

class TestShortMarkersEndToEnd:
    """Short markers must work with marker-position semantics after
    normalization — the caller sees full short-form support."""

    def test_short_python_marker_hash_dots(self):
        """``#...`` at end of snippet inserts new lines at top."""
        original = textwrap.dedent("""\
            def handle(request):
                validated = validate(request)
                return validated
            """)
        snippet = textwrap.dedent("""\
            def handle(request):
                log.info("top-hook")
                #...
            """)
        # chunked_merge normalizes; for unit test, normalize manually.
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "def handle(request):"
        assert lines[1] == '    log.info("top-hook")'
        assert lines[2] == "    validated = validate(request)"
        assert lines[3] == "    return validated"

    def test_short_js_marker_slash_slash_dots(self):
        """``//...`` at start of snippet inserts new lines at bottom."""
        original = textwrap.dedent("""\
            function handle(req) {
                const v = validate(req);
                return v;
            }
            """)
        snippet = textwrap.dedent("""\
            function handle(req) {
                //...
                metrics.bump("handle");
            }
            """)
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None
        # Bottom insertion: new line goes after the preserved body
        # (right before the closing brace, which is a trailing anchor).
        # Because ``}`` is ambiguous, it won't match as an anchor — so
        # marker-position mode triggers and new goes at the bottom.
        assert 'metrics.bump("handle")' in result
        assert "const v = validate(req);" in result
        assert "return v;" in result

    def test_unicode_ellipsis_marker(self):
        """Unicode ellipsis (U+2026) is recognized universally."""
        original = textwrap.dedent("""\
            def process(data):
                cleaned = sanitize(data)
                return cleaned
            """)
        snippet = "def process(data):\n    log.info(\"start\")\n    …\n"
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None
        lines = result.splitlines()
        assert lines[1] == '    log.info("start")'
        assert lines[2] == "    cleaned = sanitize(data)"

    def test_legacy_long_marker_still_works(self):
        """Legacy long-form markers continue to function."""
        original = textwrap.dedent("""\
            def foo():
                a = 1
                b = 2
                return a + b
            """)
        snippet = textwrap.dedent("""\
            def foo():
                log.debug("entry")
                # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        assert lines[1] == '    log.debug("entry")'
        assert lines[2] == "    a = 1"
        assert lines[3] == "    b = 2"

    def test_short_marker_with_position_semantics(self):
        """``<new> + #...`` combination — short form + top insertion."""
        original = textwrap.dedent("""\
            def compute(x):
                y = x * 2
                z = y + 1
                return z
            """)
        snippet = textwrap.dedent("""\
            def compute(x):
                assert x is not None
                #...
            """)
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None
        lines = result.splitlines()
        assert lines[1] == "    assert x is not None"
        assert lines[2] == "    y = x * 2"
        assert lines[3] == "    z = y + 1"
        assert lines[4] == "    return z"


# ---------------------------------------------------------------------------
# Cross-language
# ---------------------------------------------------------------------------

class TestCrossLanguage:
    """Rust fixture exercising ``//...`` + marker position semantics."""

    def test_rust_short_slash_marker_top_insert(self):
        original = textwrap.dedent("""\
            fn handle(req: Request) -> Response {
                let validated = validate(req);
                let processed = process(validated);
                processed
            }
            """)
        snippet = textwrap.dedent("""\
            fn handle(req: Request) -> Response {
                log::info!("handling");
                //...
            }
            """)
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None
        assert 'log::info!("handling")' in result
        assert "let validated = validate(req);" in result
        assert "let processed = process(validated);" in result
