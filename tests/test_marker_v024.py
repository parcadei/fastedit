"""Tests for v0.2.4 marker improvements.

Part 1 — marker-position semantics: snippet with a marker but zero body
anchors uses marker placement to infer top vs bottom insertion.
"""

from __future__ import annotations

import textwrap

from fastedit.inference.text_match import deterministic_edit


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
