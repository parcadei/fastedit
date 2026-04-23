"""Regression tests for v0.2.5 add-guard refinement of the wrap_block guard.

The 0.2.4 wrap_block guard (introduced to prevent flat-peer insertion when
the user really meant to wrap the body in a new scope) was over-aggressive.
It fired whenever the last new-line before the marker ended with ``:`` or
``{``, which caught the very common "add-guard" pattern — inserting an
early-return guard at the top of a function.

The 0.2.5 refinement compares indent alignment between the block-opener
and the marker:

  * marker_indent  >  opener_indent → genuine wrap_block (fall through
    to model).
  * marker_indent <= opener_indent → add-guard (deterministic path
    proceeds).

These tests lock in both halves of the behavior: add-guard patterns must
produce correct output via the deterministic path, and genuine
wrap_block patterns must continue to fall through.
"""

from __future__ import annotations

import textwrap

from fastedit.inference.text_match import (
    _normalize_markers,
    deterministic_edit,
)


class TestAddGuardDeterministic:
    """Add-guard patterns must run on the deterministic path (0 tokens)."""

    def test_add_guard_python_if_block(self):
        """Canonical failing case from the 0.2.5 bug report.

        Inserting an ``if not records: return []`` guard at the top of a
        function that begins with a ``while`` loop. Before 0.2.5 this
        hit the wrap_block guard (``if not records:`` ends with ``:``)
        and fell through to the model. Now the indent comparison
        (opener at 4, marker at 4 → same indent → add-guard) sends it
        down the deterministic top-insert path.
        """
        original = textwrap.dedent("""\
            def retry_batch(client, records, max_attempts=3):
                attempt = 0
                while attempt < max_attempts:
                    try:
                        return client.send(records)
                    except TransientError:
                        attempt += 1
                raise RuntimeError("exhausted retries")
            """)
        snippet = textwrap.dedent("""\
            def retry_batch(client, records, max_attempts=3):
                if not records:
                    return []
                # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None, (
            "Expected deterministic path to handle add-guard pattern; "
            "got None (would route to model)."
        )
        lines = result.splitlines()
        # Signature preserved, guard at top, original body intact.
        assert lines[0] == "def retry_batch(client, records, max_attempts=3):"
        assert lines[1] == "    if not records:"
        assert lines[2] == "        return []"
        assert lines[3] == "    attempt = 0"
        assert lines[4] == "    while attempt < max_attempts:"
        # Spot-check that nothing downstream was lost.
        assert 'raise RuntimeError("exhausted retries")' in result
        assert "except TransientError:" in result

    def test_add_guard_with_return_guard(self):
        """Early-return guard — simplest form of add-guard."""
        original = textwrap.dedent("""\
            def serialize(obj):
                data = dict(obj.__dict__)
                data["_type"] = type(obj).__name__
                return json.dumps(data)
            """)
        snippet = textwrap.dedent("""\
            def serialize(obj):
                if obj is None:
                    return "null"
                # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "def serialize(obj):"
        assert lines[1] == "    if obj is None:"
        assert lines[2] == '        return "null"'
        assert lines[3] == "    data = dict(obj.__dict__)"
        assert lines[4] == '    data["_type"] = type(obj).__name__'
        assert lines[5] == "    return json.dumps(data)"

    def test_add_guard_javascript(self):
        """C-family variant — ``{`` opener with same-indent marker."""
        original = textwrap.dedent("""\
            function fetchData(url) {
              const response = request(url);
              return response.body;
            }
            """)
        snippet = textwrap.dedent("""\
            function fetchData(url) {
              if (!url) {
                return null;
              }
              // ... existing code ...
            }
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None, (
            "Expected deterministic path to handle JS add-guard; "
            "got None."
        )
        # Guard should be inserted at the top of the body; original
        # body must be preserved verbatim below it.
        assert "if (!url) {" in result
        assert "return null;" in result
        assert "const response = request(url);" in result
        assert "return response.body;" in result
        # Order check: guard appears before the original body.
        idx_guard = result.index("if (!url)")
        idx_body = result.index("const response = request(url)")
        assert idx_guard < idx_body

    def test_nested_add_guard_python(self):
        """Add-guard whose body itself contains a nested ``if``.

        The block-opener for guard purposes is the LAST new-line before
        the marker. Nested openers earlier in ``new_before`` don't
        change classification — the outer guard (last non-marker new
        line) is what we compare against the marker indent.
        """
        original = textwrap.dedent("""\
            def handle_event(event):
                result = dispatch(event)
                return result
            """)
        snippet = textwrap.dedent("""\
            def handle_event(event):
                if event is None:
                    if strict_mode:
                        raise ValueError("event required")
                    return None
                # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "def handle_event(event):"
        assert lines[1] == "    if event is None:"
        assert lines[2] == "        if strict_mode:"
        assert lines[3] == '            raise ValueError("event required")'
        assert lines[4] == "        return None"
        assert lines[5] == "    result = dispatch(event)"
        assert lines[6] == "    return result"

    def test_add_guard_with_short_marker(self):
        """Short-marker form (``#...``) must compose with the refined guard."""
        original = textwrap.dedent("""\
            def compute(values):
                total = sum(values)
                return total / len(values)
            """)
        snippet = textwrap.dedent("""\
            def compute(values):
                if not values:
                    return 0
                #...
            """)
        normalized = _normalize_markers(snippet)
        result = deterministic_edit(original, normalized)
        assert result is not None, (
            "Short marker + add-guard should still hit deterministic path."
        )
        lines = result.splitlines()
        assert lines[0] == "def compute(values):"
        assert lines[1] == "    if not values:"
        assert lines[2] == "        return 0"
        assert lines[3] == "    total = sum(values)"
        assert lines[4] == "    return total / len(values)"


class TestGenuineWrapBlockStillFallsThrough:
    """Critical: genuine wrap_block patterns MUST still route to the model.

    The fix must not weaken the guard for real wrap cases (marker is
    nested inside the opener). These tests would pass even before 0.2.5;
    they lock in that 0.2.5 didn't break them.
    """

    def test_genuine_wrap_block_still_routes(self):
        """Python ``with`` wrap-block: marker indented deeper than opener."""
        original = textwrap.dedent("""\
            def critical_section():
                mutate_shared_state()
                update_counter()
                return True
            """)
        # Marker is at indent 8, wrapped INSIDE `with lock:` at indent 4.
        snippet = textwrap.dedent("""\
            def critical_section():
                with lock:
                    # ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        # Must fall through — wrapping requires semantic understanding
        # the deterministic path can't supply.
        assert result is None, (
            "Genuine wrap_block must fall through to model. "
            f"Got deterministic output: {result!r}"
        )

    def test_genuine_wrap_block_javascript(self):
        """C-family wrap_block via the position-mode path.

        Construct a snippet that cleanly enters position mode (single
        marker, all new lines BEFORE the marker, zero body anchors).
        The opener ``try {`` is at indent 2; the marker is at indent 4
        (deeper — nested inside the opened scope). The refined guard
        must recognize this as genuine wrap_block and decline,
        preserving the 0.2.4 contract that actual wrap_block patterns
        route to the model.
        """
        original = textwrap.dedent("""\
            function process(data) {
              const result = transform(data);
              return persist(result);
            }
            """)
        # Position-mode shape: no trailing closer in the snippet, so
        # the last-line ``}`` anchor the standard path would use is
        # absent. All new content is BEFORE the marker, marker is
        # nested deeper than the opener → wrap_block.
        snippet = textwrap.dedent("""\
            function process(data) {
              try {
                // ... existing code ...
            """)
        result = deterministic_edit(original, snippet)
        assert result is None, (
            "Genuine JS wrap_block (marker nested deeper than opener) "
            f"must fall through to model. Got: {result!r}"
        )
