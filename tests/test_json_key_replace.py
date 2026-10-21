"""Issue #4: JSON key-level edits on package.json.

The reported failures, per shape:

* a bare value snippet (``"new"`` or ``new``) was REFUSED — the parse gate
  called ``new`` invalid JSON and the definition-line guard called ``"new"``
  signature-less;
* an object snippet (``{"name": "new"}``) either fell to the model or — on
  the direct-swap path — replaced the WHOLE file (a compact file's pairs
  share one line, so the pair's "span" was the entire document), losing
  every sibling key;
* multi-line snippets made the model retry and return the file unchanged.

The fix is a JSON-aware key replacement path: when the target language is
json and ``--replace <key>`` resolves to a ``pair`` node, a snippet that is
a bare JSON VALUE (validated via a synthetic ``{"k": <snippet>}`` wrapper —
no document parse required) splices ONLY the pair's value (and the key when
the snippet restates ``"key": value`` with a different key). Sibling keys,
commas, indentation, the trailing-newline state and the file's compact /
pretty style are preserved by construction: the splice replaces exactly the
value node's byte span, and a JSON serializer never emits commas.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastedit.cli import _try_deterministic_replace
from fastedit.inference.chunked_merge import _try_json_key_replace
from fastedit.mcp.backup import BackupStore

PRETTY = """{
  "name": "old-pkg",
  "version": "1.0.0",
  "private": true,
  "config": {
    "debug": false,
    "level": 3
  },
  "tags": [
    "a",
    "b"
  ]
}
"""

COMPACT = '{"name": "old-pkg","version": "1.0.0","private": true}\n'


def _edit(code: str, snippet: str, replace: str, suffix: str = ".json") -> str:
    """Run the CLI's deterministic replace on an in-memory file; return the
    merged file text. The model path is never reached (the JSON fast path
    returns before it, and merge_fn is not installed here)."""
    path = Path("/virtual") / f"package{suffix}"
    lines = code.splitlines(keepends=True)
    result = _try_deterministic_replace(
        path, code, lines, snippet, replace, "json", BackupStore(),
    )
    assert result is not None, (
        f"JSON key replace for {replace!r} with snippet {snippet!r} fell "
        f"through to the model path"
    )
    assert result.parse_valid
    return result.merged_code


# ---------------------------------------------------------------------------
# Bare JSON value snippets splice ONLY the value
# ---------------------------------------------------------------------------


class TestBareValueSnippets:
    def test_pretty_string_value_change_keeps_siblings_byte_exact(self):
        merged = _edit(PRETTY, '"new-pkg"', "name")
        assert '"name": "new-pkg",' in merged
        # Every sibling byte-exact.
        assert '"version": "1.0.0",' in merged
        assert '"private": true,' in merged
        assert '"debug": false,' in merged
        assert '"level": 3' in merged
        assert '"a",' in merged and '"b"' in merged
        # Exactly one "name" key — no duplication, no wipe.
        assert merged.count('"name"') == 1
        # Document still parses and the value really changed.
        assert json.loads(merged)["name"] == "new-pkg"
        assert json.loads(merged)["config"]["level"] == 3

    def test_bare_unquoted_string_becomes_a_json_string(self):
        merged = _edit(PRETTY, "new-pkg", "name")
        assert json.loads(merged)["name"] == "new-pkg"

    def test_number_value(self):
        merged = _edit(PRETTY, "42", "version")
        assert json.loads(merged)["version"] == 42

    def test_bool_value(self):
        merged = _edit(PRETTY, "false", "private")
        assert json.loads(merged)["private"] is False

    def test_null_value(self):
        merged = _edit(PRETTY, "null", "version")
        assert json.loads(merged)["version"] is None

    def test_nested_key_via_dotted_path(self):
        merged = _edit(PRETTY, "true", "config.debug")
        doc = json.loads(merged)
        assert doc["config"]["debug"] is True
        # The nested sibling survives.
        assert doc["config"]["level"] == 3

    def test_array_value_replaces_whole_value(self):
        merged = _edit(PRETTY, '["x", "y", "z"]', "tags")
        doc = json.loads(merged)
        assert doc["tags"] == ["x", "y", "z"]
        assert doc["name"] == "old-pkg"

    def test_compact_file_value_change_keeps_siblings(self):
        """The reported whole-file wipe: a compact file's pairs share one
        line, so the old direct swap replaced the ENTIRE document."""
        merged = _edit(COMPACT, '"new-pkg"', "name")
        doc = json.loads(merged)
        assert doc["name"] == "new-pkg"
        assert doc["version"] == "1.0.0"
        assert doc["private"] is True

    def test_trailing_newline_state_preserved(self):
        with_nl = _edit(COMPACT, '"x"', "name")
        assert with_nl.endswith("\n")
        without_nl = _edit(COMPACT.rstrip("\n"), '"x"', "name")
        assert not without_nl.endswith("\n")
        assert json.loads(without_nl)["name"] == "x"

    def test_multiline_pretty_object_value_snippet(self):
        snippet = '{\n  "debug": true,\n  "level": 9\n}'
        merged = _edit(PRETTY, snippet, "config")
        doc = json.loads(merged)
        assert doc["config"] == {"debug": True, "level": 9}
        assert doc["name"] == "old-pkg"
        # The old value's pretty shape is preserved: braces on their own
        # lines at the pair's indent, members indented deeper.
        assert '  "config": {\n    "debug": true,\n    "level": 9\n  },' in merged


# ---------------------------------------------------------------------------
# Restated pair snippets splice the value (and the key on rename)
# ---------------------------------------------------------------------------


class TestRestatedPairSnippets:
    def test_object_snippet_with_target_key_splices_value_only(self):
        """The reported corruption shape: ``{"name": "new"}`` used to wipe
        the whole file / fall to the model."""
        merged = _edit(PRETTY, '{"name": "new"}', "name")
        doc = json.loads(merged)
        assert doc["name"] == "new"
        assert doc["version"] == "1.0.0"
        assert doc["private"] is True
        assert doc["config"]["level"] == 3

    def test_key_value_text_snippet_without_braces(self):
        merged = _edit(PRETTY, '"name": "new"', "name")
        doc = json.loads(merged)
        assert doc["name"] == "new"
        assert doc["version"] == "1.0.0"

    def test_key_rename_via_restated_pair(self):
        merged = _edit(PRETTY, '"title": "new"', "name")
        doc = json.loads(merged)
        assert "name" not in doc
        assert doc["title"] == "new"
        assert doc["version"] == "1.0.0"

    def test_restated_multiline_value_is_reprinted_in_file_style(self):
        snippet = '"tags": ["x", "y"]'
        merged = _edit(PRETTY, snippet, "tags")
        doc = json.loads(merged)
        assert doc["tags"] == ["x", "y"]
        # Pretty file → pretty value rendering.
        assert '  "tags": [\n    "x",\n    "y"\n  ]' in merged


# ---------------------------------------------------------------------------
# Structural safety: commas, siblings, style
# ---------------------------------------------------------------------------


class TestStructuralSafety:
    def test_no_double_comma_ever(self):
        for snippet, replace in (
            ('"x"', "name"), ("null", "version"), ("false", "private"),
            ('{"name": "x"}', "name"),
        ):
            merged = _edit(PRETTY, snippet, replace)
            assert ",," not in merged, f",, introduced by {snippet!r}"
            json.loads(merged)  # always parses

    def test_dangling_comma_in_original_is_not_propagated_or_doubled(self):
        """A pre-existing trailing comma after the last pair must survive
        EXACTLY as it was (neither doubled nor 'fixed'), and the edit must
        still land."""
        original = '{\n  "name": "old-pkg",\n  "version": "1.0.0",\n}\n'
        merged = _edit(original, '"new-pkg"', "name")
        assert ",," not in merged
        # The pre-existing trailing comma is untouched (it sits outside the
        # spliced value span).
        assert '"version": "1.0.0",\n}' in merged
        assert '"name": "new-pkg",' in merged

    def test_compact_style_stays_compact(self):
        merged = _edit(COMPACT, '"x"', "version")
        # No newlines introduced into a single-line document.
        assert merged.count("\n") == 1
        assert json.loads(merged)["version"] == "x"
        assert json.loads(merged)["name"] == "old-pkg"

    def test_pretty_style_stays_pretty(self):
        merged = _edit(PRETTY, '"x"', "version")
        assert json.loads(merged)["version"] == "x"
        # The file keeps its 2-space indentation on untouched lines.
        assert '\n  "name": "old-pkg",' in merged

    def test_siblings_byte_exact_in_pretty_file(self):
        merged = _edit(PRETTY, '"x"', "version")
        before = PRETTY.splitlines(keepends=True)
        after = merged.splitlines(keepends=True)
        assert len(before) == len(after)
        for i, (b, a) in enumerate(zip(before, after)):
            if '"version"' in b or '"version"' in a:
                continue
            assert b == a, f"line {i + 1} changed without a declared edit: {b!r} -> {a!r}"


    def test_flat_key_containing_a_dot_is_edited_literally(self):
        """A JSON key that LITERALLY contains a dot (``"a.b"``) is one flat
        key, not the nested path a.b: the dotted-target leaf split
        (``config.debug`` -> leaf ``debug``) must not hide it from the
        value-splice matcher. _resolve_symbol resolves the literal name
        (B3 literal pass); the helper must match the pair by its FULL name
        too."""
        doc = '{\n  "a.b": 1,\n  "c": 2\n}\n'
        merged = _edit(doc, '"new"', "a.b")
        assert json.loads(merged) == {"a.b": "new", "c": 2}
        # Sibling byte-exact.
        assert '"c": 2' in merged

    def test_flat_dotted_key_and_nested_leaf_coexist(self):
        """replace="a.b" targets the LITERAL flat key even when a nested
        a -> b pair also exists (the resolved target's span disambiguates)."""
        doc = '{\n  "a.b": 1,\n  "a": {"b": 2}\n}\n'
        merged = _edit(doc, '"flat"', "a.b")
        doc_out = json.loads(merged)
        assert doc_out["a.b"] == "flat"
        assert doc_out["a"]["b"] == 2

    def test_value_containing_double_comma_is_content_not_a_splice_bug(self):
        """The dangling-comma safety belt must not refuse a value whose
        STRING CONTENT contains ',,' — the belt hunts structural commas,
        and string literals are content (the splice replaces exactly the
        value span, so it cannot introduce a structural double comma)."""
        merged = _edit(PRETTY, '"wait,, what"', "name")
        doc = json.loads(merged)
        assert doc["name"] == "wait,, what"
        assert doc["config"]["level"] == 3


# ---------------------------------------------------------------------------
# The chunked_merge path (MCP route) gets the same fix
# ---------------------------------------------------------------------------


class TestChunkedMergeRoute:
    def test_chunked_merge_replace_key_splices_value(self):
        result = _try_json_key_replace(
            PRETTY, '"new-pkg"', "name",
        )
        assert result is not None
        merged, parse_valid = result
        assert parse_valid
        assert json.loads(merged)["name"] == "new-pkg"
        assert json.loads(merged)["version"] == "1.0.0"

    def test_chunked_merge_compact_file_no_whole_file_wipe(self):
        result = _try_json_key_replace(COMPACT, '"new-pkg"', "name")
        assert result is not None
        merged, _ = result
        doc = json.loads(merged)
        assert doc == {"name": "new-pkg", "version": "1.0.0", "private": True}


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def run_cli(*args: str):
    import os
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        capture_output=True, text=True, timeout=30, env=env, check=False,
    )


class TestCliEndToEnd:
    def test_edit_replace_with_bare_string_value(self, tmp_path: Path):
        target = tmp_path / "package.json"
        target.write_text(PRETTY)
        result = run_cli(
            "edit", str(target), "--snippet", '"new-pkg"', "--replace", "name",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        doc = json.loads(target.read_text())
        assert doc["name"] == "new-pkg"
        assert doc["config"]["level"] == 3

    def test_edit_replace_with_object_snippet_keeps_siblings(self, tmp_path: Path):
        target = tmp_path / "package.json"
        target.write_text(PRETTY)
        result = run_cli(
            "edit", str(target), "--snippet", '{"name": "new"}', "--replace", "name",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        doc = json.loads(target.read_text())
        assert doc["name"] == "new"
        assert doc["version"] == "1.0.0"
        assert doc["tags"] == ["a", "b"]
