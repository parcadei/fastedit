"""Issue #3: ``--replace`` duplicates a target's leading doc comment.

A function's leading JSDoc / docstring comment sits ABOVE the symbol's AST
span (the span starts at the ``def``/``function`` line — the definition
line). When the snippet RESTATES that comment along with the signature and
body, the old splice emitted the snippet verbatim over the span: the
original comment above the span survived AND the snippet's copy was spliced
in — the JSDoc appeared twice (once above the spliced function, once
attached to it).

The fix strips the snippet's leading lines when they duplicate the file's
lines immediately above the replaced span (reverse-order match), so the
definition line is where the splice content starts and the comment above
the span stays single. A Python snippet restating a docstring above the
def also used to be REFUSED outright (the single-definition refinement
counted the docstring's expression statement as a second top-level node);
stripping the duplicate first fixes that false refusal too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastedit.cli import _try_deterministic_replace
from fastedit.data_gen.ast_analyzer import detect_language
from fastedit.inference.chunked_merge import (
    _strip_leading_doc_duplicate,
    chunked_merge,
)
from fastedit.mcp.backup import BackupStore

JS_WITH_JSDOC = """/**
 * Docs for greet.
 */
function greet(name) {
  return "hi " + name;
}
"""

PY_WITH_DOC_ABOVE = '''"""Docstring above the function."""

def greet(name):
    return "hi " + name
'''


class WithMethodDocInside:
    pass


PY_METHOD_DOC = '''class Store:
    """Store docs."""

    """Docs for save."""

    def save(self):
        return 1
'''


def _cli_replace(code: str, snippet: str, replace: str, filename: str) -> str:
    path = Path("/virtual") / filename
    lines = code.splitlines(keepends=True)
    result = _try_deterministic_replace(
        path, code, lines, snippet, replace,
        detect_language(path), BackupStore(),
    )
    assert result is not None, (
        f"doc-restating snippet for {replace!r} fell through to the model path"
    )
    return result.merged_code


# ---------------------------------------------------------------------------
# CLI deterministic path (_try_deterministic_replace)
# ---------------------------------------------------------------------------


class TestCliDeterministicPath:
    def test_jsdoc_restatement_not_duplicated(self):
        snippet = (
            "/**\n * Docs for greet.\n */\n"
            'function greet(name) {\n  return "hello " + name;\n}\n'
        )
        merged = _cli_replace(JS_WITH_JSDOC, snippet, "greet", "app.js")
        assert merged.count("Docs for greet.") == 1, repr(merged)
        assert '"hello " + name' in merged
        # The surviving comment sits ABOVE the definition line.
        assert merged.index("Docs for greet.") < merged.index("function greet")

    def test_py_docstring_above_not_duplicated_and_not_refused(self):
        """The reported Python shape: a ``\"\"\"Docstring\"\"\"`` above the
        function restated by the snippet. Used to be REFUSED ("has no
        definition line of its own") because the single-definition
        refinement counted the docstring statement as a second node."""
        snippet = (
            '"""Docstring above the function."""\n\n'
            'def greet(name):\n    return "hello " + name\n'
        )
        merged = _cli_replace(PY_WITH_DOC_ABOVE, snippet, "greet", "app.py")
        assert merged.count("Docstring above the function.") == 1, repr(merged)
        assert '"hello " + name' in merged

    def test_indented_doc_above_method_not_duplicated(self):
        snippet = (
            '    """Docs for save."""\n\n'
            "    def save(self):\n        return 2\n"
        )
        merged = _cli_replace(PY_METHOD_DOC, snippet, "save", "app.py")
        assert merged.count("Docs for save.") == 1, repr(merged)
        assert "return 2" in merged
        # The class and its own docstring are untouched.
        assert merged.count("Store docs.") == 1

    def test_doc_added_when_none_exists_is_kept(self):
        """A snippet ADDING a doc comment where the file has none must not
        be stripped — nothing above the span matches it."""
        code = 'function greet(name) {\n  return "hi " + name;\n}\n'
        snippet = (
            "/**\n * Docs for greet.\n */\n"
            'function greet(name) {\n  return "hello " + name;\n}\n'
        )
        merged = _cli_replace(code, snippet, "greet", "app.js")
        assert merged.count("Docs for greet.") == 1
        assert merged.startswith("/**\n * Docs for greet.")
        assert '"hello " + name' in merged

    def test_body_only_snippet_still_drops_inside_docstring_is_not_this_bug(self):
        """A snippet WITHOUT the function's own docstring (which lives
        INSIDE the body span) is a body restatement; the inside-body
        docstring is part of the symbol, not a comment above it. The
        splice replaces the span, so the restated body is what lands."""
        code = 'def greet(name):\n    """Greets the user."""\n    return "hi " + name\n'
        snippet = 'def greet(name):\n    """Greets the user."""\n    return "hello " + name\n'
        merged = _cli_replace(code, snippet, "greet", "app.py")
        assert merged.count("Greets the user.") == 1
        assert '"hello " + name' in merged


# ---------------------------------------------------------------------------
# The strip helper itself
# ---------------------------------------------------------------------------


class TestStripLeadingDocDuplicate:
    def test_strips_reversed_order_match_above_span(self):
        lines = JS_WITH_JSDOC.splitlines(keepends=True)
        # Span starts at the function line (index 3).
        stripped, n = _strip_leading_doc_duplicate(
            "/**\n * Docs for greet.\n */\nfunction greet(name) {\n",
            lines, 3,
        )
        assert n == 3
        assert stripped.startswith("function greet(name) {")

    def test_stops_at_first_mismatch(self):
        lines = ["a\n", "b\n", "def f():\n"]
        stripped, n = _strip_leading_doc_duplicate("b\nx\ndef f():\n", lines, 2)
        # Only "b" matches the line above the span; "x" does not match "a".
        assert n == 1
        assert stripped == "x\ndef f():\n"

    def test_never_strips_the_whole_snippet(self):
        lines = ["def f():\n"]
        stripped, n = _strip_leading_doc_duplicate("def f():\n", lines, 0)
        assert n == 0  # nothing above the span
        assert stripped == "def f():\n"

    def test_span_at_top_of_file_strips_nothing(self):
        lines = ["def f():\n", "    pass\n"]
        stripped, n = _strip_leading_doc_duplicate("def f():\n    pass\n", lines, 0)
        assert n == 0
        assert stripped == "def f():\n    pass\n"


# ---------------------------------------------------------------------------
# chunked_merge replace= path (the MCP route shares it)
# ---------------------------------------------------------------------------


class TestChunkedMergePath:
    def test_chunked_merge_jsdoc_restatement_not_duplicated(self):
        snippet = (
            "/**\n * Docs for greet.\n */\n"
            'function greet(name) {\n  return "hello " + name;\n}\n'
        )
        result = chunked_merge(
            original_code=JS_WITH_JSDOC,
            snippet=snippet,
            file_path="app.js",
            merge_fn=lambda *a, **k: pytest.fail("model must not be called"),
            language="javascript",
            replace="greet",
        )
        merged = result.merged_code
        assert merged.count("Docs for greet.") == 1, repr(merged)
        assert '"hello " + name' in merged

    def test_chunked_merge_doc_restatement_without_signature_prepends_cleanly(self):
        """A snippet restating the doc but NOT the signature used to have
        the signature prepended ABOVE the doc (signature + doc + body).
        After the strip, the prepend lands on the body-only snippet."""
        snippet = '"""Docstring above the function."""\n\nreturn "hello " + name\n'
        result = chunked_merge(
            original_code=PY_WITH_DOC_ABOVE,
            snippet=snippet,
            file_path="app.py",
            merge_fn=lambda *a, **k: pytest.fail("model must not be called"),
            language="python",
            replace="greet",
        )
        merged = result.merged_code
        assert merged.count("Docstring above the function.") == 1, repr(merged)
        assert '"hello " + name' in merged
        # The signature the prepend restored sits BELOW the doc, not above.
        assert merged.index("Docstring above") < merged.index("def greet")


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
    def test_edit_replace_jsdoc_restatement_single_copy(self, tmp_path: Path):
        target = tmp_path / "app.js"
        target.write_text(JS_WITH_JSDOC)
        snippet = (
            "/**\n * Docs for greet.\n */\n"
            'function greet(name) {\n  return "hello " + name;\n}\n'
        )
        result = run_cli(
            "edit", str(target), "--snippet", snippet, "--replace", "greet",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text()
        assert content.count("Docs for greet.") == 1, repr(content)
        assert '"hello " + name' in content

    def test_edit_replace_py_docstring_restatement_single_copy(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text(PY_WITH_DOC_ABOVE)
        snippet = (
            '"""Docstring above the function."""\n\n'
            'def greet(name):\n    return "hello " + name\n'
        )
        result = run_cli(
            "edit", str(target), "--snippet", snippet, "--replace", "greet",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text()
        assert content.count("Docstring above the function.") == 1, repr(content)
