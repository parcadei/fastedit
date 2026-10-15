"""Regression tests for TRDD-CMRMA2YG (fastedit edit --replace silent truncation).

`_try_deterministic_replace`'s direct-replacement fallback used to assume "the
snippet IS the new symbol" and splice it verbatim over the symbol's line range.
A snippet containing a keep-marker (`#...`, `//...`, ...) is NOT a complete
symbol -- the marker stands in for lines the author deliberately omitted -- so
splicing it verbatim silently deleted those lines while still exiting 0. These
tests pin the fix: `snippet_has_keep_marker` refuses the unsafe case with a
clean CLI error and leaves the file byte-for-byte unchanged, while every
legitimate case (trailing marker via the deterministic merge path, real code
that merely contains marker-like text, non-Python languages, and a bare `...`
Python stub body) keeps working.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastedit.inference.text_match import snippet_has_keep_marker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None):
    """Run `python -m fastedit <args>` and return CompletedProcess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )

def test_rust_lifetime_paired_with_a_later_apostrophe_is_detected() -> None:
    """A lifetime pairing with an apostrophe AFTER the marker must NOT hide it.

    `let s: &'static str;  // <marker> don't` -- the apostrophe in `'static`
    pairs with the one in `don't`, which sits AFTER the marker. A scanner
    that accepted that pairing entered string mode and never saw the `//`,
    so the marker was missed, the partial snippet was spliced, and code was
    lost silently at exit 0 (TRDD-CMRMA2YG, B42b).

    Fixed: an apostrophe opens a string only when its partner appears
    BEFORE any comment opener on the line. A pairing that SPANS a comment
    opener is lexical noise -- a Rust lifetime accidentally paired with
    English prose -- not a string, so the `//` is reached and the marker
    detected. (Un-pinned from xfail(strict) when the hole was closed.)
    """
    slash = "// " + "... existing code ..."
    a = chr(39)
    line = "let s: &" + a + "static str;  " + slash + " don" + a + "t"
    assert snippet_has_keep_marker(line) is True

def test_rust_lifetimes_do_not_hide_a_marker(tmp_path: Path) -> None:
    """A Rust lifetime must not swallow the rest of the line and hide a marker.

    MEASURED code-loss regression, and the fifth distinct shape to reach the
    silent path. Rust writes lifetimes with a lone apostrophe -- `&'static str`,
    `&'a mut T` -- which has no closing partner. A scanner that treats every
    apostrophe as a string opener therefore stays "inside a string" for the rest
    of the line, never reaches the `//`, and misses the marker; the partial
    snippet is then spliced and code is lost. Rust is a supported language.

    The rule: an apostrophe opens a string only when a partner appears later on
    the SAME line, so a genuine char literal still counts and a lifetime does not.

    Note the near-miss that makes this worth pinning: `fn f<'a>(x: &'a str)` has
    TWO apostrophes which accidentally pair, so it passed even while the
    single-lifetime form silently failed. Testing only that shape would have
    reported success.
    """
    slash = "// " + "... existing code ..."
    a = chr(39)

    assert snippet_has_keep_marker("let s: &" + a + "static str = x; " + slash) is True
    assert snippet_has_keep_marker("fn f<" + a + "a>(x: &" + a + "a str) { " + slash) is True
    assert snippet_has_keep_marker("let c = " + a + "x" + a + "; " + slash) is True
    assert snippet_has_keep_marker("    this.#count = 1;  " + slash) is True

def test_marker_detection_survives_awkward_string_syntax(tmp_path: Path) -> None:
    """Raw strings, escaped quotes and f-strings must not hide or invent a marker.

    The whole defect history of this predicate is UNCONSIDERED INPUT SHAPES, not
    broken lines -- three versions shipped, each correct on the cases its author
    imagined and wrong on one nobody had tried. Mutating the implementation
    cannot catch that class; only widening the input space can. So this pins the
    awkward shapes directly.

    The three that could LOSE code (a real marker going undetected, letting a
    partial snippet splice) are the first three. The last three must NOT be
    treated as markers: a marker that lives inside a string literal is data, not
    an instruction to keep lines.
    """
    canonical = "# " + "... existing code ..."
    q = chr(34)
    a = chr(39)
    bs = chr(92)

    # Must DETECT -- a real marker in a comment, after awkward-but-closed syntax.
    assert snippet_has_keep_marker("    x = r" + a + "a" + a + "  " + canonical) is True
    assert snippet_has_keep_marker("    x = " + q + "a" + bs + q + "b" + q + "  " + canonical) is True
    assert snippet_has_keep_marker("    x = f" + q + "{a}#b" + q + "  " + canonical) is True

    # Must NOT detect -- the marker text is inside a string, so it is data.
    assert snippet_has_keep_marker("    s = " + q * 3 + "text " + canonical) is False
    assert snippet_has_keep_marker("    x = " + q + "#not a marker" + q) is False
    assert snippet_has_keep_marker("    x = f" + q + "{" + canonical + "}" + q) is False

def test_marker_after_a_comment_containing_an_apostrophe_is_refused(tmp_path: Path) -> None:
    """A marker after an ordinary comment with an apostrophe must still be caught.

    Regression for a MEASURED silent-truncation path, and the third distinct
    shape to reach it. A guard that decided comment-vs-string by asking whether
    any quote character appeared before the marker treated the apostrophe in
    everyday English -- don't, won't, it's, the caller's -- as an open string,
    so the marker after it was invisible and the partial snippet was spliced.
    The same hole swallowed a marker following any earlier string literal.

    The fix scans the line with real quote state to find where the comment
    actually opens, which is the property itself rather than a proxy. This test
    pins the two shapes that defeated the proxy.
    """
    canonical = "# " + "... existing code ..."
    apostrophe = chr(39)
    quote = chr(34)

    assert snippet_has_keep_marker("    # don" + apostrophe + "t touch  " + canonical) is True
    assert snippet_has_keep_marker("    x = " + quote + "a" + quote + "  " + canonical) is True

    target = tmp_path / ("mod" + "." + "py")
    original = b"def f():\n    a = 1\n    b = 2\n    return a + b\n"
    target.write_bytes(original)
    body = "def f():\n    # don" + apostrophe + "t touch  " + canonical + "\n"
    result = run_cli("edit", str(target), "--replace", "f", "--snippet", body)
    assert result.returncode != 0
    assert target.read_bytes() == original, "target must be byte-identical on refusal"

def test_marker_after_code_on_the_same_line_is_refused(tmp_path: Path) -> None:
    """A marker sitting AFTER code on the same line must still be caught.

    Regression for a MEASURED silent-truncation path. An earlier guard tested
    the canonical marker with `startswith` on the stripped line, so a marker
    preceded by code was invisible to it. End-to-end, that snippet reduced a
    four-line body to one line and exited 0 -- the exact data loss this guard
    exists to prevent, reachable again through a guard meant to stop it.

    The two failure directions are NOT symmetric: a false positive refuses a
    valid edit (loud, recoverable), a false negative destroys code (silent).
    This asserts the dangerous direction; its sibling test asserts the other.
    """
    canonical = "# " + "... existing code ..."
    assert snippet_has_keep_marker("    x = 1  " + canonical) is True

    target = tmp_path / ("mod" + "." + "py")
    original = b"def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    target.write_bytes(original)
    body = "def f():\n    x = 1  " + canonical + "\n"
    result = run_cli("edit", str(target), "--replace", "f", "--snippet", body)
    assert result.returncode != 0
    assert target.read_bytes() == original, "target must be byte-identical on refusal"

def test_docstring_mentioning_the_marker_phrase_is_not_refused(tmp_path: Path) -> None:
    """A complete snippet whose DOCSTRING mentions a marker phrase must still apply.

    Regression for a MEASURED false positive. The guard originally tested the
    canonical long-form marker with `in` rather than `startswith`, so a valid,
    complete replacement body whose docstring merely described the marker was
    refused. Not academic: this project's own source documents markers in
    docstrings, and the containment version refused the very edit that fixed it.

    A refusal predicate's defect space is the inputs nobody thought to try, so
    this asserts the NEGATIVE case -- that a legitimate snippet is ACCEPTED --
    which is the half that behavioural testing of a guard usually omits.
    """
    marker_words = "# " + "... existing code ..."
    assert snippet_has_keep_marker(marker_words) is True
    assert snippet_has_keep_marker('    """Docs: write ' + marker_words + ' here."""') is False

    target = tmp_path / ("mod" + "." + "py")
    target.write_bytes(b"def f():\n    return 1\n")
    body = (
        "def f():\n"
        '    """Docs: write ' + marker_words + ' to keep lines."""\n'
        "    return 42\n"
    )
    result = run_cli("edit", str(target), "--replace", "f", "--snippet", body)
    assert result.returncode == 0, result.stderr
    assert b"return 42" in target.read_bytes()


class TestSnippetHasKeepMarker:
    """Unit tests for the predicate itself, independent of the CLI."""

    def test_short_hash_marker_is_detected(self):
        """A bare `#...` line is a keep-marker."""
        assert snippet_has_keep_marker("#...") is True

    def test_indented_short_hash_marker_is_detected(self):
        """An indented `#...` line is still a keep-marker."""
        assert snippet_has_keep_marker("  #...") is True

    def test_short_slash_marker_is_detected(self):
        """A bare `//...` line is a keep-marker."""
        assert snippet_has_keep_marker("//...") is True

    def test_canonical_hash_marker_is_detected(self):
        """The canonical long-form hash marker is a keep-marker."""
        assert snippet_has_keep_marker("# ... existing code ...") is True

    def test_canonical_slash_marker_is_detected(self):
        """The canonical long-form slash marker is a keep-marker."""
        assert snippet_has_keep_marker("// ... existing code ...") is True

    def test_bare_short_ellipsis_phrase_is_not_a_marker(self):
        """A bare `# ...` (no trailing `existing code`) is real code, not a marker."""
        assert snippet_has_keep_marker("# ...") is False

    def test_bare_short_slash_ellipsis_phrase_is_not_a_marker(self):
        """A bare `// ...` (no trailing `existing code`) is real code, not a marker."""
        assert snippet_has_keep_marker("// ...") is False

    def test_real_trailing_comment_is_not_a_marker(self):
        """A real code line whose comment happens to contain `# ...` is not a marker."""
        line = "x = 1  # ... and then some real trailing prose here"
        assert snippet_has_keep_marker(line) is False

    def test_ordinary_code_line_is_not_a_marker(self):
        """A plain code line is not a marker."""
        assert snippet_has_keep_marker("y = 2") is False

    def test_string_literal_ellipsis_is_not_a_marker(self):
        """A string literal containing three dots is not a marker."""
        assert snippet_has_keep_marker("return '...'") is False

    def test_bare_ellipsis_stub_body_is_not_a_marker(self):
        """A bare `...` line (Python Ellipsis stub body) is real code, not a marker."""
        assert snippet_has_keep_marker("...") is False

    def test_protocol_stub_method_is_not_a_marker(self):
        """A full stub method body (`def f(): ...`) is not a marker."""
        snippet = "def f(self) -> int:\n    ...\n"
        assert snippet_has_keep_marker(snippet) is False


class TestMidSnippetMarkerRefusal:
    """The exact TRDD-CMRMA2YG repro: a mid-snippet marker must refuse, not truncate."""

    def test_mid_snippet_marker_refuses_with_nonzero_exit(self, tmp_path: Path):
        """`--replace` with a mid-snippet `#...` marker must exit non-zero."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert result.returncode != 0

    def test_mid_snippet_marker_refusal_names_the_symbol(self, tmp_path: Path):
        """The refusal message must name the symbol so the user can act on it."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert "g" in result.stderr
        assert "keep-marker" in result.stderr

    def test_mid_snippet_marker_refusal_prints_no_traceback(self, tmp_path: Path):
        """The refusal must be a clean CLI diagnostic, never a raw Python traceback."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert "Traceback" not in result.stderr

    def test_mid_snippet_marker_refusal_leaves_file_unchanged(self, tmp_path: Path):
        """The file on disk must be byte-for-byte unchanged after a refused edit."""
        target = tmp_path / "mid.py"
        original = "def g():\n    x = 1\n    y = 2\n    return x + y\n"
        target.write_text(original, encoding="utf-8")
        run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert target.read_text(encoding="utf-8") == original


class TestTrailingMarkerStillWorks:
    """The trailing-marker case (never reaches the guard) must not regress."""

    def test_trailing_marker_preserves_untouched_tail(self, tmp_path: Path):
        """A trailing `#...` merges via deterministic_edit and preserves b, c, return."""
        target = tmp_path / "tail.py"
        target.write_text(
            "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n",
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    a = 99\n#...\n",
        )
        assert result.returncode == 0
        merged = target.read_text(encoding="utf-8")
        assert "b = 2" in merged
        assert "c = 3" in merged
        assert "return a + b + c" in merged


class TestRealCodeWithMarkerLikeTextIsNotRefused:
    """A snippet containing real code that merely resembles a marker must apply."""

    def test_snippet_with_real_hash_ellipsis_comment_is_applied(self, tmp_path: Path):
        """A full replacement body containing `# ...` in a real comment is not refused."""
        target = tmp_path / "h.py"
        target.write_text("def h():\n    return 1\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "h",
            "--snippet",
            "def h():\n    x = 1  # ... and then some real trailing prose here\n    return x\n",
        )
        assert result.returncode == 0
        assert "return x" in target.read_text(encoding="utf-8")

    def test_ellipsis_stub_body_replacement_is_applied(self, tmp_path: Path):
        """A `--replace` snippet whose full body is a bare `...` stub is not refused."""
        target = tmp_path / "stub.py"
        target.write_text("def f(self) -> int:\n    ...\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f(self) -> int:\n    ...\n",
        )
        assert result.returncode == 0


class TestNonPythonLanguage:
    """The guard also refuses the analogous unsafe case in a non-Python language."""

    def test_js_direct_replace_marker_only_body_refuses(self, tmp_path: Path):
        """A JS `--replace` snippet consisting only of a keep-marker must refuse.

        This targets the direct-replacement fallback path specifically (a
        marker-only snippet with no other content cannot be merged by
        `deterministic_edit`, which always declines it), independent of any
        per-language merge-quality behavior of `deterministic_edit` itself.
        """
        target = tmp_path / "mid.js"
        target.write_text(
            "function g() {\n  var x = 1;\n  var y = 2;\n  return x + y;\n}\n",
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "//...",
        )
        assert result.returncode != 0
        assert "keep-marker" in result.stderr
        assert target.read_text(encoding="utf-8") == (
            "function g() {\n  var x = 1;\n  var y = 2;\n  return x + y;\n}\n"
        )
