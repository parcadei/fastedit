"""Regression tests for the multi-line signature auto-preserve bug.

Symptom (up to and including 0.3.2): a function whose parameter list spans
multiple lines, combined with ``replace='name'`` and a body-only snippet,
produced syntactically broken output because the auto-preserve path grabbed
only the first line of the signature (``def foo(``) and dropped the
continuation (``    a, b, c):``).

Fix uses tree-sitter's ``body`` field to find where the signature ends,
then prepends the full span. Regression-tested below across Python, Rust,
Go, JavaScript, and TypeScript.
"""

from __future__ import annotations

import ast as py_ast

from fastedit.inference.chunked_merge import chunked_merge


def _should_not_call_model(*args, **kwargs):
    raise AssertionError(
        "signature auto-preserve + deterministic_edit should handle this "
        "without reaching the model"
    )


class _FakeModelResult:
    """Minimal stand-in for the engine's MergeResult."""

    def __init__(self, merged_code):
        self.merged_code = merged_code
        self.tokens_generated = 11
        self.latency_ms = 1.0
        self.parse_valid = True


def _model_returns_original(code, snippet, language):
    """Fallback model stand-in: records the call, leaves the code intact."""
    _model_returns_original.called = True
    return _FakeModelResult(code)


_model_returns_original.called = False


def test_multiline_signature_python_preserved():
    original = (
        "def foo(\n"
        "    a: int,\n"
        "    b: int,\n"
        "    c: int,\n"
        "    d: int,\n"
        "):\n"
        '    """old docstring"""\n'
        "    return a + b\n"
    )
    snippet = (
        '    """new docstring"""\n'
        "    total = a + b + c + d\n"
        "    return total\n"
    )
    result = chunked_merge(
        original, snippet, "/tmp/test_multiline_sig.py",
        _should_not_call_model, language="python", replace="foo",
    )
    # Full parameter list preserved
    assert "def foo(\n    a: int,\n    b: int,\n    c: int,\n    d: int,\n):" in result.merged_code
    # Body replaced
    assert "new docstring" in result.merged_code
    assert "total = a + b + c + d" in result.merged_code
    # Parses as valid Python
    py_ast.parse(result.merged_code)


def test_single_line_signature_still_works():
    """The fix must not regress the common case."""
    original = (
        "def foo(a, b):\n"
        '    """old"""\n'
        "    return a + b\n"
    )
    snippet = (
        '    """new"""\n'
        "    return a * b\n"
    )
    result = chunked_merge(
        original, snippet, "/tmp/test_single_sig.py",
        _should_not_call_model, language="python", replace="foo",
    )
    assert "def foo(a, b):" in result.merged_code
    assert "return a * b" in result.merged_code
    py_ast.parse(result.merged_code)


def test_multiline_signature_with_return_annotation():
    """Python function with multi-line params AND a return annotation on the
    closing line. The signature ends with ``-> RetType:`` which must be kept."""
    original = (
        "def compute(\n"
        "    value: int,\n"
        "    multiplier: int = 2,\n"
        ") -> int:\n"
        "    return value * multiplier\n"
    )
    snippet = "    return value + multiplier\n"
    result = chunked_merge(
        original, snippet, "/tmp/test_return_annot.py",
        _should_not_call_model, language="python", replace="compute",
    )
    assert "def compute(\n    value: int,\n    multiplier: int = 2,\n) -> int:" in result.merged_code
    assert "return value + multiplier" in result.merged_code
    py_ast.parse(result.merged_code)


def test_multiline_signature_rust():
    """tree-sitter-rust exposes the same ``body`` field on ``function_item``.

    UPDATED (preserve-by-default, Step 5): the extracted signature span now
    ends AFTER the block opener — ``) -> i32 {`` — so the prepended lines
    match the original signature exactly and are pinned as fixed context
    instead of being misclassified as column-0 new lines.

    The deterministic editor STILL declines this particular shape, and that
    decline is correct preserve-by-default behavior: the snippet's only
    content is the bare tail expression ``a * b`` — an ambiguous rewrite of
    the preserved tail expression ``a + b`` (same leading token, no
    assignment LHS, no marker). Emitting both would silently change which
    expression is the function's return value, so the editor declines
    rather than guess. The snippet is also not a complete re-definition
    (no closing brace), so the direct-swap fast path declines it too.
    The model decides; the fast path hands it the original with the
    signature (opener included) intact.
    """
    original = (
        "fn foo(\n"
        "    a: i32,\n"
        "    b: i32,\n"
        ") -> i32 {\n"
        "    a + b\n"
        "}\n"
    )
    snippet = "    a * b\n"
    _model_returns_original.called = False
    result = chunked_merge(
        original, snippet, "/tmp/test_rust_sig.rs",
        _model_returns_original, language="rust", replace="foo",
    )
    assert _model_returns_original.called, (
        "deterministic path must decline the brace-less prepended "
        "signature and fall through to the model"
    )
    # The model fallback receives the original intact — the signature
    # (opener included) was never dropped by the fast path.
    assert "fn foo(\n    a: i32,\n    b: i32,\n) -> i32 {" in result.merged_code
    assert "a * b" not in result.merged_code or result.merged_code == original


def test_multiline_signature_typescript():
    """UPDATED (preserve-by-default, Step 5): the extracted TypeScript
    signature now ends after the ``): number {`` opener and the prepended
    lines are pinned context. The edit still completes on the MODEL path,
    and that is the correct preserve-by-default outcome: the snippet's
    only content, ``return a * b;``, is an ambiguous rewrite of the
    preserved ``return a + b;`` (same leading token, no assignment LHS,
    no marker) — the deterministic editor declines rather than emit both
    or drop one — and the snippet is not a complete re-definition (no
    closing brace), so the direct-swap fast path declines it as well.
    The model fallback receives the signature intact.
    """
    original = (
        "function foo(\n"
        "  a: number,\n"
        "  b: number,\n"
        "): number {\n"
        "  return a + b;\n"
        "}\n"
    )
    snippet = "  return a * b;\n"
    _model_returns_original.called = False
    result = chunked_merge(
        original, snippet, "/tmp/test_ts_sig.ts",
        _model_returns_original, language="typescript", replace="foo",
    )
    assert _model_returns_original.called, (
        "deterministic path must decline the brace-less prepended "
        "signature and fall through to the model"
    )
    assert (
        "function foo(\n  a: number,\n  b: number,\n): number {" in result.merged_code
    )

import pytest

from fastedit.inference.chunked_merge import _extract_signature_via_ast

# Every tree-sitter grammar FastEdit supports. For each one: a source file
# with a MULTI-LINE parameter list, the line range of the target function/
# method/class, and an assertion that the extracted signature contains the
# full parameter list (not just the opening line).
_MULTILINE_CASES = [
    (
        "python",
        "def foo(\n    a: int,\n    b: int,\n    c: int,\n):\n    return a + b + c\n",
        1, 5, ["def foo(", "a: int,", "b: int,", "c: int,", "):"],
    ),
    (
        "javascript",
        "function foo(\n  a,\n  b,\n  c,\n) {\n  return a + b + c;\n}\n",
        1, 6, ["function foo(", "a,", "b,", "c,", ")"],
    ),
    (
        "typescript",
        "function foo(\n  a: number,\n  b: number,\n): number {\n  return a + b;\n}\n",
        1, 5, ["function foo(", "a: number,", "b: number,", "): number"],
    ),
    (
        "rust",
        "fn foo(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a + b\n}\n",
        1, 6, ["fn foo(", "a: i32,", "b: i32,", "-> i32"],
    ),
    (
        "go",
        "func Foo(\n\ta int,\n\tb int,\n) int {\n\treturn a + b\n}\n",
        1, 6, ["func Foo(", "a int,", "b int,", ") int"],
    ),
    (
        "java",
        "class C {\n  void foo(\n    int a,\n    int b\n  ) {\n    return;\n  }\n}\n",
        2, 7, ["void foo(", "int a,", "int b"],
    ),
    (
        "ruby",
        "def foo(\n  a,\n  b\n)\n  a + b\nend\n",
        1, 6, ["def foo(", "a,", "b"],
    ),
    (
        "swift",
        "func foo(\n    a: Int,\n    b: Int\n) -> Int {\n    return a + b\n}\n",
        1, 6, ["func foo(", "a: Int,", "b: Int", "-> Int"],
    ),
    (
        "kotlin",
        "fun foo(\n    a: Int,\n    b: Int\n): Int {\n    return a + b\n}\n",
        1, 6, ["fun foo(", "a: Int,", "b: Int", ": Int"],
    ),
    (
        "c",
        "int foo(\n    int a,\n    int b\n) {\n    return a + b;\n}\n",
        1, 6, ["int foo(", "int a,", "int b"],
    ),
    (
        "cpp",
        "int foo(\n    int a,\n    int b\n) {\n    return a + b;\n}\n",
        1, 6, ["int foo(", "int a,", "int b"],
    ),
    (
        "c_sharp",
        "class C {\n  int Foo(\n    int a,\n    int b\n  ) {\n    return a + b;\n  }\n}\n",
        2, 7, ["int Foo(", "int a,", "int b"],
    ),
    (
        "php",
        "<?php\nfunction foo(\n    $a,\n    $b\n) {\n    return $a + $b;\n}\n",
        2, 7, ["function foo(", "$a,", "$b"],
    ),
    (
        "elixir",
        "def foo(\n  a,\n  b\n) do\n  a + b\nend\n",
        1, 6, ["def foo(", "a,", "b"],
    ),
]


@pytest.mark.parametrize("language,src,ls,le,must_contain", _MULTILINE_CASES)
def test_multiline_signature_extraction_across_all_languages(
    language, src, ls, le, must_contain,
):
    """Every supported tree-sitter grammar must preserve a multi-line parameter
    list when the AST-based signature extractor is called. Regression lock
    against the 0.3.2-era bug where only the first line of the signature was
    kept."""
    sig = _extract_signature_via_ast(src, language, ls, le, "FALLBACK")
    assert sig != "FALLBACK", f"{language}: helper fell back to single-line"
    for fragment in must_contain:
        assert fragment in sig, (
            f"{language}: missing {fragment!r} in extracted signature {sig!r}"
        )
