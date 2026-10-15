"""Golden-case generator for the B3 per-format golden matrix (req. 3 + C1).

Committed generator: `uv run python tests/golden/_generate.py` re-writes
every fixture under ``tests/golden/<lang>/``.

F2 and F3 add the census-batch tables. ``CENSUS_LANGUAGES``: one minimal
parse-clean fixture per tree-sitter-language-pack ok-name that B3 does not
already cover (batch 1 = the alphabetical range through odin; batch 2 = the
rest), plus every parse_degraded name whose REAL per-language snippet parses
clean — the census probe verdict was an artifact of the trivial probe line,
not the grammar. ``EXCLUDED_LANGUAGES``: the degraded names whose grammar is
genuinely broken, recorded with the exact parse_diagnostics errors instead
of a fabricated fixture (see ``_write_excluded`` and the fail-loud pin in
tests/test_golden_matrix.py) — empty since G1b lifted its founding member
(``test``), whose defect record now rides IN the exercised fixture as
``grammar_defect_workaround``. Languages with no declarative symbol
anchoring in ast_utils declare their anchoring ops ``unsupported`` — an
honest hole, never silent.

THE ORACLE IS INDEPENDENT OF FASTEDIT. Expected outputs are computed by
explicit line-splice arithmetic against the line indices declared in the
LANGUAGES table below — this module never imports fastedit, never runs its
pipeline, and never consults a parser. The oracle encodes the DOCUMENTED
splice semantics of the deterministic paths as plain line arithmetic:

  insert_after(symbol)
      Splice the snippet lines (preservation-marker lines dropped — they
      are merge directives, never content) after the anchor symbol's last
      line, wrapped in one blank separator line on each side where the
      adjacent original line is non-blank.

  replace_symbol(span)
      Replace the span's lines with the effective snippet: the span's own
      leading signature lines when the op declares ``prepend_signature_lines``
      (the pipeline auto-prepends a signature the snippet omitted), then the
      snippet lines minus marker lines, each newline-terminated.

  delete_symbol(span)
      Remove the span's lines, then consume the blank lines that separated
      the deleted symbol from whatever follows (the leading blanks already
      preserved before the span become the new separator).

Every declared line index is cross-checked against the authored original
(``span_head`` / ``span_tail`` / ``anchor_head`` / ``anchor_tail``
expectations) so the table cannot silently drift from the fixtures, and
every snippet is asserted pre-aligned to its target's FIRST-line indent
(the pipeline's indent alignment is a no-op on every golden op by
construction).

The manifest-driven runner (tests/test_golden_matrix.py) re-derives the
expected bytes from the manifest with THESE SAME oracle functions and
compares them against the committed expected files, so a stale artifact
fails loudly instead of pinning wrong behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent

# A preservation-marker line (canonical long forms; see
# fastedit.inference.markers) is a merge directive: the oracle drops it
# from snippet content exactly as the pipeline does.
_MARKER_LINE = re.compile(r"^(?:#|//)\s*\.\.\.\s*existing code\s*\.\.\.\s*$")


def is_marker_line(line: str) -> bool:
    """True for a preservation-marker directive line (never content)."""
    return bool(_MARKER_LINE.match(line.strip()))


def _terminated(snippet: str) -> list[str]:
    """Snippet text -> newline-terminated lines, markers dropped."""
    body = snippet.rstrip("\n")
    return [
        line + "\n"
        for line in body.split("\n")
        if not is_marker_line(line)
    ]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def oracle_insert_after(
    lines: list[str], anchor_end_line: int, snippet: str,
) -> list[str]:
    """Splice the snippet after the anchor's last line (blank separators)."""
    before = lines[:anchor_end_line]
    after = lines[anchor_end_line:]
    kept = _terminated(snippet)
    separator = ["\n"] if before and before[-1].strip() != "" else []
    trailing = ["\n"] if after and after[0].strip() != "" else []
    return before + separator + kept + trailing + after


def oracle_replace_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    snippet: str,
    prepend_signature_lines: int = 0,
) -> list[str]:
    """Replace the span's lines with the effective snippet lines."""
    replacement = list(lines[start_line - 1 : start_line - 1 + prepend_signature_lines])
    replacement += _terminated(snippet)
    return lines[: start_line - 1] + replacement + lines[end_line:]


def oracle_delete_span(
    lines: list[str], start_line: int, end_line: int,
) -> list[str]:
    """Remove the span plus the blank lines that separated it from the rest."""
    end_idx = end_line
    while end_idx < len(lines) and lines[end_idx].strip() == "":
        end_idx += 1
    return lines[: start_line - 1] + lines[end_idx:]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# The golden matrix. Fixture content and oracle indices are HAND-DECLARED:
# the generator verifies every index against the authored original before
# writing anything.
# ---------------------------------------------------------------------------

LANGUAGES: list[dict] = [
    # --- the 15 original AST languages -----------------------------------
    {
        "language": "python", "ext": "py", "filename": "original.py",
        "symbol_semantics": (
            "top-level functions, classes, methods (parent-qualified) — "
            "module-level UPPER_CASE assignments are deliberately not symbols "
            "(see the _CONST_LIKE_NODE_TYPES note in ast_utils)"
        ),
        "original": '''"""Sample module."""
CONSTANT = 42


def alpha(x):
    return x + 1


class Store:
    def save(self):
        return "saved"


def beta(y):
    return y * 2
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "def gamma(z):\n    return z - 1\n",
                "anchor_start_line": 5, "anchor_head": "def alpha(x):",
                "anchor_end_line": 6, "anchor_tail": "    return x + 1",
                "snippet_head": "def gamma(z):",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "def beta(y):\n    return y * 3\n",
                "start_line": 14, "end_line": 15,
                "span_head": "def beta(y):", "span_tail": "    return y * 2",
                "snippet_head": "def beta(y):",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 5, "end_line": 6,
                "span_head": "def alpha(x):", "span_tail": "    return x + 1",
            },
        ],
        "gigo": {
            "defect": "syntax error in an unrelated function (def broken(:) — EDIT-NOT-CORRECT: preserved byte-exact",
            "defect_lines": [4, 5],
            "original": '''"""Sample module."""


def broken(:
    return oops


def alpha(x):
    return x + 1


def beta(y):
    return y * 2
''',
            "op": {
                "op": "insert_after", "symbol": "beta",
                "snippet": "def gamma(z):\n    return z - 1\n",
                "anchor_start_line": 12, "anchor_head": "def beta(y):",
                "anchor_end_line": 13, "anchor_tail": "    return y * 2",
                "snippet_head": "def gamma(z):",
            },
        },
    },
    {
        "language": "javascript", "ext": "js", "filename": "original.js",
        "symbol_semantics": "functions, classes, methods, top-level const/let (constants)",
        "original": '''const LIMIT = 10;


function alpha(x) {
    return x + 1;
}


class Cart {
    add(item) {
        this.items.push(item);
    }
}


function beta(y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma(z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 4, "anchor_head": "function alpha(x) {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "function gamma(z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "function beta(y) {\n    return y * 3;\n}\n",
                "start_line": 16, "end_line": 18,
                "span_head": "function beta(y) {", "span_tail": "}",
                "snippet_head": "function beta(y) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "typescript", "ext": "ts", "filename": "original.ts",
        "symbol_semantics": "functions, classes, interfaces, method signatures, top-level const/let",
        "original": '''const LIMIT: number = 10;


function alpha(x: number): number {
    return x + 1;
}


interface Repo {
    save(): boolean;
}


function beta(y: number): number {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma(z: number): number {\n    return z - 1;\n}\n",
                "anchor_start_line": 4,
                "anchor_head": "function alpha(x: number): number {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "function gamma(z: number): number {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "function beta(y: number): number {\n    return y * 3;\n}\n",
                "start_line": 14, "end_line": 16,
                "span_head": "function beta(y: number): number {", "span_tail": "}",
                "snippet_head": "function beta(y: number): number {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT: number = 10;",
                "span_tail": "const LIMIT: number = 10;",
            },
        ],
    },
    {
        "language": "tsx", "ext": "tsx", "filename": "original.tsx",
        "symbol_semantics": "components (functions), top-level const/let",
        "original": '''const LIMIT = 10;


function Badge({ label }: { label: string }) {
    return <span className="badge">{label}</span>;
}


function Panel({ title }: { title: string }) {
    return (
        <div className="panel">
            <Badge label={title} />
        </div>
    );
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Badge",
                "snippet": 'function Tag({ label }: { label: string }) {\n    return <span className="tag">{label}</span>;\n}\n',
                "anchor_start_line": 4,
                "anchor_head": "function Badge({ label }: { label: string }) {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": 'function Tag({ label }: { label: string }) {',
            },
            {
                "op": "replace_symbol", "symbol": "Badge", "path": "direct_swap",
                "snippet": 'function Badge({ label }: { label: string }) {\n    return <span className="tag">{label}</span>;\n}\n',
                "start_line": 4, "end_line": 6,
                "span_head": "function Badge({ label }: { label: string }) {",
                "span_tail": "}",
                "snippet_head": "function Badge({ label }: { label: string }) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "rust", "ext": "rs", "filename": "original.rs",
        "symbol_semantics": "functions, structs/enums/traits/impls, const/static items",
        "original": '''const LIMIT: i32 = 10;

fn alpha(x: i32) -> i32 {
    x + 1
}

struct Item {
    name: String,
}

fn beta(y: i32) -> i32 {
    y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "fn gamma(z: i32) -> i32 {\n    z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "fn alpha(x: i32) -> i32 {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "fn gamma(z: i32) -> i32 {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "fn beta(y: i32) -> i32 {\n    y * 3\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "fn beta(y: i32) -> i32 {", "span_tail": "}",
                "snippet_head": "fn beta(y: i32) -> i32 {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const LIMIT: i32 = 10;",
                "span_tail": "const LIMIT: i32 = 10;",
            },
        ],
    },
    {
        "language": "go", "ext": "go", "filename": "original.go",
        "symbol_semantics": "functions and methods, type declarations, var/const declarations",
        "original": '''package main

const Limit = 10

func Alpha(x int) int {
\treturn x + 1
}

type Item struct {
\tName string
}

func Beta(y int) int {
\treturn y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Alpha",
                "snippet": "func Gamma(z int) int {\n\treturn z - 1\n}\n",
                "anchor_start_line": 5, "anchor_head": "func Alpha(x int) int {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": "func Gamma(z int) int {",
            },
            {
                "op": "replace_symbol", "symbol": "Beta", "path": "direct_swap",
                "snippet": "func Beta(y int) int {\n\treturn y * 3\n}\n",
                "start_line": 13, "end_line": 15,
                "span_head": "func Beta(y int) int {", "span_tail": "}",
                "snippet_head": "func Beta(y int) int {",
            },
            {
                "op": "delete_symbol", "symbol": "Limit",
                "start_line": 3, "end_line": 3,
                "span_head": "const Limit = 10", "span_tail": "const Limit = 10",
            },
        ],
    },
    {
        "language": "java", "ext": "java", "filename": "original.java",
        "symbol_semantics": (
            "classes/interfaces/enums, methods and constructors, fields "
            "(constants). The replace op passes a body-only snippet and the "
            "pipeline prepends the signature line from the AST"
        ),
        "original": '''public class Sample {
    private static final int LIMIT = 10;

    public static int alpha(int x) {
        return x + 1;
    }

    public static int beta(int y) {
        return y * 2;
    }
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "    public static int gamma(int z) {\n        return z - 1;\n    }\n",
                "anchor_start_line": 4,
                "anchor_head": "    public static int alpha(int x) {",
                "anchor_end_line": 6, "anchor_tail": "    }",
                "snippet_head": "    public static int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "        return y * 3;\n    }\n",
                "start_line": 8, "end_line": 10,
                "span_head": "    public static int beta(int y) {",
                "span_tail": "    }",
                "snippet_head": "        return y * 3;",
                "prepend_signature_lines": 1,
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 2, "end_line": 2,
                "span_head": "    private static final int LIMIT = 10;",
                "span_tail": "    private static final int LIMIT = 10;",
            },
        ],
    },
    {
        "language": "c", "ext": "c", "filename": "original.c",
        "symbol_semantics": "functions, struct/enum specifiers; preproc #define is not a symbol",
        "original": '''#include <stdio.h>

int alpha(int x) {
    return x + 1;
}

struct Item {
    int size;
};

int beta(int y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "int gamma(int z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 3, "anchor_head": "int alpha(int x) {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "int beta(int y) {\n    return y * 3;\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "int beta(int y) {", "span_tail": "}",
                "snippet_head": "int beta(int y) {",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 3, "end_line": 5,
                "span_head": "int alpha(int x) {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "cpp", "ext": "cpp", "filename": "original.cpp",
        "symbol_semantics": "functions, class/struct specifiers",
        "original": '''#include <string>

class Cart {
public:
    void add(int item);
};

int alpha(int x) {
    return x + 1;
}

int beta(int y) {
    return y * 2;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "int gamma(int z) {\n    return z - 1;\n}\n",
                "anchor_start_line": 8, "anchor_head": "int alpha(int x) {",
                "anchor_end_line": 10, "anchor_tail": "}",
                "snippet_head": "int gamma(int z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "int beta(int y) {\n    return y * 3;\n}\n",
                "start_line": 12, "end_line": 14,
                "span_head": "int beta(int y) {", "span_tail": "}",
                "snippet_head": "int beta(int y) {",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 8, "end_line": 10,
                "span_head": "int alpha(int x) {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "ruby", "ext": "rb", "filename": "original.rb",
        "symbol_semantics": "methods and singleton methods, classes and modules",
        "original": '''LIMIT = 10

def alpha(x)
  x + 1
end

class Cart
  def add(item)
    item
  end
end

def beta(y)
  y * 2
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "def gamma(z)\n  z - 1\nend\n",
                "anchor_start_line": 3, "anchor_head": "def alpha(x)",
                "anchor_end_line": 5, "anchor_tail": "end",
                "snippet_head": "def gamma(z)",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "def beta(y)\n  y * 3\nend\n",
                "start_line": 13, "end_line": 15,
                "span_head": "def beta(y)", "span_tail": "end",
                "snippet_head": "def beta(y)",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 3, "end_line": 5,
                "span_head": "def alpha(x)", "span_tail": "end",
            },
        ],
    },
    {
        "language": "swift", "ext": "swift", "filename": "original.swift",
        "symbol_semantics": "functions and initializers, classes/structs/protocols, property declarations",
        "original": '''let limit = 10

func alpha(x: Int) -> Int {
    return x + 1
}

struct Item {
    var name: String
}

func beta(y: Int) -> Int {
    return y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "func gamma(z: Int) -> Int {\n    return z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "func alpha(x: Int) -> Int {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "func gamma(z: Int) -> Int {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "func beta(y: Int) -> Int {\n    return y * 3\n}\n",
                "start_line": 11, "end_line": 13,
                "span_head": "func beta(y: Int) -> Int {", "span_tail": "}",
                "snippet_head": "func beta(y: Int) -> Int {",
            },
            {
                "op": "delete_symbol", "symbol": "limit",
                "start_line": 1, "end_line": 1,
                "span_head": "let limit = 10", "span_tail": "let limit = 10",
            },
        ],
    },
    {
        "language": "kotlin", "ext": "kt", "filename": "original.kt",
        "symbol_semantics": "functions, classes/objects, property declarations (constants)",
        "original": '''const val LIMIT = 10

fun alpha(x: Int): Int {
    return x + 1
}

class Cart {
    fun add(item: Int): Int {
        return item
    }
}

fun beta(y: Int): Int {
    return y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "fun gamma(z: Int): Int {\n    return z - 1\n}\n",
                "anchor_start_line": 3, "anchor_head": "fun alpha(x: Int): Int {",
                "anchor_end_line": 5, "anchor_tail": "}",
                "snippet_head": "fun gamma(z: Int): Int {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "fun beta(y: Int): Int {\n    return y * 3\n}\n",
                "start_line": 13, "end_line": 15,
                "span_head": "fun beta(y: Int): Int {", "span_tail": "}",
                "snippet_head": "fun beta(y: Int): Int {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 1, "end_line": 1,
                "span_head": "const val LIMIT = 10",
                "span_tail": "const val LIMIT = 10",
            },
        ],
    },
    {
        "language": "c_sharp", "ext": "cs", "filename": "original.cs",
        "symbol_semantics": (
            "classes/interfaces/structs, methods and constructors, fields "
            "(constants). Allman braces: the replace op passes a body-only "
            "snippet and the pipeline prepends the 2-line signature from the AST"
        ),
        "original": '''public class Sample
{
    private const int Limit = 10;

    public static int Alpha(int x)
    {
        return x + 1;
    }

    public static int Beta(int y)
    {
        return y * 2;
    }
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Alpha",
                "snippet": "    public static int Gamma(int z)\n    {\n        return z - 1;\n    }\n",
                "anchor_start_line": 5,
                "anchor_head": "    public static int Alpha(int x)",
                "anchor_end_line": 8, "anchor_tail": "    }",
                "snippet_head": "    public static int Gamma(int z)",
            },
            {
                "op": "replace_symbol", "symbol": "Beta", "path": "direct_swap",
                "snippet": "        return y * 3;\n    }\n",
                "start_line": 10, "end_line": 13,
                "span_head": "    public static int Beta(int y)",
                "span_tail": "    }",
                "snippet_head": "        return y * 3;",
                "prepend_signature_lines": 2,
            },
            {
                "op": "delete_symbol", "symbol": "Limit",
                "start_line": 3, "end_line": 3,
                "span_head": "    private const int Limit = 10;",
                "span_tail": "    private const int Limit = 10;",
            },
        ],
    },
    {
        "language": "php", "ext": "php", "filename": "original.php",
        "symbol_semantics": (
            "functions, classes/interfaces/traits, methods, const declarations. "
            "The grammar parses script content only after <?php, so a "
            "standalone-parsed redefinition snippet cannot serve direct-swap; "
            "the replace op uses the marker-bearing text-match form with an "
            "assignment body (unique replacement key)"
        ),
        "original": '''<?php

const LIMIT = 10;

function alpha($x) {
    return $x + 1;
}

function beta($y) {
    $result = $y * 2;
    return $result;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "function gamma($z) {\n    return $z - 1;\n}\n",
                "anchor_start_line": 5, "anchor_head": "function alpha($x) {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": "function gamma($z) {",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "text_match",
                "snippet": (
                    "function beta($y) {\n"
                    "// ... existing code ...\n"
                    "    $result = $y * 3;\n"
                    "    return $result;\n"
                    "// ... existing code ...\n"
                    "}\n"
                ),
                "start_line": 9, "end_line": 12,
                "span_head": "function beta($y) {",
                "span_tail": "}",
                "snippet_head": "function beta($y) {",
            },
            {
                "op": "delete_symbol", "symbol": "LIMIT",
                "start_line": 3, "end_line": 3,
                "span_head": "const LIMIT = 10;", "span_tail": "const LIMIT = 10;",
            },
        ],
    },
    {
        "language": "elixir", "ext": "ex", "filename": "original.ex",
        "symbol_semantics": "defmodule (module), def/defp functions (methods inside a module)",
        "original": '''defmodule Sample do
  @limit 10

  def alpha(x) do
    x + 1
  end

  def beta(y) do
    y * 2
  end
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "  def gamma(z) do\n    z - 1\n  end\n",
                "anchor_start_line": 4, "anchor_head": "  def alpha(x) do",
                "anchor_end_line": 6, "anchor_tail": "  end",
                "snippet_head": "  def gamma(z) do",
            },
            {
                "op": "replace_symbol", "symbol": "beta", "path": "direct_swap",
                "snippet": "  def beta(y) do\n    y * 3\n  end\n",
                "start_line": 8, "end_line": 10,
                "span_head": "  def beta(y) do", "span_tail": "  end",
                "snippet_head": "  def beta(y) do",
            },
            {
                "op": "delete_symbol", "symbol": "alpha",
                "start_line": 4, "end_line": 6,
                "span_head": "  def alpha(x) do", "span_tail": "  end",
            },
        ],
    },
    # --- the 10 B2/B3 hard-dependency data/config/markup formats ---------
    {
        "language": "html", "ext": "html", "filename": "original.html",
        "symbol_semantics": (
            "elements carrying an `id` attribute, named by the attribute VALUE "
            "(`id` is HTML's naming mechanism; tag names repeat and address "
            "nothing). Symbols nest. A replace= snippet that restates only "
            "part of the element would be an insert by the text-match editor's "
            "preserve-by-default semantics, so the golden replace op is a "
            "complete element re-definition served by the direct-swap span swap"
        ),
        "original": '''<!DOCTYPE html>
<html lang="en">
<head>
  <title>Sample</title>
</head>
<body>
  <section id="intro">
    <h1>Intro</h1>
    <p>Welcome paragraph.</p>
  </section>
  <section id="details">
    <h1>Details</h1>
    <p>Detail paragraph.</p>
  </section>
</body>
</html>
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "intro",
                "snippet": '  <section id="extra">\n    <p>Extra paragraph.</p>\n  </section>\n',
                "anchor_start_line": 7, "anchor_head": '  <section id="intro">',
                "anchor_end_line": 10, "anchor_tail": "  </section>",
                "snippet_head": '  <section id="extra">',
            },
            {
                "op": "replace_symbol", "symbol": "details", "path": "direct_swap",
                "snippet": '  <section id="details" class="wide">\n    <h2>Details</h2>\n    <p>Expanded detail paragraph.</p>\n  </section>\n',
                "start_line": 11, "end_line": 14,
                "span_head": '  <section id="details">',
                "span_tail": "  </section>",
                "snippet_head": '  <section id="details" class="wide">',
            },
            {
                "op": "delete_symbol", "symbol": "intro",
                "start_line": 7, "end_line": 10,
                "span_head": '  <section id="intro">', "span_tail": "  </section>",
            },
        ],
        "gigo": {
            "defect": "valueless attribute (attr=) in an unrelated div — the one html breakage the grammar trips on — preserved byte-exact",
            "defect_lines": [4, 4],
            "original": '''<!DOCTYPE html>
<html lang="en">
<body>
  <div attr=></div>
  <section id="intro">
    <h1>Intro</h1>
  </section>
  <section id="details">
    <h1>Details</h1>
  </section>
</body>
</html>
''',
            "op": {
                "op": "insert_after", "symbol": "details",
                "snippet": '  <section id="extra">\n    <p>Extra paragraph.</p>\n  </section>\n',
                "anchor_start_line": 8, "anchor_head": '  <section id="details">',
                "anchor_end_line": 10, "anchor_tail": "  </section>",
                "snippet_head": '  <section id="extra">',
            },
        },
    },
    {
        "language": "xml", "ext": "xml", "filename": "original.xml",
        "symbol_semantics": (
            "every element, named by its tag name; sibling repeats are refused "
            "as ambiguous (fail loud, never first-match). Symbols nest"
        ),
        "original": '''<?xml version="1.0"?>
<config>
  <server>host1</server>
  <cache enabled="true">redis</cache>
  <limits>
    <timeout>30</timeout>
  </limits>
</config>
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "server",
                "snippet": "  <replica>host2</replica>\n",
                "anchor_start_line": 3, "anchor_head": "  <server>host1</server>",
                "anchor_end_line": 3, "anchor_tail": "  <server>host1</server>",
                "snippet_head": "  <replica>host2</replica>",
            },
            {
                "op": "replace_symbol", "symbol": "cache", "path": "direct_swap",
                "snippet": '  <cache enabled="false">memcached</cache>\n',
                "start_line": 4, "end_line": 4,
                "span_head": '  <cache enabled="true">redis</cache>',
                "span_tail": '  <cache enabled="true">redis</cache>',
                "snippet_head": '  <cache enabled="false">memcached</cache>',
            },
            {
                "op": "delete_symbol", "symbol": "limits",
                "start_line": 5, "end_line": 7,
                "span_head": "  <limits>", "span_tail": "  </limits>",
            },
        ],
    },
    {
        "language": "markdown", "ext": "md", "filename": "original.md",
        "symbol_semantics": (
            "sections (heading + body), named by the heading text; sections "
            "nest so a parent section's replace/delete takes subsections with "
            "it — target leaf sections. The markdown grammar is structurally "
            "error-tolerant (plan §3 risk 1): md validity leans on text "
            "traits, never the parse gate alone"
        ),
        "original": '''# Guide

Intro paragraph.

## Setup

Install steps.

## Usage

Usage notes.
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Setup",
                "snippet": "## Troubleshooting\n\nCommon fixes.\n",
                "anchor_start_line": 5, "anchor_head": "## Setup",
                "anchor_end_line": 8, "anchor_tail": "",
                "snippet_head": "## Troubleshooting",
            },
            {
                "op": "replace_symbol", "symbol": "Usage", "path": "direct_swap",
                "snippet": "## Usage\n\nUpdated usage notes.\n",
                "start_line": 9, "end_line": 11,
                "span_head": "## Usage", "span_tail": "Usage notes.",
                "snippet_head": "## Usage",
            },
            {
                "op": "delete_symbol", "symbol": "Usage",
                "start_line": 9, "end_line": 11,
                "span_head": "## Usage", "span_tail": "Usage notes.",
            },
        ],
        "gigo": {
            "defect": "malformed table (row missing its separator column count) — parses clean by grammar leniency, preserved byte-exact; structural md validity is D1/D3 territory",
            "defect_lines": [3, 4],
            "original": '''# Guide

| broken | table |
| --- |

## Setup

Install steps.
''',
            "op": {
                "op": "insert_after", "symbol": "Setup",
                "snippet": "## Notes\n\nNotes body.\n",
                "anchor_start_line": 6, "anchor_head": "## Setup",
                "anchor_end_line": 8, "anchor_tail": "Install steps.",
                "snippet_head": "## Notes",
            },
        },
    },
    {
        "language": "json", "ext": "json", "filename": "original.json",
        "symbol_semantics": (
            "object pairs (keys), named by the key; nested objects' pairs "
            "become nested symbols. JSON comma semantics constrain insertion "
            "anchors: an inserted pair must carry its own trailing comma and "
            "the anchor pair must already have one (the last pair cannot "
            "anchor an insert)"
        ),
        "original": '''{
  "name": "fastedit",
  "version": "0.5.0",
  "config": {
    "debug": true,
    "level": 3
  },
  "tags": ["ast", "edit"]
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "name",
                "snippet": '  "license": "MIT",\n',
                "anchor_start_line": 2, "anchor_head": '  "name": "fastedit",',
                "anchor_end_line": 2, "anchor_tail": '  "name": "fastedit",',
                "snippet_head": '  "license": "MIT",',
            },
            {
                "op": "delete_symbol", "symbol": "version",
                "start_line": 3, "end_line": 3,
                "span_head": '  "version": "0.5.0",',
                "span_tail": '  "version": "0.5.0",',
            },
        ],
        "unsupported": [
            {
                "op": "replace_symbol",
                "reason": (
                    "no deterministic path can re-define a JSON pair: a bare "
                    "pair is not a standalone-parsable JSON document (the "
                    "direct-swap gate's parse precondition fails), and the "
                    "text-match editor cannot justify a same-key value "
                    "rewrite — JSON's `key: value` separator is not an "
                    "assignment (`=`), so the pair has no replacement key, "
                    "and a marker-free rewrite of a preserved line is "
                    "declined as ambiguous. JSON value updates are served by "
                    "the model path, which is out of the deterministic golden "
                    "matrix's scope"
                ),
            },
        ],
        "gigo": {
            "defect": "missing comma between two members — preserved byte-exact",
            "defect_lines": [2, 3],
            "original": '''{
  "name": "fastedit"
  "version": "0.5.0",
  "config": {
    "debug": true
  }
}
''',
            "op": {
                "op": "insert_after", "symbol": "version",
                "snippet": '  "license": "MIT",\n',
                "anchor_start_line": 3, "anchor_head": '  "version": "0.5.0",',
                "anchor_end_line": 3, "anchor_tail": '  "version": "0.5.0",',
                "snippet_head": '  "license": "MIT",',
            },
        },
    },
    {
        "language": "yaml", "ext": "yaml", "filename": "original.yaml",
        "symbol_semantics": (
            "block-mapping pairs (keys), named by the key (quotes stripped); "
            "nested mappings' pairs become nested symbols"
        ),
        "original": '''name: fastedit
version: 0.5.0
database:
  host: localhost
  port: 5432
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "version",
                "snippet": "license: MIT\n",
                "anchor_start_line": 2, "anchor_head": "version: 0.5.0",
                "anchor_end_line": 2, "anchor_tail": "version: 0.5.0",
                "snippet_head": "license: MIT",
            },
            {
                "op": "replace_symbol", "symbol": "version", "path": "direct_swap",
                "snippet": "version: 0.6.0\n",
                "start_line": 2, "end_line": 2,
                "span_head": "version: 0.5.0", "span_tail": "version: 0.5.0",
                "snippet_head": "version: 0.6.0",
            },
            {
                "op": "delete_symbol", "symbol": "port",
                "start_line": 5, "end_line": 5,
                "span_head": "  port: 5432", "span_tail": "  port: 5432",
            },
        ],
        "gigo": {
            "defect": "unclosed flow sequence in an unrelated key — preserved byte-exact",
            "defect_lines": [5, 6],
            "original": '''name: fastedit
version: 0.5.0
database:
  host: localhost
broken: [unclosed
  - bad
''',
            "op": {
                "op": "insert_after", "symbol": "database",
                "snippet": "cache:\n  enabled: true\n",
                "anchor_start_line": 3, "anchor_head": "database:",
                "anchor_end_line": 4, "anchor_tail": "  host: localhost",
                "snippet_head": "cache:",
            },
        },
    },
    {
        "language": "css", "ext": "css", "filename": "original.css",
        "symbol_semantics": (
            "rule sets, named by their full selector text (`.header`, `body`, "
            "`a:hover`); rules inside @media blocks are found by structural "
            "descent — @media itself is a container, not a symbol"
        ),
        "original": '''body {
  color: #333;
  margin: 0 auto;
}

.header {
  font-weight: bold;
}

.footer {
  padding: 4px;
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": ".header",
                "snippet": ".nav {\n  display: flex;\n}\n",
                "anchor_start_line": 6, "anchor_head": ".header {",
                "anchor_end_line": 8, "anchor_tail": "}",
                "snippet_head": ".nav {",
            },
            {
                "op": "replace_symbol", "symbol": ".footer", "path": "direct_swap",
                "snippet": ".footer {\n  padding: 8px;\n}\n",
                "start_line": 10, "end_line": 12,
                "span_head": ".footer {", "span_tail": "}",
                "snippet_head": ".footer {",
            },
            {
                "op": "delete_symbol", "symbol": "body",
                "start_line": 1, "end_line": 4,
                "span_head": "body {", "span_tail": "}",
            },
        ],
        "gigo": {
            "defect": "unclosed rule block (.broken) at EOF — preserved byte-exact",
            "defect_lines": [9, 10],
            "original": '''.header {
  font-weight: bold;
}

.footer {
  padding: 4px;
}

.broken {
  color: red;
''',
            "op": {
                "op": "insert_after", "symbol": ".footer",
                "snippet": ".nav {\n  display: flex;\n}\n",
                "anchor_start_line": 5, "anchor_head": ".footer {",
                "anchor_end_line": 7, "anchor_tail": "}",
                "snippet_head": ".nav {",
            },
        },
    },
    {
        "language": "bash", "ext": "sh", "filename": "original.sh",
        "symbol_semantics": (
            "function definitions (the `name` field is conventional). Shell "
            "variable assignments are deliberately not symbols"
        ),
        "original": '''#!/usr/bin/env bash
VERSION="1.2.3"

run_build() {
  echo "building ${VERSION}"
}

run_test() {
  echo "testing"
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "run_build",
                "snippet": 'run_clean() {\n  echo "cleaning"\n}\n',
                "anchor_start_line": 4, "anchor_head": "run_build() {",
                "anchor_end_line": 6, "anchor_tail": "}",
                "snippet_head": "run_clean() {",
            },
            {
                "op": "replace_symbol", "symbol": "run_test", "path": "direct_swap",
                "snippet": 'run_test() {\n  echo "running tests"\n}\n',
                "start_line": 8, "end_line": 10,
                "span_head": "run_test() {", "span_tail": "}",
                "snippet_head": "run_test() {",
            },
            {
                "op": "delete_symbol", "symbol": "run_build",
                "start_line": 4, "end_line": 6,
                "span_head": "run_build() {", "span_tail": "}",
            },
        ],
    },
    {
        "language": "toml", "ext": "toml", "filename": "original.toml",
        "symbol_semantics": (
            "tables and table-array elements, named by the (possibly dotted) "
            "key text; top-level bare pairs are deliberately not symbols — "
            "the table is the TOML addressing unit"
        ),
        "original": '''title = "fastedit"

[package]
name = "fastedit"
version = "0.5.0"

[dependencies]
serde = "1"

[tool.pytest]
minversion = "7.0"
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "dependencies",
                "snippet": '[dev-dependencies]\npytest = "7.0"\n',
                "anchor_start_line": 7, "anchor_head": "[dependencies]",
                "anchor_end_line": 9, "anchor_tail": "",
                "snippet_head": "[dev-dependencies]",
            },
            {
                "op": "replace_symbol", "symbol": "tool.pytest", "path": "direct_swap",
                "snippet": '[tool.pytest]\nminversion = "8.0"\n',
                "start_line": 10, "end_line": 11,
                "span_head": "[tool.pytest]", "span_tail": 'minversion = "7.0"',
                "snippet_head": "[tool.pytest]",
            },
            {
                "op": "delete_symbol", "symbol": "package",
                "start_line": 3, "end_line": 6,
                "span_head": "[package]", "span_tail": "",
            },
        ],
    },
    {
        "language": "sql", "ext": "sql", "filename": "original.sql",
        "symbol_semantics": (
            "CREATE statements that define a schema object (table / view / "
            "materialized view / index), named by the object name; DML "
            "(SELECT/INSERT/UPDATE) is an operation, never a symbol"
        ),
        "original": '''CREATE TABLE users (
  id INT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT
);

CREATE INDEX idx_orders_user ON orders (user_id);
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "users",
                "snippet": "CREATE TABLE sessions (\n  id INT PRIMARY KEY\n);\n",
                "anchor_start_line": 1, "anchor_head": "CREATE TABLE users (",
                "anchor_end_line": 4, "anchor_tail": ");",
                "snippet_head": "CREATE TABLE sessions (",
            },
            {
                "op": "replace_symbol", "symbol": "orders", "path": "direct_swap",
                "snippet": "CREATE TABLE orders (\n  id INT PRIMARY KEY,\n  user_id INT NOT NULL\n);\n",
                "start_line": 6, "end_line": 9,
                "span_head": "CREATE TABLE orders (", "span_tail": ");",
                "snippet_head": "CREATE TABLE orders (",
            },
            {
                "op": "delete_symbol", "symbol": "idx_orders_user",
                "start_line": 11, "end_line": 11,
                "span_head": "CREATE INDEX idx_orders_user ON orders (user_id);",
                "span_tail": "CREATE INDEX idx_orders_user ON orders (user_id);",
            },
        ],
        "gigo": {
            "defect": "broken CREATE TABLE (missing closing paren) in an unrelated statement — preserved byte-exact",
            "defect_lines": [7, 9],
            "original": '''CREATE TABLE users (
  id INT PRIMARY KEY
);

CREATE INDEX idx_users ON users (id);

CREATE TABLE broken (
  id INT
;
''',
            "op": {
                "op": "insert_after", "symbol": "users",
                "snippet": "CREATE TABLE sessions (\n  id INT PRIMARY KEY\n);\n",
                "anchor_start_line": 1, "anchor_head": "CREATE TABLE users (",
                "anchor_end_line": 3, "anchor_tail": ");",
                "snippet_head": "CREATE TABLE sessions (",
            },
        },
    },
    {
        "language": "dockerfile", "ext": "dockerfile", "filename": "original.dockerfile",
        "symbol_semantics": (
            "FROM instructions (build stages), named by the `AS` alias when "
            "present, else by the image spec; other instructions are steps, "
            "not symbols"
        ),
        "original": '''FROM python:3.12 AS builder

RUN pip install requests

FROM alpine AS runtime

COPY --from=builder /app /app

CMD ["python"]
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "builder",
                "snippet": "WORKDIR /build\n",
                "anchor_start_line": 1, "anchor_head": "FROM python:3.12 AS builder",
                "anchor_end_line": 1, "anchor_tail": "FROM python:3.12 AS builder",
                "snippet_head": "WORKDIR /build",
            },
            {
                "op": "replace_symbol", "symbol": "runtime", "path": "direct_swap",
                "snippet": "FROM alpine:3.19 AS runtime\n",
                "start_line": 5, "end_line": 5,
                "span_head": "FROM alpine AS runtime",
                "span_tail": "FROM alpine AS runtime",
                "snippet_head": "FROM alpine:3.19 AS runtime",
            },
            {
                "op": "delete_symbol", "symbol": "builder",
                "start_line": 1, "end_line": 1,
                "span_head": "FROM python:3.12 AS builder",
                "span_tail": "FROM python:3.12 AS builder",
            },
        ],
    },
    # --- all-grammars extra samples (one representative op each) ---------
    {
        "language": "lua", "ext": "lua", "filename": "original.lua",
        "requires_wheel": "tree_sitter_lua",
        "symbol_semantics": (
            "function declarations, dotted names verbatim (`M.alpha`). "
            "all-grammars extra: extension-unwired by design, so the golden "
            "runner passes the language explicitly (the resolver's "
            "`resolve when requested explicitly` contract)"
        ),
        "original": '''local M = {}

function M.alpha(x)
  return x + 1
end

function M.beta(y)
  return y * 2
end
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "M.alpha",
                "snippet": "function M.gamma(z)\n  return z - 1\nend\n",
                "anchor_start_line": 3, "anchor_head": "function M.alpha(x)",
                "anchor_end_line": 5, "anchor_tail": "end",
                "snippet_head": "function M.gamma(z)",
            },
        ],
    },
    {
        "language": "scala", "ext": "scala", "filename": "original.scala",
        "requires_wheel": "tree_sitter_scala",
        "symbol_semantics": "objects/classes/traits, defs (methods inside a type)",
        "original": '''object Sample {
  val limit = 10

  def alpha(x: Int): Int = x + 1

  def beta(y: Int): Int = y * 2
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "alpha",
                "snippet": "  def gamma(z: Int): Int = z - 1\n",
                "anchor_start_line": 4,
                "anchor_head": "  def alpha(x: Int): Int = x + 1",
                "anchor_end_line": 4, "anchor_tail": "  def alpha(x: Int): Int = x + 1",
                "snippet_head": "  def gamma(z: Int): Int = z - 1",
            },
        ],
    },
    {
        "language": "graphql", "ext": "graphql", "filename": "original.graphql",
        "requires_wheel": "tree_sitter_graphql",
        "symbol_semantics": "schema type definitions (object/interface/enum/union/input/scalar), named by their `name` field",
        "original": '''type User {
  id: ID!
  name: String
}

type Post {
  id: ID!
  author: User
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "User",
                "snippet": "type Comment {\n  id: ID!\n}\n",
                "anchor_start_line": 1, "anchor_head": "type User {",
                "anchor_end_line": 4, "anchor_tail": "}",
                "snippet_head": "type Comment {",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# F2/F3: census-batch golden fixtures.
#
# Split (tests/golden/pack_census.json, probe_version 1):
#   * batch 1 (F2) = the FIRST 87 of the 144 ok-names (alphabetically
#     actionscript .. odin). The 15 B3-covered names inside that range
#     (bash, c, c_sharp, cpp, css, dockerfile, elixir, go, html, java,
#     javascript, json, kotlin, lua, markdown) are SKIPPED here — they
#     already have manifests. That left 72 languages, authored below.
#   * batch 2 (F3) = the remaining 57 ok-names minus the 12 B3-covered names
#     in that range (php, python, ruby, rust, scala, sql, swift, toml, tsx,
#     typescript, xml, yaml) = 45 languages, authored below; plus the 14
#     parse_degraded names left after F2's batch-1 retries (pony, prisma,
#     proto, qmljs, query, smali, smithy, test, ungrammar, uxntal, vhdl,
#     wast, wat, xml_dtd).
#   * F3 degraded retries: 13 of the 14 parse with zero error traits using a
#     REAL per-language snippet (the census's trivial probe line was the
#     problem, not the grammar) — authored below with a census_note, and
#     their census verdicts are flipped to ok in tests/golden/
#     pack_census.json, as are F2's 15 batch-1 retries and graphql (whose
#     B3 fixture already proves its grammar). The one genuine grammar
#     defect — `test` — stayed un-fabricated through F3 (recorded with the
#     exact parse errors); G1b lifted it: G1a's artifact filter makes
#     same-record edits land, G1b adds the declarative anchoring row, and
#     the exercised fixture below carries the defect record as
#     ``grammar_defect_workaround`` (fail-loud documentation, re-pinned by
#     tests/test_golden_matrix.py::test_grammar_defect_workaround_is_real).
#
# Anchoring honesty: fastedit's declarative symbol anchoring
# (_FUNCTION_LIKE_NODE_TYPES / _CLASS_LIKE_NODE_TYPES / _CONST_LIKE_NODE_TYPES
# / _FORMAT_SYMBOL_SPECS in src/fastedit/inference/ast_utils.py) has rows for
# neither of these languages yet, so after=/replace=/delete= cannot resolve a
# symbol span (the in-memory symbol map is empty and the ops fail loud with
# "Symbol not found"). The ONLY census name that anchors today is
# "csharp" — a LANGUAGE_NAME_ALIASES spelling of the canonical c_sharp, whose
# B3 anchoring serves it through the pack-name alias path (exercised e2e
# below). Every other census language (batch 1 and batch 2 alike) declares
# its anchoring ops ``unsupported`` with the reason — the honest hole
# F-series anchoring work will fill by adding one declarative table row and
# flipping these to exercised ops.
# ---------------------------------------------------------------------------

_NO_ANCHORING_REASON = (
    "fastedit's declarative symbol anchoring (_FUNCTION_LIKE_NODE_TYPES / "
    "_CLASS_LIKE_NODE_TYPES / _CONST_LIKE_NODE_TYPES / _FORMAT_SYMBOL_SPECS "
    "in src/fastedit/inference/ast_utils.py) has no row for this language "
    "yet: the in-memory symbol map is empty and the op fails loud with "
    "'Symbol not found' rather than guessing an anchor. Adding the one "
    "declarative anchoring row and flipping this to an exercised golden op "
    "is future F-series work — declared here, never silent."
)


def _anchoring_unsupported() -> list[dict]:
    """The three anchoring ops, all declared unsupported (no symbol row)."""
    return [
        {"op": op, "reason": _NO_ANCHORING_REASON}
        for op in ("insert_after", "replace_symbol", "delete_symbol")
    ]


_CENSUS_PACK = "tree_sitter_language_pack"

CENSUS_LANGUAGES: list[dict] = [
    # --- batch-1 ok-names, actionscript .. odin (72; csharp exercises ops) -
    {
        "language": "actionscript", "ext": "as", "filename": "original.as",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "packages, classes, functions/methods — but fastedit anchoring: "
            "none yet (see the unsupported reasons below)"
        ),
        "original": '''package {
    public class Greeter {
        public function greet(name:String):String {
            return "Hello, " + name;
        }
    }
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "agda", "ext": "agda", "filename": "original.agda",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, typed function clauses — but fastedit anchoring: none yet"
        ),
        "original": '''module Sample where

open import Agda.Builtin.Nat

double : Nat -> Nat
double zero = zero
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "apex", "ext": "cls", "filename": "original.cls",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "classes, static methods (Java-like, single-quoted strings) — "
            "but fastedit anchoring: none yet"
        ),
        "original": '''public class Greeter {
    public static String greet(String name) {
        return 'Hello, ' + name;
    }
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "arduino", "ext": "ino", "filename": "original.ino",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "setup()/loop() sketch functions — but fastedit anchoring: none yet"
        ),
        "original": '''void setup() {
  pinMode(13, OUTPUT);
}

void loop() {
  digitalWrite(13, HIGH);
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "asm", "ext": "asm", "filename": "original.asm",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "labels and instructions (NASM-style text section) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''section .text
global add
add:
    mov rax, rdi
    ret
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "astro", "ext": "astro", "filename": "original.astro",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "component frontmatter (--- fenced JS) + template — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''---
const name = "world";
---
<p>Hello {name}</p>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "beancount", "ext": "beancount", "filename": "original.beancount",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "account directives and transactions (line-oriented ledger) — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''2024-01-01 open Assets:Cash

2024-01-15 * "Pay"
  Assets:Cash             100 USD
  Income:Work
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "bibtex", "ext": "bib", "filename": "original.bib",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "@type{key, field = value} entries — but fastedit anchoring: none yet"
        ),
        "original": '''@article{key2020,
  title = {A Title},
  year = {2020}
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "bicep", "ext": "bicep", "filename": "original.bicep",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "param/variable/resource declarations — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''param location string = 'eastus'

resource sa 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'sample'
  location: location
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "bitbake", "ext": "bb", "filename": "original.bb",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "recipe variables + do_<task> shell/python tasks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''SUMMARY = "Sample recipe"
LICENSE = "MIT"

do_install() {
    install -d ${D}${bindir}
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "bsl", "ext": "bsl", "filename": "original.bsl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "1C:Enterprise Procedure/EndProcedure blocks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''Procedure Hello()
    Message("hello");
EndProcedure
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "cairo", "ext": "cairo", "filename": "original.cairo",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Cairo 1.0 functions (felt252 args) — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''fn add(a: felt252, b: felt252) -> felt252 {
    a + b
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "capnp", "ext": "capnp", "filename": "original.capnp",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Cap'n Proto structs with numbered fields — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''@0xdbb9ad1f14bf0b36;

struct Point {
  x @0 :Int32;
  y @1 :Int32;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "chatito", "ext": "chatito", "filename": "original.chatito",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) classified this language parse_degraded from the "
            "trivial probe lines; this real %[intent:...] block parses with "
            "zero error traits — the census probe was the problem, the "
            "grammar is healthy; the committed census snapshot already "
            "classifies chatito ok"
        ),
        "symbol_semantics": (
            "%[intent:...] blocks with query/alias lines — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''%[intent:greet]
    query: hello
    query: hi there
    alias: hi
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "clojure", "ext": "clj", "filename": "original.clj",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "defn forms, def vars — but fastedit anchoring: none yet"
        ),
        "original": '''(defn add [x y]
  (+ x y))

(def limit 10)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "cmake", "ext": "cmake", "filename": "original.cmake",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function()/macro() blocks and set() calls — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''cmake_minimum_required(VERSION 3.20)
project(sample C)

set(LIMIT 10)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "comment", "ext": "comment", "filename": "original.comment",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "the aggregate `comment` grammar: bare comment lines (no symbol "
            "notion at all — a pure text format); fastedit anchoring: none yet"
        ),
        "original": '''# a leading comment
# a second comment line
# a third comment line
# and a fourth
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "commonlisp", "ext": "lisp", "filename": "original.lisp",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "defun forms, defparameter vars — but fastedit anchoring: none yet"
        ),
        "original": '''(defun add (x y)
  (+ x y))

(defvar *limit* 10)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "cpon", "ext": "cpon", "filename": "original.cpon",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "OpenCAP Cpon: JSON-shaped objects with name/value pairs — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''{
  "server": "host1",
  "port": 8080
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "csv", "ext": "csv", "filename": "original.csv",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "RFC4180 rows with a header line (no symbol notion at all — a "
            "pure data grid); fastedit anchoring: none yet"
        ),
        "original": '''id,name
1,fastedit
2,golden
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "cuda", "ext": "cu", "filename": "original.cu",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "__global__ kernels and device functions — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''__global__ void scale(float* x, int n) {
    int i = threadIdx.x;
    x[i] = x[i] * 2.0f;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "d", "ext": "d", "filename": "original.d",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, functions — but fastedit anchoring: none yet"
        ),
        "original": '''module sample;

int add(int a, int b) {
    return a + b;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "dart", "ext": "dart", "filename": "original.dart",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "top-level functions, classes — but fastedit anchoring: none yet"
        ),
        "original": '''int add(int a, int b) {
  return a + b;
}

int twice(int x) {
  return x * 2;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "elisp", "ext": "el", "filename": "original.el",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "defun forms, defvar vars — but fastedit anchoring: none yet"
        ),
        "original": '''(defun add (x y)
  (+ x y))

(defvar limit 10)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "elm", "ext": "elm", "filename": "original.elm",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, typed top-level functions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''module Sample exposing (add)

add : Int -> Int -> Int
add x y =
    x + y
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "embeddedtemplate", "ext": "et", "filename": "original.et",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "embedded-template content with <%= expr %> codelets — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''Hello <%= name %>!

Items:
  <%= first %>
  <%= second %>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "erlang", "ext": "erl", "filename": "original.erl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "-module attributes, function clauses — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''-module(sample).
-export([add/2]).

add(A, B) ->
    A + B.
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "fennel", "ext": "fnl", "filename": "original.fnl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "(fn ...) forms, (local ...) bindings — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''(local limit 10)

(fn add [x y]
  (+ x y))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "fish", "ext": "fish", "filename": "original.fish",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function ... end definitions — but fastedit anchoring: none yet"
        ),
        "original": '''function add
    echo (math $argv[1] + $argv[2])
end
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "fortran", "ext": "f90", "filename": "original.f90",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "modules, contains'd functions — but fastedit anchoring: none yet"
        ),
        "original": '''module sample_mod
  implicit none
contains
  function add(a, b) result(c)
    integer, intent(in) :: a, b
    integer :: c
    c = a + b
  end function add
end module sample_mod
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "fsharp", "ext": "fs", "filename": "original.fs",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, let-bound functions — but fastedit anchoring: none yet"
        ),
        "original": '''module Sample

let add x y = x + y

let twice x = x * 2
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "func", "ext": "func", "filename": "original.func",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "FunC (TON) C-like function definitions — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''int add(int a, int b) {
  return a + b;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gdscript", "ext": "gd", "filename": "original.gd",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "extends + func declarations — but fastedit anchoring: none yet"
        ),
        "original": '''extends Node

func add(a, b):
    return a + b
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gitattributes", "ext": "gitattributes", "filename": "original.gitattributes",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "pattern + attribute lines (no symbol notion at all — a pure "
            "data format); fastedit anchoring: none yet"
        ),
        "original": '''*.py text eol=lf
*.png binary
*.md text diff
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gitcommit", "ext": "gitcommit", "filename": "original.gitcommit",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "subject line + body paragraphs (no symbol notion at all); "
            "fastedit anchoring: none yet"
        ),
        "original": '''Add golden fixtures

Body line explaining the change.
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gitignore", "ext": "gitignore", "filename": "original.gitignore",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "pattern lines (no symbol notion at all — a pure data format); "
            "fastedit anchoring: none yet"
        ),
        "original": '''node_modules/
__pycache__/
*.pyc
dist/
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gleam", "ext": "gleam", "filename": "original.gleam",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "import + pub fn definitions — but fastedit anchoring: none yet"
        ),
        "original": '''import gleam/io

pub fn main() {
  io.println("hello")
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "glsl", "ext": "glsl", "filename": "original.glsl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "shader entry points (void main()) — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''#version 330 core
void main() {
    gl_Position = vec4(0.0);
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gn", "ext": "gn", "filename": "original.gn",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "target blocks (executable(\"name\") { ... }) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''executable("sample") {
  sources = [ "main.cc" ]
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "groovy", "ext": "groovy", "filename": "original.groovy",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "def methods and script variables — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''def add(x, y) {
    return x + y
}

def limit = 10
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gstlaunch", "ext": "gstlaunch", "filename": "original.gstlaunch",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "gst-launch pipeline descriptions (element ! element — no "
            "symbol notion at all); fastedit anchoring: none yet"
        ),
        "original": '''fakesrc num-buffers=10 ! fakesink

fakesrc ! queue ! fakesink
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "hack", "ext": "hack", "filename": "original.hack",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "<?hh functions with typed signatures — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''<?hh

function add(int $x, int $y): int {
  return $x + $y;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "haskell", "ext": "hs", "filename": "original.hs",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, typed top-level functions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''module Sample where

add :: Int -> Int -> Int
add x y = x + y
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "haxe", "ext": "hx", "filename": "original.hx",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "classes, static functions — but fastedit anchoring: none yet"
        ),
        "original": '''class Greeter {
    static function main() {
        trace("hello");
    }
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "hcl", "ext": "hcl", "filename": "original.hcl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "block bodies (variable \"name\" { ... }) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''variable "name" {
  type = string
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "heex", "ext": "heex", "filename": "original.heex",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Phoenix HEEx template elements with interpolation — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''<div>
  <p>{@greeting}</p>
</div>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "hlsl", "ext": "hlsl", "filename": "original.hlsl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "shader functions with semantics annotations — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''float4 main(float4 pos : SV_POSITION) : SV_POSITION {
    return pos;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "hyprlang", "ext": "hyprlang", "filename": "original.hyprlang",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Hyprland config sections and key = value lines — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''general {
    gaps_in = 5
}

bind = SUPER, Q, exec, kitty
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ini", "ext": "ini", "filename": "original.ini",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "sections ([name]) with key = value pairs — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''[core]
name = fastedit

[server]
host = localhost
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ispc", "ext": "ispc", "filename": "original.ispc",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "export/task functions — but fastedit anchoring: none yet"
        ),
        "original": '''export float scale(float x) {
    return x * 2.0;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "janet", "ext": "janet", "filename": "original.janet",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "(defn ...) forms, (def ...) bindings — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''(defn add [x y]
  (+ x y))

(def limit 10)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "jsonnet", "ext": "jsonnet", "filename": "original.jsonnet",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "object fields and methods — but fastedit anchoring: none yet"
        ),
        "original": '''{
  name: "fastedit",
  add(a, b): a + b,
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "julia", "ext": "jl", "filename": "original.jl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function/short-form definitions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''function add(x, y)
    return x + y
end

twice(x) = x * 2
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "kconfig", "ext": "kconfig", "filename": "original.kconfig",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "config blocks with typed prompts — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''config FASTEDIT
	bool "Enable fastedit"
	default y
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "kdl", "ext": "kdl", "filename": "original.kdl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "nodes with arguments and child blocks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''name "fastedit"
server {
    host "localhost"
    port 5432
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "latex", "ext": "tex", "filename": "original.tex",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "document environment with sections — but fastedit anchoring: "
            "none yet"
        ),
        "original": r'''\documentclass{article}
\begin{document}
Hello.
\end{document}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "luap", "ext": "luap", "filename": "original.luap",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Lua pattern expressions (captures/classes — no symbol notion "
            "at all); fastedit anchoring: none yet"
        ),
        "original": '''(%a+)%s+(%d+)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "luau", "ext": "luau", "filename": "original.luau",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "local functions with type annotations — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''local function add(a: number, b: number): number
    return a + b
end
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "magik", "ext": "magik", "filename": "original.magik",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "_method/_endmethod blocks — but fastedit anchoring: none yet"
        ),
        "original": '''_method point.add(other)
	_return _self.x + other.x
_endmethod
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "make", "ext": "mak", "filename": "original.mak",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "rules (targets, recipes) and variables — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''all: build

build:
	echo building
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "markdown_inline", "ext": "md_inline", "filename": "original.md_inline",
        "census_fixture": True,
        "symbol_semantics": (
            "the markdown INLINE sub-grammar (emphasis/links inside one "
            "block of prose — no symbol notion at all); fastedit anchoring: "
            "none yet. Served by the hard-dependency tree_sitter_markdown "
            "wheel's inline_language entry, so this fixture needs no pack"
        ),
        "original": '''a *bold* word
and _italic_ text
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "matlab", "ext": "matlab", "filename": "original.matlab",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function definitions — but fastedit anchoring: none yet"
        ),
        "original": '''function c = add(a, b)
    c = a + b;
end
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "meson", "ext": "meson", "filename": "original.meson",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "project()/executable() calls — but fastedit anchoring: none yet"
        ),
        "original": '''project('sample', 'c')

executable('app', 'main.c')
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "netlinx", "ext": "axs", "filename": "original.axs",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "DEFINE_DEVICE/DEFINE_START sections — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''DEFINE_DEVICE

dvPanel = 128:1:0
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "nim", "ext": "nim", "filename": "original.nim",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "proc definitions — but fastedit anchoring: none yet"
        ),
        "original": '''proc add(x, y: int): int =
  x + y

proc twice(x: int): int =
  x * 2
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ninja", "ext": "ninja", "filename": "original.ninja",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "rule/build statements (no function symbol notion — a build "
            "graph); fastedit anchoring: none yet"
        ),
        "original": '''rule cc
  command = gcc -c $in -o $out

build foo.o: cc foo.c
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "nix", "ext": "nix", "filename": "original.nix",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "attribute sets with bindings — but fastedit anchoring: none yet"
        ),
        "original": '''{
  name = "fastedit";
  version = "0.5.0";
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "nqc", "ext": "nqc", "filename": "original.nqc",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "task main() blocks (C-like) — but fastedit anchoring: none yet"
        ),
        "original": '''task main() {
    OnFwd(OUT_A);
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "objc", "ext": "m", "filename": "original.m",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "@interface/@implementation blocks, methods — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''#import <Foundation/Foundation.h>

@interface Greeter : NSObject
- (NSString *)greet;
@end
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ocaml", "ext": "ml", "filename": "original.ml",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "let-bound functions — but fastedit anchoring: none yet"
        ),
        "original": '''let add x y = x + y

let twice x = x * 2
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "odin", "ext": "odin", "filename": "original.odin",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "package + proc declarations — but fastedit anchoring: none yet"
        ),
        "original": '''package sample

add :: proc(a, b: int) -> int {
    return a + b
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "csharp", "ext": "cs", "filename": "original.cs",
        "census_fixture": True,
        "symbol_semantics": (
            "same canonical grammar as c_sharp (LANGUAGE_NAME_ALIASES maps "
            "the pack's 'csharp' spelling to c_sharp): classes, methods, "
            "fields (constants). Allman braces; the replace op passes a "
            "body-only snippet and the pipeline prepends the 2-line "
            "signature from the AST. This fixture exercises the ALIAS path "
            "end to end — language='csharp' canonicalizes before the parser "
            "cache key and the B3 anchoring serves it"
        ),
        "census_note": (
            "census name 'csharp' canonicalizes to c_sharp, which B3's "
            "golden dir already covers via the direct wheel name; this "
            "batch-1 fixture additionally proves the pack-spelling alias "
            "resolves, anchors, and edits byte-exactly"
        ),
        "original": '''public class Cart
{
    private const int Capacity = 4;

    public static int Add(int items)
    {
        return items + 1;
    }

    public static int Fill(int items)
    {
        return items * 2;
    }
}
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "Add",
                "snippet": "    public static int Remove(int z)\n    {\n        return z - 1;\n    }\n",
                "anchor_start_line": 5, "anchor_head": "    public static int Add(int items)",
                "anchor_end_line": 8, "anchor_tail": "    }",
                "snippet_head": "    public static int Remove(int z)",
            },
            {
                "op": "replace_symbol", "symbol": "Fill", "path": "direct_swap",
                "snippet": "        return items * 3;\n    }\n",
                "start_line": 10, "end_line": 13,
                "span_head": "    public static int Fill(int items)",
                "span_tail": "    }",
                "snippet_head": "        return items * 3;",
                "prepend_signature_lines": 2,
            },
            {
                "op": "delete_symbol", "symbol": "Capacity",
                "start_line": 3, "end_line": 3,
                "span_head": "    private const int Capacity = 4;",
                "span_tail": "    private const int Capacity = 4;",
            },
        ],
    },
    # --- parse_degraded names inside batch 1's range, retried with REAL ---
    # --- snippets (the census probe line was the problem, not the grammar)
    {
        "language": "ada", "ext": "adb", "filename": "original.adb",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real procedure body parses with zero error traits "
            "— the census probe was the problem, the grammar is healthy. "
            "Census verdict flipped to ok in F3 (a --force re-probe "
            "re-derives the trivial-probe verdict; this fixture is the "
            "durable proof)"
        ),
        "symbol_semantics": (
            "procedure bodies with declarations — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''with Ada.Text_IO;

procedure Sample is
begin
   Ada.Text_IO.Put_Line ("hello");
end Sample;
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "clarity", "ext": "clar", "filename": "original.clar",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real define-public contract function parses with "
            "zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "(define-public ...) contract functions — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''(define-public (increment (val uint))
  (ok (+ val u1))
)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "doxygen", "ext": "doxygen", "filename": "original.doxygen",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real @brief/@param comment parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "doxygen comment text (@brief/@param/@return tags — no code "
            "symbol notion at all); fastedit anchoring: none yet"
        ),
        "original": '''/**
 * @brief Adds two numbers.
 * @param x first
 * @return the sum
 */
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "dtd", "ext": "dtd", "filename": "original.dtd",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real <!ELEMENT> document parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "<!ELEMENT> declarations (no function symbol notion); fastedit "
            "anchoring: none yet"
        ),
        "original": '''<!ELEMENT note (to,from)>
<!ELEMENT to (#PCDATA)>
<!ELEMENT from (#PCDATA)>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "firrtl", "ext": "fir", "filename": "original.fir",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real circuit/module body parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "circuit/module hardware blocks — but fastedit anchoring: none yet"
        ),
        "original": '''circuit Sample :
  module Sample :
    output out : UInt<8>
    out <= UInt<8>(1)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "fsharp_signature", "ext": "fsi", "filename": "original.fsi",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real namespace + val signature file parses with "
            "zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "signature files: namespace + val type signatures — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''namespace Sample

val limit : int
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gomod", "ext": "gomod", "filename": "original.gomod",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real go.mod parses with zero error traits — the "
            "census probe was the problem, the grammar is healthy. Census "
            "verdict flipped to ok in F3 (a --force re-probe re-derives the "
            "trivial-probe verdict; this fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "module/require directives (line-oriented — no function symbol "
            "notion); fastedit anchoring: none yet"
        ),
        "original": '''module example.com/sample

go 1.21
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "gosum", "ext": "gosum", "filename": "original.gosum",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real go.sum content parses with zero error traits "
            "— the census probe was the problem, the grammar is healthy. "
            "Census verdict flipped to ok in F3 (a --force re-probe "
            "re-derives the trivial-probe verdict; this fixture is the "
            "durable proof)"
        ),
        "symbol_semantics": (
            "module version/hash lines (no symbol notion at all — a pure "
            "data format); fastedit anchoring: none yet"
        ),
        "original": '''example.com/mod v1.0.0 h1:abc=
example.com/mod v1.0.0/go.mod h1:def=
example.com/other v2.1.3 h1:ghi=
example.com/other v2.1.3/go.mod h1:jkl=
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "hare", "ext": "ha", "filename": "original.ha",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real exported function parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "use declarations, fn definitions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''use strings;

export fn add(x: int, y: int) int = {
	return x + y;
};
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "jsdoc", "ext": "jsdoc", "filename": "original.jsdoc",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real @param/@returns block parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "JSDoc comment text (@param/@returns tags — no code symbol "
            "notion at all); fastedit anchoring: none yet"
        ),
        "original": '''/**
 * Adds numbers.
 * @param {number} x first
 * @returns {number} sum
 */
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "linkerscript", "ext": "ld", "filename": "original.ld",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real SECTIONS script parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "SECTIONS/output-section blocks — but fastedit anchoring: none yet"
        ),
        "original": '''SECTIONS
{
  .text : { *(.text) }
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "llvm", "ext": "ll", "filename": "original.ll",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real define block parses with zero error traits — "
            "the census probe was the problem, the grammar is healthy. "
            "Census verdict flipped to ok in F3 (a --force re-probe "
            "re-derives the trivial-probe verdict; this fixture is the "
            "durable proof)"
        ),
        "symbol_semantics": (
            "define @function blocks — but fastedit anchoring: none yet"
        ),
        "original": '''define i32 @add(i32 %a, i32 %b) {
  %s = add i32 %a, %b
  ret i32 %s
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "luadoc", "ext": "luadoc", "filename": "original.luadoc",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real @class/@field doc content parses with zero "
            "error traits — the census probe was the problem (the grammar "
            "parses the doc-comment CONTENT without the leading dashes, "
            "and cleanly accepts a single @field per document). Census "
            "verdict flipped to ok in F3 (a --force re-probe re-derives the "
            "trivial-probe verdict; this fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "lua doc-comment annotations (@class/@field — no code symbol "
            "notion at all); fastedit anchoring: none yet"
        ),
        "original": '''@class Point
@field x number the x coordinate
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "mermaid", "ext": "mmd", "filename": "original.mmd",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real flowchart parses with zero error traits — "
            "the census probe was the problem, the grammar is healthy. "
            "Census verdict flipped to ok in F3 (a --force re-probe "
            "re-derives the trivial-probe verdict; this fixture is the "
            "durable proof)"
        ),
        "symbol_semantics": (
            "diagram declarations with nodes and edges — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''flowchart TD
    A[Start] --> B[End]
    B --> C{Done}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ocaml_interface", "ext": "mli", "filename": "original.mli",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real .mli signature (val + type) parses with zero "
            "error traits — the census probe was the problem, the grammar "
            "is healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "signature values (val ... : type) and type declarations — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''val add : int -> int -> int

type point = { x : int; y : int }
''',
        "unsupported": _anchoring_unsupported(),
    },
    # --- batch-2 (F3) ok-names, org .. zig (45; none anchor yet) ----------
    {
        "language": "org", "ext": "org", "filename": "original.org",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "outline headlines with body text — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''* Heading one
Some text.

* Heading two
More text.
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "pascal", "ext": "pas", "filename": "original.pas",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "program blocks, procedures/functions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''program Sample;
var
  X: Integer;
begin
  X := 1;
end.
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "pem", "ext": "pem", "filename": "original.pem",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "BEGIN/END PEM blocks (certificates and keys — no code symbol "
            "notion at all); fastedit anchoring: none yet"
        ),
        "original": '''-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAMlyFqJGvW6K
TQIDBAAK
-----END CERTIFICATE-----
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "perl", "ext": "pl", "filename": "original.pl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "sub definitions, package blocks — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''use strict;
use warnings;

sub add {
    my ($x, $y) = @_;
    return $x + $y;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "pgn", "ext": "pgn", "filename": "original.pgn",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "chess games: tag pairs + movetext (no code symbol notion at "
            "all); fastedit anchoring: none yet"
        ),
        "original": '''[Event "Sample"]

1. e4 e5 2. Nf3 Nc6
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "po", "ext": "po", "filename": "original.po",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "msgid/msgstr translation entries (no code symbol notion at "
            "all); fastedit anchoring: none yet"
        ),
        "original": '''msgid "hello"
msgstr "ciao"
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "powershell", "ext": "ps1", "filename": "original.ps1",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function definitions with param blocks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''function Add-Numbers {
    param($x, $y)
    return $x + $y
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "printf", "ext": "printf", "filename": "original.printf",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "printf format-string conversion specs (no code symbol notion "
            "at all); fastedit anchoring: none yet"
        ),
        "original": '''Hello %s, %d items (%5.2f%%)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "properties", "ext": "properties", "filename": "original.properties",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "key=value lines (no code symbol notion at all — a pure data "
            "format); fastedit anchoring: none yet"
        ),
        "original": '''name=fastedit
server.host=localhost
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "psv", "ext": "psv", "filename": "original.psv",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "pipe-delimited rows with a header line (no symbol notion at "
            "all — a pure data grid); fastedit anchoring: none yet"
        ),
        "original": '''id|name
1|fastedit
2|golden
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "puppet", "ext": "pp", "filename": "original.pp",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "resource declarations, class definitions — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''package { 'nginx':
  ensure => installed,
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "purescript", "ext": "purs", "filename": "original.purs",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module, typed top-level functions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''module Sample where

add :: Int -> Int -> Int
add x y = x + y
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "pymanifest", "ext": "pymanifest", "filename": "original.pymanifest",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "sdist manifest directives (include/recursive-include — no code "
            "symbol notion at all); fastedit anchoring: none yet"
        ),
        "original": '''include README.md
recursive-include src *.py
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "qmldir", "ext": "qmldir", "filename": "original.qmldir",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "QML module directory directives (type registrations — no code "
            "symbol notion at all); fastedit anchoring: none yet"
        ),
        "original": '''module Sample
Constants 1.0 Constants.qml
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "r", "ext": "r", "filename": "original.r",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function assignments, library calls — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''add <- function(x, y) {
  x + y
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "racket", "ext": "rkt", "filename": "original.rkt",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "#lang modules, define forms — but fastedit anchoring: none yet"
        ),
        "original": '''#lang racket

(define (add x y)
  (+ x y))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "re2c", "ext": "re2c", "filename": "original.re2c",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "lexer rules (regex { action }) and configurations — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''[0-9]+ { return 1; }
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "readline", "ext": "inputrc", "filename": "original.inputrc",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "key bindings and variable settings (no code symbol notion at "
            "all); fastedit anchoring: none yet"
        ),
        "original": '''set editing-mode vi
"\\C-a": beginning-of-line
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "rego", "ext": "rego", "filename": "original.rego",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "package + rule definitions — but fastedit anchoring: none yet"
        ),
        "original": '''package sample

allow {
    input.role == "admin"
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "requirements", "ext": "requirements", "filename": "original.requirements",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "pinned requirement lines (no code symbol notion at all — a "
            "pure data format); fastedit anchoring: none yet"
        ),
        "original": '''pytest>=7.0
ruff==0.16.6
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ron", "ext": "ron", "filename": "original.ron",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "Rusty Object Notation structs/fields (data — no code symbol "
            "notion at all); fastedit anchoring: none yet"
        ),
        "original": '''(
    name: "fastedit",
    version: 5,
)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "rst", "ext": "rst", "filename": "original.rst",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "sections (title + underline) and directives — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''Title
=====

Section
-------

Some text.
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "scheme", "ext": "scm", "filename": "original.scm",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "define forms — but fastedit anchoring: none yet"
        ),
        "original": '''(define (add x y)
  (+ x y))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "scss", "ext": "scss", "filename": "original.scss",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "rule sets, variables, nested rules — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''$header: #333;

.header {
  color: $header;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "solidity", "ext": "sol", "filename": "original.sol",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "contracts, functions, state variables — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''pragma solidity ^0.8.0;

contract Counter {
    uint256 public count;

    function increment() public {
        count += 1;
    }
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "sparql", "ext": "sparql", "filename": "original.sparql",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "SELECT query forms (no code symbol notion at all); fastedit "
            "anchoring: none yet"
        ),
        "original": '''SELECT ?name
WHERE {
    ?person a foaf:Person .
    ?person foaf:name ?name .
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "squirrel", "ext": "nut", "filename": "original.nut",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function and class definitions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''function add(x, y) {
    return x + y;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "starlark", "ext": "bzl", "filename": "original.bzl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "def statements, top-level assignments — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''def add(x, y):
    return x + y

LIMIT = 10
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "svelte", "ext": "svelte", "filename": "original.svelte",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "component script/markup/style sections — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''<script>
  let name = "world";
</script>

<p>Hello {name}</p>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "tablegen", "ext": "td", "filename": "original.td",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "class/def records — but fastedit anchoring: none yet"
        ),
        "original": '''class Instruction<string name> {
  string Name = name;
}

def ADD : Instruction<"add">;
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "tcl", "ext": "tcl", "filename": "original.tcl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "proc definitions — but fastedit anchoring: none yet"
        ),
        "original": '''proc add {x y} {
    return [expr {$x + $y}]
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "terraform", "ext": "tf", "filename": "original.tf",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "resource/data/provider blocks — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''resource "aws_s3_bucket" "sample" {
  bucket = "sample-bucket"
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "thrift", "ext": "thrift", "filename": "original.thrift",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "structs/services/enums — but fastedit anchoring: none yet"
        ),
        "original": '''namespace py sample

struct Point {
  1: i32 x,
  2: i32 y,
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "tsv", "ext": "tsv", "filename": "original.tsv",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "tab-delimited rows with a header line (no symbol notion at "
            "all — a pure data grid); fastedit anchoring: none yet"
        ),
        "original": "id\tname\n1\tfastedit\n2\tgolden\n",
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "twig", "ext": "twig", "filename": "original.twig",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "template tags/blocks ({% ... %}, {{ ... }}) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''{% if items %}
  {% for item in items %}
    {{ item }}
  {% endfor %}
{% endif %}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "typst", "ext": "typ", "filename": "original.typ",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "markup headings and #let bindings — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''#let add(x, y) = x + y

= Heading
Hello #add(1, 2)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "udev", "ext": "rules", "filename": "original.rules",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "match/assign rule lines (no code symbol notion at all); "
            "fastedit anchoring: none yet"
        ),
        "original": '''ACTION=="add", SUBSYSTEM=="usb", RUN+="/usr/bin/script.sh"
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "v", "ext": "v", "filename": "original.v",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "fn declarations, struct types — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''fn add(x int, y int) int {
    return x + y
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "verilog", "ext": "v", "filename": "original.v",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "module declarations — but fastedit anchoring: none yet"
        ),
        "original": '''module adder(
    input wire a,
    input wire b,
    output wire y
);
    assign y = a & b;
endmodule
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "vim", "ext": "vim", "filename": "original.vim",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "function! blocks — but fastedit anchoring: none yet"
        ),
        "original": '''function! Add(x, y) abort
    return a:x + a:y
endfunction
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "vue", "ext": "vue", "filename": "original.vue",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "component template/script/style blocks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''<template>
  <p>Hello {{ name }}</p>
</template>

<script>
export default {
  name: "Sample",
};
</script>
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "wgsl", "ext": "wgsl", "filename": "original.wgsl",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "shader entry points (fn with @ attributes) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''@fragment
fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "xcompose", "ext": "xcompose", "filename": "original.xcompose",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "compose sequence lines (no code symbol notion at all); "
            "fastedit anchoring: none yet"
        ),
        "original": '''<Multi_key> <a> <e> : "ae"
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "yuck", "ext": "yuck", "filename": "original.yuck",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "defwidget/defwindow s-expression blocks — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''(defwidget bar []
  (box :orientation "h"
    (label :text "hello")))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "zig", "ext": "zig", "filename": "original.zig",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "symbol_semantics": (
            "fn declarations, const top-levels — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''const std = @import("std");

fn add(x: i32, y: i32) i32 {
    return x + y;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    # --- parse_degraded names inside batch 2's range, retried with REAL ---
    # --- snippets: 13 of 14 parse clean (the one genuine grammar defect, --
    # --- `test`, is recorded in EXCLUDED_LANGUAGES below, never fabricated)
    {
        "language": "pony", "ext": "pony", "filename": "original.pony",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real actor Main with a create constructor parses "
            "with zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "actors/classes/primitives with methods — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''actor Main
  new create(env: Env) =>
    env.out.print("hello")
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "prisma", "ext": "prisma", "filename": "original.prisma",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real datasource + model schema parses with zero "
            "error traits — the census probe was the problem, the grammar "
            "is healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "datasource/generator/model blocks — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''datasource db {
  provider = "sqlite"
  url      = "file:dev.db"
}

model User {
  id    Int    @id
  name  String
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "proto", "ext": "proto", "filename": "original.proto",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real proto3 message file parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "message/enum/service definitions — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''syntax = "proto3";

package sample;

message Point {
  int32 x = 1;
  int32 y = 2;
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "qmljs", "ext": "qml", "filename": "original.qml",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real QML import + Rectangle object tree parses "
            "with zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "QML object declarations with property bindings — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''import QtQuick

Rectangle {
    width: 200
    height: 200
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "query", "ext": "scm", "filename": "original.scm",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real tree-sitter query pattern with captures "
            "parses with zero error traits — the census probe was the "
            "problem, the grammar is healthy. Census verdict flipped to ok "
            "in F3 (a --force re-probe re-derives the trivial-probe "
            "verdict; this fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "query patterns with captures (@name) — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''(function_declaration name: (identifier) @name)
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "smali", "ext": "smali", "filename": "original.smali",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real Dalvik class with a method body parses with "
            "zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            ".class/.super/.method directives — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''.class public LSample;
.super Ljava/lang/Object;

.method public static add(II)I
    .locals 0
    add-int v0, p0, p1
    return v0
.end method
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "smithy", "ext": "smithy", "filename": "original.smithy",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real Smithy IDL namespace + structure parses with "
            "zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "namespace + shape definitions (structure/service) — but "
            "fastedit anchoring: none yet"
        ),
        "original": '''$version: "2"

namespace sample

structure Point {
    x: Integer
}
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "ungrammar", "ext": "ungram", "filename": "original.ungram",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real ungrammar rule file parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "grammar rules (Name = nodes/tokens) — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''SourceFile =
  'fn' Name

Name = identifier
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "uxntal", "ext": "tal", "filename": "original.tal",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real uxn assembly routine with a label and BRK "
            "parses with zero error traits — the census probe was the "
            "problem, the grammar is healthy. Census verdict flipped to ok "
            "in F3 (a --force re-probe re-derives the trivial-probe "
            "verdict; this fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "labels (@Main) and opcode lines — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''( hello )
|0100 @Main
    #80 DEO
    BRK
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "vhdl", "ext": "vhd", "filename": "original.vhd",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real entity + architecture parses with zero error "
            "traits — the census probe was the problem, the grammar is "
            "healthy. Census verdict flipped to ok in F3 (a --force "
            "re-probe re-derives the trivial-probe verdict; this fixture is "
            "the durable proof)"
        ),
        "symbol_semantics": (
            "entity/architecture/package units — but fastedit anchoring: "
            "none yet"
        ),
        "original": '''entity adder is
end entity;

architecture rtl of adder is
begin
  y <= a;
end architecture;
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "wast", "ext": "wast", "filename": "original.wast",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real folded wasm module with a function parses "
            "with zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "(module ...) s-expressions with funcs — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''(module
  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "wat", "ext": "wat", "filename": "original.wat",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real wasm module with an exported function parses "
            "with zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "(module ...) s-expressions with exported funcs — but fastedit "
            "anchoring: none yet"
        ),
        "original": '''(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))
''',
        "unsupported": _anchoring_unsupported(),
    },
    {
        "language": "xml_dtd", "ext": "dtd", "filename": "original.dtd",
        "census_fixture": True,
        "census_note": (
            "census (F1) verdict was parse_degraded from the trivial probe "
            "lines; this real DTD (ELEMENT + ATTLIST declarations) parses "
            "with zero error traits — the census probe was the problem, the "
            "grammar is healthy. Census verdict flipped to ok in F3 (a "
            "--force re-probe re-derives the trivial-probe verdict; this "
            "fixture is the durable proof)"
        ),
        "symbol_semantics": (
            "<!ELEMENT>/<!ATTLIST> declarations (no function symbol "
            "notion); fastedit anchoring: none yet. Served by the "
            "hard-dependency tree_sitter_xml wheel's dtd language entry, so "
            "this fixture needs no pack"
        ),
        "original": '''<!ELEMENT note (to,from)>
<!ELEMENT to (#PCDATA)>
<!ATTLIST note version CDATA #IMPLIED>
''',
        "unsupported": _anchoring_unsupported(),
    },
    # --- G1b: the two census exclusions flip to exercised fixtures --------
    #
    # G1a bounded both grammars' quirks (cobol's error recovery wedges ->
    # PATHOLOGICAL_RECOVERY_GRAMMARS subprocess watchdog; `test`'s
    # systematic MISSING-at-EOF emission -> _GRAMMAR_ARTIFACT_FILTERS).
    # G1b adds the declarative anchoring rows (ast_utils
    # _FORMAT_SYMBOL_SPECS) and these exercised goldens, flipping the last
    # two census exclusions: pack_census.json now accounts for 175/175
    # (ok — plain or guarded) with zero exclusions.
    {
        "language": "cobol", "ext": "cob", "filename": "original.cob",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was unresolvable/timeout: the grammar "
            "resolves instantly and parses REAL cobol in 0.0s, but its "
            "error recovery LOOPS FOREVER on unrecoverable input — every "
            "trivial probe line wedges the parse. G1a bounded the grammar "
            "behind the subprocess parse watchdog "
            "(PATHOLOGICAL_RECOVERY_GRAMMARS); the v2 census probe feeds "
            "guarded grammars their real snippet ONLY (a wedge costs the "
            "whole probe) and this program parses with zero error traits "
            "under the watchdog — healthy-with-watchdog, verdict ok. G1b "
            "adds declarative anchoring (program_definition/paragraph_"
            "header) and this exercised op"
        ),
        "symbol_semantics": (
            "program_definition (named by its PROGRAM-ID, span = the whole "
            "compilation unit) and paragraph_header (named by the header "
            "text minus its terminating period; the grammar brackets ONLY "
            "the header line — statements are siblings inside procedure_"
            "division — so a paragraph anchor spans its header line alone). "
            "after=<PROGRAM-ID> therefore appends at EOF: the one insert "
            "shape whose splice cannot mangle paragraph ownership"
        ),
        "original": '''       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
       MAIN-PARA.
           DISPLAY "HELLO".
           PERFORM SETUP-PARA.
           STOP RUN.
       SETUP-PARA.
           MOVE 1 TO WS-COUNT.
           DISPLAY "SETUP".
''',
        "ops": [
            {
                "op": "insert_after", "symbol": "HELLO",
                "snippet": (
                    '       NEW-PARA.\n'
                    '           DISPLAY "ADDED".\n'
                ),
                "anchor_start_line": 1,
                "anchor_head": "       IDENTIFICATION DIVISION.",
                "anchor_end_line": 10,
                "anchor_tail": '           DISPLAY "SETUP".',
                "snippet_head": "       NEW-PARA.",
            },
        ],
        "unsupported": [
            {
                "op": "replace_symbol",
                "reason": (
                    "G1b anchoring resolves the spans (program = whole "
                    "compilation unit; paragraph = its header line only), "
                    "but a program replace is a whole-file rewrite and a "
                    "paragraph replace would splice around the header "
                    "while the body lines are siblings of it — declared "
                    "here, not exercised in the deterministic golden scope"
                ),
            },
            {
                "op": "delete_symbol",
                "reason": (
                    "same span shapes as replace_symbol: deleting a "
                    "paragraph header would promote its body into the "
                    "previous paragraph (the grammar brackets only the "
                    "header), and deleting the program deletes the file — "
                    "declared here, not exercised in the golden scope"
                ),
            },
        ],
    },
    {
        "language": "test", "ext": "test", "filename": "original.test",
        "census_fixture": True, "requires_wheel": _CENSUS_PACK,
        "census_note": (
            "census (F1) verdict was parse_degraded and F3 kept the "
            "exclusion fail-loud: every trivial probe line is an outright "
            "ERROR node for this grammar, and the grammar's own complete "
            "record ends in one systematic zero-width MISSING trait at EOF "
            "(the defect record below). G1a's artifact filter removes that "
            "systematic trait from the trait lists (tagged in "
            ".grammar_artifacts), so a same-record edit on a .test file now "
            "LANDS; G1b adds the declarative record-anchoring row "
            "(_FORMAT_SYMBOL_SPECS['test']) and this exercised op. Census "
            "verdict flips to ok — the artifact is FILTERED, not absent "
            "(the v2 probe reports the filtered count)"
        ),
        "symbol_semantics": (
            "the `test` record (header separator/name/separator + input + "
            "closing separator), named by its header's `name` line. The "
            "grammar models a whole file as ONE record whose input swallows "
            "inner separators, so the record span is the whole file and "
            "after=<record name> appends a new record at EOF"
        ),
        "original": (
            "================\nSample test\n================\n"
            'print("hi")\n================\n'
        ),
        "ops": [
            {
                "op": "insert_after", "symbol": "Sample test",
                "snippet": (
                    "================\nSecond record\n================\n"
                    'print("bye")\n================\n'
                ),
                "anchor_start_line": 1, "anchor_head": "================",
                "anchor_end_line": 5, "anchor_tail": "================",
                "snippet_head": "================",
            },
        ],
        "unsupported": [
            {
                "op": "replace_symbol",
                "reason": (
                    "G1b record anchoring resolves the span, but the "
                    "record's span IS the whole file (one record per file "
                    "in this grammar), so a replace is a whole-file rewrite "
                    "— declared here, not exercised in the golden scope"
                ),
            },
            {
                "op": "delete_symbol",
                "reason": (
                    "same span shape as replace_symbol: deleting the record "
                    "deletes the file — declared here, not exercised"
                ),
            },
        ],
        "grammar_defect_workaround": {
            "reason": (
                "pack grammar defect (tree-sitter-language-pack 0.13.0): "
                "the `test` grammar parses the header (separator / name / "
                "separator) but its `input` region greedily consumes the "
                "closing separator line, so EVERY complete test record ends "
                "in a MISSING-separator error trait at EOF, and a "
                "header-only file is an outright ERROR node. F3 verified a "
                "matrix of real-syntax shapes (recorded below with their "
                "exact parse_diagnostics errors) — none parse clean, so F3 "
                "excluded the language instead of fabricating a fixture. "
                "G1a's _GRAMMAR_ARTIFACT_FILTERS['test'] = (('MISSING', "
                "'eof'),) removes the SYSTEMATIC EOF emission from the "
                "trait lists (tagged in .grammar_artifacts) so same-record "
                "edits land while header-only breakage still rejects; G1b "
                "flipped the exclusion into this exercised fixture. The "
                "recorded attempts stay as negative evidence: they pin the "
                "defect exactly (test_golden_matrix."
                "test_grammar_defect_workaround_is_real) so an upstream "
                "grammar fix — or a regression — surfaces loudly"
            ),
            "attempts": [
                {
                    "description": (
                        "complete test record: header + input + closing "
                        "separator"
                    ),
                    "snippet": (
                        "================\nSample test\n================\n"
                        'print("hi")\n================\n'
                    ),
                    "parse_errors": [[75, 75, "MISSING"]],
                },
                {
                    "description": (
                        "complete test record, closing separator without a "
                        "trailing newline"
                    ),
                    "snippet": (
                        "================\nSample test\n================\n"
                        'print("hi")\n================'
                    ),
                    "parse_errors": [[74, 74, "MISSING"]],
                },
                {
                    "description": (
                        "header only (no input, no closing separator)"
                    ),
                    "snippet": (
                        "================\nSample test\n================\n"
                    ),
                    "parse_errors": [[0, 46, "ERROR"]],
                },
                {
                    "description": "two complete test records",
                    "snippet": (
                        "================\nOne\n================\ndo it\n"
                        "================\nTwo\n================\nagain\n"
                        "================\n"
                    ),
                    "parse_errors": [[105, 105, "MISSING"]],
                },
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# F3: census exclusions — census languages whose grammar is genuinely BROKEN
# in the installed pack. Never fabricate a parse-clean claim: the attempted
# real-syntax snippets and their EXACT parse_diagnostics errors are recorded
# here, written into the language's manifest as ``census_excluded``, and
# re-pinned by
# tests/test_golden_matrix.py::test_census_excluded_grammar_defect_is_real so
# an upstream grammar fix surfaces as a loud test failure. When that fires,
# lift the exclusion: move the language into CENSUS_LANGUAGES with a
# census_note and flip its pack_census.json verdict.
#
# EMPTY since G1b: the founding member `test` was lifted into
# CENSUS_LANGUAGES (G1a's artifact filter made same-record edits land; G1b
# added the anchoring row and the exercised op) — its F3 defect record
# survives verbatim as the fixture's ``grammar_defect_workaround``, and the
# fail-loud pin moved to
# test_grammar_defect_workaround_is_real. The list REMAINS as the
# extension point for the next genuinely-broken grammar.
# ---------------------------------------------------------------------------

EXCLUDED_LANGUAGES: list[dict] = []


# ---------------------------------------------------------------------------
# Verification + writers
# ---------------------------------------------------------------------------

def _verify_op(original_text: str, op: dict, *, gigo: bool = False) -> None:
    """Cross-check a hand-declared op against the authored original.

    Raises AssertionError when a declared line index does not match the
    fixture it claims to describe — the oracle is only as honest as these
    checks.
    """
    lines = original_text.splitlines(keepends=True)
    op_kind = op["op"]
    where = f"{op_kind} {op['symbol']}" + (" (gigo)" if gigo else "")
    if op_kind == "insert_after":
        anchor_end = op["anchor_end_line"]
        assert 1 <= anchor_end <= len(lines), f"{where}: anchor out of range"
        tail = lines[anchor_end - 1].rstrip("\n")
        assert tail == op["anchor_tail"], (
            f"{where}: anchor line {anchor_end} is {tail!r}, "
            f"expected {op['anchor_tail']!r}"
        )
        # The pipeline aligns the snippet to the anchor's FIRST line, so the
        # pre-alignment precondition compares against that line's indent.
        anchor_head = lines[op["anchor_start_line"] - 1].rstrip("\n")
        assert anchor_head.startswith(op["anchor_head"]), (
            f"{where}: anchor head line {op['anchor_start_line']} is "
            f"{anchor_head!r}, expected to start with {op['anchor_head']!r}"
        )
        snippet_base = op.get("snippet_head", "")
        target_base = anchor_head
    else:
        start, end = op["start_line"], op["end_line"]
        assert 1 <= start <= end <= len(lines), f"{where}: span out of range"
        head = lines[start - 1].rstrip("\n")
        assert head.startswith(op["span_head"]), (
            f"{where}: line {start} is {head!r}, expected to start with "
            f"{op['span_head']!r}"
        )
        tail = lines[end - 1].rstrip("\n")
        assert tail == op["span_tail"], (
            f"{where}: span end line {end} is {tail!r}, "
            f"expected {op['span_tail']!r}"
        )
        snippet_base = op.get("snippet_head", "")
        target_base = head
    if snippet_base and not op.get("prepend_signature_lines"):
        # Pre-alignment precondition: the pipeline's indent alignment is a
        # no-op on every golden op by construction. Ops that declare a
        # prepended signature are exempt — the pipeline pins the span's own
        # (indented) signature lines ahead of the body-only snippet.
        assert _indent_of(snippet_base) == _indent_of(target_base), (
            f"{where}: snippet indent {_indent_of(snippet_base)!r} does not "
            f"match target indent {_indent_of(target_base)!r}"
        )


def _expected_bytes(original_text: str, op: dict) -> bytes:
    """The oracle's expected output for one op (pure line arithmetic)."""
    lines = original_text.splitlines(keepends=True)
    op_kind = op["op"]
    if op_kind == "insert_after":
        merged = oracle_insert_after(lines, op["anchor_end_line"], op["snippet"])
    elif op_kind == "replace_symbol":
        merged = oracle_replace_span(
            lines, op["start_line"], op["end_line"], op["snippet"],
            op.get("prepend_signature_lines", 0),
        )
    elif op_kind == "delete_symbol":
        merged = oracle_delete_span(lines, op["start_line"], op["end_line"])
    else:
        raise AssertionError(f"unknown op kind {op_kind!r}")
    return "".join(merged).encode("utf-8")


def _op_manifest(op: dict) -> dict:
    return {
        "op": op["op"],
        "symbol": op["symbol"],
        "snippet": op.get("snippet"),
        "path": op.get("path") or (
            "fast_path" if op["op"] == "insert_after" else None
        ),
        "prepend_signature_lines": op.get("prepend_signature_lines", 0),
        "oracle": {
            key: op[key]
            for key in ("anchor_start_line", "anchor_end_line", "start_line", "end_line")
            if key in op
        },
    }


def _write_lang(entry: dict) -> None:
    lang = entry["language"]
    lang_dir = GOLDEN_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    ext = entry["ext"]
    filename = entry["filename"]
    original_text = entry["original"]

    manifest: dict = {
        "language": lang,
        "ext": ext,
        "filename": filename,
        "symbol_semantics": entry["symbol_semantics"],
        "ops": [],
    }
    if entry.get("requires_wheel"):
        manifest["requires_wheel"] = entry["requires_wheel"]
    if entry.get("census_note"):
        manifest["census_note"] = entry["census_note"]
    if entry.get("census_fixture"):
        manifest["census_fixture"] = True
    if entry.get("grammar_defect_workaround"):
        # G1b: a lifted exclusion keeps its F3 defect record verbatim —
        # negative evidence, re-pinned fail-loud by the matrix runner.
        manifest["grammar_defect_workaround"] = entry["grammar_defect_workaround"]

    (lang_dir / filename).write_bytes(original_text.encode("utf-8"))

    for op in entry.get("ops") or []:
        _verify_op(original_text, op)
        expected_name = f"expected_{op['op']}_{_sanitize(op['symbol'])}.{ext}"
        (lang_dir / expected_name).write_bytes(_expected_bytes(original_text, op))
        manifest["ops"].append({
            **_op_manifest(op),
            "expected": expected_name,
        })
    if entry.get("unsupported"):
        manifest["unsupported"] = entry["unsupported"]

    if entry.get("gigo"):
        gigo = entry["gigo"]
        gigo_original = gigo["original"]
        (lang_dir / f"original_gigo.{ext}").write_bytes(
            gigo_original.encode("utf-8"),
        )
        _verify_op(gigo_original, gigo["op"], gigo=True)
        expected_name = f"expected_gigo_{_sanitize(gigo['op']['symbol'])}.{ext}"
        (lang_dir / expected_name).write_bytes(
            _expected_bytes(gigo_original, gigo["op"]),
        )
        manifest["gigo"] = {
            "original": f"original_gigo.{ext}",
            "defect": gigo["defect"],
            "defect_lines": gigo["defect_lines"],
            "op": {**_op_manifest(gigo["op"]), "expected": expected_name},
        }

    (lang_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_excluded(entry: dict) -> None:
    """Write a census-EXCLUDED language's fail-loud documentation.

    The language's grammar is genuinely broken (no real-syntax snippet
    parses clean — F3 verified a shape matrix). Nothing here claims a clean
    parse: the manifest records the exact ``parse_diagnostics`` errors per
    attempted shape, and the matrix runner re-pins every recorded failure so
    an upstream grammar fix fails loudly instead of the exclusion silently
    outliving the defect.
    """
    excluded = entry["census_excluded"]
    attempts = excluded["attempts"]
    assert excluded.get("reason"), f"{entry['language']}: exclusion needs a reason"
    assert attempts, f"{entry['language']}: exclusion needs recorded attempts"
    for attempt in attempts:
        errors = attempt["parse_errors"]
        assert errors, f"{entry['language']}: attempt without recorded errors"
        for start, end, kind in errors:
            assert isinstance(start, int) and isinstance(end, int), errors
            assert start <= end and kind in ("ERROR", "MISSING"), errors
    assert entry["original"] == attempts[0]["snippet"], (
        f"{entry['language']}: the committed original must be the primary "
        f"attempt's snippet"
    )
    lang_dir = GOLDEN_DIR / entry["language"]
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / entry["filename"]).write_bytes(entry["original"].encode("utf-8"))
    manifest = {
        "language": entry["language"],
        "ext": entry["ext"],
        "filename": entry["filename"],
        "symbol_semantics": entry["symbol_semantics"],
        "requires_wheel": entry["requires_wheel"],
        "census_excluded": excluded,
    }
    (lang_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{entry['language']}: census-excluded (grammar defect) documented")


def main() -> None:
    for entry in [*LANGUAGES, *CENSUS_LANGUAGES]:
        _write_lang(entry)
        ops = len(entry.get("ops") or []) + (1 if entry.get("gigo") else 0)
        print(f"{entry['language']}: {ops} golden op(s) written")
    for entry in EXCLUDED_LANGUAGES:
        _write_excluded(entry)


if __name__ == "__main__":
    main()
