"""B3 symbol-anchoring regression tests.

Each test pins one defect the golden matrix exposed, at unit level:

1. the declarative per-format symbol tables (_FORMAT_SYMBOL_SPECS and the
   bash row) give every new format an AST map where none existed;
2. C/C++ functions surface in the in-memory map (their name hides inside a
   ``function_declarator``, not a direct identifier child);
3. the 1-indexed end-line rule: a node whose last byte is a newline
   (markdown sections, TOML tables, YAML block values) must not claim the
   NEXT block's first line — a delete/insert-after spliced from such a span
   used to eat the following block;
4. _resolve_symbol resolves literal dotted names (CSS selectors, TOML
   dotted keys) before applying the Class.method qualification grammar;
5. the prepended signature keeps the target's indentation — without that
   the direct-swap alignment shifted the whole replacement by the missing
   columns;
6. brace-style signatures (bash ``run_test() {``, C ``int beta(int y) {``)
   are recognized, so a snippet that restates one is not signature-prepended
   a second time;
7. the snippet parser consults fastedit's in-memory resolver first, so
   direct-swap works for formats the tldr daemon has never heard of;
8. an explicit language hint drives get_ast_map_from_source for the
   extension-unwired all-grammars languages;
9. issue #5: jest it()/test()/describe() blocks, class constructors, and
   the file's import block are addressable symbols, named exactly the way
   the tldr structure pass already prints them — so every slug read (or an
   error message) suggests is one --replace/--after can resolve.
"""

from __future__ import annotations

import pytest

from fastedit.inference.ast_utils import (
    _call_slug,
    _resolve_symbol,
    get_ast_map_from_source,
)
from fastedit.inference.chunked_merge import (
    _extract_signature_via_ast,
    _snippet_has_target_signature,
)
from fastedit.inference.snippet_analysis import (
    _find_import_region,
    _snippet_parses_as_member,
    _try_tldr_snippet_parse,
)

# ---------------------------------------------------------------------------
# 1. the declarative per-format symbol maps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "language", "expected"),
    [
        (
            "page.html", "html",
            [("intro", "element", 7, 10), ("details", "element", 11, 14)],
        ),
        (
            "data.xml", "xml",
            [
                ("config", "element", 2, 8),
                ("server", "element", 3, 3),
                ("cache", "element", 4, 4),
                ("limits", "element", 5, 7),
                ("timeout", "element", 6, 6),
            ],
        ),
        (
            "doc.md", "markdown",
            # Step D2 stress: markdown section spans are trimmed to their
            # last CONTENT line — the blank separator before the next
            # heading is layout between symbols, and a span that swallowed
            # it made every replace/delete splice eat the separator byte.
            [("Guide", "section", 1, 11), ("Setup", "section", 5, 7),
             ("Usage", "section", 9, 11)],
        ),
        (
            "data.json", "json",
            [
                ("name", "key", 2, 2),
                ("version", "key", 3, 3),
                ("config", "key", 4, 7),
                ("debug", "key", 5, 5),
                ("level", "key", 6, 6),
                ("tags", "key", 8, 8),
            ],
        ),
        (
            "cfg.yaml", "yaml",
            [
                ("name", "key", 1, 1),
                ("version", "key", 2, 2),
                ("database", "key", 3, 5),
                ("host", "key", 4, 4),
                ("port", "key", 5, 5),
            ],
        ),
        (
            "style.css", "css",
            [("body", "rule", 1, 4), (".header", "rule", 6, 8),
             (".footer", "rule", 10, 12)],
        ),
        (
            "app.toml", "toml",
            [("package", "table", 3, 6), ("dependencies", "table", 7, 9),
             ("tool.pytest", "table", 10, 11)],
        ),
        (
            "q.sql", "sql",
            [("users", "table", 1, 4), ("orders", "table", 6, 9),
             ("idx_orders_user", "index", 11, 11)],
        ),
        (
            "Container.dockerfile", "dockerfile",
            [("builder", "stage", 1, 1), ("runtime", "stage", 5, 5)],
        ),
        (
            "run.sh", "bash",
            [("run_build", "function", 4, 6), ("run_test", "function", 8, 10)],
        ),
    ],
)
def test_format_symbol_map(filename, language, expected):
    """Every new format gets a symbol map with the declared semantics."""
    fixtures = {
        "html": FIXTURE_HTML,
        "xml": FIXTURE_XML,
        "markdown": FIXTURE_MARKDOWN,
        "json": FIXTURE_JSON,
        "yaml": FIXTURE_YAML,
        "css": FIXTURE_CSS,
        "toml": FIXTURE_TOML,
        "sql": FIXTURE_SQL,
        "dockerfile": FIXTURE_DOCKERFILE,
        "bash": FIXTURE_BASH,
    }
    nodes = get_ast_map_from_source(fixtures[language], filename)
    got = [(n.name, n.kind, n.line_start, n.line_end) for n in nodes]
    assert got == expected, f"{language} symbol map drifted: {got}"


# ---------------------------------------------------------------------------
# 2. C/C++ functions surface (declarator descent)
# ---------------------------------------------------------------------------

def test_c_cpp_functions_survive_the_map():
    c_src = (
        "#include <stdio.h>\n\nint alpha(int x) {\n    return x + 1;\n}\n"
    )
    nodes = get_ast_map_from_source(c_src, "m.c")
    assert [(n.name, n.kind) for n in nodes] == [("alpha", "function")]


# ---------------------------------------------------------------------------
# 3. the end-line rule for newline-terminated nodes
# ---------------------------------------------------------------------------

def test_section_span_does_not_swallow_the_next_heading():
    nodes = get_ast_map_from_source(FIXTURE_MARKDOWN, "doc.md")
    setup = next(n for n in nodes if n.name == "Setup")
    usage = next(n for n in nodes if n.name == "Usage")
    # Step D2 stress: the section span ends at its last CONTENT line — the
    # trailing blank separator is layout between symbols (a span that
    # swallowed it made replace/delete eat the separator byte).
    assert setup.line_end == 7, (
        "a markdown section claiming line 8 (the blank separator) makes "
        "replace/delete eat the layout between sections"
    )
    assert usage.line_end == 11
    assert setup.line_end < usage.line_start


def test_toml_table_span_does_not_swallow_the_next_table():
    nodes = get_ast_map_from_source(FIXTURE_TOML, "app.toml")
    package = next(n for n in nodes if n.name == "package")
    assert package.line_end == 6
    assert package.line_end < 7  # line 7 is [dependencies]'s header


# ---------------------------------------------------------------------------
# 4. literal dotted names resolve before the qualification grammar
# ---------------------------------------------------------------------------

def test_css_selector_resolves_literally():
    nodes = get_ast_map_from_source(FIXTURE_CSS, "style.css")
    node = _resolve_symbol(".header", nodes)
    assert node is not None and node.kind == "rule"


def test_toml_dotted_key_resolves_literally():
    nodes = get_ast_map_from_source(FIXTURE_TOML, "app.toml")
    node = _resolve_symbol("tool.pytest", nodes)
    assert node is not None and node.line_start == 10


def test_dotted_qualification_still_resolves_members():
    nodes = get_ast_map_from_source(FIXTURE_PYTHON, "original.py")
    node = _resolve_symbol("Store.save", nodes)
    assert node is not None and node.parent == "Store"
    # ... and the plain bare name still works
    assert _resolve_symbol("alpha", nodes) is not None


def test_duplicate_literal_names_still_refuse():
    src = "def dup():\n    return 1\n\n\ndef dup():\n    return 2\n"
    nodes = get_ast_map_from_source(src, "original.py")
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_symbol("dup", nodes)


# ---------------------------------------------------------------------------
# 5. the prepended signature keeps its indentation
# ---------------------------------------------------------------------------

CS_SRC = """public class Sample
{
    public static int Beta(int y)
    {
        return y * 2;
    }
}
"""


def test_prepended_signature_keeps_target_indent():
    sig = _extract_signature_via_ast(
        CS_SRC, "c_sharp", 3, 5, CS_SRC.splitlines(keepends=True)[2],
    )
    # Allman braces: the signature is the 2-line header including the opener,
    # and its FIRST line carries the member's 4-space indentation.
    assert sig == "    public static int Beta(int y)\n    {\n"


# ---------------------------------------------------------------------------
# 6. brace-style signatures are recognized (no double prepend)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("run_test() {", "run_test"),          # bash
        ("int beta(int y) {", "beta"),          # c
        ("int beta(int y)", "beta"),            # c Allman-style header line
    ],
)
def test_brace_style_signature_is_recognized(line, name):
    assert _snippet_has_target_signature(line + "\n", name)


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("    save(x);", "save"),               # a call, not a definition
        ("    total = total + x;", "total"),
        ("# ... existing code ...", "existing"),
    ],
)
def test_calls_are_not_signatures(line, name):
    assert not _snippet_has_target_signature(line + "\n", name)


def test_bash_full_redefinition_skips_the_prepend():
    """A bash snippet restating its signature must reach direct-swap with a
    bracket-balanced splice (the doubling made the balance 1 vs 0 and the
    op fell to the model path)."""
    from fastedit.inference.chunked_merge import chunked_merge

    src = FIXTURE_BASH

    def _no_model(*_a, **_kw):
        raise AssertionError("model path ran")

    result = chunked_merge(
        original_code=src,
        snippet='run_test() {\n  echo "running tests"\n}\n',
        file_path="tmp/run.sh",
        merge_fn=_no_model,
        language="bash",
        replace="run_test",
    )
    assert result.model_tokens == 0
    assert result.merged_code.count("run_test() {") == 1


# ---------------------------------------------------------------------------
# 7. the snippet parser consults the in-memory resolver first
# ---------------------------------------------------------------------------

def test_snippet_parse_serves_formats_the_daemon_does_not_know():
    snippet = '<section id="alpha">\n  <h1>Alpha</h1>\n</section>\n'
    assert _try_tldr_snippet_parse(snippet, ".html") == ["alpha"]


# ---------------------------------------------------------------------------
# 8. explicit language hints drive extension-unwired languages
# ---------------------------------------------------------------------------

def test_language_hint_serves_extension_unwired_languages():
    lua = (
        "local M = {}\n\nfunction M.alpha(x)\n  return x + 1\nend\n"
    )
    nodes = get_ast_map_from_source(lua, "m.lua", "lua")
    assert [(n.name, n.kind) for n in nodes] == [("M.alpha", "function")]


def test_explicit_hint_wins_over_suffix_detection():
    """A caller that names the language is authoritative even when the
    suffix also resolves (yaml content in a .txt file, say)."""
    nodes = get_ast_map_from_source(FIXTURE_YAML, "cfg.txt", "yaml")
    assert [n.name for n in nodes][:2] == ["name", "version"]


# ---------------------------------------------------------------------------
# 9. issue #5: test blocks, constructors, and the imports block are symbols
# ---------------------------------------------------------------------------

FIXTURE_JEST_TS = (
    'import { describe, it, expect } from "vitest";\n'
    'import { createServer } from "./server";\n'
    "\n"
    "const PORT = 3111;\n"
    "\n"
    'describe("constructor", () => {\n'
    '  it("should create server with correct", () => {\n'
    "    const server = createServer(PORT);\n"
    "    expect(server.port).toBe(PORT);\n"
    "  });\n"
    "\n"
    '  it("renders correctly", () => {\n'
    '    expect(true).toBe(true);\n'
    "  });\n"
    "});\n"
)

FIXTURE_TS_CLASS = (
    "export class Server {\n"
    "  readonly port: number;\n"
    "\n"
    "  constructor(port: number) {\n"
    "    this.port = port;\n"
    "  }\n"
    "\n"
    "  start(): void {\n"
    "    console.log(this.port);\n"
    "  }\n"
    "}\n"
)

FIXTURE_IMPORTS_PY = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def alpha():\n"
    "    import json  # deferred — layout, not part of the top block\n"
    "    return os\n"
)


def test_jest_blocks_are_addressable_symbols():
    """it()/test()/describe() calls are symbols named <callee>:<slug>."""
    nodes = get_ast_map_from_source(FIXTURE_JEST_TS, "server.test.ts")
    by_name = {n.name: n for n in nodes}
    assert "describe:constructor" in by_name
    assert "it:should-create-server-with-correct" in by_name
    assert "it:renders-correctly" in by_name
    it_node = by_name["it:should-create-server-with-correct"]
    assert it_node.kind == "call"
    # nesting: the it() block sits inside its describe, so the qualified
    # form disambiguates same-named tests in different suites
    assert it_node.parent == "describe:constructor"
    describe_node = by_name["describe:constructor"]
    assert describe_node.line_start < it_node.line_start
    assert describe_node.line_end >= it_node.line_end


def test_jest_blocks_resolve_as_printed():
    """Every slug the map lists resolves — bare and qualified."""
    nodes = get_ast_map_from_source(FIXTURE_JEST_TS, "server.test.ts")
    node = _resolve_symbol("it:renders-correctly", nodes)
    assert node is not None and node.kind == "call"
    qualified = _resolve_symbol("describe:constructor.it:renders-correctly", nodes)
    assert qualified is node


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("renders correctly", "renders-correctly"),
        ("Should Create Server", "should-create-server"),
        ("should   create   server", "should-create-server"),
        ("renders <Header /> correctly!", "renders-header-correctly"),
        ("renders_user_profile", "renders-user-profile"),
        ("UPPER CASE NAME", "upper-case-name"),
        ("don't fail", "don-t-fail"),
        ("a-b-c", "a-b-c"),
        ("hello, world: the sequel", "hello-world-the-sequel"),
        ("tab\tseparated", "tab-separated"),
        ("  trimmed  ", "trimmed"),
        ("multiple  spaces & symbols!!", "multiple-spaces-symbols"),
        ("héllo wörld", "h-llo-w-rld"),
        ("123 numbers 456", "123-numbers-456"),
        ("CamelCaseWord another", "camelcaseword-another"),
    ],
)
def test_call_slug_matches_the_tldr_structure_shape(name, slug):
    """The slug shape is the one `tldr structure` prints for jest calls —
    read, the error suggestions, and resolve must agree on it slug for slug."""
    assert _call_slug(name) == slug


def test_duplicate_test_names_dedupe_like_tldr():
    """Same-named tests get the daemon's #N suffix, in file order, so the
    second block stays addressable instead of refusing as ambiguous."""
    src = (
        'describe("one", () => {\n'
        '  it("same", () => { expect(1).toBe(1); });\n'
        "});\n"
        'describe("two", () => {\n'
        '  it("same", () => { expect(2).toBe(2); });\n'
        "});\n"
    )
    names = [n.name for n in get_ast_map_from_source(src, "dup.test.ts")]
    assert names == ["describe:one", "it:same", "describe:two", "it:same#2"]


def test_nameless_test_call_falls_back_to_the_bare_callee():
    """`it(() => ...)` has no string argument — the daemon names it after
    the callee alone; resolve must agree with that printed form."""
    nodes = get_ast_map_from_source('it(() => {\n  expect(1).toBe(1);\n});\n', "a.test.ts")
    assert [(n.name, n.kind) for n in nodes] == [("it", "call")]


def test_imports_pseudo_symbol_spans_the_leading_import_block():
    nodes = get_ast_map_from_source(FIXTURE_IMPORTS_PY, "app.py")
    blocks = [n for n in nodes if n.kind == "imports"]
    assert len(blocks) == 1
    block = blocks[0]
    assert block.name == "imports"
    assert (block.line_start, block.line_end) == (1, 3)
    # the deferred import inside alpha() is layout, not part of the block
    assert block.line_end < 7
    # ... and it resolves like any symbol, so --after imports anchors on it
    assert _resolve_symbol("imports", nodes) is block


def test_imports_pseudo_symbol_multiline_python_imports():
    src = (
        "from os import (\n"
        "    path,\n"
        ")\n"
        "import sys\n"
        "\n"
        "\n"
        "def f():\n"
        "    return path\n"
    )
    nodes = get_ast_map_from_source(src, "app.py")
    block = next(n for n in nodes if n.kind == "imports")
    assert (block.line_start, block.line_end) == (1, 4)


@pytest.mark.parametrize(
    ("ext", "language"),
    [("ts", "typescript"), ("tsx", "tsx"), ("js", "javascript")],
)
def test_imports_pseudo_symbol_per_js_family_language(ext, language):
    src = (
        'import a from "a";\n'
        'import b from "b";\n'
        "\n"
        "class C {}\n"
    )
    nodes = get_ast_map_from_source(src, f"app.{ext}", language)
    block = next(n for n in nodes if n.kind == "imports")
    assert (block.line_start, block.line_end) == (1, 2)


def test_no_imports_pseudo_symbol_without_imports():
    nodes = get_ast_map_from_source(FIXTURE_PYTHON, "original.py")
    assert all(n.kind != "imports" for n in nodes)


def test_import_region_ignores_the_pseudo_symbol():
    """_find_import_region measures 'imports before the first definition';
    counting the imports pseudo-symbol itself as the first definition would
    filter the whole region away."""
    nodes = get_ast_map_from_source(FIXTURE_IMPORTS_PY, "app.py")
    assert _find_import_region(FIXTURE_IMPORTS_PY, "python", nodes) == (1, 3)


CONSTRUCTOR_SNIPPET = "constructor(port: number) {\n    this.port = port * 2;\n  }\n"


def test_constructor_snippet_surfaces_its_own_name():
    """A bare `constructor(...) {...}` has no compilation-unit parse; the
    snippet-name parser must still report the member it defines."""
    assert _try_tldr_snippet_parse(CONSTRUCTOR_SNIPPET, ".ts") == ["constructor"]


def test_constructor_snippet_parses_as_member():
    assert _snippet_parses_as_member(CONSTRUCTOR_SNIPPET, "typescript")
    assert not _snippet_parses_as_member("constructor(port: number) {", "typescript")
    assert not _snippet_parses_as_member("def broken(:", "typescript")


def test_snippet_signature_check_accepts_the_constructor_line():
    assert _snippet_has_target_signature("constructor(port: number) {", "constructor")


def test_constructor_replace_reaches_the_deterministic_swap():
    """`--replace constructor` with a constructor(...) snippet must land
    deterministically (0 model tokens), never refuse as 'not valid
    typescript' nor fall to the model backend."""
    from fastedit.inference.chunked_merge import chunked_merge

    def _no_model(*_a, **_kw):
        raise AssertionError("model path ran")

    snippet = "  constructor(port: number) {\n    this.port = port * 2;\n  }\n"
    result = chunked_merge(
        original_code=FIXTURE_TS_CLASS,
        snippet=snippet,
        file_path="tmp/server.ts",
        merge_fn=_no_model,
        language="typescript",
        replace="constructor",
    )
    assert result.model_tokens == 0
    assert result.parse_valid
    assert "this.port = port * 2;" in result.merged_code
    assert "console.log(this.port);" in result.merged_code


def test_test_block_replace_reaches_the_deterministic_swap():
    """`--replace it:<slug>` with a restated it()-line snippet lands
    byte-exact on the block, deterministically."""
    from fastedit.inference.chunked_merge import chunked_merge

    def _no_model(*_a, **_kw):
        raise AssertionError("model path ran")

    original = (
        'import { describe, it, expect } from "vitest";\n'
        "\n"
        'describe("badge", () => {\n'
        '  it("renders correctly", () => {\n'
        '    const el = renderBadge("hello");\n'
        '    expect(el.textContent).toBe("hello");\n'
        "  });\n"
        "\n"
        '  it("handles empty label", () => {\n'
        '    expect(renderBadge("")).toBeNull();\n'
        "  });\n"
        "});\n"
    )
    snippet = (
        '  it("renders correctly", () => {\n'
        '    const el = renderBadge("hi");\n'
        '    expect(el.textContent).toBe("hi");\n'
        "  });\n"
    )
    result = chunked_merge(
        original_code=original,
        snippet=snippet,
        file_path="tmp/badge.test.ts",
        merge_fn=_no_model,
        language="typescript",
        replace="it:renders-correctly",
    )
    assert result.model_tokens == 0
    expected = original.replace('renderBadge("hello")', 'renderBadge("hi")').replace(
        'toBe("hello")', 'toBe("hi")'
    )
    assert result.merged_code == expected


def test_after_imports_anchors_on_the_last_import_line():
    """after= over the `imports` pseudo-symbol splices after the LAST import
    with the standard after= blank-line separators."""
    from fastedit.inference.chunked_merge import chunked_merge

    def _no_model(*_a, **_kw):
        raise AssertionError("model path ran")

    result = chunked_merge(
        original_code=(
            "import os\n"
            "import sys\n"
            "\n"
            "\n"
            "def main():\n"
            "    print(os.name)\n"
        ),
        snippet="import json\n",
        file_path="tmp/app.py",
        merge_fn=_no_model,
        language="python",
        after="imports",
    )
    assert result.model_tokens == 0
    assert result.merged_code == (
        "import os\n"
        "import sys\n"
        "\n"
        "import json\n"
        "\n"
        "\n"
        "def main():\n"
        "    print(os.name)\n"
    )


# ---------------------------------------------------------------------------
# fixtures (kept tiny; the golden matrix carries the full set)
# ---------------------------------------------------------------------------

FIXTURE_PYTHON = (
    '"""Sample module."""\nCONSTANT = 42\n\n\ndef alpha(x):\n    return x + 1\n'
    "\n\nclass Store:\n    def save(self):\n        return \"saved\"\n\n\n"
    "def beta(y):\n    return y * 2\n"
)
FIXTURE_HTML = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <title>Sample</title>\n"
    "</head>\n<body>\n  <section id=\"intro\">\n    <h1>Intro</h1>\n"
    "    <p>Welcome paragraph.</p>\n  </section>\n  <section id=\"details\">\n"
    "    <h1>Details</h1>\n    <p>Detail paragraph.</p>\n  </section>\n"
    "</body>\n</html>\n"
)
FIXTURE_XML = (
    '<?xml version="1.0"?>\n<config>\n  <server>host1</server>\n'
    '  <cache enabled="true">redis</cache>\n  <limits>\n    <timeout>30</timeout>\n'
    "  </limits>\n</config>\n"
)
FIXTURE_MARKDOWN = (
    "# Guide\n\nIntro paragraph.\n\n## Setup\n\nInstall steps.\n\n"
    "## Usage\n\nUsage notes.\n"
)
FIXTURE_JSON = (
    '{\n  "name": "fastedit",\n  "version": "0.5.0",\n  "config": {\n'
    '    "debug": true,\n    "level": 3\n  },\n  "tags": ["ast", "edit"]\n}\n'
)
FIXTURE_YAML = (
    "name: fastedit\nversion: 0.5.0\ndatabase:\n  host: localhost\n  port: 5432\n"
)
FIXTURE_CSS = (
    "body {\n  color: #333;\n  margin: 0 auto;\n}\n\n.header {\n"
    "  font-weight: bold;\n}\n\n.footer {\n  padding: 4px;\n}\n"
)
FIXTURE_TOML = (
    'title = "fastedit"\n\n[package]\nname = "fastedit"\nversion = "0.5.0"\n\n'
    '[dependencies]\nserde = "1"\n\n[tool.pytest]\nminversion = "7.0"\n'
)
FIXTURE_SQL = (
    "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  name TEXT NOT NULL\n);\n\n"
    "CREATE TABLE orders (\n  id INT PRIMARY KEY,\n  user_id INT\n);\n\n"
    "CREATE INDEX idx_orders_user ON orders (user_id);\n"
)
FIXTURE_DOCKERFILE = (
    "FROM python:3.12 AS builder\n\nRUN pip install requests\n\n"
    "FROM alpine AS runtime\n\nCOPY --from=builder /app /app\n\nCMD [\"python\"]\n"
)
FIXTURE_BASH = (
    '#!/usr/bin/env bash\nVERSION="1.2.3"\n\nrun_build() {\n'
    '  echo "building ${VERSION}"\n}\n\nrun_test() {\n  echo "testing"\n}\n'
)
