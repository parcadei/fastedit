"""Grammar resolver contract — B1 dynamic resolver + B2 broad grammar set.

fastedit must work with ANY tree-sitter grammar present in the
environment, not just the grammars named in its dependency list (req. 4 of
the implementation plan: "if a file format is supported by tree-sitter,
fastedit must support it — hundreds of formats"). These tests pin:

B1 (dynamic resolver):
* every ALREADY-INSTALLED grammar resolves and parses (parametrized);
* a deliberately broken snippet yields non-empty diagnostics (A2's
  ``parse_diagnostics`` re-proven on every installed grammar);
* alternate spellings (``csharp`` ↔ ``c_sharp``) resolve to the SAME
  parser object;
* unknown languages raise the typed :class:`GrammarUnavailableError`
  carrying the ``pip install`` hint — never a silent "no validation";
* the parser/language caches return identical objects on re-resolution;
* a grammar wheel that is installed but absent from the declarative
  table still resolves (the generic ``tree_sitter_<lang>`` +
  ``language()`` convention) — new grammars need zero code changes;
* the optional ``tree_sitter_languages``-style pack is a FALLBACK
  resolver, never a requirement.

B2 (broad grammar set — Step B2 of the implementation plan):
* the core-format grammars (html, xml, markdown, json, yaml, css, bash,
  toml, sql, dockerfile) are HARD dependencies: they resolve on a default
  install and their extensions are wired into ``EXTENSION_TO_LANGUAGE``;
* the all-grammars extra adds scala/lua/perl/julia/zig/svelte/graphql/
  hcl/make/nix + ``tree-sitter-language-pack`` (~170 bundled languages);
  their tests skip cleanly when the wheel is absent;
* extension→language round-trips for every new pair (detect_language
  never hands back a name the resolver cannot honor);
* markdown's documented leniency (plan §3 risk 1) is pinned: structural
  malformation parses clean, so md validity leans on the D1/D3 text
  traits;
* one real deterministic ``chunked_merge`` edit per major new format
  (html, json, yaml, markdown) proves locate→edit→validate end-to-end,
  with a negative control proving the relative parse gate is ARMED with
  the new grammar (markdown excluded there — it cannot be parse-broken,
  see the leniency test);
* the language pack, when installed, adds its languages through the B1
  fallback resolver (counted and reported in the test output).
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

from fastedit.data_gen import ast_analyzer
from fastedit.data_gen.ast_analyzer import (
    EXTENSION_TO_LANGUAGE,
    LANGUAGE_ALIASES,
    GrammarUnavailableError,
    detect_language,
    get_language,
    get_parser,
    parse_diagnostics,
    validate_parse,
)

# Core grammars — HARD dependencies of fastedit since B2 (pyproject
# [project.dependencies]). They must resolve in every environment.
CORE_LANGUAGES = (
    "python", "javascript", "typescript", "tsx", "rust", "go", "java",
    "c", "cpp", "ruby", "swift", "kotlin", "c_sharp", "php", "elixir",
    # B2 core formats:
    "html", "xml", "xml_dtd", "markdown", "markdown_inline", "json",
    "yaml", "css", "bash", "toml", "sql", "dockerfile",
)

# Verified B2 wheels shipped in the optional ``all-grammars`` extra (NOT
# hard dependencies). Tests skip when the wheel is absent so the default
# tier stays hermetic on a bare ``uv sync``; in this repo's dev venv the
# extra is synced, so they run.
EXTRA_LANGUAGES = (
    "scala", "lua", "perl", "julia", "zig", "svelte", "graphql", "hcl",
    "make", "nix",
)


def _wheel_present(canonical: str) -> bool:
    module = ast_analyzer._grammar_module_candidates(canonical)[0]
    return importlib.util.find_spec(module) is not None


_EXTRA_SKIP_MARKS = {
    lang: pytest.mark.skipif(
        not _wheel_present(lang),
        reason=(
            f"tree-sitter-{lang.replace('_', '-')} is an all-grammars extra "
            f"dependency, not a hard dependency — resolving it is the "
            f"optional extra's job, never the default install's"
        ),
    )
    for lang in EXTRA_LANGUAGES
}

# Minimal parse-VALID and deliberately BROKEN snippets per language. The
# broken snippets were probe-verified to produce tree-sitter error traits
# on the installed grammar versions. Lenient grammars (markdown) document
# their only trippable breakage inline.
MINIMAL_SNIPPETS: dict[str, tuple[str, str]] = {
    "python": ("def f():\n    return 1\n", "def f(:\n    return 1\n"),
    "javascript": ("function f() { return 1; }\n", "function f( { return 1; }\n"),
    "typescript": (
        "function f(x: number): number { return x; }\n",
        "function f(x: number): number { return x;\n",
    ),
    "tsx": (
        'const el = <div className="a">hi</div>;\n',
        'const el = <div className="a">hi;\n',
    ),
    "rust": ("fn f() -> i32 { 1 }\n", "fn f( -> i32 { 1 }\n"),
    "go": (
        "package main\nfunc f() int { return 1 }\n",
        "package main\nfunc f( int { return 1 }\n",
    ),
    "java": (
        "class A { int f() { return 1; } }\n",
        "class A { int f( { return 1; } }\n",
    ),
    "c": ("int f(void) { return 1; }\n", "int f( { return 1; }\n"),
    "cpp": ("int f() { return 1; }\n", "class A { int f( { return 1; } };\n"),
    "ruby": ("def f\n  1\nend\n", "def f\n  1\n"),
    "swift": ("func f() -> Int { return 1 }\n", "func f( -> Int { return 1 }\n"),
    "kotlin": ("fun f(): Int { return 1 }\n", "fun f( { return 1 }\n"),
    "c_sharp": (
        "class A { int F() { return 1; } }\n",
        "class A { int F( { return 1; } }\n",
    ),
    "php": (
        "<?php\nfunction f() { return 1; }\n",
        "<?php\nfunction f( { return 1; }\n",
    ),
    "elixir": (
        "defmodule M do\n  def f, do: 1\nend\n",
        "defmodule M do\n  def f(\n",
    ),
    # --- B2 core formats (all probe-verified) ---
    "html": (
        (
            "<!DOCTYPE html>\n<html><head><title>T</title></head>"
            "<body><p>hi</p></body></html>\n"
        ),
        "<b><i>x</b></i>\n",
    ),    "xml": (
        '<?xml version="1.0"?>\n<root><item id="1">v</item></root>\n',
        "<root><item></root>\n",
    ),
    "xml_dtd": (
        "<!ELEMENT note (to, from)>\n<!ATTLIST note id CDATA #IMPLIED>\n",
        "<!ELEMENT note (to\n",
    ),
    # Markdown's grammar is structurally error-TOLERANT: every malformed
    # construct we probed (unclosed fences, unclosed emphasis, stray
    # brackets, broken tables) parses clean. The ONLY breakage that trips
    # it is a control character (NUL) — used here so the parametrized
    # "broken → diagnostics" contract holds uniformly. The structural
    # leniency itself is pinned by
    # test_markdown_grammar_is_structurally_lenient below.
    "markdown": (
        "# Title\n\nSome *emphasis* and `code`.\n\n- item one\n- item two\n",
        "a\x00b\n",
    ),
    # Same NUL-only leniency for the inline sub-grammar (same wheel,
    # `inline_language` entry).
    "markdown_inline": (
        "*emphasis* and `code` ok\n",
        "a\x00b\n",
    ),
    "json": (
        '{"a": [1, 2, {"b": null}], "c": true}\n',
        '{"a": [1, 2, ], "b": }\n',
    ),
    "yaml": (
        "key: value\nlist:\n  - one\n  - two\nnested:\n  a: 1\n",
        "key: [unclosed\n  - bad\n",
    ),
    "css": (
        (
            "body { color: red; margin: 0 auto; }\n"
            ".cls > #id, a:hover { padding: 4px; }\n"
        ),
        "body {{ color: red; }\n",
    ),
    "bash": (
        '#!/usr/bin/env bash\nfor f in *.txt; do echo "$f"; done\n',
        "if [ -f x ]; then echo hi\n",
    ),
    "toml": (
        '[section]\nkey = "value"\nnum = 1\narr = [1, 2]\n',
        "[section\nkey = \n",
    ),
    "dockerfile": (
        'FROM python:3.12\nRUN pip install -r req.txt\nCMD ["python"]\n',
        "FROM\nRUN (\n",
    ),
    "sql": (
        "SELECT a, b FROM t WHERE a > 1 ORDER BY b;\n",
        "SELECT FROM WHERE ORDER;\n",
    ),
    # --- B2 all-grammars extra languages ---
    "scala": (
        "object A { def f(x: Int): Int = x + 1 }\n",
        "object A { def f(x: Int): = }\n",
    ),
    "lua": (
        "local function f(x) return x + 1 end\n",
        "function f( return 1 end\n",
    ),
    "perl": (
        "use strict;\nmy $x = 1;\nsub f { return $x + 1; }\n",
        "sub f { return $x + ;\n",
    ),
    "julia": (
        "function f(x)\n    return x + 1\nend\n",
        "function f(\n    return\n",
    ),
    "zig": (
        "pub fn f(x: i32) i32 { return x + 1; }\n",
        "pub fn f(x: i32) { return ;\n",
    ),
    "svelte": (
        "<script>\n  let x = 1;\n</script>\n<p>{x}</p>\n",
        "<script>\n  let x = 1;\n",
    ),
    "graphql": (
        "query Q { field(sub: 1) { nested } }\n",
        "query Q { field( { }\n",
    ),
    "hcl": (
        'resource "aws_s3_bucket" "b" {\n  bucket = "x"\n}\n',
        'resource "aws_s3_bucket" "b" {\n  bucket = \n',
    ),
    "make": (
        "all: build\n\t@echo done\n\nbuild:\n\tgcc -o app main.c\n",
        "all:\n\t@echo $(\n",
    ),
    "nix": (
        "{\n  package = import ./nix { inherit pkgs; };\n}\n",
        "{\n  package = import\n",
    ),
}


# ---------------------------------------------------------------------------
# Every installed grammar resolves and parses
# ---------------------------------------------------------------------------

def _resolved_language_params():
    """Parametrize over core + extra languages, skipping absent extras."""
    params = [pytest.param(lang, id=lang) for lang in CORE_LANGUAGES]
    params.extend(
        pytest.param(lang, id=lang, marks=_EXTRA_SKIP_MARKS[lang])
        for lang in EXTRA_LANGUAGES
    )
    return params


@pytest.mark.parametrize("language", _resolved_language_params())
def test_installed_grammar_resolves_and_parses(language):
    """Each installed grammar resolves through the resolver and parses."""
    valid, _ = MINIMAL_SNIPPETS[language]
    assert validate_parse(valid, language) is True


@pytest.mark.parametrize("language", _resolved_language_params())
def test_broken_snippet_yields_diagnostics(language):
    """A2's relative diagnostics work on EVERY installed grammar, not just
    Python: a deliberately broken snippet produces non-empty error traits
    and ``is_valid=False``."""
    _, broken = MINIMAL_SNIPPETS[language]
    diags = parse_diagnostics(broken, language)
    assert diags.errors, f"expected parse-error traits for broken {language}"
    assert diags.is_valid is False
    # Trait shape: (start_byte, end_byte, kind) with a known kind label.
    # MISSING traits are zero-width (the parser names the byte where the
    # missing token belongs), so start == end is legal; ERROR traits span.
    start, end, kind = diags.errors[0]
    assert 0 <= start <= end <= len(diags.source.encode("utf-8"))
    assert kind in ("ERROR", "MISSING")


def test_core_grammars_are_hard_dependencies():
    """The core-format languages must resolve WITHOUT the all-grammars
    extra: they are hard dependencies, wired so detect_language() can
    promise them on a default install."""
    assert set(CORE_LANGUAGES) <= set(LANGUAGE_ALIASES)
    for lang in CORE_LANGUAGES:
        assert _wheel_present(lang), (
            f"{lang} is a hard dependency — its wheel must be installed"
        )


# ---------------------------------------------------------------------------
# Alias spellings resolve to the same parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("csharp", "c_sharp"),
        ("c#", "c_sharp"),
        ("golang", "go"),
        ("c++", "cpp"),
        ("CSharp", "c_sharp"),  # case/spacing normalization
        # B2 alternate spellings for the broad grammar set.
        ("sh", "bash"),
        ("shell", "bash"),
        ("zsh", "bash"),
        ("docker", "dockerfile"),
        ("makefile", "make"),
        ("gql", "graphql"),
        ("terraform", "hcl"),
        ("yml", "yaml"),
    ],
)
def test_alias_spellings_share_one_parser(spelling, canonical):
    """Alternate spellings canonicalize BEFORE the cache key, so both
    spellings hand back the identical parser object."""
    assert get_parser(spelling) is get_parser(canonical)
    assert get_language(spelling) is get_language(canonical)


def test_alias_table_targets_are_canonical():
    """Every alias target must itself be a canonical (resolvable) name —
    an alias pointing at a non-canonical name would deadlock resolution."""
    for spelling, canonical in ast_analyzer.LANGUAGE_NAME_ALIASES.items():
        assert spelling not in LANGUAGE_ALIASES, (
            f"alias '{spelling}' must not shadow a canonical name"
        )
        assert canonical in LANGUAGE_ALIASES, (
            f"alias '{spelling}' points at unknown canonical '{canonical}'"
        )


def test_spec_table_covers_every_detected_language():
    """Extension-detected languages must all have declarative grammar
    specs — detect_language() promises resolution, the table delivers."""
    for ext, language in EXTENSION_TO_LANGUAGE.items():
        assert language in LANGUAGE_ALIASES, (
            f"extension {ext} maps to '{language}' which has no GrammarSpec"
        )


def test_spec_table_shape():
    """Declarative table hygiene: modules are wheel names, entries are
    attribute names, and multi-grammar wheels keep EXACT entries."""
    for language, spec in LANGUAGE_ALIASES.items():
        assert spec.modules, f"{language}: needs at least one module candidate"
        assert spec.entries, f"{language}: needs at least one entry candidate"
        for module in spec.modules:
            assert module.startswith("tree_sitter_")
        for entry in spec.entries:
            assert entry.isidentifier()
    # Multi-grammar wheels expose one entry point per grammar; a generic
    # fallback here would silently substitute a DIFFERENT grammar.
    ts_spec = LANGUAGE_ALIASES["tsx"]
    assert ts_spec.modules == ("tree_sitter_typescript",)
    assert ts_spec.entries == ("language_tsx",)
    assert LANGUAGE_ALIASES["typescript"].entries == ("language_typescript",)
    # B2 multi-grammar wheels: xml (xml + dtd) and markdown (block + inline).
    xml_spec = LANGUAGE_ALIASES["xml"]
    assert xml_spec.modules == ("tree_sitter_xml",)
    assert xml_spec.entries == ("language_xml",)
    dtd_spec = LANGUAGE_ALIASES["xml_dtd"]
    assert dtd_spec.modules == ("tree_sitter_xml",)
    assert dtd_spec.entries == ("language_dtd",)
    md_spec = LANGUAGE_ALIASES["markdown"]
    assert md_spec.modules == ("tree_sitter_markdown",)
    assert md_spec.entries == ("language",)
    inline_spec = LANGUAGE_ALIASES["markdown_inline"]
    assert inline_spec.modules == ("tree_sitter_markdown",)
    # The inline sub-grammar ships in the SAME wheel — there is no separate
    # markdown-inline package on PyPI (probed: 404) — and its entry point is
    # spelled `inline_language`, not `language_inline`.
    assert inline_spec.entries == ("inline_language",)


# ---------------------------------------------------------------------------
# Typed failure — unknown language
# ---------------------------------------------------------------------------

def test_unknown_language_raises_typed_error_with_install_hint():
    """Unknown language → GrammarUnavailableError naming the language and
    the pip package hint. The resolver reports truthfully; the CALLER
    decides policy (never a silent no-validation fallback).

    B2 triage: the probe language must be a name NO wheel and NO language
    pack has ever shipped. B1 used "cobol", but tree-sitter-language-pack
    (now a first-class all-grammars extra) covers cobol — with the extra
    installed the pack fallback RESOLVES it, which is req. 4 working as
    designed, not a resolver bug."""
    unknown = "definitely_not_a_language"
    with pytest.raises(GrammarUnavailableError) as excinfo:
        get_language(unknown)
    message = str(excinfo.value)
    assert unknown in message
    # The pip hint derives from the module name with underscores → dashes.
    assert "pip install tree-sitter-definitely-not-a-language" in message
    # Type-compatible with the historical ValueError so existing
    # `except (ValueError, ...)` consumers keep working.
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.language == unknown


def test_validate_parse_propagates_typed_error():
    """Validation entry points share the resolver's single code path, so an
    unresolvable language surfaces the same typed error through both."""
    with pytest.raises(GrammarUnavailableError):
        validate_parse("anything", "not_a_language")
    with pytest.raises(GrammarUnavailableError):
        parse_diagnostics("anything", "not_a_language")


def test_failed_resolution_does_not_poison_cache():
    """A failed resolution must not cache a negative entry — installing the
    grammar later in the same process must succeed without a restart.

    B2 triage: probe name is one no wheel and no language pack ships (see
    test_unknown_language_raises_typed_error_with_install_hint)."""
    canonical = "definitely_not_a_language"
    with pytest.raises(GrammarUnavailableError):
        get_language(canonical)
    assert canonical not in ast_analyzer._language_cache
    assert canonical not in ast_analyzer._parser_cache


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

def test_parser_cache_identity():
    """Parse calls happen on multi-MB files — re-resolution must return the
    cached parser (identity), never rebuild."""
    first = get_parser("go")
    second = get_parser("go")
    assert first is second
    assert get_language("go") is get_language("go")


@pytest.fixture()
def clean_resolver_caches():
    """Pristine resolver state: snapshot → clear → restore.

    Serves two kinds of tests: those that mutate the declarative tables,
    and cache-keying tests that must not see entries OTHER tests left in
    the module-level caches (the golden matrix legitimately resolves
    pack-only languages — ``fsharp`` among them — straight into
    ``_parser_cache``). The pre-test entries, same objects included, are
    put back afterwards, so warm-cache speed elsewhere is untouched."""
    saved_langs = dict(ast_analyzer._language_cache)
    saved_parsers = dict(ast_analyzer._parser_cache)
    ast_analyzer.clear_grammar_caches()
    yield
    ast_analyzer._language_cache.clear()
    ast_analyzer._language_cache.update(saved_langs)
    ast_analyzer._parser_cache.clear()
    ast_analyzer._parser_cache.update(saved_parsers)


def test_resolver_caches_are_keyed_canonically(clean_resolver_caches):
    """The cache holds ONE entry per canonical language even when reached
    through alias spellings.

    Hermetic via the fixture: ``"fsharp"`` — a DIFFERENT canonical language
    that the golden matrix's F2 census fixtures resolve into the parser
    cache earlier in the suite — also ends with ``"sharp"``, so the
    one-entry count below is only meaningful on a pristine cache. The
    production canonicalization itself is correct: ``csharp`` maps to
    ``c_sharp`` BEFORE the cache key, and ``fsharp`` is nobody's alias."""
    get_parser("csharp")
    get_parser("c_sharp")
    assert sum(1 for key in ast_analyzer._parser_cache if key.endswith("sharp")) == 1


def test_generic_convention_resolves_grammar_missing_from_table(
    monkeypatch, clean_resolver_caches,
):
    """A grammar wheel installed LATER (e.g. B2's `tree_sitter_zig`) must
    resolve with ZERO code changes: the generic
    ``tree_sitter_<lang>`` + ``language()`` convention is the default."""
    monkeypatch.delitem(LANGUAGE_ALIASES, "rust")
    assert "rust" not in LANGUAGE_ALIASES
    language = get_language("rust")
    assert language is not None
    assert validate_parse("fn f() -> i32 { 1 }\n", "rust")


def test_tabled_language_with_missing_wheel_raises_with_hint(
    monkeypatch, clean_resolver_caches,
):
    """A table'd language whose wheel is NOT installed (simulated: the
    import fails) raises the typed error naming the language and the pip
    hint derived from the DECLARED module candidate — not the generic
    convention."""
    real_import_module = importlib.import_module

    def fake_import(name, *args, **kwargs):
        # B2 triage: the pack modules must fail too — with the pack
        # installed, the B1 fallback would otherwise RESOLVE elixir (the
        # pack bundles it) and the "no resolution source" contract would
        # never be reached. The test's subject is the typed error when
        # NOTHING can serve the declared wheel, so the simulation removes
        # every resolution source, direct and aggregate.
        if name == "tree_sitter_elixir" or name in _PACK_MODULES:
            raise ImportError(f"No module named {name!r} (simulated absence)")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(GrammarUnavailableError) as excinfo:
        get_language("elixir")
    message = str(excinfo.value)
    assert "elixir" in message
    assert "pip install tree-sitter-elixir" in message


# ---------------------------------------------------------------------------
# Optional language pack (fallback resolver — never required)
# ---------------------------------------------------------------------------

_PACK_MODULES = ("tree_sitter_languages", "tree_sitter_language_pack")


def _pack_is_importable() -> bool:
    return any(
        importlib.util.find_spec(mod) is not None for mod in _PACK_MODULES
    )


def _pack_language_names() -> list[str] | None:
    """Best-effort introspection of the pack's language name list.

    The bundled 0.x line exposes ``SupportedLanguage`` (a typing.Literal of
    every bundled name); the 1.x line exposes ``SUPPORTED_LANGUAGES``.
    Returns None when the pack offers neither — callers then skip the count
    assertion rather than guessing.
    """
    try:
        pack = importlib.import_module("tree_sitter_language_pack")
    except ImportError:
        return None
    names = getattr(pack, "SUPPORTED_LANGUAGES", None)
    if names:
        return sorted(names)
    supported = getattr(pack, "SupportedLanguage", None)
    if supported is not None:
        args = sys.modules["typing"].get_args(supported)
        if args:
            return sorted(args)
    return None


@pytest.mark.skipif(
    not _pack_is_importable(),
    reason=(
        "optional dependency tree_sitter_languages / tree_sitter_language_pack "
        "not installed in this environment (fastedit never requires it; the "
        "fallback path is exercised here only when present)"
    ),
)
def test_language_pack_fallback_resolver():
    """When a pack is importable, its ``get_language``/``get_parser`` serves
    as the fallback resolver for languages the direct wheels don't cover."""
    language = ast_analyzer._resolve_from_language_pack("python")
    assert language is not None, "pack present but returned no Language for python"
    # Whatever object shape the pack returns, it must be usable as a
    # tree-sitter Language by the rest of the pipeline.
    parser = get_parser("python")  # direct wheel still wins — pack is fallback
    assert parser.parse(b"def f():\n    return 1\n") is not None


def test_language_pack_is_not_imported_at_module_import():
    """The resolver must not import any pack module when fastedit's
    ast_analyzer is imported — the pack is consulted lazily, only when
    direct wheel resolution fails.

    B2 triage: B1 asserted this in-process via ``sys.modules``, which broke
    the moment any OTHER test legitimately imported the optional pack (the
    coverage test below does). The contract is really about MODULE IMPORT
    time, so it is now proven in a fresh interpreter: import fastedit's
    analyzer there and assert no pack module was loaded as a side effect.
    The default tier stays fast: the pack is never imported in-process by
    this assertion."""
    code = (
        "import sys; import fastedit.data_gen.ast_analyzer as a; "
        "leaked = [m for m in sys.modules if m.startswith("
        "'tree_sitter_languages') or m.startswith('tree_sitter_language_pack')]; "
        "print('LEAKED:' + ','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    leaked = [
        m for m in result.stdout.split("LEAKED:", 1)[1].strip().split(",") if m
    ]
    assert leaked == [], f"pack imported at module import time: {leaked}"


@pytest.mark.skipif(
    not _pack_is_importable(),
    reason=(
        "optional dependency tree-sitter-language-pack not installed — the "
        "pack-coverage contract only applies when the all-grammars extra is "
        "present"
    ),
)
def test_language_pack_coverage_when_present():
    """THE B2 headline: with the pack installed, fastedit's resolver serves
    every bundled language through the B1 fallback path — hundreds of
    formats with zero per-language code.

    Reports (printed for the B2 ledger):
    * how many language names the pack introspection exposes;
    * how many of those are NOT served by direct wheels (the pack's NET
      addition to fastedit's coverage).

    Only ONE grammar is actually loaded (``r`` — a pack-only language with
    no PyPI wheel): introspection never materializes the other ~170, so the
    default tier stays fast."""
    names = _pack_language_names()
    if names is None:
        pytest.skip("pack exposes no introspectable language name list")
    direct_served = set(CORE_LANGUAGES) | set(EXTRA_LANGUAGES)
    pack_only = [n for n in names if n not in direct_served]
    print(
        f"\nlanguage pack: {len(names)} languages; "
        f"NET new vs direct wheels: {len(pack_only)}"
    )
    assert len(names) > 100, (
        "expected the ~170-language bundle, got a degenerate pack"
    )
    # A pack-only language (no direct wheel anywhere — probed: tree-sitter-r
    # is not on PyPI) resolves through the B1 fallback and parses.
    assert "r" in names
    assert validate_parse("x <- c(1, 2, 3)\nprint(mean(x))\n", "r") is True


@pytest.mark.skipif(
    not _pack_is_importable(),
    reason="optional dependency tree-sitter-language-pack not installed",
)
def test_pack_covered_real_language_resolves():
    """B2 triage companion: a REAL language that no direct wheel ships
    (cobol) resolves through the pack fallback when the extra is installed
    — this is exactly why the B1 unknown-language tests moved to a name no
    pack could ever have."""
    assert validate_parse(
        "       IDENTIFICATION DIVISION.\n", "cobol",
    ) is True


# ---------------------------------------------------------------------------
# B2 extension table — new pairs resolve end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # B1 trivially-aliasable extensions (kept as regression pins).
        ("app.mjs", "javascript"),
        ("app.cjs", "javascript"),
        ("app.mts", "typescript"),
        ("app.cts", "typescript"),
        ("header.hxx", "cpp"),
        ("index.phtml", "php"),
        ("legacy.php5", "php"),
        # B2 core formats.
        ("index.html", "html"),
        ("index.htm", "html"),
        ("config.xml", "xml"),
        ("icon.svg", "xml"),  # svg IS xml — parse sanity asserted separately
        ("schema.dtd", "xml_dtd"),
        ("README.md", "markdown"),
        ("notes.markdown", "markdown"),
        ("data.json", "json"),
        ("config.yaml", "yaml"),
        ("config.yml", "yaml"),
        ("style.css", "css"),
        ("run.sh", "bash"),
        ("run.bash", "bash"),
        ("app.toml", "toml"),
        ("query.sql", "sql"),
        ("Container.dockerfile", "dockerfile"),
    ],
)
def test_extension_mappings_resolve(filename, expected):
    """detect_language() must never hand back a name the resolver cannot
    honor: every wired extension round-trips to a language whose grammar is
    a HARD dependency and actually parses its minimal snippet."""
    assert detect_language(filename) == expected
    language = detect_language(filename)
    assert language is not None
    valid, _ = MINIMAL_SNIPPETS[language]
    assert validate_parse(valid, language) is True


def test_svg_is_xml_parse_sanity():
    """The mission's explicit check: .svg maps to the xml grammar, and a
    real SVG document (namespaced attributes, self-closing children) parses
    with zero error traits."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">\n'
        '  <circle cx="4" cy="4" r="3" fill="#333"/>\n'
        "</svg>\n"
    )
    assert validate_parse(svg, "xml") is True
    diags = parse_diagnostics(svg, "xml")
    assert diags.is_valid is True
    assert diags.errors == []


def test_markdown_grammar_is_structurally_lenient():
    """Implementation plan §3 risk 1, PINNED: the markdown grammar is
    error-tolerant by design — every structural malformation we probed
    (unclosed fence, unclosed emphasis/link, stray brackets, broken table)
    parses with NO error traits. Consequence (documented, not a bug): md
    parse validity is nearly always True, so markdown correctness leans on
    the D1 text heuristics / D3 structure traits, never on the parse gate
    alone. Only a control character (NUL) trips the parser (see the
    parametrized broken-snippet contract)."""
    malformed = [
        "```py\nx = 1\n",          # unclosed fence
        "[text](unclosed\n",        # unclosed link
        "*unclosed emphasis\n",     # unclosed emphasis
        "| a | b |\n| --- |\n",     # broken table row
        "](([))\n",                 # stray brackets
        "~~~\n~~~\n~~~x\n",         # fence soup
    ]
    for snippet in malformed:
        diags = parse_diagnostics(snippet, "markdown")
        assert diags.is_valid is True, f"unexpectedly flagged: {snippet!r}"
        assert diags.errors == []


def test_default_install_leaves_unwired_extensions_undetected():
    """Extensions whose grammar is an EXTRA dependency (or has no wheel at
    all) stay None: detect_language() promises resolution on a DEFAULT
    install, so it never advertises a language the default install cannot
    honor. These are honest plain-text paths — the languages still resolve
    via get_language() when the all-grammars extra is installed."""
    for filename in ("main.scala", "init.lua", "script.pl", "app.jl",
                     "main.zig", "page.svelte", "schema.graphql", "infra.hcl",
                     "Makefile.mk", "default.nix", "notes.tex", "cfg.ini",
                     "app.vue"):
        assert detect_language(filename) is None, filename


# ---------------------------------------------------------------------------
# B2 consumer contract — deterministic chunked_merge edits on new formats
# ---------------------------------------------------------------------------

def _scripted_whole_file(results):
    """A merge_fn replaying *results* (last repeats), recording calls — the
    established scripted-engine pattern (tests/test_relative_validation.py)."""
    from types import SimpleNamespace

    calls: list[tuple[str, str, str | None]] = []

    def merge_fn(code, snippet, language):
        calls.append((code, snippet, language))
        result = results[min(len(calls) - 1, len(results) - 1)]
        return SimpleNamespace(
            merged_code=result,
            parse_valid=True,
            tokens_generated=9,
            latency_ms=1.0,
            truncated=False,
        )

    return merge_fn, calls


# The new formats have NO symbol AST maps yet (ast_utils' node-type tables
# are per-language and B3's golden matrix will extend them), so
# locate_chunks hands back the whole file and the scripted engine performs
# the merge — the deterministic locate→edit→validate path for data formats.
#
# Snippet shape follows the validator's marker semantics (chunked_merge
# _check_hallucinations): a CONTEXT ANCHOR (an original line) first, then
# the new lines, then the keep-marker. New lines declared before the first
# marker must precede the surviving originals of their segment — without a
# leading anchor the declared inserts would be ranked "before the first
# anchor" and any mid-file landing would be a side-order violation.
HTML_ORIGINAL = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head><title>Page</title></head>\n"
    "<body>\n"
    "  <h1>Title</h1>\n"
    "  <p>First paragraph.</p>\n"
    "</body>\n"
    "</html>\n"
)
HTML_ANCHOR = "  <p>First paragraph.</p>\n"
HTML_INSERT = "  <p>Second paragraph.</p>\n"
HTML_GOOD_MERGE = HTML_ORIGINAL.replace(
    HTML_ANCHOR,
    HTML_ANCHOR + HTML_INSERT,
)
# Probe-verified html breakage that survives INSIDE a valid <body>
# (mismatched tags are recovered by the grammar; a valueless attribute is
# not).
HTML_BROKEN_INSERT = "  <div attr=></div>\n"

JSON_ORIGINAL = (
    '{\n'
    '  "name": "fastedit",\n'
    '  "version": "0.5.0",\n'
    '  "tags": ["ast", "edit"]\n'
    "}\n"
)
JSON_ANCHOR = '  "version": "0.5.0",\n'
JSON_INSERT = '  "license": "MIT",\n'
JSON_GOOD_MERGE = JSON_ORIGINAL.replace(
    JSON_ANCHOR,
    JSON_ANCHOR + JSON_INSERT,
)
JSON_BROKEN_INSERT = '  "broken": ,\n'

YAML_ORIGINAL = (
    "name: fastedit\n"
    "version: 0.5.0\n"
    "tags:\n"
    "  - ast\n"
    "  - edit\n"
)
YAML_ANCHOR = "version: 0.5.0\n"
YAML_INSERT = "description: ast-aware editor\n"
YAML_GOOD_MERGE = YAML_ORIGINAL.replace(
    YAML_ANCHOR,
    YAML_ANCHOR + YAML_INSERT,
)
YAML_BROKEN_INSERT = "key: [unclosed\n  - bad\n"

MD_ORIGINAL = (
    "# Title\n"
    "\n"
    "Intro paragraph.\n"
    "\n"
    "## Existing Section\n"
    "\n"
    "Body text.\n"
)
MD_ANCHOR = "Intro paragraph.\n"
MD_INSERT = "## New Section\n\nFresh content.\n\n"
MD_GOOD_MERGE = MD_ORIGINAL.replace(MD_ANCHOR, MD_ANCHOR + MD_INSERT)


def _run_whole_file_edit(tmp_path, filename, original, snippet, merge_fn):
    from fastedit.inference.chunked_merge import chunked_merge

    target = tmp_path / filename
    target.write_text(original, encoding="utf-8")
    return chunked_merge(
        original_code=original,
        snippet=snippet,
        file_path=str(target),
        merge_fn=merge_fn,
        language=detect_language(filename),
    )


@pytest.mark.parametrize(
    ("filename", "original", "anchor", "insert", "good_merge"),
    [
        ("page.html", HTML_ORIGINAL, HTML_ANCHOR, HTML_INSERT, HTML_GOOD_MERGE),
        ("data.json", JSON_ORIGINAL, JSON_ANCHOR, JSON_INSERT, JSON_GOOD_MERGE),
        ("config.yaml", YAML_ORIGINAL, YAML_ANCHOR, YAML_INSERT, YAML_GOOD_MERGE),
        ("README.md", MD_ORIGINAL, MD_ANCHOR, MD_INSERT, MD_GOOD_MERGE),
    ],
)
def test_chunked_merge_edit_end_to_end_new_formats(
    tmp_path, filename, original, anchor, insert, good_merge,
):
    """Consumer contract per B2 major format (html, json, yaml, markdown):
    one real deterministic ``chunked_merge`` edit — locate (chunk locator
    over the parsed original), edit (deterministic scripted engine), and
    the validation battery driven by the NEW grammar — lands the declared
    insertion with parse_valid True."""
    snippet = anchor + insert + "... existing code ...\n"
    merge_fn, calls = _scripted_whole_file([good_merge])
    result = _run_whole_file_edit(tmp_path, filename, original, snippet, merge_fn)

    assert result.parse_valid is True
    assert result.chunks_rejected == 0
    assert result.merged_code == good_merge
    # The edit went through the pipeline's locate step (one whole-file
    # chunk for these symbol-less formats), not around it.
    assert len(calls) == 1
    assert calls[0][2] == detect_language(filename)
    # The bytes that would be written re-validate through the resolver's
    # grammar — what the write gate re-checks.
    assert validate_parse(result.merged_code, detect_language(filename)) is True


@pytest.mark.parametrize(
    ("filename", "original", "anchor", "insert", "broken_insert"),
    [
        ("page.html", HTML_ORIGINAL, HTML_ANCHOR, HTML_INSERT,
         HTML_BROKEN_INSERT),
        ("data.json", JSON_ORIGINAL, JSON_ANCHOR, JSON_INSERT,
         JSON_BROKEN_INSERT),
        ("config.yaml", YAML_ORIGINAL, YAML_ANCHOR, YAML_INSERT,
         YAML_BROKEN_INSERT),
        # markdown deliberately ABSENT: its grammar cannot be parse-broken
        # (see test_markdown_grammar_is_structurally_lenient) — there is no
        # snippet whose declared content the md parse gate could reject; md
        # validity leans on D1/D3 per plan §3 risk 1.
    ],
)
def test_chunked_merge_rejects_broken_new_format_edits(
    tmp_path, filename, original, anchor, insert, broken_insert,
):
    """Negative control: the snippet DECLARES parse-broken new content
    (anchor + broken line + marker), so the content-faithfulness check
    passes — what must catch it is the RELATIVE PARSE GATE armed with the
    B2 grammar. The gate retries with the parse failure as the corrective
    note, then rejects and keeps the original byte-exact: the grammar is
    genuinely consulted for the new formats, never vacuously skipped."""
    snippet = anchor + broken_insert + "... existing code ...\n"
    # The scripted engine keeps returning the parse-broken merge it
    # "believes" is right — the loop must reject it on parse grounds.
    # Content-faithful by construction (it contains exactly the declared
    # anchor and new lines), parse-broken by the B2 grammar's verdict.
    broken_merge = original.replace(anchor, anchor + broken_insert)
    merge_fn, calls = _scripted_whole_file([broken_merge])
    result = _run_whole_file_edit(tmp_path, filename, original, snippet, merge_fn)

    assert result.chunks_rejected == 1
    assert result.merged_code == original, (
        "a parse-broken merge must never reach the file"
    )
    assert len(calls) >= 2, "the gate must have retried before rejecting"
    # The corrective note names the PARSE failure — proving the rejection
    # came from the (new-grammar) relative parse gate, not faithfulness.
    language = detect_language(filename)
    assert f"does not parse as {language}" in calls[1][1], calls[1][1]


# ---------------------------------------------------------------------------
# B1 regression pins — extension mappings added before B2
# ---------------------------------------------------------------------------

def test_chunked_merge_go_edit_end_to_end(tmp_path):
    """Consumer contract: one real deterministic ``chunked_merge`` edit
    through Go — AST anchor resolution (``get_ast_map_from_source`` →
    ``get_parser``) and the relative parse gate (``parse_diagnostics``)
    both resolve their grammar through the B1 resolver."""
    from fastedit.inference.chunked_merge import chunked_merge

    go_original = (
        "package main\n"
        "\n"
        'import "fmt"\n'
        "\n"
        "// Greeter greets by name.\n"
        "func Greeter(name string) string {\n"
        '\treturn fmt.Sprintf("hi %s", name)\n'
        "}\n"
        "\n"
        "func main() {\n"
        '\tfmt.Println(Greeter("world"))\n'
        "}\n"
    )
    go_insert = (
        "// Farewell says goodbye.\n"
        "func Farewell(name string) string {\n"
        '\treturn "bye " + name\n'
        "}\n"
    )
    go_expected = go_original.replace(
        "func main() {",
        "// Farewell says goodbye.\n"
        "func Farewell(name string) string {\n"
        '\treturn "bye " + name\n'
        "}\n"
        "\n"
        "func main() {",
    )

    def _no_model(*_args, **_kwargs):
        raise AssertionError("merge_fn must not be called on the deterministic path")

    file_path = tmp_path / "main.go"
    file_path.write_text(go_original, encoding="utf-8")

    result = chunked_merge(
        original_code=go_original,
        snippet=go_insert,
        file_path=str(file_path),
        merge_fn=_no_model,
        language="go",
        after="Greeter",
    )

    assert result.model_tokens == 0, "after= is the zero-model fast path"
    assert result.parse_valid is True
    assert result.merged_code == go_expected
    # The edit must also survive the file round-trip with the resolver's
    # grammar (what the write gate re-validates).
    assert validate_parse(file_path.read_text(encoding="utf-8"), "go") is True


def test_chunked_merge_go_edit_rejects_broken_snippet(tmp_path):
    """Consumer contract, relative-rule side (req. 9): a broken Go snippet
    inserted after an anchor introduces a NEW error trait → the gate flags
    the merge unparse-valid instead of silently writing corruption."""
    from fastedit.inference.chunked_merge import chunked_merge

    go_original = (
        "package main\n"
        "\n"
        'import "fmt"\n'
        "\n"
        "// Greeter greets by name.\n"
        "func Greeter(name string) string {\n"
        '\treturn fmt.Sprintf("hi %s", name)\n'
        "}\n"
        "\n"
        "func main() {\n"
        '\tfmt.Println(Greeter("world"))\n'
        "}\n"
    )

    def _no_model(*_args, **_kwargs):
        raise AssertionError("merge_fn must not be called on the deterministic path")

    file_path = tmp_path / "main.go"
    file_path.write_text(go_original, encoding="utf-8")

    result = chunked_merge(
        original_code=go_original,
        snippet="func Broken( string {\n\treturn\n",
        file_path=str(file_path),
        merge_fn=_no_model,
        language="go",
        after="Greeter",
    )

    assert result.parse_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
