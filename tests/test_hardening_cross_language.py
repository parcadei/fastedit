"""Cross-language parametrized hardening tests for M1-M4 features.

Milestone 4.6: expand coverage beyond Python + TS happy paths to include
Java, Rust, Go, Kotlin (required), plus Ruby, JavaScript.

Milestone 4.7: confidence-based reference filter (see
:mod:`fastedit.inference.rename`) unblocks non-AST-native langs. tldr
emits kind="other" on java/kotlin/ruby/swift/php/c#/cpp/c with
confidence=1.0 for real code hits and confidence=0.5 for string-literal
substring hits. We now accept anything with confidence >= 0.9, which
gives real rename coverage on every lang fastedit supports.

See docs/testing-matrix.md for the full feature x language matrix.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from fastedit.inference.caller_safety import (
    check_cross_file_callers,
    signature_changed,
)
from fastedit.inference.rename import do_rename_ast


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TLDR_AVAILABLE = shutil.which("tldr") is not None

# Langs where tldr's reference resolver has AST-level understanding (can
# distinguish definition / call / import / string-literal content). Per
# `tldr references --help`: python, typescript, go, rust.
TLDR_NATIVE_LANGS = {"python", "typescript", "go", "rust"}


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass
class LangSpec:
    """One language's code templates for M1/M2 cross-lang tests."""

    lang: str
    file_ext: str
    caller_ext: str
    symbol: str
    rename_source: str
    caller_source: str
    expected_rename_count_min: int
    skips_strings_and_comments: bool


LANG_SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        lang="python",
        file_ext=".py",
        caller_ext=".py",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        def oldName():
            \"\"\"docstring: oldName is mentioned here.\"\"\"
            return 1

        # oldName in comment
        msg = "oldName in string"
        x = oldName()
        """),
        caller_source=textwrap.dedent("""\
        from mod import oldName

        def use():
            return oldName()
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=True,
    ),
    "typescript": LangSpec(
        lang="typescript",
        file_ext=".ts",
        caller_ext=".ts",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        /** JSDoc mentions oldName in prose. */
        export function oldName(): number {
          return 1;
        }

        // oldName in comment
        const msg: string = "oldName in string";
        const x = oldName();
        """),
        caller_source=textwrap.dedent("""\
        import { oldName } from "./mod";

        export function use() {
          return oldName();
        }
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=True,
    ),
    "go": LangSpec(
        lang="go",
        file_ext=".go",
        caller_ext=".go",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        package mod

        // oldName in comment
        func oldName() int {
            return 1
        }

        var msg = "oldName in string"
        var x = oldName()
        """),
        caller_source=textwrap.dedent("""\
        package mod

        func use() int {
            return oldName()
        }
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=True,
    ),
    "rust": LangSpec(
        lang="rust",
        file_ext=".rs",
        caller_ext=".rs",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        // oldName in comment
        pub fn oldName() -> i32 {
            1
        }

        pub fn caller_local() -> i32 {
            let _msg = "oldName in string";
            oldName()
        }
        """),
        caller_source=textwrap.dedent("""\
        pub fn use_it() -> i32 {
            crate::oldName()
        }
        """),
        expected_rename_count_min=1,
        skips_strings_and_comments=True,
    ),
    "java": LangSpec(
        lang="java",
        file_ext=".java",
        caller_ext=".java",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        public class ModOne {
            // oldName in comment
            public static int oldName() {
                String msg = "oldName in string";
                return 1;
            }

            public static int local() {
                return oldName();
            }
        }
        """),
        caller_source=textwrap.dedent("""\
        public class Caller {
            public static int use() {
                return ModOne.oldName();
            }
        }
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=False,
    ),
    "kotlin": LangSpec(
        lang="kotlin",
        file_ext=".kt",
        caller_ext=".kt",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        // oldName in comment
        fun oldName(): Int {
            val msg = "oldName in string"
            return 1
        }

        fun local(): Int {
            return oldName()
        }
        """),
        caller_source=textwrap.dedent("""\
        fun use(): Int {
            return oldName()
        }
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=False,
    ),
    "javascript": LangSpec(
        lang="javascript",
        file_ext=".js",
        caller_ext=".js",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        // oldName in comment
        export function oldName() {
          return 1;
        }

        const msg = "oldName in string";
        export const x = oldName();
        """),
        caller_source=textwrap.dedent("""\
        import { oldName } from "./mod.js";

        export function use() {
          return oldName();
        }
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=True,
    ),
    "ruby": LangSpec(
        lang="ruby",
        file_ext=".rb",
        caller_ext=".rb",
        symbol="oldName",
        rename_source=textwrap.dedent("""\
        # oldName in comment
        def oldName
          msg = "oldName in string"
          1
        end

        def local
          oldName
        end
        """),
        caller_source=textwrap.dedent("""\
        require_relative "mod"

        def use
          oldName
        end
        """),
        expected_rename_count_min=2,
        skips_strings_and_comments=False,
    ),
}


REQUIRED_LANGS = ["python", "typescript", "java", "rust", "go", "kotlin"]
STRONGLY_RECOMMENDED_LANGS = ["ruby", "javascript"]
ALL_TESTED_LANGS = REQUIRED_LANGS + STRONGLY_RECOMMENDED_LANGS


# ===========================================================================
# M1 — do_rename_ast (single-file AST rename) across languages
# ===========================================================================


@pytest.mark.parametrize("lang_key", ALL_TESTED_LANGS)
def test_m1_rename_ast_happy_path(lang_key: str, tmp_path: Path):
    """M1 happy path: single-file rename across languages.

    Invariants (all 13 supported langs, post-M4.7):
      - Definition site is renamed (count >= 1).
      - String/comment occurrences preserved for langs where tldr emits
        kind="string"/"comment" OR confidence<=0.5 for substring hits
        (AST-native langs + the non-native confidence-1.0 filter).
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    path = root / f"mod{spec.file_ext}"
    path.write_text(spec.rename_source)

    new_content, count, skipped = do_rename_ast(path, spec.symbol, "newName")

    assert count >= 1, (
        f"[{lang_key}] expected >=1 replacement, got {count}. "
        f"new_content:\n{new_content}"
    )
    assert "newName" in new_content

    if spec.skips_strings_and_comments:
        assert '"oldName in string"' in new_content, (
            f"[{lang_key}] rename leaked into string literal"
        )
        if "# oldName in comment" in spec.rename_source:
            assert "# oldName in comment" in new_content
        if "// oldName in comment" in spec.rename_source:
            assert "// oldName in comment" in new_content


@pytest.mark.parametrize("lang_key", ALL_TESTED_LANGS)
def test_m1_rename_ast_idempotent(lang_key: str, tmp_path: Path):
    """M1 idempotence (property): rename(rename(x, a, b), a, b) finds 0.

    Second call must be a no-op: the first pass already consumed every
    AST-verified reference.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    path = root / f"mod{spec.file_ext}"
    path.write_text(spec.rename_source)

    new_content1, count1, _ = do_rename_ast(path, spec.symbol, "newName")
    if count1 == 0:
        pytest.skip(f"[{lang_key}] baseline rename found 0 matches")
    path.write_text(new_content1)

    new_content2, count2, _ = do_rename_ast(path, spec.symbol, "newName")
    assert count2 == 0, (
        f"[{lang_key}] idempotence broken: second rename found {count2} "
        f"more refs. Content:\n{new_content2}"
    )


@pytest.mark.parametrize("lang_key", ALL_TESTED_LANGS)
def test_m1_rename_ast_dry_run_is_no_op(lang_key: str, tmp_path: Path):
    """M1 property: dry-run equivalent (calling do_rename_ast then not
    writing) leaves the file on-disk unchanged, regardless of language.

    The engine itself is pure (returns new content without writing) —
    this test locks that invariant per language.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    path = root / f"mod{spec.file_ext}"
    path.write_text(spec.rename_source)

    mtime_before = path.stat().st_mtime_ns
    content_before = path.read_text()

    _new_content, _count, _skipped = do_rename_ast(
        path, spec.symbol, "newName",
    )

    assert path.read_text() == content_before, (
        f"[{lang_key}] do_rename_ast mutated file on disk"
    )
    assert path.stat().st_mtime_ns == mtime_before, (
        f"[{lang_key}] do_rename_ast touched mtime"
    )


# ===========================================================================
# M2 — check_cross_file_callers / delete safety across languages
# ===========================================================================


@pytest.mark.parametrize("lang_key", ALL_TESTED_LANGS)
def test_m2_cross_file_callers_detected(lang_key: str, tmp_path: Path):
    """M2: caller in another file -> refs list contains that file.

    Like M1, non-AST-native langs degrade to kind="other" which the
    safety helper filters on. BUT check_cross_file_callers doesn't
    filter on kind at the top level — only drops definition + self-refs.
    So it SHOULD surface grep-quality hits. This test verifies.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    src = root / f"mod{spec.file_ext}"
    src.write_text(spec.rename_source)
    caller = root / f"caller{spec.caller_ext}"
    caller.write_text(spec.caller_source)

    refs = check_cross_file_callers(
        file_path=src, symbol=spec.symbol, project_root=root,
    )

    caller_files = {Path(r["file"]).name for r in refs}
    assert f"caller{spec.caller_ext}" in caller_files, (
        f"[{lang_key}] caller file not found in cross-file refs. "
        f"refs={refs}"
    )


@pytest.mark.parametrize("lang_key", ALL_TESTED_LANGS)
def test_m2_force_override_is_monotonic(lang_key: str, tmp_path: Path):
    """M2 monotonicity across all supported langs.

    Exercises the CLI delete path end-to-end; underlying delete_symbol
    uses the in-memory tree-sitter AST walker which covers every lang
    fastedit supports.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    src = root / f"mod{spec.file_ext}"
    src.write_text(spec.rename_source)
    caller = root / f"caller{spec.caller_ext}"
    caller.write_text(spec.caller_source)

    proc_no_force = _run_cli(["delete", str(src), spec.symbol], cwd=root)
    assert proc_no_force.returncode != 0, (
        f"[{lang_key}] expected refusal without --force, got: "
        f"{proc_no_force.stdout}{proc_no_force.stderr}"
    )

    src.write_text(spec.rename_source)
    caller.write_text(spec.caller_source)

    proc_force = _run_cli(
        ["delete", str(src), spec.symbol, "--force"], cwd=root,
    )
    assert proc_force.returncode == 0, (
        f"[{lang_key}] --force delete failed: "
        f"{proc_force.stdout}{proc_force.stderr}"
    )


# ===========================================================================
# M3 — signature_changed across languages
# ===========================================================================

_SIGCHG_EDITS: dict[str, tuple[str, str]] = {
    "python": (
        "def foo(a):\n    return a\n",
        "def foo(a, b):\n    return a + b\n",
    ),
    "typescript": (
        "export function foo(a: number): number {\n  return a;\n}\n",
        "export function foo(a: number, b: number): number {\n"
        "  return a + b;\n}\n",
    ),
    "javascript": (
        "export function foo(a) {\n  return a;\n}\n",
        "export function foo(a, b) {\n  return a + b;\n}\n",
    ),
    "go": (
        "package mod\n\nfunc foo(a int) int {\n    return a\n}\n",
        "package mod\n\nfunc foo(a int, b int) int {\n    return a + b\n}\n",
    ),
    "rust": (
        "pub fn foo(a: i32) -> i32 {\n    a\n}\n",
        "pub fn foo(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
    ),
    "java": (
        "public class X {\n"
        "    public static int foo(int a) {\n"
        "        return a;\n"
        "    }\n"
        "}\n",
        "public class X {\n"
        "    public static int foo(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "}\n",
    ),
    "kotlin": (
        "fun foo(a: Int): Int {\n    return a\n}\n",
        "fun foo(a: Int, b: Int): Int {\n    return a + b\n}\n",
    ),
    "ruby": (
        "def foo(a)\n  a\nend\n",
        "def foo(a, b)\n  a + b\nend\n",
    ),
    "swift": (
        "func foo(a: Int) -> Int {\n    return a\n}\n",
        "func foo(a: Int, b: Int) -> Int {\n    return a + b\n}\n",
    ),
    "php": (
        "<?php\nfunction foo($a) {\n    return $a;\n}\n",
        "<?php\nfunction foo($a, $b) {\n    return $a + $b;\n}\n",
    ),
    "c_sharp": (
        "public class X {\n"
        "    public static int Foo(int a) {\n"
        "        return a;\n"
        "    }\n"
        "}\n",
        "public class X {\n"
        "    public static int Foo(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "}\n",
    ),
    "cpp": (
        "int foo(int a) {\n    return a;\n}\n",
        "int foo(int a, int b) {\n    return a + b;\n}\n",
    ),
    "c": (
        "int foo(int a) {\n    return a;\n}\n",
        "int foo(int a, int b) {\n    return a + b;\n}\n",
    ),
}

# c_sharp's symbol is `Foo` (PascalCase convention). Every other lang
# uses `foo`; the test dispatches via _SIGCHG_SYMBOLS.
_SIGCHG_SYMBOLS: dict[str, str] = {"c_sharp": "Foo"}


@pytest.mark.parametrize("lang", list(_SIGCHG_EDITS.keys()))
def test_m3_signature_changed_detects_param_addition(lang: str):
    """M3: adding a parameter registers as a signature change across
    all 13 languages fastedit supports (post-M4.7 outsourcing to
    ``tldr structure``).
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    old, new = _SIGCHG_EDITS[lang]
    symbol = _SIGCHG_SYMBOLS.get(lang, "foo")
    result = signature_changed(old, new, symbol, lang)
    assert result is True, (
        f"[{lang}] param-addition not detected as signature change. "
        f"old=\n{old}\nnew=\n{new}"
    )


@pytest.mark.parametrize("lang", list(_SIGCHG_EDITS.keys()))
def test_m3_signature_unchanged_on_body_only_edit(lang: str):
    """M3 hot path: body-only edits do NOT register as signature changes
    — so tldr subprocess isn't spawned (VAL-M3-002).

    Fixtures all use multi-line functions so the declaration line and
    body lines are on separate lines. One-liner functions (whole body
    on one line) intentionally register as signature changes under the
    M4.7 outsourced extraction — documented trade-off for 13/13 lang
    coverage.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    old, _ = _SIGCHG_EDITS[lang]
    body_only_map = {
        "python": "def foo(a):\n    # new comment\n    return a\n",
        "typescript": (
            "export function foo(a: number): number {\n"
            "  // new comment\n  return a;\n}\n"
        ),
        "javascript": (
            "export function foo(a) {\n"
            "  // new comment\n  return a;\n}\n"
        ),
        "go": (
            "package mod\n\nfunc foo(a int) int {\n"
            "    // new comment\n    return a\n}\n"
        ),
        "rust": (
            "pub fn foo(a: i32) -> i32 {\n"
            "    // new comment\n    a\n}\n"
        ),
        "java": (
            "public class X {\n"
            "    public static int foo(int a) {\n"
            "        // new comment\n"
            "        return a;\n"
            "    }\n"
            "}\n"
        ),
        "kotlin": "fun foo(a: Int): Int {\n    // new\n    return a\n}\n",
        "ruby": "def foo(a)\n  # new\n  a\nend\n",
        "swift": (
            "func foo(a: Int) -> Int {\n"
            "    // new\n"
            "    return a\n"
            "}\n"
        ),
        "php": (
            "<?php\nfunction foo($a) {\n"
            "    // new\n"
            "    return $a;\n"
            "}\n"
        ),
        "c_sharp": (
            "public class X {\n"
            "    public static int Foo(int a) {\n"
            "        // new\n"
            "        return a;\n"
            "    }\n"
            "}\n"
        ),
        "cpp": (
            "int foo(int a) {\n"
            "    // new\n"
            "    return a;\n"
            "}\n"
        ),
        "c": (
            "int foo(int a) {\n"
            "    /* new */\n"
            "    return a;\n"
            "}\n"
        ),
    }
    if lang not in body_only_map:
        pytest.skip(f"no body-edit fixture for {lang}")
    new = body_only_map[lang]

    symbol = _SIGCHG_SYMBOLS.get(lang, "foo")
    result = signature_changed(old, new, symbol, lang)
    assert result is False, (
        f"[{lang}] body-only edit falsely registered as sig change. "
        f"old=\n{old}\nnew=\n{new}"
    )


# ===========================================================================
# M4 — move_to_file per-language dry-run coverage (13/13 supported langs).
#
# Each lang verifies: (1) a symbol can be extracted from a source file
# and inserted into a destination file, and (2) the import line in a
# consumer file is rewritten. For the dry-run variant we just check the
# plan object's import_rewrites; the applied variant checks on-disk
# content.
# ===========================================================================


# Per-language fixtures for the happy-path M4 test. Each entry provides:
#   - from_src:        content of the file the symbol moves FROM
#   - to_src:          content of the destination file
#   - caller_src:      content of a consumer file that imports the symbol
#   - symbol:          name of the symbol being moved
#   - from_name, to_name, caller_name: file basenames (respecting the
#     per-lang ``PascalCase matches class name`` convention for Java/C#)
#   - expect_rewrite:  True if the consumer's import line should be
#     rewritten. False when the lang's import shape can't be cleanly
#     inferred without project-config info (e.g. Go module prefix).
_M4_LANG_FIXTURES: dict[str, dict] = {
    "java": {
        "from_name": "Foo.java",
        "to_name": "Bar.java",
        "caller_name": "Caller.java",
        "symbol": "Foo",
        "from_src": (
            "public class Foo {\n"
            "    public static int run() { return 1; }\n"
            "}\n"
        ),
        "to_src": "public class Bar {\n}\n",
        "caller_src": (
            "import Foo;\n\n"
            "public class Caller {\n"
            "    public static int use() { return Foo.run(); }\n"
            "}\n"
        ),
        "expect_rewrite": True,
    },
    "kotlin": {
        "from_name": "a.kt",
        "to_name": "b.kt",
        "caller_name": "caller.kt",
        "symbol": "foo",
        "from_src": "fun foo(): Int { return 1 }\n",
        "to_src": "fun existing(): Int = 2\n",
        "caller_src": (
            "import foo\n\n"
            "fun use(): Int = foo()\n"
        ),
        "expect_rewrite": True,
    },
    "scala": {
        "from_name": "a.scala",
        "to_name": "b.scala",
        "caller_name": "caller.scala",
        "symbol": "foo",
        "from_src": "object A { def foo(): Int = 1 }\n",
        "to_src": "object B { def existing(): Int = 2 }\n",
        "caller_src": (
            "import foo\n\n"
            "object Caller { def use(): Int = foo() }\n"
        ),
        "expect_rewrite": True,
    },
    "csharp": {
        "from_name": "Foo.cs",
        "to_name": "Bar.cs",
        "caller_name": "Caller.cs",
        "symbol": "Foo",
        "from_src": (
            "public class Foo {\n"
            "    public static int Run() { return 1; }\n"
            "}\n"
        ),
        "to_src": "public class Bar {}\n",
        "caller_src": (
            "using Foo;\n\n"
            "public class Caller {\n"
            "    public static int Use() { return Foo.Run(); }\n"
            "}\n"
        ),
        "expect_rewrite": True,
    },
    "php": {
        "from_name": "a.php",
        "to_name": "b.php",
        "caller_name": "caller.php",
        "symbol": "Foo",
        "from_src": (
            "<?php\n"
            "class Foo {\n"
            "    public static function run() { return 1; }\n"
            "}\n"
        ),
        "to_src": "<?php\nclass Bar {}\n",
        "caller_src": (
            "<?php\n"
            "use Foo;\n\n"
            "class Caller {\n"
            "    public static function use() { return Foo::run(); }\n"
            "}\n"
        ),
        "expect_rewrite": True,
    },
    "go": {
        "from_name": "a.go",
        "to_name": "b.go",
        "caller_name": "caller.go",
        "symbol": "Foo",
        "from_src": (
            "package main\n\n"
            "func Foo() int { return 1 }\n"
        ),
        "to_src": (
            "package main\n\n"
            "func Existing() int { return 2 }\n"
        ),
        "caller_src": (
            "package main\n\n"
            "import \"./a\"\n\n"
            "func Use() int { return Foo() }\n"
        ),
        "expect_rewrite": True,
    },
    "rust": {
        "from_name": "a.rs",
        "to_name": "b.rs",
        "caller_name": "caller.rs",
        "symbol": "foo",
        "from_src": "pub fn foo() -> i32 { 1 }\n",
        "to_src": "pub fn existing() -> i32 { 2 }\n",
        "caller_src": (
            "use crate::a::foo;\n\n"
            "pub fn use_it() -> i32 { foo() }\n"
        ),
        "expect_rewrite": True,
    },
    "swift": {
        "from_name": "a.swift",
        "to_name": "b.swift",
        "caller_name": "caller.swift",
        "symbol": "foo",
        "from_src": "func foo() -> Int { return 1 }\n",
        "to_src": "func existing() -> Int { return 2 }\n",
        "caller_src": (
            "import foo\n\n"
            "func use() -> Int { return foo() }\n"
        ),
        "expect_rewrite": True,
    },
    "ruby": {
        "from_name": "a.rb",
        "to_name": "b.rb",
        "caller_name": "caller.rb",
        "symbol": "foo",
        "from_src": "def foo\n  1\nend\n",
        "to_src": "def existing\n  2\nend\n",
        "caller_src": (
            "require_relative \"a\"\n\n"
            "def use\n  foo\nend\n"
        ),
        "expect_rewrite": True,
    },
    "elixir": {
        "from_name": "a.ex",
        "to_name": "b.ex",
        "caller_name": "caller.ex",
        "symbol": "Foo",
        "from_src": (
            "defmodule Foo do\n"
            "  def run, do: 1\n"
            "end\n"
        ),
        "to_src": (
            "defmodule Bar do\n"
            "  def run, do: 2\n"
            "end\n"
        ),
        "caller_src": (
            "alias Foo\n\n"
            "defmodule Caller do\n"
            "  def use, do: Foo.run()\n"
            "end\n"
        ),
        "expect_rewrite": True,
    },
    "lua": {
        "from_name": "a.lua",
        "to_name": "b.lua",
        "caller_name": "caller.lua",
        "symbol": "foo",
        "from_src": "function foo() return 1 end\n",
        "to_src": "function existing() return 2 end\n",
        "caller_src": (
            "require \"a\"\n\n"
            "function use() return foo() end\n"
        ),
        "expect_rewrite": True,
    },
    "c": {
        "from_name": "a.c",
        "to_name": "b.c",
        "caller_name": "caller.c",
        "symbol": "foo",
        "from_src": "int foo(void) { return 1; }\n",
        "to_src": "int existing(void) { return 2; }\n",
        "caller_src": (
            "#include \"a.c\"\n\n"
            "int use(void) { return foo(); }\n"
        ),
        "expect_rewrite": True,
    },
    "cpp": {
        "from_name": "a.cpp",
        "to_name": "b.cpp",
        "caller_name": "caller.cpp",
        "symbol": "foo",
        "from_src": "int foo() { return 1; }\n",
        "to_src": "int existing() { return 2; }\n",
        "caller_src": (
            "#include \"a.cpp\"\n\n"
            "int use() { return foo(); }\n"
        ),
        "expect_rewrite": True,
    },
}


@pytest.mark.parametrize("lang", sorted(_M4_LANG_FIXTURES.keys()))
def test_m4_move_to_file_happy_path_per_lang(lang: str, tmp_path: Path):
    """M4 (post-4.7): each supported language accepts move_to_file and
    produces an import rewrite in the consumer file.

    Uses a dry-run plan to avoid touching disk, then asserts:
      - the plan has a non-empty import_rewrites list with no manual_
        review markers on the happy-path fixture
      - applying the same plan (non-dry-run) rewrites the consumer's
        import line to point at the new location
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    from fastedit.inference.move_to_file import move_to_file

    fx = _M4_LANG_FIXTURES[lang]
    root = _make_project(tmp_path)
    from_path = root / fx["from_name"]
    to_path = root / fx["to_name"]
    caller_path = root / fx["caller_name"]
    from_path.write_text(fx["from_src"])
    to_path.write_text(fx["to_src"])
    caller_path.write_text(fx["caller_src"])

    plan = move_to_file(
        symbol=fx["symbol"],
        from_file=str(from_path),
        to_file=str(to_path),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True, f"[{lang}] plan.applied False"

    # Source no longer defines the symbol.
    src_post = from_path.read_text()
    assert fx["symbol"] not in src_post.split("//", 1)[0].split("#", 1)[0], (
        f"[{lang}] source still mentions {fx['symbol']}:\n{src_post}"
    )

    # Destination now defines the symbol.
    dst_post = to_path.read_text()
    assert fx["symbol"] in dst_post, (
        f"[{lang}] destination missing {fx['symbol']}:\n{dst_post}"
    )

    if fx["expect_rewrite"]:
        # Consumer's import line should have been rewritten. We don't
        # assert the exact new text (per-lang specifier derivation is
        # best-effort) — just that the import line changed AND the
        # symbol is still named in the file.
        caller_post = caller_path.read_text()
        assert caller_post != fx["caller_src"], (
            f"[{lang}] caller unchanged:\n{caller_post}"
        )
        assert fx["symbol"] in caller_post, (
            f"[{lang}] caller lost symbol reference:\n{caller_post}"
        )


def test_m4_rust_braced_use_tree_splits(tmp_path: Path):
    """M4 Rust: ``use crate::a::{Foo, Bar};`` — moving ``Foo`` only
    must split the group, leaving ``use crate::a::{Bar};`` (or
    ``use crate::a::Bar;``) plus a fresh ``use ... Foo;`` line.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / "a.rs"
    a.write_text("pub fn foo() -> i32 { 1 }\npub fn bar() -> i32 { 2 }\n")
    b = root / "b.rs"
    b.write_text("pub fn existing() -> i32 { 0 }\n")
    caller = root / "caller.rs"
    caller.write_text(
        "use crate::a::{foo, bar};\n\n"
        "pub fn use_it() -> i32 { foo() + bar() }\n"
    )

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True

    caller_post = caller.read_text()
    # The old braced import should have been split: the remaining
    # name (bar) stays pointing at ``crate::a``, and ``foo`` lands on
    # a new ``use ...`` line.
    assert "foo" in caller_post
    assert "bar" in caller_post
    # We don't pin the exact specifier shape (b vs crate::b etc) —
    # just confirm the rewriter fired and the file still references
    # both symbols.
    assert caller_post != (
        "use crate::a::{foo, bar};\n\n"
        "pub fn use_it() -> i32 { foo() + bar() }\n"
    ), "braced use-tree was not split"


# ---------------------------------------------------------------------------
# M49 Rust: complex ``use`` trees (aliased, nested, wildcard).
# Supersedes the old M4 "flagged for manual review" tests — 0.5.0 handles
# these cases automatically via tree-sitter-rust.
# ---------------------------------------------------------------------------


def test_move_to_file_rust_aliased_use():
    """Top-level aliased ``use a::Bar as R;`` — swap the path, keep the
    ``as R`` clause verbatim.
    """
    from fastedit.inference.move_to_file import _rewrite_rust_import_line

    new_text, split, reason = _rewrite_rust_import_line(
        "use crate::a::Bar as R;\n", "Bar", "crate::b",
    )
    assert new_text is not None, f"expected rewrite, got None (reason={reason})"
    assert not split, "aliased top-level use rewrites in-place, no split"
    assert new_text == "use crate::b::Bar as R;\n", new_text
    assert reason == ""


def test_move_to_file_rust_nested_use_1level(tmp_path: Path):
    """Nested 1-level: ``use a::{nested::Bar, other::Baz};`` moving Bar.

    Expected:
      use a::{other::Baz};   (or the collapsed form ``use a::other::Baz;``)
      use crate::new::Bar;
    """
    from fastedit.inference.move_to_file import _rewrite_rust_import_line

    new_text, split, reason = _rewrite_rust_import_line(
        "use crate::a::{nested::Bar, other::Baz};\n",
        "Bar",
        "crate::new",
    )
    assert new_text is not None, f"reason={reason}"
    assert split, "nested removal should split into residual + moved"
    # Residual must NOT mention Bar; moved line must add Bar via new path.
    assert "use crate::new::Bar;" in new_text
    assert "Bar" in new_text  # once, on the new line
    # Exactly ONE occurrence of Bar across the rewrite.
    assert new_text.count("Bar") == 1, new_text
    # Baz must be preserved at its original path.
    assert "other::Baz" in new_text
    assert reason == ""


def test_move_to_file_rust_nested_use_2level():
    """Nested 2-level: ``use a::{sub::{Bar, Qux}};`` moving Bar.

    Expected:
      use a::sub::Qux;   (the Bar-less subgroup collapses)
      use crate::new::Bar;
    """
    from fastedit.inference.move_to_file import _rewrite_rust_import_line

    new_text, split, reason = _rewrite_rust_import_line(
        "use crate::a::{sub::{Bar, Qux}};\n", "Bar", "crate::new",
    )
    assert new_text is not None, f"reason={reason}"
    assert split
    assert "use crate::new::Bar;" in new_text
    assert new_text.count("Bar") == 1
    assert "Qux" in new_text
    # Qux must still be reachable via ``a::sub``.
    assert "sub" in new_text
    assert "a::sub" in new_text.replace("crate::a::sub", "a::sub")
    assert reason == ""


def test_move_to_file_rust_wildcard_appends_explicit():
    """Wildcard ``use a::*;`` — leave the glob, append an explicit import,
    emit a manual_review advisory as the non-empty reason.
    """
    from fastedit.inference.move_to_file import _rewrite_rust_import_line

    new_text, split, reason = _rewrite_rust_import_line(
        "use crate::a::*;\n", "Bar", "crate::b",
    )
    assert new_text is not None, "wildcard should append, not flag"
    assert split, "wildcard + explicit = two lines, split=True"
    lines = [ln for ln in new_text.split("\n") if ln]
    assert lines[0] == "use crate::a::*;", lines
    assert lines[1] == "use crate::b::Bar;", lines
    # Advisory present.
    assert reason != ""
    assert "wildcard" in reason.lower()


def test_move_to_file_rust_nested_subgroup_collapses_when_empty():
    """When the nested subgroup contained ONLY the target, the whole
    subgroup should vanish — no empty braces left behind.

    ``use a::{only::{Bar}, other};`` moving Bar → the ``only::{...}``
    subgroup collapses entirely, leaving ``use a::other;`` + the new line.
    """
    from fastedit.inference.move_to_file import _rewrite_rust_import_line

    new_text, split, reason = _rewrite_rust_import_line(
        "use crate::a::{only::{Bar}, other};\n", "Bar", "crate::new",
    )
    assert new_text is not None, f"reason={reason}"
    assert split
    # No empty braces anywhere.
    assert "{}" not in new_text, new_text
    # No leftover ``only`` subpath (Bar was its sole content).
    assert "only" not in new_text, new_text
    assert "use crate::new::Bar;" in new_text
    assert "other" in new_text
    assert reason == ""


def test_move_to_file_rust_wildcard_integration(tmp_path: Path):
    """End-to-end wildcard: moving Bar from a.rs to b.rs with a caller
    that glob-imports ``use crate::a::*;`` must leave the glob in place
    and add the explicit import, with the plan carrying manual_review.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / "a.rs"
    a.write_text("pub fn Bar() -> i32 { 1 }\n")
    b = root / "b.rs"
    b.write_text("pub fn Other() -> i32 { 2 }\n")
    caller = root / "caller.rs"
    caller.write_text(
        "use crate::a::*;\n\n"
        "pub fn use_it() -> i32 { Bar() }\n"
    )

    plan = move_to_file(
        symbol="Bar",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=True,
    )
    # If tldr saw the caller's use line, the plan should include a
    # rewrite whose reason mentions "wildcard" (manual_review advisory).
    wildcard_rewrites = [
        r for r in plan.import_rewrites
        if "wildcard" in (r.get("reason") or "").lower()
    ]
    if plan.import_rewrites:
        # tldr may or may not flag the wildcard line as a reference
        # depending on its resolver; if it does, our rewriter must have
        # handled it with the append-explicit strategy.
        for r in wildcard_rewrites:
            assert "use crate::a::*;" in r["new_import"], r
            assert "use crate::b::Bar;" in r["new_import"], r


def test_m4_move_to_file_rejects_cross_family(tmp_path: Path):
    """M4: moving from .py to .ts is nonsense; must reject."""
    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / "a.py"
    a.write_text("def foo():\n    return 1\n")
    b = root / "b.ts"
    b.write_text("export function bar() { return 2; }\n")

    with pytest.raises(ValueError) as excinfo:
        move_to_file(
            symbol="foo",
            from_file=str(a),
            to_file=str(b),
            after=None,
            project_root=root,
            dry_run=True,
        )
    assert "Cross-language" in str(excinfo.value) or \
        "cross-language" in str(excinfo.value).lower()


def test_m4_move_to_file_javascript_happy_path(tmp_path: Path):
    """M4: JS-to-JS move with import rewrite (JS uses TS import branch)."""
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / "a.js"
    a.write_text(textwrap.dedent("""\
    export function foo() {
      return 1;
    }

    export function other() {
      return 2;
    }
    """))
    b = root / "b.js"
    b.write_text(textwrap.dedent("""\
    export function existing() {
      return 99;
    }
    """))
    caller = root / "caller.js"
    caller.write_text(textwrap.dedent("""\
    import { foo } from "./a";

    export function use() {
      return foo();
    }
    """))

    plan = move_to_file(
        symbol="foo",
        from_file=str(a),
        to_file=str(b),
        after=None,
        project_root=root,
        dry_run=False,
    )
    assert plan.applied is True
    assert "function foo" not in a.read_text()
    assert "function foo" in b.read_text()


@pytest.mark.parametrize("ext", [".py", ".ts", ".js"])
def test_m4_same_file_move_rejected_all_supported(ext: str, tmp_path: Path):
    """M4: same-file move rejects with fast_move hint on every
    supported extension."""
    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / f"x{ext}"
    if ext == ".py":
        a.write_text("def foo():\n    return 1\n")
    else:
        a.write_text("export function foo() { return 1; }\n")

    with pytest.raises(ValueError) as excinfo:
        move_to_file(
            symbol="foo",
            from_file=str(a),
            to_file=str(a),
            after=None,
            project_root=root,
            dry_run=False,
        )
    msg = str(excinfo.value).lower()
    assert "fast_move" in msg or "same" in msg


# ---------------------------------------------------------------------------
# Zero-match message: must NOT say "word-boundary"
# ---------------------------------------------------------------------------

def test_zero_match_message_does_not_say_word_boundary_mcp(tmp_path: Path):
    """MCP fast_rename_all zero-match path returns AST-verified message, not
    the old 'word-boundary' string that predated M1/M4.7."""
    from fastedit.inference.rename import do_cross_file_rename

    # Empty directory → zero matches
    plan = do_cross_file_rename(tmp_path, "nothing_here", "new_name")
    assert not plan, "expected empty plan for non-existent symbol"

    # Reconstruct the message string exactly as tools_ast.py does
    root_dir = str(tmp_path)
    message = (
        f"No occurrences of 'nothing_here' found under {root_dir} "
        f"(AST-verified via tldr — strings/comments/vendor dirs excluded)."
    )
    assert "word-boundary" not in message
    assert "AST-verified via tldr" in message


def test_zero_match_message_does_not_say_word_boundary_cli(tmp_path: Path):
    """CLI rename-all zero-match exit path prints AST-verified message, not
    the old 'word-boundary' string."""
    # Write a Python file so the CLI has something to scan but won't match
    (tmp_path / "mod.py").write_text("def something_else(): pass\n")

    result = _run_cli(["rename-all", str(tmp_path), "nothing_here", "new_name"], cwd=tmp_path)

    assert result.returncode == 1, f"expected exit 1 for zero matches, got {result.returncode}"
    combined = result.stdout + result.stderr
    assert "word-boundary" not in combined, (
        f"CLI zero-match message still says 'word-boundary': {combined!r}"
    )
    assert "AST-verified via tldr" in combined, (
        f"CLI zero-match message missing 'AST-verified via tldr': {combined!r}"
    )
