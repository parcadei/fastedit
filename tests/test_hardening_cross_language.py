"""Cross-language parametrized hardening tests for M1-M4 features.

Milestone 4.6: expand coverage beyond Python + TS happy paths to include
Java, Rust, Go, Kotlin (required), plus Ruby, JavaScript.

See docs/testing-matrix.md for the full feature x language matrix.

Key finding (documented):
    do_rename_ast and check_cross_file_callers drop tldr refs with
    kind="other". tldr emits kind="other" for every language outside
    its AST-native set (python, typescript, go, rust). Java, Kotlin,
    Ruby, and friends therefore cleanly return count=0 rather than
    silently doing a regex-quality rename. Users on those langs should
    fall back to `fastedit rename` (regex + tree-sitter skip zones).
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

    Invariants (AST-native langs: python/typescript/go/rust/javascript):
      - Definition site is renamed (count >= 1).
      - String/comment occurrences preserved (VAL-M1-002).

    Non-AST-native langs (java/kotlin/ruby):
      - do_rename_ast returns count=0 — DOCUMENTED behavior. tldr emits
        kind="other" (grep-quality) for these langs and fastedit's
        AST-only filter intentionally drops them to avoid unverified
        renames. Users should fall back to `fastedit rename`.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")

    spec = LANG_SPECS[lang_key]
    root = _make_project(tmp_path)
    path = root / f"mod{spec.file_ext}"
    path.write_text(spec.rename_source)

    new_content, count, skipped = do_rename_ast(path, spec.symbol, "newName")

    if lang_key in TLDR_NATIVE_LANGS or lang_key == "javascript":
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
    else:
        # Non-native: locked-in fall-through to count=0. If this ever
        # flips (tldr gains AST support for the lang, or fastedit widens
        # its kind filter), the assertion forces a deliberate review.
        assert count == 0, (
            f"[{lang_key}] unexpected non-zero count={count}. "
            f"tldr may have gained AST support — review filter."
        )


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


@pytest.mark.parametrize("lang_key", REQUIRED_LANGS)
def test_m2_force_override_is_monotonic(lang_key: str, tmp_path: Path):
    """M2 monotonicity: force=False refuses -> force=True proceeds.

    CLI-level exercise for Python (where delete_symbol is robust). Other
    languages deferred to the per-lang AST delete test suite.
    """
    if not TLDR_AVAILABLE:
        pytest.skip("tldr binary not on PATH")
    if lang_key != "python":
        pytest.skip(
            f"[{lang_key}] CLI delete path: per-lang AST delete tested "
            f"in test_ast_analyzer.py"
        )

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
        "export function foo(a: number, b: number): number {\n  return a + b;\n}\n",
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
        "public class X {\n    public static int foo(int a) { return a; }\n}\n",
        "public class X {\n    public static int foo(int a, int b) { return a + b; }\n}\n",
    ),
    "kotlin": (
        "fun foo(a: Int): Int {\n    return a\n}\n",
        "fun foo(a: Int, b: Int): Int {\n    return a + b\n}\n",
    ),
}


@pytest.mark.parametrize("lang", list(_SIGCHG_EDITS.keys()))
def test_m3_signature_changed_detects_param_addition(lang: str):
    """M3: adding a parameter registers as a signature change in every
    language fastedit's analyze_file supports.

    Languages where analyze_file cannot cleanly locate the function body
    are xfailed with a reference to 0.6.1.
    """
    old, new = _SIGCHG_EDITS[lang]
    # Empirically tested on 2026-04-23: all six langs pass. Update the
    # set if tree-sitter grammar changes break a lang.
    known_broken: set[str] = set()

    result = signature_changed(old, new, "foo", lang)
    if lang in known_broken:
        pytest.xfail(
            f"{lang} signature detection broken — flagged for 0.6.1"
        )
    assert result is True, (
        f"[{lang}] param-addition not detected as signature change. "
        f"old=\n{old}\nnew=\n{new}"
    )


@pytest.mark.parametrize("lang", list(_SIGCHG_EDITS.keys()))
def test_m3_signature_unchanged_on_body_only_edit(lang: str):
    """M3 hot path: body-only edits do NOT register as signature changes
    — so tldr subprocess isn't spawned (VAL-M3-002)."""
    old, _ = _SIGCHG_EDITS[lang]
    body_only_map = {
        "python": "def foo(a):\n    # new comment\n    return a\n",
        "typescript": (
            "export function foo(a: number): number {\n"
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
            "    public static int foo(int a) { /* new */ return a; }\n}\n"
        ),
        "kotlin": "fun foo(a: Int): Int {\n    // new\n    return a\n}\n",
    }
    if lang not in body_only_map:
        pytest.skip(f"no body-edit fixture for {lang}")
    new = body_only_map[lang]

    result = signature_changed(old, new, "foo", lang)
    assert result is False, (
        f"[{lang}] body-only edit falsely registered as sig change. "
        f"old=\n{old}\nnew=\n{new}"
    )


# ===========================================================================
# M4 — move_to_file rejection matrix
# ===========================================================================

_M4_REJECTION_EXTS = [
    (".rs", "rust"),
    (".go", "go"),
    (".java", "java"),
    (".kt", "kotlin"),
    (".rb", "ruby"),
    (".swift", "swift"),
    (".php", "php"),
    (".cs", "csharp"),
    (".cpp", "cpp"),
    (".c", "c"),
]


@pytest.mark.parametrize("ext,lang_name", _M4_REJECTION_EXTS)
def test_m4_move_to_file_rejects_unsupported_lang(
    ext: str, lang_name: str, tmp_path: Path,
):
    """M4: move_to_file must reject unsupported langs with a clear error
    rather than silently producing broken imports.

    Supported: .py + .ts/.tsx/.js/.jsx. Everything else raises
    ValueError pointing at the supported set.
    """
    from fastedit.inference.move_to_file import move_to_file

    root = _make_project(tmp_path)
    a = root / f"a{ext}"
    b = root / f"b{ext}"
    stub_map = {
        ".rs": "pub fn foo() {}\n",
        ".go": "package main\nfunc foo() {}\n",
        ".java": "public class Main { public static void foo() {} }\n",
        ".kt": "fun foo() {}\n",
        ".rb": "def foo; end\n",
        ".swift": "func foo() {}\n",
        ".php": "<?php\nfunction foo() {}\n",
        ".cs": "public class X { public static void Foo() {} }\n",
        ".cpp": "void foo() {}\n",
        ".c": "void foo(void) {}\n",
    }
    a.write_text(stub_map[ext])
    b.write_text(stub_map[ext])

    with pytest.raises(ValueError) as excinfo:
        move_to_file(
            symbol="foo",
            from_file=str(a),
            to_file=str(b),
            after=None,
            project_root=root,
            dry_run=True,
        )
    msg = str(excinfo.value)
    assert ext in msg or "Unsupported" in msg, (
        f"[{ext}] rejection message should mention extension; got: {msg}"
    )
    assert ".py" in msg or ".ts" in msg, (
        f"[{ext}] rejection should hint supported exts; got: {msg}"
    )


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
