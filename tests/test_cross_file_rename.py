"""Tests for cross-file rename via fastedit.inference.rename.do_cross_file_rename.

Verifies:
- Renames apply to every supported code file under the root.
- node_modules, .git, __pycache__, and other vendor/build dirs are pruned.
- Matches inside strings, comments, and docstrings are skipped (tree-sitter).
- Unsupported file extensions are ignored entirely.
- Binary / unreadable files do not crash the walk.
- Zero-match files are omitted from the plan.
- The function is read-only (callers apply writes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastedit.inference.rename import do_cross_file_rename


def _write(base: Path, rel: str, content: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "src/a.py", 'def target():\n    """docstring: target"""\n    return 1\n')
    _write(tmp_path, "src/sub/b.py", "from a import target\ntarget()\n# call target\n")
    _write(tmp_path, "src/c.ts", 'import { target } from "./a";\ntarget();\n')
    _write(tmp_path, "src/unrelated.py", "def other():\n    return 0\n")
    _write(tmp_path, "README.md", "The `target` function does stuff.\n")
    _write(tmp_path, "node_modules/vendor.py", "def target(): return 'vendored'\n")
    _write(tmp_path, ".git/HEAD", "ref: refs/heads/main\n")
    _write(tmp_path, "__pycache__/cache.py", "def target(): pass\n")
    return tmp_path


def test_renames_across_files_and_languages(repo: Path):
    plan = do_cross_file_rename(repo, "target", "renamed")
    paths = {p.relative_to(repo).as_posix() for p in plan}
    assert paths == {"src/a.py", "src/sub/b.py", "src/c.ts"}


def test_vendor_dirs_are_pruned(repo: Path):
    plan = do_cross_file_rename(repo, "target", "renamed")
    for path in plan:
        parts = path.relative_to(repo).parts
        assert "node_modules" not in parts
        assert ".git" not in parts
        assert "__pycache__" not in parts


def test_unsupported_extensions_ignored(repo: Path):
    plan = do_cross_file_rename(repo, "target", "renamed")
    assert not any(p.suffix == ".md" for p in plan)


def test_strings_and_comments_skipped(repo: Path):
    plan = do_cross_file_rename(repo, "target", "renamed")
    a = plan[repo / "src" / "a.py"]
    new_content, count, skipped = a
    assert count == 1
    assert skipped == 1
    assert "docstring: target" in new_content
    assert "def renamed()" in new_content


def test_zero_match_files_omitted(repo: Path):
    plan = do_cross_file_rename(repo, "target", "renamed")
    assert (repo / "src" / "unrelated.py") not in plan


def test_read_only(repo: Path):
    before = {p: p.read_text() for p in repo.rglob("*") if p.is_file()}
    do_cross_file_rename(repo, "target", "renamed")
    after = {p: p.read_text() for p in repo.rglob("*") if p.is_file()}
    assert before == after, "do_cross_file_rename must not mutate files; callers apply writes"


def test_binary_files_do_not_crash(tmp_path: Path):
    (tmp_path / "ok.py").write_text("def target(): pass\n")
    (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02\xff\xfe")
    plan = do_cross_file_rename(tmp_path, "target", "renamed")
    assert set(plan.keys()) == {tmp_path / "ok.py"}


def test_no_matches_returns_empty_dict(tmp_path: Path):
    (tmp_path / "a.py").write_text("def other(): pass\n")
    plan = do_cross_file_rename(tmp_path, "nonexistent", "replacement")
    assert plan == {}

def test_same_name_is_noop(tmp_path: Path):
    """Renaming X -> X is a no-op; returning a plan that rewrites files would
    waste IO, confuse diffs, and potentially trigger backup churn."""
    (tmp_path / "a.py").write_text("def target(): pass\n")
    plan = do_cross_file_rename(tmp_path, "target", "target")
    assert plan == {}


def test_non_standard_venv_dirs_pruned_by_pyvenv_marker(tmp_path: Path):
    """Any dir containing pyvenv.cfg (PEP 405 venv marker) must be pruned,
    not just the few names in DEFAULT_IGNORE_DIRS. Catches .venv311, myenv,
    virtualenv, etc. which otherwise escape the filter."""
    weird_venv = tmp_path / "my_weirdly_named_venv"
    weird_venv.mkdir()
    (weird_venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (weird_venv / "lib").mkdir()
    (weird_venv / "lib" / "pkg.py").write_text("def target(): return 'evil'\n")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("def target(): return 1\n")

    plan = do_cross_file_rename(tmp_path, "target", "renamed")
    paths = {p.relative_to(tmp_path).as_posix() for p in plan}
    assert paths == {"src/real.py"}, f"expected only src/real.py, got {paths}"


def test_symlinks_are_not_followed(tmp_path: Path):
    """os.walk(followlinks=False) prevents the walker from descending into a
    symlink that points back into the tree, which would otherwise produce
    duplicate plan entries for the same file content."""
    import os

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def target(): pass\n")
    (tmp_path / "sub").mkdir()
    os.symlink(tmp_path / "src", tmp_path / "sub" / "link")

    plan = do_cross_file_rename(tmp_path, "target", "renamed")
    # Only one entry, not two via the symlinked path.
    assert len(plan) == 1
    assert (tmp_path / "src" / "a.py") in plan


def test_kind_filter_class_only(tmp_path: Path):
    """kind_filter='class' must only rename when the target's definition is a class.

    Same-name collisions (a class Foo and a local variable Foo) are lumped by
    tldr into one references group keyed on the dominant definition. The filter
    is satisfied when that definition.kind matches. When it does not (e.g. the
    target resolves to a variable, but the caller asked for class), the plan is
    empty — preventing an accidental variable rename through the class knob.
    """
    # Case A: target defines a class. filter='class' should rename it.
    (tmp_path / "a.py").write_text(
        "class MySymbol:\n    pass\n\nprint(MySymbol)\n",
    )
    plan = do_cross_file_rename(tmp_path, "MySymbol", "RenamedSymbol", kind_filter="class")
    assert (tmp_path / "a.py") in plan, "class rename under kind_filter='class' must apply"
    new_content, count, _ = plan[tmp_path / "a.py"]
    assert "class RenamedSymbol" in new_content
    assert "print(RenamedSymbol)" in new_content
    assert count == 2

    # Case B: filter='function' on a class definition -> empty plan (no-op).
    plan2 = do_cross_file_rename(tmp_path, "MySymbol", "Nope", kind_filter="function")
    assert plan2 == {}, "kind_filter mismatch on definition must yield empty plan"

    # Case C: variable-only definition, filter='class' -> empty plan.
    vpath = tmp_path / "b.py"
    vpath.write_text("my_var = 1\nprint(my_var)\n")
    plan3 = do_cross_file_rename(tmp_path, "my_var", "renamed_var", kind_filter="class")
    assert vpath not in plan3, "variable must not be renamed when filter='class'"


def test_ast_verified_strings_and_comments_skipped(tmp_path: Path):
    """Regression: strings, docstrings, and substring-containing comments must
    never be rewritten. This locks AST-verified behavior — with the tldr-driven
    engine, low-confidence (kind='other') hits from string/comment substrings
    must be filtered out via --min-confidence 0.9."""
    src = (
        'def target():\n'
        '    """docstring mentions target here"""\n'
        '    # this comment contains target as a word\n'
        '    x = "string literal with target inside"\n'
        '    return 1\n'
        '\n'
        'target()\n'
    )
    path = tmp_path / "a.py"
    path.write_text(src)

    plan = do_cross_file_rename(tmp_path, "target", "renamed")
    assert path in plan
    new_content, count, _ = plan[path]

    # Only the def + the final call() site should change — 2 real refs.
    assert count == 2, f"expected 2 real refs, got {count}"
    # AST-verified preservation: docstring, comment, string literal all intact.
    assert "docstring mentions target here" in new_content
    assert "# this comment contains target as a word" in new_content
    assert '"string literal with target inside"' in new_content
    # Real refs are rewritten.
    assert "def renamed()" in new_content


def test_kind_filter_function_only_skips_class_definition(tmp_path: Path):
    """kind_filter='function' must not apply when the target is a class.

    Complements test_kind_filter_class_only: filter is bidirectional and
    defensively rejects mismatched definition kinds both ways."""
    (tmp_path / "a.py").write_text(
        "class Shape:\n    pass\n\nShape()\n",
    )
    plan = do_cross_file_rename(tmp_path, "Shape", "Renamed", kind_filter="function")
    assert plan == {}, "function filter must not match a class definition"


def test_kind_filter_none_renames_everything(tmp_path: Path):
    """kind_filter=None must preserve the default (non-filtered) behavior."""
    (tmp_path / "a.py").write_text("def worker(): pass\nworker()\n")
    plan = do_cross_file_rename(tmp_path, "worker", "labor", kind_filter=None)
    assert (tmp_path / "a.py") in plan
    new_content, count, _ = plan[tmp_path / "a.py"]
    assert count == 2
    assert "def labor()" in new_content


def test_kind_filter_invalid_value_raises(tmp_path: Path):
    """An unrecognized kind_filter value must raise ValueError, not silently
    ignore. Protects against typos like 'func' or 'fn'."""
    (tmp_path / "a.py").write_text("def t(): pass\n")
    with pytest.raises(ValueError, match="kind_filter"):
        do_cross_file_rename(tmp_path, "t", "u", kind_filter="fn")


def test_tldr_driven_rename_rewrites_imports(tmp_path: Path):
    """Import statement references (kind='import') must be rewritten, not
    just direct usages. Regression for cross-module rename propagation."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "core.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "main.py").write_text(
        "from pkg.core import helper\n\nhelper()\n",
    )
    plan = do_cross_file_rename(tmp_path, "helper", "utility")
    paths = {p.relative_to(tmp_path).as_posix() for p in plan}
    assert "pkg/core.py" in paths
    assert "main.py" in paths
    main_content = plan[tmp_path / "main.py"][0]
    assert "from pkg.core import utility" in main_content
    assert "utility()" in main_content


def test_tldr_driven_skipped_count_matches_docstring_mentions(tmp_path: Path):
    """skipped count reflects how many word-boundary hits tldr did NOT verify
    as real references — i.e. hits inside strings/comments/docstrings. This
    is the client-side computation that replaces tldr's (absent) skip count.
    """
    path = tmp_path / "a.py"
    path.write_text(
        'def widget():\n'
        '    """widget widget widget"""  # three mentions in one docstring\n'
        '    return 1\n'
        '\n'
        'widget()\n'
    )
    plan = do_cross_file_rename(tmp_path, "widget", "gadget")
    assert path in plan
    new_content, count, skipped = plan[path]
    # Two real refs: the def and the call.
    assert count == 2
    # Three docstring mentions + zero comment hits (none) = 3 skipped.
    assert skipped == 3, f"expected 3 skipped (docstring hits), got {skipped}"
    assert '"""widget widget widget"""' in new_content
