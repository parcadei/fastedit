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
