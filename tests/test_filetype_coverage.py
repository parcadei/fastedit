"""File-type coverage matrix for fastedit.

Proves fastedit handles every file type it claims to, and refuses cleanly
the ones it does not. Three axes:

1. The 13 AST-supported languages: create, edit --replace, rename, delete --
   each asserted byte-exact against a hand-computed expected file, not just
   "exit code 0" or "file is non-empty".
2. Non-parseable text types: create/duplicate/split/join must work; the AST
   verbs (edit --replace, rename, delete) must refuse cleanly.
3. Binary content: refused by create (content-based, not extension-based),
   and refused cleanly by the AST verbs against a pre-existing binary file.

All fixtures are built and compared as bytes (Path.write_bytes/read_bytes)
so a difference in line-ending or encoding can never be silently masked by
text-mode normalisation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    """Invoke the real `python -m fastedit` CLI as a subprocess, binary-safe."""
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def create_file(path: Path, content: bytes) -> subprocess.CompletedProcess:
    """`fastedit create` via stdin -- avoids argv-escaping issues for any byte content."""
    return run_cli("create", str(path), "--content-file", "-", input_bytes=content)


NO_LANGUAGE_SUPPORT_NOTE = b"has no fastedit/tldr language support -- skipped the symbol check"


# ---------------------------------------------------------------------------
# AXIS 1 -- the 13 AST languages: create, edit --replace, rename, delete.
#
# Each fixture is HEADER + TARGET_BLOCK + "\n" + KEEP_BLOCK + FOOTER. This
# makes every expected post-operation byte string mechanically derivable:
#   create  -> header + target        + "\n" + keep               + footer
#   edit    -> header + target_edited + "\n" + keep               + footer
#   rename  -> header + target        + "\n" + keep(renamed once) + footer
#   delete  -> header + keep + footer   (delete also eats exactly one
#                                         trailing blank separator line --
#                                         verified empirically per language)
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, dict[str, str]] = {
    "python": {
        "ext": "py",
        "header": "",
        "footer": "",
        "target": "def target_fn(a, b):\n    return a + b\n",
        "target_edited": "def target_fn(a, b):\n    return a - b\n",
        "keep": "def keep_fn(x):\n    return x * 2\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "javascript": {
        "ext": "js",
        "header": "",
        "footer": "",
        "target": "function target_fn(a, b) {\n    return a + b;\n}\n",
        "target_edited": "function target_fn(a, b) {\n    return a - b;\n}\n",
        "keep": "function keep_fn(x) {\n    return x * 2;\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "typescript": {
        "ext": "ts",
        "header": "",
        "footer": "",
        "target": "function target_fn(a: number, b: number): number {\n    return a + b;\n}\n",
        "target_edited": "function target_fn(a: number, b: number): number {\n    return a - b;\n}\n",
        "keep": "function keep_fn(x: number): number {\n    return x * 2;\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "rust": {
        "ext": "rs",
        "header": "",
        "footer": "",
        "target": "fn target_fn(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        "target_edited": "fn target_fn(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
        "keep": "fn keep_fn(x: i32) -> i32 {\n    x * 2\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "go": {
        "ext": "go",
        "header": "package main\n\n",
        "footer": "",
        "target": "func target_fn(a int, b int) int {\n\treturn a + b\n}\n",
        "target_edited": "func target_fn(a int, b int) int {\n\treturn a - b\n}\n",
        "keep": "func keep_fn(x int) int {\n\treturn x * 2\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "java": {
        "ext": "java",
        "header": "public class Sample {\n",
        "footer": "}\n",
        "target": "    public static int target_fn(int a, int b) {\n        return a + b;\n    }\n",
        "target_edited": "    public static int target_fn(int a, int b) {\n        return a - b;\n    }\n",
        "keep": "    public static int keep_fn(int x) {\n        return x * 2;\n    }\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "c": {
        "ext": "c",
        "header": "",
        "footer": "",
        "target": "int target_fn(int a, int b) {\n    return a + b;\n}\n",
        "target_edited": "int target_fn(int a, int b) {\n    return a - b;\n}\n",
        "keep": "int keep_fn(int x) {\n    return x * 2;\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "cpp": {
        "ext": "cpp",
        "header": "",
        "footer": "",
        "target": "int target_fn(int a, int b) {\n    return a + b;\n}\n",
        "target_edited": "int target_fn(int a, int b) {\n    return a - b;\n}\n",
        "keep": "int keep_fn(int x) {\n    return x * 2;\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "ruby": {
        "ext": "rb",
        "header": "",
        "footer": "",
        "target": "def target_fn(a, b)\n  a + b\nend\n",
        "target_edited": "def target_fn(a, b)\n  a - b\nend\n",
        "keep": "def keep_fn(x)\n  x * 2\nend\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "swift": {
        "ext": "swift",
        "header": "",
        "footer": "",
        "target": "func target_fn(a: Int, b: Int) -> Int {\n    return a + b\n}\n",
        "target_edited": "func target_fn(a: Int, b: Int) -> Int {\n    return a - b\n}\n",
        "keep": "func keep_fn(x: Int) -> Int {\n    return x * 2\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "kotlin": {
        "ext": "kt",
        "header": "",
        "footer": "",
        "target": "fun target_fn(a: Int, b: Int): Int {\n    return a + b\n}\n",
        "target_edited": "fun target_fn(a: Int, b: Int): Int {\n    return a - b\n}\n",
        "keep": "fun keep_fn(x: Int): Int {\n    return x * 2\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "csharp": {
        "ext": "cs",
        "header": "public class Sample\n{\n",
        "footer": "}\n",
        "target": "    public static int target_fn(int a, int b)\n    {\n        return a + b;\n    }\n",
        "target_edited": "    public static int target_fn(int a, int b)\n    {\n        return a - b;\n    }\n",
        "keep": "    public static int keep_fn(int x)\n    {\n        return x * 2;\n    }\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
    "php": {
        "ext": "php",
        "header": "<?php\n\n",
        "footer": "",
        "target": "function target_fn($a, $b) {\n    return $a + $b;\n}\n",
        "target_edited": "function target_fn($a, $b) {\n    return $a - $b;\n}\n",
        "keep": "function keep_fn($x) {\n    return $x * 2;\n}\n",
        "old_name": "keep_fn",
        "new_name": "kept_fn",
    },
}


def _initial_content(spec: dict[str, str]) -> bytes:
    return (spec["header"] + spec["target"] + "\n" + spec["keep"] + spec["footer"]).encode()


def _edited_content(spec: dict[str, str]) -> bytes:
    return (spec["header"] + spec["target_edited"] + "\n" + spec["keep"] + spec["footer"]).encode()


def _renamed_content(spec: dict[str, str]) -> bytes:
    keep_renamed = spec["keep"].replace(spec["old_name"], spec["new_name"], 1)
    return (spec["header"] + spec["target"] + "\n" + keep_renamed + spec["footer"]).encode()


def _deleted_content(spec: dict[str, str]) -> bytes:
    return (spec["header"] + spec["keep"] + spec["footer"]).encode()


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_create_ast_language(lang: str, tmp_path: Path) -> None:
    """`create` writes byte-exact content for each of the 13 AST-supported languages."""
    spec = LANGUAGES[lang]
    target = tmp_path / f"sample.{spec['ext']}"
    content = _initial_content(spec)
    result = create_file(target, content)
    assert result.returncode == 0, result.stderr.decode()
    assert target.read_bytes() == content


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_edit_replace_ast_language(lang: str, tmp_path: Path) -> None:
    """`edit --replace` rewrites only the targeted function; the rest of the file is byte-identical."""
    spec = LANGUAGES[lang]
    target = tmp_path / f"sample.{spec['ext']}"
    create_file(target, _initial_content(spec))
    result = run_cli("edit", str(target), "--replace", "target_fn", "--snippet", spec["target_edited"])
    assert result.returncode == 0, result.stderr.decode()
    assert target.read_bytes() == _edited_content(spec)


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_rename_ast_language(lang: str, tmp_path: Path) -> None:
    """`rename` changes only the AST-verified identifier; every other byte is untouched."""
    spec = LANGUAGES[lang]
    target = tmp_path / f"sample.{spec['ext']}"
    create_file(target, _initial_content(spec))
    result = run_cli("rename", str(target), spec["old_name"], spec["new_name"])
    assert result.returncode == 0, result.stderr.decode()
    assert target.read_bytes() == _renamed_content(spec)


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_delete_ast_language(lang: str, tmp_path: Path) -> None:
    """`delete` removes exactly the targeted function plus one trailing blank line; the kept function is byte-identical."""
    spec = LANGUAGES[lang]
    target = tmp_path / f"sample.{spec['ext']}"
    create_file(target, _initial_content(spec))
    result = run_cli("delete", str(target), "target_fn")
    assert result.returncode == 0, result.stderr.decode()
    assert target.read_bytes() == _deleted_content(spec)


# ---------------------------------------------------------------------------
# AXIS 2 -- non-parseable text types: create + duplicate must work; the AST
# verbs must refuse cleanly (clear message, non-zero exit, file untouched).
# ---------------------------------------------------------------------------

NONPARSEABLE_FILES: dict[str, str] = {
    "csv": "a,b,c\n1,2,3\n",
    "tsv": "a\tb\tc\n1\t2\t3\n",
    # B2 triage: json/md/html/xml/toml/yaml/yml/css/sql moved to
    # B2_NOW_PARSEABLE_FILES below — Step B2 shipped verified grammars for
    # them (hard dependencies), so they are no longer "non-parseable" text.
    "jsonl": '{"a": 1}\n{"a": 2}\n',
    "mdx": "# Title\n\n<Component />\n",
    "xhtml": '<?xml version="1.0"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><body/></html>\n',
    "ini": "[section]\nkey=value\n",
    "cfg": "[section]\nkey=value\n",
    "txt": "just plain text\nsecond line\n",
    "log": "2026-09-13 12:00:00 INFO started\n",
    "r": "x <- 1\nprint(x)\n",
    "tex": "\\documentclass{article}\n\\begin{document}\nhi\n\\end{document}\n",
    "sty": "\\ProvidesPackage{sample}\n",
    "cls": "\\ProvidesClass{sample}\n",
    "scss": "$x: 1;\nbody { color: $x; }\n",
    "geojson": '{"type": "FeatureCollection", "features": []}\n',
    "gltf": '{"asset": {"version": "2.0"}}\n',
    "plist": '<?xml version="1.0"?>\n<plist version="1.0"><dict/></plist>\n',
    "vsproj": '<?xml version="1.0"?>\n<VisualStudioProject></VisualStudioProject>\n',
}

# B2: extensions that USED to be plain "non-parseable" text and now have
# verified grammars (hard dependencies). Their lifecycle differs from the
# non-parseable one in exactly one observable way: `create` detects the
# language and runs the symbol check instead of printing the no-support
# note. The AST verbs still refuse cleanly (these data formats have no
# symbol AST maps yet — per-format AST coverage is B3's golden-matrix
# scope), and the file is never touched.
B2_NOW_PARSEABLE_FILES: dict[str, str] = {
    "json": '{"a": 1, "b": [1, 2, 3]}\n',
    "md": "# Title\n\nSome text.\n",
    "html": "<html><body><p>hi</p></body></html>\n",
    "xml": "<root><item>1</item></root>\n",
    "toml": '[section]\nkey = "value"\n',
    "yaml": "a: 1\nb: 2\n",
    "yml": "a: 1\nb: 2\n",
    "sql": "SELECT * FROM foo;\n",
    "css": "body { color: red; }\n",
}


@pytest.mark.parametrize("ext", sorted(NONPARSEABLE_FILES))
def test_nonparseable_type_full_lifecycle(ext: str, tmp_path: Path) -> None:
    """create+duplicate succeed byte-exactly and the AST verbs refuse cleanly, for every non-AST text extension."""
    content = NONPARSEABLE_FILES[ext].encode()
    target = tmp_path / f"sample.{ext}"

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    assert NO_LANGUAGE_SUPPORT_NOTE in create_result.stdout

    dup = tmp_path / f"sample_dup.{ext}"
    dup_result = run_cli("duplicate", str(target), str(dup))
    assert dup_result.returncode == 0, dup_result.stderr.decode()
    assert dup.read_bytes() == content

    edit_result = run_cli("edit", str(target), "--replace", "foo", "--snippet", "bar")
    assert edit_result.returncode != 0
    assert b"Error" in edit_result.stderr

    rename_result = run_cli("rename", str(target), "foo", "bar")
    assert rename_result.returncode != 0
    assert b"Error" in rename_result.stderr

    delete_result = run_cli("delete", str(target), "foo")
    assert delete_result.returncode != 0
    assert b"Error" in delete_result.stderr

    assert target.read_bytes() == content


@pytest.mark.parametrize("ext", sorted(B2_NOW_PARSEABLE_FILES))
def test_b2_parseable_type_full_lifecycle(ext: str, tmp_path: Path) -> None:
    """B2 lifecycle for formats that gained verified grammars: create and
    duplicate still work byte-exactly, but `create` now DETECTS the
    language (no no-support note), and the AST verbs still refuse cleanly
    — no symbol AST maps exist for these data formats yet (B3's per-format
    golden matrix extends them) — leaving the file untouched."""
    content = B2_NOW_PARSEABLE_FILES[ext].encode()
    target = tmp_path / f"sample.{ext}"

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    # B2: the language IS detected now — no "no language support" note.
    assert NO_LANGUAGE_SUPPORT_NOTE not in create_result.stdout

    dup = tmp_path / f"sample_dup.{ext}"
    dup_result = run_cli("duplicate", str(target), str(dup))
    assert dup_result.returncode == 0, dup_result.stderr.decode()
    assert dup.read_bytes() == content

    edit_result = run_cli("edit", str(target), "--replace", "foo", "--snippet", "bar")
    assert edit_result.returncode != 0
    assert b"Error" in edit_result.stderr

    rename_result = run_cli("rename", str(target), "foo", "bar")
    assert rename_result.returncode != 0
    assert b"Error" in rename_result.stderr

    delete_result = run_cli("delete", str(target), "foo")
    assert delete_result.returncode != 0
    assert b"Error" in delete_result.stderr

    assert target.read_bytes() == content


DOTFILES: dict[str, str] = {
    ".gitignore": "*.pyc\n__pycache__/\n",
    ".semgrepignore": "vendor/\n",
}


@pytest.mark.parametrize("name", sorted(DOTFILES))
def test_dotfile_type_full_lifecycle(name: str, tmp_path: Path) -> None:
    """Dotfiles with no conventional extension (.gitignore, .semgrepignore) create fine and refuse AST verbs cleanly."""
    content = DOTFILES[name].encode()
    target = tmp_path / name

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    assert NO_LANGUAGE_SUPPORT_NOTE in create_result.stdout

    edit_result = run_cli("edit", str(target), "--replace", "foo", "--snippet", "bar")
    assert edit_result.returncode != 0
    assert b"Error" in edit_result.stderr
    assert target.read_bytes() == content


def test_create_extensionless_text_file(tmp_path: Path) -> None:
    """A file with no extension at all still creates byte-exactly and refuses AST verbs cleanly."""
    content = b"just some plain text\nsecond line\n"
    target = tmp_path / "noext"

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    assert NO_LANGUAGE_SUPPORT_NOTE in create_result.stdout

    rename_result = run_cli("rename", str(target), "foo", "bar")
    assert rename_result.returncode != 0
    assert b"Error" in rename_result.stderr
    assert target.read_bytes() == content


def test_json_content_named_py_extension_creates_as_text(tmp_path: Path) -> None:
    """Content-based detection: JSON content saved with a .py name is created (it's text), not refused.

    The extension IS AST-supported (.py), so create's symbol step actually
    parses it as Python -- and finds it syntactically valid with zero
    functions, which is a different code path from the generic
    "no language support" extensions above.
    """
    content = b'{"a": 1}\n'
    target = tmp_path / "misleading.py"

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    assert NO_LANGUAGE_SUPPORT_NOTE not in create_result.stdout

    edit_result = run_cli("edit", str(target), "--replace", "foo", "--snippet", "bar")
    assert edit_result.returncode != 0
    assert b"Symbol 'foo' not found" in edit_result.stderr
    assert b"Available: []" in edit_result.stderr


def test_python_content_named_dat_extension_creates_and_skips_symbol_check(tmp_path: Path) -> None:
    """Extension-based detection: valid Python content saved as .dat still creates fine, but the
    language/symbol step is skipped because the language map is keyed by extension, not content.
    """
    content = b"def foo():\n    return 1\n"
    target = tmp_path / "misleading.dat"

    create_result = create_file(target, content)
    assert create_result.returncode == 0, create_result.stderr.decode()
    assert target.read_bytes() == content
    assert NO_LANGUAGE_SUPPORT_NOTE in create_result.stdout

    # rename strictly requires tldr's extension-keyed language map and refuses
    # even though the content is genuinely valid, symbol-bearing Python.
    rename_result = run_cli("rename", str(target), "foo", "bar")
    assert rename_result.returncode != 0
    assert b"Error" in rename_result.stderr
    assert target.read_bytes() == content


# ---------------------------------------------------------------------------
# split / join round-trips on non-AST text types.
# ---------------------------------------------------------------------------

def test_split_join_roundtrip_csv_by_rows(tmp_path: Path) -> None:
    """`split --rows` then `join` reconstructs a CSV byte-for-byte, header repeated per chunk."""
    content = b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
    target = tmp_path / "data.csv"
    create_file(target, content)
    out_dir = tmp_path / "parts"
    split_result = run_cli("split", str(target), "--out", str(out_dir), "--rows", "2")
    assert split_result.returncode == 0, split_result.stderr.decode()
    joined = tmp_path / "joined.csv"
    join_result = run_cli("join", str(out_dir), "-o", str(joined))
    assert join_result.returncode == 0, join_result.stderr.decode()
    assert joined.read_bytes() == content


def test_split_join_roundtrip_json_by_element(tmp_path: Path) -> None:
    """`split --by element` then `join` reconstructs a JSON array byte-for-byte."""
    content = b'[{"a": 1}, {"a": 2}, {"a": 3}]\n'
    target = tmp_path / "arr.json"
    create_file(target, content)
    out_dir = tmp_path / "parts"
    split_result = run_cli("split", str(target), "--out", str(out_dir), "--by", "element")
    assert split_result.returncode == 0, split_result.stderr.decode()
    joined = tmp_path / "joined.json"
    join_result = run_cli("join", str(out_dir), "-o", str(joined))
    assert join_result.returncode == 0, join_result.stderr.decode()
    assert joined.read_bytes() == content


def test_split_join_roundtrip_markdown_by_heading(tmp_path: Path) -> None:
    """`split --by heading` then `join` reconstructs a Markdown doc byte-for-byte."""
    content = b"# Title\n\n## Section One\ncontent one\n\n## Section Two\ncontent two\n"
    target = tmp_path / "doc.md"
    create_file(target, content)
    out_dir = tmp_path / "parts"
    split_result = run_cli("split", str(target), "--out", str(out_dir), "--by", "heading", "--level", "2")
    assert split_result.returncode == 0, split_result.stderr.decode()
    joined = tmp_path / "joined.md"
    join_result = run_cli("join", str(out_dir), "-o", str(joined))
    assert join_result.returncode == 0, join_result.stderr.decode()
    assert joined.read_bytes() == content


def test_split_join_roundtrip_generic_lines_on_yaml(tmp_path: Path) -> None:
    """`split --lines` works on any text file (here YAML) and `join` reconstructs it byte-for-byte."""
    content = b"a: 1\nb: 2\nc: 3\nd: 4\n"
    target = tmp_path / "cfg.yaml"
    create_file(target, content)
    out_dir = tmp_path / "parts"
    split_result = run_cli("split", str(target), "--out", str(out_dir), "--lines", "2")
    assert split_result.returncode == 0, split_result.stderr.decode()
    joined = tmp_path / "joined.yaml"
    join_result = run_cli("join", str(out_dir), "-o", str(joined))
    assert join_result.returncode == 0, join_result.stderr.decode()
    assert joined.read_bytes() == content


# ---------------------------------------------------------------------------
# AXIS 3 -- binary content: refused by `create` (content-based, not
# extension-based); AST verbs refuse cleanly against an existing binary file.
# ---------------------------------------------------------------------------

BINARY_FIXTURES: dict[str, tuple[str, bytes]] = {
    "docx_zip_header": ("out.docx", b"PK\x03\x04" + b"\x00" * 4 + b"binaryjunkdata"),
    "binary_plist": ("out.plist", b"bplist00" + bytes([1, 2, 3, 0, 0, 0]) + b"binarydata"),
    "png_header": ("out.png", b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"),
    "nul_byte_in_first_8kb": ("out.txt", b"hello" + b"\x00" + b"world" * 100),
    "utf16le_without_bom": ("out.txt", "hello".encode("utf-16-le")),
}


@pytest.mark.parametrize("case", sorted(BINARY_FIXTURES))
def test_create_refuses_binary_content(case: str, tmp_path: Path) -> None:
    """`create` refuses binary content with a clear message, non-zero exit, and creates no file."""
    name, content = BINARY_FIXTURES[case]
    target = tmp_path / name
    result = create_file(target, content)
    assert result.returncode != 0
    assert b"refusing to create a binary file" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("case", sorted(BINARY_FIXTURES))
def test_ast_verbs_refuse_cleanly_on_existing_binary_file(case: str, tmp_path: Path) -> None:
    """edit/rename/delete all refuse cleanly against a pre-existing binary file (written directly, bypassing `create`)."""
    name, content = BINARY_FIXTURES[case]
    target = tmp_path / name
    target.write_bytes(content)

    edit_result = run_cli("edit", str(target), "--replace", "foo", "--snippet", "bar")
    assert edit_result.returncode != 0
    assert b"Error" in edit_result.stderr

    rename_result = run_cli("rename", str(target), "foo", "bar")
    assert rename_result.returncode != 0
    assert b"Error" in rename_result.stderr

    delete_result = run_cli("delete", str(target), "foo")
    assert delete_result.returncode != 0
    assert b"Error" in delete_result.stderr

    # None of the refused AST verbs may have corrupted the pre-existing binary fixture.
    assert target.read_bytes() == content
