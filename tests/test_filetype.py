"""Tests for fastedit.filetype.is_text_file / detect_file_type.

Covers both required lists from the phase-1 spec: real content for every
text format that must be detected as text regardless of extension, and
real (programmatically-built, never committed as blobs) binary payloads
that must be detected as binary regardless of extension.
"""

from __future__ import annotations

import plistlib
import zipfile
import zlib
from io import BytesIO
from pathlib import Path

import pytest

from fastedit.filetype import detect_file_type, is_text_file

# Real content for every format the spec requires to be classified as text.
# The detector never looks at a name, so one representative snippet per
# format is enough to prove the content-sniffing path handles it.
TEXT_SAMPLES: dict[str, str] = {
    "csv": "a,b,c\n1,2,3\n",
    "mdx": "# Title\n\nSome **MDX** content with <Component />.\n",
    "html": "<!DOCTYPE html>\n<html><body>hi</body></html>\n",
    "xhtml": (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body/></html>\n'
    ),
    "xml": '<?xml version="1.0" encoding="UTF-8"?>\n<root><child/></root>\n',
    "json": '{"a": 1, "b": [true, null]}\n',
    "jsonl": '{"a": 1}\n{"b": 2}\n',
    "txt": "just plain text\nwith two lines\n",
    "toml": '[section]\nkey = "value"\n',
    "ini": "[section]\nkey=value\n",
    "cfg": "[section]\nkey=value\n",
    "plist_xml": (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict/></plist>\n'
    ),
    "vsproj": (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Project ToolsVersion="4.0" '
        'xmlns="http://schemas.microsoft.com/developer/msbuild/2003"></Project>\n'
    ),
    "gitignore": "*.pyc\n__pycache__/\n.venv/\n",
    "extensionless_script": "#!/bin/sh\necho hi\n",
    # The 13 languages fastedit's tree-sitter layer parses.
    "python": "def f(x):\n    return x + 1\n",
    "javascript": "function f(x) {\n  return x + 1;\n}\n",
    "typescript": "function f(x: number): number {\n  return x + 1;\n}\n",
    "rust": "fn f(x: i32) -> i32 {\n    x + 1\n}\n",
    "go": "package main\n\nfunc f(x int) int {\n\treturn x + 1\n}\n",
    "java": "class A {\n    int f(int x) { return x + 1; }\n}\n",
    "c": "int f(int x) {\n    return x + 1;\n}\n",
    "cpp": "int f(int x) {\n    return x + 1;\n}\n",
    "ruby": "def f(x)\n  x + 1\nend\n",
    "swift": "func f(_ x: Int) -> Int {\n    return x + 1\n}\n",
    "kotlin": "fun f(x: Int): Int {\n    return x + 1\n}\n",
    "csharp": "class A {\n    int F(int x) { return x + 1; }\n}\n",
    "php": "<?php\nfunction f($x) {\n    return $x + 1;\n}\n",
}


@pytest.mark.parametrize("label", sorted(TEXT_SAMPLES))
def test_known_text_formats_are_detected_as_text_by_content(label: str) -> None:
    """Real content for every required text format is classified as text, by content alone."""
    result = detect_file_type(TEXT_SAMPLES[label].encode("utf-8"))
    assert result.is_text, f"{label}: expected text, got binary ({result.reason})"


@pytest.mark.parametrize("suffix", [".dat", ".vsproj", ".gitignore", ""])
def test_text_content_is_still_text_under_a_misleading_extension(tmp_path: Path, suffix: str) -> None:
    """A plain-text file is detected as text via its path even under a non-.txt/no extension."""
    path = tmp_path / f"renamed{suffix}"
    path.write_text(TEXT_SAMPLES["python"], encoding="utf-8")
    result = is_text_file(path)
    assert result.is_text


def test_empty_file_is_text() -> None:
    """An empty file is trivially text (nothing to sniff, nothing to refuse)."""
    result = detect_file_type(b"")
    assert result.is_text
    assert "empty" in result.reason


@pytest.mark.parametrize(
    "bom,payload",
    [
        (b"\xef\xbb\xbf", b"hello utf-8-sig\n"),
        (b"\xff\xfe", "hello utf-16-le\n".encode("utf-16-le")),
        (b"\xfe\xff", "hello utf-16-be\n".encode("utf-16-be")),
        (b"\xff\xfe\x00\x00", "hello utf-32-le\n".encode("utf-32-le")),
        (b"\x00\x00\xfe\xff", "hello utf-32-be\n".encode("utf-32-be")),
    ],
)
def test_bom_prefixed_content_is_text(bom: bytes, payload: bytes) -> None:
    """UTF-8/16/32 BOM-prefixed content is honoured as text, per spec."""
    # payload already carries its own BOM-less encoded bytes; the BOM itself
    # is prepended separately here to test detection of the marker alone
    # against content that decodes cleanly under the matching encoding.
    encoding = {
        b"\xef\xbb\xbf": "utf-8-sig",
        b"\xff\xfe": "utf-16",
        b"\xfe\xff": "utf-16",
        b"\xff\xfe\x00\x00": "utf-32",
        b"\x00\x00\xfe\xff": "utf-32",
    }[bom]
    data = bom + payload
    result = detect_file_type(data)
    assert result.is_text, f"BOM {bom!r} ({encoding}): expected text, got binary ({result.reason})"


def test_nul_byte_in_first_8kb_is_binary() -> None:
    """A NUL byte anywhere in the first 8KB is refused as binary, regardless of the rest."""
    data = b"looks like text at first\x00but has a NUL byte"
    result = detect_file_type(data)
    assert not result.is_text
    assert "NUL" in result.reason


def test_high_control_byte_ratio_is_binary() -> None:
    """Content that isn't valid UTF-8/UTF-16 and is mostly control bytes is binary."""
    # 0x80-0xFF bytes decode under latin-1 but a high proportion of C0 control
    # bytes below 0x20 (excluding tab/LF/CR/FF) is the fallback binary signal.
    data = bytes([i % 0x1F for i in range(200)])
    result = detect_file_type(data)
    assert not result.is_text


def test_low_control_byte_ratio_non_utf8_is_text() -> None:
    """Non-UTF-8 content with only a handful of control bytes falls back to text."""
    data = "café über naïve".encode("latin-1") + b"\x01"
    result = detect_file_type(data)
    assert result.is_text


def _make_zip_bytes() -> bytes:
    """Build a real minimal zip container in memory (no committed binary blob)."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<document>" + ("x" * 500) + "</document>")
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


@pytest.mark.parametrize("label", ["docx", "xlsx", "pptx", "odt"])
def test_zip_container_office_formats_are_binary(label: str) -> None:
    """A real zip container (the shared format behind docx/xlsx/pptx/odt) is binary."""
    data = _make_zip_bytes()
    result = detect_file_type(data)
    assert not result.is_text, f"{label}: expected binary, got text ({result.reason})"


def test_binary_plist_is_binary() -> None:
    """A real bplist00 payload (built with stdlib plistlib) is detected as binary."""
    data = plistlib.dumps({"key": "value", "n": 42}, fmt=plistlib.FMT_BINARY)
    assert data.startswith(b"bplist00")
    result = detect_file_type(data)
    assert not result.is_text


def _make_png_bytes() -> bytes:
    """Build a real, minimal, valid PNG file in memory (8x8 IHDR + empty IDAT)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", (8).to_bytes(4, "big") + (8).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0]))
    idat = chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 24))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def test_image_bytes_are_binary() -> None:
    """A real minimal PNG (magic bytes + valid chunks) is detected as binary."""
    data = _make_png_bytes()
    result = detect_file_type(data)
    assert not result.is_text


def test_is_text_file_accepts_bytes_directly() -> None:
    """is_text_file works on raw bytes, not just a path, for both outcomes."""
    assert is_text_file(b"plain ascii text\n").is_text
    assert not is_text_file(_make_png_bytes()).is_text


def test_is_text_file_reads_a_path(tmp_path: Path) -> None:
    """is_text_file(path) reads and sniffs the file's own bytes."""
    binary_path = tmp_path / "image.bin"
    binary_path.write_bytes(_make_png_bytes())
    assert not is_text_file(binary_path).is_text

    text_path = tmp_path / "readme"
    text_path.write_text("hello\n", encoding="utf-8")
    assert is_text_file(text_path).is_text
