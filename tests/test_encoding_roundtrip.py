"""Step 15 (B21, B23): encoding-safe round-trip for every edit path.

The bugs:

* **B21** — source files were read with ``errors="replace"``
  (``tools_edit.py``, ``symbols.py``, ``cli.py``): every undecodable byte
  became U+FFFD and was written back as EF BF BD — permanent corruption of
  lines the edit never touched. A latin-1 file lost its é bytes; a UTF-16
  file was mojibake-fed to tree-sitter.
* **B22 bridge** — ``_atomic_write``'s backup save decoded the original with
  UTF-8, raising on latin-1 content and breaking the write step AFTER the
  merge had already succeeded.
* **B23** — ``_atomic_write`` restored a BOM but never the original
  encoding: it always wrote UTF-8.

Contract locked down here:

1. A latin-1 file survives an edit with byte-exact content OUTSIDE the
   edited span, still decodes as latin-1, and contains no EF BF BD
   replacement-character bytes anywhere.
2. A latin-1 file whose edited span itself contains the accented byte (the
   snippet restates it) round-trips that byte exactly — 0xE9 stays 0xE9,
   it is not re-encoded as UTF-8's 0xC3 0xA9.
3. A latin-1 edit's backup/undo round-trips: ``fastedit undo`` restores the
   original bytes exactly (the B22 bridge through ``BackupStore``).
4. A UTF-8 file with a BOM keeps its BOM after an edit (may already have
   passed — it pins the behavior through the new io_utils path).
5. A UTF-16 file is REFUSED with a clear error (naming UTF-16, saying the
   file was not modified) — not mojibake-fed to the merge, nothing written.
6. A binary file is refused the same way.
7. An ASCII file round-trips unchanged (control).

The latin-1/BOM/ASCII round-trips go through the REAL edit path end-to-end
(the real CLI process: read → deterministic merge → parse gate → atomic
write, the same harness tests/test_encoding_matrix.py uses). The refusal
contract is tested both directly against ``fastedit.io_utils`` helpers and
through the MCP ``fast_edit`` tool with ``chunked_merge`` stubbed to explode
if the merge is ever reached — proving an unsupported encoding never gets
that far. No model, no network.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from fastedit.io_utils import UnsupportedEncodingError, read_source, write_source
from fastedit.mcp import tools_edit
from fastedit.mcp.backup import BackupStore, _atomic_write

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Invoke the real fastedit CLI (python -m fastedit) as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # the return code IS the assertion target
        env=env,
    )


# --- fixture byte constants (every one built from bytes / explicit codecs) ---

# é is a single 0xE9 byte in latin-1 and a 0xC3 0xA9 pair in UTF-8: any
# encoding rewrite of the file is visible in the bytes.
LATIN1_A = "def a():\n    x = 'café'\n    return x\n"
LATIN1_B = "def b():\n    return 2\n"
LATIN1_SRC = (LATIN1_A + "\n\n" + LATIN1_B).encode("latin-1")
assert LATIN1_SRC.count(b"\xe9") == 1
assert b"\xef\xbf\xbd" not in LATIN1_SRC

ASCII_SRC = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
BOM_ASCII_SRC = b"\xef\xbb\xbf" + ASCII_SRC
UTF16LE_SRC = codecs.BOM_UTF16_LE + "def a():\n    return 1\n".encode("utf-16-le")
UTF16BE_SRC = codecs.BOM_UTF16_BE + "def a():\n    return 1\n".encode("utf-16-be")
# Invalid UTF-8 AND dense with NULs: filetype.py's NUL heuristic classifies
# it as binary long before any codec question arises.
BINARY_SRC = b"def a():\x00\x01\x02\xff\xfe\x00bin\x00ary\x9c\xc5\x8d\x9d"


# ---------------------------------------------------------------------------
# (a)/(e) latin-1 and ASCII round-trips through the REAL edit path end-to-end
# ---------------------------------------------------------------------------


class TestLatin1RoundTrip:
    def test_edit_outside_the_span_leaves_untouched_bytes_identical(
        self, tmp_path: Path
    ) -> None:
        """Editing b leaves a's bytes — including the 0xE9 é byte — exact."""
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)
        prefix = LATIN1_SRC[: LATIN1_SRC.index(b"def b")]

        result = run_cli(
            "edit", str(target), "--replace", "b",
            "--snippet", "def b():\n    return 20\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        # Untouched region byte-exact — the B21 failure mode was exactly
        # these bytes being rewritten as EF BF BD.
        assert after[: len(prefix)] == prefix
        assert after == prefix + b"def b():\n    return 20\n"
        # Still latin-1: the accented byte survived as itself.
        assert after.count(b"\xe9") == 1
        assert b"\xef\xbf\xbd" not in after  # no U+FFFD written back
        after.decode("latin-1")  # must not raise

    def test_edit_touching_the_accented_line_round_trips_the_byte(
        self, tmp_path: Path
    ) -> None:
        """B23: the write re-encodes with the SAME codec that read the file.

        The snippet restates the é line inside the edited span; the byte
        must come back as 0xE9 (latin-1), never as UTF-8's 0xC3 0xA9.
        """
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    x = 'café'\n    return 42\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after == (
            "def a():\n    x = 'café'\n    return 42\n\n\ndef b():\n    return 2\n"
        ).encode("latin-1")
        assert after.count(b"\xe9") == 1
        assert after.count(b"\xc3\xa9") == 0
        assert b"\xef\xbf\xbd" not in after

    def test_undo_restores_the_original_latin1_bytes_exactly(
        self, tmp_path: Path
    ) -> None:
        """B22 bridge: the backup of a latin-1 file must not raise on save
        (it used to, killing the write AFTER the merge) and ``undo`` must
        restore the original bytes exactly."""
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)

        result = run_cli(
            "edit", str(target), "--replace", "b",
            "--snippet", "def b():\n    return 20\n",
        )
        assert result.returncode == 0, result.stderr
        assert target.read_bytes() != LATIN1_SRC

        undone = run_cli("undo", str(target))
        assert undone.returncode == 0, undone.stderr
        assert target.read_bytes() == LATIN1_SRC
        assert target.read_bytes().count(b"\xe9") == 1


class TestAsciiControlRoundTrip:
    def test_ascii_file_edit_is_byte_exact_outside_the_span(
        self, tmp_path: Path
    ) -> None:
        """Control: a plain ASCII file round-trips byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(ASCII_SRC)

        result = run_cli(
            "edit", str(target), "--replace", "b",
            "--snippet", "def b():\n    return 20\n",
        )
        assert result.returncode == 0, result.stderr

        prefix = ASCII_SRC[: ASCII_SRC.index(b"def b")]
        after = target.read_bytes()
        assert after == prefix + b"def b():\n    return 20\n"
        assert after[: len(prefix)] == prefix


# ---------------------------------------------------------------------------
# (b) UTF-8 BOM preserved (pins existing behavior through the new io_utils path)
# ---------------------------------------------------------------------------


class TestUtf8BomPreserved:
    def test_bom_survives_replace_of_the_first_symbol_exactly_once(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(BOM_ASCII_SRC)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    return 10\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == b"\xef\xbb\xbf"
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after == b"\xef\xbb\xbf" + b"def a():\n    return 10\n" + ASCII_SRC[
            len(b"def a():\n    return 1\n") :
        ]

    def test_write_source_with_sig_codec_never_doubles_a_leading_bom(
        self, tmp_path: Path
    ) -> None:
        """A -sig codec adds the BOM itself; text that still carries a
        leading U+FEFF must not encode into a doubled marker."""
        target = tmp_path / "f.py"
        write_source(target, "def a():\n    return 1\n", "utf-8-sig")
        assert target.read_bytes().count(b"\xef\xbb\xbf") == 1

        write_source(target, "\ufeffdef a():\n    return 1\n", "utf-8-sig")
        data = target.read_bytes()
        assert data.startswith(b"\xef\xbb\xbf")
        assert data.count(b"\xef\xbb\xbf") == 1


# ---------------------------------------------------------------------------
# (c)/(d) UTF-16 and binary are refused, loudly, without writing
# ---------------------------------------------------------------------------


class TestReadSourceRefusals:
    def test_utf16le_bom_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)
        with pytest.raises(UnsupportedEncodingError, match="not modified"):
            read_source(target)

    def test_utf16be_bom_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(UTF16BE_SRC)
        with pytest.raises(UnsupportedEncodingError, match="not modified"):
            read_source(target)

    def test_bomless_utf16_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes("def a():\n    return 1\n".encode("utf-16-le"))
        with pytest.raises(UnsupportedEncodingError, match="not modified"):
            read_source(target)

    def test_binary_file_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(BINARY_SRC)
        with pytest.raises(UnsupportedEncodingError, match="not modified"):
            read_source(target)

    def test_error_names_the_encoding_family(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)
        with pytest.raises(UnsupportedEncodingError, match="UTF-16"):
            read_source(target)

    def test_unsupported_encoding_error_is_a_value_error(self, tmp_path: Path) -> None:
        """So every existing ``except ValueError`` refusal path treats it as
        a clean refusal instead of a crash."""
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)
        with pytest.raises(ValueError):
            read_source(target)

    def test_original_bytes_untouched_after_a_refused_read(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)
        with contextlib.suppress(UnsupportedEncodingError):
            read_source(target)
        assert target.read_bytes() == UTF16LE_SRC


class TestReadSourceDetectionOrder:
    def test_empty_file_is_utf8_text(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(b"")
        assert read_source(target) == ("", "utf-8")

    def test_plain_utf8_detected_as_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes("café ☕\n".encode())
        assert read_source(target) == ("café ☕\n", "utf-8")

    def test_utf8_bom_detected_as_sig_and_stripped(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(BOM_ASCII_SRC)
        text, encoding = read_source(target)
        assert (text, encoding) == (ASCII_SRC.decode("utf-8"), "utf-8-sig")
        assert not text.startswith("﻿")

    def test_latin1_detected_as_latin1(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)
        text, encoding = read_source(target)
        assert encoding == "latin-1"
        assert text == (LATIN1_A + "\n\n" + LATIN1_B)
        assert "café" in text

    def test_ascii_detected_as_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(ASCII_SRC)
        assert read_source(target) == (ASCII_SRC.decode("ascii"), "utf-8")


class TestWriteSourceRoundTrip:
    def test_latin1_round_trip_is_byte_exact(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        write_source(target, LATIN1_A + "\n\n" + LATIN1_B, "latin-1")
        assert target.read_bytes() == LATIN1_SRC

    def test_utf8_sig_round_trip_restores_the_bom(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(BOM_ASCII_SRC)
        text, encoding = read_source(target)
        write_source(target, text, encoding)
        assert target.read_bytes() == BOM_ASCII_SRC

    def test_utf8_round_trip_is_byte_exact(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes("café ☕\n".encode())
        text, encoding = read_source(target)
        write_source(target, text, encoding)
        assert target.read_bytes() == "café ☕\n".encode()

    def test_unrepresentable_character_fails_loudly_without_writing(
        self, tmp_path: Path
    ) -> None:
        """A snippet that adds a character the file's codec cannot represent
        fails loudly and leaves the previous content on disk."""
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)
        with pytest.raises(UnsupportedEncodingError):
            write_source(target, "def a():\n    return '☕'\n", "latin-1")
        assert target.read_bytes() == LATIN1_SRC


class TestAtomicWriteEncoding:
    def test_str_content_encodes_with_the_given_codec(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        _atomic_write(target, LATIN1_A + "\n\n" + LATIN1_B, encoding="latin-1")
        assert target.read_bytes() == LATIN1_SRC

    def test_default_encoding_stays_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "f.py"
        _atomic_write(target, "café\n")
        assert target.read_bytes() == "café\n".encode()

    def test_bom_restore_still_applies_to_utf8_writes(self, tmp_path: Path) -> None:
        """B23 pinned old behavior that must survive: a plain-UTF-8 write to
        a BOM'd file restores the BOM (the codec was not asked to)."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_ASCII_SRC)
        _atomic_write(target, "def a():\n    return 10\n\n\ndef b():\n    return 2\n")
        assert target.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_backup_of_a_latin1_file_does_not_raise_and_round_trips(
        self, tmp_path: Path
    ) -> None:
        """B22 bridge: _atomic_write's backup save used to decode the
        original with UTF-8 and raised on latin-1 content, killing the
        write step after the merge had already succeeded."""
        store = BackupStore()
        target = tmp_path / "f.py"
        target.write_bytes(LATIN1_SRC)

        _atomic_write(
            target, LATIN1_A + "\n\n" + LATIN1_B, backups=store, encoding="latin-1",
        )  # must not raise

        assert target.read_bytes() == LATIN1_SRC  # same content rewritten
        assert str(target) in store
        # Step 19 (B22/B38) triage flip: BackupStore now stores RAW BYTES --
        # pop returns the file's exact original bytes, no decode round-trip
        # involved (the old assertion decoded the backup to text).
        backup_bytes = store.pop(str(target))
        assert backup_bytes == LATIN1_SRC

    def test_backup_of_a_bom_file_round_trips_through_sig(self, tmp_path: Path) -> None:
        store = BackupStore()
        target = tmp_path / "f.py"
        target.write_bytes(BOM_ASCII_SRC)
        _atomic_write(target, ASCII_SRC.decode("utf-8"), backups=store, encoding="utf-8-sig")
        assert target.read_bytes() == BOM_ASCII_SRC
        # Step 19 (B22/B38) triage flip: pop returns the raw backup bytes --
        # BOM included, byte-exact.
        backup_bytes = store.pop(str(target))
        assert backup_bytes == BOM_ASCII_SRC


# ---------------------------------------------------------------------------
# MCP fast_edit: unsupported encodings are refused before the merge
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeBackend:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield object()


class _FakeMcp:
    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


def _install_fake_mcp(monkeypatch) -> None:
    """Point ``tools_edit.mcp`` at a fake context and stub the merge to
    explode: an unsupported-encoding file must be refused BEFORE any merge
    (and therefore before tree-sitter ever sees mojibake)."""
    lifespan_context = {
        "backend_kind": "mlx",
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_edit, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")

    def _merge_must_not_run(*a, **kw):
        raise AssertionError(
            "chunked_merge was reached for a file whose encoding should have "
            "been refused at read time"
        )

    monkeypatch.setattr(tools_edit, "chunked_merge", _merge_must_not_run)


class TestMcpFastEditRefusals:
    @pytest.mark.parametrize("content", [UTF16LE_SRC, BINARY_SRC])
    def test_unsupported_encoding_refused_before_the_merge(
        self, tmp_path, monkeypatch, content
    ) -> None:
        _install_fake_mcp(monkeypatch)
        target = tmp_path / "mod.py"
        target.write_bytes(content)

        message = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="def a():\n    return 2\n",
            replace="a",
        ))

        assert message.startswith("Error"), message
        assert "not modified" in message, message
        assert target.read_bytes() == content

    def test_utf16_refusal_names_the_encoding(self, tmp_path, monkeypatch) -> None:
        _install_fake_mcp(monkeypatch)
        target = tmp_path / "mod.py"
        target.write_bytes(UTF16LE_SRC)

        message = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="x", replace="a",
        ))

        assert "UTF-16" in message, message
        assert target.read_bytes() == UTF16LE_SRC


class TestCliRefusals:
    @pytest.mark.parametrize("content", [UTF16LE_SRC, BINARY_SRC])
    def test_cli_edit_refuses_and_leaves_bytes_untouched(
        self, tmp_path, content
    ) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(content)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    return 2\n",
        )

        assert result.returncode != 0
        assert "not modified" in result.stderr, result.stderr
        assert target.read_bytes() == content

    def test_cli_utf16_refusal_names_the_encoding(self, tmp_path) -> None:
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    return 2\n",
        )

        assert result.returncode != 0
        assert "UTF-16" in result.stderr, result.stderr
