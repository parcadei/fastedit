"""Encoding + line-ending matrix across every FastEdit write verb.

Every fixture is built and re-read with Path.write_bytes/read_bytes -- never
text mode -- so a fixture's exact bytes (CRLF, lone CR, mixed, a BOM, UTF-16LE)
reach the CLI unmangled. Every assertion compares bytes for EQUALITY (exact
counts, exact slices, exact expected output) rather than a "count > 0" or
"BOM present" style check, because a >0 check passes on a partially corrupted
file just as happily as on a correct one.

Verbs covered: create, duplicate, edit --replace, edit --after, batch-edit,
multi-edit, delete, move, rename, rename-all, move-to-file, undo.

History note (prose only -- the tests' intent is unchanged): two defects found
while writing this matrix were first captured here as strict xfails/failures
and have since been fixed in their own remediation steps. `edit --after`
leaking bare LF bytes into a pure-CRLF file (B20) was unpinned in Step 12;
`batch-edit` / `multi-edit` dropping CR bytes on CRLF files (B19/B41) was
fixed in Step 14 via the central EOL normalizer unit-tested in
TestCentralEolNormalizer below. The batch-edit/multi-edit tests still carry
their skipif: those verbs construct a model backend unconditionally -- even
for pure exact-replace edits -- which requires `mlx` to be importable (the
same cause behind TestCLIBatchEdit / TestCLIMultiEdit in test_cli.py), so
they skip rather than fail when `mlx` is absent.
"""

from __future__ import annotations

import codecs
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastedit.inference.chunked_merge import _normalize_merged_eol
from fastedit.split_join import detect_line_ending

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None


def run_cli(*args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Invoke the real fastedit CLI (python -m fastedit) as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# --- fixture byte constants (every one built with a bytes literal, never str) ---

LF_SRC = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
CRLF_SRC = b"def a():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"
CR_SRC = b"def a():\r    return 1\r\r\rdef b():\r    return 2\r"
MIN_CR_SRC = b"def a():\r    return 1\r"
MIXED_SRC = b"def a():\r\n    return 1\n\ndef b():\r    return 2\r\n"
BOM_LF_SRC = b"\xef\xbb\xbf" + LF_SRC
NONASCII_SRC = 'def a():\n    return "éè 中文 \U0001F600"\n\n\ndef b():\n    return 2\n'.encode(
    "utf-8"
)
NO_EOL_SRC = b"def a():\n    return 1"
EMPTY_SRC = b""
UTF16LE_SRC = codecs.BOM_UTF16_LE + "def a():\n    return 1\n".encode("utf-16-le")


class TestCentralEolNormalizer:
    """Unit tests for chunked_merge._normalize_merged_eol (Step 14: B19/B31).

    Every chunked_merge return path (and the batch/multi composition over
    it, and the CLI's deterministic replace) funnels its merged_code through
    this ONE helper before the result may be surfaced, so the merged file's
    line-ending convention and trailing-newline state come from the ORIGINAL
    file -- never from a hardcoded "\\n" at a splice site (B31) and never
    from whatever ending the produced span happened to carry (B19).

    Contract under test:

    * an original with ONE line-ending convention (CRLF, lone CR, LF) whose
      merged output contains produced pieces in another convention comes out
      in the ORIGINAL's convention everywhere;
    * an original WITHOUT a trailing newline never gains one;
    * an original WITH a trailing newline keeps exactly one;
    * a MIXED-ending original has no single convention to enforce, so its
      untouched endings are never rewritten (byte-exactness beats guessing)
      while its trailing-newline state is still enforced.
    """

    def test_crlf_original_bare_lf_pieces_become_crlf_everywhere(self) -> None:
        """The B19 batch seam: produced LF-only pieces inside a CRLF file."""
        original = "def a():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"
        merged = "def a():\n    return 10\n\r\n\r\ndef b():\r\n    return 2\r\n"
        assert _normalize_merged_eol(merged, original) == (
            "def a():\r\n    return 10\r\n\r\n\r\ndef b():\r\n    return 2\r\n"
        )

    def test_lone_cr_original_bare_lf_pieces_become_cr_everywhere(self) -> None:
        """Lone-CR handling: detect_line_ending returns a lone CR and the funnel obeys it."""
        original = MIN_CR_SRC.decode()
        assert detect_line_ending(original) == "\r"
        merged = "def a():\n    return 9\n"
        out = _normalize_merged_eol(merged, original)
        assert out == "def a():\r    return 9\r"
        assert "\n" not in out

    def test_lf_original_crlf_pieces_become_lf_everywhere(self) -> None:
        """MIRROR of the CRLF check: a CRLF-produced piece never salts an LF file."""
        original = "def a():\n    return 1\n"
        merged = "def a():\r\n    return 9\r\n"
        assert _normalize_merged_eol(merged, original) == "def a():\n    return 9\n"

    def test_original_without_trailing_newline_never_gains_one(self) -> None:
        """B31: neither an LF nor a CRLF terminator may be appended at EOF."""
        original = "def x(): return 1"
        assert _normalize_merged_eol("def x(): return 2\n", original) == "def x(): return 2"
        assert _normalize_merged_eol("def x(): return 2\r\n", original) == "def x(): return 2"

    def test_original_with_trailing_newline_keeps_exactly_one(self) -> None:
        """A produced span that lost the file's terminator at EOF gets exactly one back."""
        original = "def a():\r\n    return 1\r\n"
        merged = "def a():\n    return 10"
        assert _normalize_merged_eol(merged, original) == "def a():\r\n    return 10\r\n"

    def test_mixed_original_untouched_endings_are_never_rewritten(self) -> None:
        """A mixed-ending original has no single convention: rewriting its
        untouched CRLF/CR bytes to the dominant one would corrupt bytes the
        edit never touched, so only the trailing-newline state is enforced."""
        original = MIXED_SRC.decode()
        merged = "def a():\r\n    return 1\n\ndef b():\n    return 99\n"
        assert _normalize_merged_eol(merged, original) == merged

    def test_mixed_original_still_enforces_the_trailing_newline_state(self) -> None:
        """Even for a mixed original, B31 holds: no terminator is appended to
        a file whose original had none (the produced piece's own is stripped)."""
        original = "def a():\r\n    return 1\n\ndef b():\r    return 2"
        merged = "def a():\r\n    return 1\n\ndef b():\n    return 99\n"
        assert _normalize_merged_eol(merged, original) == (
            "def a():\r\n    return 1\n\ndef b():\n    return 99"
        )

    def test_normalizer_is_a_no_op_when_merged_equals_the_original(self) -> None:
        """Rejection paths hand back the original verbatim: the funnel must
        be an exact no-op there, for every fixture shape in this matrix."""
        for original in (
            LF_SRC, CRLF_SRC, CR_SRC, MIN_CR_SRC, MIXED_SRC,
            BOM_LF_SRC, NONASCII_SRC, NO_EOL_SRC, EMPTY_SRC,
        ):
            text = original.decode("utf-8", errors="replace")
            assert _normalize_merged_eol(text, text) == text


class TestEncodingMatrixEditReplace:
    """`edit --replace` (the deterministic fast path) across the encoding axis."""

    def test_lf_file_replace_stays_lf_byte_exact_outside_span(self, tmp_path: Path) -> None:
        """An LF file's untouched tail stays byte-exact and no CR is introduced."""
        target = tmp_path / "f.py"
        target.write_bytes(LF_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 100\n")
        after = target.read_bytes()
        tail = b"\n\n\ndef b():\n    return 2\n"
        assert after.count(b"\r") == 0
        assert after.endswith(tail)

    def test_crlf_file_replace_keeps_cr_count_unchanged_and_new_line_is_crlf(
        self, tmp_path: Path
    ) -> None:
        """--replace on a CRLF file keeps the CR count unchanged and normalizes an LF snippet to CRLF."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 10\n")
        after = target.read_bytes()
        tail = CRLF_SRC[len(b"def a():\r\n    return 1\r\n") :]
        assert after.count(b"\r") == CRLF_SRC.count(b"\r")
        assert after == b"def a():\r\n    return 10\r\n" + tail

    def test_cr_only_min_fixture_replace_keeps_cr2_lf0(self, tmp_path: Path) -> None:
        """--replace on the minimal lone-CR fixture: CR=2/LF=0 must survive unchanged."""
        target = tmp_path / "f.py"
        target.write_bytes(MIN_CR_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 9\n")
        after = target.read_bytes()
        assert after.count(b"\r") == 2
        assert after.count(b"\n") == 0

    def test_mixed_file_replace_b_matches_known_good_byte_exact_result(self, tmp_path: Path) -> None:
        """On the mixed-endings fixture, replacing b with an LF snippet is a full byte-exact known result."""
        target = tmp_path / "f.py"
        target.write_bytes(MIXED_SRC)
        run_cli("edit", str(target), "--replace", "b", "--snippet", "def b():\n    return 99\n")
        after = target.read_bytes()
        assert MIXED_SRC.count(b"\r") == 3
        assert after.count(b"\r") == 1
        assert after == b"def a():\r\n    return 1\n\ndef b():\n    return 99\n"

    def test_bom_file_replace_symbol_on_first_line_bom_survives_exactly_once(
        self, tmp_path: Path
    ) -> None:
        """A BOM riding on the replaced (first) symbol's own line still survives, exactly once."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 1000\n")
        after = target.read_bytes()
        assert after[:3] == b"\xef\xbb\xbf"
        assert after.count(b"\xef\xbb\xbf") == 1

    def test_bom_file_replace_symbol_not_on_first_line_leaves_prefix_untouched(
        self, tmp_path: Path
    ) -> None:
        """Replacing a later symbol leaves the BOM and every byte before it untouched."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        prefix = BOM_LF_SRC[: BOM_LF_SRC.index(b"def b()")]
        run_cli("edit", str(target), "--replace", "b", "--snippet", "def b():\n    return 2000\n")
        after = target.read_bytes()
        assert after[:3] == b"\xef\xbb\xbf"
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after[: len(prefix)] == prefix

    def test_utf8_no_bom_replace_never_introduces_a_bom(self, tmp_path: Path) -> None:
        """A plain UTF-8 file with no BOM must not gain one from a --replace edit."""
        target = tmp_path / "f.py"
        target.write_bytes(LF_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 1\n")
        after = target.read_bytes()
        assert after[:3] != b"\xef\xbb\xbf"
        assert after.count(b"\xef\xbb\xbf") == 0

    def test_non_ascii_content_survives_a_replace_edit_byte_exact(self, tmp_path: Path) -> None:
        """Accented Latin, CJK and an emoji in the untouched function survive a sibling replace byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        prefix = NONASCII_SRC[: NONASCII_SRC.index(b"def b()")]
        run_cli("edit", str(target), "--replace", "b", "--snippet", "def b():\n    return 22\n")
        after = target.read_bytes()
        assert after[: len(prefix)] == prefix

    def test_no_trailing_newline_file_replace_result_is_exact(self, tmp_path: Path) -> None:
        """Replacing the sole symbol of a no-trailing-newline file: the exact resulting bytes are pinned.

        EXPECTATION UPDATED (Step 14, B31): this test previously pinned the
        old destructive behaviour where the replaced file GAINED a trailing
        newline (b"def a():\\n    return 9\\n") that was in neither the
        original file nor the snippet's own text. That byte was the forced
        "\\n" the deterministic splice appended -- B31. The trailing-newline
        state now comes from the ORIGINAL via the central normalizer, so the
        no-trailing-newline file stays without one.
        """
        target = tmp_path / "f.py"
        target.write_bytes(NO_EOL_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 9")
        after = target.read_bytes()
        assert after == b"def a():\n    return 9"

    def test_empty_file_replace_refuses_cleanly_and_leaves_file_untouched(self, tmp_path: Path) -> None:
        """An empty file has no symbols: --replace must fail fast and leave 0 bytes untouched."""
        target = tmp_path / "f.py"
        target.write_bytes(EMPTY_SRC)
        result = run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 1\n")
        assert result.returncode != 0
        assert target.read_bytes() == b""

    def test_utf16le_file_replace_refuses_cleanly_and_leaves_bytes_byte_exact(
        self, tmp_path: Path
    ) -> None:
        """A UTF-16LE file cannot be parsed as Python source: --replace refuses without corrupting a byte."""
        target = tmp_path / "f.py"
        target.write_bytes(UTF16LE_SRC)
        result = run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 1\n")
        assert result.returncode != 0
        assert target.read_bytes() == UTF16LE_SRC

    def test_body_only_snippet_must_not_delete_the_symbol_signature(
        self, tmp_path: Path
    ) -> None:
        """Regression test: a body-only snippet must not destroy the target symbol's signature line.

        Documents TRDD-8M0MXRJO: the deterministic text-match branch of
        _try_deterministic_replace splices `snippet` in as a literal
        statement-level replacement of the ENTIRE symbol span (including its
        `def` line), so a snippet that is only the new body -- no `def f():`
        line of its own -- deletes the signature outright. tree-sitter's
        Python grammar does not set `has_error` on a bare over-indented
        top-level statement, so `validate_parse` reports the result as valid
        and C1's `_refuse_if_edit_broke_parse` guard never fires. The fix
        makes fastedit refuse such an edit, leaving the file byte-identical.

        Measured limit: an XPASS proves the fix landed, but continued failure
        would NOT have proven it had not -- a fix that refused AND still wrote,
        or that refused only for some snippet shapes, would have left this
        test failing too. XPASS implies fixed; fixed does not imply XPASS.
        """
        target = tmp_path / ("fixture" + ".py")
        target.write_bytes(b"def f():\n    return 1\n")
        run_cli("edit", str(target), "--replace", "f", "--snippet", "    return 99")
        assert b"def f(" in target.read_bytes()

class TestEncodingMatrixEditAfter:
    """`edit --after` (insertion) across the encoding axis."""

    def test_lf_file_after_insert_at_end_is_a_byte_exact_prefix(self, tmp_path: Path) -> None:
        """Inserting after the last symbol of an LF file leaves the whole original file as an exact prefix."""
        target = tmp_path / "f.py"
        target.write_bytes(LF_SRC)
        run_cli("edit", str(target), "--after", "b", "--snippet", "def m():\n    return 5\n")
        after = target.read_bytes()
        assert after.startswith(LF_SRC)

    def test_bom_file_after_insert_leaves_bom_and_original_content_as_exact_prefix(
        self, tmp_path: Path
    ) -> None:
        """Inserting after the last symbol of a BOM file: BOM plus everything before it is an exact prefix."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("edit", str(target), "--after", "b", "--snippet", "def m():\n    return 5\n")
        after = target.read_bytes()
        assert after.startswith(BOM_LF_SRC)
        assert after.count(b"\xef\xbb\xbf") == 1

    def test_non_ascii_after_insert_leaves_original_content_as_exact_prefix(
        self, tmp_path: Path
    ) -> None:
        """Inserting after the last symbol of a non-ASCII file leaves every original byte as an exact prefix."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        run_cli("edit", str(target), "--after", "b", "--snippet", "def m():\n    return 5\n")
        after = target.read_bytes()
        assert after.startswith(NONASCII_SRC)

    def test_empty_file_after_insert_refuses_cleanly_no_anchor_symbol(self, tmp_path: Path) -> None:
        """An empty file has no anchor symbol: --after must fail fast and leave 0 bytes untouched."""
        target = tmp_path / "f.py"
        target.write_bytes(EMPTY_SRC)
        result = run_cli("edit", str(target), "--after", "a", "--snippet", "def m():\n    return 5\n")
        assert result.returncode != 0
        assert target.read_bytes() == b""

    def test_crlf_file_after_insert_should_stay_all_crlf_but_does_not(self, tmp_path: Path) -> None:
        """--after on a CRLF file must introduce zero bare LF bytes.

        Was a strict-xfail defect reproducer (B20: the after= splice hardcoded
        "\\n" separators and never normalized the snippet). Unpinned in Step 12:
        the after= fast path now derives its separators and the inserted piece's
        terminators from the original's line-ending convention, so this is a
        permanent regression test.
        """
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("edit", str(target), "--after", "a", "--snippet", "def m():\n    return 5\n")
        after = target.read_bytes()
        bare_lf_count = after.count(b"\n") - after.count(b"\r\n")
        assert bare_lf_count == 0

    def test_lf_file_after_insert_has_no_bare_cr_mirror_of_the_crlf_check(self, tmp_path: Path) -> None:
        """MIRROR of the CRLF check: an LF file must contain no bare CR after an insertion.

        Catches a normalize-everything-to-CRLF overcorrection that the CRLF-only
        assertion above would miss entirely.
        """
        target = tmp_path / "f.py"
        target.write_bytes(LF_SRC)
        result = run_cli("edit", str(target), "--after", "a", "--snippet", "def m():\n    return 5\n")
        assert result.returncode == 0
        after = target.read_bytes()
        assert after.count(b"\r") == 0


class TestEncodingMatrixDelete:
    """`delete` across the encoding axis."""

    def test_crlf_file_delete_leaves_remaining_lines_byte_exact(self, tmp_path: Path) -> None:
        """Deleting the last symbol of a CRLF file leaves the untouched prefix byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("delete", str(target), "b")
        after = target.read_bytes()
        assert after == CRLF_SRC[: CRLF_SRC.index(b"def b()")]

    def test_cr_only_file_delete_leaves_remaining_lines_byte_exact(self, tmp_path: Path) -> None:
        """Deleting the last symbol of a lone-CR file leaves the untouched prefix byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(CR_SRC)
        run_cli("delete", str(target), "b")
        after = target.read_bytes()
        assert after == CR_SRC[: CR_SRC.index(b"def b()")]

    def test_mixed_file_delete_leaves_remaining_region_byte_exact(self, tmp_path: Path) -> None:
        """Deleting b from the mixed-endings fixture leaves the a-region byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(MIXED_SRC)
        run_cli("delete", str(target), "b")
        after = target.read_bytes()
        assert after == MIXED_SRC[: MIXED_SRC.index(b"def b()")]

    def test_bom_file_delete_keeps_bom_exactly_once_and_prefix_exact(self, tmp_path: Path) -> None:
        """Deleting a later symbol from a BOM file preserves the BOM once and the prefix byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("delete", str(target), "b")
        after = target.read_bytes()
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after == BOM_LF_SRC[: BOM_LF_SRC.index(b"def b()")]

    def test_non_ascii_file_delete_leaves_prefix_byte_exact(self, tmp_path: Path) -> None:
        """Deleting b leaves a's accented/CJK/emoji content byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        run_cli("delete", str(target), "b")
        after = target.read_bytes()
        assert after == NONASCII_SRC[: NONASCII_SRC.index(b"def b()")]

    def test_empty_file_delete_refuses_cleanly_and_leaves_file_untouched(self, tmp_path: Path) -> None:
        """An empty file has no symbols: delete must fail fast and leave 0 bytes untouched."""
        target = tmp_path / "f.py"
        target.write_bytes(EMPTY_SRC)
        result = run_cli("delete", str(target), "a")
        assert result.returncode != 0
        assert target.read_bytes() == b""


class TestEncodingMatrixMove:
    """`move` (single-file reorder) across the encoding axis."""

    def test_crlf_file_move_is_byte_exact_reorder(self, tmp_path: Path) -> None:
        """Moving a after b in a CRLF file reorders bytes exactly, CR count unchanged."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("move", str(target), "a", "--after", "b")
        after = target.read_bytes()
        assert len(after) == len(CRLF_SRC)
        assert after.count(b"\r") == CRLF_SRC.count(b"\r")
        assert after.count(b"\n") == CRLF_SRC.count(b"\n")

    def test_cr_only_file_move_is_byte_exact_reorder(self, tmp_path: Path) -> None:
        """Moving a after b in a lone-CR file reorders bytes exactly, introducing no LF."""
        target = tmp_path / "f.py"
        target.write_bytes(CR_SRC)
        run_cli("move", str(target), "a", "--after", "b")
        after = target.read_bytes()
        assert len(after) == len(CR_SRC)
        assert after.count(b"\r") == CR_SRC.count(b"\r")
        assert after.count(b"\n") == 0

    def test_mixed_file_move_preserves_exact_cr_and_lf_counts(self, tmp_path: Path) -> None:
        """Moving a after b in the mixed-endings fixture preserves the exact CR and LF counts."""
        target = tmp_path / "f.py"
        target.write_bytes(MIXED_SRC)
        run_cli("move", str(target), "a", "--after", "b")
        after = target.read_bytes()
        assert after.count(b"\r") == MIXED_SRC.count(b"\r")
        assert after.count(b"\n") == MIXED_SRC.count(b"\n")

    def test_bom_file_move_keeps_bom_exactly_once_when_its_own_symbol_moves(
        self, tmp_path: Path
    ) -> None:
        """Moving the symbol the BOM is attached to still leaves exactly one BOM in the file."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("move", str(target), "a", "--after", "b")
        after = target.read_bytes()
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after.startswith(b"\xef\xbb\xbf")

    def test_non_ascii_file_move_preserves_all_non_ascii_bytes(self, tmp_path: Path) -> None:
        """Moving a after b in the non-ASCII fixture: every accented/CJK/emoji byte survives somewhere in the result."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        needle = NONASCII_SRC[NONASCII_SRC.index(b'"') : NONASCII_SRC.index(b'"\n') + 1]
        run_cli("move", str(target), "a", "--after", "b")
        after = target.read_bytes()
        assert needle in after
        assert len(after) == len(NONASCII_SRC)


class TestEncodingMatrixRename:
    """`rename` (single-file, AST-verified identifier rename) across the encoding axis."""

    def test_crlf_file_rename_is_byte_exact_outside_the_identifier(self, tmp_path: Path) -> None:
        """Renaming a->aa in a CRLF file changes only the 2 added identifier bytes; CR count is unchanged."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("rename", str(target), "a", "aa")
        after = target.read_bytes()
        assert after == b"def aa():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"
        assert after.count(b"\r") == CRLF_SRC.count(b"\r")

    def test_cr_only_file_rename_is_byte_exact_outside_the_identifier(self, tmp_path: Path) -> None:
        """Renaming a->aa in a lone-CR file changes only the identifier; no LF is introduced."""
        target = tmp_path / "f.py"
        target.write_bytes(CR_SRC)
        run_cli("rename", str(target), "a", "aa")
        after = target.read_bytes()
        assert after == b"def aa():\r    return 1\r\r\rdef b():\r    return 2\r"
        assert after.count(b"\n") == 0

    def test_mixed_file_rename_is_byte_exact_outside_the_identifier(self, tmp_path: Path) -> None:
        """Renaming a->aa in the mixed-endings fixture changes only the identifier bytes."""
        target = tmp_path / "f.py"
        target.write_bytes(MIXED_SRC)
        run_cli("rename", str(target), "a", "aa")
        after = target.read_bytes()
        assert after == b"def aa():\r\n    return 1\n\ndef b():\r    return 2\r\n"

    def test_bom_file_rename_of_first_line_symbol_keeps_bom_exactly_once(self, tmp_path: Path) -> None:
        """Renaming the symbol whose line the BOM is attached to still leaves exactly one BOM."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("rename", str(target), "a", "aa")
        after = target.read_bytes()
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after == b"\xef\xbb\xbfdef aa():\n    return 1\n\n\ndef b():\n    return 2\n"

    def test_non_ascii_file_rename_leaves_non_ascii_prefix_byte_exact(self, tmp_path: Path) -> None:
        """Renaming b->bb leaves a's accented/CJK/emoji content byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        prefix = NONASCII_SRC[: NONASCII_SRC.index(b"def b()")]
        run_cli("rename", str(target), "b", "bb")
        after = target.read_bytes()
        assert after[: len(prefix)] == prefix

    def test_empty_file_rename_refuses_cleanly_and_leaves_file_untouched(self, tmp_path: Path) -> None:
        """An empty file has no identifier to rename: rename must fail fast and leave 0 bytes untouched."""
        target = tmp_path / "f.py"
        target.write_bytes(EMPTY_SRC)
        result = run_cli("rename", str(target), "a", "aa")
        assert result.returncode != 0
        assert target.read_bytes() == b""


class TestEncodingMatrixRenameAll:
    """`rename-all` (directory-wide rename) across the encoding axis."""

    def test_crlf_file_rename_all_is_byte_exact_outside_the_identifier(self, tmp_path: Path) -> None:
        """rename-all a->aa on a directory containing a CRLF file changes only the identifier."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("rename-all", str(tmp_path), "a", "aa")
        after = target.read_bytes()
        assert after == b"def aa():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n"

    def test_mixed_file_rename_all_is_byte_exact_outside_the_identifier(self, tmp_path: Path) -> None:
        """rename-all a->aa on a directory containing a mixed-endings file changes only the identifier."""
        target = tmp_path / "f.py"
        target.write_bytes(MIXED_SRC)
        run_cli("rename-all", str(tmp_path), "a", "aa")
        after = target.read_bytes()
        assert after == b"def aa():\r\n    return 1\n\ndef b():\r    return 2\r\n"

    def test_bom_file_rename_all_keeps_bom_exactly_once(self, tmp_path: Path) -> None:
        """rename-all a->aa on a directory containing a BOM file leaves exactly one BOM."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("rename-all", str(tmp_path), "a", "aa")
        after = target.read_bytes()
        assert after.count(b"\xef\xbb\xbf") == 1
        assert after == b"\xef\xbb\xbfdef aa():\n    return 1\n\n\ndef b():\n    return 2\n"

    def test_non_ascii_file_rename_all_leaves_non_ascii_prefix_byte_exact(self, tmp_path: Path) -> None:
        """rename-all b->bb on a directory containing a non-ASCII file leaves a's content byte-exact."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        prefix = NONASCII_SRC[: NONASCII_SRC.index(b"def b()")]
        run_cli("rename-all", str(tmp_path), "b", "bb")
        after = target.read_bytes()
        assert after[: len(prefix)] == prefix


class TestEncodingMatrixMoveToFile:
    """`move-to-file` across the encoding axis: the moved block adopts the destination's convention."""

    def test_crlf_source_moves_into_bom_lf_destination_normalized_and_bom_preserved(
        self, tmp_path: Path
    ) -> None:
        """Moving c from a CRLF source into a BOM+LF destination: the block is LF-normalized, dst BOM stays exactly once."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(CRLF_SRC.replace(b"def b()", b"def c()"))
        dest_file = tmp_path / "dest.py"
        dest_file.write_bytes(b"\xef\xbb\xbfdef z():\n    return 0\n")
        run_cli("move-to-file", "c", str(source_file), str(dest_file))
        dest_after = dest_file.read_bytes()
        assert dest_after == b"\xef\xbb\xbfdef z():\n    return 0\n\ndef c():\n    return 2\n"
        assert dest_after.count(b"\xef\xbb\xbf") == 1
        source_after = source_file.read_bytes()
        assert source_after == (CRLF_SRC.replace(b"def b()", b"def c()"))[
            : CRLF_SRC.index(b"def b()")
        ]

    def test_cr_only_source_moves_into_crlf_destination_normalized_to_crlf(
        self, tmp_path: Path
    ) -> None:
        """Moving b from a lone-CR source into a CRLF destination: the moved block becomes CRLF."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(CR_SRC)
        dest_file = tmp_path / "dest.py"
        dest_file.write_bytes(b"def z():\r\n    return 0\r\n")
        run_cli("move-to-file", "b", str(source_file), str(dest_file))
        dest_after = dest_file.read_bytes()
        assert dest_after == b"def z():\r\n    return 0\r\n\r\ndef b():\r\n    return 2\r\n"

    def test_symbol_name_conflict_refuses_cleanly_leaving_both_files_untouched(
        self, tmp_path: Path
    ) -> None:
        """A destination that already defines the moved symbol's name refuses and touches neither file."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(CRLF_SRC)
        dest_file = tmp_path / "dest.py"
        dest_file.write_bytes(LF_SRC)
        result = run_cli("move-to-file", "b", str(source_file), str(dest_file))
        assert result.returncode != 0
        assert source_file.read_bytes() == CRLF_SRC
        assert dest_file.read_bytes() == LF_SRC


class TestEncodingMatrixUndo:
    """`undo` across the encoding axis: it must restore the exact original bytes."""

    def test_undo_restores_crlf_file_byte_exact(self, tmp_path: Path) -> None:
        """undo after an edit on a CRLF file restores the exact original bytes."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        run_cli("edit", str(target), "--replace", "a", "--snippet", "def a():\n    return 99\n")
        assert target.read_bytes() != CRLF_SRC
        run_cli("undo", str(target))
        assert target.read_bytes() == CRLF_SRC

    def test_undo_restores_bom_file_byte_exact(self, tmp_path: Path) -> None:
        """undo after an edit on a BOM file restores the exact original bytes, BOM included."""
        target = tmp_path / "f.py"
        target.write_bytes(BOM_LF_SRC)
        run_cli("edit", str(target), "--replace", "b", "--snippet", "def b():\n    return 77\n")
        assert target.read_bytes() != BOM_LF_SRC
        run_cli("undo", str(target))
        assert target.read_bytes() == BOM_LF_SRC

    def test_undo_restores_non_ascii_file_byte_exact(self, tmp_path: Path) -> None:
        """undo after an edit on a non-ASCII file restores the exact original bytes."""
        target = tmp_path / "f.py"
        target.write_bytes(NONASCII_SRC)
        run_cli("edit", str(target), "--replace", "b", "--snippet", "def b():\n    return 55\n")
        assert target.read_bytes() != NONASCII_SRC
        run_cli("undo", str(target))
        assert target.read_bytes() == NONASCII_SRC


class TestEncodingMatrixCreate:
    """`create` across the encoding axis: content is passed through, validated as UTF-8 text."""

    def test_create_inline_content_lf_is_byte_exact(self, tmp_path: Path) -> None:
        """--content with an LF-terminated string writes exactly those bytes."""
        target = tmp_path / "created.py"
        run_cli("create", str(target), "--content", "x = 1\n")
        assert target.read_bytes() == b"x = 1\n"

    def test_create_content_file_crlf_is_byte_exact(self, tmp_path: Path) -> None:
        """--content-file with CRLF bytes writes exactly those bytes, unnormalized."""
        content_file = tmp_path / "content.bin"
        content_file.write_bytes(CRLF_SRC)
        target = tmp_path / "created.py"
        run_cli("create", str(target), "--content-file", str(content_file))
        assert target.read_bytes() == CRLF_SRC

    def test_create_content_file_cr_only_is_byte_exact(self, tmp_path: Path) -> None:
        """--content-file with lone-CR bytes writes exactly those bytes, unnormalized."""
        content_file = tmp_path / "content.bin"
        content_file.write_bytes(CR_SRC)
        target = tmp_path / "created.py"
        run_cli("create", str(target), "--content-file", str(content_file))
        assert target.read_bytes() == CR_SRC

    def test_create_content_file_mixed_is_byte_exact(self, tmp_path: Path) -> None:
        """--content-file with mixed-ending bytes writes exactly those bytes, unnormalized."""
        content_file = tmp_path / "content.bin"
        content_file.write_bytes(MIXED_SRC)
        target = tmp_path / "created.py"
        run_cli("create", str(target), "--content-file", str(content_file))
        assert target.read_bytes() == MIXED_SRC

    def test_create_content_file_utf16le_is_rejected_and_no_file_is_written(
        self, tmp_path: Path
    ) -> None:
        """--content-file with UTF-16LE bytes (not valid UTF-8) is refused; no file is left behind."""
        content_file = tmp_path / "content.bin"
        content_file.write_bytes(UTF16LE_SRC)
        target = tmp_path / "created.py"
        result = run_cli("create", str(target), "--content-file", str(content_file))
        assert result.returncode != 0
        assert not target.exists()

    def test_create_empty_content_writes_a_zero_byte_file(self, tmp_path: Path) -> None:
        """--content "" creates the file with exactly 0 bytes."""
        target = tmp_path / "created.py"
        run_cli("create", str(target), "--content", "")
        assert target.read_bytes() == b""


class TestEncodingMatrixDuplicate:
    """`duplicate` across the encoding axis: a pure byte-for-byte copy, encoding-agnostic."""

    def test_duplicate_crlf_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a CRLF file produces byte-identical output."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(CRLF_SRC)
        dest_file = tmp_path / "dest.py"
        run_cli("duplicate", str(source_file), str(dest_file))
        assert dest_file.read_bytes() == CRLF_SRC

    def test_duplicate_cr_only_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a lone-CR file produces byte-identical output."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(CR_SRC)
        dest_file = tmp_path / "dest.py"
        run_cli("duplicate", str(source_file), str(dest_file))
        assert dest_file.read_bytes() == CR_SRC

    def test_duplicate_mixed_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a mixed-endings file produces byte-identical output."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(MIXED_SRC)
        dest_file = tmp_path / "dest.py"
        run_cli("duplicate", str(source_file), str(dest_file))
        assert dest_file.read_bytes() == MIXED_SRC

    def test_duplicate_bom_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a BOM file produces byte-identical output, BOM included exactly once."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(BOM_LF_SRC)
        dest_file = tmp_path / "dest.py"
        run_cli("duplicate", str(source_file), str(dest_file))
        after = dest_file.read_bytes()
        assert after == BOM_LF_SRC
        assert after.count(b"\xef\xbb\xbf") == 1

    def test_duplicate_utf16le_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a UTF-16LE file is a pure byte copy, unlike create which validates UTF-8."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(UTF16LE_SRC)
        dest_file = tmp_path / "dest.py"
        result = run_cli("duplicate", str(source_file), str(dest_file))
        assert result.returncode == 0
        assert dest_file.read_bytes() == UTF16LE_SRC

    def test_duplicate_empty_file_is_byte_identical(self, tmp_path: Path) -> None:
        """Duplicating a 0-byte file produces a 0-byte destination."""
        source_file = tmp_path / "source.py"
        source_file.write_bytes(EMPTY_SRC)
        dest_file = tmp_path / "dest.py"
        run_cli("duplicate", str(source_file), str(dest_file))
        assert dest_file.read_bytes() == b""


class TestEncodingMatrixBatchAndMultiEdit:
    """`batch-edit` / `multi-edit`: both construct a model backend unconditionally, even for
    pure exact-replace edits, which requires `mlx` (or a reachable vLLM endpoint) to be
    importable. This is the same root cause as this repo's known 4-failure baseline
    (TestCLIBatchEdit / TestCLIMultiEdit in test_cli.py), so these are skipped rather than
    failed when mlx is unavailable -- matching the "green = the known 4 and nothing else"
    contract instead of adding a 5th/6th failure class."""

    @pytest.mark.skipif(not _MLX_AVAILABLE, reason="batch-edit constructs a backend unconditionally; needs mlx")
    def test_batch_edit_on_crlf_file_both_edits_normalized_to_crlf(self, tmp_path: Path) -> None:
        """batch-edit with two LF snippets on a CRLF file: both replacements normalize to CRLF."""
        target = tmp_path / "f.py"
        target.write_bytes(CRLF_SRC)
        edits = json.dumps(
            [
                {"snippet": "def a():\n    return 10\n", "replace": "a"},
                {"snippet": "def b():\n    return 20\n", "replace": "b"},
            ]
        )
        run_cli("batch-edit", str(target), "--edits", edits)
        after = target.read_bytes()
        assert after.count(b"\r") == CRLF_SRC.count(b"\r")
        assert after.count(b"\n") - after.count(b"\r\n") == 0

    @pytest.mark.skipif(not _MLX_AVAILABLE, reason="multi-edit constructs a backend unconditionally; needs mlx")
    def test_multi_edit_across_two_crlf_files_each_normalized_to_crlf(self, tmp_path: Path) -> None:
        """multi-edit with an LF snippet on two separate CRLF files: each file normalizes independently."""
        first = tmp_path / "one.py"
        first.write_bytes(CRLF_SRC)
        second = tmp_path / "two.py"
        second.write_bytes(CRLF_SRC)
        file_edits = json.dumps(
            [
                {"file_path": str(first), "edits": [{"snippet": "def a():\n    return 111\n", "replace": "a"}]},
                {"file_path": str(second), "edits": [{"snippet": "def b():\n    return 222\n", "replace": "b"}]},
            ]
        )
        run_cli("multi-edit", "--file-edits", file_edits)
        after_first = first.read_bytes()
        after_second = second.read_bytes()
        assert after_first.count(b"\r") == CRLF_SRC.count(b"\r")
        assert after_second.count(b"\r") == CRLF_SRC.count(b"\r")
