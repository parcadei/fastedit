from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]

_UTF8_BOM = b"\xef\xbb\xbf"


def run_cli(*args: str, input_text: str | None = None, env_extra: dict | None = None):
    """Invoke the real fastedit CLI as a subprocess, mirroring test_cli.py's helper."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


class TestCLIEditPreservesLineEndings:
    """fastedit edit --replace (the deterministic fast path) is byte-exact
    outside the replaced symbol and normalizes the substituted text to the
    file's own prevailing line ending."""

    def test_crlf_file_replace_stays_all_crlf(self, tmp_path: Path) -> None:
        """--replace on a CRLF file keeps untouched lines CRLF and normalizes an LF snippet to CRLF too."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n    return 3\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.count(b"\r\n") == after.count(b"\n"), (
            "result has a mix of CRLF and bare LF lines: " + repr(after)
        )
        assert after.startswith(b"def f():\r\n    return 1\r\n\r\ndef g():\r\n")
        assert b"return 3" in after
        assert b"return 2" not in after

    def test_cr_only_file_replace_keeps_cr_count_and_introduces_no_lf(self, tmp_path: Path) -> None:
        """--replace on a lone-CR (classic-Mac) file must not turn any CR into LF.

        Regression test for the defect the coordinator measured directly against
        commit 096d7ef: CR=2/LF=0 before, CR=1/LF=2 after. Root cause was
        get_ast_map (tldr structure, shelled out to for --replace's AST
        resolution) counting rows by scanning for '\\n' only, so it saw the
        whole 2-line CR-only file as one line and resolved the target symbol's
        span wrong (line_end=1 instead of 2) -- _try_deterministic_replace then
        bailed out (returned None) and cmd_edit fell through to the unfixed
        chunked_merge path, which spliced in a plain-LF snippet while leaving
        the untouched second line as raw CR. Fixed in get_ast_map by feeding
        tldr an LF-normalized temp copy (a same-length, same-position "\\r"
        -> "\\n" substitution) whenever a bare CR is present.
        """
        original = b"def a():\r    return 1\r"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        # Counts hoisted out of the f-string: backslashes inside f-string
        # expressions are a Python 3.12+ syntax (PEP 701) and fail to parse
        # under the 3.11 target. Pass/fail semantics are unchanged.
        before_crs = original.count(b"\r")
        after_crs = after.count(b"\r")
        assert after_crs == before_crs, (
            f"CR count changed: before={before_crs} after={after_crs}: {after!r}"
        )
        assert after.count(b"\n") == 0, "a bare LF leaked into a CR-only file: " + repr(after)
        assert after == b"def a():\r    return 2\r"

    def test_mixed_endings_file_replace_keeps_untouched_region_mixed(self, tmp_path: Path) -> None:
        """A file that was already mixed before the edit stays mixed outside the replaced span."""
        original = b"def f():\r\n    return 1\n\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 99\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        # The untouched tail (def g and its body) must be byte-identical to
        # the original -- the edit's blast radius is the replaced symbol only.
        assert after.endswith(b"\ndef g():\r\n    return 2\r\n")
        assert b"return 99" in after

    def test_bom_survives_replacing_the_symbol_the_bom_is_attached_to(self, tmp_path: Path) -> None:
        """A BOM riding on the file's only (and replaced) symbol must still survive.

        Regression test for the defect the coordinator measured directly
        against commit 096d7ef: first 3 bytes ef bb bf before, 64 65 66
        ('d','e','f') after -- the BOM was gone entirely. Root cause: the BOM
        is not a separate line, it is the first 3 bytes of line 1's own text,
        so a --replace of the symbol occupying line 1 discarded it along with
        the rest of that line. Fixed in _atomic_write (the single write path
        every verb funnels through): if the on-disk file already starts with
        a UTF-8 BOM and the new content does not, the BOM is re-prepended
        there, independent of whether the merge logic touched line 1.
        """
        original = _UTF8_BOM + b"def a():\r\n    return 1\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "a",
            "--snippet", "def a():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM, f"BOM was stripped: first 3 bytes = {after[:3]!r}"
        assert after == _UTF8_BOM + b"def a():\r\n    return 2\r\n"

    def test_bom_is_preserved_when_the_edit_does_not_touch_the_first_line(self, tmp_path: Path) -> None:
        """A UTF-8 BOM on an untouched first line also survives an edit elsewhere."""
        original = _UTF8_BOM + b"# header comment\ndef f():\n    return 1\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM, f"BOM was stripped: first 3 bytes = {after[:3]!r}"
        assert after.startswith(_UTF8_BOM + b"# header comment\n")
        assert b"return 2" in after

    def test_non_ascii_content_survives_a_replace_edit_byte_exact(self, tmp_path: Path) -> None:
        """Non-ASCII UTF-8 content outside the replaced span is untouched byte-for-byte."""
        original = "def greet():\r\n    return \"café ☃ 日本語\"\r\n\r\ndef f():\r\n    return 1\r\n".encode()
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert "café ☃ 日本語".encode() in after
        assert b"\r\n" in after
        assert b"return 2" in after

    def test_after_insert_leaves_untouched_crlf_regions_byte_exact(self, tmp_path: Path) -> None:
        """--after (the model-merge path) never rewrites the CRLF/CR endings of lines it does not touch."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--after", "f",
            "--snippet", "def h():\n    return 3\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.startswith(b"def f():\r\n    return 1\r\n")
        assert after.endswith(b"def g():\r\n    return 2\r\n")
        assert b"def h" in after


class TestCLIDeletePreservesLineEndings:
    """fastedit delete only removes lines; every remaining line must be byte-exact."""

    def test_crlf_file_delete_keeps_remaining_lines_crlf(self, tmp_path: Path) -> None:
        """Deleting a symbol from a CRLF file leaves the surviving lines CRLF, untranslated."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n\r\ndef h():\r\n    return 3\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("delete", str(target), "g")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "a bare LF leaked into a CRLF file: " + repr(after)
        )
        assert after.startswith(b"def f():\r\n    return 1\r\n")
        assert after.rstrip(b"\r\n").endswith(b"def h():\r\n    return 3")
        assert b"return 2" not in after

    def test_cr_only_file_delete_correctly_spans_the_whole_symbol(self, tmp_path: Path) -> None:
        """Deleting a symbol from a lone-CR file must remove exactly its own lines, no more, no less.

        Same get_ast_map root cause as the --replace CR-only regression above:
        before the fix, get_ast_map resolved a 2-line CR-only function as
        spanning only 1 line, so deleting it left the second line ("    return
        1\\r") behind as orphaned, uncorrupted-looking but wrong content.
        """
        original = b"def keep_this_fn():\r    return 1\r\rdef delete_this_fn():\r    return 2\r"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("delete", str(target), "delete_this_fn", "--force")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after, "a bare LF leaked into a CR-only file: " + repr(after)
        assert after == b"def keep_this_fn():\r    return 1\r\r"

    def test_bom_survives_a_delete(self, tmp_path: Path) -> None:
        """A UTF-8 BOM survives a delete that touches a later symbol."""
        original = _UTF8_BOM + b"def f():\n    return 1\n\n\ndef g():\n    return 2\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("delete", str(target), "g")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM
        assert after.startswith(_UTF8_BOM + b"def f():\n")


class TestCLIMovePreservesLineEndings:
    """fastedit move relocates a symbol within one file."""

    def test_crlf_file_move_keeps_untouched_and_moved_lines_crlf(self, tmp_path: Path) -> None:
        """Moving a symbol in a CRLF file never introduces a bare-LF line."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n\r\ndef h():\r\n    return 3\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("move", str(target), "g", "--after", "h")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b"")
        assert after.count(b"\r\n") == after.count(b"\n")
        assert after.index(b"def h") < after.index(b"def g")

    def test_cr_only_file_move_inserts_cr_separator_not_lf(self, tmp_path: Path) -> None:
        """Moving a symbol in a CR-only file uses a CR separator, not a hardcoded LF."""
        original = b"def f():\r    return 1\r\rdef g():\r    return 2\r\rdef h():\r    return 3\r"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("move", str(target), "g", "--after", "h")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after, "move_symbol used a hardcoded LF separator on a CR-only file"

    def test_bom_stays_at_file_start_not_duplicated_mid_file_when_its_symbol_moves(self, tmp_path: Path) -> None:
        """Moving the symbol the BOM is attached to must not carry the BOM along as embedded text.

        A file-level BOM riding on line 1's symbol, once that symbol is
        relocated elsewhere in the file, must still appear exactly once, at
        byte offset 0 -- never re-embedded at the new (non-zero) position the
        moved text ends up at. Caught by hand while re-checking the BOM fix
        across every write verb, not just edit: move_symbol read the file
        without stripping the BOM first, so it rode along inside `extracted`
        and reappeared mid-file after the splice, on top of _atomic_write's
        own (correct) restoration of the file-level BOM at position 0 --
        producing two BOM sequences instead of one.
        """
        original = _UTF8_BOM + b"def a():\r\n    x = 1\r\n    return x\r\n\r\ndef b():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("move", str(target), "a", "--after", "b")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.count(_UTF8_BOM) == 1, f"BOM appears {after.count(_UTF8_BOM)} times: {after!r}"
        assert after[:3] == _UTF8_BOM, f"BOM is not at the start: {after[:20]!r}"
        assert after == _UTF8_BOM + b"def b():\r\n    return 2\r\ndef a():\r\n    x = 1\r\n    return x\r\n\r\n"


class TestCLIRenamePreservesLineEndings:
    """fastedit rename (single file, AST-verified references)."""

    def test_crlf_file_rename_stays_byte_exact_outside_renamed_identifiers(self, tmp_path: Path) -> None:
        """Renaming references in a CRLF file changes only the identifier text, never the line endings."""
        original = b"def f():\r\n    x = 1\r\n    return x\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("rename", str(target), "x", "y")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after == b"def f():\r\n    y = 1\r\n    return y\r\n"

    def test_cr_only_file_rename_stays_byte_exact(self, tmp_path: Path) -> None:
        """Renaming a symbol in a lone-CR file must not introduce any LF.

        Same class of fix as the other AST-based verbs: the external tldr
        binary counts rows by scanning for '\\n', so a single-file `tldr
        references` lookup against a bare-CR source saw the whole file as one
        line and matched nothing. Fixed in _run_tldr_references the same way
        as get_ast_map -- an LF-normalized temp copy for the file-scope
        lookup only (workspace-scope, a whole directory, is unaffected).
        """
        original = b"def a():\r    x = 1\r    return x\r"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("rename", str(target), "a", "renamed_a")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after
        assert after == b"def renamed_a():\r    x = 1\r    return x\r"


class TestCLIRenameAllPreservesLineEndings:
    """fastedit rename-all (AST-verified, across a directory tree)."""

    def test_crlf_files_rename_all_stays_byte_exact_outside_renamed_identifiers(self, tmp_path: Path) -> None:
        """rename-all across two CRLF files never collapses their line endings."""
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_bytes(b"def use():\r\n    return shared_name(1)\r\n")
        b.write_bytes(b"def shared_name(n):\r\n    return n\r\n")

        result = run_cli("rename-all", str(tmp_path), "shared_name", "renamed_fn")
        assert result.returncode == 0, result.stderr

        assert a.read_bytes() == b"def use():\r\n    return renamed_fn(1)\r\n"
        assert b.read_bytes() == b"def renamed_fn(n):\r\n    return n\r\n"


class TestCLIMoveToFilePreservesLineEndings:
    """fastedit move-to-file (the one verb that genuinely crosses files)."""

    def test_moved_block_is_normalized_to_destination_crlf_convention(self, tmp_path: Path) -> None:
        """An LF-sourced symbol moved into a CRLF destination comes out CRLF, not mixed."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(b"def moved():\n    return 1\n")
        dst.write_bytes(b"def existing():\r\n    return 0\r\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after = dst.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "moved block kept its source LF endings instead of the destination's CRLF: " + repr(after)
        )
        assert after.startswith(b"def existing():\r\n    return 0\r\n")
        assert b"def moved" in after

    def test_source_file_remaining_content_stays_crlf_byte_exact(self, tmp_path: Path) -> None:
        """The source file's untouched remainder keeps its own CRLF endings after the move."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(b"def keep():\r\n    return 0\r\n\r\ndef moved():\r\n    return 1\r\n")
        dst.write_bytes(b"def existing():\n    return 0\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after = src.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "a bare LF leaked into the source file's remaining CRLF content: " + repr(after)
        )
        assert after.rstrip(b"\r\n") == b"def keep():\r\n    return 0"

    def test_source_bom_is_kept_and_not_duplicated_into_destination(self, tmp_path: Path) -> None:
        """The BOM-carrying symbol's own file keeps its BOM; the destination gets none.

        Regression test added alongside the move BOM fix: the moved symbol's
        text must not carry an embedded BOM into the destination (which never
        had one), and the source file (which loses that symbol but keeps its
        others) keeps its own file-level BOM.
        """
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(_UTF8_BOM + b"def moved():\r\n    x = 1\r\n    return x\r\n\r\ndef keep():\r\n    return 2\r\n")
        dst.write_bytes(b"def existing():\r\n    return 0\r\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after_src = src.read_bytes()
        after_dst = dst.read_bytes()
        assert after_src[:3] == _UTF8_BOM, f"source lost its own BOM: {after_src[:20]!r}"
        assert after_dst.count(_UTF8_BOM) == 0, (
            "destination gained a BOM it never had: " + repr(after_dst)
        )

    def test_destination_own_bom_survives_and_is_not_duplicated(self, tmp_path: Path) -> None:
        """A destination file that already has its own BOM keeps exactly one, at the start."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(b"def moved():\n    return 1\n")
        dst.write_bytes(_UTF8_BOM + b"def existing():\r\n    return 0\r\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after = dst.read_bytes()
        assert after.count(_UTF8_BOM) == 1, f"BOM appears {after.count(_UTF8_BOM)} times: {after!r}"
        assert after.startswith(_UTF8_BOM + b"def existing():\r\n    return 0\r\n")


class TestCLICreatePreservesBOM:
    """fastedit create's three input paths (--content, --content-file, stdin)
    all round-trip a caller-supplied BOM without stripping or duplicating it."""

    def test_content_file_preserves_a_leading_bom(self, tmp_path: Path) -> None:
        """--content-file with BOM-bearing bytes preserves the BOM."""
        content_file = tmp_path / "src.txt"
        content_file.write_bytes(_UTF8_BOM + b"def a():\n    return 1\n")
        target = tmp_path / "out.py"

        result = run_cli("create", str(target), "--content-file", str(content_file))
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM
        assert after == _UTF8_BOM + b"def a():\n    return 1\n"

    def test_inline_content_preserves_a_leading_bom(self, tmp_path: Path) -> None:
        """--content with a BOM character inline preserves the BOM."""
        target = tmp_path / "out.py"

        result = run_cli("create", str(target), "--content", "﻿def a():\n    return 1\n")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM
        assert after == _UTF8_BOM + b"def a():\n    return 1\n"

    def test_stdin_content_preserves_a_leading_bom(self, tmp_path: Path) -> None:
        """Bare stdin content (no --content/--content-file) with a BOM preserves it."""
        target = tmp_path / "out.py"

        result = run_cli("create", str(target), input_text="﻿def a():\n    return 1\n")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after[:3] == _UTF8_BOM
        assert after == _UTF8_BOM + b"def a():\n    return 1\n"
