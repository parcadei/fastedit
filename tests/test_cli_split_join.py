"""End-to-end CLI tests for `fastedit split` and `fastedit join`.

Runs the real CLI via subprocess (same convention as test_cli.py's and
test_cli_create_duplicate.py's run_cli), so these prove the actual
binary's behavior, not just the split_join module's internals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


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


class TestCLISplitJoinRoundTrip:
    def test_json_array_round_trip(self, tmp_path: Path) -> None:
        """split --by element then join reproduces a pretty-printed JSON array byte-for-byte."""
        original = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, [1, 2, 3]], indent=2) + "\n"
        source = tmp_path / "data.json"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--by", "element")
        assert result.returncode == 0
        assert (out_dir / "manifest.json").exists()

        rejoined = tmp_path / "rejoined.json"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original.encode("utf-8")

    def test_json_array_compact_round_trip(self, tmp_path: Path) -> None:
        """A compact (no whitespace) JSON array round-trips too -- the split is text-exact, not re-serialized."""
        original = '[1,2,3,"four",{"five":5}]'
        source = tmp_path / "compact.json"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        run_cli("split", str(source), "--out", str(out_dir), "--by", "element")
        rejoined = tmp_path / "rejoined.json"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        """split --rows then join reproduces a CSV file byte-for-byte, de-duplicating the repeated header."""
        original = "h1,h2,h3\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n"
        source = tmp_path / "data.csv"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--rows", "2")
        assert result.returncode == 0
        parts = sorted(p.name for p in out_dir.glob("part-*.csv"))
        assert len(parts) == 2
        # Every chunk must repeat the header row.
        for name in parts:
            assert (out_dir / name).read_text(encoding="utf-8").startswith("h1,h2,h3\n")

        rejoined = tmp_path / "rejoined.csv"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_markdown_heading_round_trip(self, tmp_path: Path) -> None:
        """split --by heading then join reproduces a Markdown file byte-for-byte."""
        original = "# Title\n\nintro text\n\n## Section A\n\nbody a\n\n## Section B\n\nbody b\n"
        source = tmp_path / "doc.md"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--by", "heading")
        assert result.returncode == 0
        assert len(list(out_dir.glob("part-*.md"))) == 3  # intro + 2 sections

        rejoined = tmp_path / "rejoined.md"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_markdown_custom_heading_level_round_trip(self, tmp_path: Path) -> None:
        """--level controls which heading depth splits, and join still round-trips it."""
        original = "# Title\n\n## A\n\ntext a\n\n### A.1\n\ndeep\n\n## B\n\ntext b\n"
        source = tmp_path / "doc.md"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--by", "heading", "--level", "3")
        assert result.returncode == 0
        rejoined = tmp_path / "rejoined.md"
        run_cli("join", str(out_dir), "-o", str(rejoined))
        assert rejoined.read_text(encoding="utf-8") == original

    def test_lines_round_trip_any_text_file(self, tmp_path: Path) -> None:
        """--lines chunks any text file, and join concatenates it back byte-for-byte."""
        original = "line1\nline2\nline3\nline4\nline5\n"
        source = tmp_path / "plain.txt"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--lines", "2")
        assert result.returncode == 0
        assert len(list(out_dir.glob("part-*.txt"))) == 3

        rejoined = tmp_path / "rejoined.txt"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_jsonl_round_trip_via_lines(self, tmp_path: Path) -> None:
        """JSONL splits and joins cleanly with --lines 1 -- one JSON object per physical line."""
        original = '{"a":1}\n{"b":2}\n{"c":3}\n'
        source = tmp_path / "data.jsonl"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        run_cli("split", str(source), "--out", str(out_dir), "--lines", "1")
        rejoined = tmp_path / "rejoined.jsonl"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_join_from_explicit_part_files_in_given_order(self, tmp_path: Path) -> None:
        """join also accepts explicit part files (not just a directory), concatenated in argv order."""
        original = "line1\nline2\nline3\nline4\n"
        source = tmp_path / "plain.txt"
        source.write_text(original, encoding="utf-8")
        out_dir = tmp_path / "parts"
        run_cli("split", str(source), "--out", str(out_dir), "--lines", "2")
        parts = sorted(out_dir.glob("part-*.txt"))
        rejoined = tmp_path / "rejoined.txt"
        result = run_cli("join", *[str(p) for p in parts], "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_text(encoding="utf-8") == original

    def test_crlf_round_trip_via_lines(self, tmp_path: Path) -> None:
        """--lines split+join is byte-exact on CRLF line endings (no universal-newline translation)."""
        original = b"l1\r\nl2\r\nl3\r\n"
        source = tmp_path / "crlf.txt"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--lines", "2")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.txt"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original

    def test_cr_only_round_trip_via_lines(self, tmp_path: Path) -> None:
        """--lines split+join is byte-exact on lone-CR (classic Mac) line endings."""
        original = b"l1\rl2\rl3\r"
        source = tmp_path / "cronly.txt"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--lines", "1")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.txt"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original

    def test_mixed_line_endings_round_trip_via_lines(self, tmp_path: Path) -> None:
        """--lines split+join is byte-exact when a file mixes CRLF, LF and lone-CR endings."""
        original = b"l1\r\nl2\nl3\r"
        source = tmp_path / "mixed.txt"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--lines", "1")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.txt"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original

    def test_crlf_round_trip_via_rows(self, tmp_path: Path) -> None:
        """--rows (CSV) split+join is byte-exact on CRLF line endings, including header de-dup."""
        original = b"h1,h2\r\nr1a,r1b\r\nr2a,r2b\r\n"
        source = tmp_path / "crlf.csv"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--rows", "1")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.csv"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original

    def test_cr_only_round_trip_via_rows(self, tmp_path: Path) -> None:
        """--rows (CSV) split+join is byte-exact on lone-CR endings, including header de-dup."""
        original = b"h1,h2\rr1a,r1b\rr2a,r2b\r"
        source = tmp_path / "cronly.csv"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--rows", "1")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.csv"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original

    def test_mixed_line_endings_round_trip_via_rows(self, tmp_path: Path) -> None:
        """--rows (CSV) split+join is byte-exact when the file mixes CRLF and lone-CR endings."""
        original = b"h1,h2\r\nr1a,r1b\rr2a,r2b\r\n"
        source = tmp_path / "mixed.csv"
        source.write_bytes(original)

        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--rows", "1")
        assert result.returncode == 0

        rejoined = tmp_path / "rejoined.csv"
        result = run_cli("join", str(out_dir), "-o", str(rejoined))
        assert result.returncode == 0
        assert rejoined.read_bytes() == original


class TestCLISplitXmlHtmlElement:
    def test_xml_by_element_splits_top_level_children(self, tmp_path: Path) -> None:
        """--by element on XML writes one file per top-level child, with a warning about lossiness."""
        source = tmp_path / "data.xml"
        source.write_text('<root><item id="1">A</item><item id="2">B</item></root>', encoding="utf-8")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--by", "element")
        assert result.returncode == 0
        assert "lossy" in result.stderr
        assert "read-only" in result.stderr
        parts = sorted(out_dir.glob("part-*.xml"))
        assert len(parts) == 2
        assert parts[0].read_text(encoding="utf-8") == '<item id="1">A</item>'
        assert parts[1].read_text(encoding="utf-8") == '<item id="2">B</item>'

    def test_xml_by_element_join_is_refused(self, tmp_path: Path) -> None:
        """join refuses an XML element split explicitly, rather than silently producing a wrong file."""
        source = tmp_path / "data.xml"
        source.write_text("<root><a>1</a><b>2</b></root>", encoding="utf-8")
        out_dir = tmp_path / "parts"
        run_cli("split", str(source), "--out", str(out_dir), "--by", "element")
        result = run_cli("join", str(out_dir), "-o", str(tmp_path / "out.xml"))
        assert result.returncode == 1
        assert "not joinable" in result.stderr
        assert not (tmp_path / "out.xml").exists()

    def test_html_by_element_splits_top_level_children(self, tmp_path: Path) -> None:
        """--by element on HTML also works, treating <html>'s direct children as the elements."""
        source = tmp_path / "page.html"
        source.write_text(
            "<html><head><title>t</title></head><body><div>x</div></body></html>", encoding="utf-8",
        )
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--by", "element")
        assert result.returncode == 0
        parts = sorted(out_dir.glob("part-*.html"))
        assert len(parts) == 2


class TestCLISplitJoinRefusals:
    def test_split_refuses_binary_source(self, tmp_path: Path) -> None:
        """A NUL-bearing binary source is refused before any part is written."""
        source = tmp_path / "payload.dat"
        source.write_bytes(b"abc\x00def")
        out_dir = tmp_path / "parts"
        result = run_cli("split", str(source), "--out", str(out_dir), "--lines", "1")
        assert result.returncode == 1
        assert "binary" in result.stderr
        assert not out_dir.exists() or not any(out_dir.iterdir())

    def test_by_element_refuses_a_json_object(self, tmp_path: Path) -> None:
        """--by element on a JSON file whose top-level value is an object (not an array) is refused."""
        source = tmp_path / "obj.json"
        source.write_text('{"a": 1}', encoding="utf-8")
        result = run_cli("split", str(source), "--out", str(tmp_path / "parts"), "--by", "element")
        assert result.returncode == 1
        assert "array" in result.stderr

    def test_by_element_refuses_a_csv_file(self, tmp_path: Path) -> None:
        """--by element does not apply to CSV -- only json/xml/html."""
        source = tmp_path / "data.csv"
        source.write_text("h1,h2\n1,2\n", encoding="utf-8")
        result = run_cli("split", str(source), "--out", str(tmp_path / "parts"), "--by", "element")
        assert result.returncode == 1
        assert "--by element does not apply to a csv file" in result.stderr

    def test_by_heading_refuses_a_non_markdown_file(self, tmp_path: Path) -> None:
        """--by heading only applies to markdown/mdx files."""
        source = tmp_path / "plain.txt"
        source.write_text("just text\n", encoding="utf-8")
        result = run_cli("split", str(source), "--out", str(tmp_path / "parts"), "--by", "heading")
        assert result.returncode == 1
        assert "--by heading does not apply to a text file" in result.stderr

    def test_rows_refuses_a_non_csv_file(self, tmp_path: Path) -> None:
        """--rows only applies to CSV/TSV files."""
        source = tmp_path / "plain.txt"
        source.write_text("a\nb\nc\n", encoding="utf-8")
        result = run_cli("split", str(source), "--out", str(tmp_path / "parts"), "--rows", "2")
        assert result.returncode == 1
        assert "--rows does not apply to a text file" in result.stderr

    def test_by_heading_refuses_when_no_matching_heading_exists(self, tmp_path: Path) -> None:
        """--by heading --level 4 on a file with only level-2 headings is refused, not silently a single chunk."""
        source = tmp_path / "doc.md"
        source.write_text("# Title\n\n## Section\n\nbody\n", encoding="utf-8")
        result = run_cli("split", str(source), "--out", str(tmp_path / "parts"), "--by", "heading", "--level", "4")
        assert result.returncode == 1
        assert "no level-4 headings" in result.stderr

    def test_split_refuses_missing_source(self, tmp_path: Path) -> None:
        """A source file that doesn't exist is refused with a clear message."""
        result = run_cli("split", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "parts"), "--lines", "1")
        assert result.returncode == 1
        assert "file not found" in result.stderr

    def test_join_refuses_missing_manifest_for_directory_input(self, tmp_path: Path) -> None:
        """join on a directory with no manifest.json is refused, not silently misjoined."""
        empty_dir = tmp_path / "not_a_split_dir"
        empty_dir.mkdir()
        (empty_dir / "stray.txt").write_text("hi\n", encoding="utf-8")
        result = run_cli("join", str(empty_dir), "-o", str(tmp_path / "out.txt"))
        assert result.returncode == 1
        assert "manifest.json" in result.stderr
