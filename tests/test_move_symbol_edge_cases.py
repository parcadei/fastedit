"""move_symbol edge cases: splice-safety at the file's unterminated last line.

``splitlines(keepends=True)`` keeps every line terminated EXCEPT possibly
the file's last one. A file whose final symbol line carries no newline
(being edited by a tool that doesn't add one, say) put move_symbol's
"blank line separator" decision in a state where the terminator was never
added: the moved block was spliced directly onto the unterminated line,
concatenating two statements into one — silent corruption, exit 0
(parse_valid catches it only when the caller passed a language).
"""

from __future__ import annotations

from pathlib import Path

from fastedit.inference.symbols import move_symbol


def test_move_to_eof_terminates_unterminated_last_line(tmp_path: Path) -> None:
    """Moving a symbol after the file's LAST symbol must not concatenate
    that symbol's (unterminated) final line with the moved block."""
    file_path = tmp_path / "m.py"
    file_path.write_text("def a():\n    pass\ndef b():\n    pass")  # no trailing \n

    result = move_symbol(str(file_path), "a", "b", language="python")

    assert result.parse_valid is True
    assert "passdef a():" not in result.merged_code
    assert result.merged_code == "def b():\n    pass\ndef a():\n    pass\n\n"


def test_move_midfile_is_unchanged_by_terminator_fix(tmp_path: Path) -> None:
    """A normal file (every line terminated) moves exactly as before."""
    file_path = tmp_path / "m.py"
    file_path.write_text(
        "def f():\n    return 1\n\n"
        "def g():\n    return 2\n\n"
        "def h():\n    return 3\n",
    )

    result = move_symbol(str(file_path), "g", "h", language="python")

    assert result.parse_valid is True
    assert "def h():" in result.merged_code
    assert result.merged_code.index("def h") < result.merged_code.index("def g")
    # No bare-LF/mixed-ending or doubled-content corruption.
    assert result.merged_code.count("def g") == 1
    assert result.merged_code.count("def h") == 1
