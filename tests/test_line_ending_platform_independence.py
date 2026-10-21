"""Regression tests: detect_line_ending must never depend on the host OS.

TRDD-J8Q28MED: detect_line_ending() used to fall back to os.linesep on two
paths -- an exact count tie between styles, and text with NO line ending at
all (the more common of the two). os.linesep is "\n" on macOS/Linux and
"\r\n" on Windows, so the same edit to the same file emitted different
bytes depending on which machine ran fastedit. Both paths must now resolve
to a fixed, host-independent value ("\n") regardless of os.linesep.
"""

from __future__ import annotations

import os

from fastedit.split_join import detect_line_ending


def test_dominant_lf_wins_over_minority_crlf():
    """A file where LF clearly outnumbers CRLF must resolve to LF."""
    text = "a\nb\nc\nd\r\n"
    assert detect_line_ending(text) == "\n"


def test_dominant_crlf_wins_over_minority_lf():
    """A file where CRLF clearly outnumbers LF must resolve to CRLF."""
    text = "a\r\nb\r\nc\r\nd\n"
    assert detect_line_ending(text) == "\r\n"


def test_exact_tie_is_deterministic_not_os_linesep():
    """An exact count tie must resolve to a fixed value, not os.linesep."""
    tied = "a\nb\r\nc"  # lone_lf=1, crlf=1 -> tie
    assert detect_line_ending(tied) == "\n"


def test_no_line_ending_at_all_is_deterministic_not_os_linesep():
    """Text with zero line endings (single line) must resolve to a fixed value."""
    assert detect_line_ending("single line, no terminator") == "\n"


def test_cr_only_file():
    """A file using only classic-Mac lone CR terminators must resolve to CR."""
    text = "a\rb\rc\r"
    assert detect_line_ending(text) == "\r"


def test_crlf_only_file():
    """A file using only CRLF terminators must resolve to CRLF."""
    text = "a\r\nb\r\nc\r\n"
    assert detect_line_ending(text) == "\r\n"


def test_lf_only_file():
    """A file using only LF terminators must resolve to LF."""
    text = "a\nb\nc\n"
    assert detect_line_ending(text) == "\n"


def test_empty_string_is_deterministic_not_os_linesep():
    """An empty string carries no line-ending signal and must resolve to a fixed value."""
    assert detect_line_ending("") == "\n"


def test_result_unchanged_when_os_linesep_is_crlf(monkeypatch):
    """KEY REGRESSION TEST: monkeypatch os.linesep to CRLF and confirm the
    output bytes are unaffected on both the zero-count and tie paths -- this
    is exactly what would go red if os.linesep were reintroduced anywhere in
    detect_line_ending.
    """
    monkeypatch.setattr(os, "linesep", "\r\n")
    assert detect_line_ending("single line, no terminator") == "\n"
    assert detect_line_ending("a\nb\r\nc") == "\n"  # exact tie: lone_lf=1, crlf=1


def test_result_unchanged_when_os_linesep_is_lf(monkeypatch):
    """Same regression check with os.linesep forced to LF, so a test run on
    a real macOS/Linux host cannot coincidentally pass by matching the host
    default instead of the fixed contract.
    """
    monkeypatch.setattr(os, "linesep", "\n")
    assert detect_line_ending("single line, no terminator") == "\n"
    assert detect_line_ending("a\nb\r\nc") == "\n"
