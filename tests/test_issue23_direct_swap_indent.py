"""GitHub issue #23: the ``--replace`` direct-swap splice landed a snippet
VERBATIM (``result_lines[func_start:func_end] = snippet_lines``), so a
column-0 snippet put the replacement symbol and its whole body at column
0 — OUTSIDE the class/method container it replaces — and the tool reported
success with exit 0 and no warning. Still parses (brace languages), still
type-checks; only reading the file shows the method left its class.

Fix contract (fail-safe reindent):

  * the direct-swap splice auto-re-indents the snippet to the TARGET
    symbol's own leading-indent level, shifting every non-blank line by
    the same column delta (relative structure preserved; blank lines pass
    through);
  * a snippet already at the target's level is a byte-for-byte NO-OP;
  * ``--indent N`` (spaces) is the explicit override of that level;
  * the same alignment governs the chunked_merge direct-swap branch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["FASTEDIT_NO_UPDATE_CHECK"] = "1"
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


TS_COUNTER = (
    "export class Counter {\n"
    "  private n = 0;\n"
    "\n"
    "  add(a: number): void {\n"
    "    this.n = a;\n"
    "  }\n"
    "}\n"
)

TS_COLUMN0_SNIPPET = (
    "add(a: number): void {\n"
    "this.n = a + 1\n"
    "}\n"
)

# Relative structure preserved: the snippet carried a FLAT body (column-0
# body line under a column-0 head), so after the +2 shift the body stays
# at the head's level — the issue's table: emitted indent == target level.
TS_COUNTER_EXPECTED = (
    "export class Counter {\n"
    "  private n = 0;\n"
    "\n"
    "  add(a: number): void {\n"
    "  this.n = a + 1\n"
    "  }\n"
    "}\n"
)

PYTHON_CLASS = (
    "class Counter:\n"
    "\n"
    "    def add(self, a):\n"
    "        return a\n"
    "\n"
    "    def sub(self, a):\n"
    "        return a\n"
)


class TestIssue23DirectSwapAutoReindent:
    def test_direct_swap_rename_column0_snippet_lands_at_target_level(
        self, tmp_path: Path,
    ):
        """A rename routes the edit past the text-match editor to the
        direct swap (issue #23's trigger boundary list): a column-0
        snippet must still land at the target symbol's level."""
        target = tmp_path / "mod.py"
        target.write_text(PYTHON_CLASS, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--replace", "Counter.add",
            "--snippet", "def increment(self, a):\n    return a + 1\n",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        content = target.read_text(encoding="utf-8")
        assert content == (
            "class Counter:\n"
            "\n"
            "    def increment(self, a):\n"
            "        return a + 1\n"
            "\n"
            "    def sub(self, a):\n"
            "        return a\n"
        ), content

    def test_ts_column0_snippet_lands_inside_class(self, tmp_path: Path):
        """The issue's exact repro: a column-0 method snippet --replace'd
        onto a 2-space-indented class method must land INSIDE the class at
        the 2-space level, not at column 0, with exit 0."""
        target = tmp_path / "demo.ts"
        target.write_text(TS_COUNTER, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--replace", "Counter.add",
            "--snippet", TS_COLUMN0_SNIPPET,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Applied edit to" in result.stdout
        # Byte-exact: head of the splice carries the target's 2-space level.
        assert target.read_text(encoding="utf-8") == TS_COUNTER_EXPECTED, (
            target.read_text(encoding="utf-8")
        )

    def test_python_method_column0_snippet_keeps_class_membership(
        self, tmp_path: Path,
    ):
        """Python variant: a column-0 snippet replacing a class method must
        land INSIDE the class at the method's 4-space level (whichever
        deterministic route serves it), never flat at column 0."""
        target = tmp_path / "mod.py"
        target.write_text(PYTHON_CLASS, encoding="utf-8")

        result = run_cli(
            "edit", str(target),
            "--replace", "Counter.add",
            "--snippet", "def add(self, a):\n    return a + 1\n",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        content = target.read_text(encoding="utf-8")
        # Method stayed a member of the class (4-space level), body kept.
        assert "    def add(self, a):\n" in content, content
        assert "return a + 1" in content, content
        assert "\ndef add(self, a):" not in content, content

    def test_deeper_snippet_normalizes_down_to_target_level(
        self, tmp_path: Path,
    ):
        """A snippet written deeper than its target is normalized DOWN to
        the target's level (relative structure preserved)."""
        target = tmp_path / "mod.py"
        target.write_text(
            "class Widget:\n"
            "\n"
            "    def paint(self):\n"
            "        return True\n"
        )
        snippet = (
            "        def paint(self):\n"
            "            return False\n"
        )

        result = run_cli(
            "edit", str(target),
            "--replace", "Widget.paint",
            "--snippet", snippet,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert target.read_text(encoding="utf-8") == (
            "class Widget:\n"
            "\n"
            "    def paint(self):\n"
            "        return False\n"
        )

    def test_snippet_already_at_target_indent_is_a_noop_reindent(
        self, tmp_path: Path,
    ):
        """A snippet already written at the target's level is spliced
        byte-exact: the auto-reindent adds no drift and applies no shift."""
        target = tmp_path / "mod.py"
        target.write_text(
            "class Widget:\n"
            "\n"
            "    def paint(self):\n"
            "        return True\n"
        )
        snippet = (
            "    def paint(self):\n"
            "        return False\n"
        )

        result = run_cli(
            "edit", str(target),
            "--replace", "Widget.paint",
            "--snippet", snippet,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert target.read_text(encoding="utf-8") == (
            "class Widget:\n"
            "\n"
            "    def paint(self):\n"
            "        return False\n"
        )


class TestIssue23IndentOverride:
    def test_indent_n_overrides_the_target_derived_level(self, tmp_path: Path):
        """--indent N is the escape hatch for a DIRECT-SWAP splice (a
        rename routes past the text-match editor): the head gets N leading
        spaces instead of the target's level. Top-level target so the
        override produces parse-valid output (an in-class method re-level
        with mixed member indents would be refused, correctly)."""
        target = tmp_path / "mod.py"
        target.write_text(
            "def greet(name):\n"
            "    return name\n"
            "\n"
            "def other():\n"
            "    return 1\n"
        )

        result = run_cli(
            "edit", str(target),
            "--replace", "greet",
            "--snippet", "def welcome(name):\n    return name\n",
            "--indent", "4",
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        content = target.read_text(encoding="utf-8")
        assert content == (
            "    def welcome(name):\n"
            "        return name\n"
            "\n"
            "def other():\n"
            "    return 1\n"
        ), content

    def test_indent_flag_is_documented(self):
        result = run_cli("edit", "--help")
        assert result.returncode == 0
        assert "--indent" in result.stdout


class TestIssue23RelativeStructurePreserved:
    def test_nested_body_keeps_relative_depth(self, tmp_path: Path):
        """Interior lines DEEPER than the head in the snippet keep that
        relative depth after the shift."""
        target = tmp_path / "mod.py"
        target.write_text(PYTHON_CLASS, encoding="utf-8")
        snippet = (
            "def add(self, a):\n"
            "    if a:\n"
            "        return 1\n"
            "    return 0\n"
        )

        result = run_cli(
            "edit", str(target),
            "--replace", "Counter.add",
            "--snippet", snippet,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        content = target.read_text(encoding="utf-8")
        assert content == (
            "class Counter:\n"
            "\n"
            "    def add(self, a):\n"
            "        if a:\n"
            "            return 1\n"
            "        return 0\n"
            "\n"
            "    def sub(self, a):\n"
            "        return a\n"
        ), content
