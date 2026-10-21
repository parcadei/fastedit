"""Hermetic e2e coverage for issues #17, #18, #19 (defects 1-2), #21, #22.

Every test below reproduces the exact shapes the corresponding GitHub issue
reports, runs `fastedit edit` as a subprocess (the same seam the reports
exercised), and asserts the fixed behavior — no model, no network.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [*CLI, *args], input=input_text, capture_output=True, text=True,
        check=False,
    )


@pytest.fixture
def backup_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return Path(os.environ["FASTEDIT_BACKUP_DIR"])


# ---------------------------------------------------------------------------
# Issue #17: bash non-symbol assignment is targetable by --replace-literal
# ---------------------------------------------------------------------------

BASH_FIXTURE = (
    "MODULE=old\n"
    "run() {\n"
    "  :\n"
    "}\n"
    "GPIO=17\n"
    "for pid in 1 2 3; do\n"
    "  PID=oldpid\n"
    "  echo $PID\n"
    "done\n"
)


class TestIssue17ReplaceLiteral:
    def test_top_level_assignment_replaces_hermetically(
        self, tmp_path: Path, backup_dir,
    ):
        target = tmp_path / "sample.sh"
        target.write_text(BASH_FIXTURE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", "PID=oldpid",
            "--snippet", "  PID=123",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        content = target.read_text(encoding="utf-8")
        assert content == BASH_FIXTURE.replace("  PID=oldpid", "  PID=123")
        assert "0 tokens" in result.stdout

    def test_missing_literal_refuses_and_preserves(self, tmp_path, backup_dir):
        target = tmp_path / "sample.sh"
        target.write_text(BASH_FIXTURE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", "PID=never",
            "--snippet", "  PID=123",
        )
        assert result.returncode == 1
        assert "literal not found" in result.stderr
        assert target.read_text(encoding="utf-8") == BASH_FIXTURE

    def test_empty_snippet_refuses_with_delete_remedy(self, tmp_path, backup_dir):
        target = tmp_path / "sample.sh"
        target.write_text(BASH_FIXTURE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", "PID=oldpid",
            "--snippet", "",
        )
        assert result.returncode == 1
        assert "--lines" in result.stderr
        assert target.read_text(encoding="utf-8") == BASH_FIXTURE

    def test_ordinal_suffix_wins_when_line_ends_in_hash_digit(
        self, tmp_path, backup_dir,
    ):
        """A literal whose own text carries `#N` is read as the Nth
        occurrence of the head (documented in --replace-literal help);
        --occurrence is the unambiguous spelling."""
        target = tmp_path / "sample.sh"
        target.write_text(BASH_FIXTURE, encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", "echo $PID#1",
            "--snippet", "echo hi",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "echo hi" in target.read_text(encoding="utf-8")


class TestIssue18RepeatedLinesAndRanges:
    TS_FIXTURE = (
        "export function summarize(mode: number, failed: string[], total: number): string[] {\n"
        "  const summary: string[] = [];\n"
        "  if (mode === 1) {\n"
        '    summary.push("GROUP REPORTS");\n'
        "    if (failed.length > 0) {\n"
        '      summary.push("", "FAILED:");\n'
        "    }\n"
        "  } else {\n"
        '    summary.push("MERGED REPORT");\n'
        "    if (failed.length > 0) {\n"
        '      summary.push("", "FAILED:");\n'
        "    }\n"
        "  }\n"
        "  function advanceModel(id: string): void {\n"
        "    summary.push(id);\n"
        "  }\n"
        '  advanceModel("done");\n'
        "  return summary;\n"
        "}\n"
    )

    def _fixture(self, tmp_path: Path) -> Path:
        target = tmp_path / "sample.ts"
        target.write_text(self.TS_FIXTURE, encoding="utf-8")
        return target

    def test_uncircumscribed_repeat_refuses_counting_occurrences(
        self, tmp_path, backup_dir,
    ):
        target = self._fixture(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", 'summary.push("", "FAILED:");',
            "--snippet", '      summary.push("X");',
        )
        assert result.returncode == 1
        assert "matches 2 lines" in result.stderr
        assert "#1" in result.stderr and "#2" in result.stderr
        assert target.read_text(encoding="utf-8") == before

    def test_ordinal_suffix_replaces_exactly_one_occurrence(
        self, tmp_path, backup_dir,
    ):
        target = self._fixture(tmp_path)
        new = '      summary.push("", `FAILED (${failed.length} of ${total}):`);'
        result = run_cli(
            "edit", str(target),
            "--replace-literal", 'summary.push("", "FAILED:");#1',
            "--snippet", new,
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[5] == new
        assert lines[10] == '      summary.push("", "FAILED:");'

    def test_occurrence_flag_replaces_exactly_one_occurrence(
        self, tmp_path, backup_dir,
    ):
        target = self._fixture(tmp_path)
        new = '      summary.push("", `FAILED (${failed.length} of ${total}):`);'
        result = run_cli(
            "edit", str(target),
            "--occurrence", "2",
            "--replace-literal", 'summary.push("", "FAILED:");',
            "--snippet", new,
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[5] == '      summary.push("", "FAILED:");'
        assert lines[10] == new

    def test_occurrence_out_of_range_refuses(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--replace-literal", 'summary.push("", "FAILED:");#3',
            "--snippet", "x",
        )
        assert result.returncode == 1
        assert "occurrence #3 does not exist" in result.stderr
        assert target.read_text(encoding="utf-8") == before


    def test_lines_range_with_snippet_replaces_two_lines(
        self, tmp_path, backup_dir,
    ):
        """The deletion-only --lines gap (#18): one occurrence of the
        repeated label replaced by line range."""
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--lines", "6:6",
            "--snippet", '      summary.push("FAILED:");',
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[5] == '      summary.push("FAILED:");'
        assert lines[10] == '      summary.push("", "FAILED:");'

    def test_lines_range_rejects_parse_breaking_snippet(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--lines", "2:2",
            "--snippet", "const summary = [",
        )
        assert result.returncode == 1
        assert "parse errors" in result.stderr
        assert target.read_text(encoding="utf-8") == before

    def test_lines_replace_noop_refuses(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target),
            "--lines", "2:2",
            "--snippet", "  const summary: string[] = [];",
        )
        assert result.returncode == 1
        assert "No change resulted" in result.stderr
        assert target.read_text(encoding="utf-8") == before

    def test_nested_function_not_found_states_recipe(self, tmp_path, backup_dir):
        """#18 item 5: the actionable alternative, not a bare not-found."""
        target = self._fixture(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli("edit", str(target), "--replace", "advanceModel",
                         "--snippet", "  function advanceModel(id: string): void {}")
        assert result.returncode == 1
        assert "not found" in result.stderr
        assert "nested INSIDE another function's body" in result.stderr
        assert "--replace-literal" in result.stderr
        assert target.read_text(encoding="utf-8") == before


class TestIssue22IndentInheritance:
    TS_FIXTURE = (
        "function a() {\n"
        "  // a comment line\n"
        "  const y = 1\n"
        "}\n"
    )

    def _fixture(self, tmp_path: Path) -> Path:
        target = tmp_path / "demo.ts"
        target.write_text(self.TS_FIXTURE, encoding="utf-8")
        return target

    def test_column0_snippet_inherits_anchor_indent(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--insert-after", "// a comment line",
            "--snippet", "// WHY: carried comment\n// second line",
            input_text=None,
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "indentation inherited" in result.stdout
        assert target.read_text(encoding="utf-8") == (
            "function a() {\n"
            "  // a comment line\n"
            "  // WHY: carried comment\n"
            "  // second line\n"
            "  const y = 1\n"
            "}\n"
        )

    def test_insert_before_inherits_too(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--insert-before", "const y = 1",
            "--snippet", "const guard = check()",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert target.read_text(encoding="utf-8") == (
            "function a() {\n"
            "  // a comment line\n"
            "  const guard = check()\n"
            "  const y = 1\n"
            "}\n"
        )

    def test_own_indentation_stays_verbatim(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--insert-after", "// a comment line",
            "--snippet", "      const deep = 1",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "indentation inherited" not in result.stdout
        assert target.read_text(encoding="utf-8") == (
            "function a() {\n"
            "  // a comment line\n"
            "      const deep = 1\n"
            "  const y = 1\n"
            "}\n"
        )

    def test_column0_anchor_inherits_nothing(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--insert-after", "}",
            "--snippet", "// appended",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "indentation inherited" not in result.stdout
        assert target.read_text(encoding="utf-8") == (
            "function a() {\n"
            "  // a comment line\n"
            "  const y = 1\n"
            "}\n"
            "// appended\n"
        )

    def test_replace_literal_inherits_replaced_line_indent(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target),
            "--replace-literal", "const y = 1",
            "--snippet", "const y = 1, z = 2",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "indentation inherited" in result.stdout
        assert target.read_text(encoding="utf-8") == (
            "function a() {\n"
            "  // a comment line\n"
            "  const y = 1, z = 2\n"
            "}\n"
        )


class TestIssue21BlankLineNoopHonesty:
    FIXTURE = (
        "def a():\n"
        "    return 1\n"
        "\n"
        "\n"
        "\n"
        "def b():\n"
        '    """Doc."""\n'
        "    return 2\n"
    )

    def _fixture(self, tmp_path: Path) -> Path:
        target = tmp_path / "repro.py"
        target.write_text(self.FIXTURE, encoding="utf-8")
        return target

    def test_exact_restatement_generic_noop(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target), "--replace", "b",
            "--snippet", "def b():\n    \"\"\"Doc.\"\"\"\n    return 2",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert "No change resulted" in result.stdout
        assert target.read_text(encoding="utf-8") == self.FIXTURE

    def test_blank_difference_names_recipe_not_match_claim(self, tmp_path, backup_dir):
        target = self._fixture(tmp_path)
        result = run_cli(
            "edit", str(target), "--replace", "b",
            "--snippet",
            "\ndef b():\n    \"\"\"Doc.\"\"\"\n    return 2",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        out = result.stdout
        assert "blank line" in out
        assert "OUTSIDE the symbol's span" in out
        assert "--lines" in out
        assert "matches the current content" not in out
        assert target.read_text(encoding="utf-8") == self.FIXTURE


class TestIssue19FailFastAndFragmentKeys:
    MD_FIXTURE = (
        "## Scenario\n"
        "\n"
        "The REVIEW fork uses `subagent_type: \"fork\"`, prompt starting `ROLE: REVIEW`, and runs\n"
        "**synchronously — never `run_in_background`** (the one sanctioned exception) so the\n"
        "main agent below can react at once.\n"
        "\n"
    )

    def test_rewording_collapsing_lines_refuses_quickly_and_names_cause(
        self, tmp_path, backup_dir,
    ):
        """The 11-20-minute futile retry loop: an M-lines -> one-line
        markdown rewording cannot converge; the battery repeats its
        rejection, the retry loop stops early instead of exhausting the
        budget, and the CLI refuses naming the dropped line."""
        target = tmp_path / "rules.md"
        target.write_text(self.MD_FIXTURE, encoding="utf-8")
        snippet = (
            "## Scenario\n"
            "\n"
            "One reworded line replacing the three above.\n"
        )
        result = run_cli(
            "edit", str(target), "--replace", "Scenario", "--snippet", snippet,
        )
        assert result.returncode == 1, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        # The early-stop mechanism is the fix: the hopeless shape is
        # refused after TWO attempts (metrics name "2 validation
        # retries"), not nine — and the file is untouched.
        assert "2 validation retries" in result.stderr
        assert target.read_text(encoding="utf-8") == self.MD_FIXTURE

    def test_repeated_rejection_stops_retries_early(self):
        """Unit: two identical battery rejections end the loop after two
        attempts instead of exhausting the whole retry budget (the
        mechanism that turned an 11-20-minute refusal into seconds)."""
        import os

        from fastedit.inference.chunked_merge import chunked_merge
        from fastedit.inference.merge import MergeResult

        original = "## H\n\nline one.\nline two.\n"
        snippet = "## H\n\none reworded line.\n"

        def merge_fn(chunk, s, language):
            # Always echo the snippet's intent as a merge that drops the
            # second line — a battery rejection, word after word.
            return MergeResult(
                merged_code="## H\n\none reworded line.\n",
                parse_valid=True,
                tokens_generated=1,
                latency_ms=1.0,
                tokens_per_second=1.0,
            )

        # FASTEDIT_MAX_RETRIES must not matter: the identical-rejection
        # streak ends the loop at attempt two.
        os.environ["FASTEDIT_MAX_RETRIES"] = "8"
        result = chunked_merge(
            original_code=original,
            snippet=snippet,
            file_path="hopeless.md",
            merge_fn=merge_fn,
            language="markdown",
        )
        assert result.chunks_rejected >= 1
        assert result.retries == 1, result

    def test_fragment_frontmatter_keys_vouched_by_document(self):
        """#19 (defect 1): `key:`-shaped PROSE lines in a frontmatter-less
        document are not frontmatter keys — and would no longer refuse the
        whole merge. Behind a real frontmatter document they still are."""
        from fastedit.inference.chunked_merge import _derived_attribute_changes

        prose = "Some intro.\nWhy: the reason.\nTail.\n"
        # vouched: a document that starts with --- still declares them.
        declared = _derived_attribute_changes(
            "---\nname: x\n---\nbody\n", prose, "markdown",
        )
        assert any(
            d.kind == "md_frontmatter_key" and d.value == "Why: the reason."
            for d in declared.declared
        )
        # NOT vouched: the reported 118-line agent-rules file has no
        # frontmatter, so the wrapped paragraph's key:-shaped word is prose.
        declared = _derived_attribute_changes(prose, prose, "markdown")
        assert not any(
            d.kind == "md_frontmatter_key" for d in declared.declared
        )

    def test_real_md_document_without_frontmatter_merges_now(
        self, tmp_path, backup_dir,
    ):
        """End-to-end: editing one line in a frontmatter-less md file that
        contains an `Applies when:`-shaped prose line used to fail the
        fragment key declaration; the line-range replace path plus the
        battery now accept a total restatement of the same shape."""
        doc = (
            "## Section\n"
            "\n"
            "questions: *a* and *b*\n"
            "\n"
            "Detail line.\n"
            "\n"
        )
        target = tmp_path / "rules.md"
        target.write_text(doc, encoding="utf-8")
        from fastedit.inference.chunked_merge import (
            _attribute_rejection_reason,
        )

        # The full restatement with identical prose must be attribute-OK:
        reason = _attribute_rejection_reason(doc, doc, doc, "markdown")
        assert reason is None, reason
