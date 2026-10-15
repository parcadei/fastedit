"""Issue #8: markdown ``--replace`` tail drops + ``fastedit diff`` coverage.

Part (a): a snippet restating a section's heading + SOME body lines but
ending MID-SENTENCE (its final line is a strict prefix of an original body
line) used to land as a direct swap: the text-match editor declined the
shape (leading-token rewrite conflict), and the completeness check for
format symbols only COUNTED declared-vs-deleted content lines — the
truncated line counted as a new line covering the deletion, so the tail
(``... config is valid.``) was dropped with exit 0.

Part (b): ``fastedit diff`` used to diff the current file against the NEWEST
backup (``BackupStore.peek``), i.e. against the LAST edit's pre-state. A
loss from an EARLIER edit vanished from the diff as soon as any later edit
landed — the diff showed only the newest intended hunk. Issue #8 fixed that
by diffing against the OLDEST surviving backup so the rendered diff covers
every change still in the undo history.

Issue #15 made that loss-detection view OPT-IN: the default base is now the
NEWEST backup (`--base last`, exactly the last edit's change — what an
agent verifying an edit expects), while `--base all` keeps the issue-#8
whole-history view. Context n stays the standard 3: adjacent hunks merge
automatically, distant hunks each print in full — every changed line
appears as a removal or addition either way; the base, not the context,
was the gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastedit.inference.chunked_merge import (
    _snippet_splice_covers_deleted_lines,
    chunked_merge,
)

MD_ORIGINAL = """# Guide

## Ambiguity handling

When a snippet ends mid-sentence the editor must
refuse the edit instead of dropping the tail
without ambiguity the config is valid.

## Next section

Still here.
"""

MD_MID_SENTENCE_SNIPPET = """## Ambiguity handling

When a snippet ends mid-sentence the editor must
refuse the edit instead of dropping the tail
without ambiguity the
"""

SECTION_SPAN = (
    "## Ambiguity handling\n"
    "\n"
    "When a snippet ends mid-sentence the editor must\n"
    "refuse the edit instead of dropping the tail\n"
    "without ambiguity the config is valid.\n"
)


# ---------------------------------------------------------------------------
# (a) the completeness gate catches the mid-sentence truncation
# ---------------------------------------------------------------------------


class TestCompletenessGate:
    def test_mid_sentence_suffix_does_not_cover_the_deletion(self):
        """The exact reported shape: the snippet's final line is a strict
        prefix of the deleted original line."""
        assert _snippet_splice_covers_deleted_lines(
            SECTION_SPAN, MD_MID_SENTENCE_SNIPPET.rstrip("\n") + "\n",
        ) is False

    def test_full_restatement_still_passes(self):
        complete = MD_MID_SENTENCE_SNIPPET.replace(
            "without ambiguity the\n",
            "without ambiguity the config is valid.\n",
        )
        assert _snippet_splice_covers_deleted_lines(
            SECTION_SPAN, complete,
        ) is True

    def test_legitimate_full_rewrite_still_passes(self):
        rewritten = (
            "## Ambiguity handling\n"
            "\n"
            "The editor now refuses truncated snippets\n"
            "instead of silently dropping the tail\n"
            "so nothing is lost.\n"
        )
        assert _snippet_splice_covers_deleted_lines(
            SECTION_SPAN, rewritten,
        ) is True

    def test_dropping_a_whole_line_still_fails_the_count(self):
        partial = (
            "## Ambiguity handling\n"
            "\n"
            "When a snippet ends mid-sentence the editor must\n"
        )
        assert _snippet_splice_covers_deleted_lines(
            SECTION_SPAN, partial,
        ) is False


class TestDirectSwapDeclines:
    def test_chunked_merge_direct_swap_declines_mid_sentence_shape(self):
        """chunked_merge's replace= direct swap must decline (merge_fn
        raising proves the swap never landed)."""
        def _no_model(*a, **k):
            raise RuntimeError("model must not be called")

        with pytest.raises(RuntimeError, match="model must not be called"):
            chunked_merge(
                original_code=MD_ORIGINAL,
                snippet=MD_MID_SENTENCE_SNIPPET,
                file_path="doc.md",
                merge_fn=_no_model,
                language="markdown",
                replace="Ambiguity handling",
            )


# ---------------------------------------------------------------------------
# (b) fastedit diff surfaces losses from EARLIER edits
# ---------------------------------------------------------------------------


def run_cli(*args: str, env_extra: dict | None = None):
    import os
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        capture_output=True, text=True, timeout=30, env=env, check=False,
    )


PY_ORIGINAL = """def target():
    a = 1

def helper():
    b = 2
"""


class TestDiffCoverage:
    """The reported symptom: after edit 2 landed, `fastedit diff` showed
    only edit 2's hunk — the line edit 1 REMOVED was identical on both
    sides of the newest-backup base and vanished from the diff."""

    def _edit(self, target: Path, snippet: str, replace: str):
        result = run_cli(
            "edit", str(target), "--snippet", snippet, "--replace", replace,
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )

    def test_diff_surfaces_removal_from_previous_edit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL)

        # Edit 1 (a sanctioned deterministic deletion): removes the whole
        # `helper` function.
        deleted = run_cli("delete", str(target), "helper")
        assert deleted.returncode == 0, (
            f"stdout: {deleted.stdout!r} stderr: {deleted.stderr!r}"
        )
        assert "b = 2" not in target.read_text()

        # Edit 2: rewrites `a = 1` -> `a = 2`.
        self._edit(target, "def target():\n    a = 2\n", "target")
        content = target.read_text()
        assert "a = 2" in content and "b = 2" not in content

        diff = run_cli("diff", str(target), "--base", "all")
        assert diff.returncode == 0, diff.stderr
        # The removal from the EARLIER edit appears even though the newest
        # edit's pre-state still contained it.
        assert "-def helper():" in diff.stdout, diff.stdout
        assert "-    b = 2" in diff.stdout, diff.stdout
        # The newest edit's change is surfaced too.
        assert "-    a = 1" in diff.stdout
        assert "+    a = 2" in diff.stdout

    def test_diff_after_single_edit_shows_that_edit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL)
        self._edit(target, "def target():\n    a = 2\n", "target")

        diff = run_cli("diff", str(target))
        assert diff.returncode == 0, diff.stderr
        assert "-    a = 1" in diff.stdout
        assert "+    a = 2" in diff.stdout

    def test_no_backup_recorded_message_when_history_is_empty(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
        target = tmp_path / "app.py"
        target.write_text(PY_ORIGINAL)
        diff = run_cli("diff", str(target))
        assert diff.returncode == 0
        assert "No backup recorded" in diff.stdout
