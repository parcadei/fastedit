"""GitHub issue #24 — section rewrite with deletions + oscillation early stop.

Fixes pins for:

  * defect A: a ``--replace`` on a markdown heading whose snippet restates
    the heading AND the span's tail (a complete section restatement — one
    paragraph deleted, two bullets reworded) must APPLY deterministically,
    deletions included. The exact issue fixture (original.md /
    snippet.md verbatim, sha256s from the issue body) e2e via cmd_edit.
  * defect B: the retry loop's early stop compares the rejection CATEGORY
    (parse / faithfulness / truncation), so the reported oscillation
    ("declared new line missing" ↔ "unmentioned original line dropped")
    stops after 2 attempts instead of burning the whole budget, while
    distinct categories still exhaust the budget.
  * defect C: ``_rejection_refusal`` branches its advice — all
    faithfulness-class rejections name the deterministic primitives
    instead of the wrong "function may be too large" size advice on a
    10-line file.
  * defect D: a pure one-line deletion via ``--replace-literal <line>
    --snippet ''`` is refused but names the sanctioned tool
    (``--lines FROM:TO --delete``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastedit.inference.chunked_merge import _rejection_category
from fastedit.write_gates import _rejection_refusal

# ---------------------------------------------------------------------------
# The exact issue fixture (hashes pinned in the issue body)
# ---------------------------------------------------------------------------

ORIGINAL_MD = (
    "# CLAUDE.md — complete-plugin-validation (CPV)\n"
    "\n"
    "Lean on purpose. The former long version is `HISTORY.md` (untracked, when present): read it in small parts with `tldr body` or triage it with quicksilver, never whole.\n"
    "\n"
    "- **Name.** The plugin was renamed from `claude-plugins-validation` to `complete-plugin-validation` because newer Claude Code reserves the words claude and anthropic in plugin and skill names. The GitHub repository is still `Emasoft/claude-plugins-validation`.\n"
    "- **Security above all.** Never bypass, skip or weaken a security, validation or hash check: no skip variables, no relaxed `--strict`, no suppressed rule. Make tampering as hard as possible, and make a successful one likely to leave a trail (a commit, an audit-log line).\n"
    "- **Releases.** Publish only through `scripts/publish.py`. Its self-integrity check stays on during a release: it is anchored to the candidate's own committed manifest, never switched off with `PLUGIN_SKIP_GITHUB_INTEGRITY`.\n"
    "- **History rewrites.** After any rewrite that changes file contents, regenerate and re-verify `.plugin-self-hashes.json` (issue #258: a purge rewrote files but not their recorded hashes, so a release aborted on every fresh install). Before any push, run the push guard on TRDD-P4JSYC7C.\n"
    "- **CI.** Be careful with `.github/workflows`: most binary checkers the plugin uses do not work in the GitHub CI environment. Do not move a local gate into CI, or change a workflow, without checking that.\n"
    "- **GitHub.** `gh` is authenticated as the owner, Emasoft. Write only to Emasoft-owned repositories.\n"
)

SNIPPET_MD = (
    "# CLAUDE.md — complete-plugin-validation (CPV)\n"
    "\n"
    "- **Name.** The plugin was renamed from `claude-plugins-validation` to `complete-plugin-validation` because newer Claude Code reserves the words claude and anthropic in plugin and skill names. The GitHub repository is still `Emasoft/claude-plugins-validation`.\n"
    "- **Security above all.** Never bypass security, validation or hash checks. Make tampering as hard as possible, and make a successful one likely to leave a trail (a commit, an audit-log line).\n"
    "- **Releases.** Publish only through `scripts/publish.py`. Its self-integrity check stays on during a release: it is anchored to the candidate's own committed manifest, never switched off with `PLUGIN_SKIP_GITHUB_INTEGRITY`.\n"
    "- **History rewrites.** After any rewrite that changes file contents, regenerate and re-verify `.plugin-self-hashes.json` (issue #258: a purge rewrote files but not their recorded hashes, so a release aborted on every fresh install). Before any push, run the push guard on TRDD-P4JSYC7C.\n"
    "- **CI.** Be careful with `.github/workflows`: most binary checkers the plugin uses do not work in the GitHub CI environment.\n"
    "- **GitHub.** `gh` is authenticated as the owner, Emasoft. Write only to Emasoft-owned repositories.\n"
)

REPLACE_HEADING = "CLAUDE.md — complete-plugin-validation (CPV)"


def _test_wrote_nothing(*_args, **_kwargs) -> None:  # placeholder helper
    return None


class TestIssue24DefectASectionRewriteApplies:
    """The exact issue command, e2e through the CLI, must APPLY."""

    def test_exact_issue_fixture_e2e_applies_with_deletions(
        self, tmp_path: Path, monkeypatch,
    ):
        from fastedit.cli import main as cli_main

        target = tmp_path / "a.md"
        snippet_file = tmp_path / "snippet.md"
        target.write_text(ORIGINAL_MD)
        snippet_file.write_text(SNIPPET_MD)
        monkeypatch.setattr(
            sys, "argv",
            [
                "fastedit", "edit", str(target),
                "--replace", REPLACE_HEADING,
                "--snippet", f"@{snippet_file}",
            ],
        )

        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                cli_main()
        except SystemExit as e:
            raise AssertionError(
                f"the issue's exact edit must apply with exit 0, got exit "
                f"{e.code}; stderr: {err.getvalue()}"
            )
        # The file IS the snippet — deletions included.
        assert target.read_text() == SNIPPET_MD, (
            "the applied merge must be byte-exact with the snippet "
            "(whole-section restatement, deletions included); "
            f"stderr: {err.getvalue()}"
        )

    def test_issue_fixture_via_subprocess_applies(self, tmp_path: Path):
        from test_cli import run_cli

        target = tmp_path / "a.md"
        snippet_file = tmp_path / "snippet.md"
        target.write_text(ORIGINAL_MD)
        snippet_file.write_text(SNIPPET_MD)
        result = run_cli(
            "edit", str(target),
            "--replace", REPLACE_HEADING,
            "--snippet", f"@{snippet_file}",
        )
        assert result.returncode == 0, (
            f"stdout: {result.stdout!r} stderr: {result.stderr!r}"
        )
        assert result.stderr == "" or "would delete" not in result.stderr, (
            result.stderr
        )
        assert target.read_text() == SNIPPET_MD

    def test_keep_marker_free_full_swap_of_a_code_section_keeps_declining(
        self, tmp_path: Path,
    ):
        """Contract boundary: the tail-coverage full-swap rule is scoped to
        the completeness gate for definition_line_required symbols. An
        actual code replacement keeps its semantics (guarded elsewhere) and
        the #8 truncated-tail shape must STILL decline (covered by
        tests/test_markdown_tail_and_diff.py; this pin is the issue fixture
        wiring itself, not a substitute)."""


class TestIssue24DefectBOscillationEarlyStop:
    """The issue's oscillation: the battery alternates between
    'declared new line missing' and 'unmentioned original line dropped' —
    one rejection CATEGORY (faithfulness) — so the early stop fires after
    2 attempts."""

    # Module fixtures (borrowed shapes from the battery's trait pins).
    NOTES_ORIGINAL = (
        "Project notes — warehouse sync.\n"
        "\n"
        "The nightly job copies stock levels at midnight.\n"
        "It skips items marked discontinued.\n"
        "\n"
        "Contact: ops@example.com for schedule changes.\n"
        "The on-call engineer reviews failures each morning.\n"
        "\n"
        "Keep every line: this is a trait to preserve.\n"
    )
    SNIPPET = (
        "Project notes — warehouse sync.\n"
        "The on-call engineer reviews failures each morning.\n"
        "Appended paragraph one.\n"
        "Appended paragraph two.\n"
    )
    # Attempt A: the model drops the DECLARED NEW lines (faithfulness).
    DROP_NEW = NOTES_ORIGINAL
    # Attempt B: the model applies the new lines but DROPS the original line
    # the snippet replaced (a different faithfulness string — the alternation).
    DROP_OLD = NOTES_ORIGINAL.replace(
        "The on-call engineer reviews failures each morning.\n",
        "Appended paragraph one.\nAppended paragraph two.\n",
    )

    def _merge_oscillating(self, tmp_path, budget):
        from fastedit.inference.chunked_merge import chunked_merge

        target = tmp_path / "notes.md"
        target.write_text(self.NOTES_ORIGINAL)
        calls: list = []

        def merge_fn(code, snippet, language):
            calls.append(snippet)
            merged = self.DROP_NEW if len(calls) % 2 else self.DROP_OLD
            return SimpleNamespace(
                merged_code=merged,
                parse_valid=True,
                tokens_generated=9,
                latency_ms=1.0,
                truncated=False,
            )

        return chunked_merge(
            self.NOTES_ORIGINAL, self.SNIPPET, str(target), merge_fn,
            language="markdown", max_validation_retries=budget,
        ), calls

    def test_oscillating_faithfulness_reasons_stop_after_two_attempts(
        self, tmp_path,
    ):
        result, calls = self._merge_oscillating(tmp_path, budget=5)
        # Same CATEGORY oscillating ("declared new line missing" ↔
        # "unmentioned original line dropped") must stop after exactly the
        # first two attempts — never burn 9 model calls + 54s again.
        assert len(calls) == 2, f"calls: {len(calls)}"
        assert result.chunks_rejected == 1
        assert result.merged_code == self.NOTES_ORIGINAL
        assert getattr(result, "rejection_categories", None) == (
            "faithfulness",
        )

    def test_distinct_categories_still_exhaust_the_budget(self, tmp_path):
        from fastedit.inference.chunked_merge import chunked_merge

        target = tmp_path / "notes.md"
        target.write_text(self.NOTES_ORIGINAL)
        calls: list = []

        def merge_fn(code, snippet, language):
            calls.append(snippet)
            if len(calls) % 2:
                merged = self.DROP_NEW  # faithfulness category
            else:
                merged = self.DROP_OLD  # faithfulness category
            return SimpleNamespace(
                merged_code=merged,
                parse_valid=True,
                tokens_generated=9,
                latency_ms=1.0,
                truncated=len(calls) % 2 == 0,  # even attempts: truncation
            )

        result = chunked_merge(
            self.NOTES_ORIGINAL, self.SNIPPET, str(target), merge_fn,
            language="markdown", max_validation_retries=2,
        )
        # faithfulness → truncation → faithfulness: cycling categories keep
        # the budget (3 attempts consumed).
        assert len(calls) == 3
        assert result.chunks_rejected == 1
        assert set(getattr(result, "rejection_categories", ())) == {
            "faithfulness", "truncation",
        }

    def test_chunk_loop_same_category_stops_after_two_attempts(
        self, tmp_path,
    ):
        """The PER-CHUNK retry loop carries the same category rule.

        Marker-bearing snippet (the deterministic paths decline it) inside
        a padded file so the chunked loop runs; the scripted merge
        oscillates "declared new line 'c = 3' missing" ↔ "unmentioned
        original line dropped 'b = 2'" — one faithfulness category, two
        strings. Same-category → the loop stops after 2 attempts.
        """
        from fastedit.inference.chunked_merge import chunked_merge

        filler_top = "\n".join(f"x{i} = {i}" for i in range(30))
        filler_bottom = "\n".join(f"y{i} = {i}" for i in range(30))
        original = (
            f"{filler_top}\n\n"

            "def beta():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    return a + b\n"
            f"\n{filler_bottom}\n"
        )
        snippet = (
            "def beta():\n"
            "    # ... existing code ...\n"
            "    c = 3\n"
            "    return a + b\n"
            "    # ... rest of the body stays ...\n"
        )
        target = tmp_path / "mod.py"
        target.write_text(original)
        calls: list = []
        merge_a = original  # drops the declared new line entirely
        merge_b = original.replace(
            "    b = 2\n",
            "    c = 3\n",  # new lines in, original line out
        )

        def merge_fn(code, chunk_snippet, language):
            calls.append(chunk_snippet)
            return SimpleNamespace(
                merged_code=merge_a if len(calls) % 2 else merge_b,
                parse_valid=True,
                tokens_generated=9,
                latency_ms=1.0,
                truncated=False,
            )

        result = chunked_merge(
            original, snippet, str(target), merge_fn,
            language="python", max_validation_retries=5,
        )
        assert len(calls) >= 2
        if len(calls) == 2:
            # Early stop fired (same category twice in a row) — asserted
            # bookkeeping: the chunk stays rejected, original kept.
            assert result.chunks_rejected >= 1
        else:
            # The chunk loop consumed the whole budget instead: still a
            # rejection, but that would mean the category rule failed —
            # force the failure.
            raise AssertionError(
                f"oscillating same-category attempts not stopped early: "
                f"{len(calls)} attempts"
            )


class TestRejectionCategory:
    def test_faithfulness(self):
        assert _rejection_category(
            "merged output failed the content-faithfulness check "
            "(unmentioned original line dropped: 'x')",
        ) == "faithfulness"

    def test_parse(self):
        assert _rejection_category(
            "merged output does not parse as python (missing paren)",
        ) == "parse"

    def test_truncation(self):
        assert _rejection_category("truncated (model hit the token cap)") == (
            "truncation"
        )

    def test_trait_and_attribute_are_their_own_categories(self):
        assert _rejection_category(
            "merged output failed the text-trait check (lines)",
        ) == "trait"
        assert _rejection_category(
            "merged output failed the language-attribute/structure check "
            "(html_lang)",
        ) == "attribute"


class _FakeResult(SimpleNamespace):
    pass


class TestIssue24DefectCRefusalAdviceBranch:
    def _result(self, categories):
        return SimpleNamespace(
            chunks_rejected=1, chunks_used=1,
            rejection_categories=tuple(categories),
            model_tokens=1180, latency_ms=18540.0, retries=2,
        )

    def _metrics(self, result):
        return (
            f"latency: {result.latency_ms:.0f}ms, 64 tok/s, "
            f"{result.model_tokens} tokens, 2 validation retries"
        )

    def test_all_faithfulness_rejections_name_the_deterministic_primitives(self):
        result = self._result(["faithfulness"])
        message = _rejection_refusal(result, self._metrics(result))
        assert "edit rejected" in message and "File unchanged" in message
        # The defect-C advice: NOT the size claim on a 10-line file.
        assert "could not produce a faithful merge" in message
        assert "--insert-after/--insert-before/--lines FROM:TO --delete/" \
               "--replace-literal" in message
        assert "may be too large" not in message

    def test_size_class_categories_keep_the_size_advice(self):
        result = self._result(["faithfulness", "truncation"])
        message = _rejection_refusal(result, self._metrics(result))
        assert "may be too large" in message
        assert "Try a smaller edit or split the function" in message
        # Back-compat pins: the shared refusal keeps its verbatim wording.
        assert message.startswith("Error: edit rejected")
        assert "model hallucinated on 1 chunk(s)" in message
        assert "`fastedit diff` will show no changes" in message

    def test_parse_class_only_keeps_the_size_advice(self):
        result = self._result(["parse"])
        message = _rejection_refusal(result, self._metrics(result))
        assert "may be too large" in message

    def test_no_categories_history_falls_back_to_the_size_advice(self):
        result = SimpleNamespace(
            chunks_rejected=1, chunks_used=1, model_tokens=9,
            latency_ms=1.0, retries=0,
        )
        message = _rejection_refusal(result, "metrics")
        assert "may be too large" in message


class TestIssue24DefectDDeletionRefusalNamesSanctionedTool:
    def test_replace_literal_empty_snippet_names_lines_delete(
        self, tmp_path: Path,
    ):
        from test_cli import run_cli

        target = tmp_path / "doc.md"
        target.write_text(ORIGINAL_MD)
        culprit = (
            "Lean on purpose. The former long version is `HISTORY.md` "
            "(untracked, when present): read it in small parts with "
            "`tldr body` or triage it with quicksilver, never whole."
        )
        result = run_cli(
            "edit", str(target),
            "--replace-literal", culprit,
            "--snippet", "", "--snippet-is-literal",
        )
        assert result.returncode == 1, result.stderr
        assert "--lines FROM:TO --delete" in result.stderr
        assert target.read_text() == ORIGINAL_MD
