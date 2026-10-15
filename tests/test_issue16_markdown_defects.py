"""Issue #16 — four markdown editing defects.

Defect 1 — silent no-op: ``--replace '<heading>'`` with a snippet that
restates the section IDENTICALLY landed through the deterministic path,
produced ``merged_code == original`` and the CLI printed "Applied edit to
..." with exit 0 while the file was byte-identical (``fastedit diff`` then
said "No changes detected"). A true no-op must be REPORTED as one: honest
message, exit 0, no write — never a success line for work that did not
happen.

Defect 2 — renames in prose are unlandable: the faithfulness battery's
exact-line matching reads any rename as an "unmentioned original line
dropped" even though the snippet restates the line WITH the token renamed
(the issue shape: ``- **Action:** ... ${appUrl}/contact ...`` renamed to
``dashboardUrl``). The fix is a declarative, bounded RENAME-AWARE pairing
pass in the deletion-justification logic: a dropped identity-free original
with no exact survivor is justified when EXACTLY ONE otherwise-unmatched
declared new line is a near-restatement of it (difflib SequenceMatcher
quick_ratio >= 0.80 AND ratio >= 0.85 on the stripped lines) — and the
pairing FAILS CLOSED when several candidates compete (never guess which
one replaced the line). Caps: minimum line length 12, at most 25% of a
segment's lines may pair, marker-free-or-same-segment only. Both sides of
the op arithmetic (the content validator and the D1 trait oracle) credit
the same pairings so they can never disagree.

Defect 3 — ``--after <heading-symbol>`` inserted at the END of the
section span (the symbol's line_end), duplicating a heading when the
snippet is itself a heading. For a format symbol whose spec row declares
``definition_line_required`` (markdown sections, TOML tables, JSON/YAML
keys, ...), when the snippet's first line is itself a definition line of
the same kind and names the SAME symbol, the insertion point is the
symbol's DEFINITION LINE — "after the heading" — not the span end.
Non-heading snippets and non-format languages keep the existing
span-end semantics.

Defect 4 — nondeterminism between identical-shaped edits: the 5x-refusal
shape was defect 2 (the failing file's section carried the rename, the
succeeding one did not); after the rename-pairing fix the pipeline's
accept/reject verdict for a given SHAPE is a function of the input, so
two structurally identical fixtures must get the same outcome. Pinned by
running the same-shaped edit against two identical-shape fixtures and
asserting both land — and against the still-invalid rewrite shape and
asserting both refuse.
"""

from __future__ import annotations

import argparse

import pytest

from fastedit.inference.chunked_merge import (
    _merge_unfaithful_reason,
    chunked_merge,
)
from fastedit.inference.merge import MergeResult

# ---------------------------------------------------------------------------
# Fixtures — the issue's reported .scen.md shape (heading DSL + prose)
# ---------------------------------------------------------------------------

SCEN_MD = """# Test scenario: contact form

## Scenario: Example form flow

Some preamble prose for the scenario section.

- **Given:** the user opens the app
- **Action:** navigate to ${appUrl}/contact and submit the form
- **Then:** the confirmation page renders

#### S001: Open the contact page

Step details for S001.
"""

SECTION_BODY = """## Scenario: Example form flow

Some preamble prose for the scenario section.

- **Given:** the user opens the app
- **Action:** navigate to ${appUrl}/contact and submit the form
- **Then:** the confirmation page renders
"""

RENAME_SNIPPET = SECTION_BODY.replace("appUrl", "dashboardUrl")
NOOP_SNIPPET = SECTION_BODY
S001_SNIPPET = "#### S001: Open the contact page\n"

ACTION_ORIGINAL = "- **Action:** navigate to ${appUrl}/contact and submit the form"
ACTION_RENAMED = "- **Action:** navigate to ${dashboardUrl}/contact and submit the form"
# A SECOND qualifying near-restatement of the same original (case-variant
# token): both candidates clear the similarity thresholds, so which one
# "replaced" the line is a guess — the pairing must fail closed.
ACTION_RENAMED_ALT = "- **Action:** navigate to ${DashboardUrl}/contact and submit the form"

# A full rewrite masquerading as a rename: same section, but the action
# line is replaced by unrelated prose (low similarity).
REWRITE_SNIPPET = SECTION_BODY.replace(
    ACTION_ORIGINAL,
    "Complete the onboarding survey before proceeding",
)


def _section_merge(chunk: str, token_old: str, token_new: str) -> str:
    """The merge a model would honestly produce for a rename snippet:
    the chunk with the token renamed on the action line, everything else
    preserved verbatim."""
    return chunk.replace(token_old, token_new, 1)


def _stub_args(**overrides) -> argparse.Namespace:
    """The argparse namespace cmd_edit reads (CLI surface tests)."""
    base = {
        "file": "", "snippet": "", "replace": "", "after": "",
        "after_imports": False, "snippet_is_literal": False,
        "insert_after": None, "insert_before": None,
        "lines": None, "delete": False,
        "backend": None, "model_path": None, "api_base": None, "api_model": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _install_stub_backend(monkeypatch, responder):
    """Replace the CLI backend factory with a scripted merge responder.

    ``responder(chunk, snippet, language) -> str`` returns the merge the
    "model" produces. The stub engine raises if anything tries to load a
    real model.
    """
    import fastedit.cli as cli_module
    from fastedit.inference.merge import MergeResult as _MR

    class _StubEngine:
        def merge_auto(self, chunk, snippet, language):
            return _MR(
                merged_code=responder(chunk, snippet, language),
                parse_valid=True,
                tokens_generated=5,
                latency_ms=10.0,
                tokens_per_second=500.0,
            )

    monkeypatch.setattr(
        cli_module, "_make_backend_with_overrides",
        lambda args: ("stub", _StubEngine()),
    )


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")


@pytest.fixture()
def backup_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", str(tmp_path / "backups"))
    return tmp_path / "backups"


# ---------------------------------------------------------------------------
# Defect 2 — the rename-aware pairing in the faithfulness battery
# ---------------------------------------------------------------------------


class TestRenamePairingBattery:
    """The battery-level rename pairing on the exact issue shape."""

    def test_true_rename_is_justified(self):
        """The issue shape: the action line restated with the token
        renamed, every other line an exact context anchor — the dropped
        original pairs with the renamed line and the merge is faithful."""
        merged = RENAME_SNIPPET
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, RENAME_SNIPPET),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is None, reason

    def test_rename_with_unrelated_new_line_still_pairs(self):
        """A snippet that renames the action line AND adds an unrelated
        new line: the dropped original still has exactly one high-similarity
        candidate — the pairing holds."""
        snippet = RENAME_SNIPPET.replace(
            "- **Then:** the confirmation page renders",
            "- **Then:** the confirmation page renders\n"
            "- **Notes:** created for the rename regression suite",
        )
        merged = snippet
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, snippet),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is None, reason

    def test_full_rewrite_masquerading_as_rename_rejected(self):
        """A low-similarity replacement is NOT a rename: the dropped
        original has no near-restatement candidate and the merge fails
        closed."""
        merged = REWRITE_SNIPPET
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, REWRITE_SNIPPET),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is not None
        assert "unmentioned original line dropped" in reason

    def test_many_to_one_pairing_rejected(self):
        """Two competing renamed variants for ONE dropped original: which
        one replaced the line is a guess — the pairing fails closed."""
        snippet = SECTION_BODY.replace(
            ACTION_ORIGINAL,
            f"{ACTION_RENAMED}\n{ACTION_RENAMED_ALT}",
        )
        merged = snippet
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, snippet),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is not None

    def test_rename_cap_per_segment(self):
        """At most 25% of a segment's lines may pair as renames: a snippet
        that 'renames' 5 of 8 content lines (62.5% > 25% of the segment)
        is a rewrite, not a rename — fails closed."""
        original = (
            "## Section\n"
            "\n"
            "alpha line one\n"
            "alpha line two\n"
            "alpha line three\n"
            "alpha line four\n"
            "alpha line five\n"
            "omega line\n"
        )
        snippet = original
        for word in ("one", "two", "three", "four", "five"):
            snippet = snippet.replace(f"alpha line {word}", f"alpha line {word} renamed")
        merged = snippet
        reason = _merge_unfaithful_reason(
            _content_lines(original),
            _content_lines(merged),
            _classified_tokens(original, snippet),
            _raw_lines(original),
            _raw_lines(merged),
        )
        assert reason is not None

    def test_rename_below_minimum_length_rejected(self):
        """Very short lines have unstable similarity — a 2-token line swap
        is below the minimum-length guard and stays an unjustified drop."""
        original = "## Section\n\nopen x\n- keep this bullet\n- and this bullet\n"
        snippet = original.replace("open x", "open y")
        reason = _merge_unfaithful_reason(
            _content_lines(original),
            _content_lines(snippet),
            _classified_tokens(original, snippet),
            _raw_lines(original),
            _raw_lines(snippet),
        )
        assert reason is not None

    def test_rename_pairing_cannot_mask_a_dropped_survivor(self):
        """A rename pairing must not launder a REAL deletion: the snippet
        renames the action line AND drops the Given line — the Given drop
        still fails closed."""
        snippet = RENAME_SNIPPET.replace(
            "- **Given:** the user opens the app\n",
            "",
        )
        merged = snippet
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, snippet),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is not None

    def test_rename_with_survivor_reindent_still_rejected(self):
        """The rename pairing replaces the deleted-line justification; the
        surviving-line indent-delta rule still applies independently."""
        merged = RENAME_SNIPPET.replace(
            "- **Given:** the user opens the app",
            "    - **Given:** the user opens the app",
        )
        reason = _merge_unfaithful_reason(
            _content_lines(SECTION_BODY),
            _content_lines(merged),
            _classified_tokens(SECTION_BODY, RENAME_SNIPPET),
            _raw_lines(SECTION_BODY),
            _raw_lines(merged),
        )
        assert reason is not None


def _content_lines(text: str) -> list[str]:
    from fastedit.inference.chunked_merge import _real_lines
    return _real_lines(text)


def _raw_lines(text: str) -> list[str]:
    from fastedit.inference.chunked_merge import _raw_content_lines
    return _raw_content_lines(text)[1]


def _classified_tokens(original: str, snippet: str):
    from fastedit.inference.chunked_merge import _classify_snippet
    return _classify_snippet(snippet, _content_lines(original))


# ---------------------------------------------------------------------------
# Defect 2 — op-side consistency: the D1/D3 oracles credit the same pairings
# ---------------------------------------------------------------------------


class TestRenamePairingOpSide:
    """The trait oracles derive removal capacity from the snippet alone;
    they must credit exactly the removals the pairing justifies."""

    def test_op_side_capacity_covers_a_justified_rename(self):
        from fastedit.inference.chunked_merge import _derived_text_op
        _op, _slack, removable = _derived_text_op(SECTION_BODY, RENAME_SNIPPET)
        # The action line is 8 words / 64 bytes (measured on this fixture):
        # its exact trait sizes must be covered by the rename capacity.
        assert removable.get("lines", 0) >= 1
        assert removable.get("words", 0) >= 8
        assert removable.get("bytes", 0) >= 64

    def test_op_side_credits_nothing_for_a_masqueraded_rewrite(self):
        """The pairing capacity (the rename-specific part of the floor) is
        zero for a masqueraded rewrite — the shape may still carry the
        issue-#6 complete-restatement credit, but that is the pre-existing
        rule, not the rename pairing."""
        from fastedit.inference.chunked_merge import (
            _classify_snippet_raw,
            _find_rename_pairs,
            _raw_content_lines,
        )
        orig, _raw = _raw_content_lines(SECTION_BODY)
        for label, snippet in (
            ("rewrite", REWRITE_SNIPPET),
            ("rename", RENAME_SNIPPET),
        ):
            raw_tokens = _classify_snippet_raw(snippet, orig)
            seg_new = [
                t[2] for t in raw_tokens
                if t[0] == "new" and t[2] is not None
            ]
            deleted = [i for i in range(len(orig))]
            pairs = _find_rename_pairs(orig, deleted, seg_new)
            if label == "rewrite":
                assert pairs == {}, pairs
            else:
                # The rename shape pairs exactly the action line.
                action_idx = orig.index(ACTION_ORIGINAL.strip())
                assert set(pairs) == {action_idx}, pairs

    def test_attribute_side_covers_a_justified_rename(self):
        """The D3 declared-changes derivation routes through the SAME
        ``_snippet_justifiable_removals`` helper; on this prose shape the
        markdown trait extractors see no traits — pin that the helper (not
        the snippet text alone) decides, so the two oracles can never
        disagree about which removal a rename justifies."""
        from fastedit.inference.chunked_merge import (
            _classify_snippet_raw,
            _raw_content_lines,
            _snippet_justifiable_removals,
        )
        from fastedit.lang_attributes import extract_attributes
        orig, orig_raw = _raw_content_lines(SECTION_BODY)
        raw_tokens = _classify_snippet_raw(RENAME_SNIPPET, orig)
        removable = _snippet_justifiable_removals(orig, orig_raw, raw_tokens)
        assert ACTION_ORIGINAL in removable
        # The action line carries no markdown trait at all (prose), so the
        # D3 gate is inert for it — pinned explicitly.
        assert extract_attributes(ACTION_ORIGINAL + "\n", "markdown", fragment=True) == []


# ---------------------------------------------------------------------------
# Defect 2 — end to end: the model path and the deterministic path land renames
# ---------------------------------------------------------------------------


class TestRenameLandsEndToEnd:
    def test_model_path_lands_a_rename_in_a_markdown_section(self):
        """The 5x-refused shape: a marker-free section-replace snippet that
        renames one token. A model that renames exactly that line and
        preserves everything else converges on attempt 1."""

        def scripted_merge(chunk, snippet, language):
            return MergeResult(
                merged_code=_section_merge(chunk, "appUrl", "dashboardUrl"),
                parse_valid=True,
                tokens_generated=5,
                latency_ms=10.0,
                tokens_per_second=500.0,
            )

        calls = {"n": 0}

        def counting_merge(chunk, snippet, language):
            calls["n"] += 1
            return scripted_merge(chunk, snippet, language)

        result = chunked_merge(
            original_code=SCEN_MD,
            snippet=RENAME_SNIPPET,
            file_path="case.md",
            merge_fn=counting_merge,
            language="markdown",
            replace="Scenario: Example form flow",
        )

        assert result.chunks_rejected == 0, (
            f"the rename merge must land; retries={result.retries}"
        )
        assert calls["n"] == 1
        lines = result.merged_code.splitlines()
        action_lines = [ln for ln in lines if ln.startswith("- **Action:**")]
        assert action_lines == [ACTION_RENAMED], action_lines
        assert "${appUrl}/contact" not in result.merged_code.splitlines()[
            next(
                i for i, ln in enumerate(lines, 1)
                if ln.startswith("- **Action:**")
            ) - 1
        ]
        # Every other line preserved byte-exact.
        assert "# Test scenario: contact form" in result.merged_code
        assert "Some preamble prose" in result.merged_code
        assert "- **Given:** the user opens the app" in result.merged_code
        assert "- **Then:** the confirmation page renders" in result.merged_code

    def test_rename_lands_through_cmd_edit(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """The renamed line lands through cmd_edit's whole flow: the
        deterministic editor declines the shape (leading-token rewrite
        conflict), the direct swap declines (the snippet does not restate
        the nested subsection), and the validated model path — whose
        battery consumed the pairing — writes the rename."""

        target = tmp_path / "case.md"
        target.write_text(SCEN_MD)

        _install_stub_backend(
            monkeypatch,
            lambda chunk, snippet, language: _section_merge(
                chunk, "appUrl", "dashboardUrl",
            ),
        )
        from fastedit.cli import cmd_edit

        cmd_edit(_stub_args(
            file=str(target), snippet=RENAME_SNIPPET,
            replace="Scenario: Example form flow",
        ))

        content = target.read_text()
        assert ACTION_RENAMED in content
        # The old token is gone from the action line ONLY — every other
        # line is byte-preserved.
        lines = content.splitlines()
        action_lines = [ln for ln in lines if ln.startswith("- **Action:**")]
        assert action_lines == [ACTION_RENAMED]
        assert "Some preamble prose for the scenario section." in content
        assert "- **Given:** the user opens the app" in content
        out = capsys.readouterr().out
        assert "Applied edit" in out

    def test_pipeline_refuses_a_many_to_one_rename_merge(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """The many-to-one abuse through the whole pipeline: a scripted
        model that emits TWO renamed variants of one original line is
        refused, the file is untouched, and the CLI exits 1 (fail closed,
        never a silent ratification)."""

        def abusive_merge(chunk, snippet, language):
            return chunk.replace(
                ACTION_ORIGINAL,
                f"{ACTION_RENAMED}\n{ACTION_RENAMED_ALT}",
            )

        _install_stub_backend(monkeypatch, abusive_merge)
        from fastedit.cli import cmd_edit

        target = tmp_path / "case.md"
        target.write_text(SCEN_MD)

        with pytest.raises(SystemExit) as excinfo:
            cmd_edit(_stub_args(
                file=str(target), snippet=RENAME_SNIPPET,
                replace="Scenario: Example form flow",
            ))

        assert excinfo.value.code == 1
        assert target.read_text() == SCEN_MD
        err = capsys.readouterr().err
        assert "edit rejected" in err
        assert "Applied edit" not in capsys.readouterr().out


class _ForbiddenBackend:
    def __init__(self, fn):
        self._fn = fn

    def merge_auto(self, *a, **kw):
        self._fn(*a, **kw)


# ---------------------------------------------------------------------------
# Defect 1 — the silent no-op is reported honestly
# ---------------------------------------------------------------------------


class TestNoOpHonesty:
    def test_model_path_noop_section_replace_reports_no_change(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """A model-path edit whose merged output is byte-identical to the
        original must NOT print "Applied edit": it reports the no-op
        truthfully, exits 0, and writes nothing."""
        _install_stub_backend(monkeypatch, lambda chunk, snippet, language: chunk)
        from fastedit.cli import cmd_edit

        target = tmp_path / "case.md"
        target.write_text(SCEN_MD)

        cmd_edit(_stub_args(
            file=str(target), snippet=NOOP_SNIPPET,
            replace="Scenario: Example form flow",
        ))

        captured = capsys.readouterr()
        assert target.read_text() == SCEN_MD  # byte-identical, no write
        assert "No change resulted from this edit" in captured.out, captured.out
        assert "the snippet matches the current content" in captured.out
        assert "Applied edit" not in captured.out, captured.out

    def test_deterministic_noop_section_replace_reports_no_change(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """The defect-1 reported shape end to end: --replace with a snippet
        restating the section identically. The deterministic text-match
        editor produces merged == original; the CLI must report the no-op
        honestly instead of printing a success line for an untouched file."""

        def _forbidden(*a, **kw):  # pragma: no cover - assertion guard
            raise AssertionError("a no-op must never reach the model")

        import fastedit.cli as cli_module
        monkeypatch.setattr(
            cli_module, "_make_backend_with_overrides",
            lambda args: ("stub", _ForbiddenBackend(_forbidden)),
        )
        from fastedit.cli import cmd_edit

        target = tmp_path / "case.md"
        target.write_text(SCEN_MD)

        cmd_edit(_stub_args(
            file=str(target), snippet=NOOP_SNIPPET,
            replace="Scenario: Example form flow",
        ))

        captured = capsys.readouterr()
        assert target.read_text() == SCEN_MD
        assert "No change resulted from this edit" in captured.out, captured.out
        assert "Applied edit" not in captured.out, captured.out

    def test_real_change_still_reports_applied(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """Guard: the honest no-op report must not swallow real edits."""
        _install_stub_backend(
            monkeypatch,
            lambda chunk, snippet, language: _section_merge(
                chunk, "appUrl", "dashboardUrl",
            ),
        )
        from fastedit.cli import cmd_edit

        target = tmp_path / "case.md"
        target.write_text(SCEN_MD)

        cmd_edit(_stub_args(
            file=str(target), snippet=RENAME_SNIPPET,
            replace="Scenario: Example form flow",
        ))

        captured = capsys.readouterr()
        assert "Applied edit" in captured.out, captured.out
        assert ACTION_RENAMED in target.read_text()


# ---------------------------------------------------------------------------
# Defect 3 — --after a heading symbol inserts after the heading LINE
# ---------------------------------------------------------------------------


class TestAfterHeadingInsertionPoint:
    def test_after_heading_symbol_inserts_after_the_heading_line(self):
        """The reported duplication: inserting a '#### S001' snippet after
        the '## Scenario' symbol must land immediately after the heading
        line — no duplicate heading, section content intact."""
        result = chunked_merge(
            original_code=SCEN_MD,
            snippet=S001_SNIPPET,
            file_path="case.md",
            merge_fn=_never_called_merge,
            language="markdown",
            after="Scenario: Example form flow",
        )

        assert result.chunks_used == 0  # fast path, no model
        lines = result.merged_code.splitlines()
        heading_idx = lines.index("## Scenario: Example form flow")
        # The snippet lands on the line immediately after the heading.
        assert lines[heading_idx + 1] == "#### S001: Open the contact page", (
            "\n".join(f"L{i+1}: {ln}" for i, ln in enumerate(lines))
        )
        # No duplicated heading anywhere in the file.
        s001 = [ln for ln in lines if "S001: Open the contact page" in ln]
        assert len(s001) == 1, s001
        # The section's own content is intact.
        assert "#### S001: Open the contact page" in lines
        body = "\n".join(lines)
        assert "- **Action:** navigate to ${appUrl}/contact and submit the form" in body
        assert "Step details for S001." in body

    def test_after_heading_non_heading_snippet_keeps_span_end_semantics(self):
        """A NON-heading snippet keeps the existing 'after the section'
        semantics — only a snippet whose first line is itself a definition
        line of the same kind takes the heading-line insertion point."""
        snippet = "- **And:** the second contact step lands\n"
        result = chunked_merge(
            original_code=SCEN_MD,
            snippet=snippet,
            file_path="case.md",
            merge_fn=_never_called_merge,
            language="markdown",
            after="Scenario: Example form flow",
        )
        lines = result.merged_code.splitlines()
        heading_idx = lines.index("## Scenario: Example form flow")
        # NOT immediately after the heading: the line after the heading is
        # the section's own preamble.
        assert lines[heading_idx + 1] != snippet.strip()
        # ...and the snippet landed at the section's end (span-end
        # semantics, unchanged).
        assert snippet.strip() in lines
        assert lines.index(snippet.strip()) > heading_idx
        s001 = [ln for ln in lines if "S001: Open the contact page" in ln]
        assert len(s001) == 1

    def test_after_symbol_non_format_language_unchanged(self):
        """Code languages keep the span-end insertion point (no
        definition-line special case without a declarative spec row)."""
        py_original = (
            "def anchor():\n"
            "    a = 1\n"
            "\n"
            "def other():\n"
            "    b = 2\n"
        )
        result = chunked_merge(
            original_code=py_original,
            snippet="def inserted():\n    c = 3\n",
            file_path="mod.py",
            merge_fn=_never_called_merge,
            language="python",
            after="anchor",
        )
        lines = result.merged_code.splitlines()
        # The insertion lands after the WHOLE anchor function (a = 1), not
        # after its definition line.
        assert lines.index("def inserted():") == 3, "\n".join(lines)


def _never_called_merge(*a, **kw):  # pragma: no cover - assertion guard
    raise AssertionError("the after= fast path must not call the model")


# ---------------------------------------------------------------------------
# Defect 4 — the same input SHAPE gets the same verdict
# ---------------------------------------------------------------------------


class TestSameShapeDeterminism:
    def test_identical_shape_edits_deterministically_land(self):
        """Two structurally identical fixtures, the same rename edit, the
        same scripted model: both land. The pipeline's accept/reject for a
        given shape is a function of the input, not of the attempt count."""
        results = []
        for _ in range(2):
            results.append(chunked_merge(
                original_code=SCEN_MD,
                snippet=RENAME_SNIPPET,
                file_path="case.md",
                merge_fn=_scripted_rename_merge,
                language="markdown",
                replace="Scenario: Example form flow",
            ))
        assert all(r.chunks_rejected == 0 for r in results), [
            r.chunks_rejected for r in results
        ]
        assert all(r.merged_code == results[0].merged_code for r in results)
        assert all(r.retries == 0 for r in results)
        assert all(
            ACTION_RENAMED in r.merged_code and r.merged_code != SCEN_MD
            for r in results
        )

    def test_identical_shape_edits_via_real_pipeline_both_land(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """The issue's SCEN-001/SCEN-002 pair: two identically-shaped
        files, the same --replace rename on both — both land (before the
        fix, one landed and the other refused 5x)."""
        _install_stub_backend(
            monkeypatch,
            lambda chunk, snippet, language: _section_merge(
                chunk, "appUrl", "dashboardUrl",
            ),
        )
        from fastedit.cli import cmd_edit

        outcomes = []
        for name in ("scen_one.md", "scen_two.md"):
            target = tmp_path / name
            target.write_text(SCEN_MD)
            cmd_edit(_stub_args(
                file=str(target), snippet=RENAME_SNIPPET,
                replace="Scenario: Example form flow",
            ))
            outcomes.append(target.read_text())
            capsys.readouterr()

        assert all(ACTION_RENAMED in content for content in outcomes)
        assert outcomes[0] == outcomes[1]

    def test_rejection_is_input_determined_for_the_same_shape(
        self, tmp_path, monkeypatch, capsys, backup_dir,
    ):
        """The same-shape contract cuts both ways: the still-invalid
        rewrite shape refuses IDENTICALLY on structurally identical
        fixtures — the verdict is determined by the input class."""
        _install_stub_backend(
            monkeypatch,
            lambda chunk, snippet, language: chunk.replace(
                ACTION_ORIGINAL,
                "Complete the onboarding survey before proceeding",
            ),
        )
        from fastedit.cli import cmd_edit

        verdicts = []
        for name in ("scen_one.md", "scen_two.md"):
            target = tmp_path / name
            target.write_text(SCEN_MD)
            with pytest.raises(SystemExit) as excinfo:
                cmd_edit(_stub_args(
                    file=str(target), snippet=REWRITE_SNIPPET,
                    replace="Scenario: Example form flow",
                ))
            verdicts.append((excinfo.value.code, target.read_text()))
            capsys.readouterr()

        assert all(code == 1 for code, _content in verdicts)
        assert all(content == SCEN_MD for _code, content in verdicts)


def _scripted_rename_merge(chunk, snippet, language):
    return MergeResult(
        merged_code=_section_merge(chunk, "appUrl", "dashboardUrl"),
        parse_valid=True,
        tokens_generated=5,
        latency_ms=10.0,
        tokens_per_second=500.0,
    )
