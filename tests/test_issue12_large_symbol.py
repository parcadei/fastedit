"""Issue #12 — editing a very large symbol: OOM, opaque refusals, noise.

The reported failure (10,000-line TS file, symbol ``ChangeTitle`` spanning
~2,235 lines, tiny ``catch``-block edit inside it):

  1. the process died with exit 137 (SIGKILL) with NO message — the memory
     hog is the O(n·m) ``_lcs_pair_map`` DP table (n·m Python ints; with the
     model engine resident the multi-GB spike OOM-kills the process) and it
     is allocated on EVERY validation attempt;
  2. deterministic splice failures printed only "failed the
     content-faithfulness check" with no line and no rule named, so the
     snippet could not be corrected;
  3. "no anchor line matched the original body" fired when the anchor line
     EXISTS but is repeated across the symbol (ambiguous) — the message
     must say what would make the edit placeable;
  4. the transformers tokenizer warning (``fix_mistral_regex``) prints on
     every engine init, burying real diagnostics;
  5. ``fastedit diff`` after the refusal read as if the edit might have
     half-applied ("No backup recorded ... Run an edit command first").

Fixes pinned here (all fail-loud + bounded-memory):

  (a) ``_lcs_pair_map`` caps its DP at :data:`_LCS_MAX_CELLS` cells: under
      the cap the exact DP runs; over it a banded LCS (off-diagonal band
      :data:`_LCS_BAND`) runs — bounded memory, documented approximation;
      when even the band cannot fit, a loud ``ValueError`` names the size
      and the remedy. Banded results are validated against a reference DP
      and can only ever be STRICTER for the battery (a missed survivor
      pair reads as a deletion, which the deletion-justification rules
      must then justify) — never more permissive.
  (b) content-faithfulness failures name the FIRST failing line and the
      rule (``unmentioned original line dropped: '<content>'``); the
      keep-marker anchor refusal names the repeated anchor, its match
      count and the remedy.
  (c) the transformers verbosity is dropped to ERROR, scoped to the model
      load, and restored afterwards (tested with a stubbed loader that
      emits the real warning path).
  (d) cmd_edit's refusal paths say the file is unchanged and that
      ``fastedit diff`` will show no changes.

Hermetic: no model, no backend — merge functions are stubs.
"""

from __future__ import annotations

import logging
import tracemalloc
from types import SimpleNamespace

import pytest

from fastedit.inference.chunked_merge import (
    _LCS_BAND,
    _LCS_MAX_CELLS,
    _lcs_matched,
    _lcs_pair_map,
    _lcs_pair_map_banded,
    _merge_rejection_reason,
    chunked_merge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _large_ts_function(lines: int = 2235) -> str:
    """A synthetic TS function of exactly *lines* lines with unique bodies.

    Mirrors the reported shape: one huge function (``ChangeTitle``),
    thousands of structurally similar but textually unique lines, so every
    line can anchor unambiguously.
    """
    body = ["export function changeTitle(input: Input): Result {"]
    for k in range(1, lines - 2):
        body.append(f"  ops.push(buildStep({k}, 'step-{k}'));")
    body.append("  return finalize(ops);")
    body.append("}")
    return "\n".join(body) + "\n"


def _reference_lcs_pairs(a: list[str], b: list[str]) -> dict[int, int]:
    """The unbounded textbook DP — the oracle the bounded paths must match."""
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            dp[i][j] = (
                dp[i + 1][j + 1] + 1
                if a[i] == b[j]
                else max(dp[i + 1][j], dp[i][j + 1])
            )
    pairs: dict[int, int] = {}
    i = j = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            pairs[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _assert_valid_pairing(a: list[str], b: list[str], pairs: dict[int, int]):
    """pairs is a common subsequence: strictly increasing, content-equal."""
    last_i = last_j = -1
    for i in sorted(pairs):
        j = pairs[i]
        assert i > last_i and j > last_j
        assert a[i] == b[j]
        last_i, last_j = i, j


# ---------------------------------------------------------------------------
# (a) bounded LCS
# ---------------------------------------------------------------------------


class TestBoundedLcs:
    def test_reported_span_size_takes_the_bounded_path_and_completes(self):
        """2235x2235 exceeds the cell cap → banded DP, correct result, no OOM."""
        a = [f"line {i}" for i in range(2235)]
        b = [f"line {i}" for i in range(2235)]
        assert len(a) * len(b) > _LCS_MAX_CELLS  # the exact DP must NOT run

        tracemalloc.start()
        pairs = _lcs_pair_map(a, b)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(pairs) == 2235
        _assert_valid_pairing(a, b, pairs)
        # Bounded: nowhere near the multi-GB table the unbounded DP built.
        assert peak < 500 * 1024 * 1024

    def test_banded_matches_reference_on_in_band_drift(self):
        """Block insert/delete under the band: banded == exact LCS length."""
        a = [f"line {i}" for i in range(3000)]
        # 200-line block deleted and a 150-line block inserted: every
        # alignment stays well inside the 500-line band.
        b = a[:500] + a[700:]  # drop 200
        b = b[:1200] + [f"inserted {k}" for k in range(150)] + b[1200:]

        expected = _reference_lcs_pairs(a, b)
        assert len(expected) == 2800  # every original line of b survives

        pairs = _lcs_pair_map_banded(a, b, _LCS_BAND)
        assert len(pairs) == len(expected)
        _assert_valid_pairing(a, b, pairs)

    def test_lcs_matched_agrees_with_pair_map(self):
        a = [f"line {i}" for i in range(2500)]
        b = a[:1000] + a[1500:]
        kept_a, kept_b = _lcs_matched(a, b)
        pairs = _lcs_pair_map(a, b)
        assert kept_a == set(pairs)
        assert kept_b == set(pairs.values())

    def test_small_inputs_still_take_the_exact_dp(self):
        """Under the cap the exact DP runs — identical to the reference."""
        a = [f"line {i}" for i in range(300)]
        b = a[:100] + ["x"] + a[100:]
        assert len(a) * len(b) <= _LCS_MAX_CELLS
        pairs = _lcs_pair_map(a, b)
        assert pairs == _reference_lcs_pairs(a, b)

    def test_raises_loud_when_even_the_band_cannot_fit(self):
        """6000-line spans: neither DP arm fits → loud, actionable refusal."""
        a = [f"line {i}" for i in range(6000)]
        b = [f"line {i}" for i in range(6000)]
        with pytest.raises(ValueError, match=r"symbol too large for exact line "
                                             r"matching \(6000 lines\)"):
            _lcs_pair_map(a, b)

    def test_error_message_names_the_remedy(self):
        a = [f"line {i}" for i in range(6000)]
        with pytest.raises(ValueError, match="narrow the span"):
            _lcs_pair_map(a, a[:6000])


# ---------------------------------------------------------------------------
# (b) refusals name the failing line, the rule, and the remedy
# ---------------------------------------------------------------------------


class TestFaithfulnessRefusalNaming:
    ORIGINAL = "alpha one\nbeta two\ngamma three\ndelta four\n"
    # beta is UNMENTIONED by the snippet (the issue's shape: the merge drops
    # a line the snippet never spoke about).
    SNIPPET = "alpha one\ngamma three\ndelta four\n"
    DROPPED = "beta two"

    def test_rejection_reason_names_the_dropped_line_and_rule(self):
        merged = self.ORIGINAL.replace(self.DROPPED + "\n", "")
        reason = _merge_rejection_reason(
            self.ORIGINAL, merged, self.SNIPPET, None, False,
        )
        assert reason is not None
        assert "content-faithfulness" in reason
        assert "unmentioned original line dropped" in reason
        assert self.DROPPED in reason

    def test_clean_merge_has_no_reason(self):
        assert (
            _merge_rejection_reason(
                self.ORIGINAL, self.ORIGINAL, self.ORIGINAL, None, False,
            )
            is None
        )

    def test_invention_is_named(self):
        merged = self.ORIGINAL.replace(
            "gamma three\n", "gamma three\ninvented line\n",
        )
        reason = _merge_rejection_reason(
            self.ORIGINAL, merged, self.ORIGINAL, None, False,
        )
        assert reason is not None
        assert "invented line" in reason


# ---------------------------------------------------------------------------
# (b3) the keep-marker anchor refusal says what would make it work
# ---------------------------------------------------------------------------


class TestAnchorRefusalRemedy:
    def _refusal(self, original_func: str, snippet: str) -> ValueError:
        from fastedit.inference.text_match import (
            _marker_snippet_placement_refusal,
        )

        return ValueError(_marker_snippet_placement_refusal(original_func, snippet))

    def test_repeated_anchor_names_count_and_remedy(self):
        """The reported case: '} catch {' exists but repeats 3 times."""
        original = (
            "function f() {\n"
            "  try {\n"
            "    a();\n"
            "  } catch {\n"
            "    b();\n"
            "  }\n"
            "  try {\n"
            "    c();\n"
            "  } catch {\n"
            "    d();\n"
            "  }\n"
            "  try {\n"
            "    e();\n"
            "  } catch {\n"
            "    g();\n"
            "  }\n"
            "}\n"
        )
        snippet = (
            "// ... existing code ...\n"
            "} catch {\n"
            "  ops.push(1);\n"
            "}\n"
            "// ... existing code ...\n"
        )
        message = str(self._refusal(original, snippet))
        assert "} catch {" in message
        assert "3 original lines" in message
        assert "include more surrounding unique lines" in message

    def test_zero_matches_keeps_the_classic_message(self):
        original = "function f() {\n  return 1;\n}\n"
        snippet = (
            "// ... existing code ...\n"
            "totally absent line\n"
            "// ... existing code ...\n"
        )
        message = str(self._refusal(original, snippet))
        assert "no anchor line matched the original body" in message

    def test_single_unique_anchor_names_the_two_anchor_floor(self):
        original = "function f() {\n  unique one\n  unique two\n}\n"
        snippet = (
            "// ... existing code ...\n"
            "unique one\n"
            "ops.push(1);\n"
            "// ... existing code ...\n"
        )
        message = str(self._refusal(original, snippet))
        assert "at least two" in message

    def test_two_unique_anchors_declined_names_the_other_cause(self):
        """Enough anchors, but a rewrite conflict declined the editor."""
        original = "function f() {\n  unique one\n  unique two\n}\n"
        snippet = (
            "unique one\n"
            "unique two = 9;\n"
            "unique two\n"
        )
        message = str(self._refusal(original, snippet))
        assert "deterministic editor" in message
        assert "full replacement body" in message

    def test_cli_refusal_prints_the_remedy(self, tmp_path, capsys):
        """End-to-end: cmd_edit exits 1 with the remedy on stderr."""
        import argparse

        from fastedit import cli

        target = tmp_path / "mod.py"
        target.write_text(
            "def handler():\n"
            "    try:\n"
            "        run(1)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    try:\n"
            "        run(2)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    try:\n"
            "        run(3)\n"
            "    except ValueError:\n"
            "        pass\n",
        )
        args = argparse.Namespace(
            file=str(target),
            snippet=(
                "# ... existing code ...\n"
                "    except ValueError:\n"
                "        ops.append(9)\n"
                "# ... existing code ...\n"
            ),
            replace="handler",
            after="",
            backend=None, model_path=None, api_base=None, api_model=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_edit(args)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "except ValueError" in err
        assert "include more surrounding unique lines" in err


# ---------------------------------------------------------------------------
# The reported end-to-end shape: 2,235-line symbol, tiny inner edit
# ---------------------------------------------------------------------------


class TestLargeSymbolEndToEnd:
    def test_deterministic_edit_of_reported_shape_completes_bounded(self):
        """The reported flow, hermetic: marker snippet, two anchors, catch edit.

        Asserts completion, the correct 0-token deterministic merge, and a
        bounded allocation peak (the unbounded DP the issue reported would
        build a multi-GB table here; the bounded one stays two orders of
        magnitude smaller).
        """
        original = _large_ts_function(2235)
        anchor_1 = "  ops.push(buildStep(100, 'step-100'));"
        anchor_2 = "  ops.push(buildStep(900, 'step-900'));"
        snippet = (
            "// ... existing code ...\n"
            f"{anchor_1}\n"
            "  try {\n"
            "    tracker.mark(101);\n"
            "  } catch (err) {\n"
            "    failures.push('step-100');\n"
            "  }\n"
            f"{anchor_2}\n"
            "// ... existing code ...\n"
        )

        tracemalloc.start()
        result = chunked_merge(
            original_code="const x = 1;\n" + original,
            snippet=snippet,
            file_path="/repo/service.ts",
            merge_fn=lambda *_a, **_kw: pytest.fail(
                "the deterministic path must win; no model may be invoked",
            ),
            language="typescript",
            replace="changeTitle",
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result.model_tokens == 0  # deterministic path, no model
        assert result.chunks_used == 0
        assert result.parse_valid
        assert "tracker.mark(101);" in result.merged_code
        assert "failures.push('step-100');" in result.merged_code
        # Everything outside the insertion survives.
        for k in (1, 100, 900, 2232):
            assert f"ops.push(buildStep({k}, 'step-{k}'));" in result.merged_code
        assert "  return finalize(ops);" in result.merged_code
        assert peak < 500 * 1024 * 1024

    def test_faithfulness_check_on_reported_span_completes(self):
        """The gate itself, at the reported size: completes and stays sound."""
        original = _large_ts_function(2235)
        merged = original.replace(
            "  ops.push(buildStep(1500, 'step-1500'));\n",
            "  ops.push(buildStep(1500, 'step-1500-rewritten'));\n",
        )
        snippet = (
            "// ... existing code ...\n"
            "  ops.push(buildStep(10, 'step-10'));\n"
            "  ops.push(buildStep(1500, 'step-1500-rewritten'));\n"
            "  ops.push(buildStep(2000, 'step-2000'));\n"
            "// ... existing code ...\n"
        )
        # The editor output here inserts the rewritten line while the
        # original survives: a faithful INSERTION of a new line.
        reason = _merge_rejection_reason(original, merged, snippet, None, False)
        assert reason is not None  # rewritten line is an undeclared mutation
        assert "content-faithfulness" in reason


# ---------------------------------------------------------------------------
# (d) the refusal makes clear the diff is empty
# ---------------------------------------------------------------------------


class TestRefusalDiffNote:
    def test_all_chunks_rejected_refusal_mentions_diff(self, tmp_path, monkeypatch,
                                                       capsys):
        import argparse

        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit import cli

        target = tmp_path / "mod.py"
        target.write_text("def existing():\n    return 1\n")
        monkeypatch.setattr(
            chunked_merge_module,
            "chunked_merge",
            lambda *a, **kw: SimpleNamespace(
                merged_code="def existing():\n    return 1\n",
                parse_valid=False,
                chunks_used=1,
                chunk_regions=[(1, 2)],
                model_tokens=12,
                latency_ms=40.0,
                chunks_rejected=1,
                retries=8,
            ),
        )
        args = argparse.Namespace(
            file=str(target), snippet="x", replace="", after="existing",
            backend=None, model_path=None, api_base=None, api_model=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_edit(args)
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "edit rejected" in err
        assert "fastedit diff" in err
        assert "no changes" in err


# ---------------------------------------------------------------------------
# (c) the transformers tokenizer warning is scoped out of engine init
# ---------------------------------------------------------------------------


class TestTokenizerWarningSuppression:
    def test_init_suppresses_transformers_warnings_and_restores_verbosity(
        self, tmp_path, monkeypatch,
    ):
        import transformers

        hf_logging = transformers.logging

        from fastedit.inference import mlx_engine

        emitted: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                emitted.append(record.getMessage())

        handler = _Capture()
        root = logging.getLogger()
        root.addHandler(handler)
        prev_verbosity = hf_logging.get_verbosity()
        try:

            def fake_load(_model_path, **_kw):
                # The exact channel the real warning uses: transformers'
                # own logger, at WARNING — the level the fix_mistral_regex
                # notice prints at.
                hf_logging.get_logger(
                    "transformers.tokenization_utils_base",
                ).warning(
                    "The tokenizer class you load from this checkpoint is not "
                    "the current default (fix_mistral_regex=True).",
                )
                return "model", "tokenizer"

            monkeypatch.setattr(mlx_engine, "load", fake_load)
            engine = mlx_engine.MLXEngine(
                model_path="unused", cache_dir=str(tmp_path),
            )
        finally:
            root.removeHandler(handler)
            hf_logging.set_verbosity(prev_verbosity)

        assert engine.model == "model"
        assert engine.tokenizer == "tokenizer"
        # The tokenizer warning never surfaced...
        assert emitted == []
        # ...and the suppression was scoped: verbosity restored after init.
        assert hf_logging.get_verbosity() == prev_verbosity
