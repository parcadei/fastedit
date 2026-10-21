"""Issue #13 — memory explosion on a large file with no edit target.

The reported failure ("fastedit grows to 80–97 GB on a large file with no
target"): ``fastedit edit <big-file> --snippet -`` with no ``--replace``/
``--after`` and no snippet anchor that can match. Two unbounded hand-offs
to the merge model survived the issue #12 LCS cap (which bounds the
VALIDATION DP, not the model prompts):

  1. **whole-file path** — the >150-line gate is a LINE gate. A 100 MB
     single-line file (minified JSON/log) has ``total_lines == 1`` and
     sails through it; the whole-file path then hands the ENTIRE file to
     ``merge_fn`` on every one of the 8 retry attempts. With the mlx
     engine the prompt is tokenized (~25M tokens → a multi-GB Python int
     list) and prefilled (prompt KV cache + activation arrays all scale
     with prompt tokens) — the measured 80–97 GB climb, ending in exit 137.
  2. **per-chunk path** — a ``replace=`` chunk (a whole symbol, or a window
     over megabyte-long lines) is only size-checked at the LCS DP gate
     AFTER the model has already been called with it.

Fix pinned here: every text handed to ``merge_fn`` is bounded by
:data:`fastedit.inference.chunked_merge._MAX_MERGE_PROMPT_CHARS` BEFORE the
model is invoked (whole-file path and every per-chunk prompt), and the
snippet itself is bounded by
:data:`fastedit.inference.chunked_merge._MAX_SNIPPET_CHARS` before the
marker normalizer's per-character string-span masker runs. Each overflow
fails LOUD with a ``ValueError`` naming the size, the budget and the
remedy — never a silent multi-GB climb.

Scaling/RSS tests run the real window flow in-process and assert the
process peak RSS stays far under a 2 GB budget (ru_maxrss: bytes on macOS,
KiB elsewhere — the tests/corpus.py convention).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from fastedit.inference.chunked_merge import (
    _MAX_MERGE_PROMPT_CHARS,
    _MAX_SNIPPET_CHARS,
    chunked_merge,
)
from fastedit.inference.merge import MergeResult
from fastedit.inference.text_match import deterministic_edit

_RSS_BUDGET_MB = 2_000

# The RSS guard runs the scenario in a DEDICATED child process:
# ``resource.getrusage(...).ru_maxrss`` is a whole-PROCESS watermark, so
# measured inside the pytest process it would report the cumulative peak of
# every test that ran before it (measured: ~4 GB after the full suite) —
# a number that says nothing about THIS scenario. A fresh child isolates
# the scenario's own footprint, which is the quantity the 80–97 GB bug is
# about. Units follow the tests/corpus.py convention (bytes on macOS, KiB
# elsewhere).
_CHILD_SCRIPT = """import json, resource, sys

def _rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024

file_path, snippet_path, language, replace = sys.argv[1:5]
language = None if language == "none" else language
replace = None if replace == "none" else replace

from fastedit.inference.chunked_merge import chunked_merge
from fastedit.inference.merge import MergeResult
from fastedit.inference.text_match import deterministic_edit

out = {"calls": 0, "rejected": None, "raised": None}

def merge_fn(original_code, update_snippet, language_=None):
    out["calls"] += 1
    edited = deterministic_edit(original_code, update_snippet)
    return MergeResult(
        merged_code=edited if edited is not None else original_code,
        parse_valid=True, tokens_generated=1, latency_ms=1.0,
        tokens_per_second=1000.0)

try:
    result = chunked_merge(
        original_code=open(file_path, encoding="utf-8").read(),
        snippet=open(snippet_path, encoding="utf-8").read(),
        file_path=file_path,
        merge_fn=merge_fn,
        language=language,
        replace=replace,
    )
    out["rejected"] = result.chunks_rejected
    out["merged_has_new"] = "NEW LINE INSERTED" in result.merged_code
except ValueError as exc:
    out["raised"] = str(exc)
out["rss_mb"] = round(_rss_mb(), 1)
print("CHILD: " + json.dumps(out))
"""


def _run_in_child(tmp_path, file_text: str, snippet: str, language, replace):
    """Run chunked_merge in a fresh child; return its JSON report dict."""
    f = tmp_path / "child_input.bin"
    f.write_text(file_text, encoding="utf-8")
    s = tmp_path / "child_snippet.txt"
    s.write_text(snippet, encoding="utf-8")
    script = tmp_path / "child_scenario.py"
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, str(script),
            str(f), str(s),
            language or "none", replace or "none",
        ],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("CHILD: "):
            return json.loads(line[len("CHILD: "):])
    raise AssertionError(f"child produced no report: {proc.stdout[-400:]}")


class _RecordingMergeFn:
    """Never-should-be-called sentinel: records every prompt it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, original_code, update_snippet, language=None):
        self.calls.append((len(original_code), len(update_snippet)))
        return MergeResult(
            merged_code=original_code,
            parse_valid=True,
            tokens_generated=1,
            latency_ms=1.0,
            tokens_per_second=1000.0,
        )


class _PerfectMergeFn:
    """Emulates a correct small model: deterministic_edit, else echo.

    Records the LARGEST prompt it was handed so tests can pin the
    every-prompt-within-budget invariant.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.max_prompt_chars = 0

    def __call__(self, original_code, update_snippet, language=None):
        self.calls += 1
        self.max_prompt_chars = max(self.max_prompt_chars, len(original_code))
        edited = deterministic_edit(original_code, update_snippet)
        return MergeResult(
            merged_code=edited if edited is not None else original_code,
            parse_valid=True,
            tokens_generated=1,
            latency_ms=1.0,
            tokens_per_second=1000.0,
        )


# ---------------------------------------------------------------------------
# 1. The whole-file path: the character budget the LINE gate could not see
# ---------------------------------------------------------------------------


def test_single_line_oversized_file_refuses_before_model_call(tmp_path):
    """A 1-line file over the prompt budget must fail loud, model untouched.

    This is the reported shape: total_lines == 1 passes the 150-LINE
    whole-file gate, so before the fix the whole 100 MB line went to
    merge_fn on every retry attempt (80–97 GB with the real engine).
    """
    f = tmp_path / "big.log"
    f.write_text("x" * (_MAX_MERGE_PROMPT_CHARS + 1), encoding="utf-8")
    fn = _RecordingMergeFn()
    with pytest.raises(ValueError, match="file too large for a whole-file merge"):
        chunked_merge(
            original_code=f.read_text(),
            snippet="a brand new log line\n",
            file_path=str(f),
            merge_fn=fn,
            language=None,
        )
    assert fn.calls == []


def test_whole_file_gate_fires_before_tag_escape(monkeypatch):
    """The budget gate runs BEFORE the tag-escape copy (issue #13 recheck).

    _escape_tags is a full-text transform of the original; on oversized
    input it is wasted work (a full scan, and a full copy when the text
    carries literal tags). Escape never shortens (the placeholders are
    longer than the tags), so a file over budget is over budget escaped
    too -- the gate must refuse before the escape runs at all.
    """
    import fastedit.inference.chunked_merge as cm

    def _no_escape(text, nonce=""):  # pragma: no cover - must never run
        raise AssertionError(
            "tag escape must not run on input over the prompt budget"
        )

    monkeypatch.setattr(cm, "_escape_tags", _no_escape)
    fn = _RecordingMergeFn()
    with pytest.raises(ValueError, match="file too large for a whole-file merge"):
        cm.chunked_merge(
            original_code="x" * (cm._MAX_MERGE_PROMPT_CHARS + 1),
            snippet="a brand new log line\n",
            file_path="big.log",
            merge_fn=fn,
            language=None,
        )
    assert fn.calls == []


def test_few_line_oversized_file_refuses_on_character_budget(tmp_path):
    """≤150 LINES but over the character budget: the char gate must catch it.

    100 lines × 3 KB — the line gate passes, the model could never re-emit
    300 KB within its 16,384-token output envelope.
    """
    f = tmp_path / "wide.txt"
    line = "y" * 3000
    f.write_text("\n".join(line for _ in range(100)) + "\n", encoding="utf-8")
    fn = _RecordingMergeFn()
    with pytest.raises(ValueError, match="file too large for a whole-file merge"):
        chunked_merge(
            original_code=f.read_text(),
            snippet="a brand new line\n",
            file_path=str(f),
            merge_fn=fn,
            language=None,
        )
    assert fn.calls == []


# ---------------------------------------------------------------------------
# 2. The per-chunk path: a giant symbol chunk is refused BEFORE the model
# ---------------------------------------------------------------------------


def test_oversized_symbol_chunk_refuses_before_model_call(tmp_path):
    """A replace= chunk over the prompt budget raises before merge_fn runs.

    Before the fix the model was called with the whole-symbol chunk and the
    size problem surfaced only afterwards at the LCS DP gate (issue #12's
    ``symbol too large`` ValueError) — after burning 8 model attempts.
    """
    body = _MAX_MERGE_PROMPT_CHARS // 12 + 100
    parts = ["def big():\n"]
    parts += [f"    x{i} = {i}\n" for i in range(body)]
    parts.append("    return x0\n\n\n")
    # Tail symbols keep the file from being one whole-file chunk (that gate
    # has its own 150-line refusal): the chunk below is the SYMBOL chunk.
    parts += [f"def tail_{k}(a, b):\n    return a + b + {k}\n\n\n" for k in range(3)]
    f = tmp_path / "mod.py"
    f.write_text("".join(parts), encoding="utf-8")
    assert len(f.read_text()) > _MAX_MERGE_PROMPT_CHARS
    fn = _RecordingMergeFn()
    with pytest.raises(ValueError, match=r"chunk 1-\d+ too large for a merge"):
        chunked_merge(
            original_code=f.read_text(),
            snippet="def big():\n    # ... existing code ...\n",
            file_path=str(f),
            merge_fn=fn,
            language="python",
            replace="big",
        )
    assert fn.calls == []


# ---------------------------------------------------------------------------
# 3. The snippet itself: bounded before the per-character masker runs
# ---------------------------------------------------------------------------


def test_oversized_snippet_refuses_loudly(tmp_path):
    """A snippet over the budget is refused before any O(n)-per-char work."""
    f = tmp_path / "small.txt"
    f.write_text("hello\n", encoding="utf-8")
    fn = _RecordingMergeFn()
    with pytest.raises(ValueError, match="snippet too large"):
        chunked_merge(
            original_code="hello\n",
            snippet="z" * (_MAX_SNIPPET_CHARS + 1),
            file_path=str(f),
            merge_fn=fn,
            language=None,
        )
    assert fn.calls == []


# ---------------------------------------------------------------------------
# 4. No regression: normal edits still merge end-to-end
# ---------------------------------------------------------------------------


def test_small_whole_file_merge_still_merges(tmp_path):
    """A small anchored whole-file merge (the D1 shape) still converges."""
    lines = [f"Paragraph {i}: the quick brown fox {i}." for i in range(1, 61)]
    lines[29] = "UNIQUE ANCHOR LINE ALPHA"
    lines[30] = "UNIQUE ANCHOR LINE BETA"
    f = tmp_path / "small.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snippet = (
        "UNIQUE ANCHOR LINE ALPHA\nNEW LINE INSERTED\nUNIQUE ANCHOR LINE BETA\n"
    )
    fn = _PerfectMergeFn()
    result = chunked_merge(
        original_code=f.read_text(),
        snippet=snippet,
        file_path=str(f),
        merge_fn=fn,
        language=None,
    )
    assert result.chunks_used == 1
    assert result.chunks_rejected == 0
    assert "NEW LINE INSERTED" in result.merged_code
    assert fn.max_prompt_chars <= _MAX_MERGE_PROMPT_CHARS


@pytest.mark.parametrize("n_lines", [5_000, 50_000])
def test_window_flow_scales_within_rss_budget(tmp_path, n_lines):
    """The windowed flow completes on 5k/50k-line files within the RSS cap.

    Scaling guard for the growth curve: the scenario's own peak RSS must
    stay far under 2 GB (the reported bug climbed to 80–97 GB), and every
    prompt handed to the model must sit inside the character budget.
    """
    lines = [f"Paragraph {i}: the quick brown fox {i}." for i in range(1, n_lines + 1)]
    mid = n_lines // 2
    lines[mid - 1] = "UNIQUE ANCHOR LINE ALPHA"
    lines[mid] = "UNIQUE ANCHOR LINE BETA"
    snippet = (
        "UNIQUE ANCHOR LINE ALPHA\nNEW LINE INSERTED\nUNIQUE ANCHOR LINE BETA\n"
    )
    report = _run_in_child(tmp_path, "\n".join(lines) + "\n", snippet, None, None)
    assert report["calls"] >= 1
    assert report["rejected"] == 0
    assert report["merged_has_new"] is True
    assert report["rss_mb"] < _RSS_BUDGET_MB


def test_single_line_50mb_refuses_within_rss_budget(tmp_path):
    """The 50 MB single-line repro: loud refusal, no model call, RSS bound.

    "Completes" means the pipeline finishes promptly and cleanly — a
    ValueError with the remedy — instead of climbing toward the reported
    80–97 GB (or an exit-137 kill) on the way to an inevitable refusal.
    """
    record = '{"level":"info","msg":"xxxxxxxxxxxxxxxxxxxx","ts":1234567890}'
    single_line = record * ((50 * 1024 * 1024) // len(record) + 1)
    report = _run_in_child(tmp_path, single_line, "a brand new log line\n", None, None)
    assert report["calls"] == 0
    assert report["raised"] is not None
    assert "file too large for a whole-file merge" in report["raised"]
    assert report["rss_mb"] < _RSS_BUDGET_MB
