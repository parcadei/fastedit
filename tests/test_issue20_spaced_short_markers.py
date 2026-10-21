"""Issue #20: spaced whole-line keep-markers must refuse a --replace edit.

The predicate gap -- ``snippet_has_keep_marker`` scanned for marker phrases
mid-line but missed whole-line spellings ``is_marker_line`` accepts (``//
...``, ``# ...``) -- let a partial ``--replace`` snippet reach the
direct-replacement splice. The WHOLE symbol was replaced by the snippet, the
``// ...`` lines written literally, exit 0: silent data loss.

The fix pins three behaviors:

1. Unit: every whole-line spelling ``is_marker_line`` accepts is a
   keep-marker for ``snippet_has_keep_marker`` too.
2. Escape (the loud-direction tradeoff, per the issue's step 1): a
   marker line the ORIGINAL symbol body already carries identically is
   restated content -- the snippet is a full-body replacement that keeps
   the body's own marker-comment line -- so it must NOT refuse.
3. E2E (TypeScript, the reported shape): ``// ...`` plus a changed line
   between markers refuses with exit != 0 and an unchanged file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastedit.inference.markers import is_marker_line
from fastedit.inference.text_match import snippet_has_keep_marker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


class TestWholeLineMarkerSpellingsAreKeepMarkers:
    """1. Every whole-line spelling ``is_marker_line`` accepts is detected."""

    def test_spaced_short_markers_are_keep_markers(self):
        """`// ...` and `# ...` are markers for is_marker_line, so they are
        keep-markers for snippet_has_keep_marker too (the reported gap)."""
        for spelling in ("// ...", "//  ...", "# ...", "//...", "#...", "…"):
            line = "        " + spelling
            assert is_marker_line(line) is True, spelling
            assert snippet_has_keep_marker(line) is True, spelling

    def test_long_form_under_indentation_still_detected(self):
        assert snippet_has_keep_marker("    // ... existing code ...") is True


class TestRestatedMarkerCommentIsContent:
    """2. A marker line the original body already carries is content."""

    def test_body_contents_own_marker_comment_does_not_refuse(self):
        body = "def f(x):\n    # ...\n    return x + 1\n"
        full_body = "def f(x):\n    # ...\n    return x + 2\n"
        assert snippet_has_keep_marker(full_body, body) is False

    def test_extra_marker_not_in_the_body_still_refuses(self):
        body = "def f(x):\n    return x + 1\n"
        snippet = "def f(x):\n    # ...\n    return x + 1\n"
        assert snippet_has_keep_marker(snippet, body) is True

    def test_no_original_keeps_the_loud_direction(self):
        assert snippet_has_keep_marker("x = 1\n// ...\n") is True


MIN_TS = """export async function poll(get: () => Promise<number>): Promise<number> {
  let st = 0
  let tries = 0
  const deadline = Date.now() + 1000
  while (Date.now() < deadline) {
    try {
      st = await get()
    } catch (err) {
      const name = (err as { name?: string }).name
      if (name === 'TimeoutError') continue
      throw err
    }
    tries += 1
  }
  return st + tries
}

export const other = 1
"""


class TestSpacedMarkerE2eRefuses:
    """3. The reported shape refuses end to end; the file is unchanged."""

    def _write(self, tmp_path: Path) -> Path:
        p = tmp_path / "min.ts"
        p.write_text(MIN_TS, encoding="utf-8")
        return p

    def test_spaced_short_markers_with_changed_line_refuse(self, tmp_path):
        target = self._write(tmp_path)
        before = target.read_text(encoding="utf-8")
        marker_snippet = (
            "// ...\n"
            "      const name = (err as { name?: string }).name\n"
            "      if (name === 'TimeoutError') {\n"
            "        tries += 1\n"
            "        continue\n"
            "      }\n"
            "      throw err\n"
            "// ...\n"
        )
        result = run_cli(
            "edit", str(target), "--replace", "poll",
            "--snippet", marker_snippet,
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert "keep-marker" in (result.stdout + result.stderr)
        assert target.read_text(encoding="utf-8") == before

    def test_hash_spelling_refuses_too(self, tmp_path):
        target = self._write(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "poll",
            "--snippet", "// ...\n      throw err\n# ...\n",
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert target.read_text(encoding="utf-8") == before

    def test_canonical_long_form_refusal_is_unchanged(self, tmp_path):
        target = self._write(tmp_path)
        before = target.read_text(encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "poll",
            "--snippet", (
                "// ... existing code ...\n"
                "      const name = (err as { name?: string }).name\n"
                "      if (name === 'TerminalError') continue\n"
                "      throw err\n"
                "// ... existing code ...\n"
            ),
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert target.read_text(encoding="utf-8") == before
