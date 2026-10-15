"""Step D4 mixed-language golden generator (req. 3 + 8) — INDEPENDENT oracle.

``uv run python tests/golden/mixed_md/_generate.py`` re-derives the
committed expected file from the authored original and the manifest's
translation table.

THE ORACLE IS INDEPENDENT OF FASTEDIT: it never imports fastedit, never
runs its pipeline, never consults a parser. The expected file is pure
line-splice arithmetic — the original's lines with each Chinese comment
LINE swapped for its declared English translation line, every other byte
identical (the B3/C1 oracle discipline, applied at line granularity).

Every declared translation line is cross-checked against the authored
original (exactly one occurrence, byte-exact, EOL-insensitive) before
anything is written, so the table cannot silently drift from the fixture.
The manifest's trait inventory is cross-checked the same way: the
frontmatter delimiter/key lines and every declared fence info string must
occur in the original, and the trailing unclosed fence must be the file's
LAST fence-shaped line.
"""

from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent


def oracle_swap_lines(
    original_lines: list[str],
    translations: list[dict],
) -> list[str]:
    """The line-splice oracle: swap each declared zh line for its en line.

    Line-identity arithmetic only — no parsing, no pipeline. Each zh line
    must occur EXACTLY once in the original (the call sites below enforce
    that before splicing); its single occurrence is replaced by the en
    line, everything else passes through untouched.
    """
    lines = list(original_lines)
    for entry in translations:
        zh, en = entry["zh"], entry["en"]
        hits = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == zh]
        if len(hits) != 1:
            raise AssertionError(
                f"translation table drifted from the fixture: {zh!r} occurs "
                f"{len(hits)} time(s) in original.md (expected exactly 1)"
            )
        lines[hits[0]] = en + "\n"
    return lines


def _cross_check(original_text: str, manifest: dict) -> None:
    """Assert the manifest's declared traits exist in the authored fixture."""
    lines = original_text.splitlines()
    frontmatter = manifest["traits"]["frontmatter"]
    assert lines[0] == frontmatter["delimiters"][0], (
        "fixture drift: the file must START with the frontmatter delimiter"
    )
    for key in frontmatter["keys"]:
        assert key in lines, f"fixture drift: frontmatter key {key!r} missing"
    for info in manifest["traits"]["fence_info_strings"]:
        expected_line = "```" if info == "" else f"```{info}"
        assert expected_line in lines, (
            f"fixture drift: no fence with info string {info!r}"
        )
    unclosed = manifest["traits"]["unclosed_fence_opener"]
    fence_lines = [
        line for line in lines
        if line.strip().startswith("```") or line.strip().startswith("~~~")
    ]
    assert fence_lines and fence_lines[-1] == unclosed, (
        "fixture drift: the unclosed fence must be the file's LAST fence line"
    )
    lang_attr = manifest["traits"]["embedded_html_lang"]
    assert any(lang_attr in line for line in lines), (
        f"fixture drift: embedded {lang_attr!r} missing"
    )
    for comment in manifest["preserved_english_comments"]:
        assert comment in original_text, (
            f"fixture drift: preserved English comment {comment!r} missing"
        )


def main() -> None:
    original = (DIR / "original.md").read_text(encoding="utf-8")
    manifest = json.loads((DIR / "corpus.json").read_text(encoding="utf-8"))
    _cross_check(original, manifest)

    expected_lines = oracle_swap_lines(
        original.splitlines(keepends=True), manifest["translations"],
    )
    expected = "".join(expected_lines)
    # The oracle's own identity checks: the expected file differs from the
    # original in EXACTLY the declared line positions, and nowhere else.
    assert expected != original, "the translation table changed nothing"
    orig_lines = original.splitlines()
    exp_lines = expected.splitlines()
    assert len(orig_lines) == len(exp_lines), "line count must be preserved"
    changed = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(orig_lines, exp_lines))
        if a != b
    ]
    assert len(changed) == len(manifest["translations"]), (
        f"the oracle changed {len(changed)} line(s); the table declares "
        f"{len(manifest['translations'])}"
    )

    out = DIR / manifest["expected"]
    out.write_text(expected, encoding="utf-8")
    print(f"wrote {out} ({len(expected.encode('utf-8'))} bytes, "
          f"{len(changed)} line(s) swapped)")


if __name__ == "__main__":
    main()
