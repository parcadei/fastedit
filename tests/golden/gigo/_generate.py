"""Generate the committed D5 GIGO goldens (tests/golden/gigo/).

The originals are byte-exact transcriptions of the fixtures measured
against the real model (see tests/test_real_llm_gigo_e2e.py's module
docstring); the expected files are computed by the INDEPENDENT
line-splice arithmetic below — never fastedit. Both test suites
(``test_real_llm_gigo_e2e.py``, ``test_gigo_hermetic.py``) re-derive the
expected bytes with their own splice oracle and assert the committed
files agree (the B3 oracle-agreement discipline).
"""

from __future__ import annotations

from pathlib import Path

GIGO = Path(__file__).resolve().parent

# ── py_defects: syntax error + typo + style crimes (case a) ──────────────
PY_DEFECTS = (
    "#!/usr/bin/env python3\n"
    '"""Inventory report utilities."""\n'
    "\n"
    "\n"
    "def alpha()\n"
    "\t# the misspelled accumulator ships as-is\n"
    "\treuslt = 0\n"
    "\t\treuslt += 1\n"
    "\treturn reuslt  \n"
    "   \n"
    "\n"
    "def audit_lock():\n"
    "    return contextlib.nullcontext()\n"
    "\n"
    "\n"
    "def beta(items):\n"
    "    lines = []\n"
    "    for item in items:\n"
    "        lines.append(str(item))\n"
    "    lines.append(str(len(items)))\n"
    "    return \" \".join(lines)\n"
)


def wrap_beta(source: str) -> str:
    lines = source.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def beta"))
    body = lines[start + 1:]
    return "".join(
        lines[:start + 1] + ["    with audit_lock():\n"]
        + ["    " + ln for ln in body],
    )


# ── md_frontmatter: malformed frontmatter (cases b, c) ───────────────────
MD_ORIGINAL = (
    "---\n"
    "title: [unclosed list\n"
    "tags: [a, b\n"
    "date: 2024-13-45\n"
    "\n"
    "# Field notes\n"
    "Body paragraph one opens the sensor log.\n"
    "Body paragraph two records the drift.\n"
    "\n"
    "## Appendix\n"
    "Closing prose.\n"
)
MD_BODY_EDIT = MD_ORIGINAL.replace(
    "Body paragraph one opens the sensor log.\n",
    "Body paragraph ONE was edited by the command.\n",
)
MD_FRONT_FIX = (
    "---\n"
    "title: Release Notes\n"
    "tags: [a, b]\n"
    "date: 2024-01-15\n"
    "---\n"
    "\n"
    "# Field notes\n"
    "Body paragraph one opens the sensor log.\n"
    "Body paragraph two records the drift.\n"
    "\n"
    "## Appendix\n"
    "Closing prose.\n"
)

# ── md_fence: the D4 corpus shape — unclosed fence (case d) ──────────────
MD_FENCE = (
    "---\n"
    "title: Field notes\n"
    "---\n"
    "\n"
    "# Field notes\n"
    "Body paragraph one lives here.\n"
    "Body paragraph two lives here.\n"
    "\n"
    "## Appendix\n"
    "\n"
    "```text\n"
    "leftover snippet\n"
)
MD_FENCE_EDIT = MD_FENCE.replace(
    "Body paragraph one lives here.\n",
    "Body paragraph ONE was edited by the command.\n",
)

# ── json_trailing_comma (case e) ──────────────────────────────────────────
JSON_ORIGINAL = (
    "{\n"
    "  \"name\": \"report\",\n"
    "  \"totals\": {\n"
    "    \"sum\": 42,\n"
    "  }\n"
    "}\n"
)
JSON_EDIT = JSON_ORIGINAL.replace('"name": "report"', '"name": "final report"')

# ── py_typo: replacement target carries the typo (case f) ────────────────
PY_TYPO = (
    "#!/usr/bin/env python3\n"
    '"""Inventory report utilities."""\n'
    "\n"
    "\n"
    "def alpha():\n"
    "    reuslt = 0\n"
    "    return reuslt\n"
    "\n"
    "\n"
    "def beta(items):\n"
    "    reuslt = 0\n"
    "    for item in items:\n"
    "        reuslt += len(items)\n"
    "    return reuslt\n"
)
PY_TYPO_EDIT = PY_TYPO.replace(
    "def beta(items):\n"
    "    reuslt = 0\n"
    "    for item in items:\n"
    "        reuslt += len(items)\n"
    "    return reuslt\n",
    "def beta(items):\n"
    "    total = len(items)\n"
    "    return f\"count={total}\"\n",
)


def main() -> None:
    cases = {
        "py_defects/original.py": PY_DEFECTS,
        "py_defects/expected_wrap_beta.py": wrap_beta(PY_DEFECTS),
        "md_frontmatter/original.md": MD_ORIGINAL,
        "md_frontmatter/expected_body_edit.md": MD_BODY_EDIT,
        "md_frontmatter/expected_frontmatter_fix.md": MD_FRONT_FIX,
        "md_fence/original.md": MD_FENCE,
        "md_fence/expected_body_edit.md": MD_FENCE_EDIT,
        "json_trailing_comma/original.json": JSON_ORIGINAL,
        "json_trailing_comma/expected_edit.json": JSON_EDIT,
        "py_typo/original.py": PY_TYPO,
        "py_typo/expected_replace_beta.py": PY_TYPO_EDIT,
    }
    for rel, text in cases.items():
        path = GIGO / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({len(text.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
