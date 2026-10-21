"""Tests for the PreToolUse hook: fastedit/hook.py.

Verifies that:
- Supported code extensions (.py, .ts, .rs, ...) are blocked with a redirect.
- Unsupported extensions (.toml, .md, .yaml, .json, .sh, ...) fall through
  silently so Claude Code's built-in Edit tool can proceed normally.
- Missing file_path also falls through to the block branch (preserves
  prior behavior when no path is available to inspect).
"""

import json
import subprocess
import sys

import pytest


def run_hook(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fastedit.hook"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def run_hook_raw(input_text: str) -> subprocess.CompletedProcess:
    """Run the hook with arbitrary (possibly unparsable) stdin."""
    return subprocess.run(
        [sys.executable, "-m", "fastedit.hook"],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "ext",
    [".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go",
     ".java", ".cpp", ".rb", ".swift", ".kt", ".cs", ".php", ".ex",
     # B2 triage: these extensions used to fall through, but fastedit now
     # ships verified grammars for them (html/xml/markdown/json/yaml/css/
     # bash/toml/sql/dockerfile are hard dependencies), so the hook blocks
     # built-in Edit exactly as it does for the code languages — the
     # supported set IS the extension table by design (hook.py derives
     # SUPPORTED_EXTS from EXTENSION_TO_LANGUAGE).
     ".html", ".htm", ".xml", ".md", ".json", ".yaml", ".yml",
     ".css", ".sh", ".bash", ".toml", ".sql"],
)
def test_supported_extensions_are_blocked(ext):
    result = run_hook({"tool_input": {"file_path": f"/tmp/example{ext}",
                                      "old_string": "a",
                                      "new_string": "b"}})
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    decision = payload["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "fast_edit" in decision["permissionDecisionReason"]


@pytest.mark.parametrize(
    "ext",
    # B2 triage: .toml/.md/.yaml/.yml/.json/.sh moved to the BLOCKED list
    # above (fastedit now ships grammars for them). What remains here are
    # genuinely grammar-less formats (plain text and formats with no
    # published tree-sitter wheel — probed in B2: ini/tex/vue/etc.).
    [".txt", ".ini", ".cfg", ".lock", ".env",
     ".rst", ".tex", ".vue", ".scala", ".pl"],
)
def test_unsupported_extensions_fall_through(ext):
    result = run_hook({"tool_input": {"file_path": f"/tmp/example{ext}",
                                      "old_string": "a",
                                      "new_string": "b"}})
    assert result.returncode == 0
    assert result.stdout == ""


@pytest.mark.parametrize(
    "bad_input",
    ["not-json{[", "", "[]", '"just a string"', '{"tool_input": "not-a-dict"}'],
)
def test_unparsable_input_falls_through_silently(bad_input):
    """An unparsable or non-object stdin payload falls open (exit 0, no
    output) instead of crashing the hook with a traceback."""
    result = run_hook_raw(bad_input)
    assert result.returncode == 0
    assert result.stdout == ""
    assert "Traceback" not in result.stderr

@pytest.mark.parametrize(
    "file_name",
    [".env", ".bashrc", ".gitignore", "Makefile", "Dockerfile", "README"],
)
def test_dotfiles_and_extensionless_fall_through(file_name):
    """Path.suffix is empty for leading-dot names and names without a '.',
    so these paths must still fall through to built-in Edit."""
    result = run_hook({"tool_input": {"file_path": f"/tmp/{file_name}",
                                      "old_string": "a",
                                      "new_string": "b"}})
    assert result.returncode == 0
    assert result.stdout == ""


def test_uppercase_extension_still_matches():
    result = run_hook({"tool_input": {"file_path": "/tmp/example.PY"}})
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_missing_file_path_blocks_by_default():
    result = run_hook({"tool_input": {"old_string": "a", "new_string": "b"}})
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"
