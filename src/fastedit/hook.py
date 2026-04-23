"""PreToolUse hook: redirect Edit → fast_edit (MCP).

Blocks Claude Code's built-in Edit tool and tells Claude to use fast_edit
instead — but only for file types fastedit actually supports. For unsupported
extensions (.toml, .md, .yaml, .json, .sh, etc.) the hook exits silently so
Edit proceeds normally.

Works on Mac, Linux, and Windows (pure Python, no dependencies).

Install:
    Add to .claude/settings.json or project .claude.json:
    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Edit",
            "hooks": [{"type": "command", "command": "fastedit-hook"}]
          }
        ]
      }
    }
"""

import json
import sys
from pathlib import Path

try:
    from .data_gen.ast_analyzer import EXTENSION_TO_LANGUAGE
    SUPPORTED_EXTS = set(EXTENSION_TO_LANGUAGE.keys())
except ImportError:
    # If ast_analyzer has a real bug (SyntaxError, NameError, …) we want that
    # to surface — silently falling back would hide it. The fallback list is
    # only for install-time issues where the package isn't fully assembled.
    SUPPORTED_EXTS = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java",
        ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
        ".rb", ".swift", ".kt", ".kts", ".cs", ".php", ".ex", ".exs",
    }


def main():
    inp = json.load(sys.stdin)
    tool_input = inp.get("tool_input", {})

    file_path = tool_input.get("file_path") or tool_input.get("path") or ""
    ext = Path(file_path).suffix.lower() if file_path else ""

    # Fall through to built-in Edit for any path with an extension outside
    # the supported set AND for extensionless files / dotfiles (Makefile,
    # .env, .bashrc, etc. — Path.suffix is "" for names starting with a dot
    # or with no '.' at all). Only block when we have a file_path whose
    # extension FastEdit can actually parse; block by default if no path
    # was provided at all.
    if file_path and ext not in SUPPORTED_EXTS:
        sys.exit(0)

    hint = "Use fast_edit (MCP) instead of Edit."
    if tool_input.get("old_string") and tool_input.get("new_string"):
        hint += " Use replace= for the containing function with context markers around the change."

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Use fast_edit MCP tool instead — AST-scoped, faster, fewer tokens.",
                "additionalContext": hint,
            }
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
