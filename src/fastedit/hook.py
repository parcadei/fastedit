"""PreToolUse hook: redirect Edit → fast_edit (MCP) for code files only.

Blocks Claude Code's built-in Edit tool and tells Claude to use fast_edit
instead — but only when the target is a code file FastEdit can actually
handle. Falls through silently for Markdown, YAML, JSON, plain text,
and anything else outside the language whitelist.

Zero wasted tokens — Edit never executes on code files; everything else
routes normally.

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
import os
import sys

# Extensions FastEdit can actually parse + edit structurally.
# Keep in sync with fastedit.data_gen.ast_analyzer.detect_language.
_CODE_EXTENSIONS = frozenset({
    ".py",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".rs",
    ".go",
    ".java",
    ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".rb",
    ".php",
    ".swift",
    ".kt", ".kts",
    ".cs",
    ".ex", ".exs",
})


def _should_redirect(file_path: str) -> bool:
    if not file_path:
        return False
    _, ext = os.path.splitext(file_path)
    return ext.lower() in _CODE_EXTENSIONS


def main():
    inp = json.load(sys.stdin)
    tool_input = inp.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not _should_redirect(file_path):
        # Not a FastEdit-supported code file — let the Edit tool run
        # normally. Emit an empty hook response (no permissionDecision).
        json.dump({}, sys.stdout)
        return

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
