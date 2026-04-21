"""PreToolUse hook: redirect Edit → fast_edit (MCP).

Blocks Claude Code's built-in Edit tool and tells Claude to use fast_edit
instead. Zero wasted tokens — Edit never executes.

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


def main():
    inp = json.load(sys.stdin)
    tool_input = inp.get("tool_input", {})

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
