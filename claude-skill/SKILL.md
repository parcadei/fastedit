---
name: fastedit
description: Install + configure FastEdit — AST-scoped code edits via 1.7B model
---

# FastEdit Setup

FastEdit 0.6.0+ adds cross-file caller-safety on deletes, AST-verified renames, signature-impact notes on edits, and a cross-file `fast_move_to_file` that rewrites importers.

## When to Use
- "install fastedit", "set up fast edit", "configure fastedit"
- "add fastedit to claude code", "fastedit mcp", "fastedit cli"

## Instructions

Ask user: **MCP server** (recommended — tools appear in Claude Code) or **CLI** (standalone commands)?

### Install

```bash
# Mac (Apple Silicon)
pip install fastedits[mlx]

# Mac + MCP server
pip install fastedits[mlx,mcp]

# Linux GPU server
pip install fastedits[vllm]
```

### Pull model (~3GB, one-time)

```bash
fastedit pull
```

74% of edits resolve deterministically (0 tokens, <1ms). Model handles the remaining 26%.

### Option A: MCP Server (recommended)

Add to `~/.claude.json` or project `.claude.json`:

```json
{
  "mcpServers": {
    "fastedit": {
      "command": "python3",
      "args": ["-m", "fastedit.mcp_server"]
    }
  }
}
```

12 tools: `fast_edit` (replace=/after=), `fast_batch_edit`, `fast_multi_edit`, `fast_read`, `fast_search`, `fast_delete`, `fast_move`, `fast_move_to_file`, `fast_rename`, `fast_rename_all`, `fast_diff`, `fast_undo`

Optional — auto-redirect Edit → fast_edit (add to same `.claude.json`):

```json
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
```

Key patterns:
- `replace='func_name'` + context markers → AST-scoped edit (~1-2s)
- `after='func_name'` + new code → deterministic insert (0 tokens)
- `fast_read` first → learn symbol names → then edit

### Option B: CLI

```bash
fastedit read src/app.py                              # structure map
fastedit edit src/app.py --replace fn --snippet '...'  # edit function
fastedit edit src/app.py --after fn --snippet '...'    # insert after
fastedit search "query" src/                           # find symbols
fastedit delete src/app.py old_func                    # remove symbol
fastedit rename src/app.py old_name new_name           # rename symbol
fastedit diff src/app.py                               # verify last edit
fastedit undo src/app.py                               # revert last edit
```

### Remote / Local LLM Server (optional)

Any OpenAI-compatible server works: vLLM, LM Studio, llama.cpp, Ollama.

```bash
export FASTEDIT_BACKEND=llm
export FASTEDIT_LLM_API_BASE=http://localhost:1234/v1  # LM Studio default
```

| Server | Default URL |
|--------|-------------|
| vLLM | `http://localhost:8000/v1` |
| LM Studio | `http://localhost:1234/v1` |
| llama.cpp | `http://localhost:8080/v1` |
| Ollama | `http://localhost:11434/v1` |

Load the FastEdit GGUF/MLX model in your server, set the URL, done.
