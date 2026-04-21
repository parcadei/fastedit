# FastEdit

AST-aware code editing powered by a fine-tuned 1.7B small language model.

## The problem

Every AI code editor today makes the model output old code to locate edits. Whether it's unified diffs, SEARCH/REPLACE blocks, or `apply_patch` — the model has to repeat back the lines it wants to change:

```
# Claude Code / Codex — model outputs old AND new code
@@ -1,4 +1,6 @@
 def process(data):
-    result = transform(data)
-    return result
+    try:
+        result = transform(data)
+        return result
+    except Error as e:
+        return {"error": str(e)}
```

You're paying double: the model writes every old line (to say "find this") plus every new line (to say "put this"). On a 50-line function where you change 3 lines, that's 50 lines of output just for location, plus 3 lines of actual edit. **~94% of output tokens are wasted on telling the model where to put the code.**

## How FastEdit works

FastEdit eliminates location tokens entirely. Instead of making the model repeat old code, it uses two things:

1. **AST awareness** — tree-sitter parses the file and finds the target function/class by name. No need to output old lines for location.
2. **A fine-tuned 1.7B SLM** — when the edit is complex, a small merge model takes the original chunk (~35 lines) + edit snippet and produces the merged result.

```python
# FastEdit — model writes ONLY the change
fastedit edit api.py --replace process --snippet '
def process(data):
    try:
        # ... existing code ...
    except Error as e:
        return {"error": str(e)}
'
```

`--replace process` uses tree-sitter to find the function. `# ... existing code ...` tells the system to preserve untouched lines. The model never outputs old code — zero tokens spent on location.

## The three edit modes

| Mode | What happens | Tokens | Speed |
|------|-------------|--------|-------|
| `--after symbol` | Text insertion after the named symbol | 0 | Instant |
| `--replace symbol` (deterministic) | Context anchors splice new lines in | 0 | Instant |
| `--replace symbol` (model) | 1.7B SLM merges snippet into ~35-line chunk | ~40 | <1s |

The system tries deterministic text-matching first. It classifies each snippet line as "context" (matches the original) or "new" (the edit), then splices new lines between the matched anchors. This handles **74% of real edits** with zero model calls.

When deterministic matching can't resolve the edit (indent structure changes, full rewrites, <2 matching lines), the 1.7B model takes over. It only ever sees a ~35-line function — never the whole file — so it's fast and accurate.

## Install

```bash
pip install fastedits
fastedit pull          # downloads the 1.7B model (~3GB, one-time)
```

For Apple Silicon (recommended for local use):
```bash
pip install fastedits[mlx]
fastedit pull
```

For GPU servers (vLLM, TGI) or local servers (LM Studio, llama.cpp, Ollama):
```bash
# Optional: install vLLM for GPU serving
pip install fastedits[vllm]

# Or point at any OpenAI-compatible server:
FASTEDIT_BACKEND=llm FASTEDIT_LLM_API_BASE=http://localhost:1234/v1 fastedit edit ...
```

## CLI

```bash
# View file structure (functions, classes, line ranges)
fastedit read src/app.py

# Edit a function (AST-scoped merge)
fastedit edit src/app.py --replace handle_request --snippet '
def handle_request(data):
    validate(data)
    # ... existing code ...
    logger.info("done")
'

# Insert new code after a symbol (0 tokens)
fastedit edit src/app.py --after handle_request --snippet '
def health_check():
    return {"status": "ok"}
'

# Batch edits to one file
fastedit batch-edit src/app.py --edits '[
  {"snippet": "import redis", "after": "import json"},
  {"snippet": "def cache_get(key): ...", "after": "connect"}
]'

# Delete, move, rename (all instant, no model)
fastedit delete src/app.py deprecated_handler
fastedit move src/app.py helper_func --after main
fastedit rename src/app.py old_name new_name

# Undo last edit / show diff
fastedit undo src/app.py
fastedit diff src/app.py
```

## MCP server

FastEdit runs as an MCP server for AI agents (Claude Code, Cursor, etc.):

```bash
pip install fastedits[mcp]
python -m fastedit.mcp_server
```

Add to Claude Code config:
```json
{
  "mcpServers": {
    "fastedit": {
      "command": "python",
      "args": ["-m", "fastedit.mcp_server"]
    }
  }
}
```

10 tools: `fast_edit`, `fast_batch_edit`, `fast_multi_edit`, `fast_read`, `fast_search`, `fast_diff`, `fast_delete`, `fast_move`, `fast_rename`, `fast_undo`

### Auto-redirect Edit → fast_edit (optional)

A PreToolUse hook intercepts Claude's built-in `Edit` tool and redirects to `fast_edit`. Zero tokens wasted — Edit never executes. Works on Mac, Linux, and Windows (PowerShell too).

Add to `.claude/settings.json` or your project `.claude.json`:

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

`fastedit-hook` is installed automatically with `pip install fastedits` — no paths, no `python3` vs `python` issues.

## The model

FastEdit includes a fine-tuned 1.7B parameter model (Qwen2.5-Coder-1.5B architecture) trained specifically for code merging. It takes an original code chunk + edit snippet and produces the merged result.

Most edits never reach the model:
- `--after` is pure text insertion (0 tokens, instant)
- `--replace` tries deterministic text matching first (0 tokens, instant)
- Only when the snippet has complex structural changes does the 1.7B model activate

The model is scoped to ~35-line chunks via AST, so it runs in <1s on Apple Silicon (MLX) or GPU (vLLM).

## Accuracy

Tested across 22 structurally distinct edit patterns (73 cases):

| Path | Accuracy | Tokens | Latency |
|------|----------|--------|---------|
| Deterministic (74% of edits) | 100% | 0 | <1ms |
| Model (26% of edits) | 92% | ~40 | ~500ms |
| **Combined (production)** | **~98%** | **~10 avg** | **~130ms avg** |

The deterministic path handles the easy majority perfectly and for free. The model handles the complex minority. The AST scoping prevents the failure modes that plague whole-file approaches (ordering errors, content loss).

Per-language model accuracy (156-example benchmark):

| Language | Accuracy |
|----------|----------|
| Python, Java, Kotlin, C, PHP | 92% |
| JavaScript, TypeScript, Rust, Swift | 85% |
| Go, C++, Ruby | 77% |

## How it compares

|  | FastEdit | Claude Code / Codex | Aider SEARCH/REPLACE |
|--|---------|-------------------|---------------------|
| **How it locates the edit** | AST — names the symbol | Model outputs old lines | Model outputs SEARCH block |
| **Tokens for location** | 0 | ~50% of output | ~50% of output |
| **What the model sees** | ~35-line chunk | Entire file context | Entire file context |
| **Failure mode** | Symbol not found (immediate, clear error) | Can't find old lines (silent misapply) | Can't find SEARCH block |
| **Languages** | 13 | Any | Any |

## Supported languages

Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, Swift, Kotlin, C#, PHP

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTEDIT_MODEL_PATH` | `~/.cache/fastedit/models/...` | Path to model |
| `FASTEDIT_BACKEND` | `mlx` | Backend: `mlx` or `llm` |
| `FASTEDIT_LLM_API_BASE` | `http://127.0.0.1:8000/v1` | LLM server URL (any OpenAI-compatible) |
| `FASTEDIT_LLM_MODEL` | `fastedit` | Model name to send in API requests |
| `FASTEDIT_LLM_API_KEY` | `not-needed` | API key (if server requires one) |

## License

MIT
