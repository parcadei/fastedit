# FastEdit

AST-aware code editing powered by a fine-tuned 1.7B model. Diffs, SEARCH/REPLACE, and apply_patch all force the agent to repeat back old code to say *where* the edit goes. FastEdit uses tree-sitter to find the target by name — the agent writes only the change plus a line or two of context.

### Agent output token savings

| Model | Edit tool tokens | FastEdit tokens | Saved | Reduction |
|-------|-----------------|-----------------|-------|-----------|
| GPT-5.4 | 3,404 | 1,557 | 1,847 | **54.3%** |
| Opus 4.6 | 4,286 | 2,291 | 1,995 | **46.5%** |
| Opus 4.7 | 4,771 | 2,645 | 2,126 | **44.6%** |
| Grok 4.20 | 2,946 | 1,661 | 1,285 | **43.6%** |

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
    try:
        #...
    except Error as e:
        return {"error": str(e)}
'
```

`--replace process` uses tree-sitter to find the function and auto-preserves its signature. `#...` (short form) tells the system to preserve untouched lines. The model never outputs old code — zero tokens spent on location, zero tokens spent on the signature.

## The three edit modes

| Mode | What happens | Tokens | Speed |
|------|-------------|--------|-------|
| `--after symbol` | Text insertion after the named symbol | 0 | Instant |
| `--replace symbol` (deterministic) | Context anchors splice new lines in | 0 | Instant |
| `--replace symbol` (model) | 1.7B SLM merges snippet into ~35-line chunk | ~40 | <1s |

The system tries deterministic text-matching first. It classifies each snippet line as "context" (matches the original) or "new" (the edit), then splices new lines between the matched anchors. This handles **74% of real edits** with zero model calls.

When deterministic matching can't resolve the edit (indent structure changes, full rewrites, <2 matching lines), the 1.7B model takes over. It only ever sees a ~35-line function — never the whole file — so it's fast and accurate.

## Install

**Prerequisite:** [tldr](https://github.com/parcadei/tldr-code) must be on PATH (used for AST analysis).

### Recommended — `uv tool` (handles the venv for you)

```bash
# Apple Silicon (local 1.7B model via MLX) + MCP server:
uv tool install 'fastedits[mlx,mcp]'

# GPU servers (vLLM backend) + MCP server:
uv tool install 'fastedits[vllm,mcp]'

# Generic (external OpenAI-compatible server) + MCP server:
uv tool install 'fastedits[mcp]'

# Download the 1.7B merge model (~3 GB, one-time):
fastedit pull --model mlx-8bit    # Apple Silicon
fastedit pull --model bf16        # Linux / GPU
```

The CLI lands at `~/.local/bin/fastedit`. Upgrade later with:

```bash
uv tool upgrade fastedits
```

If it says "Nothing to upgrade" right after a fresh release lands on PyPI, uv's 10-minute package-index cache is stale. Clear it and retry:

```bash
uv cache clean fastedits && uv tool upgrade fastedits
```

Or force a clean reinstall:

```bash
uv tool install --reinstall 'fastedits[mlx,mcp]'
```

Drop the `mcp` extra if you only want the CLI. Drop `mlx` / `vllm` if you only want to point at an external LLM server.

### Alternatives

```bash
# pipx (same idea, different tool):
pipx install 'fastedits[mlx,mcp]' && fastedit pull --model mlx-8bit

# Plain venv:
python3 -m venv ~/.venvs/fastedit
source ~/.venvs/fastedit/bin/activate
pip install 'fastedits[mlx,mcp]'
fastedit pull --model mlx-8bit
```

Avoid `pip install fastedits` into a Homebrew / distro-managed Python — it will fail with `error: externally-managed-environment` (PEP 668).

### Pointing at an external LLM server

```bash
FASTEDIT_BACKEND=llm FASTEDIT_LLM_API_BASE=http://localhost:1234/v1 fastedit edit ...
```

Works with LM Studio, llama.cpp, Ollama (via OpenAI-compatible endpoint), vLLM, TGI, any OpenAI-API-compatible server.

## CLI

```bash
# View file structure (functions, classes, line ranges)
fastedit read src/app.py

# Edit a function (AST-scoped merge)
# replace= auto-preserves the signature; #... marks the rest of the body.
fastedit edit src/app.py --replace handle_request --snippet '
    validate(data)
    #...
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
fastedit delete src/app.py deprecated_handler        # refuses if cross-file callers exist
fastedit delete src/app.py deprecated_handler --force # override the safety check
fastedit move src/app.py helper_func --after main
fastedit rename src/app.py old_name new_name          # AST-verified, skips strings/comments
fastedit rename src/app.py old_name new_name --dry-run

# Cross-file rename — walk a directory, skip vendor/build dirs
fastedit rename-all src/ old_name new_name
fastedit rename-all src/ old_name new_name --dry-run
fastedit rename-all src/ old_name new_name --only function  # narrow to a definition kind

# Cross-file move — relocate a symbol and rewrite importers automatically
fastedit move-to-file foo src/a.py src/b.py --dry-run
fastedit move-to-file foo src/a.py src/b.py

# Undo last edit / show diff
fastedit undo src/app.py
fastedit diff src/app.py
```

## MCP server

FastEdit runs as an MCP server for AI agents (Claude Code, Cursor, etc.). If you installed with the `mcp` extra above, the server binary is already on PATH as `fastedit-mcp`.

### One-liner setup

```bash
fastedit mcp-install                  # writes ~/.claude.json (user scope)
fastedit mcp-install --scope project  # writes ./.mcp.json (project scope)
```

Idempotent — safe to re-run. Backs up existing config before modifying.

### Manual setup

Or add the entry to `~/.claude.json` / project `.mcp.json` by hand:
```json
{
  "mcpServers": {
    "fastedit": {
      "command": "fastedit-mcp",
      "type": "stdio"
    }
  }
}
```

No hardcoded python paths, no `-m` invocation. Works wherever `fastedit` is on PATH.

12 tools: `fast_edit`, `fast_batch_edit`, `fast_multi_edit`, `fast_read`, `fast_search`, `fast_diff`, `fast_delete`, `fast_move`, `fast_move_to_file`, `fast_rename`, `fast_rename_all`, `fast_undo`

### Diagnosing issues

```bash
fastedit doctor
```

Checks binaries, Python version, backend extras, model cache state, MCP config sanity, and tldr. Run this first when anything breaks.

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

`fastedit-hook` is installed automatically with `uv tool install fastedits` — no paths, no `python3` vs `python` issues.

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
