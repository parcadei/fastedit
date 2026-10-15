# FastEdit

> **Install from this fork** (not on PyPI — this curl one-liner is the install path):
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/Emasoft/fastedit/feat/create-file/scripts/install-dev.sh | bash
> ```
>
> Interactive: picks install source, asks about the 173-language grammar pack, installs the model + agent skill, verifies with `fastedit doctor`. Verify after: `fastedit --version`.

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

## Validation & retries

Every merge output is validated before anything is written:

- **Relative parse check (edit-not-correct)** — the merged file's tree-sitter diagnostics are compared against the *original's*, never against an absolute "must parse clean" ideal. A pre-existing syntax error elsewhere in the file is a preserved trait, not something to fix: an edit next to it still lands, byte-exact defect included. What is refused is an edit that *introduces* new breakage — in a clean file or a broken one. FastEdit validates and edits; it never silently corrects your source.
- **Content faithfulness** — untouched lines must survive byte-exact: dropped, invented, reordered or selectively re-indented lines, and leaked `...` markers, all fail the merge.

A rejected attempt is retried with its failure reason appended to the prompt (retry-until-valid), up to `FASTEDIT_MAX_RETRIES` attempts (default 8). On exhaustion the edit is refused loudly and the file is left unchanged — no partial or guessed output is ever written. When retries were consumed, the result message reports it in the metrics segment: `Applied edit to app.py. latency: 900ms, 44 tok/s, 40 tokens, 2 validation retries`.

## Concurrent instances

FastEdit holds a per-file cross-process lock for the whole read→edit→write window. A second instance (CLI or MCP) exits immediately with `another fastedit instance (pid N, running Xs) is editing <path>; wait for it to finish — file unchanged` — nothing is read, merged, or written. The lock is a kernel `flock` on a central lock file (`~/.fastedit/locks/`), so it is released when the holder dies: crash-safe by construction, never stale. `--force` never bypasses it — it only opts out of the parse/caller gates. The `ConcurrentModificationError` stat guard remains the second line of defense, for files changed by a non-fastedit writer.

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

### This fork

This fork is not published to PyPI — PyPI stays upstream's release channel. Install it with **`install-dev.sh`**, the repo's one installer. There is ONE install shape: an **editable install from a local clone** — `uv tool install --force --editable <clone>[extras]` — so what's on your system is always exactly what's in the repo.

> **Branch pin — read this first:** the fork's GitHub **default branch (`main`) is stale** — the repo's landing page shows old code. The installable branch is **`feat/create-file`**. Verify any install with `fastedit --version`.

**From-scratch install (no clone needed)** — one line, pinned to the installable branch:

```bash
curl -fsSL https://raw.githubusercontent.com/Emasoft/fastedit/feat/create-file/scripts/install-dev.sh | bash
```

The installer maintains a **managed clone** at `~/.fastedit/src` (override: `FASTEDIT_CLONE_DIR`): cloned on the first run, then fetched and reset to the tracked branch on every run, and installed editable from it. Run from inside a fastedit checkout instead, and that working tree is the install source (no managed clone is created, and `--revert` never touches your checkout).

**Update the fork** — re-run the same one-liner (idempotent: it re-syncs the managed clone to the branch tip and reinstalls), or pull directly:

```bash
git -C ~/.fastedit/src pull && uv tool install --force --editable ~/.fastedit/src
```

Run from a checkout? Update with `git pull` — the install tracks it.

Also worth knowing:

```bash
scripts/install-dev.sh --check        # read-only state report — installs nothing
scripts/install-dev.sh --revert       # back to upstream fastedits from PyPI (also removes the managed clone + agent skill)
fastedit init                         # (re)install the agent skill from the installed package
```

`--dev` still exists as a no-op compatibility flag (every install is editable now). The uvx try-before-install one-liner: `uvx --from 'fastedits[mcp] @ git+https://github.com/Emasoft/fastedit@feat/create-file' fastedit --help`. Download-then-run, if you prefer to read the script before running it: `curl -fsSL <same URL> -o /tmp/install-dev.sh && bash /tmp/install-dev.sh` — piping straight into a shell executes a partial script if the connection drops; downloading first makes the fetch succeed or fail as a whole.

After a successful install, the installer prints the two from-scratch one-liners (for people on machines without a clone) and the branch-pin note; `fastedit --version` confirms what landed. `--check` reports the current state without touching anything: installed `fastedits` per method, which `fastedit` binary wins on PATH and what `fastedit --version` reports on it, model caches (VALID/STALE), the grammar pack, the local clone, and whether the installable branch exists on the fork remote — plus the same two from-scratch one-liners. It needs no `npx` and works offline (an unreachable remote is a warning, not a failure).

### Grammars: the one question the installer asks

By default the script installs **every extra this platform can install** — `mlx,mcp` on Apple Silicon, `vllm,mcp` on Linux with an NVIDIA driver, `mcp` elsewhere — because an extras-less install leaves the downloaded merge model unloadable. On top of that it asks one yes/no question:

```
Install all grammars? (offline 173-language tree-sitter pack — every tree-sitter-supported format works) [Y/n]
```

**Enter = yes.** Answering yes adds the `all-grammars` extra, so every tree-sitter-supported language resolves offline (see "Supported languages" below). Answer `n`, or pre-answer with `--all-grammars no` to skip the pack. In non-interactive runs (CI, piped stdin) there is no prompt: all grammars default **on**, and the installer says so in one line — pass `--all-grammars no` to opt out. `--revert` never asks and never installs the pack.

After installing, the installer verifies the grammar axis honestly via the tool's own Python and warns if the pack was selected but is missing. Note `--ref` names the branch a **managed clone** tracks (standalone runs; default `feat/create-file`). From inside a checkout it has no effect — the working tree is the install source.

The fork ships under the same PyPI name and console-script names as upstream, so the script uninstalls any existing `fastedits` from every method it finds (uv tool, pipx, pip) before installing, then verifies the `fastedit` that actually ends up on PATH is the fork — not a shadowed leftover. Add `--dry-run` to any of the above to see the commands without running them.

### Pointing at an external LLM server

```bash
fastedit edit ... --backend vllm --api-base http://localhost:1234/v1
# or via env: FASTEDIT_BACKEND=vllm FASTEDIT_VLLM_API_BASE=http://localhost:1234/v1
```

Works with LM Studio, llama.cpp, Ollama (via OpenAI-compatible endpoint), vLLM, TGI, any OpenAI-API-compatible server.

## Development

To hack on FastEdit, clone the repo and use the editable dev install, so your changes take effect without reinstalling:

```bash
git clone https://github.com/Emasoft/fastedit && cd fastedit
uv sync --extra mlx --extra mcp --extra all-grammars   # extras are platform-marked; resolves everywhere
uv run pytest -q                                       # hermetic default tier
scripts/install-dev.sh --dev                           # editable install of your working tree
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the three test tiers, linting, the PR workflow, and how `scripts/install-dev.sh` works (source menu, branch autodetect, preflight/sweep/postflight, model-cache handling, per-platform extras).

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
fastedit diff src/app.py            # base: the newest backup = the last edit's change
fastedit diff src/app.py --base all # every change still in the undo history —
                                    # use this to detect losses across multiple edits

# Hermetic line primitives — 0 tokens, no model (issue #14):
# insert next to a UNIQUE line (full stripped content must match exactly)
fastedit edit src/lib.rs --snippet '    audit_trail.record(x);' --insert-after 'let x = audit(a);'
fastedit edit src/lib.rs --snippet '    audit_trail.begin();' --insert-before 'let x = audit(a);'
# delete an inclusive 1-indexed line range
fastedit edit src/app.py --lines 12:14 --delete
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

## Agent skill

Coding agents can install FastEdit as a skill — it teaches the agent the CLI surface: symbol-addressed edits (write only the change), the three edit modes, snippet idioms, validation/retries, and troubleshooting.

```bash
fastedit init                                                     # preferred: ships with the installed CLI — global + yes, no branch ambiguity
npx --yes skills add "<repo>/skills/fastedit" -g -a claude-code -y      # contributors working from a clone of this repo
```

Installed automatically by `scripts/install-dev.sh` from the repo's working tree (`skills/fastedit` — no GitHub fetch, so no branch question); pass `--no-skill` to skip, and `--revert` removes it again.

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
| **Languages** | 26 grammars wired by extension (default); 175 with all-grammars (173 e2e-proven) | Any | Any |

## Supported languages

**Default install** (hard dependencies, wired by file extension): Python, JavaScript, TypeScript/TSX, Rust, Go, Java, C, C++, Ruby, Swift, Kotlin, C#, PHP, Elixir — plus the core formats HTML, XML/SVG/DTD, Markdown, JSON, YAML, CSS, Bash, TOML, SQL, and Dockerfile (26 grammars wired by extension).

**With the `all-grammars` extra** (`pip install fastedits[all-grammars]`): Scala, Lua, Perl, Julia, Zig, Svelte, GraphQL, HCL/Terraform, Make, Nix, and every language in the bundled tree-sitter pack resolve on explicit `language=` requests (verified against the e2e census in `tests/golden/pack_census.json`).

## Testing

The suite runs in three runtime tiers (pytest markers; heavy tiers are deselected from the default run so `uv run pytest -q` stays hermetic):

```bash
uv run pytest -q                                            # default tier: hermetic unit/integration suite (~2 min, never loads the model)
uv run pytest -m llm                                        # llm tier: adds real-model tests (needs the mlx extra + `fastedit pull`)
FASTEDIT_RUN_STRESS=1 uv run pytest -m "llm and stress" -q  # stress tier: adds the 100MB real-LLM stress suites (~1-2 h)
```

- The default tier never loads the model and never touches the network.
- The `llm` tier drives the real trained 1.7B model end-to-end — there is no fake LLM anywhere on the LLM path.
- Stress cases (100MB files: code seams, CJK×CRLF, deep nesting, txt/md) additionally require `FASTEDIT_RUN_STRESS=1`; without it they skip loudly, even under `-m llm`.
- The suite sets `FASTEDIT_BACKUP_DIR` itself, so test runs never write into `~/.fastedit/backups`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTEDIT_MODEL_PATH` | `~/.cache/fastedit/models/...` | Path to model |
| `FASTEDIT_BACKEND` | `mlx` | Backend: `mlx` or `vllm` |
| `FASTEDIT_VLLM_API_BASE` | `http://127.0.0.1:8000/v1` | vLLM server URL (any OpenAI-compatible) |
| `FASTEDIT_VLLM_MODEL` | `/root/fastedit-merged` | Model name to send in API requests |
| `FASTEDIT_VLLM_API_KEY` | `not-needed` | API key (if server requires one) |
| `FASTEDIT_VLLM_MAX_TOKENS` | `16384` | Max tokens the vLLM backend requests per merge |
| `FASTEDIT_NO_UPDATE_CHECK` | unset | Set to `1` to disable the PyPI update check (CLI notice + MCP banner) |
| `FASTEDIT_BACKUP_DIR` | `~/.fastedit/backups` | Directory for undo/diff backups; must be an absolute path (a leading `~` is expanded). `fastedit undo` and `diff` only see backups stored in this directory. |
| `FASTEDIT_MAX_RETRIES` | `8` | Validation-retry budget per merge site for the retry-until-valid loop (see "Validation & retries" above). Malformed or negative values fail loudly. |
| `FASTEDIT_RUN_STRESS` | unset | Set to `1` to enable the 100MB stress tests (see "Testing" above). |

## License

MIT
