---
# ONE SKILL, TWO LOCATIONS (why this copy exists): skills/fastedit/SKILL.md
# is the single source of truth (the directory `npx skills add` reads);
# src/fastedit/skill/SKILL.md is package data shipped in the wheel and
# staged by `fastedit init`. A build-time force-include cannot replace the
# copy: editable installs (uv sync / uv run) anchor importlib.resources at
# src/, so tests/conftest.py re-copies skills/ -> src/ at every test
# session instead. EDIT THE skills/ COPY ONLY.
name: fastedit
description: >-
  AST-verified code editing via the `fastedit` CLI: edit, insert, replace, rename, move or delete a symbol in a
  source file WITHOUT repeating the old code — tree-sitter locates the target by name (175 languages classified,
  173 e2e-proven offline; 26 wired by extension by default), so the agent writes only the change plus a line or
  two of context (~45% fewer output tokens than diff/SEARCH-REPLACE tools). Insert and anchor-splice paths are 0
  tokens and instant; a 1.7B merge model handles structural rewrites in a ~35-line chunk. Results are validated
  byte-exact (relative tree-sitter parse + content-trait battery) with retry-until-valid; per-file cross-process
  locks and undo backups protect every write. Reach for it when editing code with an LLM without repeating old
  code. Use when the user asks to edit, insert, replace, rename, move or delete a symbol in a source file; when
  asked to fix or modify functions in place; when byte-level safety, concurrent-instance protection and undo
  matter.
license: MIT
---

# fastedit — write only the change

`fastedit` is AST-aware code editing: tree-sitter finds the target symbol by
name, so there is no old code to repeat back. The CLI is the primary surface.

## Core idea

- **You write ONLY the change** — `--after SYMBOL` / `--replace SYMBOL` target
  the function, class or method by name; `--replace` auto-preserves the signature.
- **Preserve by default (GIGO)** — input defects are preserved byte-exact unless
  targeted. FastEdit validates and edits; it never corrects your source. An edit
  that introduces new parse errors is refused.
- **Fail loud** — every refusal (unknown symbol, unmatched marker, lock conflict,
  parse regression) is a clear non-zero exit; the file is left unchanged.

## Quickstart

```bash
scripts/install-dev.sh
fastedit pull --model mlx-8bit
fastedit doctor
```

`install-dev.sh` is the fork's one installer (extras, grammars, model; `--dev`
for editable). `pull` downloads the ~3 GB merge model once — `mlx-8bit` on
Apple Silicon, `bf16` on Linux/GPU. External server instead of a local model:
`fastedit edit ... --backend vllm --api-base http://localhost:1234/v1` (or env:
`FASTEDIT_BACKEND=vllm FASTEDIT_VLLM_API_BASE=http://localhost:1234/v1`).

## The three edit modes

Every edit targets a symbol fastedit finds by name with tree-sitter.

1. **`--after SYMBOL`** — insert the snippet verbatim below the symbol.
   0 tokens, instant.

```bash
fastedit edit app.py --after main --snippet 'def health_check(): return "ok"'
```

2. **`--replace SYMBOL`** (deterministic) — snippet lines matching the original
   act as context anchors; new lines are spliced between them. 0 tokens,
   instant. Handles ~74% of real edits.

```bash
fastedit edit app.py --replace process --snippet 'def process(data): return transform(data)'
```

3. **`--replace SYMBOL`** (model) — when anchors cannot resolve the edit, a
   1.7B merge model rewrites just the ~35-line chunk around the symbol.
   ~40 tokens, <1s.

```bash
fastedit edit api.py --replace process --snippet '
    try:
        #...
    except Error as e:
        return {"error": str(e)}
'
```

## Snippet idioms

- Keep-marker `# ... existing code ...` (short form `#...`; `// ...` in C-family)
  means "keep the untouched lines that belong here". Honored only where an
  anchor line matches the original body — an unmatched marker is refused, never
  guessed.
- Wrap shape — a couple of real anchor lines plus keep-markers:

```bash
fastedit edit src/app.py --replace handle_request --snippet '
    validate(data)
    #...
    logger.info("done")
'
```

- Full replacement — the snippet IS the new symbol, definition line included
  (also how you rename via `--replace`); a body-only snippet is refused for
  definition targets: it would delete the signature. `--snippet @file.py`
  reads the snippet from a file; `--snippet -` reads PIPED stdin (a terminal
  is refused; never pass `-` without piping). A snippet that names an existing
  file is auto-read from it (stderr note) — pass `--snippet-is-literal` to use
  the text verbatim. Same `@file`/`-`/`--content-is-literal` convention applies
  to `create --content`.

## Commands

| Command | Purpose |
|---|---|
| `read <file>` | Structure: full content (small files) or symbol map with line ranges |
| `search <query> [path]` | Symbols, references, regex or hybrid matches (`--mode`, `--top-k`) |
| `diff <file>` | Unified diff vs a pre-edit backup (read-only). `--base last` (default) = the NEWEST backup, i.e. the last edit's change; `--base all` = every change still in the undo history — loss detection across edits |
| `edit <file>` | One edit: `--after` / `--replace SYMBOL` + `--snippet`; hermetic literal primitives `--insert-after`/`--insert-before LITERAL` and `--lines FROM:TO --delete` are 0 tokens |
| `batch-edit <file>` | Ordered edits to ONE file: `--edits` JSON list (`-` for stdin) |
| `multi-edit` | Edits across files, all-or-nothing: `--file-edits` JSON list |
| `delete <file> <sym>` | Remove a symbol by AST; refuses while cross-file callers exist (`--force` skips the check) |
| `move <file> <sym>` | Reorder within a file: `--after TARGET` |
| `rename <file> <old> <new>` | AST-verified rename in one file; strings/comments untouched (`--dry-run`) |
| `rename-all <dir> <old> <new>` | Rename across a tree; vendor/build dirs skipped (`--dry-run`, `--only kind`) |
| `move-to-file <sym> <src> <dst>` | Move a symbol to another file, rewrite importers (`--dry-run`, `--after`) |
| `create <file>` | New text file from `--content`, `--content-file` or stdin (`--force`, `--parents`) |
| `duplicate <src> <dst>` | Byte-for-byte copy, binary-safe (`--force`, `--parents`) |
| `split <file> --out DIR` | Format-aware split: `--rows`, `--by heading`, `--by element`, `--lines` |
| `join PARTS... -o FILE` | Invert split via the manifest, or explicit parts in order |
| `undo <file>` | Revert the last edit byte-for-byte; repeat to step back |
| `pull --model M` | Download the merge model: `mlx-8bit` (Apple Silicon) or `bf16` (Linux/GPU) |
| `doctor` | Self-diagnostics: binaries, extras, model cache, MCP config, tldr |
| `mcp-install` | Write the MCP entry for Claude Code (`--scope user` or `project`) |
| `init` | One-shot setup: install the fastedit agent skill for your coding agent |

Workflow: `fastedit read` before writing, `--dry-run` where offered, then `fastedit diff` / `fastedit undo` to verify or step back. After SEVERAL edits on one file run `fastedit diff --base all` — the default diff base is the NEWEST backup (the last edit's change), so loss detection across the whole undo history needs `--base all` (base: the OLDEST backup).

## Validation & retries

Every merged result is checked before anything is written:

- **Relative parse check** — the merged file's tree-sitter diagnostics are
  compared against the original's, never against an absolute "must parse clean"
  ideal. Pre-existing defects elsewhere are preserved traits; only *introduced*
  breakage is refused (`parse errors; refusing to write`).
- **Content-trait battery** — untouched lines must survive byte-exact: dropped,
  invented, reordered or selectively re-indented lines, and leaked `...`
  markers, all fail the merge.
- **Retry-until-valid** — a rejected attempt is retried with its failure reason
  appended, up to `FASTEDIT_MAX_RETRIES` attempts (default 8). On exhaustion
  the edit is refused and the file is left unchanged.

## Concurrency

FastEdit holds a per-file cross-process lock (kernel `flock` under
`~/.fastedit/locks/`) for the whole read→edit→write window. A second instance
exits 1 before anything is read or written:
`another fastedit instance (pid N, running Xs) is editing <path>; wait for it to finish — file unchanged`
The lock self-releases if the holder crashes; `--force` never bypasses it.
The stat guard (`file changed on disk since it was read`) covers non-fastedit
writers.

## Limits & constraints

- The model only ever sees a ~35-line chunk around the edit — never the file.
- Files >150 lines without a usable AST refuse whole-file merges: include a
  snippet line that matches the file (an anchor) so a window can be located.
- Backups: 5 per file, pruned after 24h, under `~/.fastedit/backups`
  (`FASTEDIT_BACKUP_DIR`) — undo depth is bounded by both.
- Unsupported extensions refuse symbol-targeted edits (no grammar = no AST);
  plain-text files need anchors for every edit.
- Not a formatter or linter (no style normalization, no auto-fixes); not a
  cross-file transaction (a crash mid-write can leave earlier targets written).

## Troubleshooting

| Error string | Fix |
|---|---|
| `another fastedit instance (pid N, running Xs) is editing <path>` | Wait for the holder; the lock self-releases if it dies, then re-run |
| `file changed on disk since it was read` | A non-fastedit writer changed the file; nothing was written; re-run against current content |
| `parse errors; refusing to write` | The edit introduced new syntax errors; fix the snippet; the file is unchanged |
| `Symbol 'x' not found` | Check the name — the error lists available symbols; `fastedit read` shows the map |
| `contains a keep-marker but the edit cannot be placed deterministically` / `has no definition` | The refusal names the fix (more surrounding unique lines, or the full replacement); or use `--after` to insert |
| `literal not found: <lit>` | The `--insert-after`/`--insert-before` anchor must equal one line's FULL STRIPPED content; check `fastedit read` for the exact line |
| `literal matches N lines` | The anchor is ambiguous — include more context in the literal, or target the spot with `--lines FROM:TO --delete` |
| `Model not found locally` | `fastedit pull --model mlx-8bit` (Apple Silicon) or `--model bf16` (Linux/GPU) |
| anything else | `fastedit doctor` |

## Install & update

Fork: `scripts/install-dev.sh` (`--dev` editable, `--revert` back to upstream
PyPI). Upstream PyPI: `uv tool install 'fastedits[mlx,mcp]'`. Upgrade: `uv
tool upgrade fastedits` or re-run the installer.
Agent skill: `fastedit init` (preferred — ships with the installed CLI;
installs global+yes, no branch ambiguity). From a clone:
`npx --yes skills add "<repo>/skills/fastedit" -g -a claude-code -y`.

MCP: a server also exists — `fastedit mcp-install` writes the entry for Claude
Code (12 `fast_*` tools). This skill's flow is intentional CLI use; the CLI
stays the primary surface.
