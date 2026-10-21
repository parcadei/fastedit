# Contributing to FastEdit

## Development setup

```bash
git clone https://github.com/Emasoft/fastedit && cd fastedit
uv sync --extra mlx --extra mcp --extra all-grammars
uv run pytest -q
```

Extras are platform-marked in `pyproject.toml`, so this same command works everywhere: `mlx` resolves on macOS (Apple Silicon) only, `vllm` on Linux only, `mcp` (fastmcp) anywhere, and `all-grammars` adds the offline tree-sitter grammar pack. `uv sync` also installs the `dev` group (pytest, ruff).

Concurrent-instance safety lives in `src/fastedit/file_lock.py` — read its module docstring before touching the edit path: the lock file is never unlinked (release is unlock + close only, dodging the unlink race) and acquisition is reentrant per process, so nested acquires can't self-deadlock.

## Tests

The suite runs in three runtime tiers (pytest markers; heavy tiers are deselected from the default run so it stays hermetic):

```bash
uv run pytest -q                                            # default tier: hermetic unit/integration suite (~2 min, never loads the model, never touches the network)
uv run pytest -m llm                                        # llm tier: adds real-model tests (needs the mlx extra + `fastedit pull`)
FASTEDIT_RUN_STRESS=1 uv run pytest -m "llm and stress" -q  # stress tier: adds the 100MB real-LLM stress suites (~1-2 h)
```

- The default tier never loads the model and never touches the network.
- The `llm` tier drives the real trained 1.7B model end-to-end — there is no fake LLM anywhere on the LLM path.
- Stress cases additionally require `FASTEDIT_RUN_STRESS=1`; without it they skip loudly, even under `-m llm`.
- The suite sets `FASTEDIT_BACKUP_DIR` itself, so test runs never write into `~/.fastedit/backups`.

Lint before pushing:

```bash
uv run ruff check .
```

(`tests/golden/**` is excluded from ruff — those are committed data fixtures that deliberately contain syntax defects.)

## The installer: `scripts/install-dev.sh`

This fork is deliberately not published to PyPI — PyPI stays upstream's release channel, and this script **is** the fork's distribution mechanism: it points `uv tool install` at the fork's git repo. It has two forward modes plus a revert:

- **Fork install** (default) — installs `fastedits` from the fork's git repo, pinned to a ref.
- **Dev install** (`--dev`) — installs **editable** from the repo's working tree, so local changes take effect without reinstalling.
- **Revert** (`--revert`) — uninstalls the fork and reinstalls upstream `fastedits` from PyPI; never touches model weights.

### Source menu

On an interactive run (terminal on stdin) with neither `--dev` nor `--revert`, and the script sitting inside a fastedit clone, you are asked:

```
Install from: [1] local working tree (editable, tracks your changes) [2] remote GitHub fork (pinned branch)
```

**Enter = 1**, the detected local tree. A standalone copy of the script (curl, no clone next to it) has no local tree to offer and skips the menu, installing from the remote fork. Without a terminal there is no menu: `--dev` decides, otherwise the remote fork. `--revert` never shows the menu.

### Branch autodetect

In fork mode, run from inside a clone with no explicit `--ref`, the installer defaults to **that clone's current branch** (`git rev-parse --abbrev-ref HEAD`) — so re-running it tracks the branch you are actually on. On a real run that branch is verified to exist on the fork remote (`git ls-remote --heads`, bounded) **before** anything is uninstalled; a missing branch aborts loudly listing the remote's actual heads, and an unreachable remote only warns. A detached HEAD falls back to the built-in default (`feat/create-file`). An explicit `--ref` overrides autodetect and skips the remote check — it may name a tag or a SHA, which `--heads` cannot see. `--dry-run` prints the check it would run and skips the network.

### The grammar prompt

The one question the installer asks:

```
Install all grammars? (offline 173-language tree-sitter pack — every tree-sitter-supported format works) [Y/n]
```

**Enter = yes**; answering yes appends the `all-grammars` extra. In non-interactive runs there is no prompt: all grammars default **on** and the installer says so in one line. `--all-grammars yes|no` pre-answers it; `--revert` never asks and never installs the pack.

### Preflight, sweep, postflight

Before anything is touched, the script reports:

- every installed `fastedits` (uv tool, pipx, pip) plus which `fastedit` binary wins on PATH right now;
- every model cache under `~/.cache/fastedit/models`, labelled **VALID** (holds `*.safetensors`) or **STALE** (a partial download);
- a warning if `uv` is older than 0.5, where bracket extras on a tool install may be silently dropped.

It then **sweeps** every install method (uv tool, pipx, pip) before installing, so no leftover install shadows the new one. **STALE caches are removed** — asked on a terminal `[Y/n]`, removed by default without one, printed as "would remove" by `--dry-run` — so the next pull reinstalls cleanly; **VALID caches are kept**, because they are shared between fork and upstream. `--revert` keeps every cache. After installing, the **postflight** resolves the `fastedit` that actually wins on PATH and confirms it lists the fork subcommands (`create`, `duplicate`, `split`, `join`), warning about shadowed binaries. It also installs the **agent skill** (`skills/fastedit`) with the Vercel skills CLI, straight from this clone's working tree — `npx --yes skills add <repo_root>/skills/fastedit -g -a claude-code -y` (no GitHub fetch: the repo shorthand and tree-URL forms would resolve the fork's default branch, which carries legacy skill content; a standalone run without a clone skips this axis and points at `fastedit init`, which installs the copy shipped with the package) — where a missing `npx` or a failed install only warns (never fails the installer), `--no-skill` skips the axis, and the postflight reports the result (`agent skill: installed (Claude Code, global)` / `NOT FOUND` / `skipped`); `--revert` removes the skill best-effort.

### Extras, platform mapping, model

By default the script installs **every extra this platform can install** — `mlx,mcp` on Darwin/arm64, `vllm,mcp` on Linux with an NVIDIA driver, `mcp` elsewhere — because the extras are not co-installable across platforms and an extras-less install leaves the downloaded merge model unloadable. `--extras LIST` overrides this; `--extras ""` installs bare and skips the model pull. The model pull itself (`fastedit pull`) picks `mlx-8bit` on Apple Silicon and `bf16` on Linux/GPU; `--no-model` skips the ~3 GB download (the stale-cache cleanup still runs).

### Cross-platform notes

- **Windows**: run the script under Git Bash/WSL (bash required, no native path).
- **macOS** ships no `timeout`; the script uses `gtimeout` (coreutils) when present and runs network-bounded commands unbounded otherwise — the bound is a safety net for a hung network call, not a correctness requirement.

## Pull requests

1. Branch from the current default branch and keep the change focused.
2. Make `uv run ruff check .` and `uv run pytest -q` green; run `uv run pytest -m llm` if you touched anything on the model path.
3. If you changed `scripts/install-dev.sh`, add or adjust its contract tests in `tests/test_install_dev_script.py` — they drive the real script, so keep real-run cases hermetic (stubbed `uv`/`pipx`/`fastedit` on PATH, `HOME` pointed at a temp dir, an explicit `--ref` so no network call is made).
4. Push your branch and open a PR against the fork's repo. Running the installer from your clone autodetects your branch, so `scripts/install-dev.sh` (or `--dev` for an editable install) tests your branch end-to-end. The fork is never published to PyPI.
