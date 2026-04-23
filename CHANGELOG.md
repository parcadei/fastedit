# Changelog

All notable changes to FastEdit are documented in this file. Format: [Keep a Changelog](https://keepachangelog.com/).

## 0.5.0 — 2026-04-23

### Added
- **`fast_move_to_file` MCP tool + `fastedit move-to-file` CLI** — relocate a symbol cross-file with automatic import rewrites across all 13 supported languages (Python, TS/JS, Java, Kotlin, Rust, Go, Swift, C#, Ruby, PHP, Scala, Elixir, Lua, C/C++). Rust `use` trees handled via tree-sitter: simple, braced, arbitrary-depth nested, aliased (`as Name`), and wildcard (`*`) — the wildcard case appends an explicit import alongside the glob and surfaces an advisory note. Pass `--dry-run` for preview.
- **`--only <kind>` on `fast_rename_all` / `kind_filter` on `fast_rename_all` MCP tool** — restrict rename to `class | function | method | variable`. Resolved via tldr's structural classification, not regex.
- **`--dry-run` on `fastedit rename` / `dry_run=True` on `fast_rename` MCP tool** — parity with `rename-all`.
- **`--force` on `fastedit delete` / `force=True` on `fast_delete`** — override the new cross-file caller safety check.

### Changed
- **`fast_rename` and `fast_rename_all` AST-verified across all 13 supported languages** via `tldr references`. Strings/comments/docstrings are structurally skipped using tldr's confidence axis — real code hits at 1.0, string/comment substring noise at 0.5. On AST-native langs (Python, TS/JS, Go, Rust), refs are classified as function/method/call/import/read/variable; on the other langs the same confidence filter still gives AST-verified-quality matching. No per-language AST walk in fastedit.
- **`fast_delete` runs a cross-file caller safety check** before deleting. Refuses with a structured caller-location message (file:line:kind, up to 10 rows) if the symbol has references in other files, and suggests `fast_rename_all` as a migration path. Pass `--force` / `force=True` to bypass. Falls open with a warning note when tldr is unavailable — infra failures never fail-closed. Works across all 13 languages.
- **`fast_edit` appends an impact note** when `replace=<name>` changes the function's signature line. Reports caller counts and points at `tldr impact` / `fast_search` for detailed review. Purely informational — the edit always proceeds. Signature detection outsourced to `tldr structure` (13/13 lang coverage via tldr's per-definition `signature` field). Hot path stays subprocess-free for body-only edits (0.052ms p50 overhead verified on unchanged-signature edits).
- **Architectural refactor:** fastedit now outsources per-language AST work to tldr wherever tldr already exposes the primitive. Kind classification, signature extraction, reference discovery, and import location all go through tldr. Net code reduction in `caller_safety.py` and `rename.py` despite the feature additions.
- **MCP server instructions, per-tool descriptions, CLI help, README, and `claude-skill/SKILL.md`** all synchronized to reflect the new capabilities.

### Fixed
- **UTF-8 BOM line-1 rename bug** — `_char_col_to_byte_offset` was translating character index to byte offset while tldr already emits byte columns, causing an off-by-3 misalignment on line-1 symbols of BOM-prefixed files. Replaced with `_tldr_col_to_byte_offset` that uses tldr's column directly. Multi-byte chars before the symbol on later lines are handled correctly too.
- **`delete_symbol` no longer drops `@decorator` / `@annotation` lines** above the deleted symbol. Root cause: upstream `tldr structure` reported the wrong line-span for decorated defs. Switched to in-memory AST via `get_ast_map_from_source`.
- **`get_ast_map` debug log no longer leaks to stdout** on empty/near-empty files (surfaced during `fast_move_to_file` smoke testing). Demoted from `_log.warning` to `_log.debug`; a regression test locks stdout cleanliness.
- **Zero-match `rename-all` message** no longer says "word-boundary" (stale from the pre-AST-verified impl). Updated to "AST-verified via tldr — strings/comments/vendor dirs excluded"; regression test locks the wording for both the MCP and CLI surfaces.
- **Rust `use` tree rewriting** for `fast_move_to_file` now handles nested (arbitrary depth), aliased (`as Name`), and wildcard (`*`) cases via tree-sitter-rust. Wildcard case appends an explicit import for the moved symbol alongside the existing glob and surfaces an advisory note in the plan.

### Tests
- **756 passed** (up from 598 in 0.4.1, +158 new). Zero failures, zero xfails. 29 skipped (all intentional — per-language paths where a subsystem genuinely doesn't apply, not hidden failures).
- 91 tests from the hardening milestone (65 cross-language parametrized, 15 adversarial, 5 property/invariant, 6 tool-integration), 24 from the outsource refactor, plus regression locks on message wording, BOM line-1, decorator preservation, and Rust complex `use` cases.
- Language coverage matrix published at `docs/testing-matrix.md`: **M1, M2, M3, and M4 all verified at 13/13 languages**.

## 0.4.1 — 2026-04-23

### Fixed
- **`fast_edit(replace='ClassName', ...)` no longer spuriously rejects field-only class snippets.** The safety guard that blocks multi-symbol `replace=` snippets (to prevent silent deletion of nested methods) now uses `tldr structure`'s per-definition `kind` classification: methods and functions count as extras, field/variable/constant declarations do not. Previously the guard couldn't distinguish `private int size = 100;` from `public void set(...)`, so any `replace=ClassName` snippet containing a full class body with only a field change was rejected even though the edit was a safe direct swap. tldr is already a fastedit prerequisite (used by `fast_read`, `fast_search`, `fast_references`), so this inherits accurate per-language extraction across all 13 supported languages at ~10 ms per call. Falls back to the in-process AST analyzer if tldr is temporarily unavailable. Closes the `test_batch_edit_mixed_preserve_siblings` regression that had been carried in the backlog.
- **Signature-presence check now recognizes Java / C# / TypeScript access modifiers.** The internal helper that decides whether to auto-preserve a function/class signature (omitted signature → prepend from AST) only knew Rust's `pub` and TypeScript's `export`. `public class Cache { ... }`, `private static final class Foo { ... }`, and similar snippets were classified as "signature missing" and got their declaration prepended a second time, producing duplicated lines once the guard above stopped blocking them. The helper now accepts any sequence of modifier keywords before `class` / `struct` / `interface` / `trait` / `enum` / `impl`, and skips comment-prefixed lines (`//`, `#`, `*`, `--`) to avoid false positives from declarations referenced inside doc comments.

### Changed
- **`fastedit doctor` now reports the tldr version** (`tldr /path/to/tldr (v0.1.4)`) and treats tldr as a required dependency rather than optional, reflecting its role in the edit guard, `fast_read`, and `fast_search`. Missing-tldr now produces a failing row (exit code 1) instead of a skip.

### Tests
- **7-case regression lock for the field-vs-method guard invariant** across Python, Java, Rust, and TypeScript. Field-only class snippets pass silently; method-containing snippets raise `ValueError` with the same `additional symbol(s)` message as before. Locks both halves of the invariant so future changes to the guard can't regress one in favor of the other.

## 0.4.0 — 2026-04-23

### Added
- **`fast_rename_all` MCP tool + `fastedit rename-all` CLI** — cross-file symbol rename. Walks every supported code file under a directory, applies word-boundary rename, skips matches inside strings/comments/docstrings via tree-sitter. Prunes `.git`, `node_modules`, `__pycache__`, `target`, `dist`, `vendor`, and other common vendor/build dirs. Also prunes any directory containing a `pyvenv.cfg` (PEP 405 venv marker) so non-standard venv names like `.venv311`, `myenv`, `virtualenv` don't escape the filter. Pass `--dry-run` / `dry_run=True` to preview. Symlinks are not followed (prevents double-rename). Same-name rename is a guarded no-op. Not scope-aware (text matching with string/comment skip zones, not LSP) — unique names are safer than short common identifiers. Instant, no model. 11-case test suite covers substring collisions, dotfile dirs, binary files, and venv pruning.

### Fixed
- **Multi-line signature auto-preserve bug.** `fast_edit` with `replace='func_name'` and a body-only snippet corrupted output when the target had a multi-line parameter list (`def foo(\n    a,\n    b,\n):`). The auto-preserve path at `chunked_merge.py:496` grabbed only the first line of the signature (`def foo(`) and dropped the continuation, producing unclosed parens. Replaced the line-grab heuristic with a tree-sitter lookup of the function's `body` field — the signature is now everything from the node's start to the body's start, regardless of how many lines the parameter list spans. Verified across all 13 supported languages: Python, JavaScript, TypeScript, Rust, Go, Java, Ruby, Swift, C, C++, C#, PHP, plus Kotlin (uses unnamed `function_body` child — handled by fallback) and Elixir (parses `def foo(a) do ... end` as a call with a `do_block` child — handled by fallback). Falls back to single-line behavior for unknown grammars. 14-case parametrized regression lock plus 5 end-to-end scenario tests (single-line preserve no-regression, multi-line params, return annotations, Rust, TypeScript).

## 0.3.2 — 2026-04-23

### Fixed
- **Hook supported-extension list stays synced with the AST parser.** Thanks to [@chapayevdauren](https://github.com/chapayevdauren) (PR [#1](https://github.com/parcadei/fastedit/pull/1)) for the refactor: the hook now imports `EXTENSION_TO_LANGUAGE` directly from `fastedit.data_gen.ast_analyzer` with a hardcoded fallback for edge cases. This eliminates three silent drift bugs we'd shipped in 0.2.7 — `.mjs` and `.cjs` were in the hook's allow-list but not in the parser (edits redirected to `fast_edit`, which then errored); `.hxx` same; `.hh` was in the parser but not in the hook (silently missed the redirect). Single source of truth now.
- **Dotfiles and extensionless files fall through to built-in Edit.** `Makefile`, `Dockerfile`, `.env`, `.bashrc`, `.gitignore`, `README` — `Path.suffix` is empty for any name that starts with a dot or has no dot at all, so under the 0.3.0 logic (`if ext and ext not in SUPPORTED: fall_through`) they hit the empty-`ext` short-circuit and got blocked by the default deny path. Flagged by CodeRabbit on PR #1. Condition is now `if file_path and ext not in SUPPORTED: fall_through`, which treats any file without a recognized code extension as "not ours" — correct for every real non-code file type.

### Changed
- **Narrowed the hook's fallback `except Exception` to `except ImportError`**, also per CodeRabbit review. If `ast_analyzer` ever has a real bug (SyntaxError, NameError) we want it to surface, not be silently masked by the hardcoded fallback.

### Tests
- **First test suite in the repo.** `tests/test_hook.py` covers all 23 supported extensions (blocked with redirect), 11 common unsupported extensions (fall through silently), 6 dotfile / extensionless names (fall through), uppercase extensions, and missing `file_path`. 34 cases total, all passing. Contributed by Dauren, extended here for the dotfile regression.

## 0.3.1 — 2026-04-23

### Added
- **`fastedit doctor` now includes a remote version check.** The `fastedits package` row shows `{version} (up to date)` when current, or `{version} → {latest} available (uv tool upgrade fastedits)` when a newer release exists. Shares the 24-hour PyPI-fetch cache (`~/.cache/fastedit/update-check.json`) and `FASTEDIT_NO_UPDATE_CHECK=1` escape hatch with the existing CLI exit notice and MCP banner — one HTTP call per day across all three surfaces. Falls back gracefully to `{version} (remote version check unavailable)` when PyPI can't be reached.

### Changed
- **`update_check` module grew a public `get_version_info()` helper** returning a `(current, latest)` tuple. `get_update_notice()` now builds on it instead of duplicating the cache-and-fetch logic.

## 0.3.0 — 2026-04-23

### Added
- **`fastedit doctor`** — self-diagnostic that reports install health at a glance. Checks the three binaries on PATH (`fastedit`, `fastedit-mcp`, `fastedit-hook`), Python version, installed package version, optional backend extras (`mlx`, `vllm`, `fastmcp`), model cache (path, size, actual `.safetensors` presence, honors `FASTEDIT_MODEL_PATH`), Claude Code MCP config sanity (both `~/.claude.json` and `$(pwd)/.mcp.json`, with stale-path detection), and tldr availability. Prints per-section rows with ✓/!/✗/– markers and exits non-zero if any required check fails. Use as the first diagnostic when anything goes wrong — most common install issues (wrong extras, missing weights, stale MCP config) are now visible in one command.
- **`fastedit mcp-install`** — writes the fastedit entry into Claude Code's MCP config in one command. Defaults to `--scope user` (`~/.claude.json`); pass `--scope project` to write `$(pwd)/.mcp.json`. Idempotent: detects an existing matching entry and exits. Safe on upgrades: if a stale entry exists (e.g. hardcoded venv path from an older install), the old entry is printed, the config is backed up with a timestamped `.bak-*` suffix, and the entry is replaced with `{"command": "fastedit-mcp", "type": "stdio"}`. Warns but still writes if `fastedit-mcp` isn't on PATH yet.

## 0.2.9 — 2026-04-23

### Added
- **`fastedit-mcp` console entry point.** The MCP server is now installable as a standalone binary alongside `fastedit` and `fastedit-hook`, so MCP config files can use `"command": "fastedit-mcp"` with no hardcoded venv paths, no `python -m` invocation, no `python` vs `python3` ambiguity. Works identically in Claude Code, Cursor, Continue, and any other MCP host.

### Fixed
- **`fastedit pull` silently skipping weight download.** The cache-hit check used `any(cached_path.iterdir())`, which returned True on a partial cache (config.json + tokenizer.json present, `.safetensors` shards missing). `fastedit pull` then reported `Model ready` without downloading the weights, and every subsequent edit call failed at inference time with a cryptic model-load error. The check now requires at least one `*.safetensors` file in the cache before returning, so partial caches trigger a real download.

### Changed
- **README install + MCP setup aligned with how the package actually ships.** Primary install commands now include the `mcp` extra (`uv tool install 'fastedits[mlx,mcp]'` on Apple Silicon, `[vllm,mcp]` on GPU, `[mcp]` for external-server setups) — previously users who followed the README ended up with the MCP server importable but its `fastmcp` dependency missing, and got `ModuleNotFoundError: No module named 'mcp'` on first connect. `fastedit pull` examples now use the required `--model mlx-8bit` / `--model bf16` flag (`--model` became required in 0.2.8). The MCP config example now uses `"command": "fastedit-mcp"` (the new entry point) instead of `"command": "python", "args": ["-m", "fastedit.mcp_server"]`.

## 0.2.8 — 2026-04-23

### Fixed
- **`fastedit pull` broken default.** The CLI's `--model` default was `fastedit-1.7b-mlx-8bit`, a legacy long-form name that `model_download.py` no longer registers (valid names are `mlx-8bit` and `bf16`). Running `fastedit pull` with no args raised `RuntimeError: Unknown model 'fastedit-1.7b-mlx-8bit'`. Fixed by making `--model` required with `choices=["mlx-8bit", "bf16"]` — no silent platform default, argparse prints both options on error, and Linux/GPU users aren't tricked into downloading MLX weights they can't run.

## 0.2.7 — 2026-04-23

### Fixed
- **`fastedit-hook` blanket-denying non-code Edits.** The PreToolUse hook that
  redirects Claude Code's built-in `Edit` tool to `fast_edit` was firing on every
  file — Markdown, YAML, TOML, JSON, plain text — even though FastEdit can't
  parse those formats. Callers saw `error: Use fast_edit MCP tool instead` when
  trying to edit READMEs, changelogs, or config files. Fixed by whitelisting
  the extensions FastEdit actually handles (.py, .js/.jsx/.mjs/.cjs, .ts/.tsx,
  .rs, .go, .java, .c/.h, .cpp/.cc/.cxx/.hpp, .rb, .php, .swift, .kt/.kts,
  .cs, .ex/.exs). Non-code Edits now fall through to the built-in tool as
  they should.

### Changed
- **README install section is now PEP 668-aware.** Dropped the bare
  `pip install fastedits` instructions — they fail out of the box on Homebrew
  and distro-managed Python. New default: `uv tool install 'fastedits[mlx]'`,
  with pipx and plain-venv alternatives listed underneath. tldr is correctly
  labeled as optional (only read/search tools need it; editing runs pure
  in-memory tree-sitter).

## 0.2.6 — 2026-04-23

### Added
- **PyPI version check with MCP-aware surfacing.** On the first successful `fast_edit` /
  `fast_batch_edit` call in an MCP server session, if a newer release exists on PyPI, the
  tool response gets a one-line banner appended so the host LLM can relay the notice to the
  human who actually owns the environment. The model can't `pip install`, so the notice
  targets the user through the model's next turn. Subsequent calls in the same session are
  silent. CLI runs get a stderr notice on exit. Checks cache for 24 h in
  `~/.cache/fastedit/update-check.json`. Silent when up-to-date, on network failure, or when
  `FASTEDIT_NO_UPDATE_CHECK=1` is set. Uses stdlib `urllib` — no new dependencies.

### Changed
- **MCP tool instructions rewritten around minimal-payload patterns.** The `fast_edit` tool
  description and the top-level FastMCP instructions now lead with the six use-case patterns
  (add-guard / append / middle / full-replace / after / preserve_siblings) paired with the
  minimal snippet shape for each. The marker-position semantics (top/bottom/middle insert)
  and short marker forms (`#...` / `//...` / `…`) shipped in 0.2.4 are now surfaced to every
  LLM that picks up the tool, not buried in the changelog. Signature auto-preserve (0.2.2)
  is stated explicitly: `replace='name'` carries the def/fn/class line, so the snippet should
  omit it.
- **README teaser updated** to use the minimal form — short `#...` marker, no signature line.
  Shows what the calling agent actually writes today, not the 0.2.0 long form.

No breaking changes — all existing payloads keep working.

## 0.2.5 — 2026-04-23

### Fixed
- **Refined `wrap_block` guard.** The guard introduced in 0.2.4 was over-aggressive — it would block any marker-position snippet whose first new line ended with `:` or `{`, which caught the very common "add-guard" pattern (inserting an early-return guard at the top of a function). The guard now compares indent alignment between the block-opener and the marker: if they're at the same indent, it's an add-guard (deterministic path runs); if the marker is deeper than the block-opener, it's a genuine wrap-block (falls through to the model, as before). Unblocks the single most common `modify_inside` fixture pattern for tool-call savings.

## 0.2.4 — 2026-04-23

### Added
- **Marker-position semantics.** Snippets with a `# ... existing code ...` marker but no anchor lines can now express position implicitly:
  - `<new_lines> + marker` inserts new_lines at the top of the function body
  - `marker + <new_lines>` inserts new_lines at the bottom
  - Existing anchor-based and full-body-replace behaviors are unchanged.
  This drops the redundant "one anchor line required" for simple insertions.
- **Short marker support.** Three new marker forms are accepted, each fewer tokens than the original:
  - `#...` (Python / Ruby / Elixir) — 2 tokens
  - `//...` (JS / TS / Rust / Go / Java / C / C++ / Swift / Kotlin / C# / PHP) — 2 tokens
  - `…` (Unicode ellipsis — language-agnostic) — 1 token
  The legacy `# ... existing code ...` / `// ... existing code ...` forms continue to work. Internally, all forms are normalized to the canonical long form before processing.

These two features combined let calling agents emit snippets that are just `<new_code>` + one short marker — meaningfully closer to the "just the change" ideal for tool-call payloads.

## [0.2.3] — 2026-04-23

### Fixed
- `fast_edit` with `replace=<method_name>` now correctly handles partial edits inside a class method. The deterministic fast path previously retained old method body lines alongside the new ones when the snippet used `# ... existing code ...` markers with minimal context, silently corrupting the output. Regression test covers both Python methods and one non-Python equivalent.
- Added `[tool.hatch.build.targets.sdist]` config to pyproject.toml — prevents a 1.6 GB sdist on release builds by explicitly listing what to include/exclude. Previously this config was only applied locally in the release workspace and got lost on clean builds.

## [0.2.2] — 2026-04-23

### Fixed
- **Silent signature corruption in `replace=` fast path** — `fast_edit` with `replace=<name>` no longer silently corrupts the file when the snippet omits the function signature. FastEdit now auto-prepends the signature from the AST when it's missing from the snippet — matching the design intent of `replace=` as a structural targeting kwarg. Previously the deterministic fast path treated a body-only snippet as the full new body, wiping the `def`/`fn`/`func` line and reporting success anyway. Scoped to function/method/class-like kinds; constants and other value-like targets keep their full-replacement semantics. Regression covered by `tests/test_replace_without_signature.py` (Python + Rust, happy path, top-marker variant).

## [0.2.1] — 2026-04-22

### Fixed
- **Stale-AST corruption on chained edits** — chained `fast_edit` calls on the same file no longer corrupt output when prior edits shift line numbers. Root cause: `chunked_merge` relied on the `tldr` daemon's AST cache, which can return stale line numbers after the file is modified (file-watcher invalidation race). Fixed by parsing the source in-memory with tree-sitter directly in the edit path — no subprocess, no daemon, no cache staleness. Affected paths: `replace=` deterministic text-match, `replace=` direct-swap, `after=` insertion, and `_merge_preserve_siblings`. Regression covered by `tests/test_chained_edits_stale_ast.py` (Python + Rust fixtures plus a monkeypatched stale-AST simulation).

## [0.2.0] — 2026-04-22

### Added
- **Elixir language support** — `.ex` / `.exs` extension registration, tree-sitter-elixir dependency, call-node filtering for `def`/`defp`/`defmacro`/`defmodule`/`defprotocol`/`defimpl`. (FASTEDIT-003, commits a956476, 542248f, fa393ea)
- **Direct-swap fast path** — when `replace=X` is set and the snippet is a complete new definition of X with no markers, chunked_merge now performs a pure AST region swap with zero model tokens. Applies to `change_signature`, `extend_literal`, and full-function `wrap_block` patterns. (FASTEDIT-004, commit 2b98be3)
- **`preserve_siblings=True` flag** — new kwarg on `chunked_merge` (and exposed via `fast_edit`/`fast_batch_edit` MCP tools) for surgical edits to a single class member without enumerating siblings. The snippet provides a minimal class shell containing only the changing members; unmentioned siblings are carried over from the original. (FASTEDIT-005, commits acd55f2, 333c651, b621370)
- **`BatchEdit.preserve_siblings` field** — threaded through `batch_chunked_merge` for batch-mode per-edit surgical edits.

### Fixed
- **Preserved-gap indent mismatch** — when marker-mode gap lines need re-indentation (e.g. wrapping in try/except), they were being emitted at original indent. Now the indent delta is computed from marker position and applied. (FASTEDIT-001, commit b063141)
- **Two-marker section ambiguity** — sections containing ≥2 markers had ambiguous insertion positions. Now defer to model with semantic context instead of producing wrong deterministic output. (FASTEDIT-002, commit 4d8989c)
- **Context-anchor indent preservation** — context-anchor lines that match between snippet and original at different indents (e.g., snippet wraps the original body in a closure) were emitted at original indent. Now re-indented by per-anchor delta. (FASTEDIT-006, commit 214ffe8)

### Measured
- All improvements were validated against fasteditBench (156 fixtures across 13 languages). End-state: **tier-1 pass rate 100.0% / tier-2 pass rate 100.0% / 1,977 total model tokens / 120 zero-token (pure AST) cases**. See the fasteditBench repo for methodology.

## [0.1.0]
Initial release.

[0.2.3]: https://github.com/parcadei/fastedit/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/parcadei/fastedit/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/parcadei/fastedit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/parcadei/fastedit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/parcadei/fastedit/releases/tag/v0.1.0
