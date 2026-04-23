# Changelog

All notable changes to FastEdit are documented in this file. Format: [Keep a Changelog](https://keepachangelog.com/).

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
