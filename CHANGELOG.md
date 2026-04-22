# Changelog

All notable changes to FastEdit are documented in this file. Format: [Keep a Changelog](https://keepachangelog.com/).

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

[0.2.0]: https://github.com/parcadei/fastedit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/parcadei/fastedit/releases/tag/v0.1.0
