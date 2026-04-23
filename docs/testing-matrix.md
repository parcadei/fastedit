# Testing Matrix — M1-M4 Cross-Language Coverage

Milestone 4.7 (final pre-release) outsources per-language work to
`tldr` wherever it already exposes the primitive. Result: 13/13 lang
coverage on M1/M3 and 13/13 common-case on M4 with Rust complex cases
flagged as manual review.

## Language × Feature Matrix

| Feature                   | py  | ts  | js  | go  | rust | java | kotlin | ruby | swift | php | c# | cpp | c  |
| ------------------------- | --- | --- | --- | --- | ---- | ---- | ------ | ---- | ----- | --- | -- | --- | -- |
| M1 `fast_rename` (single) | ✅  | ✅  | ✅  | ✅  | ✅   | ✅   | ✅     | ✅   | ✅    | ✅  | ✅ | ✅  | ✅ |
| M2 delete-safety (refs)   | ✅  | ✅  | ✅  | ✅  | ✅   | ✅   | ✅     | ✅   | ✅    | ✅  | ✅ | ✅  | ✅ |
| M2 CLI delete end-to-end  | ✅  | —   | —   | —   | —    | —    | —      | —    | —     | —   | —  | —   | —  |
| M3 signature-change       | ✅  | ✅  | ✅  | ✅  | ✅   | ✅   | ✅     | ✅   | ✅    | ✅  | ✅ | ✅  | ✅ |
| M4 move-to-file (imports) | ✅  | ✅  | ✅  | ✅  | ✅*  | ✅   | ✅     | ✅   | ✅    | ✅  | ✅ | ✅  | ✅ |

Legend:
- ✅ = tested + passes
- ✅* = common case supported; complex cases (Rust nested use-trees,
  wildcard `use *`, aliased `use foo::X as Y`) are flagged for manual
  review rather than half-handled.
- — = not tested (out of scope; see note)

## M4.7 outsourcing — what changed

Three features previously degraded on non-AST-native languages. Each
is now driven off a tldr primitive instead of per-language AST
walking inside fastedit.

### M1 — confidence-based reference filter

**Before:** `do_rename_ast` filtered tldr's references by
`kind ∈ {call, read, write, import, type, definition}`. On the 9
non-AST-native langs (java, kotlin, ruby, swift, php, c#, cpp, c, and
scala/elixir/lua when present), tldr emits `kind="other"` with
`confidence=1.0` for real code hits and `confidence=0.5` for
string/comment substring hits. The kind allowlist dropped everything,
producing silent count=0 no-op renames.

**After:** filter by confidence (>= 0.9). The `--min-confidence 0.9`
CLI flag we pass to tldr already excludes the 0.5 hits; the
client-side check is belt-and-suspenders. The slice-match safety
guard in `_apply_refs_to_content` still verifies the target bytes
equal `old_name` before any rewrite, so column-math surprises remain
caught.

Coverage: 5/13 → 13/13.

### M3 — outsourced signature detection

**Before:** `_get_signature_text` walked a tree-sitter AST per
language to extract the signature span. Covered 6 langs —
python, ts, go, rust, java, kotlin — and the other 7 (javascript,
ruby, swift, php, c#, cpp, c) silently returned False from
`signature_changed`, disabling the pre-flight impact note on those
langs.

**After:** `tldr structure` returns per-definition `line_start` and
`signature` across all 13 langs. fastedit reads `line_start` and
re-extracts the signature text directly from source — this fixes
tldr's first-line-only truncation on multi-line parameter lists
like:

```python
def multi(
    a,
    b,
):
```

where tldr alone emits `signature='def multi('` but the real
signature spans four lines. The per-family body-opener set (`{`,
`:`, `;`, or bracket-balanced EOL for Ruby's `def foo(a)\n  ...\nend`)
makes the scan language-agnostic.

Hot-path discipline preserved via an in-process pre-check
(`_extract_declaration_block`). It compares the symbol's declaration
span between old and new source (first symbol line extended through
multi-line parens to the body opener). When the blocks are byte-
identical, no subprocess is spawned — body-only edits remain free of
tldr overhead (VAL-M3-002).

One documented trade-off: one-liner functions (whole body on the
same line as the signature, e.g. `int foo(int a) { return a; }`) now
register body-only edits as signature changes, because the extractor
stops at the `{` and differences exist entirely after the stop point
— the block-diff pre-check then fires. Real multi-line code (the
common case) is unaffected.

Coverage: 6/13 → 13/13.

### M4 — per-language import rewriters

**Before:** `move_to_file` supported Python + JS/TS only. The other
10 langs were hard-rejected with a `ValueError` hint to use the
supported set.

**After:** per-family rewriters for java, kotlin, scala, c#, php, go,
rust, swift, ruby, elixir, lua, c, cpp. Families group by import
syntax:

- **Dotted path** (java/kotlin/scala/c#/php): `import pkg.sub.Sym;`
  or `using Pkg.Sub.Cls;` or `use Foo\Bar\Sym;`. Shared rewriter
  with configurable keyword, trailing semicolon, and path separator.
- **Quoted path** (go): `import "./pkg"`. Substring swap inside
  quotes.
- **Rust `use` tree**: simple (`use crate::foo::Bar;`), braced
  multi-import (`use crate::foo::{Bar, Baz};` → split), and braced
  single. Nested / wildcard / renamed forms flagged for manual review.
- **Module-name-only** (swift): `import Foo` — swap module name.
- **File-path require** (ruby, lua, c/cpp): `require_relative "a"`,
  `require "a"`, `#include "a.h"`. tldr can't find these via the
  symbol because the import line names the FILE, not the symbol;
  fastedit scans project files for import-shaped lines matching the
  source file's basename and rewrites them.
- **Elixir alias/import**: `alias Foo.Bar` / `import Foo.Bar`.
  Braced `alias Foo.{Bar, Baz}` flagged.

Module-path derivation is best-effort and convention-based — we
can't see build configs (Cargo.toml, pom.xml, go.mod, .csproj),
so we infer from filesystem layout. Dotted path for JVM langs,
`crate::` for rust (dropping a leading `src/` segment and `mod`
file stem by convention), backslash for php, top-level-dir for
swift modules, PascalCase directory path for elixir. Users on
non-trivial build configs may need to adjust the rewritten
specifier post-move, but the import line *is* rewritten — no more
hard rejection.

Coverage: 3/13 → 13/13 common-case; Rust nested/wildcard/rename flagged.

## Adversarial cases

12 cases shipped at M4.6; results unchanged at M4.7.

| # | Case                                          | Outcome            | Disposition                          |
| - | --------------------------------------------- | ------------------ | ------------------------------------ |
| 1 | Unicode identifier `café`                     | Passes (skip-path) | Degrades to count=0; safe            |
| 2 | Keyword-adjacent name `class_`                | Passes             |                                      |
| 3 | Substring collision `get` vs `get_all`        | Passes             | Word-boundary + AST hold             |
| 4 | Decorator-wrapped delete (`@log` + `def foo`) | Passes             | Fixed at M4.6 (symbols.py)           |
| 5 | Multi-line signature rename + impact          | Passes             |                                      |
| 6 | Mixed line endings (CRLF + LF)                | Passes             |                                      |
| 7 | UTF-8 BOM prefix                              | **xfail**          | Flagged for 0.6.1                    |
| 8 | Very long identifier (500 chars)              | Passes             |                                      |
| 9 | Symbol inside `__main__` block                | Passes             |                                      |
| 10| Module vs local-scope shadow                  | Passes             | Locks current behavior (rename both) |
| 11| Builtin shadow `list`                         | Passes             |                                      |
| 12| Non-ASCII content in strings/comments         | Passes             |                                      |

## Bugs discovered

### Fixed at M4.6

**Bug 1: decorator lost on delete.** When `delete_symbol` was called on
a `@decorator`-wrapped function, tldr's `tldr structure` output mis-
reported line ranges — the decorated function was missing from
structure output, and the NEXT function inherited its line numbers.
Fixed by routing delete through in-memory `get_ast_map_from_source`.
Locked with `test_adv_decorator_wrapped_function_delete_removes_decorator`.

### Flagged for 0.6.1

**Bug 2: UTF-8 BOM misaligns first-line rename.** When a file has a
UTF-8 BOM, tldr reports column positions computed against the BOM-less
content, but the on-disk bytes include the BOM. The column-to-byte
math then misaligns on line 1; the safety guard correctly skips the
broken replacement rather than corrupting output.

- **Severity:** low. Failure mode is safe (no corruption, just silent
  missed rename on line 1). BOM-prefixed source files are uncommon.
- **Fix path:** strip BOM before computing line-0 byte offsets, or
  pass the BOM-less content to `_apply_refs_to_content` and offset
  the result.
- **Test:** `test_adv_utf8_bom_prefix` — locked as `xfail(strict=True)`
  so the test FAILS the suite if the bug is silently fixed without
  removing the xfail marker.

## Properties verified

- Rename idempotence (single-file + cross-file) across 8 langs
- Move-then-move-back preserves def set
- Dry-run no-op: mtimes + bytes unchanged (move + rename-all CLI)
- `kind_filter` monotonicity: class-only ⊆ unfiltered
- `--force` monotonicity: accept-set widens, never narrows
- Rename-then-rename-back restores byte-exact content
- M3 hot-path (VAL-M3-002): body-only edits never spawn tldr

## Tool-integration compositions verified

1. Move-then-rename: symbol moved + renamed across all callers
2. Force-delete leaves dangling imports visible (opt-in verified)
3. Edit-signature-then-rename: sig change + rename compose
4. Rename-class-then-delete-unrelated: kind_filter safety
5. Move-then-check-source-empty: extraction cleans up

## Test counts (post-M4.7)

Baseline pre-M4.6: 641 passed.
Post-M4.6: 725 passed + 1 xfail + 32 skipped.
Post-M4.7: 749 passed + 1 xfail + 29 skipped.

| File                                      | Tests | Notes                                    |
| ----------------------------------------- | ----- | ---------------------------------------- |
| `test_hardening_cross_language.py`        | 81    | 13-lang parametrize on M1/M3; 13-lang M4 |
| `test_hardening_adversarial.py`           | 15    | 1 xfail (BOM, flagged for 0.6.1)         |
| `test_hardening_properties_integration.py`| 13    | 5 prop + 5 integ + 3                     |
