# Testing Matrix — M1-M4 Cross-Language Coverage

Milestone 4.6 hardening pass. Documents which features are verified
against which languages, which adversarial cases revealed bugs, and
which bugs are fixed in-scope vs flagged for 0.6.1.

## Language × Feature Matrix

| Feature                   | py  | ts  | js  | go  | rust | java | kotlin | ruby | swift | php | c# | cpp | c  |
| ------------------------- | --- | --- | --- | --- | ---- | ---- | ------ | ---- | ----- | --- | -- | --- | -- |
| M1 `fast_rename` (single) | ✅  | ✅  | ✅  | ✅  | ✅   | 🔻   | 🔻     | 🔻   | 🔻    | 🔻  | 🔻 | 🔻  | 🔻 |
| M2 delete-safety (refs)   | ✅  | ✅  | ✅  | ✅  | ✅   | ✅   | ✅     | ✅   | ✅    | ✅  | ✅ | ✅  | ✅ |
| M2 CLI delete end-to-end  | ✅  | —   | —   | —   | —    | —    | —      | —    | —     | —   | —  | —   | —  |
| M3 signature-change       | ✅  | ✅  | —   | ✅  | ✅   | ✅   | ✅     | —    | —     | —   | —  | —   | —  |
| M4 move-to-file (imports) | ✅  | ✅  | ✅  | 🚫  | 🚫   | 🚫   | 🚫     | 🚫   | 🚫    | 🚫  | 🚫 | 🚫  | 🚫 |

Legend:
- ✅ = tested + passes
- 🔻 = gracefully degrades to count=0 (documented below)
- 🚫 = rejected with a helpful error (tested to fail loudly, not silently)
- —  = not tested (out of scope; see note)

## M1 rename limitation: kind="other" filter

`do_rename_ast` and `do_cross_file_rename` intentionally drop tldr refs
with `kind="other"`. tldr emits `kind="other"` for every language
outside its AST-native set (python, typescript, go, rust). On
java/kotlin/ruby the engine therefore returns `count=0` rather than
silently doing a regex-quality rename.

**Workaround:** users on non-AST-native langs should run
`fastedit rename` (regex + tree-sitter skip zones), which is the
pre-M1 path kept for this reason.

**Future work (0.6.1+):** either broaden the kind filter with a
confidence threshold (tldr reports confidence 1.0 on `kind=other` when
the grep hit is exact) or ship per-lang AST renamers.

## Adversarial cases

12 cases shipped; results summarized below.

| # | Case                                          | Outcome            | Disposition                          |
| - | --------------------------------------------- | ------------------ | ------------------------------------ |
| 1 | Unicode identifier `café`                     | Passes (skip-path) | Degrades to count=0; safe            |
| 2 | Keyword-adjacent name `class_`                | Passes             |                                      |
| 3 | Substring collision `get` vs `get_all`        | Passes             | Word-boundary + AST hold             |
| 4 | Decorator-wrapped delete (`@log` + `def foo`) | **Bug found**      | **Fixed in scope** (symbols.py)      |
| 5 | Multi-line signature rename + impact          | Passes             |                                      |
| 6 | Mixed line endings (CRLF + LF)                | Passes             |                                      |
| 7 | UTF-8 BOM prefix                              | **Bug found**      | **xfail → flagged for 0.6.1**        |
| 8 | Very long identifier (500 chars)              | Passes             |                                      |
| 9 | Symbol inside `__main__` block                | Passes             |                                      |
| 10| Module vs local-scope shadow                  | Passes             | Locks current behavior (rename both) |
| 11| Builtin shadow `list`                         | Passes             |                                      |
| 12| Non-ASCII content in strings/comments         | Passes             |                                      |

## Bugs discovered

### Fixed in scope

**Bug 1: decorator lost on delete.** When `delete_symbol` was called on
a `@decorator`-wrapped function, tldr's `tldr structure` output mis-
reported line ranges — the decorated function was missing from
structure output, and the NEXT function inherited its line numbers.
Result: delete removed lines that belonged to the wrong function AND
left the orphaned `@decorator` attached to the surviving next function,
silently changing semantics.

- **Root cause:** upstream tldr bug (tldr 0.1.5).
- **Fix:** `delete_symbol` now uses in-memory `get_ast_map_from_source`
  instead of shelling out to `tldr structure`. Tree-sitter walks the
  `decorated_definition` node, which correctly spans the decorator.
- **Commit:** included in the test hardening commit.
- **File:** `src/fastedit/inference/symbols.py`.
- **Test:** `test_adv_decorator_wrapped_function_delete_removes_decorator`
  in `tests/test_hardening_adversarial.py`.

### Flagged for 0.6.1

**Bug 2: UTF-8 BOM misaligns first-line rename.** When a file has a
UTF-8 BOM, tldr reports column positions computed against the BOM-less
content, but the on-disk bytes include the BOM. The column-to-byte math
in `_apply_refs_to_content` then misaligns on line 1, and the guard
(`if raw[byte_start:byte_end] != old_bytes: continue`) correctly skips
the broken replacement. Net effect: renames inside a BOM file succeed
on every line except the first.

- **Severity:** low. Failure mode is safe (no corruption, just silent
  missed rename on line 1). BOM-prefixed Python files are uncommon.
- **Fix path:** strip BOM before computing line-0 byte offsets, or pass
  the BOM-less content to `_apply_refs_to_content` and offset the
  result.
- **Test:** `test_adv_utf8_bom_prefix` in
  `tests/test_hardening_adversarial.py` — locked in as `xfail(strict=True)`
  so the test FAILS the suite if the bug is silently fixed without
  removing the xfail marker.

## Properties verified

- Rename idempotence (single-file + cross-file) across 8 langs
- Move-then-move-back preserves def set
- Dry-run no-op: mtimes + bytes unchanged (move + rename-all CLI)
- `kind_filter` monotonicity: class-only ⊆ unfiltered
- `--force` monotonicity: accept-set widens, never narrows
- Rename-then-rename-back restores byte-exact content

## Tool-integration compositions verified

1. Move-then-rename: symbol moved + renamed across all callers
2. Force-delete leaves dangling imports visible (opt-in verified)
3. Edit-signature-then-rename: sig change + rename compose
4. Rename-class-then-delete-unrelated: kind_filter safety
5. Move-then-check-source-empty: extraction cleans up

## Test counts (post-M4.6)

| File                                      | Tests | Notes                   |
| ----------------------------------------- | ----- | ----------------------- |
| `test_hardening_cross_language.py`        | 65    | Parametrized (8 langs)  |
| `test_hardening_adversarial.py`           | 15    | 1 xfail (BOM)           |
| `test_hardening_properties_integration.py`| 13    | 5 prop + 5 integ + 3    |
| **Total new**                             | **93**|                         |

Baseline pre-M4.6: 641 passed. Post-M4.6: 725 passed + 1 xfail + 32 skipped.
