"""Seeded corpus generator + INDEPENDENT golden oracle (Step C1, plan Phase C).

Mission: the 100MB-class corpus generator, the independent expected-output
oracle, and the backward-reconstruction data — INFRASTRUCTURE ONLY. The
100MB runs themselves are C2/C3 (``llm``/``stress`` tier) and are not
shipped here; the default tier exercises this module through small/medium
sizes and the committed ``tests/golden/big/`` cases.

THE ORACLE IS INDEPENDENT OF FASTEDIT (the B3 doctrine, tests/golden/
_generate.py): ``apply_op_oracle`` computes expected bytes by explicit
line-splice arithmetic on line indices the GENERATOR recorded. It never
imports fastedit, never runs the pipeline, and never consults a parser.

THE ORACLE CONTRACT (loud, per the plan's golden-infrastructure doctrine):

1. ``generate_big_source(language, target_bytes, ...)`` builds a corpus of
   small TOP-LEVEL functions/records — every symbol at most
   :data:`MAX_SYMBOL_LINES` lines (the chunk locator's parent-snap cap,
   ``chunk_locator._find_enclosing_parent``) — with unique symbol names and
   per-line content that embeds the owning symbol's name (mostly-unique
   lines keep the pipeline validator's LCS spans trivial), the requested
   line-ending convention (LF / CRLF, with or without a final EOL) and an
   optional sprinkle of Chinese comments (multibyte safety everywhere).
   It returns a :class:`CorpusSource` — a ``str`` subclass whose
   ``.manifest`` is the symbol -> line-span table the generator recorded
   WHILE assembling the text.
2. ``make_ops(source, ...)`` resolves every requested op's target through
   THAT manifest (``manifest.span(name)`` — never by parsing) and records
   everything the oracle needs: the resolved span, the payload in the
   file's own EOL convention, and the inverse data (the recorded original
   span bytes, the blank-separator flags, the anchor's raw last line).
3. ``apply_op_oracle(original_text, op)`` performs pure line arithmetic on
   the recorded indices and CROSS-CHECKS every recorded value against the
   text it is handed (B3's ``_verify_op`` doctrine: the oracle is only as
   honest as these checks — a drifted manifest fails loudly instead of
   pinning wrong behavior).
4. ``inverse_op(op)`` returns a :class:`LineOp` — concrete line-splice
   data for the EDITED text — so
   ``apply_line_op(apply_op_oracle(original, op), inverse_op(op))``
   regenerates the ORIGINAL byte-for-byte (req. 3's backward
   reconstruction). ``apply_op_sequence`` extends this to several ops:
   it applies them in order (shifting recorded indices by the accumulated
   line delta) and materializes inverse LineOps that regenerate the
   original when applied in reverse.
5. The ONE fastedit behavior the oracle mirrors on purpose is the
   pipeline's central EOL funnel (``_normalize_merged_eol``'s DOCUMENTED
   contract): a uniform original's ending convention is preserved and the
   trailing-newline state is taken from the ORIGINAL (see
   :func:`_enforce_trailing_state`). Everything else is plain splice
   arithmetic. On LF + final-EOL corpora the funnel is a no-op and the
   oracle reduces exactly to the B3 splice semantics.

Op kinds: ``insert_after_symbol`` (the pipeline's zero-token fast path),
``replace_symbol_body`` (the deterministic direct swap), ``delete_symbol``
(the AST delete) and ``append_at_eof`` (insert anchored on the LAST
symbol — the generator guarantees the corpus ends with a function).

Step C3 added ADVERSARIAL RECIPES (``recipe=`` kwarg): the seams are where
corruption lives — chunks are cut on ORIGINAL line indexing and spliced in
reverse (``chunked_merge``'s assembly loop) — so the C3 corpora place the
danger AT the seams:

* ``"boilerplate"`` — every function shares one byte-identical body
  (docstring, statement lines, ``return``): the duplicated-line trap. The
  only per-symbol line is the signature, so a wrong-occurrence edit is
  invisible to any content-level check and only a byte-exact full-file
  golden at the manifest-recorded span can catch it.
* ``"deep"`` — every :data:`DEEP_EVERY`-th symbol is a >100-line
  multi-block function (``for`` blocks containing ``if`` blocks) whose
  sub-block spans are RECORDED in :attr:`SymbolSpan.blocks`; a
  ``replace=`` + marker snippet on one drives the chunk locator's
  ``_narrow_large_node`` windowing path (``_MAX_BLOCK_LINES=100``) and the
  expected narrow cut is computable from the recorded blocks.
* ``"prefix"`` — every :data:`PREFIX_EVERY`-th function index emits a
  ``<base>``, ``<base>_data``, ``<base>_data_v2`` family: symbols whose
  names are prefixes of each other, targeting resolution ambiguity at
  scale.
* ``"cjk_seams"`` — every function carries Chinese text ADJACENT TO the
  chunk boundaries: a CJK docstring (python) / comment (brace languages)
  as the first body line and a CJK trailing comment on the closing
  ``return`` line. Built with ``eol=EOL_CRLF`` by the C3 suite so the
  multibyte × line-ending × seam interaction is stressed at once.
* ``"text"`` — the PLAIN-TEXT dialect (Step D2, requires
  ``language="text"``): a natural shift log whose paragraphs are
  timestamped log entries. The entry line is each paragraph's one
  file-wide-unique line — the D2 anchor matcher anchors windows on it —
  and the prose texture is what the real model measurably merges
  faithfully (tests/test_real_llm_text_chunks.py). Body sentences are
  drawn per :data:`_TEXT_TEMPLATE_STRIDE` blocks so any window-sized run
  of consecutive paragraphs is repeat-free BY CONSTRUCTION — the real
  model lossily compresses a chunk that repeats a line within itself and
  the edit can never converge (measured: free ``rng.choice`` draws put
  4-10 repeated lines in every ~40-line window and every real merge was
  rejected; the stride-block draw removes them).

All non-plain recipes also emit a two-line header import block (recorded
as :attr:`CorpusManifest.import_lines`) so a snippet adding one more
import drives the locator's import-region + code-region split — the ONE
shape that hands ``chunked_merge`` several chunk regions to splice in
reverse order.

``recipe="plain"`` (the default) is BYTE-COMPATIBLE with the C1 output:
the RNG key deliberately excludes the plain recipe so the committed
``tests/golden/big/`` fixtures regenerate identically.

Documented constraint: on a no-final-EOL corpus the delete target is never
the file's LAST symbol. The funnel would strip the preceding line's
terminator once the span (which contains the file's unterminated last
line) is removed, changing line structure beyond the splice and making the
insert-inverse unable to restore the bytes. ``make_ops`` enforces this by
never choosing the last symbol for insert/replace/delete (``append_at_eof``
owns it); ``_make_delete_op`` asserts it.

Schema decision (task 2): the corpus goldens under ``tests/golden/big/``
use a PARALLEL manifest schema (:data:`CORPUS_GOLDEN_SCHEMA`) instead of
extending B3's hand-authored ``tests/golden/<lang>/`` schema.
Justification: B3 manifests describe hand-written fixtures whose line
indices are verified against authored text; corpus manifests are generator
OUTPUT with different invariants (byte-exact seed regeneration, an
eol/final-eol/indent/sprinkle recipe, recorded inverse payloads). Both
share the conventions that matter for tooling: ``manifest.json`` per case,
``original.<ext>`` + ``expected_<op>_<symbol>.<ext>`` files, ``oracle``
line-index blocks, and ``expected`` file references — so one runner
pattern serves both, and neither schema loosens the other's validation.

Memory discipline: the generator accumulates symbol chunks in a list and
joins ONCE — no repeated string concatenation, no intermediate full-size
copies beyond the final join. Measured 100MB build (python, 246 542
symbols, ``uv run python tests/corpus.py --bench``): ~0.9 s wall time
(~113 MiB/s); peak process RSS ~395-414 MiB — approximately 4x the corpus
size (chunk list + joined copy are both alive at the join ≈ 2.1x, the rest
is CPython allocator/arena overhead from the ~250k small chunks and the
millions of short-lived render strings; the chunk list is released when
the generator returns). tracemalloc reports a ~3.6x traced peak but adds
~10x wall-time overhead tracing the small allocations. In other words: a
100MB corpus is materialized in about a second and comfortably under a
gigabyte of RAM — build-time cost is never the C2/C3 bottleneck.
Measure it: ``uv run python tests/corpus.py --bench``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

__all__ = [
    "BOILERPLATE_MONSTER_LINES",
    "CORPUS_GOLDEN_SCHEMA",
    "DEEP_BLOCK_COUNT",
    "DEEP_EVERY",
    "DEEP_INNER_STMTS",
    "DEEP_OUTER_STMTS",
    "EOL_CRLF",
    "EOL_LF",
    "GOLDEN_BIG_CASES",
    "MAX_SYMBOL_LINES",
    "OP_KINDS",
    "PREFIX_EVERY",
    "RECIPES",
    "RECIPE_BOILERPLATE",
    "RECIPE_CJK_SEAMS",
    "RECIPE_DEEP",
    "RECIPE_PLAIN",
    "RECIPE_PREFIX",
    "RECIPE_TEXT",
    "CorpusCase",
    "CorpusManifest",
    "CorpusOp",
    "CorpusSource",
    "LineOp",
    "SymbolSpan",
    "apply_line_op",
    "apply_op_oracle",
    "apply_op_sequence",
    "build_corpus_case",
    "case_manifest",
    "expected_filename",
    "generate_big_source",
    "inverse_op",
    "line_op_from_manifest",
    "line_op_to_manifest",
    "make_ops",
    "op_from_manifest",
    "op_line_delta",
    "op_to_manifest",
    "verify_recipe_structures",
    "verify_symbol_spans",
    "write_golden_big_cases",
]

# The chunk locator refuses to snap method edits to parents larger than
# this (chunk_locator._find_enclosing_parent) — the corpus keeps every
# top-level symbol at or below the cap so parent resolution stays cheap
# and method-granular.
MAX_SYMBOL_LINES = 150

EOL_LF = "\n"
EOL_CRLF = "\r\n"

OP_KINDS: tuple[str, ...] = (
    "insert_after_symbol",
    "replace_symbol_body",
    "delete_symbol",
    "append_at_eof",
)

CORPUS_GOLDEN_SCHEMA = "fastedit-corpus-golden/1"

# ---------------------------------------------------------------------------
# Adversarial recipes (Step C3) — declarative layout constants
# ---------------------------------------------------------------------------

RECIPE_PLAIN = "plain"
RECIPE_BOILERPLATE = "boilerplate"
RECIPE_DEEP = "deep"
RECIPE_PREFIX = "prefix"
RECIPE_CJK_SEAMS = "cjk_seams"
RECIPE_TEXT = "text"

RECIPES: tuple[str, ...] = (
    RECIPE_PLAIN, RECIPE_BOILERPLATE, RECIPE_DEEP, RECIPE_PREFIX,
    RECIPE_CJK_SEAMS, RECIPE_TEXT,
)

BOILERPLATE_MONSTER_LINES = 96
"""Shared-pool body lines for a boilerplate ``index % 97`` monster symbol.

96 distinct assignments + signature + shared docstring + shared ``return``
= 99 lines (python; 100 with the brace languages' declaration and closer)
— big enough to be a chunk seam neighbour (>= 90 lines) while staying
under the :data:`MAX_SYMBOL_LINES` parent-snap cap."""

DEEP_EVERY = 23
"""Every 23rd non-record function index renders as a deep-nested symbol."""

DEEP_BLOCK_COUNT = 6
"""Outer ``for`` blocks per deep symbol (siblings, never nested in each
other — the narrow's enclosing-block walk must have exactly one candidate
containing the snippet window for the cut to be computable). The blocks are
deliberately IDENTICAL to one another: the duplicated content makes the
deterministic editor's anchor binding a B24 tie (it declines, so the real
model runs) while the locator's sliding window still resolves — its first
best-scoring offset is the FIRST block, which makes the expected narrow cut
computable from the manifest."""

DEEP_INNER_STMTS = 2
"""Statements inside each deep symbol's inner ``if`` block."""

DEEP_OUTER_STMTS = 16
"""Statements per deep block after the inner ``if``. Block size:
1 opener + 1 inner opener + 2 inner statements + 16 outer statements
(+ 1 inner closer + 1 block closer on brace languages) = 20/22 lines —
above the 20-line adequate floor of ``_find_enclosing_block`` (and the
only enclosing candidate for the snippet window, so the cut stays
computable) — and six blocks put the symbol at 128-133 lines, safely above
the locator's ``_MAX_BLOCK_LINES = 100`` narrow trigger and under
:data:`MAX_SYMBOL_LINES`. The blocks are deliberately small: the narrowed
chunk's wrap is a model-sized edit (the C3 probe measured the real model
lossily dropping the tail of a 60-line mid-block wrap on the brace
languages, while a ~12-line wrap converges on every language)."""

PREFIX_EVERY = 11
"""Every 11th non-record function index emits a ``<base>`` /
``<base>_data`` / ``<base>_data_v2`` prefix family."""


# ---------------------------------------------------------------------------
# Recorded data structures — the manifest and the op records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolSpan:
    """One symbol's recorded line span (1-indexed, inclusive on both ends).

    Emitted by the generator while it assembles the text; the oracle
    resolves every op target through these spans and never parses code.
    (``slots=True``: a 100MB corpus carries ~250k of these.)
    """

    name: str
    kind: str  # "function" | "record"
    start_line: int
    end_line: int
    blocks: tuple[SymbolSpan, ...] = ()
    """Sub-block spans RECORDED for deep-recipe symbols (kind ``"block"``):
    the outer ``for`` blocks whose spans make the chunk locator's
    ``_narrow_large_node`` cut computable from the manifest alone. Empty
    for every other symbol."""

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass(frozen=True)
class CorpusManifest:
    """The symbol -> line-span table the generator emits with the text."""

    language: str
    ext: str
    eol: str
    final_eol: bool
    indent: str
    sprinkle_cjk: bool
    seed: int | str
    target_bytes: int
    header_line_count: int
    sep_blanks: int
    symbols: tuple[SymbolSpan, ...]
    recipe: str = RECIPE_PLAIN
    import_lines: tuple[int, int] | None = None
    """The (first, last) 1-indexed lines of the header import block, recorded
    while assembling; ``None`` for the plain recipe. A snippet that adds one
    more import makes the chunk locator split the edit into an import
    region (exactly this span) plus the code region — the expected split is
    computable from this recording."""

    def span(self, name: str) -> SymbolSpan:
        """Resolve a symbol name against the recorded spans (fail loud)."""
        for sym in self.symbols:
            if sym.name == name:
                return sym
        raise KeyError(
            f"symbol {name!r} is not in the corpus manifest "
            f"({self.language}, {self.symbol_count} symbols)"
        )

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def max_symbol_lines(self) -> int:
        return max(sym.line_count for sym in self.symbols)

    @property
    def last_symbol(self) -> SymbolSpan:
        return self.symbols[-1]


class CorpusSource(str):
    """The generated source text plus the manifest recorded for it.

    A ``str`` subclass so the text flows through every str API unchanged;
    ``.manifest`` rides along as the generator's symbol -> line-span table.
    """

    __slots__ = ("manifest",)

    def __new__(cls, text: str, manifest: CorpusManifest) -> Self:
        obj = super().__new__(cls, text)
        obj.manifest = manifest
        return obj


@dataclass(frozen=True)
class CorpusOp:
    """One corpus edit — data only, resolved through the generator's manifest.

    The resolved span fields are recorded by ``make_ops`` from the
    manifest; the oracle re-derives the same values from the text it is
    handed and asserts equality (drift fails loudly). The ``original_*``
    fields are the recorded INVERSE data: the exact bytes the op removed
    or overwrote, so backward reconstruction never has to re-derive them.
    """

    kind: str  # one of OP_KINDS
    symbol: str
    language: str
    ext: str
    eol: str
    final_eol: bool

    # --- resolved symbol-position data (insert/append) ---
    anchor_start_line: int | None = None
    anchor_end_line: int | None = None  # splice AFTER this line
    anchor_last_line: str | None = None  # raw keepends form of that line
    anchor_is_last_line: bool = False

    # --- resolved symbol-position data (replace/delete) ---
    start_line: int | None = None
    end_line: int | None = None
    consumed_end_line: int | None = None  # delete: last line removed (incl. blanks)

    # --- payload (the new content, in the file's own EOL convention) ---
    new_text: str | None = None

    # --- recorded inverse data / cross-check values ---
    original_span_text: str | None = None
    sep_before: bool = False
    sep_after: bool = False
    snippet_line_count: int = 0
    note: str = ""


@dataclass(frozen=True)
class LineOp:
    """A concrete line-splice instruction — data, no symbols, no parsing.

    Line numbers are 1-indexed and INCLUSIVE; ``after_line`` counts the
    lines BEFORE the splice point (0 = beginning of file). ``text`` carries
    the EXACT bytes to insert or write over the span — the inverse path
    restores recorded bytes verbatim and never re-derives endings.
    """

    kind: str  # "insert" | "replace" | "delete"
    start_line: int = 0
    end_line: int = 0
    after_line: int = 0
    text: str = ""


@dataclass(frozen=True)
class CorpusCase:
    """A generated corpus plus its ops — one C2-ready test case."""

    language: str
    ext: str
    eol: str
    final_eol: bool
    indent: str
    sprinkle_cjk: bool
    seed: int | str
    target_bytes: int
    source: CorpusSource
    ops: tuple[CorpusOp, ...]
    recipe: str = RECIPE_PLAIN

    def build_kwargs(self) -> dict:
        """The exact ``build_corpus_case`` kwargs that regenerate this case."""
        return {
            "language": self.language,
            "target_bytes": self.target_bytes,
            "eol": self.eol,
            "final_eol": self.final_eol,
            "indent": self.indent,
            "seed": self.seed,
            "sprinkle_cjk": self.sprinkle_cjk,
            "recipe": self.recipe,
        }


# ---------------------------------------------------------------------------
# Declarative per-language rendering table (no per-language branches in the
# generator core — the variance lives here, per CLAUDE.md's no-hardcoding rule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolRequest:
    """Everything one symbol's renderer needs."""

    name: str
    steps: int = 4  # body statement lines (functions); body-block repetitions
    # for the boilerplate renderer
    fields: int = 3  # field lines (records)
    indent: str = "    "
    eol: str = EOL_LF
    variant: str = "plain"  # "plain" | "edited" (replace payloads)
    cjk_step: int | None = None  # when set, emit a Chinese comment line
    comment_prefix: str = "#"


@dataclass(frozen=True)
class LanguageSpec:
    """One language's corpus dialect: header, renderers, verification shapes.

    The ``*_lines`` callables beyond ``function_lines``/``record_lines``
    render the C3 adversarial recipes; the generator core stays
    recipe-agnostic and dispatches through this table (CLAUDE.md's
    no-hardcoding rule — the variance lives in the spec, not in branches).
    """

    language: str
    ext: str
    comment_prefix: str
    indent: str  # default body indentation (top-level symbols are column 0)
    sep_blanks: int  # blank lines between top-level symbols
    header: Callable[[str, str], list[str]]  # (seed, eol) -> terminated lines
    function_lines: Callable[[SymbolRequest], list[str]]
    record_lines: Callable[[SymbolRequest], list[str]]
    signature_re: str  # format template; {name} is the (escaped) symbol name
    closer: str | None  # exact last line of a top-level symbol; None = python rule
    fn_name: Callable[[int], str]
    rec_name: Callable[[int], str]
    op_fn_name: Callable[[str, str], str]
    imports: tuple[str, ...] = ()
    """The header import lines (no EOLs) a non-plain recipe emits after the
    language's own header — the recorded :attr:`CorpusManifest.import_lines`
    region the C3 multi-region case adds one more import to."""
    boilerplate_lines: Callable[[SymbolRequest], list[str]] | None = None
    deep_lines: (
        Callable[[SymbolRequest], tuple[list[str], list[tuple[int, int]]]]
        | None
    ) = None
    cjk_function_lines: Callable[[SymbolRequest], list[str]] | None = None


def _cjk_phrase(name: str, step: int) -> str:
    """A Chinese comment phrase — multibyte safety everywhere (C1)."""
    return f"中文注释：{name} 第 {step} 行 — 快速编辑语料，多字节安全 {step:04d}"


def _cjk_line(req: SymbolRequest) -> str:
    return f"{req.indent}{req.comment_prefix} {_cjk_phrase(req.name, req.cjk_step)}{req.eol}"


def _py_header(seed: str, eol: str) -> list[str]:
    return [f'"""Deterministic fastedit corpus (tests/corpus.py, seed {seed})."""{eol}']


def _py_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    edited = req.variant == "edited"
    lines = [f"def {n}(value_a, value_b):{eol}"]
    lines.append(f'{ind}"""{n}: deterministic corpus function ({req.variant})."""{eol}')
    if req.cjk_step is not None:
        lines.append(_cjk_line(req))
    for i in range(1, req.steps + 1):
        if edited:
            if i == 1:
                lines.append(
                    f"{ind}acc_{n} = value_a * {i} + 7  # {n} edited step {i}{eol}",
                )
            else:
                lines.append(f"{ind}acc_{n} += {i * 3}  # {n} edited step {i}{eol}")
        elif i == 1:
            lines.append(f"{ind}acc_{n} = value_a + value_b + {i}  # {n} step {i}{eol}")
        else:
            lines.append(f"{ind}acc_{n} += {i}  # {n} step {i}{eol}")
    if edited:
        lines.append(f"{ind}return acc_{n} + 900001  # {n} edited result{eol}")
    else:
        lines.append(f"{ind}return acc_{n}  # {n} result{eol}")
    return lines


def _py_record_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"class {n}:{eol}"]
    lines.append(f'{ind}"""{n}: deterministic corpus record."""{eol}')
    for j in range(1, req.fields + 1):
        lines.append(f"{ind}field_{chr(96 + j)}_{n} = {j * 11}{eol}")
    return lines


def _rs_header(seed: str, eol: str) -> list[str]:
    return [f"// Deterministic fastedit corpus (tests/corpus.py, seed {seed}).{eol}"]


def _rs_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    edited = req.variant == "edited"
    lines = [f"fn {n}(value_a: i32, value_b: i32) -> i32 {{{eol}"]
    if req.cjk_step is not None:
        lines.append(_cjk_line(req))
    for i in range(1, req.steps + 1):
        if edited:
            if i == 1:
                lines.append(
                    f"{ind}let mut acc_{n} = value_a * {i} + 7; // {n} edited step {i}{eol}",
                )
            else:
                lines.append(f"{ind}acc_{n} += {i * 3}; // {n} edited step {i}{eol}")
        elif i == 1:
            lines.append(f"{ind}let mut acc_{n} = value_a + value_b + {i}; // {n} step {i}{eol}")
        else:
            lines.append(f"{ind}acc_{n} += {i}; // {n} step {i}{eol}")
    if edited:
        lines.append(f"{ind}acc_{n} + 900001{eol}")
    else:
        lines.append(f"{ind}acc_{n}{eol}")
    lines.append(f"}}{eol}")
    return lines


def _rs_record_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"struct {n} {{{eol}"]
    for j in range(1, req.fields + 1):
        lines.append(f"{ind}field_{chr(96 + j)}_{n}: i32,{eol}")
    lines.append(f"}}{eol}")
    return lines


def _go_header(seed: str, eol: str) -> list[str]:
    return [
        f"// Deterministic fastedit corpus (tests/corpus.py, seed {seed}).{eol}",
        f"package corpus{eol}",
    ]


def _go_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    edited = req.variant == "edited"
    lines = [f"func {n}(valueA int, valueB int) int {{{eol}"]
    if req.cjk_step is not None:
        lines.append(_cjk_line(req))
    for i in range(1, req.steps + 1):
        if edited:
            if i == 1:
                lines.append(f"{ind}acc{n} := valueA*{i} + 7 // {n} edited step {i}{eol}")
            else:
                lines.append(f"{ind}acc{n} += {i * 3} // {n} edited step {i}{eol}")
        elif i == 1:
            lines.append(f"{ind}acc{n} := valueA + valueB + {i} // {n} step {i}{eol}")
        else:
            lines.append(f"{ind}acc{n} += {i} // {n} step {i}{eol}")
    if edited:
        lines.append(f"{ind}return acc{n} + 900001{eol}")
    else:
        lines.append(f"{ind}return acc{n}{eol}")
    lines.append(f"}}{eol}")
    return lines


def _go_record_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"type {n} struct {{{eol}"]
    for j in range(1, req.fields + 1):
        lines.append(f"{ind}Field{chr(96 + j).upper()}{n} int{eol}")
    lines.append(f"}}{eol}")
    return lines


def _ts_header(seed: str, eol: str) -> list[str]:
    return [f"// Deterministic fastedit corpus (tests/corpus.py, seed {seed}).{eol}"]


def _ts_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    edited = req.variant == "edited"
    lines = [f"export function {n}(valueA: number, valueB: number): number {{{eol}"]
    if req.cjk_step is not None:
        lines.append(_cjk_line(req))
    for i in range(1, req.steps + 1):
        if edited:
            if i == 1:
                lines.append(f"{ind}let acc{n} = valueA * {i} + 7; // {n} edited step {i}{eol}")
            else:
                lines.append(f"{ind}acc{n} += {i * 3}; // {n} edited step {i}{eol}")
        elif i == 1:
            lines.append(f"{ind}let acc{n} = valueA + valueB + {i}; // {n} step {i}{eol}")
        else:
            lines.append(f"{ind}acc{n} += {i}; // {n} step {i}{eol}")
    if edited:
        lines.append(f"{ind}return acc{n} + 900001;{eol}")
    else:
        lines.append(f"{ind}return acc{n};{eol}")
    lines.append(f"}}{eol}")
    return lines


def _ts_record_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"export interface {n} {{{eol}"]
    for j in range(1, req.fields + 1):
        lines.append(f"{ind}field{chr(96 + j).upper()}{n}: number;{eol}")
    lines.append(f"}}{eol}")
    return lines


# ---------------------------------------------------------------------------
# C3 recipe renderers — duplicated-line trap, deep nesting, CJK at the seams
# ---------------------------------------------------------------------------

_BOILERPLATE_DOCSTRING = (
    "Boilerplate corpus function: identical body across all symbols."
)

# The shared body-line pool: every boilerplate function's body is a PREFIX
# of one shared assignment sequence (regular functions take the first
# ``steps`` lines, the every-97th monster takes all
# :data:`BOILERPLATE_MONSTER_LINES` of them). The lines are DISTINCT WITHIN
# a body — a chunk the real model must re-emit must never repeat itself (a
# repetitive chunk makes the model lossily compress it and the validator
# rightly rejects every attempt — measured in the C3 probe) — while staying
# IDENTICAL ACROSS functions, which is the duplicated-line trap under test:
# body bytes cannot identify the symbol they belong to.
BOILERPLATE_MONSTER_LINES = 96


def _py_boilerplate_lines(req: SymbolRequest) -> list[str]:
    ind, eol = req.indent, req.eol
    lines = [f"def {req.name}(value_a, value_b):{eol}"]
    lines.append(f'{ind}"""{_BOILERPLATE_DOCSTRING}"""{eol}')
    lines += [f"{ind}acc = value_a + {k}{eol}" for k in range(1, req.steps + 1)]
    lines.append(f"{ind}return acc{eol}")
    return lines


def _rs_boilerplate_lines(req: SymbolRequest) -> list[str]:
    ind, eol = req.indent, req.eol
    lines = [f"fn {req.name}(value_a: i32, value_b: i32) -> i32 {{{eol}"]
    lines.append(f"{ind}let mut acc = value_a;{eol}")
    lines += [f"{ind}acc = value_a + {k};{eol}" for k in range(1, req.steps + 1)]
    lines.append(f"{ind}acc{eol}")
    lines.append(f"}}{eol}")
    return lines


def _go_boilerplate_lines(req: SymbolRequest) -> list[str]:
    ind, eol = req.indent, req.eol
    lines = [f"func {req.name}(valueA int, valueB int) int {{{eol}"]
    lines.append(f"{ind}acc := valueA{eol}")
    lines += [f"{ind}acc = valueA + {k}{eol}" for k in range(1, req.steps + 1)]
    lines.append(f"{ind}return acc{eol}")
    lines.append(f"}}{eol}")
    return lines


def _ts_boilerplate_lines(req: SymbolRequest) -> list[str]:
    ind, eol = req.indent, req.eol
    lines = [
        (
            f"export function {req.name}(valueA: number, valueB: number): "
            f"number {{{eol}"
        ),
    ]
    lines.append(f"{ind}let acc = valueA;{eol}")
    lines += [f"{ind}acc = valueA + {k};{eol}" for k in range(1, req.steps + 1)]
    lines.append(f"{ind}return acc;{eol}")
    lines.append(f"}}{eol}")
    return lines


def _deep_stmt_values(count: int) -> list[int]:
    """Statement values for deep blocks: a CONSTANT (``2``) per line.

    Two measured model behaviors drove this (C3 seams stress):

    * ARBITRARY values (pi digits) defeat the real model's COPY: re-emitting
      ``acc += 3 // ... seam 07``-style lines, it hallucinated neighbouring
      values (``seam 07`` emitted with ``8``) and the validator rightly
      rejected every attempt.
    * SEQUENTIAL values (``value == index``) invite sequence bleed: the go
      model stuttered a bare ``08`` fragment after the chunk's closer.

    A CONSTANT value gives the copy nothing to mis-read and nothing to
    continue: the statement index embedded in each line (``seam {s:02d}``)
    keeps every line distinct within its block (the lossy-compression trap)
    while blocks stay IDENTICAL to one another (the B24 anchor tie that
    keeps the deterministic editor off the narrowed deep edit). The deep
    recipe's purpose — deep nesting driving the chunk locator's
    ``_narrow_large_node`` windowing — does not depend on the values.
    """
    return [2] * count


def _py_deep_lines(req: SymbolRequest) -> tuple[list[str], list[tuple[int, int]]]:
    ind, eol, n = req.indent, req.eol, req.name
    values = _deep_stmt_values(DEEP_OUTER_STMTS)
    lines = [f"def {n}(value_a, value_b):{eol}"]
    lines.append(f'{ind}"""{n}: deep-nesting corpus symbol (multi-block)."""{eol}')
    blocks: list[tuple[int, int]] = []
    for _b in range(1, DEEP_BLOCK_COUNT + 1):
        start = len(lines)
        lines.append(f"{ind}for _outer in range(2):{eol}")  # block opener
        lines.append(f"{ind * 2}if value_a > 1:{eol}")  # inner opener
        for s in range(1, DEEP_INNER_STMTS + 1):
            lines.append(f"{ind * 3}acc_{n}_i{s} = value_a * {s}{eol}")
        for s in range(1, DEEP_OUTER_STMTS + 1):
            lines.append(
                f"{ind * 2}acc_{n} += {values[s - 1]}  # {n} seam {s:02d}{eol}",
            )
        blocks.append((start, len(lines) - 1))
    return lines, blocks


def _rs_deep_lines(req: SymbolRequest) -> tuple[list[str], list[tuple[int, int]]]:
    ind, eol, n = req.indent, req.eol, req.name
    values = _deep_stmt_values(DEEP_OUTER_STMTS)
    lines = [f"fn {n}(value_a: i32, value_b: i32) -> i32 {{{eol}"]
    blocks: list[tuple[int, int]] = []
    for _b in range(1, DEEP_BLOCK_COUNT + 1):
        start = len(lines)
        lines.append(f"{ind}for _i in 0..2 {{{eol}")
        lines.append(f"{ind * 2}if value_a > 1 {{{eol}")
        for s in range(1, DEEP_INNER_STMTS + 1):
            lines.append(f"{ind * 3}acc += {s}; // {n} inner {s}{eol}")
        lines.append(f"{ind * 2}}}{eol}")  # inner closer: ends the if-suite
        for s in range(1, DEEP_OUTER_STMTS + 1):
            lines.append(
                f"{ind * 2}acc += {values[s - 1]}; // {n} seam {s:02d}{eol}",
            )
        lines.append(f"{ind}}}{eol}")  # block closer
        blocks.append((start, len(lines) - 1))
    lines.append(f"{ind}acc{eol}")
    lines.append(f"}}{eol}")
    return lines, blocks


def _go_deep_lines(req: SymbolRequest) -> tuple[list[str], list[tuple[int, int]]]:
    ind, eol, n = req.indent, req.eol, req.name
    values = _deep_stmt_values(DEEP_OUTER_STMTS)
    lines = [f"func {n}(valueA int, valueB int) int {{{eol}"]
    blocks: list[tuple[int, int]] = []
    for _b in range(1, DEEP_BLOCK_COUNT + 1):
        start = len(lines)
        lines.append(f"{ind}for i := 0; i < 2; i++ {{{eol}")
        lines.append(f"{ind * 2}if valueA > 1 {{{eol}")
        for s in range(1, DEEP_INNER_STMTS + 1):
            lines.append(f"{ind * 3}acc += {s} // {n} inner {s}{eol}")
        lines.append(f"{ind * 2}}}{eol}")  # inner closer: ends the if-suite
        for s in range(1, DEEP_OUTER_STMTS + 1):
            lines.append(
                f"{ind * 2}acc += {values[s - 1]} // {n} seam {s:02d}{eol}",
            )
        lines.append(f"{ind}}}{eol}")  # block closer
        blocks.append((start, len(lines) - 1))
    lines.append(f"{ind}return acc{eol}")
    lines.append(f"}}{eol}")
    return lines, blocks


def _ts_deep_lines(req: SymbolRequest) -> tuple[list[str], list[tuple[int, int]]]:
    ind, eol, n = req.indent, req.eol, req.name
    values = _deep_stmt_values(DEEP_OUTER_STMTS)
    lines = [
        f"export function {n}(valueA: number, valueB: number): number {{{eol}",
    ]
    blocks: list[tuple[int, int]] = []
    for _b in range(1, DEEP_BLOCK_COUNT + 1):
        start = len(lines)
        lines.append(f"{ind}for (let i = 0; i < 2; i++) {{{eol}")
        lines.append(f"{ind * 2}if (valueA > 1) {{{eol}")
        for s in range(1, DEEP_INNER_STMTS + 1):
            lines.append(f"{ind * 3}acc += {s}; // {n} inner {s}{eol}")
        lines.append(f"{ind * 2}}}{eol}")  # inner closer: ends the if-suite
        for s in range(1, DEEP_OUTER_STMTS + 1):
            lines.append(
                f"{ind * 2}acc += {values[s - 1]}; // {n} seam {s:02d}{eol}",
            )
        lines.append(f"{ind}}}{eol}")  # block closer
        blocks.append((start, len(lines) - 1))
    lines.append(f"{ind}return acc;{eol}")
    lines.append(f"}}{eol}")
    return lines, blocks


def _py_cjk_function_lines(req: SymbolRequest) -> list[str]:
    """Plain body shape with Chinese text on BOTH chunk-boundary sides."""
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"def {n}(value_a, value_b):{eol}"]
    lines.append(
        f'{ind}"""{n}：中文语料函数 — 分块边界多字节安全测试。"""{eol}',
    )
    lines.append(f"{ind}# 中文注释：边界首行 — {n} 快速编辑语料{eol}")
    for i in range(1, req.steps + 1):
        if i == 1:
            lines.append(
                f"{ind}acc_{n} = value_a + value_b + {i}  # {n} step {i}{eol}",
            )
        else:
            lines.append(f"{ind}acc_{n} += {i}  # {n} step {i}{eol}")
    lines.append(f"{ind}return acc_{n}  # 中文注释：边界末行 — {n} 多字节安全{eol}")
    return lines


def _rs_cjk_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"fn {n}(value_a: i32, value_b: i32) -> i32 {{{eol}"]
    lines.append(f"{ind}// 中文注释：边界首行 — {n} 快速编辑语料{eol}")
    for i in range(1, req.steps + 1):
        if i == 1:
            lines.append(
                f"{ind}let mut acc_{n} = value_a + value_b + {i}; "
                f"// {n} step {i}{eol}",
            )
        else:
            lines.append(f"{ind}acc_{n} += {i}; // {n} step {i}{eol}")
    lines.append(f"{ind}acc_{n} // 中文注释：边界末行 — {n} 多字节安全{eol}")
    lines.append(f"}}{eol}")
    return lines


def _go_cjk_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [f"func {n}(valueA int, valueB int) int {{{eol}"]
    lines.append(f"{ind}// 中文注释：边界首行 — {n} 快速编辑语料{eol}")
    for i in range(1, req.steps + 1):
        if i == 1:
            lines.append(
                f"{ind}acc{n} := valueA + valueB + {i} // {n} step {i}{eol}",
            )
        else:
            lines.append(f"{ind}acc{n} += {i} // {n} step {i}{eol}")
    lines.append(f"{ind}return acc{n} // 中文注释：边界末行 — {n} 多字节安全{eol}")
    lines.append(f"}}{eol}")
    return lines


def _ts_cjk_function_lines(req: SymbolRequest) -> list[str]:
    ind, eol, n = req.indent, req.eol, req.name
    lines = [
        f"export function {n}(valueA: number, valueB: number): number {{{eol}",
    ]
    lines.append(f"{ind}// 中文注释：边界首行 — {n} 快速编辑语料{eol}")
    for i in range(1, req.steps + 1):
        if i == 1:
            lines.append(
                f"{ind}let acc{n} = valueA + valueB + {i}; // {n} step {i}{eol}",
            )
        else:
            lines.append(f"{ind}acc{n} += {i}; // {n} step {i}{eol}")
    lines.append(
        f"{ind}return acc{n}; // 中文注释：边界末行 — {n} 多字节安全{eol}",
    )
    lines.append(f"}}{eol}")
    return lines


# ---------------------------------------------------------------------------
# Step D2: the TEXT dialect — a natural shift log (timestamped entries)
# ---------------------------------------------------------------------------

_TEXT_TEMPLATES: tuple[str, ...] = (
    "The dock-two compressor is still on the service list this week.",
    "Dana walked the mezzanine and found two pallets of unlabeled returns.",
    "The labeling printer on the north wall jammed twice before lunch.",
    "Luis patched the manifest exporter and filed the ticket with the desk.",
    "The cycle-count variance in aisle four came in under one percent.",
    "The carrier audit flagged three manifests carrying stale rates.",
    "Priya renegotiated the express surcharge with the regional desk.",
    "The mezzanine elevator passed its inspection with one advisory note.",
    "A forklift battery died mid-shift; the spare was charged by midnight.",
    "The storm delayed the inbound freight by nearly four hours.",
    "The dock crew split the backlog before the second shift started.",
    "The night crew inventoried the overflow cage down to the last carton.",
    "The new scanner firmware bricked two handhelds right at boot.",
    "Stock levels for the spring promo were locked this afternoon.",
    "The export to the planning sheet finished without a single warning.",
    "Friday's retro ran long but the action list came out clean.",
    "The rota needs one more volunteer for the holiday week Sundays.",
    "The broken label applicator finally shipped out for repair today.",
    "The auditors return next week for the annual safety walkthrough.",
    "The replacement part for the compressor is scheduled to arrive.",
    "The returns ledger still carries the typo the auditors laughed about.",
    "The handoff notes now live on the shared drive instead of the board.",
    "The night shift wants the conveyor belt sensor recalibrated.",
    "The pallet wrap order arrived late and held up the outbound lane.",
    "The freezer door seal was replaced during the deep clean.",
    "The courier route through the industrial park changed on Monday.",
    "The training room projector stays broken until the part lands.",
    "The safety walkthrough checklist moved onto the wall by the ramp.",
    "The office coffee machine is back after two weeks in the shop.",
    "The parking lot striping crew finished the far row overnight.",
    "The vendor rep left samples of the new crate liner at the desk.",
    "The weighbridge calibration certificate is posted by the scale.",
    "The second forklift passed inspection after the brake job.",
    "The racking in aisle nine was re-shelved over the weekend.",
    "The loading dock light on the north corner finally got fixed.",
    "The paper towel order for the break room doubled this month.",
    "The shipping schedule for December went out to the carriers.",
    "The lost-and-found bin by the guard shack was cleared out.",
    "The pallet jack with the squeaky wheel is back in service.",
    "The weather station on the roof survived the last storm intact.",
    "The weekend audit of the cage labels found nothing out of place.",
    "The window repairs on the office floor start next Tuesday.",
    "The winter boots order for the yard crew arrived two sizes wrong.",
    "The wire cage for the recycling moved beside the baler at last.",
    "The yard tractor finally got its hydraulic hose replaced.",
    "The yellow pallet tags ran out mid-afternoon again.",
    "The youth apprentices start their forklift training next month.",
    "The zip ties drawer in the workshop was restocked on Wednesday.",
    "Three carriers quoted the palletized freight to the coast depot.",
    "Three damaged cartons came back from the outlet with no paperwork.",
    "Three new benches arrived for the packing line by the ramp.",
    "Turf repair outside the visitor entrance finished before the rain.",
    "Two aisles of seasonal stock were consolidated into one bay.",
    "Two of the loading dock lights are still waiting on a part.",
    "Two pallets of returned paint were quarantined behind the cage.",
    "Voltage checks on the charging bays passed without a single fault.",
    "Volunteers from the office helped clear the overflow lane.",
    "Wall anchors for the new racking arrived without their bolts.",
    "Waste collection moved to Fridays without anyone telling the yard.",
    "Water bottles for the floor crew were restocked by the cooler.",
    "Weighing errors on the dock scale were traced to a worn load cell.",
    "Wind damage to the yard fence was reported to the landlord.",
    "Wireless scanners dropped their signal in the freezer aisle again.",
    "Wooden pallets are stacking up behind the building again.",
    "Work orders for the conveyor overhaul were signed off Thursday.",
    "Wrist seals for the cold store were issued to the evening crew.",
    "Yard markings for the new trailer bays were painted overnight.",
    "Yellow floor paint is peeling near the pedestrian crossing.",
    "Yield signs at the ramp junction were cleaned and re-posted.",
    "A leaking roof tile above the returns desk was patched Monday.",
    "An empty stillage frame is blocking the access to bay twelve.",
    "Battery watering for the electric trucks is due this weekend.",
    "Cardboard baler twine snapped twice during the morning run.",
    "Chemical spill kits in the east aisle were inspected and tagged.",
    "Conveyor belt two squeals whenever the speed drops below half.",
    "Damp patches on the ceiling above aisle seventeen were reported.",
    "Deliveries for the fit-out crew should use the service road now.",
    "Door closers on the cold store were adjusted by maintenance.",
    "Drainage along the north apron backed up during the downpour.",
    "Dust sheets cover the mezzanine railings until painting ends.",
    "Exit signage on the mezzanine failed its weekly bulb check.",
    "Fire extinguisher tags in the east stairwell are a month stale.",
    "Floor markings around the charger bays are fading fast.",
    "Forklift horns are being tested at the start of every shift.",
    "Gate passes for the concrete lorries need two signatures now.",
    "Glass panels in the boardroom door were replaced yesterday.",
    "Grease nipples on the pallet stacker were serviced Tuesday.",
    "Gravel delivery for the yard potholes arrives Thursday morning.",
    "Hand sanitiser dispensers by the clocking points were refilled.",
    "Hoist inspections for the mezzanine lift are booked for Friday.",
    "Ice on the ramp steps was gritted before the early shift.",
    "Insulation offcuts from the office refit are bound for recycling.",
    "Keys for the overflow yard were handed to the night supervisor.",
    "Kit inspections for the climbing harnesses happen next week.",
    "Ladder registers in the packing annex were up to date Monday.",
    "Lockers by the changing room were rekeyed after the audit.",
)

_TEXT_ZH: tuple[str, ...] = (
    "中文备注：夜班组长确认退货区域在闭店前完成复核。",
    "中文备注：周五复盘确认库存差异已降至千分之三，无需追加盘点。",
    "中文备注：传送带传感器已重新校准，夜班交接已记录。",
)


def _text_header(seed: str, eol: str) -> list[str]:
    return [
        f"Fastedit shift-log corpus (tests/corpus.py, seed {seed}).{eol}",
        eol,
    ]


def _text_entry_id(name: str) -> str:
    """The log entry's ID: the paragraph name's numeric suffix.

    Corpus paragraph names (``para_%06d``) yield unique, natural-looking
    entry numbers; non-conforming names (the op payload functions) fall
    back to a deterministic hash so any renderer call stays stable.
    """
    match = re.fullmatch(r"para_(\d+)", name)
    if match:
        return match.group(1)
    rng = random.Random(f"fastedit-text-entry-id|{name}")
    return str(rng.randrange(100000, 999999))


_TEXT_TEMPLATE_STRIDE = 11
"""Per-paragraph stride over the template pool (window-distinctness).

The D2 real-model measurements (tests/test_real_llm_text_chunks.py, and
the C3 seams probe before them) fixed one property the ~40-line windows
the real model merges must have: a chunk the model must re-emit may not
repeat a line within itself — a repetitive chunk makes the model lossily
compress it (echoed markers, dropped tails) and the validation battery
rightly rejects every attempt, so the edit can never converge. Free
``rng.choice`` draws collide: with a 40-template pool and 2-5-sentence
paragraphs, measured windows carried 4-10 repeated lines and every real
merge was rejected.

Each paragraph therefore draws a CONSECUTIVE block of template indices —
``stride * entry_id + line`` mod the pool size — with the stride greater
than the maximum body length (:data:`_TEXT_MAX_BODY_SENTENCES`, 10).
Collision between paragraphs ``k`` and ``k+d`` needs
``d * stride mod pool_size`` within a body length of zero: for the
96-template pool the reachable distances (a window spans at most 8
consecutive 3-sentence paragraphs inside its ~40-line budget) give
11..77 — never inside ±9. Every window-sized run of consecutive
paragraphs is therefore repeat-free BY CONSTRUCTION, while paragraphs
exactly 96 ids apart still share sentences (file-wide duplicates stay —
exact repeats are cheap for the model to copy across the spec-decode
draft, and the D2 anchor matcher anchors windows on entry lines only).
The stride block is deterministic per name and needs no generator state:
the renderer stays a pure function of its :class:`SymbolRequest`.
"""

_TEXT_MAX_BODY_SENTENCES = 10
"""The text dialect's maximum body length (sentences per log entry).

Mirrors the largest value ``_next_body_steps`` can draw for a non-record
symbol; the stride-disjointness argument above needs the stride (11) to
exceed it, and the 90-130-line monster branch is skipped for the text
recipe — a hundred-sentence "log entry" is not a log entry, and a
paragraph longer than the window budget would wrap the template ring
inside one paragraph and reintroduce window repeats.
"""

_TEXT_WINDOW_SPAN_LINES = 40
"""The corpus-side mirror of ``chunk_locator._MAX_TEXT_CHUNK_LINES``.

``verify_recipe_structures`` asserts the window-distinctness invariant
against the maximal single-anchor window the D2 locator can produce (the
budget ± its half-context); the value is mirrored here because the oracle
never imports fastedit. If the locator's budget changes, change this with
it — the verify step then re-checks the new windows.
"""


def _text_paragraph_lines(req: SymbolRequest) -> list[str]:
    """One paragraph of the text dialect: a NATURAL shift-log entry.

    The dialect is modeled on real .txt/.log shift logs (the D1 real-model
    measurements — tests/test_real_llm_text.py — showed the trained model
    reproduces and edits NATURAL prose faithfully while synthetic-looking
    text (dense unique numbers, machine-style prefixes) makes it drift or
    echo markers). Each paragraph is one log entry:

    * the ENTRY LINE — ``2024-06-DD HH:MM — shift log entry <id> opened by
      the duty supervisor.`` — carries the paragraph's unique ID (its name's
      numeric suffix): the one file-wide-unique line a D2 anchor matcher can
      anchor a window on;
    * the BODY is 2-5 natural sentences drawn from the fixed template pool
      through the :data:`_TEXT_TEMPLATE_STRIDE` block (deterministic per
      name — see that constant for the window-distinctness argument), plus
      one Chinese line on sprinkled entries (multibyte coverage). Body
      sentences repeat ACROSS distant entries only — a window-sized run of
      consecutive paragraphs never repeats a line, which the real-model
      measurements make load-bearing (a repetitive chunk is lossily
      compressed by the model and the edit can never converge).

    ``req.variant == "edited"`` (the replace oracle's payload) restates the
    ENTRY LINE byte-exactly (variant-independent — the oracle's
    signature-restatement precondition) and REVISION-labels the body.
    """
    eol, name = req.eol, req.name
    body = max(2, req.steps)
    clock = random.Random(f"fastedit-text-clock|{name}")
    base = clock.randrange(60000)
    stamp = (
        f"2024-06-{11 + base % 9:02d} {(base // 60) % 24:02d}:{base % 60:02d}"
    )
    entry = (
        f"{stamp} — shift log entry {_text_entry_id(name)} "
        f"opened by the duty supervisor.{eol}"
    )
    lines = [entry]
    if req.cjk_step is not None:
        lines.append(f"中文备注：{stamp} 之后的复核记录，多字节安全。{eol}")
    edited = req.variant == "edited"
    block_start = (
        _TEXT_TEMPLATE_STRIDE * int(_text_entry_id(name))
    ) % len(_TEXT_TEMPLATES)
    for i in range(body):
        if edited:
            text = (
                f"REVISION {i + 1}: the declared op replaced this entry's "
                f"notes; the duty supervisor signed the change off."
            )
        else:
            text = _TEXT_TEMPLATES[(block_start + i) % len(_TEXT_TEMPLATES)]
            text = text[0].upper() + text[1:]
        lines.append(f"{text}{eol}")
    return lines


# ---------------------------------------------------------------------------
# Step D2 stress: the MARKDOWN dialect renderers — flat level-1 sections
# ---------------------------------------------------------------------------

_MD_SECTION_PROSE_LINES = 2
"""Natural prose lines per markdown section (stride-drawn — see the text
dialect for why the stride draw is load-bearing)."""

_MD_FENCE_LINES = 3
"""Code lines inside each section's fenced block."""


def _md_header(seed: str, eol: str) -> list[str]:
    return [
        f"# Fastedit operations corpus (tests/corpus.py, seed {seed}).{eol}",
        eol,
    ]


def _md_section_id(name: str) -> int:
    """The numeric suffix of a ``Section-%06d`` name (deterministic fallback)."""
    match = re.fullmatch(r"Section-(\d+)", name)
    if match:
        return int(match.group(1))
    return random.Random(f"fastedit-md-section-id|{name}").randrange(10 ** 6)


def _md_section_lines(req: SymbolRequest) -> list[str]:
    """One markdown section: heading, prose, one fenced block.

    The heading restates the section's own name byte-exactly in BOTH
    variants (the replace oracle's signature-restatement precondition);
    ``variant == "edited"`` revises only the prose. The fence lines embed
    the section name, so every line of a section is unique within it and
    code-shaped for the model.
    """
    eol, name = req.eol, req.name
    lines = [f"# {name}{eol}", eol]
    block_start = (
        _TEXT_TEMPLATE_STRIDE * _md_section_id(name)
    ) % len(_TEXT_TEMPLATES)
    edited = req.variant == "edited"
    for i in range(_MD_SECTION_PROSE_LINES):
        if edited:
            text = (
                f"REVISION {i + 1}: the declared op rewrote this section's "
                f"notes; the operations lead signed the change off."
            )
        else:
            text = _TEXT_TEMPLATES[(block_start + i) % len(_TEXT_TEMPLATES)]
            text = text[0].upper() + text[1:]
        lines.append(f"{text}{eol}")
    lines.append(eol)
    lines.append(f"```{eol}")
    for k in range(1, _MD_FENCE_LINES + 1):
        lines.append(f"{name} step {k}: ratio_{k} = {k} * 2  # deterministic{eol}")
    lines.append(f"```{eol}")
    return lines


_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        language="python",
        ext="py",
        comment_prefix="#",
        indent="    ",
        sep_blanks=2,
        header=_py_header,
        function_lines=_py_function_lines,
        record_lines=_py_record_lines,
        signature_re=r"^(?:def|class)\s+{name}\b",
        closer=None,  # last line is an indented body line
        fn_name=lambda i: f"fn_{i:06d}",
        rec_name=lambda i: f"Rec_{i:06d}",
        op_fn_name=lambda tag, suffix: f"fn_ins_{tag}_{suffix}",
        imports=("import os", "import json"),
        boilerplate_lines=_py_boilerplate_lines,
        deep_lines=_py_deep_lines,
        cjk_function_lines=_py_cjk_function_lines,
    ),
    "rust": LanguageSpec(
        language="rust",
        ext="rs",
        comment_prefix="//",
        indent="    ",
        sep_blanks=1,
        header=_rs_header,
        function_lines=_rs_function_lines,
        record_lines=_rs_record_lines,
        signature_re=r"^(?:fn|struct|enum|trait|impl)\s+{name}\b",
        closer="}",
        fn_name=lambda i: f"fn_{i:06d}",
        rec_name=lambda i: f"Rec{i:06d}",
        op_fn_name=lambda tag, suffix: f"fn_ins_{tag}_{suffix}",
        imports=(
            "use std::collections::HashMap;",
            "use std::fmt;",
        ),
        boilerplate_lines=_rs_boilerplate_lines,
        deep_lines=_rs_deep_lines,
        cjk_function_lines=_rs_cjk_function_lines,
    ),
    "go": LanguageSpec(
        language="go",
        ext="go",
        comment_prefix="//",
        indent="\t",
        sep_blanks=1,
        header=_go_header,
        function_lines=_go_function_lines,
        record_lines=_go_record_lines,
        signature_re=r"^(?:func|type)\s+{name}\b",
        closer="}",
        fn_name=lambda i: f"Fn{i:06d}",
        rec_name=lambda i: f"Rec{i:06d}",
        op_fn_name=lambda tag, suffix: f"FnIns{tag}{suffix.upper()}",
        imports=('import "fmt"', 'import "strings"'),
        boilerplate_lines=_go_boilerplate_lines,
        deep_lines=_go_deep_lines,
        cjk_function_lines=_go_cjk_function_lines,
    ),
    "typescript": LanguageSpec(
        language="typescript",
        ext="ts",
        comment_prefix="//",
        indent="  ",
        sep_blanks=2,
        header=_ts_header,
        function_lines=_ts_function_lines,
        record_lines=_ts_record_lines,
        signature_re=r"^export\s+(?:function|interface|class)\s+{name}\b",
        closer="}",
        fn_name=lambda i: f"fn{i:06d}",
        rec_name=lambda i: f"Rec{i:06d}",
        op_fn_name=lambda tag, suffix: f"fnIns{tag}{suffix.upper()}",
        imports=(
            'import { readFileSync } from "node:fs";',
            'import { join } from "node:path";',
        ),
        boilerplate_lines=_ts_boilerplate_lines,
        deep_lines=_ts_deep_lines,
        cjk_function_lines=_ts_cjk_function_lines,
    ),
    # Step D2: the TEXT dialect — paragraphs, not code. "Symbols" are
    # paragraphs: each one's ENTRY LINE is file-wide unique (the timestamped
    # log-entry shape carrying the paragraph's own ID — variant-independently,
    # so the replace oracle's signature-restatement precondition holds
    # verbatim) and is the line a D2 anchor matcher anchors a window on;
    # body sentences are natural shift-log prose drawn per
    # :data:`_TEXT_TEMPLATE_STRIDE` blocks, so a window-sized run of
    # consecutive paragraphs never repeats a line (the real model lossily
    # compresses repetitive chunks — the C3 distinctness doctrine applies to
    # the model's WINDOW). The ``recipe="text"`` dispatch in
    # ``generate_big_source`` selects this spec; ``language="text"`` is not a
    # grammar, it is the corpus's name for plain prose (the pipeline sees
    # ``language=None`` for it).
    "text": LanguageSpec(
        language="text",
        ext="txt",
        comment_prefix="#",
        indent="",
        sep_blanks=1,
        header=_text_header,
        function_lines=_text_paragraph_lines,
        record_lines=_text_paragraph_lines,
        signature_re=r"^2024-\d\d-\d\d \d\d:\d\d — shift log entry \d+ opened",
        closer=None,
        fn_name=lambda i: f"para_{i:06d}",
        rec_name=lambda i: f"para_{i:06d}",
        op_fn_name=lambda tag, suffix: f"para_ins_{tag}_{suffix}",
        imports=(),
        cjk_function_lines=_text_paragraph_lines,
    ),
    # Step D2 stress: the MARKDOWN dialect — grammar-backed since B2, so it
    # exercises the AST chunk path at 100MB (heading-anchored `section`
    # symbols; ast_utils._FORMAT_SYMBOL_SPECS names a section by its heading
    # text). Each "symbol" is one flat level-1 section: heading + two
    # stride-drawn natural prose lines + one fenced block whose lines embed
    # the section's name (unique, code-like — the model merges them
    # faithfully). Sections stay far below the parent-snap cap, and the
    # prose never repeats within a section (the stride draw).
    "markdown": LanguageSpec(
        language="markdown",
        ext="md",
        comment_prefix="",
        indent="",
        sep_blanks=1,
        header=_md_header,
        function_lines=_md_section_lines,
        record_lines=_md_section_lines,
        signature_re=r"^# {name}\b",
        closer=None,  # the section's last line is its closing fence
        fn_name=lambda i: f"Section-{i:06d}",
        rec_name=lambda i: f"Section-{i:06d}",
        op_fn_name=lambda tag, suffix: f"Section-ins-{tag}-{suffix}",
        imports=(),
        cjk_function_lines=_md_section_lines,
    ),
}


def _spec_of(language: str) -> LanguageSpec:
    try:
        return _SPECS[language]
    except KeyError:
        raise ValueError(
            f"unsupported corpus language {language!r}; supported: {sorted(_SPECS)}"
        ) from None


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def _rng_key(
    language: str, seed: int | str, eol: str, final_eol: bool, sprinkle_cjk: bool,
    indent: str, recipe: str = RECIPE_PLAIN,
) -> str:
    """Deterministic RNG key — same seed + options -> byte-identical output."""
    base = (
        f"fastedit-corpus|{language}|{seed}|{eol!r}|{final_eol}|{sprinkle_cjk}"
        f"|{indent!r}"
    )
    if recipe == RECIPE_PLAIN:
        # BYTE-COMPAT: the committed tests/golden/big/ fixtures were
        # generated with exactly this key; the plain recipe must never
        # perturb it or they stop regenerating byte-exactly.
        return base
    return f"{base}|{recipe}"


def _next_body_steps(
    rng: random.Random, index: int, recipe: str = RECIPE_PLAIN,
) -> int:
    """Body-statement count for symbol ``index`` (total stays <= the cap).

    Boilerplate semantics: the count is the number of SHARED 4-line body
    block repetitions (the monster repeats the block
    :data:`BOILERPLATE_MONSTER_REPEATS` times), so every boilerplate body
    stays free of the symbol's name — the duplicated-line trap.
    """
    if recipe == RECIPE_BOILERPLATE:
        if index % 97 == 0:
            return BOILERPLATE_MONSTER_LINES
        return rng.choice((2, 3, 4, 5))
    if index % 97 == 0 and recipe != RECIPE_TEXT:
        # Deliberately large (but <= MAX_SYMBOL_LINES even with signature,
        # docstring, CJK comment, tail and closer lines) so chunk sizes vary.
        # Skipped for the text recipe: a 90-130-sentence "log entry" is not
        # a log entry, and a paragraph longer than the D2 window budget
        # would wrap the template ring inside one paragraph and
        # reintroduce window repeats (see _TEXT_TEMPLATE_STRIDE).
        return rng.randrange(90, 131)
    return rng.choice((3, 4, 5, 6, 8, 10))


def _assert_terminated(lines: list[str], eol: str, name: str) -> None:
    for line in lines:
        assert line.endswith(eol), f"{name}: line is not {eol!r}-terminated: {line!r}"
        assert eol not in line[: -len(eol)], f"{name}: embedded line break: {line!r}"


def generate_big_source(
    language: str,
    target_bytes: int,
    *,
    eol: str = EOL_LF,
    final_eol: bool = True,
    indent: str | None = None,
    seed: int | str = 0,
    sprinkle_cjk: bool = False,
    min_symbols: int = 24,
    recipe: str = RECIPE_PLAIN,
) -> CorpusSource:
    """Build a seeded corpus of small top-level functions/records.

    Args:
        language: one of the corpus languages (python/rust/go/typescript —
            the C2 targets).
        target_bytes: approximate UTF-8 byte budget. The generator stops
            BEFORE exceeding it, so the result is at most one symbol short;
            under-run is bounded by the largest single symbol (well under
            16 KiB — see the size-accuracy self-test).
        eol: the file's line-ending convention (``"\\n"`` or ``"\\r\\n"``);
            every emitted line, separator and payload uses it.
        final_eol: when False the file's last line carries no terminator.
        indent: body indentation; ``None`` uses the language default
            (python 4 spaces, rust 4 spaces, go tab, typescript 2 spaces).
        seed: deterministic seed — same seed and options produce
            byte-identical output.
        sprinkle_cjk: add a Chinese comment line to every 5th function.
        min_symbols: lower bound on emitted symbols so op targeting has
            disjoint candidates even for tiny budgets.
        recipe: one of :data:`RECIPES`. ``"plain"`` (default) is the C1
            generator, byte-compatible with the committed goldens; the C3
            recipes are described in the module docstring.

    Returns:
        :class:`CorpusSource` — the text (a ``str``) carrying ``.manifest``,
        the symbol -> line-span table the oracle's arithmetic relies on.
    """
    spec = _spec_of("text" if recipe == RECIPE_TEXT else language)
    if recipe == RECIPE_TEXT and language != "text":
        raise ValueError(
            'recipe="text" requires language="text" (the corpus name for '
            "plain prose; the pipeline sees language=None for it)"
        )
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; supported: {RECIPES}")
    if eol not in (EOL_LF, EOL_CRLF):
        raise ValueError(f"eol must be {EOL_LF!r} or {EOL_CRLF!r}, got {eol!r}")
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes}")
    if min_symbols < 8:
        raise ValueError(
            "min_symbols must be >= 8 so make_ops can pick four disjoint targets",
        )
    body_indent = spec.indent if indent is None else indent
    rng = random.Random(
        _rng_key(language, seed, eol, final_eol, sprinkle_cjk, body_indent, recipe),
    )
    seed_str = str(seed)

    # Memory discipline: symbol chunks accumulate in this list and are
    # joined exactly once at the end — never string +=, never a second
    # full-size copy beyond the join itself.
    parts: list[str] = []
    header_lines = spec.header(seed_str, eol)
    import_lines: tuple[int, int] | None = None
    if recipe not in (RECIPE_PLAIN, RECIPE_TEXT):
        # Plain is byte-compatible with the committed goldens; text is
        # prose — a header import block is a code concept and the txt
        # dialect has none (the D2 text windows anchor on paragraph
        # lines, never on imports).
        import_block = [line + eol for line in spec.imports]
        assert import_block, f"{language}: recipe {recipe!r} needs header imports"
        import_lines = (
            len(header_lines) + 1, len(header_lines) + len(import_block),
        )
        header_lines = header_lines + import_block
    parts.extend(header_lines)
    parts.extend([eol] * spec.sep_blanks)
    byte_len = sum(len(part.encode("utf-8")) for part in parts)
    line_no = len(header_lines) + spec.sep_blanks

    symbols: list[SymbolSpan] = []
    index = 0
    while True:
        index += 1
        is_record = index % 7 == 0
        steps = _next_body_steps(rng, index, recipe)
        fields = 3 + (index % 3)
        is_deep = (
            recipe == RECIPE_DEEP and not is_record and index % DEEP_EVERY == 0
        )
        is_family = (
            recipe == RECIPE_PREFIX and not is_record and index % PREFIX_EVERY == 0
        )
        cjk_step = (
            2 if (sprinkle_cjk and not is_record and index % 5 == 0) else None
        )

        # One loop iteration emits ONE symbol — or, for a prefix family,
        # THREE sibling symbols sharing the base name as a common prefix.
        # Each entry is (name, kind, lines, sub-block offsets 0-indexed).
        emissions: list[tuple[str, str, list[str], tuple[tuple[int, int], ...]]] = []
        if is_family:
            base = spec.fn_name(index)
            for suffix in ("", "_data", "_data_v2"):
                name = base + suffix
                req = SymbolRequest(
                    name=name, steps=steps, fields=fields, indent=body_indent,
                    eol=eol, variant="plain", cjk_step=None,
                    comment_prefix=spec.comment_prefix,
                )
                emissions.append((name, "function", spec.function_lines(req), ()))
        else:
            name = spec.rec_name(index) if is_record else spec.fn_name(index)
            req = SymbolRequest(
                name=name, steps=steps, fields=fields, indent=body_indent,
                eol=eol, variant="plain", cjk_step=cjk_step,
                comment_prefix=spec.comment_prefix,
            )
            if is_record:
                emissions.append((name, "record", spec.record_lines(req), ()))
            elif is_deep:
                sym_lines, offsets = spec.deep_lines(req)
                emissions.append((name, "function", sym_lines, tuple(offsets)))
            elif recipe == RECIPE_BOILERPLATE:
                emissions.append(
                    (name, "function", spec.boilerplate_lines(req), ()),
                )
            elif recipe == RECIPE_CJK_SEAMS:
                emissions.append(
                    (name, "function", spec.cjk_function_lines(req), ()),
                )
            else:
                emissions.append((name, "function", spec.function_lines(req), ()))

        sep = [eol] * (spec.sep_blanks if symbols else 0)
        # Family members are blank-line separated like any other symbol, so
        # the corpus layout invariant (verify_symbol_spans) keeps holding.
        joined_lines: list[str] = []
        for pos, (_n, _k, sym_lines, _b) in enumerate(emissions):
            if pos:
                joined_lines.extend([eol] * spec.sep_blanks)
            joined_lines.extend(sym_lines)
        chunk = "".join(sep + joined_lines)
        chunk_bytes = len(chunk.encode("utf-8"))
        if len(symbols) >= min_symbols and byte_len + chunk_bytes > target_bytes:
            break
        for em_name, _k, sym_lines, _b in emissions:
            _assert_terminated(sym_lines, eol, em_name)
        parts.append(chunk)
        byte_len += chunk_bytes

        line_cursor = line_no + len(sep)
        for pos, (em_name, em_kind, em_lines, offsets) in enumerate(emissions):
            if pos:
                line_cursor += spec.sep_blanks  # the gap before this member
            start_line = line_cursor + 1
            end_line = start_line + len(em_lines) - 1
            blocks = tuple(
                SymbolSpan(
                    name=f"{em_name}#block{b + 1}", kind="block",
                    start_line=start_line + s, end_line=start_line + e,
                )
                for b, (s, e) in enumerate(offsets)
            )
            symbols.append(SymbolSpan(
                name=em_name, kind=em_kind,
                start_line=start_line, end_line=end_line, blocks=blocks,
            ))
            line_cursor = end_line
        line_no = symbols[-1].end_line

    # The corpus must END with a function symbol: append_at_eof resolves to
    # the last symbol and op targets are function-kind only.
    if symbols[-1].kind != "function":
        index += 1
        name = spec.fn_name(index)
        fixup_steps = 1 if recipe == RECIPE_BOILERPLATE else 4
        req = SymbolRequest(
            name=name, steps=fixup_steps, indent=body_indent, eol=eol,
            comment_prefix=spec.comment_prefix,
        )
        if recipe == RECIPE_BOILERPLATE:
            sym_lines = spec.boilerplate_lines(req)
        elif recipe == RECIPE_CJK_SEAMS:
            sym_lines = spec.cjk_function_lines(req)
        else:
            sym_lines = spec.function_lines(req)
        chunk = "".join([eol] * spec.sep_blanks + sym_lines)
        parts.append(chunk)
        byte_len += len(chunk.encode("utf-8"))
        start_line = line_no + spec.sep_blanks + 1
        symbols.append(SymbolSpan(
            name=name, kind="function",
            start_line=start_line, end_line=start_line + len(sym_lines) - 1,
        ))
        line_no = symbols[-1].end_line

    text = "".join(parts)
    if not final_eol:
        assert text.endswith(eol), "generator invariant: last line is terminated"
        text = text[: -len(eol)]

    manifest = CorpusManifest(
        language=language,
        ext=spec.ext,
        eol=eol,
        final_eol=final_eol,
        indent=body_indent,
        sprinkle_cjk=sprinkle_cjk,
        seed=seed,
        target_bytes=target_bytes,
        header_line_count=len(header_lines),
        sep_blanks=spec.sep_blanks,
        symbols=tuple(symbols),
        recipe=recipe,
        import_lines=import_lines,
    )
    return CorpusSource(text, manifest)


# ---------------------------------------------------------------------------
# Op construction — resolve targets through the manifest, record everything
# ---------------------------------------------------------------------------


def _terminated(text: str, eol: str) -> list[str]:
    """Payload text -> EOL-terminated lines (B3's ``_terminated`` shape)."""
    body = text.rstrip("\r\n")
    if not body:
        return []
    if eol == EOL_CRLF:
        # A CRLF payload must be uniformly CRLF — an LF payload here would
        # be a mixed-ending seam the funnel could not repair.
        assert body.count("\n") == body.count(EOL_CRLF), (
            f"payload is not uniformly CRLF: {body[:80]!r}"
        )
    return [line + eol for line in body.split(eol)]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _op_tag(seed: int | str) -> str:
    digits = "".join(ch for ch in str(seed) if ch.isdigit())[:8]
    return digits or "00000000"


def _make_splice_op(
    kind: str,
    manifest: CorpusManifest,
    lines: list[str],
    anchor: SymbolSpan,
    payload: str,
    note: str,
) -> CorpusOp:
    """Build an insert/append op: splice the payload after the anchor."""
    anchor_last_line = lines[anchor.end_line - 1]
    assert anchor_last_line.strip() != "", (
        f"{anchor.name}: anchor's last line is blank — corpus layout invariant"
    )
    following = lines[anchor.end_line:]
    # Corpus layout invariant: symbols are blank-line separated, so a
    # mid-file anchor is followed by a blank line and no trailing separator
    # is ever added (the original's own blanks stay).
    assert not (following and following[0].strip() != ""), (
        f"{anchor.name}: expected a blank separator after the anchor"
    )
    sep_before = anchor_last_line.strip() != ""
    sep_after = bool(following) and following[0].strip() != ""
    return CorpusOp(
        kind=kind,
        symbol=anchor.name,
        language=manifest.language,
        ext=manifest.ext,
        eol=manifest.eol,
        final_eol=manifest.final_eol,
        anchor_start_line=anchor.start_line,
        anchor_end_line=anchor.end_line,
        anchor_last_line=anchor_last_line,
        anchor_is_last_line=(anchor.end_line == len(lines)),
        new_text=payload,
        sep_before=sep_before,
        sep_after=sep_after,
        snippet_line_count=len(_terminated(payload, manifest.eol)),
        note=note,
    )


def _make_replace_op(
    manifest: CorpusManifest,
    spec: LanguageSpec,
    lines: list[str],
    target: SymbolSpan,
    tag: str,
) -> CorpusOp:
    """Build a replace op: a complete re-definition of the target function.

    The payload restates the target's signature line byte-exactly (the
    direct-swap gate's precondition) and changes every other line, so the
    text-match editor has a single context anchor and declines in favor of
    the deterministic direct swap.
    """
    payload = "".join(spec.function_lines(SymbolRequest(
        name=target.name, steps=3, indent=manifest.indent, eol=manifest.eol,
        variant="edited", comment_prefix=spec.comment_prefix,
    )))
    start, end = target.start_line, target.end_line
    original_span_text = "".join(lines[start - 1:end])
    first_payload = payload.split(manifest.eol, 1)[0]
    first_span = original_span_text.split(manifest.eol, 1)[0]
    assert first_payload == first_span, (
        f"{target.name}: replace payload must restate the signature line "
        f"byte-exactly: {first_payload!r} != {first_span!r}"
    )
    return CorpusOp(
        kind="replace_symbol_body",
        symbol=target.name,
        language=manifest.language,
        ext=manifest.ext,
        eol=manifest.eol,
        final_eol=manifest.final_eol,
        start_line=start,
        end_line=end,
        new_text=payload,
        original_span_text=original_span_text,
        note=(
            f"replace the body of '{target.name}' with an edited variant "
            f"(tag {tag}); signature restated for the direct-swap path"
        ),
    )


def _make_delete_op(
    manifest: CorpusManifest, lines: list[str], target: SymbolSpan,
) -> CorpusOp:
    """Build a delete op: remove the span plus its trailing blank separator."""
    start, end = target.start_line, target.end_line
    if not manifest.final_eol:
        assert end < len(lines), (
            f"{target.name}: on a no-final-EOL corpus the delete target must "
            f"not be the file's last symbol — the EOL funnel would strip the "
            f"preceding line's terminator and the insert-inverse could not "
            f"restore the original bytes"
        )
    end_idx = end
    while end_idx < len(lines) and lines[end_idx].strip() == "":
        end_idx += 1
    return CorpusOp(
        kind="delete_symbol",
        symbol=target.name,
        language=manifest.language,
        ext=manifest.ext,
        eol=manifest.eol,
        final_eol=manifest.final_eol,
        start_line=start,
        end_line=end,
        consumed_end_line=end_idx,
        original_span_text="".join(lines[start - 1:end_idx]),
        note=(
            f"delete '{target.name}' and the blank lines that separated it "
            f"from what follows"
        ),
    )


def make_ops(
    source: CorpusSource,
    *,
    kinds: Sequence[str] = OP_KINDS,
    seed: int | str = 0,
) -> tuple[CorpusOp, ...]:
    """Build op records for a generated corpus.

    Every target is resolved through ``source.manifest`` (the generator's
    symbol -> line-span table) — never by parsing. Targets are
    function-kind symbols, pairwise disjoint, at deterministic positions
    (insert ~1/4, replace ~1/2, delete ~3/4, append at the last symbol);
    ops are returned in that canonical order so
    :func:`apply_op_sequence` can apply them front-to-back.
    """
    manifest = source.manifest
    spec = _spec_of(manifest.language)
    unknown = [kind for kind in kinds if kind not in OP_KINDS]
    if unknown:
        raise ValueError(f"unknown op kind(s) {unknown}; supported: {OP_KINDS}")
    lines = source.splitlines(keepends=True)
    rng = random.Random(f"fastedit-corpus-ops|{manifest.language}|{seed}")
    tag = _op_tag(seed)

    fn_indices = [
        i for i, sym in enumerate(manifest.symbols) if sym.kind == "function"
    ]
    assert fn_indices, "corpus has no function symbols to target"
    last_index = len(manifest.symbols) - 1
    assert manifest.symbols[last_index].kind == "function", (
        "generator invariant: the corpus ends with a function symbol"
    )

    taken: set[int] = set()

    def pick(fraction: float) -> int:
        """Pick an untaken function index near ``fraction`` of the corpus.

        The LAST symbol is reserved for append_at_eof (and kept clear of the
        delete target on no-final-EOL corpora — see the constraint in
        ``_make_delete_op``).
        """
        center = int(len(manifest.symbols) * fraction)
        window = [
            i for i in fn_indices
            if abs(i - center) <= max(8, len(manifest.symbols) // 20)
            and i not in taken and i != last_index
        ]
        if not window:
            window = [i for i in fn_indices if i not in taken and i != last_index]
        assert window, "corpus too small to pick disjoint op targets"
        choice = rng.choice(window)
        taken.add(choice)
        return choice

    def payload_function(name: str) -> str:
        return "".join(spec.function_lines(SymbolRequest(
            name=name, steps=4, indent=manifest.indent, eol=manifest.eol,
            variant="plain", comment_prefix=spec.comment_prefix,
        )))

    ops: list[CorpusOp] = []
    if "insert_after_symbol" in kinds:
        anchor = manifest.symbols[pick(0.25)]
        ops.append(_make_splice_op(
            "insert_after_symbol", manifest, lines, anchor,
            payload_function(spec.op_fn_name(tag, "a")),
            f"insert a fresh {manifest.language} function after '{anchor.name}'",
        ))
    if "replace_symbol_body" in kinds:
        target = manifest.symbols[pick(0.5)]
        ops.append(_make_replace_op(manifest, spec, lines, target, tag))
    if "delete_symbol" in kinds:
        target = manifest.symbols[pick(0.75)]
        ops.append(_make_delete_op(manifest, lines, target))
    if "append_at_eof" in kinds:
        anchor = manifest.symbols[last_index]
        ops.append(_make_splice_op(
            "append_at_eof", manifest, lines, anchor,
            payload_function(spec.op_fn_name(tag, "b")),
            f"append a fresh {manifest.language} function at EOF "
            f"(after the last symbol '{anchor.name}')",
        ))
    return tuple(ops)


def build_corpus_case(
    language: str,
    target_bytes: int,
    *,
    eol: str = EOL_LF,
    final_eol: bool = True,
    indent: str | None = None,
    seed: int | str = 0,
    sprinkle_cjk: bool = False,
    kinds: Sequence[str] = OP_KINDS,
    recipe: str = RECIPE_PLAIN,
) -> CorpusCase:
    """Generate a corpus and its ops in one call (C2/C3-ready)."""
    source = generate_big_source(
        language, target_bytes,
        eol=eol, final_eol=final_eol, indent=indent, seed=seed,
        sprinkle_cjk=sprinkle_cjk, recipe=recipe,
    )
    ops = make_ops(source, kinds=kinds, seed=seed)
    manifest = source.manifest
    return CorpusCase(
        language=language,
        ext=manifest.ext,
        eol=manifest.eol,
        final_eol=manifest.final_eol,
        indent=manifest.indent,
        sprinkle_cjk=manifest.sprinkle_cjk,
        seed=seed,
        target_bytes=target_bytes,
        source=source,
        ops=ops,
        recipe=manifest.recipe,
    )


# ---------------------------------------------------------------------------
# THE ORACLE — independent line-splice arithmetic on recorded indices
# ---------------------------------------------------------------------------


def _enforce_trailing_state(result_text: str, original_text: str, eol: str) -> str:
    """Mirror the pipeline's central EOL funnel — its DOCUMENTED contract.

    fastedit's ``_normalize_merged_eol`` enforces two policies on every
    return path: a uniform original's ending convention is applied to the
    produced pieces, and the file's trailing-newline state is taken from
    the ORIGINAL (no trailing terminator in -> none out, trailing EOL
    characters stripped; terminator in -> exactly one out). The corpus
    generator only produces uniform-convention files, and every payload is
    built in the file's own convention, so the normalization half is a
    no-op here; only the trailing rule can bite, and only for EOF-span ops
    on no-final-EOL corpora.
    """
    if original_text.endswith(("\n", "\r")):
        if result_text and not result_text.endswith(("\n", "\r")):
            return result_text + eol
        return result_text
    return result_text.rstrip("\r\n")


def apply_op_oracle(original_text: str, op: CorpusOp) -> str:
    """Expected output bytes for one op — pure line arithmetic.

    INDEPENDENT of fastedit: no import, no pipeline run, no parser. The
    op's target comes from the generator-recorded span data; every recorded
    value is cross-checked against ``original_text`` so a drifted manifest
    fails loudly instead of pinning wrong behavior.

    Semantics per kind (all splice arithmetic on keepends lines):

    * ``insert_after_symbol`` / ``append_at_eof`` — splice the payload's
      EOL-terminated lines after the anchor's last line, wrapped in one
      blank separator line on each side where the adjacent original line is
      non-blank (the pipeline's zero-token fast path).
    * ``replace_symbol_body`` — replace the recorded span's lines with the
      payload's terminated lines (the deterministic direct swap). The
      payload's first line must already carry the target's indent (the
      pipeline's alignment is a no-op by construction).
    * ``delete_symbol`` — remove the span plus the blank lines that
      separated it from whatever follows (the AST delete).

    The result then passes through :func:`_enforce_trailing_state` (the
    funnel's documented trailing-newline rule).
    """
    lines = original_text.splitlines(keepends=True)

    if op.kind in ("insert_after_symbol", "append_at_eof"):
        anchor_end = op.anchor_end_line
        if anchor_end is None or not 1 <= anchor_end <= len(lines):
            raise ValueError(
                f"{op.kind} {op.symbol}: anchor_end_line {anchor_end!r} out of "
                f"range for a {len(lines)}-line text",
            )
        raw_anchor = lines[anchor_end - 1]
        assert raw_anchor == op.anchor_last_line, (
            f"{op.kind} {op.symbol}: anchor line {anchor_end} drifted: "
            f"{raw_anchor!r} != recorded {op.anchor_last_line!r}"
        )
        snippet_lines = _terminated(op.new_text, op.eol)
        assert len(snippet_lines) == op.snippet_line_count, (
            f"{op.kind} {op.symbol}: payload line count drifted"
        )
        sep_before = [op.eol] if raw_anchor.strip() != "" else []
        following = lines[anchor_end:]
        sep_after = [op.eol] if following and following[0].strip() != "" else []
        assert bool(sep_before) == op.sep_before and bool(sep_after) == op.sep_after, (
            f"{op.kind} {op.symbol}: separator flags drifted "
            f"(got before={bool(sep_before)} after={bool(sep_after)}, "
            f"recorded before={op.sep_before} after={op.sep_after})"
        )
        merged = lines[:anchor_end] + sep_before + snippet_lines + sep_after + following
        return _enforce_trailing_state("".join(merged), original_text, op.eol)

    if op.kind == "replace_symbol_body":
        start, end = op.start_line, op.end_line
        if start is None or end is None or not 1 <= start <= end <= len(lines):
            raise ValueError(
                f"replace_symbol_body {op.symbol}: span "
                f"({start!r}, {end!r}) out of range for a {len(lines)}-line text",
            )
        span_text = "".join(lines[start - 1:end])
        assert span_text == op.original_span_text, (
            f"replace_symbol_body {op.symbol}: span {start}-{end} drifted from "
            f"the recorded original span"
        )
        first_target = lines[start - 1]
        first_payload = op.new_text.split(op.eol, 1)[0]
        assert _indent_of(first_payload) == _indent_of(first_target), (
            f"replace_symbol_body {op.symbol}: payload indent "
            f"{_indent_of(first_payload)!r} does not match the target indent "
            f"{_indent_of(first_target)!r} (pre-alignment precondition)"
        )
        merged = lines[: start - 1] + _terminated(op.new_text, op.eol) + lines[end:]
        return _enforce_trailing_state("".join(merged), original_text, op.eol)

    if op.kind == "delete_symbol":
        start, end = op.start_line, op.end_line
        if start is None or end is None or not 1 <= start <= end <= len(lines):
            raise ValueError(
                f"delete_symbol {op.symbol}: span ({start!r}, {end!r}) out of "
                f"range for a {len(lines)}-line text",
            )
        end_idx = end
        while end_idx < len(lines) and lines[end_idx].strip() == "":
            end_idx += 1
        removed = "".join(lines[start - 1:end_idx])
        assert removed == op.original_span_text, (
            f"delete_symbol {op.symbol}: removed span drifted from the "
            f"recorded original bytes"
        )
        assert end_idx == op.consumed_end_line, (
            f"delete_symbol {op.symbol}: consumed blank span drifted "
            f"({end_idx} != {op.consumed_end_line})"
        )
        merged = lines[: start - 1] + lines[end_idx:]
        return _enforce_trailing_state("".join(merged), original_text, op.eol)

    raise ValueError(f"unknown op kind {op.kind!r}; supported: {OP_KINDS}")


# ---------------------------------------------------------------------------
# Backward reconstruction — inverse ops as concrete line-splice data
# ---------------------------------------------------------------------------


def inverse_op(op: CorpusOp) -> LineOp:
    """The op's inverse: concrete line-splice data for the EDITED text.

    Applying the returned :class:`LineOp` with :func:`apply_line_op` to
    ``apply_op_oracle(original_text, op)`` regenerates ``original_text``
    byte-for-byte. The data is entirely recorded (never re-parsed):

    * insert/append inverse = DELETE of exactly the lines the splice added
      (separator lines included);
    * ... except an EOF append on a no-final-EOL corpus, where the splice
      TERMINATED the anchor's last line and the funnel stripped the
      payload's trailing EOL — the inverse is a REPLACE of the anchor line
      plus the added lines with the anchor's recorded (unterminated)
      original bytes;
    * replace inverse = REPLACE of the payload's span with the recorded
      original span bytes;
    * delete inverse = INSERT of the recorded removed bytes (span plus the
      consumed blank separators) after the line that preceded the span.
    """
    if op.kind in ("insert_after_symbol", "append_at_eof"):
        if not op.final_eol and op.anchor_is_last_line:
            return LineOp(
                kind="replace",
                start_line=op.anchor_end_line,
                end_line=op.anchor_end_line + op.snippet_line_count,
                text=op.anchor_last_line,
            )
        added = int(op.sep_before) + op.snippet_line_count + int(op.sep_after)
        first_added = op.anchor_end_line + 1
        return LineOp(
            kind="delete",
            start_line=first_added,
            end_line=first_added + added - 1,
        )
    if op.kind == "replace_symbol_body":
        new_count = len(_terminated(op.new_text, op.eol))
        return LineOp(
            kind="replace",
            start_line=op.start_line,
            end_line=op.start_line + new_count - 1,
            text=op.original_span_text,
        )
    if op.kind == "delete_symbol":
        return LineOp(
            kind="insert",
            after_line=op.start_line - 1,
            text=op.original_span_text,
        )
    raise ValueError(f"unknown op kind {op.kind!r}; supported: {OP_KINDS}")


def apply_line_op(text: str, line_op: LineOp) -> str:
    """Execute one :class:`LineOp` — exact-bytes line arithmetic.

    This is the inverse executor: payloads are spliced VERBATIM (they carry
    their own endings, including a deliberately unterminated last line for
    no-final-EOL corpora). No EOL funnel runs here — backward
    reconstruction restores recorded bytes, it does not re-derive them.
    """
    lines = text.splitlines(keepends=True)
    if line_op.kind == "insert":
        if not 0 <= line_op.after_line <= len(lines):
            raise ValueError(
                f"insert: after_line {line_op.after_line} out of range for a "
                f"{len(lines)}-line text",
            )
        idx = line_op.after_line
        return "".join(lines[:idx] + [line_op.text] + lines[idx:])
    if line_op.kind == "replace":
        if not 1 <= line_op.start_line <= line_op.end_line <= len(lines):
            raise ValueError(
                f"replace: span ({line_op.start_line}, {line_op.end_line}) out "
                f"of range for a {len(lines)}-line text",
            )
        return "".join(
            lines[: line_op.start_line - 1] + [line_op.text] + lines[line_op.end_line:],
        )
    if line_op.kind == "delete":
        if not 1 <= line_op.start_line <= line_op.end_line <= len(lines):
            raise ValueError(
                f"delete: span ({line_op.start_line}, {line_op.end_line}) out "
                f"of range for a {len(lines)}-line text",
            )
        return "".join(lines[: line_op.start_line - 1] + lines[line_op.end_line:])
    raise ValueError(f"unknown line-op kind {line_op.kind!r}")


def op_line_delta(op: CorpusOp) -> int:
    """Net line-count change one op's splice produces (funnel included)."""
    if op.kind in ("insert_after_symbol", "append_at_eof"):
        if not op.final_eol and op.anchor_is_last_line:
            # The separator line-ending merged INTO the anchor's last line.
            return op.snippet_line_count
        return int(op.sep_before) + op.snippet_line_count + int(op.sep_after)
    if op.kind == "replace_symbol_body":
        return len(_terminated(op.new_text, op.eol)) - (op.end_line - op.start_line + 1)
    if op.kind == "delete_symbol":
        return -(op.consumed_end_line - op.start_line + 1)
    raise ValueError(f"unknown op kind {op.kind!r}; supported: {OP_KINDS}")


def _op_position(op: CorpusOp) -> int:
    if op.kind in ("insert_after_symbol", "append_at_eof"):
        return op.anchor_end_line
    return op.start_line


def _shift_op(op: CorpusOp, shift: int) -> CorpusOp:
    """Re-express an op's recorded indices in a shifted line coordinate."""
    return replace(
        op,
        anchor_start_line=(
            op.anchor_start_line + shift if op.anchor_start_line is not None else None
        ),
        anchor_end_line=(
            op.anchor_end_line + shift if op.anchor_end_line is not None else None
        ),
        start_line=op.start_line + shift if op.start_line is not None else None,
        end_line=op.end_line + shift if op.end_line is not None else None,
        consumed_end_line=(
            op.consumed_end_line + shift if op.consumed_end_line is not None else None
        ),
    )


def apply_op_sequence(
    original_text: str, ops: Sequence[CorpusOp],
) -> tuple[str, list[LineOp]]:
    """Apply symbol-resolved ops in order; return the edited text and inverses.

    The ops' recorded line indices are ORIGINAL-text coordinates; each
    application is shifted by the accumulated line delta of the ops before
    it. Requires ops sorted by position and non-overlapping (the canonical
    order :func:`make_ops` returns satisfies both). The returned inverse
    :class:`LineOp` list is materialized in POST-application coordinates:
    applying it in REVERSE with :func:`apply_line_op` regenerates the
    original byte-for-byte.
    """
    text = original_text
    shift = 0
    prev_position = 0
    inverses: list[LineOp] = []
    for op in ops:
        position = _op_position(op)
        if position is None or position <= prev_position:
            raise ValueError(
                f"ops must be sorted by position and non-overlapping; "
                f"{op.kind} {op.symbol} sits at {position!r}",
            )
        shifted = _shift_op(op, shift)
        text = apply_op_oracle(text, shifted)
        inverses.append(inverse_op(shifted))
        shift += op_line_delta(shifted)
        prev_position = position
    return text, inverses


# ---------------------------------------------------------------------------
# Manifest verification — the recorded spans are only as honest as these checks
# ---------------------------------------------------------------------------


def verify_symbol_spans(text: str, manifest: CorpusManifest) -> None:
    """Cross-check the recorded manifest against the text it describes.

    B3's generator doctrine: the oracle rests on recorded indices, so the
    indices are verified — every symbol's first line must match its
    language's signature shape at column 0 and carry the symbol's name,
    every last line must be the language's closer (or an indented body line
    for Python), symbols must be ordered, blank-line separated and unique,
    nothing but blank lines may follow the last symbol, and no symbol may
    exceed :data:`MAX_SYMBOL_LINES` (the chunk locator's parent-snap cap).
    """
    lines = text.splitlines()
    spec = _spec_of(manifest.language)
    seen: set[str] = set()
    prev_end = 0
    for sym in manifest.symbols:
        assert sym.name not in seen, f"duplicate symbol name {sym.name!r}"
        seen.add(sym.name)
        assert 1 <= sym.start_line <= sym.end_line <= len(lines), (
            f"{sym.name}: span {sym.start_line}-{sym.end_line} out of range "
            f"for a {len(lines)}-line text"
        )
        assert sym.line_count <= MAX_SYMBOL_LINES, (
            f"{sym.name}: {sym.line_count} lines exceeds the "
            f"{MAX_SYMBOL_LINES}-line parent-snap cap"
        )
        first = lines[sym.start_line - 1]
        assert re.match(
            spec.signature_re.format(name=re.escape(sym.name)), first,
        ), (
            f"{sym.name}: line {sym.start_line} is {first!r}, expected a "
            f"signature matching {spec.signature_re.format(name=sym.name)!r}"
        )
        last = lines[sym.end_line - 1]
        if spec.closer is None:
            assert last.startswith(manifest.indent) and last.strip(), (
                f"{sym.name}: last line {sym.end_line} is {last!r}, expected "
                f"an indented body line"
            )
        else:
            assert last == spec.closer, (
                f"{sym.name}: last line {sym.end_line} is {last!r}, expected "
                f"{spec.closer!r}"
            )
        if prev_end:
            between = lines[prev_end : sym.start_line - 1]
            assert all(line.strip() == "" for line in between), (
                f"{sym.name}: non-blank lines between symbols at "
                f"{prev_end + 1}-{sym.start_line - 1}"
            )
        prev_end = sym.end_line
    assert all(line.strip() == "" for line in lines[prev_end:]), (
        "non-blank content after the last recorded symbol"
    )


def verify_recipe_structures(text: str, manifest: CorpusManifest) -> None:
    """Cross-check the C3 recipe structures the manifest RECORDS against the
    text they describe — the recorded-indices doctrine extended to the
    adversarial layouts. A drifted renderer must fail loudly here, before
    any pipeline run can pin wrong behavior on a stale corpus.

    * ``deep`` — every recorded sub-block span sits inside its symbol, in
      order, its first line opens with a ``for`` block opener and (on brace
      languages) its last line is exactly the language's closer;
    * ``boilerplate`` — no function-kind body line carries its own symbol's
      name (the duplicated-line trap: body lines cannot identify the symbol
      they belong to, so only the manifest span + a byte-exact golden can
      prove the edit landed on the right occurrence);
    * ``cjk_seams`` — every function's first and last body lines carry CJK
      text adjacent to the chunk boundary, and no U+FFFD crept in;
    * ``prefix`` — every ``<base>_data_v2`` member has its ``<base>`` and
      ``<base>_data`` siblings recorded;
    * ``import_lines`` (non-plain recipes) — the recorded region's lines all
      carry import syntax and sit before the first symbol.
    """
    lines = text.splitlines()
    spec = _spec_of(manifest.language)
    by_name = {sym.name: sym for sym in manifest.symbols}

    for sym in manifest.symbols:
        for block in sym.blocks:
            assert sym.start_line <= block.start_line <= block.end_line <= sym.end_line, (
                f"{sym.name}: block {block.name} span {block.start_line}-"
                f"{block.end_line} escapes the symbol span {sym.start_line}-"
                f"{sym.end_line}"
            )
            opener = lines[block.start_line - 1]
            assert "for " in opener, (
                f"{block.name}: recorded block opener {opener!r} is not a "
                f"for-block"
            )
            if spec.closer is not None:
                assert lines[block.end_line - 1].strip() == spec.closer, (
                    f"{block.name}: recorded block last line "
                    f"{lines[block.end_line - 1]!r} != closer {spec.closer!r}"
                )

    if manifest.recipe == RECIPE_DEEP:
        # The deep-recipe line-distinctness doctrine (C3 seams stress): a
        # block's lines must be distinct WITHIN the block — a near-identical
        # template repeated 16x makes the real model lossily compress its
        # copy (mangled comment tails, dropped closers) and the validator
        # rightly rejects every attempt, so the edit can never converge —
        # while the blocks stay IDENTICAL to one another (one shared opener),
        # the B24 anchor tie that keeps the deterministic editor off the
        # narrowed edit so the real model runs.
        for sym in manifest.symbols:
            if not sym.blocks:
                continue
            openers = {lines[b.start_line - 1] for b in sym.blocks}
            assert len(openers) == 1, (
                f"{sym.name}: deep blocks must share ONE opener line (the "
                f"B24 tie), got {sorted(openers)!r}"
            )
            for block in sym.blocks:
                block_lines = lines[block.start_line - 1: block.end_line]
                assert len(set(block_lines)) == len(block_lines), (
                    f"{block.name}: a deep block repeats a line within "
                    f"itself — the real model lossily compresses repetitive "
                    f"chunks and the edit can never converge"
                )
                if spec.closer is not None:
                    # Brace languages: the block's layout must be
                    # indentation-CONSISTENT — a statement inside an open
                    # `{` sits deeper than the opener's indent and a closer
                    # sits at its opener's indent. The broken layout (an
                    # if-closer emitted after sibling-level statements) made
                    # the real model "helpfully" re-indent the copy and the
                    # validator rightly rejected every attempt (C3 seams
                    # stress).
                    stack: list[int] = []
                    for raw in block_lines:
                        stripped = raw.rstrip()
                        width = len(stripped) - len(stripped.lstrip())
                        if stripped.endswith("{"):
                            assert not stack or width > stack[-1], (
                                f"{block.name}: opener {stripped!r} is not "
                                f"indented deeper than its enclosing block"
                            )
                            stack.append(width)
                        elif stripped.lstrip().startswith("}"):
                            assert stack and width == stack[-1], (
                                f"{block.name}: closer {stripped!r} does not "
                                f"close the innermost open block at its own "
                                f"indent"
                            )
                            stack.pop()
                        else:
                            assert stack and width > stack[-1], (
                                f"{block.name}: statement {stripped!r} is "
                                f"not indented deeper than its enclosing "
                                f"block opener"
                            )
                    assert not stack, (
                        f"{block.name}: unbalanced braces after the recorded "
                        f"block span"
                    )

    if manifest.recipe == RECIPE_BOILERPLATE:
        bodies: list[tuple[str, ...]] = []
        for sym in manifest.symbols:
            if sym.kind != "function":
                continue
            body = lines[sym.start_line : sym.end_line]
            assert body and all(sym.name not in line for line in body), (
                f"{sym.name}: a boilerplate body line embeds the symbol name "
                f"— the duplicated-line trap is defeated"
            )
            assert len(set(body)) == len(body), (
                f"{sym.name}: a boilerplate body repeats a line within "
                f"itself — the real model lossily compresses repetitive "
                f"chunks and the edit can never converge"
            )
            bodies.append((sym.name, tuple(body)))
        # Every body is the shared assignment sequence's first N lines plus
        # a shared closing tail (the ``return``/closer lines): the
        # duplicated-line trap in its strongest form — body bytes cannot
        # identify the symbol they belong to.
        min_len = min(len(body) for _name, body in bodies)
        first_body = bodies[0][1]
        shared_suffix = 0
        while shared_suffix < min_len and all(
            body[-1 - shared_suffix] == first_body[-1 - shared_suffix]
            for _name, body in bodies
        ):
            shared_suffix += 1
        assert shared_suffix, "boilerplate bodies share no closing tail"
        cores = [body[: len(body) - shared_suffix] for _name, body in bodies]
        longest_core = max(cores, key=len)
        for (name, _body), core in zip(bodies, cores):
            assert longest_core[: len(core)] == core, (
                f"{name}: boilerplate body is not a prefix-plus-closer of "
                f"the shared body sequence: {_body[:2]}"
            )
        assert len({body for _name, body in bodies}) < len(bodies), (
            "the boilerplate corpus carries no duplicated bodies — the "
            "duplicated-line trap is absent"
        )

    if manifest.recipe == RECIPE_CJK_SEAMS:
        for sym in manifest.symbols:
            if sym.kind != "function":
                continue
            body = lines[sym.start_line : sym.end_line]
            assert body, f"{sym.name}: no body lines"
            if spec.closer is not None:
                assert body[-1].strip() == spec.closer
                body = body[:-1]
            assert "中文" in body[0] and "中文" in body[-1], (
                f"{sym.name}: CJK lines not adjacent to both chunk "
                f"boundaries: first={body[0]!r} last={body[-1]!r}"
            )
        assert "\ufffd" not in text, "U+FFFD replacement character in corpus"

    if manifest.recipe == RECIPE_PREFIX:
        for sym in manifest.symbols:
            if sym.kind != "function" or not sym.name.endswith("_data_v2"):
                continue
            base = sym.name[: -len("_data_v2")]
            assert base in by_name and f"{base}_data" in by_name, (
                f"{sym.name}: prefix family incomplete (missing {base!r} or "
                f"{base + '_data'!r})"
            )

    if manifest.recipe == RECIPE_TEXT:
        # The text dialect's log-entry doctrine (Step D2): each paragraph
        # is a shift-log ENTRY whose ENTRY LINE is the paragraph's one
        # file-wide-unique line (the D2 anchor matcher anchors windows on
        # it); entry numbers must match the manifest spans. Body sentences
        # come from a fixed natural template pool and MAY repeat across
        # distant entries — but the stride-block draw makes every
        # WINDOW-sized run of consecutive paragraphs repeat-free (see
        # _TEXT_TEMPLATE_STRIDE): the real model lossily compresses a chunk
        # that repeats a line within itself and the edit can never converge
        # (the C3 distinctness doctrine applies to the model's WINDOW; see
        # chunk_locator._MAX_TEXT_CHUNK_LINES).
        entry_ids: set[str] = set()
        for sym in manifest.symbols:
            first = lines[sym.start_line - 1]
            match = re.match(
                r"^2024-\d\d-\d\d \d\d:\d\d — shift log entry (\d+) opened",
                first,
            )
            assert match, (
                f"{sym.name}: first entry line is {first!r}, expected the "
                f"timestamped log-entry shape"
            )
            entry_id = match.group(1)
            assert entry_id not in entry_ids, (
                f"{sym.name}: duplicate entry ID {entry_id!r} — the D2 "
                f"anchor matcher needs unique entry lines"
            )
            entry_ids.add(entry_id)
            assert entry_id == sym.name.rsplit("_", 1)[-1], (
                f"{sym.name}: entry ID {entry_id!r} does not match the "
                f"recorded span name"
            )
        # THE window-distinctness invariant (measured load-bearing): the
        # maximal single-anchor window the D2 locator can produce around
        # any entry (± the locator's half-budget; chunk_locator
        # _MAX_TEXT_CHUNK_LINES = 40 → ±19 lines) never repeats a
        # non-blank line. A drifted renderer that reintroduces window
        # repeats fails loudly HERE instead of as a real-model
        # non-convergence minutes into the llm tier.
        half_window = _TEXT_WINDOW_SPAN_LINES // 2
        for sym in manifest.symbols:
            lo = max(0, sym.start_line - 1 - half_window)
            hi = min(len(lines), sym.start_line - 1 + half_window + 1)
            window = [ln.strip() for ln in lines[lo:hi] if ln.strip()]
            repeats = {ln for ln in window if window.count(ln) > 1}
            assert not repeats, (
                f"{sym.name}: the ±{half_window}-line window around its "
                f"entry repeats {len(repeats)} line(s) — the real model "
                f"lossily compresses repetitive chunks and the edit can "
                f"never converge: {sorted(repeats)[:3]}"
            )
        assert "\ufffd" not in text, "U+FFFD replacement character in corpus"

    if manifest.import_lines is not None:
        first, last = manifest.import_lines
        assert 1 <= first <= last < manifest.symbols[0].start_line, (
            f"import region {first}-{last} not before the first symbol at "
            f"{manifest.symbols[0].start_line}"
        )
        for line_no in range(first, last + 1):
            line = lines[line_no - 1]
            assert "import" in line or line.lstrip().startswith("use "), (
                f"line {line_no} in the recorded import region carries no "
                f"import syntax: {line!r}"
            )


# ---------------------------------------------------------------------------
# Manifest (de)serialization — the op records ARE the golden manifest ops
# ---------------------------------------------------------------------------


def line_op_to_manifest(line_op: LineOp) -> dict:
    return {
        "kind": line_op.kind,
        "start_line": line_op.start_line,
        "end_line": line_op.end_line,
        "after_line": line_op.after_line,
        "text": line_op.text,
    }


def line_op_from_manifest(data: dict) -> LineOp:
    return LineOp(
        kind=data["kind"],
        start_line=data["start_line"],
        end_line=data["end_line"],
        after_line=data["after_line"],
        text=data["text"],
    )


def op_to_manifest(op: CorpusOp, expected: str | None = None) -> dict:
    """Serialize one op (payload + recorded oracle data + inverse data)."""
    return {
        "op": op.kind,
        "symbol": op.symbol,
        "language": op.language,
        "ext": op.ext,
        "eol": op.eol,
        "final_eol": op.final_eol,
        "note": op.note,
        "payload": op.new_text,
        "oracle": {
            "anchor_start_line": op.anchor_start_line,
            "anchor_end_line": op.anchor_end_line,
            "anchor_last_line": op.anchor_last_line,
            "anchor_is_last_line": op.anchor_is_last_line,
            "start_line": op.start_line,
            "end_line": op.end_line,
            "consumed_end_line": op.consumed_end_line,
            "original_span_text": op.original_span_text,
            "sep_before": op.sep_before,
            "sep_after": op.sep_after,
            "snippet_line_count": op.snippet_line_count,
        },
        "expected": expected,
        "inverse": line_op_to_manifest(inverse_op(op)),
    }


def op_from_manifest(data: dict) -> CorpusOp:
    """Rebuild an op from its manifest serialization (payload + oracle block)."""
    oracle = data["oracle"]
    return CorpusOp(
        kind=data["op"],
        symbol=data["symbol"],
        language=data["language"],
        ext=data["ext"],
        eol=data["eol"],
        final_eol=data["final_eol"],
        note=data.get("note", ""),
        new_text=data.get("payload"),
        anchor_start_line=oracle["anchor_start_line"],
        anchor_end_line=oracle["anchor_end_line"],
        anchor_last_line=oracle["anchor_last_line"],
        anchor_is_last_line=oracle["anchor_is_last_line"],
        start_line=oracle["start_line"],
        end_line=oracle["end_line"],
        consumed_end_line=oracle["consumed_end_line"],
        original_span_text=oracle["original_span_text"],
        sep_before=oracle["sep_before"],
        sep_after=oracle["sep_after"],
        snippet_line_count=oracle["snippet_line_count"],
    )


# ---------------------------------------------------------------------------
# Committed medium goldens (tests/golden/big/) — pinned recipes + writer
# ---------------------------------------------------------------------------

GOLDEN_BIG_CASES: tuple[dict, ...] = (
    {
        "case": "python_crlf", "language": "python", "eol": EOL_CRLF,
        "final_eol": True, "indent": "    ", "sprinkle_cjk": False,
        "seed": 20260214, "target_bytes": 1_048_576,
    },
    {
        "case": "rust_noeol", "language": "rust", "eol": EOL_LF,
        "final_eol": False, "indent": "    ", "sprinkle_cjk": False,
        "seed": 20260215, "target_bytes": 1_048_576,
    },
    {
        "case": "go_cjk_tabs", "language": "go", "eol": EOL_LF,
        "final_eol": True, "indent": "\t", "sprinkle_cjk": True,
        "seed": 20260216, "target_bytes": 1_048_576,
    },
)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "symbol"


def expected_filename(op: CorpusOp) -> str:
    return f"expected_{op.kind}_{_sanitize(op.symbol)}.{op.ext}"


def case_manifest(case: CorpusCase, case_name: str) -> dict:
    """The ``fastedit-corpus-golden/1`` manifest for one materialized case."""
    manifest = case.source.manifest
    return {
        "schema": CORPUS_GOLDEN_SCHEMA,
        "case": case_name,
        "language": case.language,
        "ext": case.ext,
        "filename": f"original.{case.ext}",
        "eol": case.eol,
        "final_eol": case.final_eol,
        "indent": case.indent,
        "sprinkle_cjk": case.sprinkle_cjk,
        "seed": case.seed,
        "target_bytes": case.target_bytes,
        "actual_bytes": len(case.source.encode("utf-8")),
        "symbol_count": manifest.symbol_count,
        "max_symbol_lines": manifest.max_symbol_lines,
        "oracle": (
            "tests/corpus.py apply_op_oracle — independent line-splice "
            "arithmetic on generator-recorded indices; never imports "
            "fastedit, never parses code"
        ),
        "regenerate": "uv run python tests/golden/big/_generate.py",
        "recipe": case.build_kwargs(),
        "ops": [op_to_manifest(op, expected_filename(op)) for op in case.ops],
    }


def write_golden_big_cases(
    base_dir: Path, cases: tuple[dict, ...] = GOLDEN_BIG_CASES,
) -> list[Path]:
    """Materialize the medium corpus goldens (original + expected + manifest)."""
    written: list[Path] = []
    for entry in cases:
        case = build_corpus_case(**{
            key: value for key, value in entry.items() if key != "case"
        })
        case_dir = base_dir / entry["case"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"original.{case.ext}").write_bytes(
            case.source.encode("utf-8"),
        )
        for op in case.ops:
            (case_dir / expected_filename(op)).write_bytes(
                apply_op_oracle(str(case.source), op).encode("utf-8"),
            )
        (case_dir / "manifest.json").write_text(
            json.dumps(case_manifest(case, entry["case"]), indent=2,
                       ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(case_dir)
        print(
            f"{entry['case']}: {len(case.ops)} corpus op(s), "
            f"{len(case.source.encode('utf-8'))} bytes, "
            f"{case.source.manifest.symbol_count} symbols, "
            f"max symbol {case.source.manifest.max_symbol_lines} lines",
        )
    return written


# ---------------------------------------------------------------------------
# CLI: build one corpus / benchmark the 100MB build (report numbers)
# ---------------------------------------------------------------------------


def _build_from_args(args: argparse.Namespace) -> CorpusSource:
    return generate_big_source(
        args.language,
        args.target_bytes,
        eol=args.eol,
        final_eol=not args.no_final_eol,
        indent=args.indent,
        seed=args.seed,
        sprinkle_cjk=args.sprinkle_cjk,
        recipe=args.recipe,
    )


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a seeded corpus (tests/corpus.py). Default prints a "
            "summary; --bench reports build time and peak memory."
        ),
    )
    parser.add_argument("--language", default="python", choices=sorted(_SPECS))
    parser.add_argument("--target-bytes", type=int, default=100_000_000)
    parser.add_argument("--eol", default=EOL_LF, choices=[EOL_LF, EOL_CRLF])
    parser.add_argument("--no-final-eol", action="store_true")
    parser.add_argument("--indent", default=None)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--sprinkle-cjk", action="store_true")
    parser.add_argument("--recipe", default=RECIPE_PLAIN, choices=list(RECIPES))
    parser.add_argument(
        "--bench", action="store_true",
        help="run the build twice (once under tracemalloc) and report",
    )
    args = parser.parse_args(argv)

    if not args.bench:
        source = _build_from_args(args)
        print(
            f"{args.language} corpus: "
            f"{len(source.encode('utf-8'))} bytes, "
            f"{source.manifest.symbol_count} symbols, "
            f"max symbol {source.manifest.max_symbol_lines} lines, "
            f"eol={args.eol!r}, final_eol={not args.no_final_eol}, "
            f"cjk={args.sprinkle_cjk}, recipe={args.recipe!r}",
        )
        return

    # Warm-up build: pins the size/symbol numbers and (after release + a
    # gc pass) leaves a clean heap for the traced run below, so the traced
    # peak describes ONE build, not two overlapping ones.
    warmup = _build_from_args(args)
    size = len(warmup.encode("utf-8"))
    symbol_count = warmup.manifest.symbol_count
    max_lines = warmup.manifest.max_symbol_lines
    del warmup
    gc.collect()

    start = time.perf_counter()
    source = _build_from_args(args)
    seconds = time.perf_counter() - start
    del source
    gc.collect()

    tracemalloc.start()
    traced_start = time.perf_counter()
    _build_from_args(args)
    traced_seconds = time.perf_counter() - traced_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # ru_maxrss is bytes on macOS, KiB on Linux (peak watermark for the whole
    # process, so it also covers the warm-up build's allocations).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_unit = "bytes" if sys.platform == "darwin" else "KiB"
    print(
        f"{args.language} corpus: {size} bytes ({size / (1 << 20):.1f} MiB), "
        f"{symbol_count} symbols, max symbol {max_lines} lines\n"
        f"build time: {seconds:.2f}s "
        f"({size / seconds / (1 << 20):.1f} MiB/s)\n"
        f"traced build: {traced_seconds:.2f}s (tracemalloc overhead included), "
        f"peak Python allocation: {peak / (1 << 20):.1f} MiB "
        f"({peak / size:.2f}x corpus size)\n"
        f"process peak RSS: {rss} {rss_unit}",
    )


if __name__ == "__main__":
    _main()
