"""Step C3 — adversarial chunk-boundary stress at 100MB (plan Phase C).

MISSION: the seams are where corruption lives — chunks are cut on ORIGINAL
line indexing and spliced in reverse order (``chunked_merge``'s assembly
loop) — so this suite places the danger exactly there. Real model only (the
trained fastedit mlx-8bit weights via :func:`llm_fixtures.real_engine`),
per plan req. 1; every case is marked ``llm`` and ``stress`` and gated on
``FASTEDIT_RUN_STRESS=1`` (plan §0 runtime tiers).

Corpus: the C3 adversarial recipes of ``tests/corpus.py`` (``recipe=``
kwarg), one ~100MB source per (recipe, language), built lazily and released
after each case (one source alive at a time — the C2 memory discipline):

* ``boilerplate`` — every function shares one byte-identical body pool
  (duplicated-line trap: body bytes cannot identify the symbol they belong
  to). LF.
* ``cjk_seams`` — Chinese text ADJACENT to both chunk boundaries of every
  function (CJK docstring / first-body comment + trailing comment on the
  closing ``return``). CRLF — the multibyte × line-ending × seam
  interaction in one corpus.
* ``deep`` — every 23rd symbol is a >100-line function of six identical
  ``for`` blocks (recorded sub-block spans in :attr:`SymbolSpan.blocks`);
  a marker snippet drives the chunk locator's ``_narrow_large_node``
  windowing path and the expected cut is computable from the manifest. LF.
* ``prefix`` — every 11th index emits ``<base>`` / ``<base>_data`` /
  ``<base>_data_v2`` (resolution ambiguity at scale). LF.

Cases (per language: python + go + typescript):

a. **seam batch** (boilerplate) — two wrap edits in ONE ``fast_batch_edit``
   call: a symbol whose chunk boundary TOUCHES a >90-line monster (the
   seam) and one in a flat region (away). ``chunks_used == 2``, the model
   only ever saw the two manifest spans (recorded chunk texts asserted
   byte-exact), untouched regions byte-identical, composed golden
   byte-exactness with outer attempts, backward reconstruction.
b. **CJK × CRLF × seam batch** (cjk_seams) — the same two-target batch on
   the CRLF corpus with CJK on both seam sides: zero bare-LF bytes, no
   U+FFFD, CJK bytes intact, golden byte-exact.
c. **prefix disambiguation batch** (prefix) — ``replace=<base>_data`` twice
   in one batch: the golden proves the MIDDLE symbol was edited and the
   ``<base>`` / ``<base>_data_v2`` siblings are byte-identical.
d. **deep-nesting narrowing** (deep) — ``replace=<deep symbol>`` with a
   marker snippet: ``_narrow_large_node`` must cut the manifest-recorded
   first block (asserted via ``result.chunk_regions``), the real model
   merges that block, golden byte-exact, backward reconstruction.
e. **multi-region reverse assembly** (cjk_seams; python + go) — ONE
   ``chunked_merge`` call whose snippet adds a header import AND wraps a
   symbol: the locator splits it into the recorded import region plus the
   code region, the assembly loop splices them in REVERSE order on original
   line indexing; golden byte-exact.
f. **undo (full stack) + backward reconstruction** — wrap → deterministic
   insert → ``fast_undo`` → ``fast_undo``: every state byte-exact at 100MB,
   and the oracle's inverse op regenerates the original from the REAL
   edited bytes. Run on three recipes across declared language instances
   (boilerplate and cjk_seams on python, deep on go); the wrap target sits
   PAST the deterministic insert's anchor (~25% of the corpus) so the
   oracle composes the two states directly on recorded indices.

Non-determinism policy (plan §0(b)): untouched-region byte-identity, op-spec
conformance, relative validity and FULL-FILE golden byte-exactness inside
``OUTER_ATTEMPTS`` complete re-edits; non-convergence is a product defect
(plan §0), reported with ``first_diff_tag``.

The golden oracle is C1's INDEPENDENT line-splice arithmetic on the
generator-recorded manifest spans (``tests/corpus.py``) — never fastedit,
never a parser. Per-language variance lives in declarative spec tables
(CLAUDE.md's no-hardcoding rule); the assertion logic has no per-language
branches.

Memory & runtime discipline: each (recipe, language) 100MB source is built
once on first use and released (cache pop + ``gc``) after its case — at
most one source alive. The recipe×language matrix cases (a)-(d) run through
the parametrized fixtures; the restricted-language cases (e)/(f) build their
declared instance through :func:`_seam_case` and release it in a ``finally``
block. Per-case wall times print as ``[C3 seams]`` lines (run pytest with
``-s``).

The op shapes are the ones MEASURED to converge against the real model per
language (the C2 doctrine — the variance lives in the shape tables, never in
assertion branches):

* python — the C2 with-suite wrap (``kept_prefix=1``: the docstring stays a
  declared context line outside the wrap).
* typescript — the C2 brace else-form wrap; on the CJK corpus with
  ``kept_prefix=1`` (the measured ts model drops a leading CJK comment it is
  asked to wrap, so the CJK corpus keeps it outside).
* go — the guard-prepend: the go model drops the ``} else {`` line of the
  else-form on every recipe, so its op declares the self-contained guard the
  model actually emits (``[sig, if{, return 0, }, MARKER]``).
* deep-narrow (d) — python/ts append a declared seam-probe line, go appends
  the measured gofmt guard form (blank separator, guard body one level
  deeper) plus the probe: the go model normalizes a mid-function guard
  toward gofmt and hallucinates past the chunk on a probe-only tail, so the
  op declares the bytes the model actually emits. The probe line carries the
  ellipsis phrase inside a STRING, which drives the locator's narrow without
  being a keep-marker.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import corpus
import pytest
from corpus import EOL_CRLF, EOL_LF
from llm_fixtures import (
    first_diff_tag,
    metrics_tag,
    require_stress_env,
    run_real_edit,
)
from test_stress_100mb_llm import (  # C2 helpers (mcp_harness rides via conftest)
    _assert_same_lines,
    _brace_wrap,
    _join,
    _py_wrap,
    _span_kept_lines,
    _StopWatch,
    _wrap_expected,
    _wrap_inverse,
    _wrap_snippet,
    _wrap_span_lines,
    _WrapSpec,
)

pytestmark = [pytest.mark.llm, pytest.mark.stress]

TARGET_BYTES = 100_000_000
"""The C3 stress size: ~100MB per source (plan req. 2)."""

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b));
non-convergence after the bound is a product defect, never flakiness."""

_DEFAULT_RETRY_BUDGET = 8
"""``chunked_merge._DEFAULT_MAX_VALIDATION_RETRIES`` — the per-site budget."""

_MONSTER_MIN_LINES = 90
"""A neighbour symbol this large makes its neighbour's chunk boundary a
SEAM (the C3 recipe corpora grow every 97th symbol to >= 90 lines)."""

_SEEDS = {
    corpus.RECIPE_BOILERPLATE: "c3-boilerplate-100mb",
    corpus.RECIPE_CJK_SEAMS: "c3-cjkseams-100mb",
    corpus.RECIPE_DEEP: "c3-deep-100mb",
    corpus.RECIPE_PREFIX: "c3-prefix-100mb",
}

# The deep-narrow tail each narrow shape appends, as (indent level relative
# to the block's base indent, line text) pairs; an empty text renders the
# BLANK separator line (the file's EOL alone). The ellipsis phrase rides
# inside a STRING so ``_snippet_has_ellipsis`` (the locator's narrow trigger)
# fires while ``markers.is_marker_line`` — an exact-match predicate — still
# classifies the line as CONTENT: the validator treats it as the declared new
# line it is and the model can emit it verbatim.
#
# The go shape is the MEASURED gofmt form (C3 seams stress): the real model
# normalizes a mid-function guard toward gofmt — it inserts a blank separator
# after the enclosing block's closer and indents the guard body one level
# deeper — and a probe-only tail makes it hallucinate a "seam 17" sequence
# continuation past the narrowed chunk (9/9 attempts rightly rejected). The
# declared tail is therefore exactly the bytes the model re-emits: measured
# byte-exact on every attempt at both deep targets (the content-faithfulness
# validator is deliberately content-level, so whitespace is unverifiable —
# the shape table must declare the model's natural form, the same doctrine
# as the C2 go guard-prepend).
_NARROW_TAIL: dict[str, tuple[tuple[int, str], ...]] = {
    "python": (
        (2, 'seam_probe = "... existing code ..."'),
    ),
    "go": (
        (0, ""),  # blank separator — the measured gofmt prior
        (1, "if valueA == 0 {"),
        (2, "return 0"),
        (1, "}"),
        (1, 'seamProbe := "... existing code ..."'),
    ),
    "typescript": (
        (1, 'const seamProbe = "... existing code ...";'),
    ),
}


def _narrow_tail_lines(language: str, base_indent: str, eol: str) -> list[str]:
    """Render one language's declared deep-narrow tail (each line WITH eol)."""
    return [
        eol if text == "" else base_indent * level + text + eol
        for level, text in _NARROW_TAIL[language]
    ]


# ---------------------------------------------------------------------------
# Declarative per-language configuration (the variance lives here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LanguageSeams:
    """One language's 100MB seam-stress recipe (seed-pinned, declarative)."""

    language: str
    ext: str
    wrap: Callable[[str], tuple[_WrapSpec, int]] | None
    """The C2 ``_WrapSpec``-based wrap (python's with-suite, typescript's
    else-form). ``None`` for go, whose measured shape is the guard-prepend
    below: the go model drops the ``} else {`` line of the else-form (its
    merge is rightly rejected 9/9), so its op declares the guard it actually
    emits."""
    guard_lines: tuple[str, ...] = ()
    """go's declared guard lines, prepended inside the symbol at body
    indent."""

    def spec(self, indent: str, kept_prefix: int | None = None) -> _WrapSpec:
        """The ``_WrapSpec`` for this language's wrap (fails loud for go)."""
        assert self.wrap is not None, (
            f"{self.language}: the seam op is the guard-prepend shape"
        )
        spec, _delta = self.wrap(indent)
        if kept_prefix is not None:
            return replace(spec, kept_prefix=kept_prefix)
        return spec


_LANG_CONFIG: dict[str, _LanguageSeams] = {
    "python": _LanguageSeams(
        language="python", ext="py", wrap=_py_wrap,
    ),
    "go": _LanguageSeams(
        language="go", ext="go", wrap=None,
        guard_lines=("if valueA > 0 {", "return 0", "}"),
    ),
    "typescript": _LanguageSeams(
        language="typescript", ext="ts",
        wrap=lambda indent: _brace_wrap("if (valueA > 0)", "return 0;"),
    ),
}
_LANGUAGES = tuple(_LANG_CONFIG)

_SEAM_MARKER = "// ... existing code ..."

# The CJK corpus's leading standalone CJK lines are declared as KEPT context
# (outside the wrap): the measured real models (python AND typescript) drop a
# standalone CJK comment they are asked to wrap, while mid-line CJK text (the
# trailing comment on the return line) survives wrapping. python's cjk
# functions carry a CJK docstring AND a CJK comment (kept_prefix=2);
# typescript carries one CJK comment (kept_prefix=1). go's guard-prepend
# wraps nothing.
_CJK_KEPT_PREFIX: dict[str, int] = {"python": 2, "typescript": 1}


def _seam_snippet(
    source: corpus.CorpusSource, span: corpus.SymbolSpan,
    kept_prefix: int | None = None,
) -> str:
    """The LLM-path snippet for one target (the per-language C3 shapes).

    go: ``[sig, guard(3), MARKER]`` — the model prepends the guard inside the
    symbol (its measured natural completion, golden-exact on every go
    recipe). python/typescript: the C2 ``_WrapSpec`` shapes.

    ``kept_prefix`` (typescript × cjk) declares the span's first body line as
    kept context: the measured ts model drops a leading CJK comment it is
    asked to wrap, so the CJK corpus keeps it outside the wrap.
    """
    manifest = source.manifest
    lines = source.splitlines(keepends=True)
    sig = lines[span.start_line - 1].rstrip("\r\n")
    if manifest.language == "go":
        ind = manifest.indent
        guard = _LANG_CONFIG["go"].guard_lines
        return "".join(
            [sig + manifest.eol]
            + [ind + g + manifest.eol for g in guard]
            + [ind + _SEAM_MARKER + manifest.eol]
        )
    spec = _LANG_CONFIG[manifest.language].spec(manifest.indent, kept_prefix)
    return _wrap_snippet(
        sig, _span_kept_lines(lines, span, spec),
        manifest.indent, spec, manifest.eol,
    )


def _seam_golden_lines(
    source: corpus.CorpusSource, span: corpus.SymbolSpan,
    kept_prefix: int | None = None,
) -> list[str]:
    """The replaced span's keepends lines — the independent golden splice."""
    manifest = source.manifest
    lines = source.splitlines(keepends=True)
    if manifest.language == "go":
        span_lines = [ln.rstrip("\r\n") for ln in
                      lines[span.start_line - 1: span.end_line]]
        ind = manifest.indent
        guard = [ind + g for g in _LANG_CONFIG["go"].guard_lines]
        return [ln + manifest.eol
                for ln in span_lines[:1] + guard + span_lines[1:]]
    spec = _LANG_CONFIG[manifest.language].spec(manifest.indent, kept_prefix)
    return _wrap_span_lines(
        lines[span.start_line - 1: span.end_line],
        manifest.indent, spec, manifest.eol,
    )


def _seam_line_delta(
    source: corpus.CorpusSource, span: corpus.SymbolSpan,
    kept_prefix: int | None = None,
) -> int:
    """The seam op's net line delta for one span."""
    return len(_seam_golden_lines(source, span, kept_prefix)) - (
        span.end_line - span.start_line + 1
    )


def _seam_declared_new(
    source: corpus.CorpusSource, kept_prefix: int | None = None,
) -> tuple[str, ...]:
    """The declared new (non-marker) lines of the language's seam op."""
    manifest = source.manifest
    ind = manifest.indent
    if manifest.language == "go":
        return tuple(ind + g for g in _LANG_CONFIG["go"].guard_lines)
    spec = _LANG_CONFIG[manifest.language].spec(manifest.indent, kept_prefix)
    declared = [spec.opener(ind)]
    for field in (spec.else_opener, spec.else_body, spec.else_closer):
        if field is not None:
            declared.append(field(ind))
    return tuple(declared)

# Corpus EOL per recipe: the CJK recipe is CRLF (the multibyte × EOL × seam
# interaction), everything else LF.
_RECIPE_EOL = {
    corpus.RECIPE_BOILERPLATE: EOL_LF,
    corpus.RECIPE_CJK_SEAMS: EOL_CRLF,
    corpus.RECIPE_DEEP: EOL_LF,
    corpus.RECIPE_PREFIX: EOL_LF,
}


# ---------------------------------------------------------------------------
# Corpus fixtures — one 100MB source alive at a time
# ---------------------------------------------------------------------------

_CASE_CACHE: dict[tuple[str, str], corpus.CorpusSource] = {}


def _seam_source(recipe: str, language: str) -> corpus.CorpusSource:
    key = (recipe, language)
    if key not in _CASE_CACHE:
        with _StopWatch(f"build 100MB corpus {language}/{recipe}"):
            _CASE_CACHE[key] = corpus.generate_big_source(
                language, TARGET_BYTES,
                eol=_RECIPE_EOL[recipe], seed=_SEEDS[recipe],
                sprinkle_cjk=(recipe == corpus.RECIPE_CJK_SEAMS),
                recipe=recipe,
            )
    return _CASE_CACHE[key]


def _seam_fixture(recipe: str):
    """A function-scoped parametrized fixture over the three languages.

    Function-scoped (not module-scoped) so the cache pop in the teardown
    keeps at most ONE 100MB source alive across the whole module run: the
    ~1s rebuild per case is far cheaper than the memory.
    """

    @pytest.fixture(params=[
        pytest.param(lang, id=lang) for lang in _LANGUAGES
    ])
    def _fixture(request):
        require_stress_env()  # belt-and-braces: skip before building anything
        language = request.param
        yield _seam_source(recipe, language)
        _CASE_CACHE.pop((recipe, language), None)
        gc.collect()

    return _fixture


boilerplate_case = _seam_fixture(corpus.RECIPE_BOILERPLATE)
cjk_case = _seam_fixture(corpus.RECIPE_CJK_SEAMS)
deep_case = _seam_fixture(corpus.RECIPE_DEEP)
prefix_case = _seam_fixture(corpus.RECIPE_PREFIX)


@contextlib.contextmanager
def _seam_case(recipe: str, language: str):
    """A restricted-language case: build its (recipe, language) source, run,
    release — the one-source-alive discipline, explicit per case.

    The recipe×language matrix cases (a)-(d) use the parametrized fixtures
    above; cases (e)/(f) run on DECLARED instances only (e.g. the undo case
    on python), so they build exactly the instance they need instead of
    skipping the other fixture params.
    """
    require_stress_env()  # skip before building anything
    try:
        yield _seam_source(recipe, language)
    finally:
        _CASE_CACHE.pop((recipe, language), None)
        gc.collect()


# ---------------------------------------------------------------------------
# Target selection — deterministic manifest scans, no cherry-picking
# ---------------------------------------------------------------------------


def _seam_target(
    source: corpus.CorpusSource, min_fraction: float = 0.0,
) -> corpus.SymbolSpan:
    """The first function symbol whose NEXT neighbour is a monster.

    Its chunk END boundary touches the monster's span — the seam. Scanning
    starts past ``min_fraction`` of the corpus (the undo cases need a wrap
    target PAST the deterministic insert's ~25% anchor so the oracle
    composes both states on unshifted recorded indices).
    """
    manifest = source.manifest
    start = int(manifest.symbol_count * min_fraction)
    for i in range(start, len(manifest.symbols) - 1):
        sym = manifest.symbols[i]
        nxt = manifest.symbols[i + 1]
        if (
            sym.kind == "function"
            and sym.line_count <= 100  # below the narrow trigger
            and nxt.line_count >= _MONSTER_MIN_LINES
        ):
            return sym
    raise AssertionError(
        f"{manifest.language}/{manifest.recipe}: no seam-adjacent target "
        f"past fraction {min_fraction}"
    )


def _away_target(
    source: corpus.CorpusSource, exclude: frozenset[str],
) -> corpus.SymbolSpan:
    """The first function symbol past 55% whose neighbours are both small.

    A flat region: no monster chunk boundary anywhere near.
    """
    manifest = source.manifest
    start = int(manifest.symbol_count * 0.55)
    for i in range(start, len(manifest.symbols)):
        sym = manifest.symbols[i]
        if sym.kind != "function" or sym.line_count > 100 or sym.name in exclude:
            continue
        prev_ok = i == 0 or manifest.symbols[i - 1].line_count < _MONSTER_MIN_LINES
        next_ok = manifest.symbols[i + 1].line_count < _MONSTER_MIN_LINES
        if prev_ok and next_ok:
            return sym
    raise AssertionError(
        f"{manifest.language}/{manifest.recipe}: no away target"
    )


def _prefix_targets(source: corpus.CorpusSource) -> tuple[
    corpus.SymbolSpan, corpus.SymbolSpan,
]:
    """Two ``<base>_data`` middle symbols (deterministic forward scans)."""
    manifest = source.manifest
    middles = [
        sym for sym in manifest.symbols
        if sym.kind == "function" and sym.name.endswith("_data")
    ]
    assert middles, (
        f"{manifest.language}/{manifest.recipe}: no prefix families"
    )
    first_cut, second_cut = int(len(middles) * 0.25), int(len(middles) * 0.65)
    first = middles[first_cut]
    second = next(
        sym for sym in middles[second_cut:] if sym.name != first.name
    )
    for sym in (first, second):
        base = sym.name[: -len("_data")]
        manifest.span(base)      # fail loud if the family is incomplete
        manifest.span(f"{base}_data_v2")
    return first, second


def _deep_target(
    source: corpus.CorpusSource, min_fraction: float = 0.0,
) -> corpus.SymbolSpan:
    """The first deep-nested symbol past ``min_fraction`` (recorded blocks
    drive the narrow)."""
    manifest = source.manifest
    start = int(manifest.symbol_count * min_fraction)
    for sym in manifest.symbols[start:]:
        if sym.blocks:
            return sym
    raise AssertionError(
        f"{manifest.language}/{manifest.recipe}: no deep symbol past "
        f"fraction {min_fraction}"
    )


def _wrap_record(
    source: corpus.CorpusSource, span: corpus.SymbolSpan,
) -> corpus.CorpusOp:
    """The recorded original-span bytes for one wrap target (inverse data)."""
    original_span_text = _join(source.splitlines(keepends=True),
                               span.start_line, span.end_line)
    assert original_span_text == "".join(
        source.splitlines(keepends=True)[span.start_line - 1: span.end_line],
    )
    return corpus.CorpusOp(
        kind="replace_symbol_body",
        symbol=span.name,
        language=source.manifest.language,
        ext=source.manifest.ext,
        eol=source.manifest.eol,
        final_eol=source.manifest.final_eol,
        start_line=span.start_line,
        end_line=span.end_line,
        original_span_text=original_span_text,
    )


# ---------------------------------------------------------------------------
# Shared batch harness — two wrap edits in one MCP fast_batch_edit call
# ---------------------------------------------------------------------------


def _batch_edits_json(
    source: corpus.CorpusSource, spans: list[corpus.SymbolSpan],
    kept_prefix: int | None = None,
) -> str:
    source_lines = source.splitlines(keepends=True)
    del source_lines  # the builders split the source themselves
    edits = []
    for span in spans:
        edits.append({
            "snippet": _seam_snippet(source, span, kept_prefix),
            "replace": span.name,
        })
    return json.dumps(edits)


def _composed_batch_golden(
    source: corpus.CorpusSource, spans: list[corpus.SymbolSpan],
    kept_prefix: int | None = None,
) -> tuple[str, list[int]]:
    """The independent composed golden for two seam edits, in order.

    Returns (merged, wrapped_line_lengths) — pure line-splice arithmetic on
    the manifest spans. ``batch_chunked_merge`` applies each edit to the
    PREVIOUS edit's output (edit 2's ``replace=`` resolves its target at the
    position the earlier edit left it at), so each span is applied at its
    position in the CURRENT text: spans below an earlier edit shift by that
    edit's line delta — the same composition C2's multi-chunk golden uses.
    """
    merged = str(source)
    wrapped_lengths: list[int] = []
    shift = 0
    for span in spans:
        golden_lines = _seam_golden_lines(source, span, kept_prefix)
        target = span
        if shift:
            target = corpus.SymbolSpan(
                name=span.name, kind=span.kind,
                start_line=span.start_line + shift,
                end_line=span.end_line + shift,
            )
        lines = merged.splitlines(keepends=True)
        merged = "".join(
            lines[: target.start_line - 1]
            + golden_lines
            + lines[target.end_line:]
        )
        wrapped_lengths.append(len(golden_lines))
        shift += len(golden_lines) - (span.end_line - span.start_line + 1)
    return merged, wrapped_lengths


def _assert_seam_op_spec(
    source: corpus.CorpusSource, merged: str, span: corpus.SymbolSpan,
    declared_count: int, merged_shift: int, kept_prefix: int | None,
) -> int:
    """Op-spec conformance of one seam edit; returns the wrapped length.

    The op's declared new lines appear exactly ``declared_count`` times
    file-wide (the declared shapes are unique in the corpus), and the edited
    span — at its position in the MERGED text — equals the independently
    spliced golden span. The composed golden carries the whole-file proof:
    for the seam recipes a file-wide survivor-line census is either
    meaningless (the boilerplate bodies are byte-identical across symbols —
    the duplicated-line trap) or subsumed by the golden (prefix/cjk embed
    the symbol name).
    """
    manifest = source.manifest
    golden_lines = _seam_golden_lines(source, span, kept_prefix)
    merged_lines = merged.splitlines(keepends=True)
    for declared in _seam_declared_new(source, kept_prefix):
        assert merged_lines.count(declared + manifest.eol) == declared_count, (
            f"declared line {declared!r} not present exactly "
            f"{declared_count} time(s)"
        )
    start = span.start_line + merged_shift
    got = merged_lines[start - 1: start - 1 + len(golden_lines)]
    assert got == golden_lines, (
        f"the edited span at merged lines {start}-"
        f"{start + len(golden_lines) - 1} is not the expected wrapped span "
        f"(wrong occurrence or wrong shape): "
        f"{first_diff_tag(''.join(golden_lines), ''.join(got))}"
    )
    return len(golden_lines)


def _shift_for(line: int, edits: list[tuple[corpus.SymbolSpan, int]]) -> int:
    """Total line shift at ``line`` after every earlier edit is applied.

    An edit's wrapped span adds ``delta`` lines; everything BELOW the
    edited span moves down by ``delta``. Used to read a sibling span's
    bytes at its position in the MERGED text.
    """
    return sum(delta for span, delta in edits if span.end_line < line)


def _run_batch_case(
    source: corpus.CorpusSource, tmp_path, mcp_harness,
    spans: list[corpus.SymbolSpan], label: str,
    kept_prefix: int | None = None, extra_assert=None,
) -> None:
    """One two-target batch case: regions, goldens, untouched identity.

    ``kept_prefix`` (typescript × cjk) declares the span's first body line as
    kept context — the measured ts model drops a leading CJK comment it is
    asked to wrap.
    """
    from fastedit.mcp import tools_edit

    case_text = str(source)
    manifest = source.manifest
    source_lines = case_text.splitlines(keepends=True)
    original_bytes = case_text.encode("utf-8")
    edits_json = _batch_edits_json(source, spans, kept_prefix)
    span_texts = [
        _join(source_lines, span.start_line, span.end_line) for span in spans
    ]

    golden, wrapped_lengths = _composed_batch_golden(source, spans, kept_prefix)
    # The composed golden's per-edit line deltas, for untouched-region math.
    first_delta = _seam_line_delta(source, spans[0], kept_prefix)
    second_delta = _seam_line_delta(source, spans[1], kept_prefix)
    span_b, span_c = spans

    target = tmp_path / f"corpus_100mb_seams.{manifest.ext}"

    def check(message: str, merged: str) -> None:
        m = f"response={message!r}"
        merged_lines = merged.splitlines(keepends=True)
        assert message.startswith(f"Applied 2 edits to {target}"), m
        assert "2 chunk(s), 2 edit(s)" in message, m
        assert "rejected" not in message and "Error" not in message, m

        # The model saw EXACTLY the two declared manifest spans.
        seen = list(dict.fromkeys(mcp_harness.recorded_chunks))
        assert len(seen) == 2, (
            f"expected two distinct chunk texts, got {len(seen)}: {m}"
        )
        assert seen[0] == span_texts[0], (
            f"chunk 1 is not the manifest span of '{span_b.name}': "
            f"{first_diff_tag(span_texts[0], seen[0])}"
        )
        assert seen[1] == span_texts[1], (
            f"chunk 2 is not the manifest span of '{span_c.name}': "
            f"{first_diff_tag(span_texts[1], seen[1])}"
        )

        # Untouched regions byte-identical: head, the gap, the tail.
        assert merged_lines[: span_b.start_line - 1] == \
            source_lines[: span_b.start_line - 1], (
            f"head before the first span changed: {m}"
        )
        _assert_same_lines(
            source_lines, merged_lines,
            span_b.end_line + 1, span_c.start_line - 1,
            span_b.end_line + first_delta + 1, "middle gap between the spans",
        )
        _assert_same_lines(
            source_lines, merged_lines,
            span_c.end_line + 1, len(source_lines),
            span_c.end_line + first_delta + second_delta + 1, "tail",
        )

        # Both edits landed (op spec + span-scoped splice; each declared
        # line appears exactly twice — two edits share the declared shape).
        # Edit 2's span sits at its post-edit-1 position in the merged text.
        for pos, span in enumerate(spans):
            _assert_seam_op_spec(
                source, merged, span, declared_count=2,
                merged_shift=first_delta if pos else 0,
                kept_prefix=kept_prefix,
            )

        # Full-file relative validity, independently of fastedit.
        from fastedit.data_gen.ast_analyzer import parse_diagnostics

        merged_diags = parse_diagnostics(merged, manifest.language)
        assert merged_diags.is_valid, (
            f"merged file has parse errors: {merged_diags.errors[:3]} | {m}"
        )

        # CRLF corpora: the funnel must have repaired every produced piece.
        if manifest.eol == EOL_CRLF:
            assert merged.count("\n") == merged.count("\r\n"), (
                f"bare LF survived the EOL funnel: {m}"
            )
        assert "\ufffd" not in merged, f"U+FFFD in the merged file: {m}"

        if extra_assert is not None:
            extra_assert(merged, m)

        # THE golden assertion: the composed file, byte-exact.
        assert merged == golden, (
            f"composed rejoin differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}"
        )

    with _StopWatch(label):
        last_error: AssertionError | None = None
        for attempt in range(1, OUTER_ATTEMPTS + 1):
            target.write_bytes(original_bytes)
            mcp_harness.recorded_chunks.clear()
            message = asyncio.run(tools_edit.fast_batch_edit(
                file_path=str(target), edits=edits_json,
            ))
            merged = target.read_bytes().decode("utf-8")
            try:
                check(message, merged)
                print(f"[C3 seams] {label}: {message.splitlines()[0]}",
                      flush=True)
                break
            except AssertionError as exc:
                last_error = exc
        else:
            raise AssertionError(
                f"{label} did not converge within {OUTER_ATTEMPTS} outer "
                f"attempts — non-convergence is a product defect (plan §0): "
                f"{last_error}"
            )

    # Backward reconstruction: the oracle's inverse ops (recorded original
    # span bytes) regenerate the original from the REAL edited content.
    records = [_wrap_record(source, span) for span in spans]
    merged = target.read_bytes().decode("utf-8")
    inverse_c = _wrap_inverse(records[1], wrapped_lengths[1])
    # edit 2's recorded span sits at its post-edit-1 position.
    inverse_c = corpus.LineOp(
        kind=inverse_c.kind,
        start_line=inverse_c.start_line + first_delta,
        end_line=inverse_c.end_line + first_delta,
        text=inverse_c.text,
    )
    inverse_b = _wrap_inverse(records[0], wrapped_lengths[0])
    for inv in (inverse_c, inverse_b):  # reverse order: highest lines first
        merged = corpus.apply_line_op(merged, inv)
    assert merged == case_text, (
        f"{label}: the oracle's inverse ops did not regenerate the original: "
        f"{first_diff_tag(case_text, merged)}"
    )


def _wrap_span_lines_for(
    source: corpus.CorpusSource, span: corpus.SymbolSpan, spec: _WrapSpec,
) -> list[str]:
    from test_stress_100mb_llm import _wrap_span_lines

    source_lines = source.splitlines(keepends=True)
    return _wrap_span_lines(
        source_lines[span.start_line - 1: span.end_line],
        source.manifest.indent, spec, source.manifest.eol,
    )


# ---------------------------------------------------------------------------
# (a) boilerplate: the duplicated-line trap at a seam and away from seams
# ---------------------------------------------------------------------------


def test_boilerplate_seam_and_away_batch(boilerplate_case, mcp_harness,
                                         tmp_path):
    """Two seam edits in one batch: a seam-adjacent trap and an away trap.

    The bodies are byte-identical across ALL symbols (only the signature
    differs), so any wrong-occurrence edit is invisible to content checks —
    the manifest span + the byte-exact composed golden are the only proof
    the edits landed on the right occurrences (the op-spec conformance is
    span-scoped: a file-wide survivor census cannot attribute a duplicated
    body line to the target).
    """
    require_stress_env()
    source = boilerplate_case
    language = source.manifest.language
    span_b = _seam_target(source)
    span_c = _away_target(source, frozenset({span_b.name}))
    _run_batch_case(
        source, tmp_path, mcp_harness,
        [span_b, span_c],
        f"(a) boilerplate seam+away batch [{language}]",
    )


# ---------------------------------------------------------------------------
# (b) cjk_seams: CJK × CRLF × seam batch
# ---------------------------------------------------------------------------


def test_cjk_crlf_seam_batch(cjk_case, mcp_harness, tmp_path):
    """Two wrap edits on the CRLF corpus with CJK on both seam sides."""
    require_stress_env()
    source = cjk_case
    language = source.manifest.language
    span_b = _seam_target(source)
    span_c = _away_target(source, frozenset({span_b.name}))
    # The CJK corpus's leading standalone CJK lines are kept context, never
    # wrapped (``_CJK_KEPT_PREFIX``): the measured real models drop a
    # standalone CJK comment they are asked to wrap.
    source_lines = source.splitlines(keepends=True)
    cjk_stripped = {line.strip() for line in source_lines if "中文" in line}
    assert cjk_stripped, "the cjk corpus carries no CJK lines"

    def extra_assert(merged: str, m: str) -> None:
        # CJK bytes intact: every CJK line's stripped content survives
        # somewhere in the merged file (the two wrapped targets re-indent
        # their CJK lines one level deeper; byte-exact placement is the
        # composed golden's job — this proves no CJK line ANYWHERE was
        # mutated, mojibake'd or lost). Single pass over each text: a
        # per-line scan of a 100MB file would be quadratic.
        got = {
            line.strip() for line in merged.splitlines(keepends=True)
            if "中文" in line
        }
        missing = cjk_stripped - got
        assert not missing, (
            f"{len(missing)} CJK line(s) mutated or lost, e.g. "
            f"{sorted(missing)[:3]!r} | {m}"
        )

    _run_batch_case(
        source, tmp_path, mcp_harness,
        [span_b, span_c],
        f"(b) cjk×crlf seam batch [{language}]",
        kept_prefix=_CJK_KEPT_PREFIX.get(language),
        extra_assert=extra_assert,
    )


# ---------------------------------------------------------------------------
# (c) prefix: resolution ambiguity at scale
# ---------------------------------------------------------------------------


def test_prefix_symbol_disambiguation_batch(prefix_case, mcp_harness,
                                            tmp_path):
    """``replace=<base>_data`` twice in one batch.

    The golden proves the MIDDLE symbol of each family was edited; the
    ``<base>`` and ``<base>_data_v2`` siblings must stay byte-identical
    (no misplaced edit despite the prefix relationships). Siblings are
    read at their position in the MERGED text: everything below an earlier
    edit shifts by that edit's line delta (``_shift_for``).
    """
    require_stress_env()
    source = prefix_case
    language = source.manifest.language
    span_b, span_c = _prefix_targets(source)
    source_lines = source.splitlines(keepends=True)
    edits = [
        (span_b, _seam_line_delta(source, span_b)),
        (span_c, _seam_line_delta(source, span_c)),
    ]

    def sibling_assert(merged: str, m: str) -> None:
        merged_lines = merged.splitlines(keepends=True)
        manifest = source.manifest
        for span, _delta in edits:
            base = span.name[: -len("_data")]
            for sibling in (base, f"{base}_data_v2"):
                sib = manifest.span(sibling)
                original = _join(source_lines, sib.start_line, sib.end_line)
                shift = _shift_for(sib.start_line, edits)
                got = _join(merged_lines, sib.start_line + shift,
                            sib.end_line + shift)
                assert got == original, (
                    f"the prefix sibling '{sibling}' was mutated "
                    f"(the edit landed on the wrong symbol): "
                    f"{first_diff_tag(original, got)} | {m}"
                )

    _run_batch_case(
        source, tmp_path, mcp_harness,
        [span_b, span_c],
        f"(c) prefix disambiguation batch [{language}]",
        extra_assert=sibling_assert,
    )


# ---------------------------------------------------------------------------
# (d) deep-nesting narrowing: _narrow_large_node over a >100-line symbol
# ---------------------------------------------------------------------------


def test_deep_nesting_narrow_block_byte_exact(deep_case, real_engine,
                                              tmp_path):
    """``replace=<deep symbol>`` + a marker snippet narrows to the recorded
    first block; the real model merges that block; golden byte-exact.

    The narrow is asserted via ``result.chunk_regions``: the region must be
    the manifest-recorded FIRST block — NOT the full 100+ line symbol (the
    narrow declined) and NOT any other block. The deterministic editor
    declines the duplicated-block shape (B24 tie), so the REAL model runs.
    """
    require_stress_env()
    case = deep_case
    language = case.manifest.language
    source = str(case)
    manifest = case.manifest
    deep = _deep_target(case)
    blk = deep.blocks[0]
    lines = source.splitlines(keepends=True)
    base_indent = "    " if language == "python" else manifest.indent
    tail_lines = _narrow_tail_lines(language, base_indent, manifest.eol)
    snippet = "".join(
        [lines[blk.start_line - 1].rstrip("\r\n") + manifest.eol]
        + [ln.rstrip("\r\n") + manifest.eol for ln in
           lines[blk.start_line: blk.end_line]]
        + tail_lines
    )
    new_span = (
        [lines[blk.start_line - 1]]
        + list(lines[blk.start_line: blk.end_line])
        + tail_lines
    )
    golden = "".join(lines[: blk.start_line - 1] + new_span
                     + lines[blk.end_line:])
    first_appended = blk.end_line + 1
    inverse = corpus.LineOp(
        kind="delete",
        start_line=first_appended,
        end_line=first_appended + len(tail_lines) - 1,
        text="",
    )
    target = tmp_path / f"corpus_100mb_seams.{manifest.ext}"
    target.write_bytes(source.encode("utf-8"))

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        # the REAL model ran, on the NARROWED block
        assert run.merge_results, (
            f"the deterministic path ran — no model, no narrow: {m}"
        )
        assert result.model_tokens > 0, f"zero model tokens: {m}"
        assert result.retries <= _DEFAULT_RETRY_BUDGET, m
        assert result.chunks_rejected == 0, m
        assert result.parse_valid, m
        assert result.chunks_used == 1, m
        assert result.chunk_regions == [(blk.start_line, blk.end_line)], (
            f"the narrow must cut the manifest-recorded FIRST block "
            f"{(blk.start_line, blk.end_line)} of '{deep.name}' "
            f"({deep.start_line}-{deep.end_line}, "
            f"{len(deep.blocks)} recorded blocks), got {result.chunk_regions}"
        )

        # untouched regions byte-identical
        merged_lines = merged.splitlines(keepends=True)
        assert merged_lines[: blk.start_line - 1] == \
            lines[: blk.start_line - 1], "head changed"
        _assert_same_lines(
            lines, merged_lines, blk.end_line + 1, len(lines),
            blk.start_line + len(new_span), "tail",
        )

        from fastedit.data_gen.ast_analyzer import parse_diagnostics

        assert parse_diagnostics(source, language).is_valid
        merged_diags = parse_diagnostics(merged, language)
        assert merged_diags.is_valid, f"parse errors: {merged_diags.errors[:3]}"

        assert merged == golden, (
            f"narrowed merge differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}"
        )

    with _StopWatch(f"(d) deep narrow [{language}]"):
        last_metrics = ""
        for attempt in range(1, OUTER_ATTEMPTS + 1):
            run = run_real_edit(
                source, snippet, file_path=str(target), language=language,
                engine=real_engine, replace=deep.name,
            )
            try:
                check(run)
                print(f"[C3 seams] (d) observed: {metrics_tag(run)}",
                      flush=True)
                break
            except AssertionError as exc:
                last_metrics = f"[outer attempt {attempt}/{OUTER_ATTEMPTS}] {exc}"
        else:
            raise AssertionError(
                f"deep-narrow did not converge within {OUTER_ATTEMPTS} outer "
                f"attempts — non-convergence is a product defect (plan §0): "
                f"{last_metrics}"
            )

    # The caller owns the write (chunked_merge never writes — the C2 (a)/(b)
    # doctrine): persist the REAL pipeline output, then reconstruct backward
    # from the bytes on disk.
    target.write_bytes(run.result.merged_code.encode("utf-8"))
    # backward reconstruction: delete the appended lines -> the original
    merged = target.read_bytes().decode("utf-8")
    restored = corpus.apply_line_op(merged, inverse)
    assert restored == source, (
        f"the narrow inverse did not regenerate the original: "
        f"{first_diff_tag(source, restored)}"
    )


# ---------------------------------------------------------------------------
# (e) multi-region reverse assembly: import region + code region in ONE call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["python", "go"])
def test_multi_region_reverse_assembly(language, real_engine, tmp_path):
    """ONE chunked_merge call, TWO chunk regions, spliced in reverse.

    The snippet adds a header import (the recorded import region) AND wraps
    a symbol (the code region): ``locate_chunks`` returns both, the assembly
    loop processes the HIGHER region first and splices both on ORIGINAL
    line indexing. Golden byte-exact; regions asserted from the manifest.
    Builds its declared (cjk_seams, language) instance — python and go.
    """
    with _seam_case(corpus.RECIPE_CJK_SEAMS, language) as case:
        manifest = case.manifest
        source = str(case)
        lines = source.splitlines(keepends=True)
        assert manifest.import_lines is not None
        target = _away_target(case, frozenset())
        import_line = {
            "python": "import collections",
            "go": 'import "collections"',
        }[language]
        snippet = import_line + "\n" + _seam_snippet(
            case, target, kept_prefix=_CJK_KEPT_PREFIX.get(language),
        )
        # golden: the appended import (measured: the model appends it after
        # the last declared import) + the seam op, spliced independently.
        _first_import, last_import = manifest.import_lines
        merged_lines = (
            lines[:last_import]
            + [import_line + manifest.eol]
            + lines[last_import:]
        )
        golden_lines = _seam_golden_lines(
            case, target, kept_prefix=_CJK_KEPT_PREFIX.get(language),
        )
        wrap_golden = "".join(
            merged_lines[: target.start_line]
            + golden_lines
            + merged_lines[target.end_line + 1:]
        )
        golden = wrap_golden
        # The located regions are in the INPUT text's coordinates: the
        # recorded import region plus the unshifted target span (the
        # appended import line shows up in the MERGED output, not in the
        # located regions).
        expected_regions = [
            manifest.import_lines,
            (target.start_line, target.end_line),
        ]
        target_path = tmp_path / f"corpus_100mb_seams.{manifest.ext}"

        def check(run):
            result = run.result
            m = metrics_tag(run)
            assert result.model_tokens > 0, f"the model must run: {m}"
            assert result.chunks_used == 2, (
                f"expected the import region + the code region (2 chunks): {m}"
            )
            assert result.chunk_regions == expected_regions, (
                f"regions must be the recorded import region plus the target "
                f"span: expected {expected_regions}, got {result.chunk_regions}"
            )
            assert result.chunks_rejected == 0 and result.parse_valid, m
            merged = result.merged_code
            assert merged.count("\n") == merged.count("\r\n"), (
                f"bare LF survived the EOL funnel: {m}"
            )
            assert merged == golden, (
                f"multi-region rejoin differs from the independent golden: "
                f"{first_diff_tag(golden, merged)} | {m}"
            )

        with _StopWatch(f"(e) multi-region reverse assembly [{language}]"):
            last_metrics = ""
            for attempt in range(1, OUTER_ATTEMPTS + 1):
                run = run_real_edit(source, snippet,
                                    file_path=str(target_path),
                                    language=language, engine=real_engine)
                try:
                    check(run)
                    print(f"[C3 seams] (e) observed: {metrics_tag(run)}",
                          flush=True)
                    break
                except AssertionError as exc:
                    last_metrics = (
                        f"[outer attempt {attempt}/{OUTER_ATTEMPTS}] {exc}"
                    )
            else:
                raise AssertionError(
                    f"multi-region assembly did not converge within "
                    f"{OUTER_ATTEMPTS} outer attempts — non-convergence is a "
                    f"product defect (plan §0): {last_metrics}"
                )

        # backward reconstruction: undo the seam op (at its +1 position),
        # then drop the appended import line.
        merged = run.result.merged_code
        record = _wrap_record(case, target)
        wrapped_len = len(
            _seam_golden_lines(
                case, target, kept_prefix=_CJK_KEPT_PREFIX.get(language),
            )
        )
        inverse_wrap = _wrap_inverse(record, wrapped_len)
        inverse_wrap = corpus.LineOp(
            kind=inverse_wrap.kind,
            start_line=inverse_wrap.start_line + 1,
            end_line=inverse_wrap.end_line + 1,
            text=inverse_wrap.text,
        )
        merged = corpus.apply_line_op(merged, inverse_wrap)
        merged = corpus.apply_line_op(merged, corpus.LineOp(
            kind="delete", start_line=last_import + 1,
            end_line=last_import + 1,
        ))
        assert merged == source, (
            f"the multi-region inverses did not regenerate the original: "
            f"{first_diff_tag(source, merged)}"
        )


# ---------------------------------------------------------------------------
# (f) undo (full stack) + backward reconstruction on seam recipes
# ---------------------------------------------------------------------------


def _undo_case(
    source: corpus.CorpusSource, tmp_path, wrap_span: tuple,
    wrap_snippet_text: str, wrap_golden: str, wrap_delta: int, label: str,
) -> None:
    """wrap (LLM) -> deterministic insert -> undo -> undo, byte-exact."""
    from fastedit.mcp import tools_ast, tools_edit

    case_text = str(source)
    manifest = source.manifest
    insert = corpus.make_ops(
        source, kinds=("insert_after_symbol",), seed=_SEEDS[manifest.recipe],
    )[0]
    anchor_end = insert.anchor_end_line
    assert anchor_end < wrap_span[0], (
        "the insert anchor must precede the wrap span so the recorded "
        "indices compose directly on the wrapped text"
    )
    golden_insert = corpus.apply_op_oracle(wrap_golden, insert)
    original_bytes = case_text.encode("utf-8")
    target = tmp_path / f"corpus_100mb_seams.{manifest.ext}"
    target.write_bytes(original_bytes)

    with _StopWatch(label):
        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=wrap_snippet_text,
            replace=wrap_span[2],
        ))
        assert response.startswith(f"Applied edit to {target}"), response
        assert "rejected" not in response and "Error" not in response, response
        assert target.read_bytes().decode("utf-8") == wrap_golden, (
            f"the wrap edit did not write the golden bytes: "
            f"{first_diff_tag(wrap_golden, target.read_bytes().decode('utf-8'))}"
        )
        print(f"[C3 seams] {label}: edit 1 {response.splitlines()[0]}",
              flush=True)

        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=insert.new_text,
            after=insert.symbol,
        ))
        assert response.startswith(f"Applied edit to {target}"), response
        assert target.read_bytes().decode("utf-8") == golden_insert, (
            "the deterministic insert did not write the oracle bytes"
        )

        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes().decode("utf-8") == wrap_golden, (
            f"undo #1 did not restore the post-wrap state byte-exactly: "
            f"{first_diff_tag(wrap_golden, target.read_bytes().decode('utf-8'))}"
        )

        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes() == original_bytes, (
            f"undo #2 did not restore the original bytes: "
            f"{first_diff_tag(case_text, target.read_bytes().decode('utf-8'))}"
        )

        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith("Error: no undo history"), response
        assert target.read_bytes() == original_bytes

    # backward reconstruction: the oracle's inverse of the wrap regenerates
    # the original from the REAL edited bytes the pipeline produced.
    lines = case_text.splitlines(keepends=True)
    record = corpus.CorpusOp(
        kind="replace_symbol_body", symbol=wrap_span[2],
        language=manifest.language, ext=manifest.ext, eol=manifest.eol,
        final_eol=manifest.final_eol,
        start_line=wrap_span[0], end_line=wrap_span[1],
        original_span_text=_join(lines, wrap_span[0], wrap_span[1]),
    )
    inverse = _wrap_inverse(record, wrap_delta)
    restored = corpus.apply_line_op(wrap_golden, inverse)
    assert restored == case_text, (
        f"{label}: the wrap inverse did not regenerate the original: "
        f"{first_diff_tag(case_text, restored)}"
    )


def test_undo_full_stack_boilerplate(mcp_harness, tmp_path):
    """Undo full stack on the duplicated-line-trap corpus (python)."""
    with _seam_case(corpus.RECIPE_BOILERPLATE, "python") as source:
        manifest = source.manifest
        spec, _delta = _py_wrap(manifest.indent)
        # past 40%: the deterministic insert's anchor sits at ~25% of the
        # corpus, so the wrap span must be BELOW it for the oracle to
        # compose both states on unshifted recorded indices.
        span = _seam_target(source, min_fraction=0.4)
        lines = source.splitlines(keepends=True)
        wrap_golden, _ = _wrap_expected(str(source), span, manifest.indent,
                                        spec, manifest.eol)
        _undo_case(
            source, tmp_path,
            (span.start_line, span.end_line, span.name),
            _wrap_snippet(
                lines[span.start_line - 1].rstrip("\r\n"),
                _span_kept_lines(lines, span, spec),
                manifest.indent, spec, manifest.eol,
            ),
            wrap_golden,
            len(_wrap_span_lines_for(source, span, spec)),
            "(f) undo full stack [python/boilerplate]",
        )


def test_undo_full_stack_cjk_crlf(mcp_harness, tmp_path):
    """Undo full stack on the CJK × CRLF corpus (python)."""
    with _seam_case(corpus.RECIPE_CJK_SEAMS, "python") as source:
        manifest = source.manifest
        spec, _delta = _py_wrap(manifest.indent)
        span = _seam_target(source, min_fraction=0.4)
        lines = source.splitlines(keepends=True)
        wrap_golden, _ = _wrap_expected(str(source), span, manifest.indent,
                                        spec, manifest.eol)
        _undo_case(
            source, tmp_path,
            (span.start_line, span.end_line, span.name),
            _wrap_snippet(
                lines[span.start_line - 1].rstrip("\r\n"),
                _span_kept_lines(lines, span, spec),
                manifest.indent, spec, manifest.eol,
            ),
            wrap_golden,
            len(_wrap_span_lines_for(source, span, spec)),
            "(f) undo full stack [python/cjk_seams]",
        )


def test_undo_full_stack_deep_narrow(mcp_harness, tmp_path):
    """Undo full stack on the deep corpus (go): the narrow edit is undone."""
    with _seam_case(corpus.RECIPE_DEEP, "go") as source:
        from fastedit.mcp import tools_ast, tools_edit

        manifest = source.manifest
        deep = _deep_target(source, min_fraction=0.4)
        blk = deep.blocks[0]
        lines = source.splitlines(keepends=True)
        tail_lines = _narrow_tail_lines("go", manifest.indent, manifest.eol)
        snippet = "".join(
            [lines[blk.start_line - 1].rstrip("\r\n") + manifest.eol]
            + [ln.rstrip("\r\n") + manifest.eol for ln in
               lines[blk.start_line: blk.end_line]]
            + tail_lines
        )
        new_span = (
            [lines[blk.start_line - 1]]
            + list(lines[blk.start_line: blk.end_line])
            + tail_lines
        )
        golden_narrow = "".join(lines[: blk.start_line - 1] + new_span
                                + lines[blk.end_line:])
        original_bytes = source.encode("utf-8")
        insert = corpus.make_ops(
            source, kinds=("insert_after_symbol",),
            seed=_SEEDS[manifest.recipe],
        )[0]
        assert insert.anchor_end_line < blk.start_line, (
            "the insert anchor must precede the narrowed block"
        )
        golden_insert = corpus.apply_op_oracle(golden_narrow, insert)
        target = tmp_path / f"corpus_100mb_seams.{manifest.ext}"
        target.write_bytes(original_bytes)

        with _StopWatch("(f) undo full stack [go/deep]"):
            response = asyncio.run(tools_edit.fast_edit(
                file_path=str(target), edit_snippet=snippet,
                replace=deep.name,
            ))
            assert response.startswith(f"Applied edit to {target}"), response
            assert target.read_bytes().decode("utf-8") == golden_narrow, (
                "the narrow edit did not write the golden bytes: "
                f"{first_diff_tag(golden_narrow, target.read_bytes().decode('utf-8'))}"
            )
            print(f"[C3 seams] (f) go/deep: edit 1 {response.splitlines()[0]}",
                  flush=True)

            response = asyncio.run(tools_edit.fast_edit(
                file_path=str(target), edit_snippet=insert.new_text,
                after=insert.symbol,
            ))
            assert response.startswith(f"Applied edit to {target}"), response
            assert target.read_bytes().decode("utf-8") == golden_insert

            response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
            assert response.startswith(f"Reverted {target}"), response
            assert target.read_bytes().decode("utf-8") == golden_narrow, (
                "undo #1 did not restore the post-narrow state byte-exactly"
            )

            response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
            assert response.startswith(f"Reverted {target}"), response
            assert target.read_bytes() == original_bytes, (
                "undo #2 did not restore the original bytes"
            )
            response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
            assert response.startswith("Error: no undo history"), response

        # backward reconstruction on the narrowed block: delete the appended
        # tail lines from the REAL narrowed output -> the original.
        inverse = corpus.LineOp(
            kind="delete",
            start_line=blk.start_line + (blk.end_line - blk.start_line) + 1,
            end_line=blk.start_line + (blk.end_line - blk.start_line)
            + len(tail_lines),
            text="",
        )
        restored = corpus.apply_line_op(golden_narrow, inverse)
        assert restored == source, (
            f"the narrow inverse did not regenerate the original: "
            f"{first_diff_tag(source, restored)}"
        )
