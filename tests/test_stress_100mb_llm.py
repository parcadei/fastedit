"""Step C2 — the 100MB REAL-LLM e2e stress suite (implementation plan Phase C).

MISSION: a ~100MB file, transparent chunking → REAL-LLM-guided edit → chunk
rejoin, proven byte-perfect. Real model only (the trained fastedit mlx-8bit
weights via :func:`llm_fixtures.real_engine`) — there is no fake on the LLM
path anywhere in this module, per plan req. 1. The deterministic ``after=``
path is asserted to consume ZERO model tokens, which a fake could not prove.

Cases (per language: python + rust + typescript), each marked ``llm`` and
``stress`` and gated on ``FASTEDIT_RUN_STRESS=1`` (plan §0 runtime tiers):

a. **LLM-path symbol replace** — a marker-bearing wrap_block snippet on a
   corpus symbol (deterministic-declined shape, so the REAL model must run:
   ``model_tokens > 0``; ``retries`` recorded and bounded). Per the
   non-determinism policy §0(b): bytes before the span start and after the
   span end byte-identical to the original (untouched-region identity), the
   replaced span conforms to the op spec (new guard block present, every
   declared kept line present exactly once at its wrapped indent), full-file
   relative validity, and FULL-FILE golden byte-exactness against the
   independent line-splice oracle inside a bounded number of OUTER attempts
   (``OUTER_ATTEMPTS`` — each attempt is a complete re-edit; non-convergence
   is a product defect per plan §0, reported with ``first_diff_tag``).
b. **after= insert (deterministic)** — byte-exact vs the C1 oracle in ONE
   attempt (``model_tokens == 0``), plus the oracle's inverse op regenerating
   the original byte-for-byte.
c. **Multi-chunk LLM edit** — two wrap edits at DISTANT symbols through the
   MCP full-stack path (:func:`fast_batch_edit` over the REAL engine):
   ``2 chunk(s)`` bookkeeping, the model only ever sees the manifest span
   bytes of the two targets (recorded chunk texts asserted byte-exact against
   the manifest spans), both edits landed, untouched regions byte-identical,
   composed golden byte-exactness with outer attempts.
d. **Undo (full stack)** — two stacked edits, then MCP ``fast_undo``
   (tools_ast) in-process: undo #1 restores the previous state byte-exact
   (the N-deep backup proof at 100MB), undo #2 restores the ORIGINAL bytes
   exactly, and the undo ledger is then empty (fail-loud).
e. **Backward reconstruction** — the oracle's inverse op (recorded original
   span bytes, executed by ``corpus.apply_line_op``) applied to the EDITED
   content the REAL pipeline produced regenerates the original byte-for-byte.

The golden oracle is C1's INDEPENDENT line-splice arithmetic
(``tests/corpus.py``): the expected files are computed here from the
generator-recorded manifest spans and never by running fastedit. The wrap
op itself is expressed as declarative per-language data (``_WrapSpec``) — no
per-language branches in the assertion logic (CLAUDE.md's no-hardcoding
rule); the variance lives in the spec table.

Memory & runtime discipline (measured on this machine, M-series / 64 GB):
* corpus build (``build_corpus_case``, 100MB): ~1.1 s and ~0.6 GB peak RSS
  per source; each language's source is built ONCE per module run via the
  parametrized module-scoped fixture and released (cache pop + ``gc``) before
  the next language's instance is built, so at most one 100MB source is alive.
* per-edit fixed overhead at 100MB: tree-sitter parses dominate —
  ``get_ast_map_from_source`` ~4.6 s and ``parse_diagnostics`` ~12.5 s per
  parse (the C2 suite-opener scan adds ~4 s) with a ~4.5 GB transient (the
  parse tree of a 2.6M-line file). A replace edit costs ~3 parses; a batch of
  two ~7. Model generation itself is seconds (tight ~10-line chunks).
  Measured per-case wall times print as ``[C2 stress] ...`` lines (run pytest
  with ``-s``): the full module ran 15/15 green in 18m37s — python
  (a) 97s (b) 33s (c) 167s (d) 109s (e) 70s; rust 63/29/99/74/39s;
  typescript 65/31/106/81/42s — well under the ~30 min tier ceiling.
* tmp files live under pytest's ``tmp_path`` (removed by pytest); edit
  backups land in conftest's session-isolated ``FASTEDIT_BACKUP_DIR``.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import corpus
import pytest
from corpus import EOL_CRLF, EOL_LF
from llm_fixtures import (
    first_diff_tag,
    metrics_tag,
    require_stress_env,
    run_real_edit,
)

pytestmark = [pytest.mark.llm, pytest.mark.stress]

TARGET_BYTES = 100_000_000
"""The C2 stress size: ~100MB per source (plan req. 2)."""

OUTER_ATTEMPTS = 3
"""Bounded outer attempts for golden byte-exactness (plan §0 policy (b)):

each outer attempt is a COMPLETE real edit (fresh model calls, fresh
validation); the non-determinism policy's golden assertion must hold within
this bound. Non-convergence after the bound is reported as a product defect
(plan §0), never absorbed as flakiness."""

_DEFAULT_RETRY_BUDGET = 8
"""``chunked_merge._DEFAULT_MAX_VALIDATION_RETRIES`` — the per-site budget."""

# ---------------------------------------------------------------------------
# Declarative per-language stress configuration (the variance lives here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WrapSpec:
    """One language's LLM-path wrap_block op, as declarative data.

    The snippet wraps the target's body in a new guard block. The shapes
    are the ones measured to converge against the REAL model while being
    declined by every deterministic path (see the per-language notes):

    * ``python`` declares the target's docstring (``kept_prefix``) as a kept
      context line and carries the marker at the anchor's own indent — the
      deterministic editor's output for this shape leaves the ``with`` suite
      empty, which the C2 suite-opener judgment rejects, so the REAL model
      runs and wraps the body (measured 4/4 attempts);    * the brace languages declare an ``else`` branch (the model's natural
      completion preserves the return type — measured on the real model) and
      a trailing marker so the new block's closer stays a DECLARED line (see
      tests/test_brace_wrap_model_path.py for the validator contracts this
      relies on).
    """

    comment: str  # comment prefix ("# " / "// ")
    kept_prefix: int  # span body lines restated as kept context anchors
    opener: Callable[[str], str]  # indent -> opener line (no EOL)
    else_opener: Callable[[str], str] | None  # None = no else branch
    else_body: Callable[[str], str] | None
    else_closer: Callable[[str], str] | None
    closer_is_span_last: bool
    """True when the span's last line is a `}` kept at its own indent (the
    brace languages); False when every span line after the kept prefix is a
    body line that moves one level deeper (python)."""
    marker_indent: Callable[[str], str]  # indent -> the marker line's indent
    trailing_marker: bool  # a second marker after the block closer


def _py_wrap(indent: str) -> tuple[_WrapSpec, int]:
    del indent
    spec = _WrapSpec(
        comment="# ",
        kept_prefix=1,  # the docstring stays outside the wrap, declared
        opener=lambda ind: f"{ind}with audit_lock:",
        else_opener=None,
        else_body=None,
        else_closer=None,
        closer_is_span_last=False,
        marker_indent=lambda ind: ind,
        trailing_marker=False,
    )
    return spec, 1  # line delta: the opener line only


def _brace_wrap(opener_text: str, else_body_text: str) -> tuple[_WrapSpec, int]:
    spec = _WrapSpec(
        comment="// ",
        kept_prefix=0,
        opener=lambda ind: f"{ind}{opener_text} {{",
        else_opener=lambda ind: f"{ind}}} else {{",
        # the else branch lives INSIDE the function: opener at one indent
        # level, its body one level deeper (measured on the real model).
        else_body=lambda ind: f"{ind * 2}{else_body_text}",
        else_closer=lambda ind: f"{ind}}}",
        closer_is_span_last=True,
        marker_indent=lambda ind: ind * 2,
        trailing_marker=True,
    )
    return spec, 4  # opener + else opener + else body + else closer


@dataclass(frozen=True)
class _LanguageStress:
    """One language's 100MB stress recipe (seed-pinned, fully declarative)."""

    language: str
    ext: str
    eol: str
    final_eol: bool
    seed: str
    wrap: Callable[[str], tuple[_WrapSpec, int]]


_LANG_CONFIG: dict[str, _LanguageStress] = {
    # python stresses the EOL funnel on the model path at scale (CRLF file).
    "python": _LanguageStress(
        language="python", ext="py", eol=EOL_CRLF, final_eol=True,
        seed="c2-py-100mb", wrap=_py_wrap,
    ),
    # rust stresses the trailing-newline rule (no final EOL) on the model path.
    "rust": _LanguageStress(
        language="rust", ext="rs", eol=EOL_LF, final_eol=False,
        seed="c2-rs-100mb", wrap=lambda indent: _brace_wrap(
            "if value_a > 0", "0",
        ),
    ),
    "typescript": _LanguageStress(
        language="typescript", ext="ts", eol=EOL_LF, final_eol=True,
        seed="c2-ts-100mb", wrap=lambda indent: _brace_wrap(
            "if (valueA > 0)", "return 0;",
        ),
    ),
}
_LANGUAGES = tuple(_LANG_CONFIG)


# ---------------------------------------------------------------------------
# Small measurement / assertion helpers
# ---------------------------------------------------------------------------


class _StopWatch:
    """Print one per-case wall-time line (plan task 2: runtime discipline)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        print(
            f"[C2 stress] {self.label}: {time.perf_counter() - self._t0:.1f}s",
            flush=True,
        )


def _join(lines: list[str], first: int, last: int) -> str:
    """1-indexed inclusive line-range join (keepends)."""
    return "".join(lines[first - 1 : last])


def _assert_same_lines(
    orig_lines: list[str],
    merged_lines: list[str],
    orig_first: int,
    orig_last: int,
    merged_first: int,
    label: str,
) -> None:
    """Untouched-region identity: two line ranges must be byte-identical."""
    want = _join(orig_lines, orig_first, orig_last)
    got = _join(merged_lines, merged_first, merged_first + (orig_last - orig_first))
    assert got == want, (
        f"{label}: untouched region changed: "
        f"{first_diff_tag(want, got)} (original lines "
        f"{orig_first}-{orig_last} vs merged from {merged_first})"
    )


def _distinct(items: list[str]) -> list[str]:
    """Order-preserving de-duplication (retry attempts repeat a chunk)."""
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# The wrap op: snippet, independent golden, and span bookkeeping
# ---------------------------------------------------------------------------


def _wrap_marker(indent: str, spec: _WrapSpec) -> str:
    return f"{spec.marker_indent(indent)}{spec.comment}... existing code ..."


def _wrap_snippet(
    sig_line: str, kept_lines: list[str], indent: str, spec: _WrapSpec, eol: str,
) -> str:
    """The marker-bearing LLM-path snippet for one target symbol.

    ``sig_line`` and every entry of ``kept_lines`` (the declared kept context
    anchors — ``spec.kept_prefix`` of them) carry no EOL; every emitted line
    is terminated with the file's own convention.
    """
    marker = _wrap_marker(indent, spec)
    lines = [sig_line] + kept_lines + [spec.opener(indent), marker]
    if spec.else_opener is not None:
        lines += [
            spec.else_opener(indent),
            spec.else_body(indent),
            spec.else_closer(indent),
        ]
        if spec.trailing_marker:
            lines.append(marker)
    return "".join(line + eol for line in lines)


def _wrap_body_slices(
    span_lines: list[str], spec: _WrapSpec,
) -> tuple[list[str], list[str]]:
    """Split a span's lines into (kept prefix, body to be wrapped).

    ``span_lines[0]`` is the signature; the kept prefix follows it; the
    brace languages' closer (the span's last line) is excluded from the
    wrapped body — it stays at its own indent, after the new block.
    """
    body = span_lines[1:]
    kept = body[: spec.kept_prefix]
    wrapped = body[spec.kept_prefix :]
    if spec.closer_is_span_last:
        wrapped = wrapped[:-1]
    return kept, wrapped


def _wrap_span_lines(
    span_lines: list[str], indent: str, spec: _WrapSpec, eol: str,
) -> list[str]:
    """The wrapped span: the independent golden's replacement lines.

    Pure line-splice arithmetic on the manifest-recorded span — the oracle's
    doctrine (never fastedit, never a parser). ``span_lines`` is the span's
    keepends lines, ``span_lines[0]`` the signature.
    """
    sig = span_lines[0]
    kept, wrapped = _wrap_body_slices(span_lines, spec)
    out = [sig] + kept + [spec.opener(indent) + eol]
    out += [indent + line for line in wrapped]
    if spec.else_opener is not None:
        out += [
            spec.else_opener(indent) + eol,
            spec.else_body(indent) + eol,
            spec.else_closer(indent) + eol,
        ]
    if spec.closer_is_span_last:
        out.append(span_lines[-1])  # the original closer, byte-identical
    return out


def _wrap_expected(
    source: str, span: corpus.SymbolSpan, indent: str, spec: _WrapSpec, eol: str,
) -> tuple[str, int]:
    """Full-file golden for the wrap op, plus the net line delta."""
    lines = source.splitlines(keepends=True)
    span_lines = lines[span.start_line - 1 : span.end_line]
    new_span = _wrap_span_lines(span_lines, indent, spec, eol)
    merged = "".join(
        lines[: span.start_line - 1] + new_span + lines[span.end_line :],
    )
    return merged, len(new_span) - len(span_lines)


def _wrap_inverse(
    op: corpus.CorpusOp, new_span_line_count: int,
) -> corpus.LineOp:
    """The wrap edit's inverse, in the oracle's own inverse machinery.

    The corpus op record carries the RECORDED original span bytes
    (``original_span_text``, cross-checked by ``apply_op_oracle``); the
    inverse is a replace of the wrapped span with those recorded bytes,
    executed by ``corpus.apply_line_op`` — req. 3's backward
    reconstruction, never re-derived from the edited text.
    """
    return corpus.LineOp(
        kind="replace",
        start_line=op.start_line,
        end_line=op.start_line + new_span_line_count - 1,
        text=op.original_span_text,
    )


def _assert_wrap_kept_lines(
    source: str, merged: str, span: corpus.SymbolSpan, indent: str,
    spec: _WrapSpec, eol: str, metrics: str,
    declared_line_count: int = 1,
    merged_shift: int = 0,
    unique_bodies: bool = True,
) -> int:
    """Op-spec conformance of the replaced span; returns the wrapped length.

    The declared kept prefix (e.g. python's docstring) survives exactly once
    at its ORIGINAL indent; the new guard block (opener / else lines) is
    present exactly ``declared_line_count`` times (1 for a single edit, once
    per edit when several wraps share the same declared line, as in case
    (c)); every other kept (original) body line survives exactly once one
    level deeper with its original-indent form gone. Line-exact comparisons:
    a substring check would count the deeper-indent line as a hit.

    ``merged_shift`` positions ``span`` inside the MERGED text: a batch
    applies each edit at its post-previous-edits position, so a span below
    an earlier edit sits ``merged_shift`` lines lower in the merged file.

    ``unique_bodies=False`` (the C3 boilerplate duplicated-line trap) swaps
    the file-wide survivor counts — impossible when body lines are
    byte-identical across ALL symbols, where no line count can attribute an
    occurrence to the target — for a span-scoped check: the target's merged
    span must equal the independently spliced wrapped span (and the
    byte-exact composed golden carries the whole-file proof).
    """
    merged_lines = merged.splitlines(keepends=True)
    for declared in (
        spec.opener(indent),
        spec.else_opener(indent) if spec.else_opener else None,
        spec.else_body(indent) if spec.else_body else None,
        spec.else_closer(indent) if spec.else_closer else None,
    ):
        if declared is None:
            continue
        assert merged_lines.count(declared + eol) == declared_line_count, (
            f"declared wrap line {declared!r} not present exactly "
            f"{declared_line_count} time(s): {metrics}"
        )
    source_lines = source.splitlines(keepends=True)
    span_lines = source_lines[span.start_line - 1 : span.end_line]
    kept, wrapped = _wrap_body_slices(span_lines, spec)
    if not unique_bodies:
        expected_span = _wrap_span_lines(span_lines, indent, spec, eol)
        span_start_merged = span.start_line + merged_shift
        got_span = merged_lines[
            span_start_merged - 1 : span_start_merged - 1 + len(expected_span)
        ]
        assert got_span == expected_span, (
            f"the edited span at merged lines {span_start_merged}-"
            f"{span_start_merged + len(expected_span) - 1} is not the "
            f"expected wrapped span (wrong occurrence or wrong shape): "
            f"{first_diff_tag(''.join(expected_span), ''.join(got_span))} | "
            f"{metrics}"
        )
        return len(expected_span)
    for kept_line in kept:
        assert merged_lines.count(kept_line) == 1, (
            f"declared kept line {kept_line.rstrip(eol)!r} not preserved "
            f"exactly once outside the wrap: {metrics}"
        )
    for body_line in wrapped:
        assert body_line.strip(), "corpus invariant: no blank body lines"
        wrapped_line = indent + body_line
        assert merged_lines.count(wrapped_line) == 1, (
            f"kept body line {body_line.rstrip(eol)!r} not preserved exactly "
            f"once at the wrapped indent: {metrics}"
        )
        assert merged_lines.count(body_line) == 0, (
            f"body line {body_line.rstrip(eol)!r} survived at its ORIGINAL "
            f"indent (the wrap did not re-indent it): {metrics}"
        )
    return len(_wrap_span_lines(span_lines, indent, spec, eol))


# ---------------------------------------------------------------------------
# Corpus fixture — each 100MB source built ONCE per module run
# ---------------------------------------------------------------------------

_CASE_CACHE: dict[str, corpus.CorpusCase] = {}
"""Holds at most ONE 100MB source at a time (see the fixture teardown)."""


@pytest.fixture(scope="module", params=_LANGUAGES)
def stress_case(request):
    """The 100MB corpus case for one language (module-scoped, built once).

    Memory discipline: the cache holds one language's source at a time — the
    teardown pops this param's entry and collects before the next param's
    setup builds its own, so peak corpus memory is one source (~0.6 GB build
    transient, ~100 MB resident as a compact ASCII str).
    """
    require_stress_env()  # belt-and-braces: skip before building anything
    language = request.param
    if language not in _CASE_CACHE:
        cfg = _LANG_CONFIG[language]
        with _StopWatch(f"build 100MB corpus {language}"):
            _CASE_CACHE[language] = corpus.build_corpus_case(
                language, TARGET_BYTES, eol=cfg.eol, final_eol=cfg.final_eol,
                seed=cfg.seed,
            )
    yield _CASE_CACHE[language]
    _CASE_CACHE.pop(language, None)
    gc.collect()


def _replace_op(case: corpus.CorpusCase) -> corpus.CorpusOp:
    """The corpus's own replace-target record (span + recorded inverse data)."""
    (op,) = corpus.make_ops(
        case.source, kinds=("replace_symbol_body",), seed=_LANG_CONFIG[
            case.language
        ].seed,
    )
    return op


def _insert_op(case: corpus.CorpusCase) -> corpus.CorpusOp:
    """The corpus's insert-after record (deterministic fast-path op)."""
    (op,) = corpus.make_ops(
        case.source, kinds=("insert_after_symbol",), seed=_LANG_CONFIG[
            case.language
        ].seed,
    )
    return op


def _distant_spans(
    case: corpus.CorpusCase, exclude: frozenset[str],
) -> tuple[corpus.SymbolSpan, corpus.SymbolSpan]:
    """Two function-kind manifest spans at ~1/3 and ~2/3 of the file.

    Deterministic forward scan (no cherry-picking beyond the declared
    constraints: function-kind, not a reserved symbol, ≤100 lines so the
    chunk locator's large-node narrow never engages and the model prompt
    stays tight).
    """
    manifest = case.source.manifest

    def pick(fraction: float, taken: set[str]) -> corpus.SymbolSpan:
        center = int(manifest.symbol_count * fraction)
        for sym in manifest.symbols[center:]:
            if (
                sym.kind == "function"
                and sym.name not in taken
                and sym.name not in exclude
                and sym.line_count <= 100
            ):
                return sym
        raise AssertionError(
            f"no eligible wrap target near fraction {fraction} of the corpus"
        )

    first = pick(1 / 3, set())
    second = pick(2 / 3, {first.name})
    assert first.end_line < second.start_line - 8, (
        "the two multi-chunk targets must sit at DISTANT symbols"
    )
    return first, second


# ---------------------------------------------------------------------------
# MCP full-stack harness — the smoke_test.py:174-177 shape, real engine
# ---------------------------------------------------------------------------


class _RealEngineBackend:
    """Lifespan backend whose ``acquire()`` yields the REAL session engine."""

    def __init__(self, engine) -> None:
        self._engine = engine

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self._engine


class _RecordingEngine:
    """Delegates to the real engine; records every chunk it is asked to merge.

    The recording is the honest way to assert ``chunk_regions`` through the
    MCP tool layer (the tool does not return the merge result): the model can
    only have been guided by the bytes it actually saw, so asserting each
    distinct recorded chunk against the manifest span bytes proves the chunk
    regions matched the manifest spans.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self.chunk_texts: list[str] = []

    def merge_auto(self, original_code: str, snippet: str, language=None):
        self.chunk_texts.append(original_code)
        return self._engine.merge_auto(original_code, snippet, language)


class _FakeRequestContext:
    def __init__(self, lifespan_context) -> None:
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context) -> None:
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeMcp:
    """Stand-in for the FastMCP instance imported into the tool modules."""

    def __init__(self, lifespan_context) -> None:
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


@dataclass
class McpHarness:
    """The installed lifespan context plus the recording engine's capture."""

    lifespan: dict
    recorded_chunks: list[str] = field(default_factory=list)


@pytest.fixture
def mcp_harness(real_engine, monkeypatch) -> McpHarness:
    """Wire tools_edit AND tools_ast to a lifespan context whose backend IS
    the real engine.

    Follows tests/test_batch_safety.py's ``_install_fake_mcp`` harness shape
    (and the _dev_smoke/smoke_test.py pattern) with one deliberate difference:
    NOTHING about the merge is stubbed — ``chunked_merge`` /
    ``batch_chunked_merge`` run for real end to end, including
    ``_atomic_write`` / ``BackupStore`` / the B37 expected-stat guard, and the
    engine behind ``backend.acquire()`` is the REAL trained model.
    """
    from fastedit.mcp import tools_ast, tools_edit
    from fastedit.mcp.backup import BackupStore

    recorder = _RecordingEngine(real_engine)
    lifespan_context = {
        "backend_kind": "mlx",
        "backend": _RealEngineBackend(recorder),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    fake = _FakeMcp(lifespan_context)
    monkeypatch.setattr(tools_edit, "mcp", fake)
    monkeypatch.setattr(tools_ast, "mcp", fake)
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return McpHarness(lifespan=lifespan_context, recorded_chunks=recorder.chunk_texts)


# ---------------------------------------------------------------------------
# (a) LLM-path symbol replace: marker-bearing wrap, byte-exact vs oracle
# ---------------------------------------------------------------------------


def _span_kept_lines(
    source_lines: list[str], span: corpus.SymbolSpan, spec: _WrapSpec,
) -> list[str]:
    """The span's declared kept context lines, stripped of their EOLs."""
    kept, _wrapped = _wrap_body_slices(
        source_lines[span.start_line - 1 : span.end_line], spec,
    )
    return [line.rstrip("\r\n") for line in kept]


def test_llm_path_replace_wrap_byte_exact(stress_case, real_engine, tmp_path):
    """The 100MB LLM-path replace: real model, byte-perfect rejoin."""
    require_stress_env()
    cfg = _LANG_CONFIG[stress_case.language]
    case = stress_case
    source = str(case.source)
    op = _replace_op(case)
    span = case.source.manifest.span(op.symbol)
    spec, _delta = cfg.wrap(case.indent)
    source_lines = source.splitlines(keepends=True)
    snippet = _wrap_snippet(
        source_lines[span.start_line - 1].rstrip("\r\n"),
        _span_kept_lines(source_lines, span, spec),
        case.indent, spec, case.eol,
    )
    golden, line_delta = _wrap_expected(source, span, case.indent, spec, case.eol)
    orig_lines = source.splitlines(keepends=True)

    target = tmp_path / f"corpus_100mb.{case.ext}"
    target.write_bytes(source.encode("utf-8"))

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code

        # ── the REAL model ran ──
        assert run.merge_results, (
            f"engine.merge_auto never invoked — the edit took a zero-token "
            f"deterministic path, so no real LLM ran: {m}"
        )
        assert any(r.tokens_generated > 0 for r in run.merge_results), (
            f"model generated zero tokens — no real inference ran: {m}"
        )
        assert result.model_tokens > 0, f"pipeline accounted zero tokens: {m}"

        # retries recorded, bounded, and reconciled with the attempt count.
        assert result.retries >= 0, f"retries must be recorded: {m}"
        assert result.retries <= _DEFAULT_RETRY_BUDGET, (
            f"validation-retry budget exceeded: {m}"
        )
        assert len(run.merge_results) == result.retries + result.chunks_used, (
            f"attempt accounting mismatch: {m}"
        )

        # ── pipeline gates ──
        assert result.parse_valid, (
            f"relative parse rule rejected the merge: {m}"
        )
        assert result.chunks_rejected == 0, (
            f"chunk(s) rejected by the validator — original kept: {m}"
        )
        assert result.chunks_used == 1, (
            f"a replace= edit must take exactly one tight chunk: {m}"
        )
        assert result.chunk_regions == [(span.start_line, span.end_line)], (
            f"chunk region must be the manifest span: {m} "
            f"got {result.chunk_regions}"
        )

        # ── op spec: new body present, declared kept-lines present ──
        _assert_wrap_kept_lines(
            source, merged, span, case.indent, spec, case.eol, m,
        )
        assert "# ... existing code ..." not in merged and (
            "// ... existing code ..." not in merged
        ), f"preservation marker leaked into the merged file: {m}"

        # ── untouched regions byte-identical (policy (b)) ──
        merged_lines = merged.splitlines(keepends=True)
        assert merged_lines[: span.start_line - 1] == orig_lines[: span.start_line - 1], (
            f"bytes before the span start changed: {m} "
            f"{first_diff_tag(_join(orig_lines, 1, span.start_line - 1), _join(merged_lines, 1, span.start_line - 1))}"
        )
        _assert_same_lines(
            orig_lines, merged_lines, span.end_line + 1, len(orig_lines),
            span.end_line + line_delta + 1, "tail after the span end",
        )

        # ── full-file relative validity, independently of fastedit ──
        from fastedit.data_gen.ast_analyzer import parse_diagnostics

        original_diags = parse_diagnostics(source, case.language)
        assert original_diags.is_valid, "the corpus must parse cleanly"
        merged_diags = parse_diagnostics(merged, case.language)
        assert merged_diags.is_valid, (
            f"merged file has parse error(s) the original does not have: "
            f"{merged_diags.errors[:3]} | {m}"
        )

        # ── THE golden assertion: full-file byte-exactness ──
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}"
        )

    with _StopWatch(f"(a) llm replace wrap [{stress_case.language}]"):
        last_metrics = ""
        for attempt in range(1, OUTER_ATTEMPTS + 1):
            run = run_real_edit(
                source, snippet, file_path=str(target),
                language=stress_case.language, engine=real_engine,
                replace=op.symbol,
            )
            try:
                check(run)
                print(f"[C2 stress] (a) observed: {metrics_tag(run)}", flush=True)
                break
            except AssertionError as exc:
                last_metrics = (
                    f"[outer attempt {attempt}/{OUTER_ATTEMPTS}] {exc}"
                )
        else:
            raise AssertionError(
                f"100MB LLM replace did not converge within {OUTER_ATTEMPTS} "
                f"outer attempts — non-convergence is a product defect "
                f"(plan §0): {last_metrics}"
            )

    # chunked_merge must not write; the caller (cli.py / the MCP tool) owns it.
    assert target.read_bytes() == source.encode("utf-8"), (
        "chunked_merge must not write; the disk copy must stay the original"
    )


# ---------------------------------------------------------------------------
# (b) after= insert: deterministic, byte-exact in ONE attempt
# ---------------------------------------------------------------------------


def test_after_insert_deterministic_byte_exact(stress_case, real_engine, tmp_path):
    """The zero-token fast path at 100MB: one attempt, oracle byte-exact."""
    require_stress_env()
    case = stress_case
    source = str(case.source)
    op = _insert_op(case)
    expected = corpus.apply_op_oracle(source, op)

    target = tmp_path / f"corpus_100mb.{case.ext}"
    target.write_bytes(source.encode("utf-8"))

    with _StopWatch(f"(b) after= insert [{stress_case.language}]"):
        run = run_real_edit(
            source, op.new_text, file_path=str(target),
            language=stress_case.language, engine=real_engine,
            after=op.symbol,
        )
        result = run.result
        m = metrics_tag(run)

        # ONE attempt, ZERO model tokens — the deterministic fast path.
        assert run.merge_results == [], (
            f"the after= fast path must never reach the model: {m}"
        )
        assert result.model_tokens == 0, f"non-zero tokens on a fast path: {m}"
        assert result.retries == 0, f"retries on a deterministic path: {m}"
        assert result.chunks_used == 0 and result.chunk_regions == [], (
            f"the fast path must not report chunk bookkeeping: {m}"
        )
        assert result.chunks_rejected == 0, m
        assert result.parse_valid, f"relative parse rule rejected: {m}"

        # Byte-exact vs the INDEPENDENT oracle.
        assert result.merged_code == expected, (
            f"after= insert diverges from the corpus oracle: "
            f"{first_diff_tag(expected, result.merged_code)} | {m}"
        )

        # Backward reconstruction with the oracle's own inverse op.
        restored = corpus.apply_line_op(result.merged_code, corpus.inverse_op(op))
        assert restored == source, (
            f"the insert inverse did not regenerate the original: "
            f"{first_diff_tag(source, restored)} | {m}"
        )
        print(f"[C2 stress] (b) observed: {m}", flush=True)

    assert target.read_bytes() == source.encode("utf-8"), (
        "chunked_merge must not write; the disk copy must stay the original"
    )


# ---------------------------------------------------------------------------
# (c) Multi-chunk LLM edit via the MCP full-stack path
# ---------------------------------------------------------------------------


def test_multi_chunk_llm_edit_two_distant_symbols(stress_case, mcp_harness, tmp_path):
    """Two distant wrap edits in ONE MCP batch call: transparent chunking,
    real model per chunk, byte-perfect rejoin, region bookkeeping proven."""
    require_stress_env()
    from fastedit.mcp import tools_edit

    cfg = _LANG_CONFIG[stress_case.language]
    case = stress_case
    source = str(case.source)
    source_lines = source.splitlines(keepends=True)
    spec, _delta = cfg.wrap(case.indent)
    span_b, span_c = _distant_spans(case, exclude={_replace_op(case).symbol})

    def snippet_for(span: corpus.SymbolSpan) -> str:
        return _wrap_snippet(
            source_lines[span.start_line - 1].rstrip("\r\n"),
            _span_kept_lines(source_lines, span, spec),
            case.indent, spec, case.eol,
        )

    edits_json = json.dumps([
        {"snippet": snippet_for(span_b), "replace": span_b.name},
        {"snippet": snippet_for(span_c), "replace": span_c.name},
    ])

    # Composed golden: wrap B, then wrap C at its post-edit-1 position.
    lines = list(source_lines)
    golden_b, delta_b = _wrap_expected("".join(lines), span_b, case.indent, spec, case.eol)
    lines = golden_b.splitlines(keepends=True)
    shifted_c = corpus.SymbolSpan(
        name=span_c.name, kind=span_c.kind,
        start_line=span_c.start_line + delta_b, end_line=span_c.end_line + delta_b,
    )
    golden_c, delta_c = _wrap_expected("".join(lines), shifted_c, case.indent, spec, case.eol)
    golden = golden_c
    expected_regions = [
        (span_b.start_line, span_b.end_line),
        (span_c.start_line + delta_b, span_c.end_line + delta_b),
    ]
    # The chunk texts the model may see: exactly the two manifest spans.
    span_b_text = _join(source_lines, span_b.start_line, span_b.end_line)
    span_c_text = _join(source_lines, span_c.start_line, span_c.end_line)

    target = tmp_path / f"corpus_100mb.{case.ext}"

    def check(message: str, merged: str) -> None:
        m = f"response={message!r}"
        merged_lines = merged.splitlines(keepends=True)

        # The tool's own bookkeeping: two edits, two chunks, all accepted.
        assert message.startswith(f"Applied 2 edits to {target}"), m
        assert "2 chunk(s), 2 edit(s)" in message, m
        assert "rejected" not in message and "Error" not in message, m

        # The model saw EXACTLY the two declared regions — nothing else.
        seen = _distinct(mcp_harness.recorded_chunks)
        assert len(seen) == 2, (
            f"expected exactly two distinct chunk texts (two sites), got "
            f"{len(seen)}: {[len(c) for c in seen]}"
        )
        assert seen[0] == span_b_text, (
            f"chunk 1 is not the manifest span of '{span_b.name}': "
            f"{first_diff_tag(span_b_text, seen[0])}"
        )
        assert seen[1] == span_c_text, (
            f"chunk 2 is not the manifest span of '{span_c.name}': "
            f"{first_diff_tag(span_c_text, seen[1])}"
        )
        assert expected_regions[0][1] - expected_regions[0][0] + 1 == len(
            span_b_text.splitlines(keepends=True),
        ), "region bookkeeping must agree with the span byte length"

        # Both edits landed (op spec, line-exact). The two wraps share the
        # declared guard lines — each must appear exactly twice.
        for span in (span_b, span_c):
            _assert_wrap_kept_lines(
                source, merged, span, case.indent, spec, case.eol, m,
                declared_line_count=2,
            )

        # Untouched regions byte-identical: head, the gap between the two
        # spans, and the tail.
        assert merged_lines[: span_b.start_line - 1] == source_lines[: span_b.start_line - 1], (
            f"head before the first span changed: {m}"
        )
        _assert_same_lines(
            source_lines, merged_lines,
            span_b.end_line + 1, span_c.start_line - 1,
            span_b.end_line + delta_b + 1, "middle gap between the two spans",
        )
        _assert_same_lines(
            source_lines, merged_lines,
            span_c.end_line + 1, len(source_lines),
            span_c.end_line + delta_b + delta_c + 1, "tail after the second span",
        )

        # Full-file relative validity, independently of fastedit.
        from fastedit.data_gen.ast_analyzer import parse_diagnostics

        merged_diags = parse_diagnostics(merged, case.language)
        assert merged_diags.is_valid, (
            f"merged file has parse errors: {merged_diags.errors[:3]} | {m}"
        )

        # THE golden assertion: the composed file, byte-exact.
        assert merged == golden, (
            f"multi-chunk rejoin differs from the composed golden: "
            f"{first_diff_tag(golden, merged)} | {m}"
        )

    with _StopWatch(f"(c) multi-chunk batch [{stress_case.language}]"):
        last_error: AssertionError | None = None
        for attempt in range(1, OUTER_ATTEMPTS + 1):
            # Each outer attempt starts from the pristine corpus on disk.
            target.write_bytes(source.encode("utf-8"))
            mcp_harness.recorded_chunks.clear()
            message = asyncio.run(tools_edit.fast_batch_edit(
                file_path=str(target), edits=edits_json,
            ))
            merged = target.read_bytes().decode("utf-8")
            try:
                check(message, merged)
                print(f"[C2 stress] (c) observed: {message.splitlines()[0]}", flush=True)
                break
            except AssertionError as exc:
                last_error = exc
        else:
            raise AssertionError(
                f"100MB multi-chunk batch did not converge within "
                f"{OUTER_ATTEMPTS} outer attempts — non-convergence is a "
                f"product defect (plan §0): {last_error}"
            )


# ---------------------------------------------------------------------------
# (d) Undo, full stack: N-deep backups at 100MB, byte-exact at every step
# ---------------------------------------------------------------------------


def test_undo_full_stack_restores_states_byte_exact(stress_case, mcp_harness, tmp_path):
    """Two stacked edits → two undos → every intermediate state byte-exact.

    Sequence: LLM wrap (one real model call, written by ``fast_edit``) →
    deterministic ``after=`` insert (second write) → ``fast_undo`` restores
    the post-wrap state (the N-deep proof: the first edit's backup survived
    the second write) → the second ``fast_undo`` restores the ORIGINAL bytes
    exactly → the ledger is empty and says so.
    """
    require_stress_env()
    from fastedit.mcp import tools_ast, tools_edit

    cfg = _LANG_CONFIG[stress_case.language]
    case = stress_case
    source = str(case.source)
    spec, _delta = cfg.wrap(case.indent)
    wrap_op = _replace_op(case)
    insert = _insert_op(case)
    span = case.source.manifest.span(wrap_op.symbol)
    source_lines = source.splitlines(keepends=True)
    golden_wrap, _ = _wrap_expected(source, span, case.indent, spec, case.eol)
    # Edit 2 (the deterministic insert) applies to the POST-WRAP file. Its
    # anchor sits before the wrap span (insert ~1/4, replace ~1/2), so the
    # recorded line indices are unchanged and the oracle composes directly
    # on the wrapped text.
    golden_insert = corpus.apply_op_oracle(golden_wrap, insert)
    original_bytes = source.encode("utf-8")

    target = tmp_path / f"corpus_100mb.{case.ext}"
    target.write_bytes(original_bytes)

    with _StopWatch(f"(d) undo full stack [{stress_case.language}]"):
        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=_wrap_snippet(
                source_lines[span.start_line - 1].rstrip("\r\n"),
                _span_kept_lines(source_lines, span, spec),
                case.indent, spec, case.eol,
            ),
            replace=wrap_op.symbol,
        ))
        assert response.startswith(f"Applied edit to {target}"), response
        assert "rejected" not in response and "Error" not in response, response
        assert target.read_bytes().decode("utf-8") == golden_wrap, (
            "the LLM wrap edit did not write the golden bytes (first_diff: "
            f"{first_diff_tag(golden_wrap, target.read_bytes().decode('utf-8'))})"
        )
        print(f"[C2 stress] (d) edit 1 observed: {response}", flush=True)

        # Edit 2: the deterministic insert — a second write on top.
        response = asyncio.run(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=insert.new_text,
            after=insert.symbol,
        ))
        assert response.startswith(f"Applied edit to {target}"), response
        assert target.read_bytes().decode("utf-8") == golden_insert, (
            "the after= insert did not write the oracle bytes"
        )

        # Undo #1 → the post-wrap state (previous state), byte-exact at 100MB.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes().decode("utf-8") == golden_wrap, (
            "undo #1 did not restore the post-wrap state byte-exactly "
            f"(first_diff: {first_diff_tag(golden_wrap, target.read_bytes().decode('utf-8'))})"
        )

        # Undo #2 → the ORIGINAL bytes, exactly.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith(f"Reverted {target}"), response
        assert target.read_bytes() == original_bytes, (
            "undo #2 did not restore the original file byte-exactly: "
            f"{first_diff_tag(source, target.read_bytes().decode('utf-8'))}"
        )

        # The undo ledger is empty — a third undo fails loud, changes nothing.
        response = asyncio.run(tools_ast.fast_undo(file_path=str(target)))
        assert response.startswith("Error: no undo history"), response
        assert target.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# (e) Backward reconstruction: the oracle's inverse op on the real output
# ---------------------------------------------------------------------------


def test_backward_reconstruction_from_real_edit(stress_case, real_engine, tmp_path):
    """The oracle's inverse op regenerates the original from the EDITED bytes
    the real pipeline produced (req. 3, backward reconstruction)."""
    require_stress_env()
    cfg = _LANG_CONFIG[stress_case.language]
    case = stress_case
    source = str(case.source)
    op = _replace_op(case)
    span = case.source.manifest.span(op.symbol)
    spec, _delta = cfg.wrap(case.indent)
    source_lines = source.splitlines(keepends=True)
    snippet = _wrap_snippet(
        source_lines[span.start_line - 1].rstrip("\r\n"),
        _span_kept_lines(source_lines, span, spec),
        case.indent, spec, case.eol,
    )
    golden, _line_delta = _wrap_expected(source, span, case.indent, spec, case.eol)
    inverse: list[corpus.LineOp] = []

    def check(run):
        result = run.result
        m = metrics_tag(run)
        merged = result.merged_code
        assert result.model_tokens > 0, f"the real model must run: {m}"
        assert result.chunks_rejected == 0 and result.parse_valid, m
        assert merged == golden, (
            f"merged file differs from the independent golden: "
            f"{first_diff_tag(golden, merged)} | {m}"
        )
        wrapped_span = _wrap_span_lines(
            source.splitlines(keepends=True)[span.start_line - 1 : span.end_line],
            case.indent, spec, case.eol,
        )
        inverse.clear()
        inverse.append(_wrap_inverse(op, len(wrapped_span)))

    with _StopWatch(f"(e) backward reconstruction [{stress_case.language}]"):
        last_metrics = ""
        for attempt in range(1, OUTER_ATTEMPTS + 1):
            run = run_real_edit(
                source, snippet, file_path=str(
                    tmp_path / f"corpus_100mb.{case.ext}",
                ),
                language=stress_case.language, engine=real_engine,
                replace=op.symbol,
            )
            try:
                check(run)
                print(f"[C2 stress] (e) observed: {metrics_tag(run)}", flush=True)
                break
            except AssertionError as exc:
                last_metrics = f"[outer attempt {attempt}/{OUTER_ATTEMPTS}] {exc}"
        else:
            raise AssertionError(
                f"backward-reconstruction edit did not converge within "
                f"{OUTER_ATTEMPTS} outer attempts — non-convergence is a "
                f"product defect (plan §0): {last_metrics}"
            )

    (inv,) = inverse
    restored = corpus.apply_line_op(golden, inv)
    assert restored == source, (
        f"the oracle's inverse op did not regenerate the original from the "
        f"REAL edited content: {first_diff_tag(source, restored)}"
    )
