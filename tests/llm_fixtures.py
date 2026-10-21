"""Real-LLM test foundation (implementation plan Phase A, Step A1).

Everything in this module drives the REAL trained fastedit model — the
mlx-8bit weights resolved through
:func:`fastedit.model_download.get_model_path` (the official
``FASTEDIT_MODEL_PATH`` → repo-local ``models/`` → ``~/.cache/fastedit``
→ auto-download-from-HF path). There is no fake LLM here and none is
permitted in the ``llm`` tier: the tier exists to exercise the model's
real context window, real tokenizer and real non-determinism.

Tier policy
-----------
``llm`` and ``stress`` are registered pytest markers (pyproject
``[tool.pytest.ini_options]``) and deselected from the DEFAULT tier via
``addopts`` — ``uv run pytest -q`` stays at the hermetic suite (model
load, prompt prefill and 100MB files are minutes-scale). The heavy tiers
run explicitly: ``uv run pytest -m llm``. The CLI ``-m`` overrides the
addopts default because pytest prepends addopts to the command line and
argparse's ``store`` action lets the later (CLI) occurrence win.
``stress`` tests additionally require ``FASTEDIT_RUN_STRESS=1``.

Skips are explicit, never silent: the :func:`real_engine` fixture skips
with a reason string ONLY after the official download path was attempted
and the model is still genuinely unavailable (no mlx runtime, download
failure, or a weight-less model directory).
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from fastedit.inference.chunked_merge import ChunkedMergeResult, chunked_merge
from fastedit.inference.merge import MergeResult

if TYPE_CHECKING:
    from fastedit.inference.mlx_engine import MLXEngine


# ---------------------------------------------------------------------------
# Availability probes (explicit-skip machinery — the repo's find_spec
# convention, cf. tests/test_cli.py's `_MLX_AVAILABLE` gating)
# ---------------------------------------------------------------------------


def mlx_runtime_available() -> bool:
    """True when the mlx + mlx-lm runtime is importable in this venv."""
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def resolve_real_model_path() -> str:
    """Resolve the trained model through the OFFICIAL resolution path.

    Mirrors ``cli.py``'s engine construction: ``get_model_path()`` walks
    ``FASTEDIT_MODEL_PATH`` → repo-local ``models/`` →
    ``~/.cache/fastedit/models/mlx-8bit`` and auto-downloads from HF
    ``continuous-lab/FastEdit`` when the cache carries no
    ``*.safetensors`` weights. Raises on genuine unavailability.
    """
    from fastedit.model_download import get_model_path

    return get_model_path()


@pytest.fixture(scope="session")
def real_engine(tmp_path_factory) -> MLXEngine:
    """ONE MLXEngine per pytest session — model load is the expensive part.

    Loads the real trained fastedit model exactly like ``cli.py`` does
    (``MLXEngine(get_model_path())``, default KV/quant settings) with one
    documented deviation: the persistent prompt-cache dir is pointed at a
    session-scoped tmp dir so tests never write into the real
    ``~/.fastedit/cache``.

    Skips — always with an explicit reason string — only when the real
    model is genuinely unavailable AFTER the official download attempt:

    * the mlx/mlx-lm runtime is not importable in this venv;
    * ``get_model_path()`` raised (download/network/auth failure);
    * the resolved directory still carries no ``*.safetensors`` weights.

    The fixture never fakes, stubs or silently degrades the tier.
    """
    if not mlx_runtime_available():
        pytest.skip(
            "mlx runtime not importable in this venv — the llm tier needs "
            "fastedit's [mlx] extra (uv sync --extra mlx); skipped with an "
            "explicit reason, never silently"
        )
    try:
        model_path = resolve_real_model_path()
    except Exception as exc:  # noqa: BLE001 -- download failures become explicit skips, not tracebacks
        pytest.skip(
            "real fastedit model unavailable after the official download "
            f"attempt ({type(exc).__name__}: {exc})"
        )
    weights = sorted(Path(model_path).glob("*.safetensors"))
    if not weights:
        pytest.skip(
            f"no *.safetensors weights at {model_path} after the official "
            "download attempt — the llm tier refuses to fake the model"
        )
    from fastedit.inference.mlx_engine import MLXEngine

    cache_dir = tmp_path_factory.mktemp("fastedit-engine-cache")
    return MLXEngine(model_path, cache_dir=str(cache_dir))


# ---------------------------------------------------------------------------
# Real edit driver — cli.py:521-529 wiring, with the real engine
# ---------------------------------------------------------------------------


@dataclass
class RealEditRun:
    """Everything one real edit produced.

    ``result`` is the pipeline-level :class:`ChunkedMergeResult`;
    ``merge_results`` captures every engine-level :class:`MergeResult`
    the pipeline produced (one per chunk attempt, including retries), so
    tests can prove the model actually ran and report measured
    tokens/latency.
    """

    result: ChunkedMergeResult
    merge_results: list[MergeResult]
    source_text: str
    snippet: str


def run_real_edit(
    source_text: str,
    snippet: str,
    file_path: str,
    language: str,
    *,
    engine: MLXEngine,
    **kwargs,
) -> RealEditRun:
    """Drive the REAL ``chunked_merge`` → ``engine.merge_auto`` pipeline.

    Wires ``merge_fn`` exactly like ``cli.py``'s lazy backend wrapper
    (cli.py:521-529): ``merge_fn(*a, **kw) -> engine.merge_auto(*a, **kw)``,
    handed to :func:`chunked_merge` as ``merge_fn`` together with
    ``language``/``after``/``replace``. The only difference from cli.py is
    laziness — the session engine is already loaded, so this wrapper
    captures each engine-level result for diagnostics while delegating
    untouched to ``merge_auto``.

    Args:
        source_text: Full original file content.
        snippet: The edit snippet (marker-bearing snippets force the model
            path — the deterministic editor declines genuine wrap_block
            shapes, see tests/test_add_guard_marker_position.py).
        file_path: Path recorded for the edit (for AST extraction; write
            the file yourself first if the edit should read from disk).
        language: Language for parse validation (e.g. ``"python"``).
        engine: The session ``real_engine`` fixture. Required — there is
            deliberately no fake default.
        **kwargs: Forwarded to ``chunked_merge`` verbatim (``after=``,
            ``replace=``, ``padding=``, ``preserve_siblings=``).

    Returns:
        :class:`RealEditRun` with the pipeline result and every captured
        engine-level :class:`MergeResult`.
    """
    captured: list[MergeResult] = []

    def merge_fn(original_code, snippet_text, lang=None, **_kw):
        # Mirrors cli.py's `_lazy_merge_fn` minus laziness: the session
        # engine is already loaded, so every chunk hits the real model.
        result = engine.merge_auto(original_code, snippet_text, lang)
        captured.append(result)
        return result

    result = chunked_merge(
        original_code=source_text,
        snippet=snippet,
        file_path=file_path,
        merge_fn=merge_fn,
        language=language,
        **kwargs,
    )
    return RealEditRun(
        result=result,
        merge_results=captured,
        source_text=source_text,
        snippet=snippet,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def first_diff_tag(a: str, b: str) -> str:
    """First differing offset for byte-exactness failures.

    Lifted from ``_dev_smoke/smoke_test.py``'s ``show_diff_tag`` pattern:
    identical inputs return ``""``; otherwise the first differing byte
    offset with both byte values, or a length tag when one side is a
    prefix of the other. The line carrying the first diff is appended so
    the failure message points straight at the corruption site.
    """
    if a == b:
        return ""
    ga, gb = a.encode("utf-8"), b.encode("utf-8")
    for i, (x, y) in enumerate(zip(ga, gb)):
        if x != y:
            line = ga[:i].count(b"\n") + 1
            return f"first diff at byte {i} (line {line}): got {x:#x} want {y:#x}"
    return f"length differs: got {len(ga)} want {len(gb)} bytes"


def metrics_tag(run: RealEditRun) -> str:
    """Measured real-model numbers for failure messages.

    Aggregates the pipeline-level accounting (model tokens, latency,
    chunk bookkeeping) with the last engine-level call's measurements, so
    every assertion failure in the llm tier carries the numbers needed to
    judge whether the model or the pipeline misbehaved.
    """
    parts = [
        f"merge_calls={len(run.merge_results)}",
        f"model_tokens={run.result.model_tokens}",
        f"latency_ms={run.result.latency_ms:.0f}",
        f"chunks_used={run.result.chunks_used}",
        f"chunks_rejected={run.result.chunks_rejected}",
    ]
    if run.merge_results:
        last = run.merge_results[-1]
        parts.append(f"last_call_tokens={last.tokens_generated}")
        parts.append(f"last_call_ttft_ms={last.ttft_ms:.0f}")
        parts.append(f"last_call_tok_per_s={last.tokens_per_second:.1f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Future-tier gate (Phase C): stress tests additionally require the env opt-in
# ---------------------------------------------------------------------------


def require_stress_env() -> None:
    """Skip helper for ``stress``-marked tests (plan §0 runtime tiers).

    ``stress`` tests are minutes-scale each (100MB files); beyond marker
    deselection they additionally require ``FASTEDIT_RUN_STRESS=1`` so no
    accidental default-tier invocation ever materializes them.
    """
    if os.environ.get("FASTEDIT_RUN_STRESS") != "1":
        pytest.skip(
            "stress tier disabled: set FASTEDIT_RUN_STRESS=1 to run the "
            "100MB stress tests"
        )
