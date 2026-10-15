"""Output-integrity tests (B11 — Step 9; B12 — Step 10; hosts Step 11 later).

The bug (B11): ``_extract_output`` in ``inference/merge.py`` returned the
text after a start tag even when the matching end tag never arrived —
"Truncated output: start tag but no end tag — take everything after
start". A truncated model response therefore became a *partial file
write*: the truncated payload flowed into ``MergeResult.merged_code``
like a clean merge and downstream consumers spliced/wrote it.

Contract locked down here (Step 9):

1. ``_extract_output`` raises the new ``TruncatedOutputError`` when a
   start tag arrives without its end tag — it never returns a partial
   payload as merged code.
2. Balanced tags → payload, unchanged.
3. Tag-less output → returned as-is (call sites own parse validity).
4. ``<think>`` blocks are still stripped — closed blocks anywhere, and
   an unclosed ``<think>`` in tag-less output (a truncated think block
   is not truncated *code*).
5. ``MergeResult`` carries ``truncated: bool = False``.
6. Engines never let ``TruncatedOutputError`` escape to callers that
   cannot handle it, and a truncated extraction never masquerades as
   success: the result is error-shaped — ``truncated=True`` and
   ``parse_valid=False`` (the universal "do not persist this" signal
   every existing consumer already gates on: chunked_merge retry
   paths, the MCP parse gate, the CLI refusal), with ``merged_code``
   keeping the best-effort payload for diagnostics only.

No network, no model, no backend: the OpenAI-compatible engines are
stubbed at the ``engine._client`` seam (the object ``_get_client``
returns / the attribute ``__init__`` sets) with plain namespaces.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from fastedit.inference.llm_engine import LLMEngine
from fastedit.inference.merge import (
    FastEditEngine,
    MergeResult,
    TruncatedOutputError,
    _extract_output,
)
from fastedit.inference.vllm_engine import VLLMEngine

# The tag families _extract_output recognizes (order matters: first match
# wins). Every family must obey the same truncation contract — a missing
# end tag is a truncated response regardless of which spelling the model
# used.
TAG_FAMILIES = [
    ("<updated-code>", "</updated-code>"),
    ("<update-code>", "</update-code>"),
    ("<updated_code>", "</updated_code>"),
]


# ---------------------------------------------------------------------------
# Step 9 — `_extract_output` strictness (B11)
# ---------------------------------------------------------------------------


class TestExtractOutputTruncation:
    """A start tag without its end tag must raise, never return partials."""

    @pytest.mark.parametrize("start_tag,end_tag", TAG_FAMILIES)
    def test_start_tag_without_end_tag_raises_truncated_output_error(
        self, start_tag, end_tag,
    ):
        raw = f"{start_tag}def foo():\n    return 1"
        with pytest.raises(TruncatedOutputError):
            _extract_output(raw)

    @pytest.mark.parametrize("start_tag,end_tag", TAG_FAMILIES)
    def test_end_tag_arriving_before_start_tag_raises(self, start_tag, end_tag):
        # A response that lost everything between the tags is truncated too:
        # the closer exists, but not after the opener.
        raw = f"{end_tag}{start_tag}def foo():"
        with pytest.raises(TruncatedOutputError):
            _extract_output(raw)

    def test_truncated_error_carries_partial_text_for_diagnostics(self):
        with pytest.raises(TruncatedOutputError) as excinfo:
            _extract_output("<updated-code>def foo():")
        assert excinfo.value.partial_text == "def foo():"

    def test_truncated_error_never_leaks_partial_as_merged_code(self):
        # The failure mode under remediation: the partial payload coming
        # back as the return value. The only sanctioned channel is the
        # exception's diagnostic attribute consumed by the engine.
        result = None
        with pytest.raises(TruncatedOutputError) as excinfo:
            result = _extract_output("<update-code>half-written}")
        assert result is None
        assert excinfo.value.partial_text == "half-written}"


class TestExtractOutputBalancedAndTagless:
    """Balanced tags and tag-less output keep today's behavior."""

    def test_balanced_tags_return_payload(self):
        raw = "<updated-code>def foo(): return 1</updated-code>"
        assert _extract_output(raw) == "def foo(): return 1"

    @pytest.mark.parametrize("start_tag,end_tag", TAG_FAMILIES)
    def test_balanced_tags_all_families_return_payload(self, start_tag, end_tag):
        raw = f"preamble {start_tag}\nbody\n{end_tag} trailing"
        assert _extract_output(raw) == "body"



    def test_no_tags_output_returned_as_is(self):
        # Content is returned verbatim (surrounding whitespace stripped —
        # pre-existing behavior); no parse gate and no rejection happens
        # here: the call-site policy owns parse validity.
        text = "def foo():\n    return 1\n"
        assert _extract_output(text) == text.strip()

    def test_tagless_output_is_not_interpreted_or_rejected(self):
        # Even prose-like output passes through untouched — the call
        # site (validate_parse / hallucination gate) decides its fate.
        assert _extract_output("I cannot perform this edit") == (
            "I cannot perform this edit"
        )

    def test_first_tag_family_wins_when_nested(self):
        raw = (
            "<updated-code>outer<update-code>inner</update-code></updated-code>"
        )
        assert _extract_output(raw) == "outer<update-code>inner</update-code>"


class TestExtractOutputThinkBlocks:
    """``<think>`` handling is preserved (Qwen3 thinking-mode leakage)."""

    def test_closed_think_block_stripped_before_tag_extraction(self):
        raw = (
            "<think>reasoning about the edit</think>"
            "<updated-code>def foo(): return 1</updated-code>"
        )
        assert _extract_output(raw) == "def foo(): return 1"

    def test_unclosed_think_block_stripped_in_tagless_output(self):
        raw = "partial reasoning<think>cut off mid-thought"
        assert _extract_output(raw) == "partial reasoning"

    def test_unclosed_think_block_does_not_hide_balanced_payload(self):
        # A truncated THINK block is not truncated code: with a balanced
        # tag pair still present, the payload must be extracted and no
        # TruncatedOutputError may fire.
        raw = "<think>runaway reasoning<updated-code>def foo(): return 1</updated-code>"
        assert _extract_output(raw) == "def foo(): return 1"


# ---------------------------------------------------------------------------
# Step 9 — MergeResult truncation contract (B11)
# ---------------------------------------------------------------------------


class TestMergeResultTruncationField:
    """``MergeResult`` gains ``truncated: bool = False`` (backward compat)."""

    def test_truncated_defaults_false(self):
        result = MergeResult(
            merged_code="x",
            parse_valid=True,
            tokens_generated=1,
            latency_ms=1.0,
            tokens_per_second=1.0,
        )
        assert result.truncated is False

    def test_truncated_is_settable(self):
        result = MergeResult(
            merged_code="x",
            parse_valid=False,
            tokens_generated=1,
            latency_ms=1.0,
            tokens_per_second=1.0,
            truncated=True,
        )
        assert result.truncated is True


# ---------------------------------------------------------------------------
# Step 9 — engines: truncated output never masquerades as success (B11)
# ---------------------------------------------------------------------------


def _fake_response(content: str, finish_reason: str | None = None) -> SimpleNamespace:
    """A minimal OpenAI chat-completion response carrying *content*.

    ``finish_reason`` mirrors the API field the engines must consult for
    B12; ``None`` simulates a stub or server that omits the field, which
    engines must treat as "not truncated".
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            ),
        ],
        usage=SimpleNamespace(completion_tokens=7),
    )


def _fake_openai_client(
    content: str, finish_reason: str | None = None,
) -> SimpleNamespace:
    """Fake OpenAI-compatible client stubbed at the engine seam.

    ``FastEditEngine._get_client`` returns ``self._client`` when already
    set, and the adapter engines keep the client on the same attribute —
    so assigning it is the boundary tests stub engines at. No network,
    no server, no model.
    """
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _fake_response(content, finish_reason),
            ),
        ),
    )


def _fake_async_create(content: str, finish_reason: str | None = None):
    async def create(**kwargs):
        return _fake_response(content, finish_reason)

    return create


class TestFastEditEngineTruncationFlag:
    """``FastEditEngine.merge`` flags truncation instead of propagating."""

    def test_truncated_output_returns_error_shaped_result(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client(
            "<updated-code>def foo():\n    return 1"
        )
        result = engine.merge(
            "def foo():\n    pass",
            "# ... existing code ...\n    return 1",
            language="python",
        )
        assert result.truncated is True
        # Error-shaped: a truncated extraction is never a clean merge.
        assert result.parse_valid is False

    def test_truncated_result_keeps_best_effort_text_for_diagnostics(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client("<updated-code>def foo():")
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is True
        assert result.merged_code == "def foo():"

    def test_balanced_output_is_not_truncated(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client(
            "<updated-code>def foo():\n    return 1</updated-code>"
        )
        result = engine.merge(
            "def foo():\n    pass", "# ... existing code ...",
        )
        assert result.truncated is False
        assert result.parse_valid is True
        assert result.merged_code == "def foo():\n    return 1"

    def test_tagless_output_is_not_truncated(self):
        # Call-site policy owns parse validity for tag-less output.
        engine = FastEditEngine()
        engine._client = _fake_openai_client("def foo():\n    return 1\n")
        result = engine.merge(
            "def foo():\n    pass", "# ... existing code ...",
        )
        assert result.truncated is False
        assert result.merged_code == "def foo():\n    return 1"

    def test_merge_async_truncated_output_sets_flag(self, monkeypatch):
        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=_fake_async_create("<updated-code>def foo():"),
                    ),
                )

        # merge_async constructs its own AsyncOpenAI; stub the import seam.
        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        engine = FastEditEngine()
        result = asyncio.run(
            engine.merge_async("def foo():\n    pass", "# ... existing code ...")
        )
        assert result.truncated is True
        assert result.parse_valid is False


class TestOpenAICompatibleAdapterTruncationFlag:
    """LLMEngine/VLLMEngine apply the same truncation contract."""

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_truncated_output_is_flagged_not_raised(self, engine_cls):
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client("<update-code>def foo(:")
        result = engine.merge(
            "def foo():\n    pass", "# ... existing code ...",
        )
        assert result.truncated is True
        assert result.parse_valid is False

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_balanced_output_is_not_truncated(self, engine_cls):
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>"
        )
        result = engine.merge(
            "def foo():\n    pass", "# ... existing code ...",
        )
        assert result.truncated is False
        assert result.parse_valid is True
        assert result.merged_code == "def foo(): return 1"


# ---------------------------------------------------------------------------
# Step 10 — engines: finish-reason / max-token truncation detection (B12)
# ---------------------------------------------------------------------------

# The bug (B12): no engine consulted `finish_reason` — an API response that
# stopped because it hit the token cap ("length") flowed into the merge as
# if it were complete. On MLX the output cap (`max(2048, input_tokens*2)`)
# could silently cut a large insertion mid-payload with no flag at all.
#
# Contract locked down here:
#
# 1. `finish_reason == "length"` (or an equivalent token-limit reason)
#    marks the MergeResult `truncated=True` even when the payload's tags
#    balance — the model never chose to stop.
# 2. `finish_reason == "stop"` (or a missing reason — stubs and older
#    servers omit it) is a model-chosen stop: `truncated=False`.
# 3. The finish-reason signal composes with the Step 9 extraction signal:
#    either one firing makes the result error-shaped (`truncated=True`,
#    `parse_valid=False`).
# 4. MLX: generation that filled the output-token cap without the model
#    emitting EOS is truncated; an EOS stop is not. The cap formula scales
#    with the input and stays finite.
# 5. chunked_merge: a truncated per-chunk result is a failed merge —
#    retried once through the existing retry machinery, never spliced,
#    and rejected (existing bookkeeping) if the retry is truncated too.
#    The whole-file path applies the same single-retry guard where its
#    retry machinery already exists (full validation is Step 11).


class TestFinishReasonTruncationAdapters:
    """LLMEngine/VLLMEngine flag length-capped responses (B12)."""

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    @pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
    def test_length_capped_response_is_truncated_even_with_balanced_tags(
        self, engine_cls, finish_reason,
    ):
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
            finish_reason=finish_reason,
        )
        result = engine.merge(
            "def foo():\n    pass", "# ... existing code ...", language="python",
        )
        # The tags balance and the payload parses — only the finish reason
        # can know this response was cut off at the token cap.
        assert result.truncated is True
        assert result.parse_valid is False

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_stop_reason_is_not_truncated(self, engine_cls):
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
            finish_reason="stop",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is False
        assert result.parse_valid is True
        assert result.merged_code == "def foo(): return 1"

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_missing_finish_reason_is_not_truncated(self, engine_cls):
        # Stubs and older OpenAI-compatible servers may omit the field
        # entirely; a missing reason must never read as truncation.
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is False

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_other_finish_reasons_are_not_truncated(self, engine_cls):
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
            finish_reason="tool_calls",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is False

    @pytest.mark.parametrize("engine_cls", [LLMEngine, VLLMEngine])
    def test_finish_reason_signal_composes_with_extraction_signal(
        self, engine_cls,
    ):
        # Either signal alone truncates: unbalanced tags with a clean
        # "stop" reason is still a truncated response (Step 9 contract).
        engine = engine_cls(
            api_base="http://localhost:9/v1", model="fastedit-4b",
        )
        engine._client = _fake_openai_client(
            "<updated-code>def foo(:", finish_reason="stop",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is True
        assert result.parse_valid is False


class TestFinishReasonTruncationFastEditEngine:
    """FastEditEngine (merge.py) applies the same finish-reason contract."""

    def test_length_capped_response_is_truncated(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
            finish_reason="length",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is True
        assert result.parse_valid is False

    def test_stop_reason_is_not_truncated(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client(
            "<updated-code>def foo(): return 1</updated-code>",
            finish_reason="stop",
        )
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.truncated is False
        assert result.parse_valid is True

    def test_merge_async_length_capped_response_is_truncated(self, monkeypatch):
        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=_fake_async_create(
                            "<updated-code>def foo(): return 1</updated-code>",
                            finish_reason="length",
                        ),
                    ),
                )

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        engine = FastEditEngine()
        result = asyncio.run(
            engine.merge_async("def foo():\n    pass", "# ... existing code ...")
        )
        assert result.truncated is True
        assert result.parse_valid is False


class TestMLXTokenCapTruncation:
    """MLX: cap exhaustion (not EOS) is truncation (B12).

    The generation loops run against the shared fake-MLX ecosystem from
    ``test_mlx_engine`` (no model download, no Apple-Silicon deps): the
    model is a callable returning pre-arranged token arrays and the
    tokenizer is a stub, so the cap decision is exercised at the exact
    seam the production loops use.
    """

    @pytest.fixture()
    def mlx_mocks(self):
        """Install the shared mlx/mlx_lm module mocks for one test."""
        from test_mlx_engine import _build_mlx_mocks

        modules, mocks = _build_mlx_mocks()
        saved = {}
        for name, mod in modules.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mod

        tok = mocks["tokenizer"]
        tok.eos_token_id = 151645  # Qwen3.5 EOS
        tok.encode.return_value = [1, 2, 3, 4, 5]

        def fake_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        ):
            contents = []
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    contents.append(message["content"])
                else:
                    contents.append(str(message))
            suffix = "<|assistant|>" if add_generation_prompt else ""
            return "".join(contents) + suffix

        tok.apply_chat_template.side_effect = fake_apply_chat_template
        # Balanced-tag payload: extraction alone would call these merges
        # clean, so only the cap signal can flag them.
        tok.decode.return_value = "<updated-code>def hello(): pass</updated-code>"

        yield mocks

        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        # Force a fresh import of the engine modules so they re-bind to
        # whatever mlx modules exist after this test (same reason
        # test_mlx_engine's fixture does this — classes are captured at
        # import time).
        for mod in (
            "fastedit.inference.mlx_engine",
            "fastedit.inference.cache_utils",
            "fastedit.inference.prefix_cache",
        ):
            sys.modules.pop(mod, None)

    def _engine(self, mlx_mocks, max_tokens=6):
        # max_tokens=6 pins the cap: min(6, max(floor, 4 * 5 input tokens))
        # = 6, so the fakes only need a handful of tokens to exhaust it.
        from fastedit.inference.mlx_engine import MLXEngine

        return MLXEngine(max_tokens=max_tokens)

    def test_ar_merge_hitting_token_cap_is_truncated(self, mlx_mocks):
        FakeArray = mlx_mocks["FakeArray"]
        # The model never emits EOS: the loop can only end at the cap.
        mlx_mocks["model"].side_effect = (
            lambda tokens, cache=None: FakeArray([[123]])
        )
        engine = self._engine(mlx_mocks)

        result = engine.merge("def foo(): pass", "def foo(): return 1")

        assert result.truncated is True
        assert result.parse_valid is False

    def test_ar_merge_stopping_on_eos_is_not_truncated(self, mlx_mocks):
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = mlx_mocks["tokenizer"].eos_token_id
        mlx_mocks["model"].side_effect = (
            lambda tokens, cache=None: FakeArray([[eos_id]])
        )
        engine = self._engine(mlx_mocks)

        result = engine.merge("def foo(): pass", "def foo(): return 1")

        assert result.truncated is False
        assert result.parse_valid is True

    def test_speculative_merge_hitting_token_cap_is_truncated(self, mlx_mocks):
        FakeArray = mlx_mocks["FakeArray"]
        # Tag-less decode so the speculative loop's tag probe never fires
        # and generation runs to the cap; the model never emits EOS.
        mlx_mocks["tokenizer"].decode.return_value = "partial merge output"
        mlx_mocks["model"].side_effect = (
            lambda tokens, cache=None: FakeArray([[123]])
        )
        engine = self._engine(mlx_mocks)

        result = engine.merge_speculative(
            "def foo(): pass", "def foo(): return 1",
        )

        assert result.truncated is True
        assert result.parse_valid is False

    def test_speculative_merge_stopping_on_eos_is_not_truncated(self, mlx_mocks):
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = mlx_mocks["tokenizer"].eos_token_id
        mlx_mocks["model"].side_effect = (
            lambda tokens, cache=None: FakeArray([[eos_id]])
        )
        engine = self._engine(mlx_mocks)

        result = engine.merge_speculative(
            "def foo(): pass", "def foo(): return 1",
        )

        assert result.truncated is False
        assert result.parse_valid is True

    def test_stopped_on_token_cap_helper_flags_cap_reached_without_eos(
        self, mlx_mocks,
    ):
        from fastedit.inference.mlx_engine import _stopped_on_token_cap

        eos = {151645}
        assert _stopped_on_token_cap([7, 7, 7], 3, eos) is True
        # EOS last: the model chose to stop — complete response.
        assert _stopped_on_token_cap([7, 151645], 3, eos) is False
        # Under the cap without EOS: the loop cannot have exited yet.
        assert _stopped_on_token_cap([7], 3, eos) is False
        assert _stopped_on_token_cap([], 3, eos) is False

    def test_output_cap_formula_is_scaled_finite_and_materially_larger(
        self, mlx_mocks,
    ):
        from fastedit.inference.mlx_engine import _compute_output_cap

        # Small inputs get the floor — materially above the old 2048.
        assert _compute_output_cap(16384, 10) == 8192
        assert _compute_output_cap(16384, 100) == 8192
        # Large inputs scale with the input but never exceed max_tokens.
        assert _compute_output_cap(16384, 5000) == 16384
        assert _compute_output_cap(30000, 5000) == 20000
        assert _compute_output_cap(4096, 10**9) == 4096
        # Materially larger than the old effective formula below saturation
        # (the old code was min(max_tokens, max(2048, 2 * input))), and
        # identical only once both saturate at the finite max_tokens bound.
        for input_tokens in (10, 100, 1000):
            old_effective_cap = min(16384, max(2048, input_tokens * 2))
            new_cap = _compute_output_cap(16384, input_tokens)
            assert new_cap > old_effective_cap
            assert new_cap <= 16384
        # Both saturate at the same finite hard bound on huge inputs.
        assert _compute_output_cap(16384, 10_000) == 16384


# ---------------------------------------------------------------------------
# Step 10 — chunked_merge: a truncated chunk result is retried, never spliced
# ---------------------------------------------------------------------------

# Three-function file: the snippet names beta, so locate_chunks yields a
# beta-sized chunk (not the whole file) and the per-chunk loop runs.
CHUNK_FILE_ORIGINAL = (
    "def alpha():\n"
    '    return "a"\n'
    "\n"
    "\n"
    "def beta():\n"
    "    total = 1\n"
    "    return total\n"
    "\n"
    "\n"
    "def gamma():\n"
    '    return "g"\n'
)

# Pure-context snippet: any output keeping beta's lines verbatim passes the
# hallucination validator, so the ONLY failure signal in these tests is the
# engine's `truncated` flag (B12) — isolating the new guard from the
# content validator.
CHUNK_FILE_SNIPPET = (
    "def beta():\n"
    "    total = 1\n"
    "# ... existing code ...\n"
    "    return total\n"
)

BETA_MERGED = "def beta():\n    total = 1\n    return total\n"

# Single-function file: the snippet names foo, the foo chunk spans the
# whole file, so the whole-file merge branch runs.
WHOLE_FILE_ORIGINAL = "def foo():\n    return 1\n"
WHOLE_FILE_SNIPPET = (
    "def foo():\n"
    "# ... existing code ...\n"
    "    return 1\n"
)


class _StubMergeResult:
    """Minimal engine-result stand-in (same shape the existing
    chunked_merge tests stub with) plus the Step 9 truncation flags."""

    def __init__(self, merged_code, truncated=False, parse_valid=True):
        self.merged_code = merged_code
        self.parse_valid = parse_valid
        self.tokens_generated = 9
        self.latency_ms = 1.0
        self.truncated = truncated


def _run_chunked_merge(fake_results, tmp_path):
    """Drive chunked_merge with a stubbed engine; return (result, calls).

    Step A2 triage: chunked_merge now runs the unified retry-until-valid
    loop (implementation plan Step A2, req. 5) with a default budget of 8
    retries (FASTEDIT_MAX_RETRIES). These unit tests pin the OLD
    single-retry budget explicitly (``max_validation_retries=1``) so they
    keep proving exactly what they proved before — one corrective retry,
    then rejection — without stubbing nine identical failures. The
    production default is unchanged; no rejection guarantee is weakened.
    """
    from fastedit.inference.chunked_merge import chunked_merge

    calls = []

    def merge_fn(code, snippet, language):
        calls.append(code)
        return fake_results[min(len(calls) - 1, len(fake_results) - 1)]

    file_path = tmp_path / "mod.py"
    file_path.write_text(CHUNK_FILE_ORIGINAL)
    result = chunked_merge(
        CHUNK_FILE_ORIGINAL,
        CHUNK_FILE_SNIPPET,
        str(file_path),
        merge_fn,
        max_validation_retries=1,
    )
    return result, calls


def _run_whole_file_merge(fake_results, tmp_path):
    """Drive the whole-file merge branch with a stubbed engine.

    Step A2 triage: see _run_chunked_merge — the old single-retry budget is
    pinned explicitly so the exact-call assertions below keep testing the
    same one-retry-then-reject contract.
    """
    from fastedit.inference.chunked_merge import chunked_merge

    calls = []

    def merge_fn(code, snippet, language):
        calls.append(code)
        return fake_results[min(len(calls) - 1, len(fake_results) - 1)]

    file_path = tmp_path / "solo.py"
    file_path.write_text(WHOLE_FILE_ORIGINAL)
    result = chunked_merge(
        WHOLE_FILE_ORIGINAL,
        WHOLE_FILE_SNIPPET,
        str(file_path),
        merge_fn,
        max_validation_retries=1,
    )
    return result, calls


class TestChunkedMergeTruncatedChunkHandling:
    """A truncated chunk result is retried once, then rejected — never
    spliced (B12). The whole-file path gets the same single-retry guard
    through its existing retry machinery."""

    def test_truncated_chunk_retried_once_then_clean_result_spliced(
        self, tmp_path,
    ):
        truncated = _StubMergeResult(
            BETA_MERGED, truncated=True, parse_valid=False,
        )
        clean = _StubMergeResult(BETA_MERGED)

        result, calls = _run_chunked_merge([truncated, clean], tmp_path)

        # Exactly one retry — the same single-retry budget parse failures get.
        assert len(calls) == 2
        assert result.chunks_used == 1
        assert result.chunks_rejected == 0

    def test_chunk_truncated_twice_is_rejected_and_never_spliced(self, tmp_path):
        truncated = _StubMergeResult(
            BETA_MERGED, truncated=True, parse_valid=False,
        )

        result, calls = _run_chunked_merge([truncated], tmp_path)

        assert len(calls) == 2
        assert result.chunks_rejected == 1
        # The beta chunk was NOT spliced: the file is untouched.
        assert result.merged_code == CHUNK_FILE_ORIGINAL

    def test_clean_chunk_result_is_not_retried(self, tmp_path):
        clean = _StubMergeResult(BETA_MERGED)

        result, calls = _run_chunked_merge([clean], tmp_path)

        assert len(calls) == 1
        assert result.chunks_rejected == 0
        assert result.merged_code == CHUNK_FILE_ORIGINAL

    def test_whole_file_truncated_result_retried_once(self, tmp_path):
        truncated = _StubMergeResult(
            "def foo():\n    return 1", truncated=True, parse_valid=False,
        )
        clean = _StubMergeResult(WHOLE_FILE_ORIGINAL.strip())

        result, calls = _run_whole_file_merge([truncated, clean], tmp_path)

        assert len(calls) == 2
        assert result.parse_valid is True

    def test_whole_file_double_truncated_stays_error_shaped(self, tmp_path):
        truncated = _StubMergeResult(
            "def foo():\n    return 1", truncated=True, parse_valid=False,
        )

        result, calls = _run_whole_file_merge([truncated], tmp_path)

        assert len(calls) == 2
        # Still error-shaped so the MCP/CLI parse gates refuse the write
        # (full whole-file content validation is Step 11).
        assert result.parse_valid is False

    def test_whole_file_clean_result_is_not_retried(self, tmp_path):
        clean = _StubMergeResult(WHOLE_FILE_ORIGINAL.strip())

        result, calls = _run_whole_file_merge([clean], tmp_path)

        assert len(calls) == 1
        assert result.parse_valid is True


# ---------------------------------------------------------------------------
# Step 11 — whole-file merge path content validation (B13, B29)
# ---------------------------------------------------------------------------

# The bug (B13/B29): the whole-file merge branch — the fallback used when
# chunk location yields a single chunk spanning the entire file (single-
# symbol files, no-AST files, unmatched snippets) — handed the ENTIRE file
# to the 1.7B model and validated the result only against the truncation
# flag and (when the language is known) a parse check. Whatever came back
# after that one retry was used as-is: a merge that silently DROPPED an
# original line the snippet never mentioned, INVENTED code, or leaked a
# preservation marker parsed fine and flowed straight into the write path.
#
# Contract locked down here:
#
# 1. The whole-file merge output must pass the SAME content validator the
#    per-chunk path is held to: `_check_hallucinations` over the FULL
#    original vs the FULL merge output with the full snippet (preserve-by-
#    default: dropped unmentioned lines, inventions, reorders and marker
#    leaks all fail).
# 2. Parse validity is required whenever the language is known.
# 3. A failed first attempt is retried ONCE with a corrective note appended
#    to the prompt (the snippet is the only prompt channel merge_fn has;
#    the retry is validated against the ORIGINAL snippet so the note cannot
#    contaminate the gate).
# 4. A second failure REJECTS the whole-file merge, using the chunk loop's
#    rejection convention: merged_code keeps the original file (never the
#    corrupted payload), parse_valid is forced False (the universal
#    "do not persist" signal), and chunks_rejected=1/chunks_used=1 makes
#    the existing MCP/CLI gates refuse the write naturally.
# 5. A truncated result on this path gets the same single retry and is
#    rejected when the retry truncates too — closing the gap flagged in
#    the Step 10 report (a truncated whole-file payload previously stayed
#    the merged_code even after the retry).
# 6. Faithful output passes through unchanged — including a legitimate
#    edit that adds declared new lines — so the gate never blocks a clean
#    merge.

# Single-function, three-line file: the snippet names foo, the foo chunk
# spans the whole file, so the whole-file merge branch runs (same seam the
# Step 10 whole-file tests use).
WHOLE3_ORIGINAL = (
    "def foo():\n"
    "    total = 1\n"
    "    return total\n"
)

# Pure-context snippet: the faithful merge for this edit IS the original.
WHOLE3_CONTEXT_SNIPPET = (
    "def foo():\n"
    "# ... existing code ...\n"
    "    return total\n"
)

# The preserve-by-default violation: the model dropped the unmentioned
# original line `total = 1` (B4-class corruption that parses fine).
WHOLE3_DROPPED_LINE = "def foo():\n    return total\n"

# Marker leakage: the model echoed the preservation marker into the file.
WHOLE3_MARKER_LEAK = (
    "def foo():\n"
    "    total = 1\n"
    "# ... existing code ...\n"
    "    return total\n"
)


def _run_whole_file_merge_stubbed(
    fake_results,
    tmp_path,
    language=None,
    original=WHOLE3_ORIGINAL,
    snippet=WHOLE3_CONTEXT_SNIPPET,
):
    """Drive the whole-file merge branch, recording (code, snippet, lang).

    Mirrors ``_run_whole_file_merge`` but records the snippet argument too,
    so tests can observe the corrective note on the retry call.

    Step A2 triage: the unified retry-until-valid loop's default budget is
    8 retries; these tests pin the OLD single-retry budget explicitly so
    their exact-call assertions (and the "retried once" contract they lock)
    stay valid without stubbing nine identical failures.
    """
    from fastedit.inference.chunked_merge import chunked_merge

    calls = []

    def merge_fn(code, snip, lang):
        calls.append((code, snip, lang))
        return fake_results[min(len(calls) - 1, len(fake_results) - 1)]

    file_path = tmp_path / "solo_whole.py"
    file_path.write_text(original)
    result = chunked_merge(
        original, snippet, str(file_path), merge_fn, language=language,
        max_validation_retries=1,
    )
    return result, calls


class TestWholeFileMergeContentValidation:
    """The whole-file merge output is content-validated, retried once with
    a corrective note, and rejected — never returned corrupted (B13/B29)."""

    def test_dropped_unmentioned_original_line_is_rejected(self, tmp_path):
        # The model (stub) keeps returning the file WITHOUT `total = 1` —
        # a line the snippet never mentioned dropping. Parses fine, but it
        # is a preserve-by-default violation (B4-class).
        drop = _StubMergeResult(WHOLE3_DROPPED_LINE)

        result, calls = _run_whole_file_merge_stubbed([drop], tmp_path)

        # Exactly one corrective retry — no more.
        assert len(calls) == 2
        # Rejection convention: error-shaped plus chunk-rejection
        # accounting, so the MCP/CLI gates refuse the write naturally.
        assert result.chunks_rejected == 1
        assert result.chunks_used == 1
        assert result.parse_valid is False
        # No usable merged output: the original file is kept — never the
        # corrupted merge.
        assert result.merged_code == WHOLE3_ORIGINAL

    def test_parse_invalid_whole_file_output_is_rejected(self, tmp_path):
        # Not truncated, content-preserved — but it does not parse. With
        # the language known, parse validity is required on this path too.
        broken = _StubMergeResult("def foo(:\n    return total\n")

        result, calls = _run_whole_file_merge_stubbed(
            [broken], tmp_path, language="python",
        )

        assert len(calls) == 2
        assert result.chunks_rejected == 1
        assert result.chunks_used == 1
        assert result.parse_valid is False
        assert result.merged_code == WHOLE3_ORIGINAL

    def test_faithful_whole_file_merge_passes_through_unchanged(self, tmp_path):
        faithful = _StubMergeResult(WHOLE3_ORIGINAL)

        result, calls = _run_whole_file_merge_stubbed([faithful], tmp_path)

        assert len(calls) == 1  # clean on the first attempt — no retry
        assert result.parse_valid is True
        assert result.chunks_rejected == 0
        assert result.chunks_used == 1
        assert result.merged_code == WHOLE3_ORIGINAL

    def test_legitimate_new_line_edit_passes_the_gate(self, tmp_path):
        # The gate must not over-reject: an edit that adds a declared new
        # line (replacing the return expression) is faithful and accepted.
        snippet = (
            "def foo():\n"
            "    total = 1\n"
            "# ... existing code ...\n"
            "    return total + 1\n"
        )
        edited = _StubMergeResult(
            "def foo():\n    total = 1\n    return total + 1\n",
        )

        result, calls = _run_whole_file_merge_stubbed(
            [edited], tmp_path, snippet=snippet,
        )

        assert len(calls) == 1
        assert result.chunks_rejected == 0
        assert result.parse_valid is True
        assert result.merged_code == (
            "def foo():\n    total = 1\n    return total + 1\n"
        )

    def test_first_attempt_hallucinated_retry_clean_is_accepted(self, tmp_path):
        drop = _StubMergeResult(WHOLE3_DROPPED_LINE)
        faithful = _StubMergeResult(WHOLE3_ORIGINAL)

        result, calls = _run_whole_file_merge_stubbed([drop, faithful], tmp_path)

        assert len(calls) == 2
        assert result.parse_valid is True
        assert result.chunks_rejected == 0
        assert result.merged_code == WHOLE3_ORIGINAL
        # The retry prompt carries a corrective note; the first call does
        # not. The note rides on the snippet (merge_fn's only prompt
        # channel).
        assert "NOTE:" not in calls[0][1]
        assert "NOTE:" in calls[1][1]
        assert "preserve every original line" in calls[1][1]

    def test_both_attempts_marker_leaking_are_rejected(self, tmp_path):
        leak = _StubMergeResult(WHOLE3_MARKER_LEAK)

        result, calls = _run_whole_file_merge_stubbed([leak], tmp_path)

        assert len(calls) == 2
        assert result.chunks_rejected == 1
        assert result.chunks_used == 1
        assert result.parse_valid is False
        assert result.merged_code == WHOLE3_ORIGINAL

    def test_truncated_whole_file_result_retried_once_then_rejected(
        self, tmp_path,
    ):
        # Closes the Step 10 gap: a truncated whole-file response used to
        # stay the merged_code even after the single retry. Now the second
        # failure rejects the merge outright.
        truncated = _StubMergeResult(
            "def foo():\n    return total", truncated=True, parse_valid=False,
        )

        result, calls = _run_whole_file_merge_stubbed([truncated], tmp_path)

        assert len(calls) == 2
        assert result.chunks_rejected == 1
        assert result.chunks_used == 1
        assert result.parse_valid is False
        assert result.merged_code == WHOLE3_ORIGINAL


class TestFastEditEngineNullContent:
    """A server that returns ``content=None`` must not crash the engine.

    OpenAI-compatible servers return ``choices[0].message.content = null``
    for refusals, tool-call turns and some error shapes. The adapter
    engines (LLMEngine/VLLMEngine) guard with ``or ""`` — FastEditEngine
    drifted from that contract and handed ``None`` straight into
    ``_extract_output``, where ``re.sub`` raised an opaque TypeError.
    """

    def test_merge_null_content_returns_empty_result(self):
        engine = FastEditEngine()
        engine._client = _fake_openai_client(None)
        result = engine.merge("def foo():\n    pass", "# ... existing code ...")
        assert result.merged_code == ""
        assert result.truncated is False

    def test_merge_async_null_content_returns_empty_result(self, monkeypatch):
        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=_fake_async_create(None),
                    ),
                )

        monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
        engine = FastEditEngine()
        result = asyncio.run(
            engine.merge_async("def foo():\n    pass", "# ... existing code ...")
        )
        assert result.merged_code == ""
        assert result.truncated is False
