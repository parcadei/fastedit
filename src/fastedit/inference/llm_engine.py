"""OpenAI-compatible inference adapter for FastEdit.

This is a thin OpenAI-compatible client that works with any server
implementing the OpenAI Chat Completions API: vLLM, LM Studio,
llama.cpp, Ollama, etc.
"""

from __future__ import annotations

from ..data_gen.ast_analyzer import validate_parse
from .merge import MergeResult, _extract_output, build_prompt


class LLMEngine:
    """OpenAI-compatible adapter for any LLM server (vLLM, LM Studio, llama.cpp, Ollama)."""

    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "not-needed",
        max_tokens: int = 16384,
        enable_thinking: bool = False,
    ):
        from openai import OpenAI

        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._client = OpenAI(api_key=api_key, base_url=self.api_base)

    def merge(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        import time

        messages = build_prompt(original_code, update_snippet)
        extra_body = {}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            extra_body=extra_body or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        raw_output = response.choices[0].message.content or ""
        merged_code = _extract_output(raw_output)
        tokens_generated = response.usage.completion_tokens if response.usage else 0
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        parse_valid = True
        if language:
            parse_valid = validate_parse(merged_code, language)

        return MergeResult(
            merged_code=merged_code,
            parse_valid=parse_valid,
            tokens_generated=tokens_generated,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            ttft_ms=0.0,
        )

    def merge_auto(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        """Remote servers handle scheduling/batching, so merge_auto == merge."""
        return self.merge(original_code, update_snippet, language)
