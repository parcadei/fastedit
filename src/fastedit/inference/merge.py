"""High-level merge API for FastEdit.

Provides the main entry point for merging code edits:
  merged = merge(original_code, update_snippet, language)

Supports OpenAI-compatible backends: oMLX (recommended), vLLM, mlx-lm server, LM Studio.
Qwen3.5 thinking mode is disabled by default for maximum inference speed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data_gen.ast_analyzer import validate_parse
from ..data_gen.prompt_templates import INFERENCE_SYSTEM_PROMPT, INFERENCE_USER_PROMPT


@dataclass
class MergeResult:
    """Result of a merge operation."""
    merged_code: str
    parse_valid: bool
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    ttft_ms: float = 0.0


def _extract_output(text: str) -> str:
    """Extract merged code from model output tags."""
    # Strip closed <think>...</think> blocks (Qwen3 thinking mode leakage)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    for start_tag, end_tag in [
        ("<updated-code>", "</updated-code>"),
        ("<update-code>", "</update-code>"),
        ("<updated_code>", "</updated_code>"),
    ]:
        start_idx = text.find(start_tag)
        if start_idx == -1:
            continue
        end_idx = text.find(end_tag, start_idx + len(start_tag))
        if end_idx != -1:
            return text[start_idx + len(start_tag):end_idx].strip()
        # Truncated output: start tag but no end tag — take everything after start
        return text[start_idx + len(start_tag):].strip()

    # No code tags found at all — strip any unclosed <think> block
    think_idx = text.find('<think>')
    if think_idx != -1:
        text = text[:think_idx].strip()
    return text.strip()


def build_prompt(original_code: str, update_snippet: str) -> list[dict]:
    """Build the chat messages for the merge model."""
    return [
        {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": INFERENCE_USER_PROMPT.format(
                original_code=original_code,
                update_snippet=update_snippet,
            ),
        },
    ]


class FastEditEngine:
    """Inference engine for FastEdit model.

    Supports any OpenAI-compatible backend:
    - oMLX (recommended for Mac — persistent KV cache, near-zero TTFT on repeat files)
    - vLLM (Linux/GPU — speculative decoding support)
    - mlx-lm server (simple MLX serving)
    - LM Studio (GUI-based)

    Qwen3.5 thinking mode is disabled via chat_template_kwargs to avoid
    generating thousands of hidden reasoning tokens (which tank throughput
    from ~200 tok/s to <1 effective tok/s).
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "fastedit-4b",
        api_key: str = "not-needed",
        max_tokens: int = 16384,
        enable_thinking: bool = False,
    ):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._client

    def merge(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        """Merge an edit snippet into the original code.

        Args:
            original_code: The original source file content.
            update_snippet: The edit snippet with ellipsis markers.
            language: Optional language for post-merge validation.

        Returns:
            MergeResult with the merged code and performance metrics.
        """
        import time

        messages = build_prompt(original_code, update_snippet)
        client = self._get_client()

        # Qwen3.5 small models (0.8B-9B) have thinking disabled by default.
        # We still pass it explicitly as a safety net for backends that may
        # override the default. With thinking ON, throughput drops from
        # ~200 tok/s to <1 effective tok/s.
        extra_body = {}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            extra_body=extra_body or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        raw_output = response.choices[0].message.content
        merged_code = _extract_output(raw_output)

        tokens_generated = response.usage.completion_tokens if response.usage else 0
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        parse_valid = True
        if language:
            parse_valid = validate_parse(merged_code, language)

        return MergeResult(
            merged_code=merged_code,
            parse_valid=parse_valid,
            tokens_generated=tokens_generated,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
        )

    async def merge_async(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        """Async version of merge for batch processing."""
        import time

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        messages = build_prompt(original_code, update_snippet)

        # Qwen3.5 small models (0.8B-9B) have thinking disabled by default.
        # We still pass it explicitly as a safety net for backends that may
        # override the default. With thinking ON, throughput drops from
        # ~200 tok/s to <1 effective tok/s.
        extra_body = {}
        if not self.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            extra_body=extra_body or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        raw_output = response.choices[0].message.content
        merged_code = _extract_output(raw_output)

        tokens_generated = response.usage.completion_tokens if response.usage else 0
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        parse_valid = True
        if language:
            parse_valid = validate_parse(merged_code, language)

        return MergeResult(
            merged_code=merged_code,
            parse_valid=parse_valid,
            tokens_generated=tokens_generated,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
        )
