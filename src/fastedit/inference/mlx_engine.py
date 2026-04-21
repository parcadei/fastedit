"""In-process MLX inference engine for FastEdit.

Replaces the HTTP-based FastEditEngine with direct mlx-lm calls
for cache control and speculative decoding support.

Requires: mlx, mlx-lm (installed in .mlx-venv, not the training venv).
"""

from __future__ import annotations

import functools
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import make_prompt_cache

from ..data_gen.ast_analyzer import validate_parse

# --- Re-export all public types and functions for backward compatibility ---
# All existing `from fastedit.inference.mlx_engine import X` imports continue to work.
from .cache_utils import (  # noqa: F401
    _clone_cache,
    _compute_cache_key,
    _find_realign_point,
    _get_prefix_boundary,
    _prefill_prompt,
    _prefill_with_snapshots,
    _restore_ssm_caches,
    _snapshot_ssm_caches,
    _trim_kv_caches,
)
from .merge import MergeResult, _extract_output, build_prompt
from .prefix_cache import (  # noqa: F401
    PromptCacheManager,
    TokenPrefixCache,
)


def _speculative_generate(
    model,
    tokenizer,
    cache: list,
    prefill_logits,
    original_draft_tokens: list[int],
    draft_batch_size: int,
    max_tokens: int,
    kv_bits: int,
    kv_group_size: int,
    eos_token_ids: set[int],
) -> list[int]:
    """Core speculative generation loop using original file tokens as draft.

    Four phases:
    1. Generate autoregressively until <updated-code> tag is detected.
    2. Use original file tokens as speculative draft for batch verification.
       On rejection, search forward in draft for re-alignment point.
    2b. If diverged (edit boundary), generate autoregressively until the
        model output re-aligns with the draft.
    3. After draft exhausted, generate remaining tokens autoregressively.

    Args:
        model: The MLX model.
        tokenizer: The tokenizer (for decoding tokens to detect the tag).
        cache: The model's cache list (modified in place).
        prefill_logits: Logits from _prefill_prompt (last prompt position).
        original_draft_tokens: Tokenized original file content to use as draft.
        draft_batch_size: Number of draft tokens to verify per batch.
        max_tokens: Maximum total tokens to generate.
        kv_bits: KV cache quantization bits.
        kv_group_size: KV cache quantization group size.
        eos_token_ids: Set of token IDs that signal end of generation.

    Returns:
        List of generated token IDs.
    """
    min_draft_batch_size = 2
    max_consecutive_rejections = 2
    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    generated_tokens: list[int] = []

    # Get first token from prefill logits (already computed by _prefill_prompt)
    first_token_arr = mx.argmax(prefill_logits[:, -1, :], axis=-1)
    mx.eval(first_token_arr)
    first_token = first_token_arr.item()
    generated_tokens.append(first_token)

    if first_token in eos_token_ids:
        return generated_tokens

    # Phase 1: Generate autoregressively until <updated-code> tag is detected.
    UPDATED_CODE_TAG = "<updated-code>"  # noqa: N806
    tag_found = False

    y = mx.array([first_token], dtype=mx.uint32)

    while len(generated_tokens) < max_tokens:
        decoded = tokenizer.decode(generated_tokens)
        if UPDATED_CODE_TAG in decoded:
            tag_found = True
            break

        logits = model(y[None], cache=cache)
        quantize_fn(cache)
        next_token_arr = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token_arr)
        next_token = next_token_arr.item()
        generated_tokens.append(next_token)

        if next_token in eos_token_ids:
            return generated_tokens

        y = mx.array([next_token], dtype=mx.uint32)

    if not tag_found:
        # Tag never appeared -- fall through to phase 3 (autoregressive finish)
        pass

    # Phase 2: Speculative verification with re-alignment on rejection
    draft_idx = 0
    current_batch_size = max(min_draft_batch_size, draft_batch_size)
    consecutive_rejections = 0
    saw_bridge = False
    y = mx.array([generated_tokens[-1]], dtype=mx.uint32)

    while draft_idx < len(original_draft_tokens) and len(generated_tokens) < max_tokens:
        n_draft = min(current_batch_size, len(original_draft_tokens) - draft_idx)
        draft = original_draft_tokens[draft_idx:draft_idx + n_draft]
        draft_mx = mx.array(draft, dtype=mx.uint32)

        # Snapshot SSM caches before verification
        ssm_snapshot = _snapshot_ssm_caches(cache)

        # Forward pass: verify current token + all draft tokens together
        verify_tokens = mx.concatenate([y, draft_mx])
        logits = model(verify_tokens[None], cache=cache)
        quantize_fn(cache)

        # Get model's top predictions at each position
        model_tokens = mx.argmax(logits[0], axis=-1)
        mx.eval(model_tokens)
        model_tokens_list = model_tokens.tolist()

        # Count consecutive accepted tokens
        n_accepted = 0
        for i in range(n_draft):
            if model_tokens_list[i] == draft[i]:
                n_accepted += 1
            else:
                break

        # Emit accepted tokens
        for i in range(n_accepted):
            generated_tokens.append(draft[i])
            if draft[i] in eos_token_ids or len(generated_tokens) >= max_tokens:
                return generated_tokens

        if n_accepted < n_draft:
            # REJECTION at position n_accepted
            consecutive_rejections += 1
            current_batch_size = max(
                min_draft_batch_size, current_batch_size // 2,
            )
            model_token = model_tokens_list[n_accepted]
            generated_tokens.append(model_token)
            if model_token in eos_token_ids or len(generated_tokens) >= max_tokens:
                return generated_tokens

            # Rewind caches: trim entire batch from KV, restore SSM,
            # replay only accepted portion to rebuild both correctly.
            _trim_kv_caches(cache, n_draft + 1)
            _restore_ssm_caches(cache, ssm_snapshot)

            replay = mx.concatenate([y, draft_mx[:n_accepted]])
            model(replay[None], cache=cache)
            quantize_fn(cache)
            mx.eval([c.state for c in cache])

            # Try to re-align: search forward in draft for model_token.
            # This handles edit boundaries where the model inserts/modifies
            # code that doesn't exist in the original file.
            search_from = draft_idx + n_accepted + 1
            realign = _find_realign_point(
                original_draft_tokens, search_from, model_token,
            )

            if realign >= 0 and realign > search_from:
                # Found alignment point ahead -- generate autoregressively
                # through the divergent section until we reach it.
                y = mx.array([model_token], dtype=mx.uint32)
                while len(generated_tokens) < max_tokens:
                    logits = model(y[None], cache=cache)
                    quantize_fn(cache)
                    nt_arr = mx.argmax(logits[:, -1, :], axis=-1)
                    mx.eval(nt_arr)
                    nt = nt_arr.item()
                    generated_tokens.append(nt)
                    if nt in eos_token_ids:
                        return generated_tokens

                    # Check if model's output matches the draft at realign point
                    if (realign < len(original_draft_tokens)
                            and nt == original_draft_tokens[realign]):
                        # Re-aligned! Skip to this position in draft.
                        saw_bridge = True
                        draft_idx = realign + 1
                        # Also skip consecutive matching tokens after realign
                        if (draft_idx < len(original_draft_tokens)
                                and nt == original_draft_tokens[draft_idx - 1]):
                            pass  # draft_idx is already past the matched token
                        y = mx.array([nt], dtype=mx.uint32)
                        break

                    # Not aligned yet -- keep generating AR.
                    # Also scan for alignment at subsequent draft positions
                    # in case the model produced the alignment token later.
                    new_realign = _find_realign_point(
                        original_draft_tokens, realign, nt,
                    )
                    if new_realign >= 0:
                        realign = new_realign
                    y = mx.array([nt], dtype=mx.uint32)
                else:
                    # Exhausted max_tokens during AR bridge
                    break
            else:
                # No alignment found -- advance by one and try again
                y = mx.array([model_token], dtype=mx.uint32)
                draft_idx = search_from

            # Once we've had to bridge through a divergent region, the draft
            # typically stops being a net win. Finish the tail autoregressively.
            if saw_bridge:
                break

            # Repeated rejections indicate the draft is not helping anymore.
            # Fall back to plain autoregressive generation for the tail.
            if consecutive_rejections >= max_consecutive_rejections:
                break
        else:
            # ALL ACCEPTED
            consecutive_rejections = 0
            if current_batch_size < draft_batch_size:
                current_batch_size += 1
            next_token = model_tokens_list[n_draft]
            generated_tokens.append(next_token)
            if next_token in eos_token_ids or len(generated_tokens) >= max_tokens:
                return generated_tokens
            y = mx.array([next_token], dtype=mx.uint32)
            draft_idx += n_draft

            # Skip bonus if it matches next draft position
            if (draft_idx < len(original_draft_tokens)
                    and next_token == original_draft_tokens[draft_idx]):
                draft_idx += 1

    # Phase 3: Autoregressive for remaining tokens after draft exhausted
    while len(generated_tokens) < max_tokens:
        logits = model(y[None], cache=cache)
        quantize_fn(cache)
        next_token_arr = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token_arr)
        next_token = next_token_arr.item()
        generated_tokens.append(next_token)
        if next_token in eos_token_ids:
            break
        y = mx.array([next_token], dtype=mx.uint32)

    return generated_tokens


class MLXEngine:
    """In-process MLX inference engine for FastEdit.

    Loads a quantized MLX model directly into memory and generates
    token-by-token via ``generate_step`` for full cache control.
    Uses TurboQuant KV cache (4-bit, group-size 64) by default.

    Supports persistent prompt caching: when editing the same file
    repeatedly, the processed KV state for the prompt prefix can be
    saved to disk and reloaded on subsequent calls, achieving
    near-zero TTFT on repeat edits.

    Args:
        model_path: Path to the MLX model directory.
        kv_bits: Number of bits for KV cache quantization.
        kv_group_size: Group size for KV cache quantization.
        max_tokens: Maximum tokens to generate per merge call.
        cache_dir: Directory for persistent prompt caches.
            Defaults to ~/.fastedit/cache/.
        max_cache_bytes: Maximum total prompt cache size in bytes.
            Default 2GB.
    """

    def __init__(
        self,
        model_path: str = "models/fastedit-1.7b-mlx-8bit",
        kv_bits: int = 8,
        kv_group_size: int = 64,
        max_tokens: int = 16384,
        cache_dir: str | None = None,
        max_cache_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        self.model, self.tokenizer = load(model_path)
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.max_tokens = max_tokens
        self.cache_manager = PromptCacheManager(
            cache_dir=cache_dir,
            max_cache_bytes=max_cache_bytes,
        )
        # Level 1 - In-memory prefix cache: {cache_key: (cloned_cache, n_prefix_tokens)}
        # For repeat edits on the same file, the prompt prefix (system + template
        # + original_code) is constant. Caching its KV state lets us skip
        # re-prefilling ~80% of the prompt on subsequent calls.
        self._prefix_cache: dict[str, tuple[list, int]] = {}

        # Level 2 - Token prefix cache for cross-version reuse (chained edits).
        # When the output of edit 1 becomes the input of edit 2, the file content
        # changes (hash miss on Level 1) but shares a large token prefix with the
        # previous version. This cache finds the longest common prefix and restores
        # from the nearest snapshot boundary.
        self._token_cache = TokenPrefixCache()

    def merge(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        """Merge an edit snippet into the original code.

        Builds a chat prompt, runs autoregressive generation with
        TurboQuant KV cache, extracts the merged code from model
        output tags, and optionally validates the parse.

        Uses in-memory prefix caching: for repeat edits on the same file,
        the prompt prefix (system + template + original_code) is loaded
        from cache instead of re-prefilled, saving ~250-350ms per call.

        Args:
            original_code: The original source file content.
            update_snippet: The edit snippet with ellipsis markers.
            language: Optional language for post-merge AST validation.

        Returns:
            MergeResult with the merged code and performance metrics.
        """
        # Build prompt from templates (reuses merge.py's build_prompt)
        messages = build_prompt(original_code, update_snippet)
        # Don't pass enable_thinking=False — on Qwen3 it injects <think></think>
        # in the prompt which causes fine-tuned models to loop on think tokens.
        # The model emits an empty think block on its own; _extract_output handles it.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_tokens = self.tokenizer.encode(prompt_text)

        # Collect EOS token IDs for stopping
        eos_token_ids: set[int] = set()
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            for tid in self.tokenizer.additional_special_tokens_ids:
                eos_token_ids.add(tid)

        cache_key = _compute_cache_key(original_code)

        start = time.perf_counter()

        if cache_key in self._prefix_cache:
            # Level 1 HIT: exact hash match -- clone prefix, process suffix only
            cached, n_prefix = self._prefix_cache[cache_key]
            cache = _clone_cache(cached)
            suffix_tokens = mx.array(full_tokens[n_prefix:])
            prefill_logits = _prefill_prompt(
                self.model, suffix_tokens, cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )
            # Save to token cache so future chained edits can partial-match
            self._token_cache.add(
                full_tokens,
                {len(full_tokens) - 1: _clone_cache(cache)},
            )

        elif (match := self._token_cache.find_match(full_tokens)) is not None:
            # Level 2 HIT: token prefix match -- restore snapshot, process remainder
            cache, n_reusable = match
            remainder_tokens = mx.array(full_tokens[n_reusable:])
            prefill_logits = _prefill_prompt(
                self.model, remainder_tokens, cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )
            # Save final state for future chained edits.
            # (We do NOT save to Level 1 here because the cache now contains
            # the full prompt state; a Level 1 hit would re-process suffix
            # tokens on top of it, corrupting the state.)
            self._token_cache.add(
                full_tokens,
                {len(full_tokens) - 1: _clone_cache(cache)},
            )

        else:
            # Level 3: cold path -- full prefill with block snapshots
            n_prefix = _get_prefix_boundary(self.tokenizer, original_code)
            cache = make_prompt_cache(self.model)

            # Process full prompt with periodic snapshots for Level 2
            prefill_logits, snapshots = _prefill_with_snapshots(
                self.model, mx.array(full_tokens), cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )

            # Save to Level 1 (exact hash): use the snapshot at the largest
            # boundary <= n_prefix. On Level 1 hit, tokens from that boundary
            # onward are re-processed (at most snapshot_interval-1 extra tokens,
            # negligible vs full prefill savings).
            prefix_boundaries = sorted(
                [k for k in snapshots if k <= n_prefix], reverse=True,
            )
            if prefix_boundaries:
                snap_boundary = prefix_boundaries[0]
                self._prefix_cache[cache_key] = (
                    _clone_cache(snapshots[snap_boundary]), snap_boundary,
                )
            else:
                # n_prefix is before the first snapshot (very short prefix).
                # Fall back: cache nothing for Level 1, next call re-prefills.
                pass

            # Save to Level 2 (token prefix) with block snapshots for
            # cross-version partial prefix reuse on chained edits.
            self._token_cache.add(full_tokens, snapshots)

        # Generate tokens autoregressively from prefill logits
        quantize_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=0,
            kv_group_size=self.kv_group_size,
            kv_bits=self.kv_bits,
        )

        # Cap output tokens: model reproduces the original code with edits,
        # so output should never exceed 2x the input code tokens.
        # This prevents OOM from runaway generation on large inputs.
        input_code_tokens = len(self.tokenizer.encode(original_code))
        output_cap = min(self.max_tokens, max(2048, input_code_tokens * 2))

        token_ids: list[int] = []
        first_token_arr = mx.argmax(prefill_logits[:, -1, :], axis=-1)
        mx.eval(first_token_arr)
        first_token = first_token_arr.item()
        token_ids.append(first_token)

        ttft_ms = (time.perf_counter() - start) * 1000

        if first_token not in eos_token_ids:
            y = mx.array([first_token], dtype=mx.uint32)
            while len(token_ids) < output_cap:
                logits = self.model(y[None], cache=cache)
                quantize_fn(cache)
                next_arr = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(next_arr)
                nt = next_arr.item()
                token_ids.append(nt)
                if nt in eos_token_ids:
                    break
                y = mx.array([nt], dtype=mx.uint32)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Decode and extract merged code
        raw_text = self.tokenizer.decode(token_ids)
        merged_code = _extract_output(raw_text)

        tokens_generated = len(token_ids)
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        # Optional AST validation
        parse_valid = True
        if language:
            parse_valid = validate_parse(merged_code, language)

        return MergeResult(
            merged_code=merged_code,
            parse_valid=parse_valid,
            tokens_generated=tokens_generated,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            ttft_ms=ttft_ms,
        )

    # Threshold: files with fewer draft tokens than this use plain AR
    # (speculative overhead exceeds batching gains on small files).
    SPECULATIVE_MIN_DRAFT_TOKENS = 50
    SPECULATIVE_DEFAULT_BATCH_SIZE = 5

    def merge_auto(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
    ) -> MergeResult:
        """Smart merge that picks AR or speculative based on file size.

        For small files (< 50 draft tokens), autoregressive is faster.
        For larger files, speculative decoding with adaptive batch sizing
        wins because most output tokens match the original.

        Args:
            original_code: The original source file content.
            update_snippet: The edit snippet with ellipsis markers.
            language: Optional language for post-merge AST validation.

        Returns:
            MergeResult with the merged code and performance metrics.
        """
        n_draft = len(self.tokenizer.encode(original_code, add_special_tokens=False))
        if n_draft < self.SPECULATIVE_MIN_DRAFT_TOKENS:
            return self.merge(original_code, update_snippet, language)
        batch_size = min(self.SPECULATIVE_DEFAULT_BATCH_SIZE, n_draft)
        return self.merge_speculative(
            original_code, update_snippet, language,
            draft_batch_size=batch_size,
        )

    def merge_speculative(
        self,
        original_code: str,
        update_snippet: str,
        language: str | None = None,
        draft_batch_size: int = SPECULATIVE_DEFAULT_BATCH_SIZE,
    ) -> MergeResult:
        """Merge with original file tokens as speculative draft.

        The key insight: since ~95% of merged output tokens are identical
        to the original file, we use original file tokens as draft
        candidates that the model verifies in parallel batches, giving
        a speedup over standard autoregressive decoding.

        Uses SSM checkpoint-and-restore for Qwen3.5's hybrid architecture:
        KVCache layers are trimmed on rejection, ArraysCache (GatedDeltaNet)
        layers are restored from snapshot. On rejection, searches forward
        in the draft for a re-alignment point and generates autoregressively
        through divergent sections (edit boundaries).

        Args:
            original_code: The original source file content.
            update_snippet: The edit snippet with ellipsis markers.
            language: Optional language for post-merge AST validation.
            draft_batch_size: Number of draft tokens to verify per batch.
                Higher values amortize forward-pass overhead but waste more
                compute on rejection. Default 5 (best measured starting point
                on the current 4-bit MLX benchmark; the loop adapts from there).

        Returns:
            MergeResult with the merged code and performance metrics.
        """
        # Build prompt
        messages = build_prompt(original_code, update_snippet)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_tokens = self.tokenizer.encode(prompt_text)

        # Tokenize original code as draft source
        original_draft_tokens = self.tokenizer.encode(
            original_code, add_special_tokens=False,
        )

        # Collect EOS token IDs for stopping
        eos_token_ids: set[int] = set()
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            for tid in self.tokenizer.additional_special_tokens_ids:
                eos_token_ids.add(tid)

        cache_key = _compute_cache_key(original_code)

        start = time.perf_counter()

        if cache_key in self._prefix_cache:
            # Level 1 HIT: exact hash match -- clone prefix, process suffix only
            cached, n_prefix = self._prefix_cache[cache_key]
            cache = _clone_cache(cached)
            suffix_tokens = mx.array(full_tokens[n_prefix:])
            prefill_logits = _prefill_prompt(
                self.model, suffix_tokens, cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )
            # Save to token cache so future chained edits can partial-match
            self._token_cache.add(
                full_tokens,
                {len(full_tokens) - 1: _clone_cache(cache)},
            )

        elif (match := self._token_cache.find_match(full_tokens)) is not None:
            # Level 2 HIT: token prefix match -- restore snapshot, process remainder
            cache, n_reusable = match
            remainder_tokens = mx.array(full_tokens[n_reusable:])
            prefill_logits = _prefill_prompt(
                self.model, remainder_tokens, cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )
            # Save final state for future chained edits.
            # (We do NOT save to Level 1 here because the cache now contains
            # the full prompt state; a Level 1 hit would re-process suffix
            # tokens on top of it, corrupting the state.)
            self._token_cache.add(
                full_tokens,
                {len(full_tokens) - 1: _clone_cache(cache)},
            )

        else:
            # Level 3: cold path -- full prefill with block snapshots
            n_prefix = _get_prefix_boundary(self.tokenizer, original_code)
            cache = make_prompt_cache(self.model)

            # Process full prompt with periodic snapshots for Level 2
            prefill_logits, snapshots = _prefill_with_snapshots(
                self.model, mx.array(full_tokens), cache,
                kv_bits=self.kv_bits, kv_group_size=self.kv_group_size,
            )

            # Save to Level 1 (exact hash): use nearest snapshot <= n_prefix
            prefix_boundaries = sorted(
                [k for k in snapshots if k <= n_prefix], reverse=True,
            )
            if prefix_boundaries:
                snap_boundary = prefix_boundaries[0]
                self._prefix_cache[cache_key] = (
                    _clone_cache(snapshots[snap_boundary]), snap_boundary,
                )
            else:
                pass

            # Save to Level 2 (token prefix) with block snapshots
            self._token_cache.add(full_tokens, snapshots)

        ttft_ms = (time.perf_counter() - start) * 1000

        # Cap output tokens: model reproduces the original code with edits,
        # so output should never exceed 2x the input code tokens.
        input_code_tokens = len(original_draft_tokens)
        output_cap = min(self.max_tokens, max(2048, input_code_tokens * 2))

        # Run speculative generation
        token_ids = _speculative_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=original_draft_tokens,
            draft_batch_size=draft_batch_size,
            max_tokens=output_cap,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            eos_token_ids=eos_token_ids,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Decode and extract merged code
        raw_text = self.tokenizer.decode(token_ids)
        merged_code = _extract_output(raw_text)

        tokens_generated = len(token_ids)
        tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        # Optional AST validation
        parse_valid = True
        if language:
            parse_valid = validate_parse(merged_code, language)

        return MergeResult(
            merged_code=merged_code,
            parse_valid=parse_valid,
            tokens_generated=tokens_generated,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            ttft_ms=ttft_ms,
        )
