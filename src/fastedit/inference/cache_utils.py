"""Cache utility functions for the MLX inference engine.

Low-level helpers for managing Qwen3.5's hybrid cache architecture:
KVCache (attention layers) and ArraysCache (GatedDeltaNet SSM layers).

Includes snapshot/restore for speculative decoding, deep cloning for
prefix caching, and cache key computation.
"""

from __future__ import annotations

import functools
import hashlib

import mlx.core as mx
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache

from .merge import build_prompt


def _snapshot_ssm_caches(cache: list) -> dict:
    """Save a copy of all ArraysCache (SSM) state for later restore.

    Qwen3.5 uses a hybrid architecture: 8 attention layers with KVCache and
    24 GatedDeltaNet layers with ArraysCache. Unlike KVCache, ArraysCache is
    NOT trimmable -- its recurrent state cannot be rewound by adjusting an
    offset. On speculative rejection, we must restore the SSM state to the
    checkpoint taken before the speculative forward pass.

    Args:
        cache: The model's cache list (mix of KVCache and ArraysCache).

    Returns:
        A dict mapping cache index to a list of copied arrays.
    """
    snapshots = {}
    all_arrays = []
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            snap = [
                mx.array(x) if x is not None else None
                for x in c.cache
            ]
            snapshots[i] = snap
            all_arrays.extend(x for x in snap if x is not None)
    # Batch-evaluate all SSM arrays in a single mx.eval call
    # instead of 24 individual evals (one per layer).
    if all_arrays:
        mx.eval(*all_arrays)
    return snapshots


def _restore_ssm_caches(cache: list, snapshots: dict) -> None:
    """Restore ArraysCache (SSM) state from a previous snapshot.

    Called on speculative rejection to rewind the SSM state to the
    pre-verification checkpoint.

    Args:
        cache: The model's cache list.
        snapshots: Dict from _snapshot_ssm_caches mapping index to state.
    """
    for i, snap in snapshots.items():
        cache[i].cache = snap


def _trim_kv_caches(cache: list, n: int) -> None:
    """Trim only trimmable cache entries (attention layers), skip ArraysCache.

    On speculative rejection, we need to rewind the KV cache by the number
    of unaccepted draft tokens. KVCache and QuantizedKVCache support trim()
    (adjusts offset), but ArraysCache does not -- SSM state is restored via
    snapshot instead.

    We use ``is_trimmable()`` instead of ``isinstance(c, KVCache)`` because
    after ``maybe_quantize_kv_cache`` runs, attention layers become
    ``QuantizedKVCache`` (a sibling class, not a KVCache subclass).

    Args:
        cache: The model's cache list.
        n: Number of tokens to trim from trimmable (attention) caches.
    """
    for c in cache:
        if not isinstance(c, ArraysCache) and c.is_trimmable():
            c.trim(n)


def _prefill_prompt(model, prompt_tokens, cache, kv_bits, kv_group_size,
                    prefill_step_size=2048):
    """Process the prompt through the model to populate cache.

    Processes in chunks of prefill_step_size to avoid memory issues on
    long prompts. Returns logits for the last prompt token.

    Args:
        model: The MLX model.
        prompt_tokens: Tokenized prompt as an mx.array.
        cache: The model's cache list (modified in place).
        kv_bits: Number of bits for KV cache quantization (None to skip).
        kv_group_size: Group size for KV cache quantization.
        prefill_step_size: Maximum tokens to process per chunk.

    Returns:
        Logits from the model's response to the final prompt token.
    """
    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    remaining = prompt_tokens
    while len(remaining) > 1:
        chunk_size = min(prefill_step_size, len(remaining) - 1)
        chunk = remaining[:chunk_size]
        model(chunk[None], cache=cache)
        quantize_fn(cache)
        mx.eval([c.state for c in cache])
        remaining = remaining[chunk_size:]
        mx.clear_cache()

    # Process last token and return logits
    logits = model(remaining[None], cache=cache)
    quantize_fn(cache)
    return logits


def _prefill_with_snapshots(
    model,
    prompt_tokens,
    cache,
    kv_bits,
    kv_group_size,
    snapshot_interval: int = 128,
) -> tuple:
    """Process prompt tokens and save cache snapshots at regular intervals.

    Like ``_prefill_prompt`` but additionally clones the cache state every
    ``snapshot_interval`` tokens so that callers can later restore from an
    intermediate checkpoint. This powers the token-level prefix cache:
    when a new prompt shares a long token prefix with a previously-seen
    prompt, we restore the nearest snapshot and only re-process the
    remaining tokens.

    The last token is processed separately (as in ``_prefill_prompt``)
    to capture the logits needed for generation.

    Args:
        model: The MLX model.
        prompt_tokens: Tokenized prompt as an mx.array.
        cache: The model's cache list (modified in place).
        kv_bits: Number of bits for KV cache quantization (None to skip).
        kv_group_size: Group size for KV cache quantization.
        snapshot_interval: Save a cache clone every N tokens.

    Returns:
        (logits, snapshots) where logits are from the final token and
        snapshots is a dict mapping {cumulative_token_count: cloned_cache}.
    """
    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    snapshots: dict[int, list] = {}
    total_len = len(prompt_tokens)

    # We need to process all-but-last through chunked prefill,
    # then the last token separately for logits.
    # Break into snapshot_interval-sized chunks, but always reserve
    # the final token for the logits call.
    processed = 0
    remaining_count = total_len

    while remaining_count > 1:
        # Chunk size: up to snapshot_interval, but leave at least 1 token
        chunk_size = min(snapshot_interval, remaining_count - 1)
        chunk = prompt_tokens[processed : processed + chunk_size]
        model(chunk[None], cache=cache)
        quantize_fn(cache)
        mx.eval([c.state for c in cache])
        processed += chunk_size
        remaining_count -= chunk_size

        # Save snapshot at this boundary
        snapshots[processed] = _clone_cache(cache)
        mx.clear_cache()

    # Process last token and return logits
    last_token = prompt_tokens[processed:]
    logits = model(last_token[None], cache=cache)
    quantize_fn(cache)

    return logits, snapshots


def _find_realign_point(
    draft_tokens: list[int],
    start_idx: int,
    target_token: int,
    ngram_window: int = 3,
    search_limit: int = 100,
) -> int:
    """Search forward in draft tokens for a re-alignment point.

    After a speculative rejection (e.g. at an edit boundary where new code
    is inserted), the draft and model output diverge. This function searches
    forward in the draft for where the original file's tokens resume,
    allowing the speculative loop to skip the divergent section.

    Uses N-gram matching: finds the first position where ``target_token``
    appears and the following (ngram_window - 1) tokens also match, to
    avoid false positives on common tokens like ``:`` or ``\\n``.

    Args:
        draft_tokens: The full list of original file tokens.
        start_idx: Position to start searching from.
        target_token: The model's correction token to search for.
        ngram_window: Minimum consecutive matching tokens required.
        search_limit: Maximum positions to search forward.

    Returns:
        The draft index where alignment was found, or -1 if not found.
    """
    end = min(start_idx + search_limit, len(draft_tokens))
    for k in range(start_idx, end):
        if draft_tokens[k] == target_token:
            if ngram_window <= 1:
                return k
            # For longer windows, we can't verify future model tokens yet.
            # Just check that the candidate isn't at the very end of draft.
            if k + ngram_window <= len(draft_tokens):
                return k
    return -1


def _clone_cache(cache: list) -> list:
    """Deep clone a model cache list for prefix caching.

    Creates independent copies of all cache entries so that generation
    on the clone doesn't mutate the saved prefix state.

    Args:
        cache: The model's cache list (mix of KVCache and ArraysCache).

    Returns:
        A new cache list with copied arrays.
    """
    cloned = []
    for c in cache:
        if isinstance(c, ArraysCache):
            new_c = ArraysCache(len(c.cache))
            new_c.cache = [mx.array(x) if x is not None else None for x in c.cache]
            cloned.append(new_c)
        elif isinstance(c, QuantizedKVCache):
            new_c = QuantizedKVCache(group_size=c.group_size, bits=c.bits)
            if c.keys is not None:
                new_c.keys = tuple(mx.array(x) for x in c.keys)
                new_c.values = tuple(mx.array(x) for x in c.values)
                new_c.offset = c.offset
            cloned.append(new_c)
        elif isinstance(c, KVCache):
            new_c = KVCache()
            if c.keys is not None:
                new_c.keys = mx.array(c.keys)
                new_c.values = mx.array(c.values)
                new_c.offset = c.offset
            cloned.append(new_c)
        else:
            cloned.append(c)
    return cloned


def _get_prefix_boundary(tokenizer, original_code: str) -> int:
    """Find the token index where the cacheable prompt prefix ends.

    The prompt template puts original_code before the snippet. Everything
    up to the snippet is constant for the same file and can be cached.
    This finds how many tokens that prefix occupies.

    Args:
        tokenizer: The model tokenizer.
        original_code: The original source code.

    Returns:
        Number of tokens in the cacheable prefix.
    """
    marker = "\x00SPLIT_HERE\x00"
    messages = build_prompt(original_code, marker)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    marker_pos = text.index(marker)
    prefix_text = text[:marker_pos]
    return len(tokenizer.encode(prefix_text))


def _compute_cache_key(original_code: str) -> str:
    """Compute a cache key from the original code content.

    The prompt for FastEdit is: system_prompt + user_template + original_code + update_snippet.
    The system_prompt and user_template prefix are constant. The original_code is the
    file being edited (same across repeat edits). Only update_snippet varies.

    We cache the prefix up to and including the original_code, keyed on its hash.

    Args:
        original_code: The source code content to hash.

    Returns:
        A 16-character hex string derived from the SHA-256 of the content.
    """
    return hashlib.sha256(original_code.encode()).hexdigest()[:16]
