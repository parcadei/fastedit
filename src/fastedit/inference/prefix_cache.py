"""Prefix cache classes for the MLX inference engine.

TokenPrefixCache: In-memory token-level prefix matching with block snapshots
for cross-version reuse (chained edits where output of edit N becomes input
of edit N+1).

PromptCacheManager: Persistent on-disk prompt cache using safetensors files
with LRU eviction for repeat-file editing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .cache_utils import _clone_cache


class TokenPrefixCache:
    """Token-level prefix cache with block snapshots for cross-version reuse.

    Stores (token_sequence, snapshots) entries. On lookup, finds the entry
    with the longest common token prefix, returns a cloned cache from the
    nearest snapshot boundary <= the common prefix length.

    This enables Scenario B (chained edits): when the output of edit 1
    becomes the input of edit 2, the two file versions share a large
    common token prefix. The hash-based Level 1 cache misses because
    the content changed, but this cache finds the shared prefix and
    restores the nearest snapshot, avoiding redundant prefill of the
    overlapping region.

    Args:
        max_entries: Maximum number of cached entries (LRU eviction).
        min_prefix_ratio: Minimum ratio of common prefix to query length
            required to use a match. Below this threshold, the savings
            are too small to justify the clone overhead.
    """

    def __init__(self, max_entries: int = 8, min_prefix_ratio: float = 0.3):
        self._entries: list[tuple[list[int], dict[int, list]]] = []
        self.max_entries = max_entries
        self.min_prefix_ratio = min_prefix_ratio

    def find_match(self, tokens: list[int]) -> tuple | None:
        """Find the best matching entry by longest common prefix.

        Iterates all entries, computes the longest common token prefix
        with each, and returns the one with the most reusable tokens
        (aligned to a snapshot boundary).

        Args:
            tokens: The full prompt token list to match against.

        Returns:
            (cloned_cache, n_reusable_tokens) where n_reusable_tokens is
            the snapshot boundary (not the raw prefix length), or None if
            no entry has sufficient overlap.
        """
        if not self._entries:
            return None

        best_idx = -1
        best_boundary = 0
        best_snapshots = None

        for entry_idx, (stored_tokens, snapshots) in enumerate(self._entries):
            # Compute longest common prefix length
            common = 0
            for a, b in zip(tokens, stored_tokens, strict=False):
                if a != b:
                    break
                common += 1

            # Check minimum overlap threshold
            if common < self.min_prefix_ratio * len(tokens):
                continue

            # Find the largest snapshot boundary <= common prefix length
            valid_boundaries = [k for k in snapshots if k <= common]
            if not valid_boundaries:
                continue

            boundary = max(valid_boundaries)
            if boundary > best_boundary:
                best_idx = entry_idx
                best_boundary = boundary
                best_snapshots = snapshots

        if best_idx < 0 or best_snapshots is None:
            return None

        # Move matched entry to end (LRU: most-recently-used at end)
        entry = self._entries.pop(best_idx)
        self._entries.append(entry)

        # Clone the cache at the best snapshot boundary
        return _clone_cache(best_snapshots[best_boundary]), best_boundary

    def add(self, tokens: list[int], snapshots: dict[int, list]):
        """Add an entry with LRU eviction.

        If the cache is at capacity, the oldest entry (index 0) is evicted.

        Args:
            tokens: The full prompt token list.
            snapshots: Dict mapping {cumulative_token_count: cloned_cache}.
        """
        self._entries.append((list(tokens), snapshots))
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def clear(self):
        """Clear all entries."""
        self._entries.clear()


class PromptCacheManager:
    """Manages persistent prompt caches for repeat-file editing.

    Caches are stored as safetensors files keyed on content hash.
    Uses LRU eviction when total size exceeds budget.

    Args:
        cache_dir: Directory for cache files. Defaults to ~/.fastedit/cache/.
        max_cache_bytes: Maximum total cache size in bytes. Default 2GB.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_cache_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB
    ):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.fastedit/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_bytes

    def get(self, cache_key: str):
        """Load a cached prompt state if it exists.

        Args:
            cache_key: The content hash key to look up.

        Returns:
            The prompt cache list, or None if not cached.
        """
        cache_path = self.cache_dir / f"{cache_key}.safetensors"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        if not cache_path.exists():
            return None

        from mlx_lm.models.cache import load_prompt_cache
        cache = load_prompt_cache(str(cache_path))

        # Touch metadata to update LRU timestamp
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["last_used"] = time.time()
                meta_path.write_text(json.dumps(meta))
            except (json.JSONDecodeError, KeyError):
                pass

        return cache

    def put(self, cache_key: str, cache, token_count: int):
        """Save a prompt cache to disk.

        Args:
            cache_key: The content hash key.
            cache: The prompt cache list from mlx-lm.
            token_count: Number of prompt tokens cached.
        """
        from mlx_lm.models.cache import save_prompt_cache

        cache_path = self.cache_dir / f"{cache_key}.safetensors"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        save_prompt_cache(str(cache_path), cache)

        meta = {
            "cache_key": cache_key,
            "token_count": token_count,
            "created": time.time(),
            "last_used": time.time(),
            "size_bytes": cache_path.stat().st_size,
        }
        meta_path.write_text(json.dumps(meta))

        # Evict old caches if over budget
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Remove least-recently-used caches until total size is under budget."""
        entries = []
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                cache_path = self.cache_dir / f"{meta['cache_key']}.safetensors"
                if cache_path.exists():
                    entries.append({
                        "key": meta["cache_key"],
                        "last_used": meta.get("last_used", 0),
                        "size": meta.get("size_bytes", cache_path.stat().st_size),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        total_size = sum(e["size"] for e in entries)
        if total_size <= self.max_cache_bytes:
            return

        # Sort by last_used ascending (oldest first)
        entries.sort(key=lambda e: e["last_used"])

        for entry in entries:
            if total_size <= self.max_cache_bytes:
                break
            cache_path = self.cache_dir / f"{entry['key']}.safetensors"
            meta_path = self.cache_dir / f"{entry['key']}.meta.json"
            if cache_path.exists():
                total_size -= entry["size"]
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

    def total_size(self) -> int:
        """Return total cache size in bytes across all safetensors files."""
        return sum(
            f.stat().st_size for f in self.cache_dir.glob("*.safetensors")
        )
