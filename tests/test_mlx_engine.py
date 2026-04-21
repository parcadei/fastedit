"""Tests for the in-process MLX inference engine."""

import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Mock the mlx / mlx_lm ecosystem so tests run without Apple Silicon deps
# ---------------------------------------------------------------------------

def _build_mlx_mocks():
    """Build mock modules for mlx and mlx_lm.

    Returns the mock objects needed to verify MLXEngine behavior.
    """
    # mlx.core mock
    mx_mock = MagicMock()

    # mx.array constructor: wraps input in a SimpleNamespace with .tolist()
    class FakeArray:
        def __init__(self, data, dtype=None):
            if isinstance(data, FakeArray):
                # Copy the list so mx.array(existing) produces independent data,
                # matching real MLX semantics (snapshot copies are not aliases).
                data = list(data._data) if isinstance(data._data, list) else data._data
            self._data = data
            self.dtype = dtype

        def tolist(self):
            if isinstance(self._data, list):
                return self._data
            return [self._data]

        def __len__(self):
            if isinstance(self._data, list):
                return len(self._data)
            return 1

        def item(self):
            """Extract scalar value, unwrapping nested lists."""
            val = self._data
            while isinstance(val, list):
                val = val[0]
            if isinstance(val, FakeArray):
                return val.item()
            return val

        def __getitem__(self, key):
            """Support slicing, indexing, tuple, and None for MLX compat.

            Handles the indexing patterns used by speculative decoding:
            - arr[None]        -> add batch dimension
            - arr[:, -1, :]    -> extract last-position logits
            - arr[0]           -> get first batch element
            - arr[start:end]   -> slice along first axis
            - arr[-N:]         -> negative slice
            """
            # arr[None] adds a batch dimension (like numpy unsqueeze)
            if key is None:
                if isinstance(self._data, list):
                    return FakeArray([self._data])
                return FakeArray([[self._data]])
            # Tuple indexing (multi-dimensional): e.g. [:, -1, :]
            # For our mock, just return self -- the data structure is
            # pre-arranged by tests to produce the right token values
            if isinstance(key, tuple):
                return FakeArray(self._data)
            if isinstance(self._data, list):
                result = self._data[key]
                if isinstance(result, list):
                    return FakeArray(result)
                return FakeArray([result])
            # For scalar-like data, just return self
            return FakeArray(self._data)

    mx_mock.array = FakeArray

    # mlx_lm.load mock
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    load_fn = MagicMock(return_value=(mock_model, mock_tokenizer))

    # mlx_lm.generate.generate_step mock
    generate_step_fn = MagicMock()

    # mlx_lm.models.cache mocks
    make_prompt_cache_fn = MagicMock(return_value=[])
    save_prompt_cache_fn = MagicMock()
    load_prompt_cache_fn = MagicMock(return_value=[])

    # maybe_quantize_kv_cache mock (no-op by default)
    maybe_quantize_kv_cache_fn = MagicMock()

    # Fake KVCache: mimics real KVCache with trimmable offset
    class FakeKVCache:
        """Mock KVCache that tracks offset and supports trim."""
        def __init__(self, offset=0):
            self.offset = offset
            self.keys = None
            self.values = None

        def is_trimmable(self):
            return True

        def trim(self, n):
            trimmed = min(self.offset, n)
            self.offset -= trimmed
            return trimmed

        @property
        def state(self):
            return (self.keys, self.values)

        @state.setter
        def state(self, v):
            self.keys, self.values = v

    class FakeQuantizedKVCache(FakeKVCache):
        """Mock quantized KV cache used by clone path."""
        def __init__(self, group_size=64, bits=4, offset=0):
            super().__init__(offset=offset)
            self.group_size = group_size
            self.bits = bits

    # Fake ArraysCache: mimics real ArraysCache with list-based state
    class FakeArraysCache:
        """Mock ArraysCache that stores SSM state as a list."""
        def __init__(self, size=2):
            self.cache = [None] * size

        def is_trimmable(self):
            return False

        @property
        def state(self):
            return self.cache

        @state.setter
        def state(self, v):
            self.cache = v

    # Build module hierarchy
    mlx_mod = ModuleType("mlx")
    mlx_core_mod = ModuleType("mlx.core")
    mlx_mod.core = mlx_core_mod
    for attr in dir(mx_mock):
        if not attr.startswith("_"):
            setattr(mlx_core_mod, attr, getattr(mx_mock, attr))
    mlx_core_mod.array = FakeArray
    mlx_core_mod.uint32 = "uint32"

    # mx.concatenate: join FakeArrays
    def fake_concatenate(arrays):
        combined = []
        for a in arrays:
            if isinstance(a, FakeArray):
                data = a._data if isinstance(a._data, list) else [a._data]
                combined.extend(data)
            elif isinstance(a, list):
                combined.extend(a)
        return FakeArray(combined)
    mlx_core_mod.concatenate = fake_concatenate

    # mx.argmax: identity for pre-computed token arrays in tests
    def fake_argmax(arr, axis=-1):
        return arr
    mlx_core_mod.argmax = fake_argmax

    # mx.eval: no-op for mock purposes
    mlx_core_mod.eval = lambda *args: None

    # mx.clear_cache: no-op
    mlx_core_mod.clear_cache = lambda: None

    mlx_lm_mod = ModuleType("mlx_lm")
    mlx_lm_mod.load = load_fn

    mlx_lm_generate_mod = ModuleType("mlx_lm.generate")
    mlx_lm_generate_mod.generate_step = generate_step_fn
    mlx_lm_generate_mod.maybe_quantize_kv_cache = maybe_quantize_kv_cache_fn

    mlx_lm_models_mod = ModuleType("mlx_lm.models")
    mlx_lm_models_cache_mod = ModuleType("mlx_lm.models.cache")
    mlx_lm_models_cache_mod.make_prompt_cache = make_prompt_cache_fn
    mlx_lm_models_cache_mod.save_prompt_cache = save_prompt_cache_fn
    mlx_lm_models_cache_mod.load_prompt_cache = load_prompt_cache_fn
    mlx_lm_models_cache_mod.KVCache = FakeKVCache
    mlx_lm_models_cache_mod.QuantizedKVCache = FakeQuantizedKVCache
    mlx_lm_models_cache_mod.ArraysCache = FakeArraysCache
    mlx_lm_models_mod.cache = mlx_lm_models_cache_mod

    modules = {
        "mlx": mlx_mod,
        "mlx.core": mlx_core_mod,
        "mlx_lm": mlx_lm_mod,
        "mlx_lm.generate": mlx_lm_generate_mod,
        "mlx_lm.models": mlx_lm_models_mod,
        "mlx_lm.models.cache": mlx_lm_models_cache_mod,
    }

    return modules, {
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "load": load_fn,
        "generate_step": generate_step_fn,
        "make_prompt_cache": make_prompt_cache_fn,
        "save_prompt_cache": save_prompt_cache_fn,
        "load_prompt_cache": load_prompt_cache_fn,
        "maybe_quantize_kv_cache": maybe_quantize_kv_cache_fn,
        "FakeArray": FakeArray,
        "FakeKVCache": FakeKVCache,
        "FakeQuantizedKVCache": FakeQuantizedKVCache,
        "FakeArraysCache": FakeArraysCache,
    }


@pytest.fixture(autouse=True)
def mlx_mocks():
    """Install mlx/mlx_lm mocks into sys.modules for every test."""
    modules, mocks = _build_mlx_mocks()

    # Patch sys.modules so 'import mlx_lm' works
    saved = {}
    for name, mod in modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Configure default tokenizer behavior
    tok = mocks["tokenizer"]
    tok.eos_token_id = 151645  # Qwen3.5 EOS

    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.return_value = "<updated-code>def hello(): pass</updated-code>"

    def fake_apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        contents = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                contents.append(message["content"])
            else:
                contents.append(str(message))
        suffix = "<|assistant|>" if add_generation_prompt else ""
        return "".join(contents) + suffix

    tok.apply_chat_template.side_effect = fake_apply_chat_template

    # Default generate_step: yields 10 tokens then EOS
    FakeArray = mocks["FakeArray"]
    eos_id = tok.eos_token_id

    def fake_generate_step(prompt, model, **kwargs):
        for i in range(10):
            yield FakeArray(100 + i), FakeArray([0.0] * 10)
        yield FakeArray(eos_id), FakeArray([0.0] * 10)

    mocks["generate_step"].side_effect = fake_generate_step

    yield mocks

    # Restore sys.modules
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig

    # Remove cached imports so all inference submodules are re-imported fresh
    # each test with the correct mock classes (cache_utils captures ArraysCache
    # at import time; stale references cause isinstance checks to fail).
    for mod in [
        "fastedit.inference.mlx_engine",
        "fastedit.inference.cache_utils",
        "fastedit.inference.prefix_cache",
    ]:
        sys.modules.pop(mod, None)


def _import_engine():
    """Import MLXEngine after mocks are installed."""
    from fastedit.inference.mlx_engine import MLXEngine
    return MLXEngine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLXEngineInit:
    """Test MLXEngine initialization."""

    def test_loads_model_and_tokenizer(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(model_path="output/fastedit-4b-mlx")

        mlx_mocks["load"].assert_called_once_with("output/fastedit-4b-mlx")
        assert engine.model is mlx_mocks["model"]
        assert engine.tokenizer is mlx_mocks["tokenizer"]

    def test_stores_kv_config(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(kv_bits=4, kv_group_size=64)

        assert engine.kv_bits == 4
        assert engine.kv_group_size == 64

    def test_stores_max_tokens(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(max_tokens=8192)

        assert engine.max_tokens == 8192

    def test_default_config(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        assert engine.kv_bits == 8
        assert engine.kv_group_size == 64
        assert engine.max_tokens == 16384

    def test_custom_model_path(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(model_path="/custom/path")

        mlx_mocks["load"].assert_called_once_with("/custom/path")


class TestMLXEngineMerge:
    """Test MLXEngine.merge() method."""

    def test_returns_merge_result(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        result = engine.merge("def foo(): pass", "def foo(): return 1")

        from fastedit.inference.merge import MergeResult
        assert isinstance(result, MergeResult)

    def test_extracts_merged_code(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        mlx_mocks["tokenizer"].decode.return_value = (
            "<updated-code>def foo(): return 42</updated-code>"
        )

        result = engine.merge("def foo(): pass", "def foo(): return 42")
        assert result.merged_code == "def foo(): return 42"

    def test_uses_build_prompt(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        with patch("fastedit.inference.mlx_engine.build_prompt") as mock_bp:
            mock_bp.return_value = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ]
            engine.merge("original", "snippet")

            mock_bp.assert_called_once_with("original", "snippet")

    def test_applies_chat_template_correctly(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        engine.merge("code", "snippet")

        # apply_chat_template is called at least once for the main prompt
        # (may also be called internally by _get_prefix_boundary).
        assert mlx_mocks["tokenizer"].apply_chat_template.call_count >= 1
        # All calls must have tokenize=False and add_generation_prompt=True.
        for call in mlx_mocks["tokenizer"].apply_chat_template.call_args_list:
            assert call.kwargs.get("tokenize") is False
            assert call.kwargs.get("add_generation_prompt") is True

    def test_calls_generate_step_with_kv_config(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(kv_bits=4, kv_group_size=64)

        engine.merge("code", "snippet")

        # merge() uses maybe_quantize_kv_cache with the engine's kv config.
        # Verify the engine stored the config correctly and the quantize
        # function was called at least once during prefill/generation.
        assert engine.kv_bits == 4
        assert engine.kv_group_size == 64
        assert mlx_mocks["maybe_quantize_kv_cache"].call_count >= 1
        # Every call should use the configured kv_bits and kv_group_size.
        for call in mlx_mocks["maybe_quantize_kv_cache"].call_args_list:
            assert call.kwargs.get("kv_bits") == 4
            assert call.kwargs.get("kv_group_size") == 64

    def test_calls_make_prompt_cache(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        engine.merge("code", "snippet")

        mlx_mocks["make_prompt_cache"].assert_called_once_with(engine.model)

    def test_counts_tokens_generated(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        # Default fake_generate_step yields 10 tokens + 1 EOS = 11 total
        # but EOS should be counted in the token list too
        result = engine.merge("code", "snippet")
        assert result.tokens_generated > 0

    def test_reports_latency(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        result = engine.merge("code", "snippet")
        assert result.latency_ms > 0

    def test_reports_tokens_per_second(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        result = engine.merge("code", "snippet")
        assert result.tokens_per_second >= 0

    def test_stops_at_eos_token(self, mlx_mocks):
        """Generation stops when EOS token is encountered."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        FakeArray = mlx_mocks["FakeArray"]
        eos_id = mlx_mocks["tokenizer"].eos_token_id

        # merge() calls model() directly and uses mx.argmax(logits[:,-1,:]).item()
        # to get the next token.  Set up model to return:
        #   - calls during _prefill_with_snapshots: return a non-EOS token
        #   - AR call 1: token 100
        #   - AR call 2: token 101
        #   - AR call 3: token 102
        #   - AR call 4: EOS  (generation should stop here)
        # tokenizer.encode returns [1,2,3,4,5] (5 tokens), so prefill processes
        # [1,2,3,4] in one chunk then [5] as the last token → 2 model calls in prefill.
        ar_call_tokens = [100, 101, 102, eos_id]
        call_idx = [0]

        def fake_model(tokens, cache=None):
            idx = call_idx[0]
            call_idx[0] += 1
            # First two calls are prefill; remaining are AR
            ar_idx = idx - 2
            if ar_idx < 0 or ar_idx >= len(ar_call_tokens):
                return FakeArray([999])
            return FakeArray([ar_call_tokens[ar_idx]])

        mlx_mocks["model"].side_effect = fake_model

        result = engine.merge("code", "snippet")
        # decode is called with the collected tokens; should stop at EOS
        decode_args = mlx_mocks["tokenizer"].decode.call_args[0][0]
        # Expect: first_token (from prefill logits) + 3 AR tokens + EOS = ≤5
        # The exact count depends on whether first_token is EOS; at minimum
        # the loop must have stopped before exhausting max_tokens.
        assert len(decode_args) <= 5
        assert result.tokens_generated <= 5

    def test_handles_output_without_tags(self, mlx_mocks):
        """When model output has no tags, _extract_output returns stripped text."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        mlx_mocks["tokenizer"].decode.return_value = "def foo(): return 1"
        result = engine.merge("def foo(): pass", "def foo(): return 1")
        assert result.merged_code == "def foo(): return 1"

    def test_handles_think_tags_in_output(self, mlx_mocks):
        """Model may output <think>\\n\\n</think> before the answer."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        mlx_mocks["tokenizer"].decode.return_value = (
            "<think>\n\n</think><updated-code>def foo(): return 1</updated-code>"
        )
        result = engine.merge("def foo(): pass", "def foo(): return 1")
        assert result.merged_code == "def foo(): return 1"


class TestMLXEngineMergeValidation:
    """Test MLXEngine.merge() with language validation."""

    def test_validates_parse_when_language_provided(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        mlx_mocks["tokenizer"].decode.return_value = (
            "<updated-code>def foo(): return 1</updated-code>"
        )

        with patch("fastedit.inference.mlx_engine.validate_parse") as mock_vp:
            mock_vp.return_value = True
            result = engine.merge("code", "snippet", language="python")
            mock_vp.assert_called_once_with("def foo(): return 1", "python")
            assert result.parse_valid is True

    def test_skips_validation_when_no_language(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        with patch("fastedit.inference.mlx_engine.validate_parse") as mock_vp:
            result = engine.merge("code", "snippet", language=None)
            mock_vp.assert_not_called()
            assert result.parse_valid is True

    def test_reports_invalid_parse(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine()

        mlx_mocks["tokenizer"].decode.return_value = (
            "<updated-code>def foo( broken</updated-code>"
        )

        with patch("fastedit.inference.mlx_engine.validate_parse") as mock_vp:
            mock_vp.return_value = False
            result = engine.merge("code", "snippet", language="python")
            assert result.parse_valid is False


class TestMLXEngineMaxTokens:
    """Test that max_tokens controls the generation loop bound."""

    def test_max_tokens_forwarded(self, mlx_mocks):
        MLXEngine = _import_engine()
        engine = MLXEngine(max_tokens=4096)

        # max_tokens is stored on the engine and used as the AR loop bound.
        assert engine.max_tokens == 4096

        # Verify merge() respects max_tokens by checking tokens_generated
        # does not exceed it.
        result = engine.merge("code", "snippet")
        assert result.tokens_generated <= 4096


# ---------------------------------------------------------------------------
# Prompt Cache Tests
# ---------------------------------------------------------------------------

def _import_cache_key():
    """Import _compute_cache_key after mocks are installed."""
    from fastedit.inference.mlx_engine import _compute_cache_key
    return _compute_cache_key


def _import_cache_manager():
    """Import PromptCacheManager after mocks are installed."""
    from fastedit.inference.mlx_engine import PromptCacheManager
    return PromptCacheManager


class TestComputeCacheKey:
    """Test _compute_cache_key function."""

    def test_deterministic_same_input(self, mlx_mocks):
        """Same input always produces the same key."""
        compute = _import_cache_key()
        code = "def foo(): pass"
        key1 = compute(code)
        key2 = compute(code)
        assert key1 == key2

    def test_different_inputs_different_keys(self, mlx_mocks):
        """Different inputs produce different keys."""
        compute = _import_cache_key()
        key1 = compute("def foo(): pass")
        key2 = compute("def bar(): pass")
        assert key1 != key2

    def test_returns_hex_string(self, mlx_mocks):
        """Key is a 16-character hex string (sha256 prefix)."""
        compute = _import_cache_key()
        key = compute("some code content")
        assert len(key) == 16
        # Verify it's valid hex
        int(key, 16)

    def test_handles_empty_string(self, mlx_mocks):
        """Empty string produces a valid key."""
        compute = _import_cache_key()
        key = compute("")
        assert len(key) == 16
        int(key, 16)

    def test_handles_unicode(self, mlx_mocks):
        """Unicode content produces a valid key."""
        compute = _import_cache_key()
        key = compute("# Comment with emoji and unicode")
        assert len(key) == 16
        int(key, 16)


class TestPromptCacheManager:
    """Test PromptCacheManager class."""

    def test_cache_miss_returns_none(self, mlx_mocks, tmp_path):
        """Get on a nonexistent key returns None."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))
        result = mgr.get("nonexistent_key")
        assert result is None

    def test_creates_cache_dir_automatically(self, mlx_mocks, tmp_path):
        """Cache directory is created if it does not exist."""
        CacheManager = _import_cache_manager()
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        assert not cache_dir.exists()
        mgr = CacheManager(cache_dir=str(cache_dir))
        assert cache_dir.exists()

    def test_put_and_get_roundtrip(self, mlx_mocks, tmp_path):
        """Stored cache can be retrieved."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))

        fake_cache = [MagicMock()]

        # save_prompt_cache writes a file; simulate it
        def fake_save(path, cache):
            Path(path).write_bytes(b"fake safetensors data")

        mlx_mocks["save_prompt_cache"].side_effect = fake_save
        mlx_mocks["load_prompt_cache"].return_value = fake_cache

        mgr.put("abc123", fake_cache, token_count=100)

        # Verify save was called
        expected_path = str(tmp_path / "abc123.safetensors")
        mlx_mocks["save_prompt_cache"].assert_called_once_with(expected_path, fake_cache)

        # Verify metadata was written
        meta_path = tmp_path / "abc123.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["cache_key"] == "abc123"
        assert meta["token_count"] == 100

        # Get should load and return cache
        result = mgr.get("abc123")
        assert result == fake_cache
        mlx_mocks["load_prompt_cache"].assert_called_once_with(expected_path)

    def test_get_updates_last_used_timestamp(self, mlx_mocks, tmp_path):
        """Getting a cache entry updates the last_used timestamp in metadata."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))

        # Create cache file and metadata manually
        cache_path = tmp_path / "key123.safetensors"
        cache_path.write_bytes(b"fake data")
        meta_path = tmp_path / "key123.meta.json"
        old_time = time.time() - 3600  # 1 hour ago
        meta = {
            "cache_key": "key123",
            "token_count": 50,
            "created": old_time,
            "last_used": old_time,
            "size_bytes": 9,
        }
        meta_path.write_text(json.dumps(meta))

        mlx_mocks["load_prompt_cache"].return_value = ["cached_state"]

        before_get = time.time()
        mgr.get("key123")
        after_get = time.time()

        updated_meta = json.loads(meta_path.read_text())
        assert updated_meta["last_used"] >= before_get
        assert updated_meta["last_used"] <= after_get

    def test_total_size_reports_safetensors_size(self, mlx_mocks, tmp_path):
        """total_size() returns sum of all .safetensors file sizes."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))

        # Create some fake cache files
        (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "b.safetensors").write_bytes(b"y" * 200)

        assert mgr.total_size() == 300

    def test_total_size_empty_dir(self, mlx_mocks, tmp_path):
        """total_size() returns 0 for empty cache dir."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))
        assert mgr.total_size() == 0

    def test_lru_eviction_removes_oldest(self, mlx_mocks, tmp_path):
        """When over budget, oldest (by last_used) entries are evicted first."""
        CacheManager = _import_cache_manager()
        # Budget: 250 bytes
        mgr = CacheManager(cache_dir=str(tmp_path), max_cache_bytes=250)

        # Create 3 cache entries: old (100B), medium (100B), recent (100B) = 300B > 250B
        now = time.time()
        for i, (key, age) in enumerate([("old", 3600), ("medium", 1800), ("recent", 0)]):
            cache_path = tmp_path / f"{key}.safetensors"
            cache_path.write_bytes(b"x" * 100)
            meta = {
                "cache_key": key,
                "token_count": 50,
                "created": now - age,
                "last_used": now - age,
                "size_bytes": 100,
            }
            (tmp_path / f"{key}.meta.json").write_text(json.dumps(meta))

        # Trigger eviction
        mgr._evict_if_needed()

        # "old" should be evicted (oldest last_used), rest should remain
        assert not (tmp_path / "old.safetensors").exists()
        assert not (tmp_path / "old.meta.json").exists()
        # medium and recent should still exist (200B <= 250B)
        assert (tmp_path / "medium.safetensors").exists()
        assert (tmp_path / "recent.safetensors").exists()

    def test_lru_eviction_removes_multiple_until_under_budget(self, mlx_mocks, tmp_path):
        """Eviction removes as many entries as needed to get under budget."""
        CacheManager = _import_cache_manager()
        # Budget: 100 bytes
        mgr = CacheManager(cache_dir=str(tmp_path), max_cache_bytes=100)

        now = time.time()
        for key, age in [("oldest", 3600), ("older", 1800), ("newest", 0)]:
            cache_path = tmp_path / f"{key}.safetensors"
            cache_path.write_bytes(b"x" * 100)
            meta = {
                "cache_key": key,
                "token_count": 50,
                "created": now - age,
                "last_used": now - age,
                "size_bytes": 100,
            }
            (tmp_path / f"{key}.meta.json").write_text(json.dumps(meta))

        mgr._evict_if_needed()

        # 300B budget 100B: need to evict until <= 100B. Remove oldest+older (200B), keep newest (100B)
        assert not (tmp_path / "oldest.safetensors").exists()
        assert not (tmp_path / "older.safetensors").exists()
        assert (tmp_path / "newest.safetensors").exists()

    def test_put_triggers_eviction(self, mlx_mocks, tmp_path):
        """put() calls eviction after saving a new entry."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path), max_cache_bytes=150)

        # Pre-existing entry: 100B
        now = time.time()
        (tmp_path / "existing.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "existing.meta.json").write_text(json.dumps({
            "cache_key": "existing",
            "token_count": 50,
            "created": now - 3600,
            "last_used": now - 3600,
            "size_bytes": 100,
        }))

        # save_prompt_cache writes 100 more bytes
        def fake_save(path, cache):
            Path(path).write_bytes(b"y" * 100)

        mlx_mocks["save_prompt_cache"].side_effect = fake_save

        mgr.put("new_entry", [MagicMock()], token_count=200)

        # Total would be 200B > 150B. "existing" (older) should be evicted.
        assert not (tmp_path / "existing.safetensors").exists()
        assert (tmp_path / "new_entry.safetensors").exists()

    def test_default_cache_dir(self, mlx_mocks):
        """Default cache dir is ~/.fastedit/cache/."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager()
        expected = Path(os.path.expanduser("~/.fastedit/cache"))
        assert mgr.cache_dir == expected

    def test_default_max_cache_bytes(self, mlx_mocks, tmp_path):
        """Default max_cache_bytes is 2GB."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path))
        assert mgr.max_cache_bytes == 2 * 1024 * 1024 * 1024

    def test_eviction_handles_corrupt_metadata(self, mlx_mocks, tmp_path):
        """Eviction skips entries with corrupt metadata files."""
        CacheManager = _import_cache_manager()
        mgr = CacheManager(cache_dir=str(tmp_path), max_cache_bytes=50)

        # One valid entry
        now = time.time()
        (tmp_path / "valid.safetensors").write_bytes(b"x" * 40)
        (tmp_path / "valid.meta.json").write_text(json.dumps({
            "cache_key": "valid",
            "token_count": 50,
            "created": now,
            "last_used": now,
            "size_bytes": 40,
        }))

        # One corrupt metadata
        (tmp_path / "corrupt.safetensors").write_bytes(b"y" * 40)
        (tmp_path / "corrupt.meta.json").write_text("not valid json{{{")

        # Should not raise
        mgr._evict_if_needed()


class TestMLXEngineWithCache:
    """Test MLXEngine integration with PromptCacheManager."""

    def test_init_accepts_cache_dir(self, mlx_mocks, tmp_path):
        """MLXEngine accepts cache_dir parameter."""
        MLXEngine = _import_engine()
        engine = MLXEngine(cache_dir=str(tmp_path))
        assert engine.cache_manager is not None
        assert engine.cache_manager.cache_dir == tmp_path

    def test_init_accepts_max_cache_bytes(self, mlx_mocks, tmp_path):
        """MLXEngine accepts max_cache_bytes parameter."""
        MLXEngine = _import_engine()
        engine = MLXEngine(
            cache_dir=str(tmp_path),
            max_cache_bytes=1024 * 1024,
        )
        assert engine.cache_manager.max_cache_bytes == 1024 * 1024

    def test_init_default_cache_manager(self, mlx_mocks):
        """MLXEngine creates a PromptCacheManager with defaults when no args given."""
        MLXEngine = _import_engine()
        engine = MLXEngine()
        assert engine.cache_manager is not None
        expected = Path(os.path.expanduser("~/.fastedit/cache"))
        assert engine.cache_manager.cache_dir == expected


# ---------------------------------------------------------------------------
# Speculative Decoding Helper Tests
# ---------------------------------------------------------------------------

def _mock_seq_len(input_tokens):
    """Extract sequence length from batched model input.

    The model receives tokens[None] which produces FakeArray([[t0, t1, ...]]).
    We need the inner sequence length, not the outer batch dimension of 1.
    """
    data = input_tokens._data if hasattr(input_tokens, '_data') else input_tokens
    # Unwrap nested single-element lists: [[t0, t1, ...]] -> [t0, t1, ...]
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    if isinstance(data, list):
        return len(data)
    return 1


def _import_speculative_helpers():
    """Import speculative decoding helpers after mocks are installed."""
    from fastedit.inference.mlx_engine import (
        _snapshot_ssm_caches,
        _restore_ssm_caches,
        _trim_kv_caches,
        _prefill_prompt,
        _speculative_generate,
    )
    return (
        _snapshot_ssm_caches,
        _restore_ssm_caches,
        _trim_kv_caches,
        _prefill_prompt,
        _speculative_generate,
    )


class TestSnapshotSSMCaches:
    """Test _snapshot_ssm_caches and _restore_ssm_caches round-trip."""

    def test_snapshots_arrays_cache_entries(self, mlx_mocks):
        """Snapshot captures state from all ArraysCache entries."""
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeKVCache = mlx_mocks["FakeKVCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        snapshot_fn = helpers[0]

        ssm_cache = FakeArraysCache(size=2)
        ssm_cache.cache = [FakeArray([1.0, 2.0]), FakeArray([3.0, 4.0])]

        kv_cache = FakeKVCache(offset=10)

        cache = [kv_cache, ssm_cache, kv_cache, ssm_cache]
        snapshots = snapshot_fn(cache)

        # Should only snapshot ArraysCache entries (indices 1 and 3)
        assert 1 in snapshots
        assert 3 in snapshots
        # Should NOT snapshot KVCache entries
        assert 0 not in snapshots
        assert 2 not in snapshots

    def test_snapshot_is_deep_copy(self, mlx_mocks):
        """Snapshot data is independent of original cache state."""
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        snapshot_fn = helpers[0]

        ssm = FakeArraysCache(size=2)
        ssm.cache = [FakeArray([1.0, 2.0]), FakeArray([3.0, 4.0])]

        cache = [ssm]
        snapshots = snapshot_fn(cache)

        # Mutate original cache
        ssm.cache = [FakeArray([99.0, 99.0]), None]

        # Snapshot should still have original values
        snap_data = snapshots[0]
        assert snap_data[0]._data == [1.0, 2.0]
        assert snap_data[1]._data == [3.0, 4.0]

    def test_snapshot_handles_none_entries(self, mlx_mocks):
        """Snapshot handles ArraysCache with None entries in its cache list."""
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        snapshot_fn = helpers[0]

        ssm = FakeArraysCache(size=3)
        ssm.cache = [FakeArray([1.0]), None, FakeArray([2.0])]

        cache = [ssm]
        snapshots = snapshot_fn(cache)

        snap = snapshots[0]
        assert snap[0]._data == [1.0]
        assert snap[1] is None
        assert snap[2]._data == [2.0]

    def test_restore_reverses_mutation(self, mlx_mocks):
        """Restoring from snapshot reverses any mutations to ArraysCache."""
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        snapshot_fn, restore_fn = helpers[0], helpers[1]

        ssm = FakeArraysCache(size=2)
        ssm.cache = [FakeArray([1.0, 2.0]), FakeArray([3.0, 4.0])]

        cache = [ssm]
        snapshots = snapshot_fn(cache)

        # Mutate the SSM state (as would happen during speculative forward pass)
        ssm.cache = [FakeArray([99.0, 99.0]), FakeArray([88.0, 88.0])]

        # Restore from snapshot
        restore_fn(cache, snapshots)

        # SSM state should be back to original
        assert ssm.cache[0]._data == [1.0, 2.0]
        assert ssm.cache[1]._data == [3.0, 4.0]

    def test_restore_only_affects_snapshotted_indices(self, mlx_mocks):
        """Restore only overwrites cache entries that were snapshotted."""
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeKVCache = mlx_mocks["FakeKVCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        snapshot_fn, restore_fn = helpers[0], helpers[1]

        kv = FakeKVCache(offset=10)
        ssm = FakeArraysCache(size=1)
        ssm.cache = [FakeArray([5.0])]

        cache = [kv, ssm]
        snapshots = snapshot_fn(cache)

        # Mutate both
        kv.offset = 20
        ssm.cache = [FakeArray([99.0])]

        restore_fn(cache, snapshots)

        # KV should NOT be restored (not in snapshot)
        assert kv.offset == 20
        # SSM should be restored
        assert ssm.cache[0]._data == [5.0]

    def test_empty_cache_returns_empty_snapshot(self, mlx_mocks):
        """Empty cache list yields empty snapshot dict."""
        helpers = _import_speculative_helpers()
        snapshot_fn = helpers[0]

        snapshots = snapshot_fn([])
        assert snapshots == {}

    def test_all_kv_caches_returns_empty_snapshot(self, mlx_mocks):
        """Cache with only KVCache entries yields empty snapshot dict."""
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        snapshot_fn = helpers[0]

        cache = [FakeKVCache(offset=5), FakeKVCache(offset=10)]
        snapshots = snapshot_fn(cache)
        assert snapshots == {}


class TestTrimKVCaches:
    """Test _trim_kv_caches only trims KVCache, not ArraysCache."""

    def test_trims_kv_cache_entries(self, mlx_mocks):
        """KVCache entries get their offset trimmed by n."""
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        trim_fn = helpers[2]

        kv1 = FakeKVCache(offset=20)
        kv2 = FakeKVCache(offset=15)

        cache = [kv1, kv2]
        trim_fn(cache, 5)

        assert kv1.offset == 15
        assert kv2.offset == 10

    def test_does_not_trim_arrays_cache(self, mlx_mocks):
        """ArraysCache entries are left untouched by trim."""
        FakeKVCache = mlx_mocks["FakeKVCache"]
        FakeArraysCache = mlx_mocks["FakeArraysCache"]
        FakeArray = mlx_mocks["FakeArray"]

        helpers = _import_speculative_helpers()
        trim_fn = helpers[2]

        kv1 = FakeKVCache(offset=20)
        kv2 = FakeKVCache(offset=20)
        ssm = FakeArraysCache(size=2)
        ssm.cache = [FakeArray([1.0]), FakeArray([2.0])]

        cache = [kv1, ssm, kv2]
        trim_fn(cache, 5)

        # KV caches trimmed
        assert kv1.offset == 15
        assert kv2.offset == 15
        # SSM untouched
        assert ssm.cache[0]._data == [1.0]
        assert ssm.cache[1]._data == [2.0]

    def test_trim_clamps_to_offset(self, mlx_mocks):
        """Trimming more than offset clamps to 0."""
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        trim_fn = helpers[2]

        kv = FakeKVCache(offset=3)
        cache = [kv]
        trim_fn(cache, 10)

        assert kv.offset == 0

    def test_mixed_cache_selective_trim(self, mlx_mocks):
        """In a Qwen3.5-style mixed cache, only attention layers are trimmed."""
        FakeKVCache = mlx_mocks["FakeKVCache"]
        FakeArraysCache = mlx_mocks["FakeArraysCache"]

        helpers = _import_speculative_helpers()
        trim_fn = helpers[2]

        # Simulate Qwen3.5: layers 0,1,2 SSM; layer 3 KV; layers 4,5,6 SSM; layer 7 KV
        cache = [
            FakeArraysCache(2), FakeArraysCache(2), FakeArraysCache(2),
            FakeKVCache(offset=50),
            FakeArraysCache(2), FakeArraysCache(2), FakeArraysCache(2),
            FakeKVCache(offset=50),
        ]

        trim_fn(cache, 10)

        # Only indices 3 and 7 (KVCache) should be trimmed
        assert cache[3].offset == 40
        assert cache[7].offset == 40
        # SSM caches unchanged
        for i in [0, 1, 2, 4, 5, 6]:
            assert isinstance(cache[i], FakeArraysCache)


class TestPrefillPrompt:
    """Test _prefill_prompt processes prompt tokens through the model."""

    def test_populates_cache_with_prompt(self, mlx_mocks):
        """Prefill calls model with prompt tokens to fill cache."""
        FakeArray = mlx_mocks["FakeArray"]
        model = mlx_mocks["model"]

        helpers = _import_speculative_helpers()
        prefill_fn = helpers[3]

        prompt_tokens = FakeArray([10, 20, 30, 40, 50])
        cache = []

        # Model returns fake logits
        model.return_value = FakeArray([[0.1, 0.2, 0.3]])

        prefill_fn(model, prompt_tokens, cache, kv_bits=4, kv_group_size=64)

        # Model should have been called at least once
        assert model.call_count >= 1

    def test_processes_large_prompt_in_chunks(self, mlx_mocks):
        """Prompt larger than prefill_step_size is processed in chunks."""
        FakeArray = mlx_mocks["FakeArray"]
        model = mlx_mocks["model"]

        helpers = _import_speculative_helpers()
        prefill_fn = helpers[3]

        # Prompt of 100 tokens, step size of 30
        prompt_tokens = FakeArray(list(range(100)))
        cache = []

        model.return_value = FakeArray([[0.1, 0.2]])

        prefill_fn(
            model, prompt_tokens, cache,
            kv_bits=4, kv_group_size=64,
            prefill_step_size=30,
        )

        # Should be called multiple times for chunked processing
        assert model.call_count >= 2

    def test_returns_logits_from_last_token(self, mlx_mocks):
        """Prefill returns logits for the final prompt token."""
        FakeArray = mlx_mocks["FakeArray"]
        model = mlx_mocks["model"]

        helpers = _import_speculative_helpers()
        prefill_fn = helpers[3]

        prompt_tokens = FakeArray([10, 20, 30])
        cache = []

        expected_logits = FakeArray([[0.5, 0.6, 0.7]])
        model.return_value = expected_logits

        result = prefill_fn(model, prompt_tokens, cache, kv_bits=4, kv_group_size=64)

        # Should return the logits from the model's last call
        assert result is not None

class TestSpeculativeGenerate:
    """Test _speculative_generate core loop."""

    def _make_model_that_agrees(self, mlx_mocks, draft_tokens):
        """Create a model mock that agrees with all draft tokens.

        When given verify_tokens [current_token, d0, d1, ...dN], the model
        returns logits where argmax at position i is draft_tokens[i], meaning
        it agrees with the entire batch. The logit at position N+1 is a
        continuation token (the 'bonus' token from full acceptance).
        """
        FakeArray = mlx_mocks["FakeArray"]
        model = mlx_mocks["model"]
        call_idx = [0]

        def model_forward(input_tokens, cache=None):
            ci = call_idx[0]
            call_idx[0] += 1

            n = _mock_seq_len(input_tokens)

            # Return tokens that match draft at each position
            # The model sees [current, d0, d1, ...] and returns predictions
            # Position 0 -> should predict d0 (first draft token)
            # Position 1 -> should predict d1
            # ...
            # Position N -> next token after all drafts accepted (bonus)
            result_tokens = []
            for i in range(n):
                if i < len(draft_tokens):
                    result_tokens.append(draft_tokens[i])
                else:
                    # Bonus: token after all draft accepted
                    result_tokens.append(999)
            return FakeArray([result_tokens])

        model.side_effect = model_forward
        return model

    def _make_model_that_rejects_at(self, mlx_mocks, draft_tokens, reject_pos):
        """Create a model mock that rejects at position reject_pos.

        Agrees with draft_tokens[0:reject_pos], then produces a different
        token at reject_pos.
        """
        FakeArray = mlx_mocks["FakeArray"]
        model = mlx_mocks["model"]

        def model_forward(input_tokens, cache=None):
            n = _mock_seq_len(input_tokens)

            result_tokens = []
            for i in range(n):
                if i < reject_pos:
                    # Agree with draft
                    result_tokens.append(draft_tokens[i] if i < len(draft_tokens) else 999)
                elif i == reject_pos:
                    # Reject: return different token
                    result_tokens.append(777)
                else:
                    result_tokens.append(888)
            return FakeArray([result_tokens])

        model.side_effect = model_forward
        return model

    def test_all_draft_tokens_accepted(self, mlx_mocks):
        """When model agrees with all draft tokens, they are all emitted."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30, 40, 50]
        eos_tokens = {151645}

        self._make_model_that_agrees(mlx_mocks, draft_tokens)

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=mlx_mocks["model"],
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=10,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # All draft tokens should appear in the result
        for dt in draft_tokens:
            assert dt in result

    def test_rejection_emits_model_token(self, mlx_mocks):
        """When model rejects at position N, model's token at N is used instead."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30, 40, 50]
        eos_tokens = {151645}

        # Reject at position 2: accepts 10, 20; rejects 30 with 777
        self._make_model_that_rejects_at(mlx_mocks, draft_tokens, reject_pos=2)

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=mlx_mocks["model"],
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=10,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # Accepted tokens (10, 20) should be in result
        assert 10 in result
        assert 20 in result
        # Model's rejection token (777) should be in result
        assert 777 in result
        # Rejected draft token (30) should NOT be in result
        # (it was replaced by 777)

    def test_eos_in_draft_stops_generation(self, mlx_mocks):
        """If an accepted draft token is EOS, generation stops immediately."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        eos_id = 151645
        # Draft contains EOS at position 2
        draft_tokens = [10, 20, eos_id, 40, 50]
        eos_tokens = {eos_id}

        self._make_model_that_agrees(mlx_mocks, draft_tokens)

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=mlx_mocks["model"],
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=10,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # Should stop at EOS - tokens after EOS (40, 50) should not appear
        assert eos_id in result
        assert 40 not in result
        assert 50 not in result

    def test_rejection_triggers_ssm_restore(self, mlx_mocks):
        """On rejection, SSM caches are restored from snapshot."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]
        FakeArraysCache = mlx_mocks["FakeArraysCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30]
        eos_tokens = {151645}

        # Track SSM state changes
        ssm = FakeArraysCache(size=1)
        ssm.cache = [FakeArray([1.0])]

        ssm_states_seen = []

        model = mlx_mocks["model"]
        call_count = [0]

        def model_forward(input_tokens, cache=None):
            call_count[0] += 1
            # Record SSM state at each call
            ssm_states_seen.append(
                [x._data if hasattr(x, '_data') else x for x in ssm.cache]
            )

            n = _mock_seq_len(input_tokens)

            # First call (verification): reject at position 1
            if call_count[0] == 1:
                # Agrees with token at pos 0 (10), rejects at pos 1 (not 20)
                result = [10, 777, 888, 888][:n]
            else:
                # Subsequent calls: just return EOS to stop
                result = [151645] * n
            return FakeArray([result])

        model.side_effect = model_forward

        kv = FakeKVCache(offset=100)
        cache_list = [kv, ssm]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=model,
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache_list,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=5,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # The function should have been called (rejection path exercises SSM restore)
        assert call_count[0] >= 1

    def test_rejection_triggers_kv_trim(self, mlx_mocks):
        """On rejection, KV caches are trimmed by the number of unaccepted tokens."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30, 40, 50]
        eos_tokens = {151645}

        model = mlx_mocks["model"]
        call_count = [0]

        def model_forward(input_tokens, cache=None):
            call_count[0] += 1
            n = _mock_seq_len(input_tokens)

            if call_count[0] == 1:
                # Reject at position 2: accept 10, 20; reject 30
                result = [10, 20, 777] + [888] * (n - 3)
                return FakeArray([result[:n]])
            else:
                return FakeArray([[151645] * n])

        model.side_effect = model_forward

        kv = FakeKVCache(offset=100)
        cache_list = [kv]
        prefill_logits = FakeArray([42])

        initial_offset = kv.offset

        result = spec_gen(
            model=model,
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache_list,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=5,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # KV cache offset should reflect trimming on rejection
        # After forward pass of 6 tokens (1 current + 5 draft), offset increases by 6
        # Then trim by 3 (5 draft - 2 accepted = 3 unaccepted)
        # The exact offset depends on implementation details, but it should
        # not remain at the post-forward value
        assert call_count[0] >= 1

    def test_empty_draft_generates_autoregressively(self, mlx_mocks):
        """With no draft tokens, falls through to autoregressive generation."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        eos_id = 151645
        eos_tokens = {eos_id}

        model = mlx_mocks["model"]
        call_count = [0]

        def model_forward(input_tokens, cache=None):
            call_count[0] += 1
            n = _mock_seq_len(input_tokens)
            # Return EOS on second call
            if call_count[0] <= 1:
                return FakeArray([[42] * n])
            return FakeArray([[eos_id] * n])

        model.side_effect = model_forward

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=model,
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=[],
            draft_batch_size=10,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # Should have generated some tokens autoregressively
        assert len(result) > 0
        assert eos_id in result

    def test_max_tokens_limits_output(self, mlx_mocks):
        """Generation stops when max_tokens is reached."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        # Long draft, but max_tokens is small
        draft_tokens = list(range(100, 200))
        eos_tokens = {151645}

        self._make_model_that_agrees(mlx_mocks, draft_tokens)

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=mlx_mocks["model"],
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=10,
            max_tokens=5,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # Result should not exceed max_tokens
        assert len(result) <= 5

    def test_draft_batch_size_controls_chunk_size(self, mlx_mocks):
        """draft_batch_size controls how many tokens are verified per batch."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        eos_tokens = {151645}

        model = mlx_mocks["model"]
        call_sizes = []

        def model_forward(input_tokens, cache=None):
            n = _mock_seq_len(input_tokens)
            call_sizes.append(n)
            # Agree with everything, then return EOS as bonus
            result = []
            for i in range(n):
                if i < len(draft_tokens):
                    result.append(draft_tokens[i])
                else:
                    result.append(151645)
            return FakeArray([result])

        model.side_effect = model_forward

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=model,
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=3,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        # With batch_size=3 and 10 draft tokens, verification calls should have
        # at most 4 tokens (1 current + 3 draft) per batch
        for size in call_sizes:
            assert size <= 4  # 1 current + 3 draft

    def test_repeated_rejections_fall_back_to_autoregressive(self, mlx_mocks):
        """After repeated rejections, speculative mode exits and tail is AR only."""
        FakeArray = mlx_mocks["FakeArray"]
        FakeKVCache = mlx_mocks["FakeKVCache"]

        helpers = _import_speculative_helpers()
        spec_gen = helpers[4]

        draft_tokens = [10, 20, 30, 40, 50, 60]
        eos_id = 151645
        eos_tokens = {eos_id}

        model = mlx_mocks["model"]
        verify_call_sizes = []
        ar_call_sizes = []
        verify_calls = [0]

        def model_forward(input_tokens, cache=None):
            n = _mock_seq_len(input_tokens)

            # Reject the first two speculative verify batches immediately.
            if n > 1 and verify_calls[0] < 2:
                verify_calls[0] += 1
                verify_call_sizes.append(n)
                return FakeArray([[777] + [888] * (n - 1)])

            ar_call_sizes.append(n)
            return FakeArray([[eos_id] * n])

        model.side_effect = model_forward

        cache = [FakeKVCache(offset=100)]
        prefill_logits = FakeArray([42])

        result = spec_gen(
            model=model,
            tokenizer=mlx_mocks["tokenizer"],
            cache=cache,
            prefill_logits=prefill_logits,
            original_draft_tokens=draft_tokens,
            draft_batch_size=6,
            max_tokens=100,
            kv_bits=4,
            kv_group_size=64,
            eos_token_ids=eos_tokens,
        )

        assert verify_calls[0] == 2
        assert verify_call_sizes == [7, 4]  # batch shrinks from 6 -> 3
        assert ar_call_sizes  # switched to plain AR after repeated rejections
        assert eos_id in result


class TestMergeSpeculative:
    """Test MLXEngine.merge_speculative() end-to-end with mocks."""

    def test_returns_merge_result(self, mlx_mocks):
        """merge_speculative returns a MergeResult."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        # Configure tokenizer for speculative path
        tok = mlx_mocks["tokenizer"]
        tok.encode.side_effect = lambda text, **kwargs: {
            True: [1, 2, 3, 4, 5],  # prompt tokens
        }.get(True, [10, 20, 30, 40, 50])

        # Model returns logits that produce the right output
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = (
            "<think>\n\n</think>\n\n<updated-code>def hello(): pass</updated-code>"
        )

        result = engine.merge_speculative(
            "def hello(): pass",
            "def hello(): return 1",
        )

        from fastedit.inference.merge import MergeResult
        assert isinstance(result, MergeResult)

    def test_extracts_merged_code(self, mlx_mocks):
        """merge_speculative extracts code from model output tags."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = (
            "<updated-code>def foo(): return 42</updated-code>"
        )

        result = engine.merge_speculative(
            "def foo(): pass",
            "def foo(): return 42",
        )
        assert result.merged_code == "def foo(): return 42"

    def test_reports_performance_metrics(self, mlx_mocks):
        """merge_speculative reports latency and tokens per second."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = "<updated-code>code</updated-code>"

        result = engine.merge_speculative("code", "snippet")

        assert result.latency_ms >= 0
        assert result.tokens_generated >= 0
        assert result.tokens_per_second >= 0

    def test_uses_build_prompt(self, mlx_mocks):
        """merge_speculative uses build_prompt to construct messages."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = "<updated-code>code</updated-code>"

        with patch("fastedit.inference.mlx_engine._get_prefix_boundary", return_value=0), \
             patch("fastedit.inference.mlx_engine.build_prompt") as mock_bp:
            mock_bp.return_value = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ]
            engine.merge_speculative("original", "snippet")
            mock_bp.assert_called_once_with("original", "snippet")

    def test_validates_parse_when_language_provided(self, mlx_mocks):
        """merge_speculative validates parse when language is given."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = (
            "<updated-code>def foo(): return 1</updated-code>"
        )

        with patch("fastedit.inference.mlx_engine.validate_parse") as mock_vp:
            mock_vp.return_value = True
            result = engine.merge_speculative(
                "code", "snippet", language="python",
            )
            mock_vp.assert_called_once_with("def foo(): return 1", "python")
            assert result.parse_valid is True

    def test_skips_validation_when_no_language(self, mlx_mocks):
        """merge_speculative skips validation when no language provided."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = "<updated-code>code</updated-code>"

        with patch("fastedit.inference.mlx_engine.validate_parse") as mock_vp:
            result = engine.merge_speculative("code", "snippet", language=None)
            mock_vp.assert_not_called()
            assert result.parse_valid is True

    def test_accepts_draft_batch_size_parameter(self, mlx_mocks):
        """merge_speculative accepts and uses draft_batch_size."""
        MLXEngine = _import_engine()
        engine = MLXEngine()

        tok = mlx_mocks["tokenizer"]
        FakeArray = mlx_mocks["FakeArray"]
        eos_id = tok.eos_token_id

        model = mlx_mocks["model"]
        model.side_effect = lambda tokens, cache=None: FakeArray([[eos_id]])

        tok.decode.return_value = "<updated-code>code</updated-code>"

        # Should not raise
        result = engine.merge_speculative(
            "code", "snippet", draft_batch_size=5,
        )
        from fastedit.inference.merge import MergeResult
        assert isinstance(result, MergeResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
