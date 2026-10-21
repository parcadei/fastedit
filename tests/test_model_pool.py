"""Step 20 (B40): ``ModelPool.acquire`` must hand out engines round-robin.

The bug: ``acquire()`` always returned ``self._engines[0]``, so a pool of
N engines never parallelized — every concurrent edit queued on the same
model instance and the extra engines only burned memory.

Contract locked down here (hermetic: no model download, no real MLX
engine). ``ModelPool._ensure_loaded`` lazily builds engines by importing
``MLXEngine`` from ``fastedit.inference.mlx_engine`` at call time, so the
smallest seam is that module import: the fixture installs a stub module
into ``sys.modules`` whose ``MLXEngine`` is a factory, so the real
lazy-creation code path runs against fakes. A ``sys.modules`` stub (not a
string-target attribute patch) is required for order-independence: other
suites pop ``fastedit.inference.mlx_engine`` from ``sys.modules`` to force
fresh re-imports, so patching a module attribute could land on a stale
module object while the first ``_ensure_loaded`` re-imports the real one
(and, with it, mlx and the network).

1. Lazy creation is preserved: nothing is built until the first acquire,
   exactly ``size`` engines are built, and only once.
2. Sequential acquisitions cycle across ALL engines in creation order
   (round-robin): 4 acquisitions over a 3-engine pool → [0, 1, 2, 0].
3. A 1-engine pool always yields that one engine.
4. Overlapping acquisitions get DIFFERENT engines when engines are free
   (2 concurrent acquires on a 2-engine pool → engines 0 and 1; 3
   concurrent acquires on a 3-engine pool → 0, 1, 2).
5. Over-subscribed concurrency still serializes safely on the semaphore
   (2 concurrent acquires on a 1-engine pool → both get the single
   engine, never inside the body at the same time).
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from fastedit.mcp.server import ModelPool

MODEL_PATH = "models/fake-merge-model"


class FakeEngine:
    """Stands in for MLXEngine: distinct instances, never loads a model."""

    def __init__(self, model_path: str, **_kwargs):
        self.model_path = model_path


@pytest.fixture
def built_engines(monkeypatch) -> list[FakeEngine]:
    """Redirect lazy engine construction to fakes via the import seam.

    ``_ensure_loaded`` does ``from ..inference.mlx_engine import MLXEngine``
    inside the method body, so a ``sys.modules`` stub takes effect at
    acquire time — no production change, no mlx import, no model, no
    network. ``monkeypatch`` restores the previous ``sys.modules`` entry
    after each test.
    """
    created: list[FakeEngine] = []

    def factory(model_path: str, **kwargs):
        engine = FakeEngine(model_path, **kwargs)
        created.append(engine)
        return engine

    stub = types.ModuleType("fastedit.inference.mlx_engine")
    stub.MLXEngine = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastedit.inference.mlx_engine", stub)
    return created


def _engine_indexes(pool: ModelPool, acquired: list[FakeEngine]) -> list[int]:
    """Map acquired fake engines to their creation order (pool index)."""
    order = {id(engine): i for i, engine in enumerate(pool._engines)}
    return [order[id(engine)] for engine in acquired]


def test_engines_are_built_lazily_and_only_once(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=3)
    assert built_engines == []  # construction must wait for first acquire

    async def scenario():
        async with pool.acquire() as engine:
            assert isinstance(engine, FakeEngine)
        async with pool.acquire() as engine:  # second acquire: no rebuild
            assert isinstance(engine, FakeEngine)

    asyncio.run(scenario())
    assert len(built_engines) == 3  # exactly ``size`` engines, once
    assert all(e.model_path == MODEL_PATH for e in built_engines)


def test_sequential_acquires_cycle_round_robin(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=3)

    async def scenario():
        acquired = []
        for _ in range(4):
            async with pool.acquire() as engine:
                acquired.append(engine)
        return acquired

    acquired = asyncio.run(scenario())
    assert _engine_indexes(pool, acquired) == [0, 1, 2, 0]


def test_single_engine_pool_always_yields_it(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=1)

    async def scenario():
        acquired = []
        for _ in range(3):
            async with pool.acquire() as engine:
                acquired.append(engine)
        return acquired

    acquired = asyncio.run(scenario())
    assert len(built_engines) == 1
    assert all(engine is built_engines[0] for engine in acquired)


def test_overlapping_acquires_get_different_engines(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=2)

    async def scenario():
        acquired = []

        async def hold():
            async with pool.acquire() as engine:
                acquired.append(engine)
                await asyncio.sleep(0.01)  # hold both leases open at once

        await asyncio.gather(hold(), hold())
        return acquired

    acquired = asyncio.run(scenario())
    assert len(acquired) == 2
    assert acquired[0] is not acquired[1]
    assert _engine_indexes(pool, acquired) == [0, 1]


def test_full_pool_overlap_hands_out_every_engine_once(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=3)

    async def scenario():
        acquired = []

        async def hold():
            async with pool.acquire() as engine:
                acquired.append(engine)
                await asyncio.sleep(0.01)

        await asyncio.gather(hold(), hold(), hold())
        return acquired

    acquired = asyncio.run(scenario())
    assert sorted(_engine_indexes(pool, acquired)) == [0, 1, 2]


def test_over_subscribed_acquires_serialize_on_the_single_engine(built_engines):
    pool = ModelPool(model_path=MODEL_PATH, size=1)
    inside = 0
    max_inside = 0

    async def scenario():
        nonlocal inside, max_inside
        acquired = []

        async def hold():
            nonlocal inside, max_inside
            async with pool.acquire() as engine:
                acquired.append(engine)
                inside += 1
                max_inside = max(max_inside, inside)
                await asyncio.sleep(0.01)
                inside -= 1

        await asyncio.gather(hold(), hold())
        return acquired

    acquired = asyncio.run(scenario())
    assert acquired[0] is acquired[1]  # only one engine exists
    assert max_inside == 1  # semaphore still serialized the bodies
