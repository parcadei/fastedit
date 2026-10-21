"""Tolerant third-party probes for the MLX engine.

The engine's ``import transformers`` (only for logging-verbosity
suppression around ``load()``) is NOT a plain availability probe: a
half-broken transformers install raises on import, and an environment
holding a spec-less module named ``mlx`` (test stub ecosystems, patched
sys.modules) makes transformers' own ``is_mlx_available`` availability
probe — ``importlib.util.find_spec("mlx")`` — raise ``ValueError:
mlx.__spec__ is None`` instead of returning None. Either failure must be
treated as "not available": the suppression is best-effort, and the
engine must still load.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def import_transformers_quietly() -> ModuleType | None:
    """Import transformers, or return None when it is not usable.

    Failure modes treated as 'not available' (never re-raised):

    * ``ImportError``/``ModuleNotFoundError`` — transformers not installed
      or a broken local install;
    * ``ValueError`` — a spec-less module named ``mlx`` (or any parent
      package in the probe chain, e.g. a stub installed into
      ``sys.modules`` without an ``__spec__``) makes transformers'
      internal ``importlib.util.find_spec`` raise
      ``ValueError: mlx.__spec__ is None`` instead of returning None.
    """
    try:
        return importlib.import_module("transformers")
    except Exception:  # noqa: BLE001 -- best-effort probe, see docstring
        return None
