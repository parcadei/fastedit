"""Materialize the C1 medium corpus goldens under tests/golden/big/.

Committed generator (the B3 convention): run

    uv run python tests/golden/big/_generate.py

to re-write every fixture from tests/corpus.py's pinned GOLDEN_BIG_CASES
recipes. The corpus module is the single source of truth — this shim only
points sys.path at tests/ and delegates. The oracle self-test
(tests/test_corpus_oracle.py) re-derives every committed file from the
seed recipe and fails loudly if a fixture drifts from the generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from corpus import write_golden_big_cases  # noqa: E402

if __name__ == "__main__":
    write_golden_big_cases(Path(__file__).resolve().parent)
