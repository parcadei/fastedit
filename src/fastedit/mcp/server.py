"""MCP server core: FastMCP instance, lifespan, ModelPool, and entrypoint.

Claude generates a compact edit snippet, this tool merges it into the
original file locally via the in-process MLX engine (~250 tok/s on M3 Max).
Result: ~47% reduction in output tokens per file edit.

Usage:
    .mlx-venv/bin/python -m fastedit.mcp_server

Then add to Claude Code settings (~/.claude.json):
    {
      "mcpServers": {
        "fastedit": {
          "command": "/path/to/FastEdit/.mlx-venv/bin/python",
          "args": ["-m", "fastedit.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .backup import (  # noqa: F401 — re-export for existing importers
    BackupStore,
    _atomic_write,
)

logger = logging.getLogger("fastedit.mcp")


class ModelPool:
    """Async-safe pool of MLX engines with lazy loading.

    Provides ``async with pool.acquire() as engine:`` for safe concurrent
    access.  The pool lazily creates engines on first acquire.
    """

    def __init__(self, model_path: str, size: int = 1):
        self._model_path = model_path
        self._size = size
        self._engines: list[Any] = []
        self._semaphore = asyncio.Semaphore(size)
        self._lock = asyncio.Lock()
        self._loaded = False

    async def _ensure_loaded(self):
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            from ..inference.mlx_engine import MLXEngine
            for _ in range(self._size):
                engine = MLXEngine(self._model_path)
                self._engines.append(engine)
            self._loaded = True
            logger.info("MLX pool loaded: %d engine(s) from %s", self._size, self._model_path)

    @contextlib.asynccontextmanager
    async def acquire(self):
        await self._ensure_loaded()
        await self._semaphore.acquire()
        engine = self._engines[0]
        try:
            yield engine
        finally:
            self._semaphore.release()


@contextlib.asynccontextmanager
async def lifespan(server: FastMCP):
    """Load the configured merge backend at startup."""
    backend_kind = os.environ.get("FASTEDIT_BACKEND", "mlx").lower()

    if backend_kind == "llm":
        from ..inference.llm_engine import LLMEngine
        api_base = os.environ.get("FASTEDIT_LLM_API_BASE", "http://127.0.0.1:8000/v1")
        model = os.environ.get("FASTEDIT_LLM_MODEL", "fastedit")
        api_key = os.environ.get("FASTEDIT_LLM_API_KEY", "not-needed")
        max_tokens = int(os.environ.get("FASTEDIT_LLM_MAX_TOKENS", "16384"))
        backend = LLMEngine(
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        logger.info("LLM backend ready: %s @ %s", model, api_base)
    else:
        from ..model_download import get_model_path
        model_path = get_model_path()
        pool_size = int(os.environ.get("FASTEDIT_POOL_SIZE", "1"))
        backend = ModelPool(model_path=model_path, size=pool_size)

    yield {
        "backend_kind": backend_kind,
        "backend": backend,
        "snapshots": {},
        # 1-deep undo buffer: persists to ~/.fastedit/backups/
        "backups": BackupStore(),
        # File locks: per-path locks preventing read-modify-write races
        # when multiple sessions edit the same file simultaneously
        "file_locks": defaultdict(asyncio.Lock),
    }


mcp = FastMCP(
    "fastedit",
    instructions=(
        "FastEdit merges compact code edit snippets into source files. "
        "Choose the right tool for each operation:\n"
        "- fast_edit: Modify or insert code. Write a short snippet with "
        "'// ... existing code ...' or '# ... existing code ...' markers "
        "for unchanged regions, plus 1-2 lines of context around each change. "
        "For inserting NEW code (new function, class, method): ALWAYS set `after` "
        "to the name of the function/class/method it should follow "
        "(e.g. after='parse_model_output'). Just write the new code — no markers needed. "
        "For replacing an entire function/class/method/constant: set `replace` to its name "
        "(e.g. replace='old_handler' or replace='MAX_RETRIES') — the snippet becomes its replacement. "
        "Do NOT insert new code without `after` — it will cause whole-file merge.\n"
        "IMPORTANT — modifying lines inside an existing function: use "
        "`replace='function_name'` with context markers. Example:\n"
        "  replace='process_file'\n"
        "  snippet:\n"
        "    def process_file(self, path):\n"
        '        if self.debug: print(f"Processing: {path}")  # new line\n'
        "        # ... existing code ...\n"
        "The model sees ONLY the ~35-line region around your change, not the whole file. "
        "This works on ANY size function — even 500+ lines. Fast (~1-2s), accurate, "
        "and preserves all existing code via automatic hallucination detection. "
        "NEVER send context markers without `after` or `replace` on large files — "
        "it triggers whole-file merge which fails on files >150 lines.\n"
        "- fast_batch_edit: When making 2+ edits to the same file, batch "
        "them in one call instead of multiple fast_edit calls. Pass `edits` as a "
        "JSON list of objects, each with a `snippet` key and optional `after` or "
        "`replace` keys. Example: "
        '[{"snippet": "import redis", "after": "import json"}, '
        '{"snippet": "def new_func(): ...", "after": "existing_func"}]. '
        "More efficient than multiple fast_edit calls — one MCP round-trip "
        "instead of N, and intermediate AST analysis is handled automatically.\n"
        "- fast_delete: Remove a function/class/method by name. Instant, "
        "no model — always prefer this for deletions.\n"
        "- fast_move: Move a function/class/method to after another symbol. "
        "Instant, no model — use for code reorganization.\n"
        "- fast_multi_edit: Edit across multiple files in one call. "
        "Sequential — changes in file A are written before editing file B.\n"
        "- fast_read: See a file's structure (functions, classes, line ranges) "
        "without reading the full content. Use FIRST to learn symbol names for "
        "after/replace/delete/move. For small files (<100 lines) returns full content.\n"
        "- fast_search: Search a codebase for functions, classes, and symbols. "
        "Four modes: 'search' (default, BM25-ranked keyword search with structural context), "
        "'regex' (exact pattern matching), 'hybrid' (BM25 ranking + regex filter — pass "
        "regex_filter param), and 'references' (find all call sites, imports, and usages "
        "of a symbol). Returns enriched context cards with signatures, callers/callees, "
        "and code previews. Runs in milliseconds, no model inference.\n"
        "- fast_diff: After editing, verify changes without re-reading "
        "the whole file. Returns a compact unified diff.\n"
        "- fast_rename: Rename all occurrences of a symbol in a file. "
        "Word-boundary aware — renaming 'get' won't touch 'get_all'. "
        "Instant, no model.\n"
        "- fast_undo: Revert the last edit to a file. One level of "
        "undo per file. Works for any edit operation (fast_edit, "
        "fast_delete, fast_move, fast_rename). Backups persist to disk — "
        "survives server restarts and cancelled edits. Instant, no model.\n"
        "Keep snippets minimal — only include changed lines plus 1-2 "
        "lines of surrounding context. The smaller the snippet, the "
        "faster and more accurate the merge."
    ),
    lifespan=lifespan,
)


def main():
    # Log to stderr so MCP stdio channel isn't polluted
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    # Import tool modules to register @mcp.tool decorators
    from . import tools_ast, tools_edit, tools_read  # noqa: F401
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
