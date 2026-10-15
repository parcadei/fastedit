"""Step 18 (B34): the batch/multi MCP tools must honor the same safety
signals ``fast_edit`` honors.

The bug: ``batch_chunked_merge`` dropped ``chunks_rejected`` on the floor,
``fast_batch_edit`` wrote parse-invalid output with only a warning (and had
no ``force`` escape hatch at all), and ``fast_multi_edit`` wrote
unconditionally, merely tagging ``parse_errors`` in its summary. Parse-invalid
merged code could reach N files without even the single-edit warning.

Contract locked down here (mirrors tests/test_mcp_fast_edit_parse_gate.py):

1. ``fast_batch_edit``: a merge whose chunks were ALL rejected (rejected >=
   used) is NOT written; the response is fast_edit's refusal, word for word.
   The pre-existing whole-call aborts for hard errors (bad JSON, missing
   file, unsupported type, undecodable bytes, ValueError) are unchanged.
2. ``fast_batch_edit``: a parse-invalid merge is NOT written unless
   ``force=True``; the escape hatch restores the tool's own
   write-with-warning behavior.
3. ``fast_multi_edit``: per-target gates — a rejected or parse-invalid
   target is NOT written, the remaining targets still write (partial-batch
   semantics; the pre-existing whole-call aborts for hard errors are
   unchanged), and the per-file statuses distinguish ok / rejected /
   parse-errors.
4. ``force=True`` on ``fast_multi_edit`` covers the parse gate only — never
   the hallucination refusal (same rule as fast_edit).
5. ``batch_chunked_merge`` surfaces accumulated ``chunks_rejected`` so the
   gates above can see it.

The unit under test is the tools' WRITE GATE, so ``batch_chunked_merge`` is
stubbed at the ``tools_edit`` module boundary (the Step-8 harness pattern);
the inner ``chunked_merge`` is stubbed for the ``batch_chunked_merge``
accounting tests. Hermetic: no model, no backend, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import defaultdict

import fastedit.inference.chunked_merge as chunked_merge_module
from fastedit.inference.ast_utils import BatchEdit, ChunkedMergeResult
from fastedit.inference.symbols import batch_chunked_merge
from fastedit.mcp import tools_edit
from fastedit.mcp.backup import BackupStore
from fastedit.mcp.server import mcp as fastmcp_server

# A valid Python file, a clean merged output for it, and a parse-invalid one
# (the kind of thing a hallucinating merge produces). The stubs hand back the
# text directly — how the merge produced it is out of scope for these gates.
PY_ORIGINAL = "def existing():\n    return 1\n"
PY_MERGED = "def existing():\n    return 2\n"
PY_BROKEN = "def existing(:\n    return 1\n"


# ---------------------------------------------------------------------------
# Harness: fake server context (no FastMCP runtime, no model, no network)
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    """Mimics ``mcp.server.fastmcp.Context.request_context``."""

    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    """Mimics the object ``mcp.get_context()`` returns inside a tool call."""

    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeEngine:
    """Engine handed out by the fake backend. Never used: batch_chunked_merge
    is stubbed in every tool test, so merge_fn is never invoked."""

    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError(
            "merge_fn must not be called; batch_chunked_merge is stubbed"
        )


class _FakeBackend:
    """Stands in for ModelPool/LLMEngine: acquire() yields a dummy engine."""

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield _FakeEngine()


class _FakeMcp:
    """Stand-in for the FastMCP instance imported into ``tools_edit``."""

    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


def _install_fake_mcp(monkeypatch):
    """Point ``tools_edit.mcp`` at a fake context; disable the update check."""
    lifespan_context = {
        "backend_kind": "mlx",
        # Never loads a model: batch_chunked_merge is stubbed in every tool
        # test, so acquire() yields an engine whose merge_auto is never used.
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_edit, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


def _batch_result(
    merged_code: str,
    parse_valid: bool,
    chunks_used: int = 2,
    chunks_rejected: int = 0,
) -> ChunkedMergeResult:
    """A batch-merge result as the tools' write gate receives it."""
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=chunks_used,
        chunk_regions=[(1, 5)] * chunks_used,
        model_tokens=12,
        latency_ms=40.0,
        chunks_rejected=chunks_rejected,
    )


def _stub_batch(monkeypatch, *results: ChunkedMergeResult):
    """Replace ``tools_edit.batch_chunked_merge`` with a stub handing back
    *results* in order — one per file the tool merges.

    Both call paths inside the tools (the direct mlx call and
    ``asyncio.to_thread``) resolve the name from module globals at call time,
    so the stub covers each of them.
    """
    it = iter(results)
    monkeypatch.setattr(tools_edit, "batch_chunked_merge", lambda *a, **kw: next(it))


def _two_file_edits(target_a, target_b) -> str:
    """The ``file_edits`` JSON for a two-target fast_multi_edit call."""
    return json.dumps([
        {"file_path": str(target_a), "edits": [{"snippet": "x"}, {"snippet": "y"}]},
        {"file_path": str(target_b), "edits": [{"snippet": "x"}, {"snippet": "y"}]},
    ])


def _call(tool_coroutine):
    """Run one tool invocation to completion."""
    return asyncio.run(tool_coroutine)


# ---------------------------------------------------------------------------
# B34: fast_batch_edit gates
# ---------------------------------------------------------------------------


class TestFastBatchEditGates:
    def test_all_chunks_rejected_refuses_and_leaves_file_untouched(
        self, tmp_path, monkeypatch,
    ):
        """(a) rejected >= used: no write, fast_edit's refusal, word for word."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(
            PY_BROKEN, parse_valid=True, chunks_used=2, chunks_rejected=2,
        ))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target), edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        assert message.startswith("Error: edit rejected"), message
        assert "model hallucinated on 2 chunk(s)" in message, message
        assert "File unchanged" in message, message
        assert "Try a smaller edit or split the function" in message, message
        assert target.read_text() == PY_ORIGINAL

    def test_all_chunks_rejected_refusal_not_overridable_by_force(
        self, tmp_path, monkeypatch,
    ):
        """(a) force=True must not buy back a fully-rejected merge."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(
            PY_BROKEN, parse_valid=True, chunks_used=2, chunks_rejected=2,
        ))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target),
            edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
            force=True,
        ))

        assert message.startswith("Error: edit rejected"), message
        assert "File unchanged" in message, message
        assert target.read_text() == PY_ORIGINAL

    def test_parse_invalid_batch_refuses_and_leaves_file_untouched(
        self, tmp_path, monkeypatch,
    ):
        """(b) Parse-invalid merged output: no write, fast_edit's refusal."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(PY_BROKEN, parse_valid=False))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target), edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        assert message.startswith("Error"), message
        assert "parse errors in python" in message, message
        assert "refusing to write" in message, message
        assert "The file is unchanged" in message, message
        assert "smaller edits" in message, message
        assert "force=True" in message, message
        assert target.read_text() == PY_ORIGINAL

    def test_parse_invalid_batch_force_true_writes_with_warning(
        self, tmp_path, monkeypatch,
    ):
        """(b) force=True restores this tool's own escape hatch: write with
        its existing warning, word for word."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(PY_BROKEN, parse_valid=False))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target),
            edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
            force=True,
        ))

        assert message.startswith("Warning"), message
        assert f"parse errors after 2 edits to {target}" in message, message
        assert "Wrote to" in message, message
        assert "anyway" in message, message
        # The (parse-invalid) merged output IS on disk — the caller opted in.
        assert target.read_text() == PY_BROKEN

    def test_partial_rejection_writes_with_warning(
        self, tmp_path, monkeypatch,
    ):
        """Partial rejection (rejected < used) writes with fast_edit's
        warning instead of silently succeeding."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(
            PY_MERGED, parse_valid=True, chunks_used=2, chunks_rejected=1,
        ))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target), edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        assert message.startswith("Warning"), message
        assert "1/2 chunk(s) rejected" in message, message
        assert "Partial edit applied" in message, message
        assert target.read_text() == PY_MERGED

    def test_clean_batch_message_shape_unchanged(self, tmp_path, monkeypatch):
        """Guard: a clean batch is unaffected — written, 'Applied N edits'."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(monkeypatch, _batch_result(PY_MERGED, parse_valid=True))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target), edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        assert message.startswith(f"Applied 2 edits to {target}"), message
        assert (
            "latency: 40ms, 300 tok/s, 12 tokens, 2 chunk(s), 2 edit(s)" in message
        ), message
        assert target.read_text() == PY_MERGED


# ---------------------------------------------------------------------------
# B34: fast_multi_edit per-target gates
# ---------------------------------------------------------------------------


class TestFastMultiEditGates:
    def test_rejected_target_not_written_other_targets_still_write(
        self, tmp_path, monkeypatch,
    ):
        """(c) Per-target refusal: the rejected file is untouched, the clean
        target still writes, and the summary distinguishes the two."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(
            monkeypatch,
            _batch_result(PY_BROKEN, parse_valid=True, chunks_used=2, chunks_rejected=2),
            _batch_result(PY_MERGED, parse_valid=True),
        )
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_multi_edit(
            file_edits=_two_file_edits(target_a, target_b),
        ))

        # Partial-batch semantics: a summary, not a whole-call abort.
        assert message.startswith(
            "Applied 2 edit(s) across 1 of 2 file(s); 1 file(s) not written. "
        ), message
        assert (
            f"{target_a}: 2 edit(s), rejected — model hallucinated on 2 chunk(s). "
            f"File unchanged." in message
        ), message
        assert "Try a smaller edit or split the function" in message, message
        assert f"{target_b}: 2 edit(s), ok" in message, message
        assert target_a.read_text() == PY_ORIGINAL
        assert target_b.read_text() == PY_MERGED

    def test_parse_invalid_target_not_written_other_targets_still_write(
        self, tmp_path, monkeypatch,
    ):
        """(c) A parse-invalid target is not written; the clean target still
        writes; the statuses distinguish parse_errors from rejected."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(
            monkeypatch,
            _batch_result(PY_BROKEN, parse_valid=False),
            _batch_result(PY_MERGED, parse_valid=True),
        )
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_multi_edit(
            file_edits=_two_file_edits(target_a, target_b),
        ))

        assert message.startswith(
            "Applied 2 edit(s) across 1 of 2 file(s); 1 file(s) not written. "
        ), message
        assert (
            f"{target_a}: 2 edit(s), parse_errors — merged output has parse errors "
            f"in python; refusing to write." in message
        ), message
        assert "pass force=True to write anyway" in message, message
        # The status is parse_errors, not a hallucination rejection.
        assert "hallucinated" not in message, message
        assert f"{target_b}: 2 edit(s), ok" in message, message
        assert target_a.read_text() == PY_ORIGINAL
        assert target_b.read_text() == PY_MERGED

    def test_force_true_writes_parse_invalid_target(self, tmp_path, monkeypatch):
        """(d) force=True preserves the parse-gate escape hatch: the target
        writes anyway and its status says so."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(
            monkeypatch,
            _batch_result(PY_BROKEN, parse_valid=False),
            _batch_result(PY_MERGED, parse_valid=True),
        )
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_multi_edit(
            file_edits=_two_file_edits(target_a, target_b), force=True,
        ))

        assert message.startswith("Applied 4 edit(s) across 2 file(s). "), message
        assert (
            f"{target_a}: 2 edit(s), parse_errors — written with force=True "
            f"despite parse errors in python." in message
        ), message
        assert f"{target_b}: 2 edit(s), ok" in message, message
        assert target_a.read_text() == PY_BROKEN
        assert target_b.read_text() == PY_MERGED

    def test_force_true_does_not_override_rejection(self, tmp_path, monkeypatch):
        """(d) force never overrides the hallucination refusal — same rule as
        fast_edit."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(
            monkeypatch,
            _batch_result(PY_BROKEN, parse_valid=True, chunks_used=2, chunks_rejected=2),
            _batch_result(PY_MERGED, parse_valid=True),
        )
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_multi_edit(
            file_edits=_two_file_edits(target_a, target_b), force=True,
        ))

        assert message.startswith(
            "Applied 2 edit(s) across 1 of 2 file(s); 1 file(s) not written. "
        ), message
        assert "rejected — model hallucinated on 2 chunk(s)" in message, message
        assert target_a.read_text() == PY_ORIGINAL
        assert target_b.read_text() == PY_MERGED

    def test_clean_multi_summary_shape_unchanged(self, tmp_path, monkeypatch):
        """Guard: the all-clean summary keeps its exact existing format."""
        _install_fake_mcp(monkeypatch)
        _stub_batch(
            monkeypatch,
            _batch_result(PY_MERGED, parse_valid=True),
            _batch_result(PY_MERGED, parse_valid=True),
        )
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_multi_edit(
            file_edits=_two_file_edits(target_a, target_b),
        ))

        assert message == (
            "Applied 4 edit(s) across 2 file(s). "
            "latency: 80ms, 300 tok/s, 24 tokens\n"
            f"{target_a}: 2 edit(s), ok\n"
            f"{target_b}: 2 edit(s), ok"
        ), message


# ---------------------------------------------------------------------------
# B34: batch_chunked_merge must surface rejection counts
# ---------------------------------------------------------------------------


def _per_edit_result(merged_code, chunks_rejected):
    """What the inner chunked_merge hands back per edit."""
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=True,
        chunks_used=1,
        chunk_regions=[(1, 3)],
        model_tokens=0,
        latency_ms=0.0,
        chunks_rejected=chunks_rejected,
    )


class TestBatchChunkedMergeSurfacesRejections:
    def test_accumulates_per_edit_rejections(self, tmp_path, monkeypatch):
        """(e) Per-edit chunks_rejected sums onto the batch result."""
        per_edit = iter([
            _per_edit_result(PY_MERGED, chunks_rejected=1),
            _per_edit_result(PY_MERGED, chunks_rejected=1),
        ])
        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", lambda *a, **kw: next(per_edit),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        result = batch_chunked_merge(
            original_code=PY_ORIGINAL,
            edits=[BatchEdit(snippet="x"), BatchEdit(snippet="y")],
            file_path=str(target),
            merge_fn=lambda *a, **kw: None,  # never called: chunked_merge stubbed
            language="python",
        )

        assert result.chunks_rejected == 2, result
        assert result.chunks_used == 2, result

    def test_zero_rejections_when_edits_are_clean(self, tmp_path, monkeypatch):
        """(e) Clean edits report zero rejections, not a missing field."""
        per_edit = iter([
            _per_edit_result(PY_MERGED, chunks_rejected=0),
            _per_edit_result(PY_MERGED, chunks_rejected=0),
        ])
        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", lambda *a, **kw: next(per_edit),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        result = batch_chunked_merge(
            original_code=PY_ORIGINAL,
            edits=[BatchEdit(snippet="x"), BatchEdit(snippet="y")],
            file_path=str(target),
            merge_fn=lambda *a, **kw: None,
            language="python",
        )

        assert result.chunks_rejected == 0, result

    def test_empty_edits_report_zero_rejections(self, tmp_path):
        """(e) The early-return path also carries the field."""
        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        result = batch_chunked_merge(
            original_code=PY_ORIGINAL,
            edits=[],
            file_path=str(target),
            merge_fn=lambda *a, **kw: None,
            language="python",
        )

        assert result.chunks_rejected == 0, result
        assert result.chunks_used == 0, result


# ---------------------------------------------------------------------------
# Schema: force must be documented in both tools' input schemas
# ---------------------------------------------------------------------------


def test_fast_batch_edit_input_schema_documents_force():
    """The registered input schema declares ``force`` as an optional boolean
    defaulting to False, with the intended-use description, so host LLMs can
    discover the escape hatch without writing it blind."""
    tools = asyncio.run(fastmcp_server.list_tools())
    batch_tool = next(t for t in tools if t.name == "fast_batch_edit")
    force = batch_tool.inputSchema["properties"]["force"]
    assert force.get("type") == "boolean"
    assert force.get("default") is False
    assert "parse check" in force.get("description", "")


def test_fast_multi_edit_input_schema_documents_force():
    """Same schema contract for fast_multi_edit."""
    tools = asyncio.run(fastmcp_server.list_tools())
    multi_tool = next(t for t in tools if t.name == "fast_multi_edit")
    force = multi_tool.inputSchema["properties"]["force"]
    assert force.get("type") == "boolean"
    assert force.get("default") is False
    assert "parse check" in force.get("description", "")
