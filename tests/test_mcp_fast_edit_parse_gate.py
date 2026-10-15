"""Step 8 (B10): MCP ``fast_edit`` must refuse to write parse-invalid output.

The bug: the CLI refuses to persist a merge that broke the parse
(``cli._refuse_if_edit_broke_parse`` — "merged output for {path} has parse
errors; refusing to write. The file is unchanged.") while the MCP
``fast_edit`` tool wrote the same garbage to disk with only a warning
("Warning: merged output has parse errors in {language}. Wrote to {file_path}
anyway."). That asymmetry is a primary cause of "corrupting the edited file
beyond saving" for MCP users.

Contract locked down here:

1. A ``fast_edit`` whose merged output fails the parse check does NOT write
   and returns an error in the CLI refusal's style, describing the parse
   failure and suggesting smaller edits. File content is unchanged on disk.
2. The escape hatch is explicit opt-in: ``force=True`` restores today's
   write-with-warning behavior, word for word.
3. Fail-loud refusals are not overridable: ``chunks_rejected >= chunks_used``
   still refuses even with ``force=True``.
4. No detected language is the Step-D2 SUPPORTED structureless path: it
   reaches the merge with ``language=None`` and the parse gate stays inert.

The unit under test is the tool's WRITE GATE, not the merge itself, so
``chunked_merge`` is stubbed at the ``tools_edit`` module boundary with a
``ChunkedMergeResult`` carrying the parse validity under test. This keeps
the tests hermetic: no model, no backend, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict

from fastedit.inference.ast_utils import ChunkedMergeResult
from fastedit.mcp import tools_edit
from fastedit.mcp.backup import BackupStore
from fastedit.mcp.server import mcp as fastmcp_server

# A valid Python file, and a parse-invalid merged output for it (the kind of
# thing a hallucinating merge produces). The stub hands back the broken text
# directly — how chunked_merge produced it is out of scope for this gate.
PY_ORIGINAL = "def existing():\n    return 1\n"
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
    """Engine handed out by the fake backend. Never used: chunked_merge is
    stubbed in every test, so merge_fn is never invoked."""

    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError("merge_fn must not be called; chunked_merge is stubbed")


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
        # Never loads a model: chunked_merge is stubbed in every test, so
        # acquire() yields an engine whose merge_auto is never invoked.
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_edit, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


def _stub_merge_result(monkeypatch, result: ChunkedMergeResult):
    """Replace ``tools_edit.chunked_merge`` with a stub returning *result*.

    Both call paths inside the tool (``asyncio.to_thread`` and the direct
    zero-model call) resolve the name from module globals at call time, so
    the stub covers each of them.
    """
    monkeypatch.setattr(tools_edit, "chunked_merge", lambda *a, **kw: result)


def _result(
    merged_code: str,
    parse_valid: bool,
    chunks_used: int = 1,
    chunks_rejected: int = 0,
    retries: int = 0,
) -> ChunkedMergeResult:
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=chunks_used,
        chunk_regions=[],
        model_tokens=12,
        latency_ms=40.0,
        chunks_rejected=chunks_rejected,
        retries=retries,
    )


def _call(tool_coroutine):
    """Run one tool invocation to completion."""
    return asyncio.run(tool_coroutine)


# ---------------------------------------------------------------------------
# B10: the parse gate
# ---------------------------------------------------------------------------


class TestFastEditParseGate:
    def test_parse_invalid_merge_refuses_and_leaves_file_untouched(
        self, tmp_path, monkeypatch,
    ):
        """(a) Parse-invalid merged output: no write, CLI-style error."""
        _install_fake_mcp(monkeypatch)
        _stub_merge_result(monkeypatch, _result(PY_BROKEN, parse_valid=False))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=PY_BROKEN, replace="existing",
        ))

        assert message.startswith("Error"), message
        # Describes the parse failure, in the CLI refusal's voice.
        assert "parse errors" in message, message
        assert "refusing to write" in message, message
        assert "unchanged" in message, message
        # Actionable: suggests smaller edits (and names the force escape hatch).
        assert "smaller edits" in message, message
        assert "force=True" in message, message
        # Disk is untouched.
        assert target.read_text() == PY_ORIGINAL

    def test_parse_invalid_merge_force_true_writes_with_warning(
        self, tmp_path, monkeypatch,
    ):
        """(b) force=True restores the old escape hatch: write + warning."""
        _install_fake_mcp(monkeypatch)
        _stub_merge_result(monkeypatch, _result(PY_BROKEN, parse_valid=False))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=PY_BROKEN, replace="existing",
            force=True,
        ))

        assert message.startswith("Warning"), message
        assert "parse errors in python" in message, message
        assert "Wrote to" in message, message
        assert "anyway" in message, message
        # The (parse-invalid) merged output IS on disk — the caller opted in.
        assert target.read_text() == PY_BROKEN

    def test_chunks_rejected_refusal_not_overridable_by_force(
        self, tmp_path, monkeypatch,
    ):
        """(c) chunks_rejected >= chunks_used refuses with AND without force.

        Fail-loud refusals are not escape hatches: force only covers the
        parse gate, never the hallucination refusal.
        """
        _install_fake_mcp(monkeypatch)
        _stub_merge_result(monkeypatch, _result(
            PY_BROKEN, parse_valid=True, chunks_used=2, chunks_rejected=2,
        ))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        without_force = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=PY_BROKEN, replace="existing",
        ))
        assert without_force.startswith("Error"), without_force
        assert "reject" in without_force, without_force
        assert target.read_text() == PY_ORIGINAL

        with_force = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet=PY_BROKEN, replace="existing",
            force=True,
        ))
        assert with_force.startswith("Error"), with_force
        assert "reject" in with_force, with_force
        assert target.read_text() == PY_ORIGINAL

    def test_partial_rejection_still_writes_with_warning_regardless_of_force(
        self, tmp_path, monkeypatch,
    ):
        """Partial rejection (rejected < used) keeps its write-with-warning
        behavior — unchanged by the new parse gate, with or without force."""
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 42\n"
        _stub_merge_result(monkeypatch, _result(
            merged, parse_valid=True, chunks_used=3, chunks_rejected=1,
        ))

        for force in (False, True):
            target = tmp_path / f"mod_{force}.py"
            target.write_text(PY_ORIGINAL)
            message = _call(tools_edit.fast_edit(
                file_path=str(target), edit_snippet="x", replace="existing",
                force=force,
            ))
            assert message.startswith("Warning"), message
            assert "rejected" in message, message
            assert "Partial edit applied" in message, message
            assert target.read_text() == merged

    def test_structureless_txt_reaches_the_merge_and_writes(self, tmp_path,
                                                            monkeypatch):
        """(d) Step D2: no detected language is the SUPPORTED AST-less path.

        The pre-D2 extension allowlist refused ``.txt`` at the door; since
        the structureless pipeline exists (D1 trait battery + D2 text
        anchor windows), ``language=None`` rides into ``chunked_merge`` and
        an accepted merge is written — the parse gate stays inert (there
        is no grammar to parse, so ``if language and not parse_valid``
        never fires; the pipeline's own battery is the gate).
        """
        _install_fake_mcp(monkeypatch)
        captured: dict = {}

        def _capture_merge(**kwargs):
            captured.update(kwargs)
            return _result("plain text, now edited.\n", parse_valid=True)

        monkeypatch.setattr(tools_edit, "chunked_merge", _capture_merge)

        target = tmp_path / "notes.txt"
        target.write_text("plain text\n")

        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="whatever",
        ))

        assert message.startswith("Applied edit to"), message
        assert target.read_text() == "plain text, now edited.\n"
        # The structureless file reached the pipeline with language=None —
        # the D2 text-anchor path's precondition.
        assert captured["language"] is None

    def test_parse_valid_edit_still_writes_and_reports_applied(
        self, tmp_path, monkeypatch,
    ):
        """Guard: a clean merge is unaffected — written, 'Applied edit'."""
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        _stub_merge_result(monkeypatch, _result(merged, parse_valid=True))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)

        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="def existing(): ...",
        ))

        assert message.startswith("Applied edit to"), message
        assert target.read_text() == merged


# ---------------------------------------------------------------------------
# Schema: force must be documented in the tool's input schema
# ---------------------------------------------------------------------------


def test_fast_edit_input_schema_documents_force():
    """The registered input schema declares ``force`` as an optional boolean
    defaulting to False, with the intended-use description, so host LLMs can
    discover the escape hatch without writing it blind."""
    tools = asyncio.run(fastmcp_server.list_tools())
    fast_edit_tool = next(t for t in tools if t.name == "fast_edit")
    force = fast_edit_tool.inputSchema["properties"]["force"]
    assert force.get("type") == "boolean"
    assert force.get("default") is False
    assert "parse check" in force.get("description", "")


# ---------------------------------------------------------------------------
# Step A3: validation retries surfaced in the metrics segment
# ---------------------------------------------------------------------------


class TestRetryMetricsSuffix:
    """When a merge consumed validation retries, the metrics segment of the
    success/warning message says so (`, N validation retries`); with no
    retries the message shape is byte-stable with pre-A3 output."""

    def test_fast_edit_reports_retries(self, tmp_path, monkeypatch):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        _stub_merge_result(monkeypatch, _result(merged, parse_valid=True, retries=3))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="x", replace="existing",
        ))

        assert message.startswith("Applied edit to"), message
        assert ", 3 validation retries" in message, message

    def test_fast_edit_reports_a_single_retry_in_the_singular(self, tmp_path, monkeypatch):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        _stub_merge_result(monkeypatch, _result(merged, parse_valid=True, retries=1))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="x", replace="existing",
        ))

        assert ", 1 validation retry" in message, message
        assert ", 1 validation retries" not in message, message

    def test_fast_edit_zero_retries_keeps_the_shape_stable(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        _stub_merge_result(monkeypatch, _result(merged, parse_valid=True, retries=0))

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_edit(
            file_path=str(target), edit_snippet="x", replace="existing",
        ))

        assert message.startswith("Applied edit to"), message
        assert "validation retr" not in message, message

    def test_fast_batch_edit_reports_retries(self, tmp_path, monkeypatch):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        monkeypatch.setattr(
            tools_edit, "batch_chunked_merge",
            lambda *a, **kw: _result(merged, parse_valid=True, retries=2),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target),
            edits='[{"snippet": "x", "replace": "existing"}]',
        ))

        assert message.startswith("Applied 1 edits to"), message
        assert ", 2 validation retries" in message, message

    def test_fast_batch_edit_zero_retries_keeps_the_shape_stable(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        monkeypatch.setattr(
            tools_edit, "batch_chunked_merge",
            lambda *a, **kw: _result(merged, parse_valid=True, retries=0),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_batch_edit(
            file_path=str(target),
            edits='[{"snippet": "x", "replace": "existing"}]',
        ))

        assert message.startswith("Applied 1 edits to"), message
        assert "validation retr" not in message, message

    def test_fast_multi_edit_aggregates_retries_in_the_summary(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        monkeypatch.setattr(
            tools_edit, "batch_chunked_merge",
            lambda *a, **kw: _result(merged, parse_valid=True, retries=2),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_multi_edit(
            file_edits=(
                f'[{{"file_path": "{target}", '
                f'"edits": [{{"snippet": "x", "replace": "existing"}}]}}]'
            ),
        ))

        assert "ok" in message, message
        assert ", 2 validation retries" in message, message

    def test_fast_multi_edit_zero_retries_keeps_the_shape_stable(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        merged = "def existing():\n    return 2\n"
        monkeypatch.setattr(
            tools_edit, "batch_chunked_merge",
            lambda *a, **kw: _result(merged, parse_valid=True, retries=0),
        )

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        message = _call(tools_edit.fast_multi_edit(
            file_edits=(
                f'[{{"file_path": "{target}", '
                f'"edits": [{{"snippet": "x", "replace": "existing"}}]}}]'
            ),
        ))

        assert "ok" in message, message
        assert "validation retr" not in message, message
