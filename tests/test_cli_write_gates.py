"""CLI write gates — Step 18 (B34) parity with the MCP edit tools.

The bug (issue #2, the CLI's model-path tail): on chunk retry-exhaustion
the merge pipeline returns the rejection convention — ``merged_code``
keeps the ORIGINAL file and ``parse_valid=False`` — so
``cli._refuse_if_edit_broke_parse`` cannot refuse (the original parses
like the original, by definition). The MCP tools refuse via
``_all_chunks_rejected``/``_rejection_refusal`` BEFORE persisting, but the
CLI's edit tail treated ``chunks_rejected`` as a truthiness warning
("Warning: 1/1 chunk(s) rejected. Partial edit applied.") and fell
through to "Applied edit to ..." + exit 0 for a file that was never
edited. ``cmd_batch_edit`` had the same warn-and-fall-through tail and
``cmd_multi_edit`` PHASE 3 had no chunks_rejected check at all.

Contract locked down here:

1. ``cmd_edit``: an all-chunks-rejected merge (rejected >= used) prints
   the MCP refusal to stderr and exits 1 BEFORE the atomic write — the
   file on disk is byte-for-byte what the caller had, and stdout never
   says "Applied edit".
2. ``cmd_edit``: partial rejection (0 < rejected < used) keeps its
   existing correct behavior — the merged output writes and the tail
   prints "Warning: N/M chunk(s) rejected. Partial edit applied."
3. ``cmd_batch_edit``: the same gate — a fully-rejected merge exits 1
   with the file unchanged; a partially-rejected one still writes.
4. ``cmd_multi_edit`` PHASE 3: per-target gate — a fully-rejected target
   is not written and is reported rejected, the remaining targets still
   write, and the command exits 1 when any target was refused.
5. Message parity: the CLI's refusal line is byte-identical to the MCP
   tool's ``_rejection_refusal`` for the same merge result, and
   ``fastedit.mcp.tools_edit`` re-imports both gate helpers from the
   shared ``fastedit.write_gates`` module.

The unit under test is the CLI WRITE TAIL, so the merge functions are
stubbed at the ``fastedit.inference.chunked_merge`` module boundary (the
tests/test_cli_retry_metrics.py harness pattern). Hermetic: no model, no
backend, no network.
"""

from __future__ import annotations

import argparse
import json

import pytest

from fastedit.inference.ast_utils import ChunkedMergeResult
from fastedit.inference.chunked_merge import _validation_retries_metric

PY_ORIGINAL = "def existing():\n    return 1\n"
PY_MERGED = "def existing():\n    return 2\n"

# The rejection convention chunked_merge returns on retry exhaustion
# (diagnosis repro: retries=8, chunks_rejected=1, chunks_used=1,
# merged_code == the original, parse_valid forced False).
REJECTED_RESULT_KWARGS = {
    "merged_code": PY_ORIGINAL,
    "parse_valid": False,
    "chunks_used": 1,
    "chunks_rejected": 1,
    "retries": 8,
}


def _result(
    merged_code: str = PY_MERGED,
    parse_valid: bool = True,
    chunks_used: int = 1,
    chunks_rejected: int = 0,
    retries: int = 0,
) -> ChunkedMergeResult:
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=chunks_used,
        chunk_regions=[(1, 1)] * max(chunks_used, 1),
        model_tokens=12,
        latency_ms=40.0,
        chunks_rejected=chunks_rejected,
        retries=retries,
    )


def _cli_args(**overrides) -> argparse.Namespace:
    """The argparse namespace cmd_edit/cmd_batch_edit/cmd_multi_edit read."""
    base = {
        "backend": None, "model_path": None, "api_base": None, "api_model": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")


class _FakeBackend:
    """Stands in for the MLX/vLLM backend; merge_auto is never reached."""

    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError("batch_chunked_merge is stubbed")


# ---------------------------------------------------------------------------
# (a) + (b) cmd_edit: the gate sits between the parse gate and the write
# ---------------------------------------------------------------------------


class TestCmdEditRejectionGate:
    def _run_with(self, tmp_path, monkeypatch, result):
        """Stub chunked_merge to hand back *result* for a real .py target."""
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_edit

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", lambda *a, **kw: result,
        )
        return target, cmd_edit

    def test_all_chunks_rejected_refuses_before_write(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(a) rejected >= used: exit 1, MCP refusal on stderr, NO write."""
        target, cmd_edit = self._run_with(
            tmp_path, monkeypatch, _result(**REJECTED_RESULT_KWARGS),
        )

        with pytest.raises(SystemExit) as excinfo:
            cmd_edit(_cli_args(
                file=str(target), snippet="x", replace="", after="existing",
            ))

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "edit rejected" in captured.err, captured.err
        assert "model hallucinated on 1 chunk(s)" in captured.err, captured.err
        assert "File unchanged" in captured.err, captured.err
        # The file was never written: byte-for-byte what the caller had.
        assert target.read_text() == PY_ORIGINAL
        # No success line, and none of the old misleading tail output.
        assert "Applied edit" not in captured.out, captured.out
        assert "Partial edit applied" not in captured.out, captured.out
        assert "Warning" not in captured.out, captured.out

    def test_refusal_line_is_byte_identical_to_the_mcp_wording(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(e) Same merge result → the exact string the MCP fast_edit tool
        returns from its ``_rejection_refusal`` gate."""
        from fastedit.mcp import tools_edit

        result = _result(**REJECTED_RESULT_KWARGS)
        target, cmd_edit = self._run_with(tmp_path, monkeypatch, result)

        with pytest.raises(SystemExit):
            cmd_edit(_cli_args(
                file=str(target), snippet="x", replace="", after="existing",
            ))

        # The metrics segment cmd_edit builds for this result: 40ms, 12
        # tokens, 1 chunk (no plural suffix), 8 retries.
        metrics = (
            "latency: 40ms, 300 tok/s, 12 tokens"
            f"{_validation_retries_metric(result.retries)}"
        )
        expected = tools_edit._rejection_refusal(result, metrics)
        err = capsys.readouterr().err.strip()
        # The CLI prints the shared refusal AS-IS: the refusal already opens
        # with "Error: " (the verbatim MCP wording), so composing another
        # prefix doubled it ("Error: Error: ..."). Pin the refusal exactly
        # once — byte-identical to the MCP tool's wording, single prefix.
        assert err == expected
        assert err.count("Error: ") == 1, err
        # The MCP wording itself is embedded verbatim.
        assert (
            "edit rejected — model hallucinated on 1 chunk(s). "
            "File unchanged. The function may be too large (1 chunk(s)) for "
            "the 1.7B model. Try a smaller edit or split the function."
        ) in err

    def test_partial_rejection_still_writes_with_warning(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(b) 0 < rejected < used keeps today's correct behavior: the
        merged output writes with the partial-edit warning. The new gate
        must not over-refuse partial rejections."""
        target, cmd_edit = self._run_with(
            tmp_path, monkeypatch,
            _result(merged_code=PY_MERGED, chunks_used=2, chunks_rejected=1),
        )

        cmd_edit(_cli_args(
            file=str(target), snippet="x", replace="", after="existing",
        ))

        captured = capsys.readouterr()
        assert "Applied edit to" in captured.out, captured.out
        assert "Warning: 1/2 chunk(s) rejected" in captured.out, captured.out
        assert "Partial edit applied" in captured.out, captured.out
        assert "edit rejected" not in captured.err, captured.err
        assert target.read_text() == PY_MERGED

    def test_zero_model_result_is_not_gated(
        self, tmp_path, monkeypatch, capsys,
    ):
        """A zero-chunk merge (pure after= splice reports chunks_used == 0)
        must not read as "everything rejected" via 0 >= 0."""
        target, cmd_edit = self._run_with(
            tmp_path, monkeypatch, _result(merged_code=PY_MERGED, chunks_used=0),
        )

        cmd_edit(_cli_args(
            file=str(target), snippet="x", replace="", after="existing",
        ))

        captured = capsys.readouterr()
        assert "Applied edit to" in captured.out, captured.out
        assert "edit rejected" not in captured.err, captured.err
        assert target.read_text() == PY_MERGED

    def test_clean_result_still_applies(self, tmp_path, monkeypatch, capsys):
        """Guard: chunks_rejected == 0 is unaffected — written, Applied."""
        target, cmd_edit = self._run_with(
            tmp_path, monkeypatch, _result(merged_code=PY_MERGED),
        )

        cmd_edit(_cli_args(
            file=str(target), snippet="x", replace="", after="existing",
        ))

        captured = capsys.readouterr()
        assert "Applied edit to" in captured.out, captured.out
        assert "Warning" not in captured.out, captured.out
        assert target.read_text() == PY_MERGED


# ---------------------------------------------------------------------------
# (c) cmd_batch_edit: the same gate on the batch tail
# ---------------------------------------------------------------------------


class TestCmdBatchEditRejectionGate:
    def _run_with(self, tmp_path, monkeypatch, result):
        """Stub batch_chunked_merge + the backend for a real .py target."""
        import fastedit.cli as cli_module
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_batch_edit

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        monkeypatch.setattr(
            chunked_merge_module, "batch_chunked_merge", lambda *a, **kw: result,
        )
        monkeypatch.setattr(
            cli_module, "_make_backend_with_overrides",
            lambda args: ("mlx", _FakeBackend()),
        )
        return target, cmd_batch_edit

    def test_all_chunks_rejected_refuses_before_write(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(c) rejected >= used: exit 1, MCP refusal on stderr, NO write."""
        target, cmd_batch_edit = self._run_with(
            tmp_path, monkeypatch,
            _result(
                merged_code=PY_ORIGINAL, parse_valid=False,
                chunks_used=2, chunks_rejected=2, retries=8,
            ),
        )

        with pytest.raises(SystemExit) as excinfo:
            cmd_batch_edit(_cli_args(
                file=str(target),
                edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
            ))

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "edit rejected" in captured.err, captured.err
        assert "model hallucinated on 2 chunk(s)" in captured.err, captured.err
        assert "File unchanged" in captured.err, captured.err
        assert target.read_text() == PY_ORIGINAL
        assert "Applied" not in captured.out, captured.out

    def test_partial_rejection_still_writes(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Partial rejection keeps batch-edit's existing behavior: the
        merged output writes and the Applied line prints (no gate fires)."""
        target, cmd_batch_edit = self._run_with(
            tmp_path, monkeypatch,
            _result(merged_code=PY_MERGED, chunks_used=2, chunks_rejected=1),
        )

        cmd_batch_edit(_cli_args(
            file=str(target),
            edits=json.dumps([{"snippet": "x"}, {"snippet": "y"}]),
        ))

        captured = capsys.readouterr()
        assert "Applied 2 edits to" in captured.out, captured.out
        assert "edit rejected" not in captured.err, captured.err
        assert target.read_text() == PY_MERGED


# ---------------------------------------------------------------------------
# (d) cmd_multi_edit: per-target gate in PHASE 3
# ---------------------------------------------------------------------------


class TestCmdMultiEditRejectionGate:
    def _install_stub(self, monkeypatch, results_by_path):
        import fastedit.cli as cli_module
        import fastedit.inference.chunked_merge as chunked_merge_module

        def fake_batch(*args, **kw):
            return results_by_path[kw["file_path"]]

        monkeypatch.setattr(chunked_merge_module, "batch_chunked_merge", fake_batch)
        monkeypatch.setattr(
            cli_module, "_make_backend_with_overrides",
            lambda args: ("mlx", _FakeBackend()),
        )

    def _write_targets(self, tmp_path):
        target_a = tmp_path / "a.py"
        target_a.write_text(PY_ORIGINAL)
        target_b = tmp_path / "b.py"
        target_b.write_text(PY_ORIGINAL)
        return target_a, target_b

    def _file_edits(self, target_a, target_b) -> str:
        return json.dumps([
            {"file_path": str(target_a), "edits": [{"snippet": "x"}, {"snippet": "y"}]},
            {"file_path": str(target_b), "edits": [{"snippet": "x"}]},
        ])

    def test_fully_rejected_target_not_written_remaining_targets_write(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(d) Per-target gate: the fully-rejected target is not written and
        is reported rejected; the clean target still writes; the command
        exits 1 so the caller knows the run did not fully land."""
        from fastedit.cli import cmd_multi_edit

        target_a, target_b = self._write_targets(tmp_path)
        self._install_stub(monkeypatch, {
            str(target_a): _result(
                merged_code=PY_ORIGINAL, parse_valid=False,
                chunks_used=2, chunks_rejected=2, retries=8,
            ),
            str(target_b): _result(merged_code=PY_MERGED),
        })

        with pytest.raises(SystemExit) as excinfo:
            cmd_multi_edit(_cli_args(
                file_edits=self._file_edits(target_a, target_b),
            ))

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "rejected" in captured.err, captured.err
        assert "model hallucinated on 2 chunk(s)" in captured.err, captured.err
        # The rejected target was NOT written...
        assert target_a.read_text() == PY_ORIGINAL
        # ...while the clean target still was.
        assert target_b.read_text() == PY_MERGED
        # The Applied line belongs to the written target only.
        assert f"Applied 1 edits to {target_b}" in captured.out, captured.out
        assert f"Applied 2 edits to {target_a}" not in captured.out, captured.out

    def test_all_targets_rejected_nothing_written(
        self, tmp_path, monkeypatch, capsys,
    ):
        """(d) Every target fully rejected: nothing is written, each target
        is reported, exit 1."""
        from fastedit.cli import cmd_multi_edit

        target_a, target_b = self._write_targets(tmp_path)
        rejected = _result(
            merged_code=PY_ORIGINAL, parse_valid=False,
            chunks_used=2, chunks_rejected=2, retries=8,
        )
        self._install_stub(monkeypatch, {
            str(target_a): rejected,
            str(target_b): rejected,
        })

        with pytest.raises(SystemExit) as excinfo:
            cmd_multi_edit(_cli_args(
                file_edits=self._file_edits(target_a, target_b),
            ))

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert target_a.read_text() == PY_ORIGINAL
        assert target_b.read_text() == PY_ORIGINAL
        assert "Applied" not in captured.out, captured.out
        assert "hallucinated" in captured.err, captured.err

    def test_clean_multi_run_is_unaffected(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Guard: all-clean runs keep the exact existing behavior — every
        target writes, every Applied line prints, no exit, no error."""
        from fastedit.cli import cmd_multi_edit

        target_a, target_b = self._write_targets(tmp_path)
        self._install_stub(monkeypatch, {
            str(target_a): _result(merged_code=PY_MERGED),
            str(target_b): _result(merged_code=PY_MERGED),
        })

        cmd_multi_edit(_cli_args(file_edits=self._file_edits(target_a, target_b)))

        captured = capsys.readouterr()
        assert target_a.read_text() == PY_MERGED
        assert target_b.read_text() == PY_MERGED
        assert f"Applied 2 edits to {target_a}" in captured.out, captured.out
        assert f"Applied 1 edits to {target_b}" in captured.out, captured.out
        assert "rejected" not in captured.err, captured.err


# ---------------------------------------------------------------------------
# (e) the shared gate module and its backward-compatible re-import
# ---------------------------------------------------------------------------


class TestSharedGateModule:
    def test_tools_edit_reimports_the_shared_gate(self):
        """Backward compat: the MCP module keeps both names, now imported
        from the shared fastedit.write_gates module (single source)."""
        from fastedit import write_gates
        from fastedit.mcp import tools_edit

        assert tools_edit._all_chunks_rejected is write_gates._all_chunks_rejected
        assert tools_edit._rejection_refusal is write_gates._rejection_refusal

    def test_gate_predicate_matrix(self):
        """rejected > 0 AND rejected >= used — the zero-model batch
        (chunks_used == 0) must never read as "all rejected"."""
        from fastedit import write_gates

        assert write_gates._all_chunks_rejected(
            _result(chunks_used=2, chunks_rejected=2),
        ) is True
        assert write_gates._all_chunks_rejected(
            _result(chunks_used=1, chunks_rejected=2),
        ) is True
        assert write_gates._all_chunks_rejected(
            _result(chunks_used=3, chunks_rejected=1),
        ) is False
        assert write_gates._all_chunks_rejected(
            _result(chunks_used=0, chunks_rejected=0),
        ) is False
        assert write_gates._all_chunks_rejected(
            _result(chunks_used=1, chunks_rejected=0),
        ) is False
