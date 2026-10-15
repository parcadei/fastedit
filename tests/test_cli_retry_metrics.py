"""Step A3 — the CLI surfaces validation retries in its result messages.

``ChunkedMergeResult.retries`` (Step A2) counts the validation-retry
attempts the unified retry-until-valid loop consumed. The budget itself is
resolved inside ``chunked_merge``/``batch_chunked_merge`` (``FASTEDIT_MAX_RETRIES``
env var or the default of 8) — the CLI only SURFACES the count, appended to
its metrics segment as ``, N validation retries`` when retries > 0 and
absent otherwise (message shapes stay byte-stable).

Locked down here for ``cmd_edit``, ``cmd_batch_edit`` and ``cmd_multi_edit``.
The merge functions are stubbed at the ``fastedit.inference.chunked_merge``
module boundary — how the retries were consumed is the Step A2 loop's
business (covered by tests/test_relative_validation.py), this pins only the
message surface. No model, no network.
"""

from __future__ import annotations

import argparse
import json

import pytest

from fastedit.inference.ast_utils import ChunkedMergeResult

PY_ORIGINAL = "def existing():\n    return 1\n"
PY_MERGED = "def existing():\n    return 2\n"


def _result(retries: int = 0) -> ChunkedMergeResult:
    return ChunkedMergeResult(
        merged_code=PY_MERGED,
        parse_valid=True,
        chunks_used=1,
        chunk_regions=[(1, 1)],
        model_tokens=12,
        latency_ms=40.0,
        retries=retries,
    )


class _FakeBackend:
    """Stands in for the MLX/vLLM backend; merge_auto is never reached."""

    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError("batch_chunked_merge is stubbed")


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


class TestCmdEditRetrySuffix:
    def test_retries_appended_to_the_metrics_segment(
        self, tmp_path, monkeypatch, capsys,
    ):
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_edit

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", lambda *a, **kw: _result(3),
        )

        cmd_edit(_cli_args(
            file=str(target), snippet="x", replace="", after="existing",
        ))

        out = capsys.readouterr().out
        assert out.startswith("Applied edit to"), out
        assert ", 3 validation retries" in out, out

    def test_zero_retries_keeps_the_shape_stable(
        self, tmp_path, monkeypatch, capsys,
    ):
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_edit

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        monkeypatch.setattr(
            chunked_merge_module, "chunked_merge", lambda *a, **kw: _result(0),
        )

        cmd_edit(_cli_args(
            file=str(target), snippet="x", replace="", after="existing",
        ))

        out = capsys.readouterr().out
        assert out.startswith("Applied edit to"), out
        assert "validation retr" not in out, out


class TestCmdBatchEditRetrySuffix:
    def _run(self, tmp_path, monkeypatch, capsys, retries):
        import fastedit.cli as cli_module
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_batch_edit

        target = tmp_path / "mod.py"
        target.write_text(PY_ORIGINAL)
        monkeypatch.setattr(
            chunked_merge_module, "batch_chunked_merge",
            lambda *a, **kw: _result(retries),
        )
        monkeypatch.setattr(
            cli_module, "_make_backend_with_overrides",
            lambda args: ("mlx", _FakeBackend()),
        )

        cmd_batch_edit(_cli_args(
            file=str(target),
            edits=json.dumps([{"snippet": "x", "replace": "existing"}]),
        ))
        return capsys.readouterr().out

    def test_retries_appended_to_the_metrics_segment(self, tmp_path, monkeypatch, capsys):
        out = self._run(tmp_path, monkeypatch, capsys, retries=2)
        assert out.startswith("Applied 1 edits to"), out
        assert ", 2 validation retries" in out, out

    def test_zero_retries_keeps_the_shape_stable(self, tmp_path, monkeypatch, capsys):
        out = self._run(tmp_path, monkeypatch, capsys, retries=0)
        assert out.startswith("Applied 1 edits to"), out
        assert "validation retr" not in out, out


class TestCmdMultiEditRetrySuffix:
    def _run(self, tmp_path, monkeypatch, capsys, retries_by_file):
        import fastedit.cli as cli_module
        import fastedit.inference.chunked_merge as chunked_merge_module
        from fastedit.cli import cmd_multi_edit

        files = {}
        for name in ("a.py", "b.py"):
            target = tmp_path / name
            target.write_text(PY_ORIGINAL)
            files[str(target)] = name

        def fake_batch(*args, **kw):
            return _result(retries_by_file[kw["file_path"]])

        monkeypatch.setattr(
            chunked_merge_module, "batch_chunked_merge", fake_batch,
        )
        monkeypatch.setattr(
            cli_module, "_make_backend_with_overrides",
            lambda args: ("mlx", _FakeBackend()),
        )

        file_edits = [
            {"file_path": str(tmp_path / name), "edits": [{"snippet": "x"}]}
            for name in ("a.py", "b.py")
        ]
        cmd_multi_edit(_cli_args(file_edits=json.dumps(file_edits)))
        return capsys.readouterr().out, files

    def test_each_file_reports_its_own_retry_count(
        self, tmp_path, monkeypatch, capsys,
    ):
        out, files = self._run(
            tmp_path, monkeypatch, capsys,
            retries_by_file={str(tmp_path / "a.py"): 2, str(tmp_path / "b.py"): 0},
        )
        a_line = next(ln for ln in out.splitlines() if files[str(tmp_path / "a.py")] in ln)
        b_line = next(ln for ln in out.splitlines() if files[str(tmp_path / "b.py")] in ln)
        assert ", 2 validation retries" in a_line, a_line
        assert "validation retr" not in b_line, b_line

    def test_zero_retries_keeps_the_shape_stable(self, tmp_path, monkeypatch, capsys):
        out, _files = self._run(
            tmp_path, monkeypatch, capsys,
            retries_by_file={str(tmp_path / "a.py"): 0, str(tmp_path / "b.py"): 0},
        )
        assert "validation retr" not in out, out
