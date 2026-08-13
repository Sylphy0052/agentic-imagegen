"""batchサブコマンドのCLI動作。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_imagegen import cli
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import GenerationFailed
from agentic_imagegen.services.batch import BatchItem, BatchOutcome

runner = CliRunner()

VALID_SPEC = """
version: "1"
task: txt2img
prompt:
  positive: 1girl, blue hair
generation:
  width: 512
  height: 512
  seed: 100
model:
  checkpoint: v1-5-pruned-emaonly.safetensors
output:
  prefix: batch_test
"""


def _write_spec(tmp_path: Path, name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(VALID_SPEC, encoding="utf-8")
    return path


def _result(tmp_path: Path, seed: int, *, with_text: bool = False) -> GenerationResult:
    return GenerationResult(
        prompt_id=f"p-{seed}",
        seed=seed,
        directory=tmp_path,
        files=(tmp_path / f"image_{seed}.png",),
        metadata_path=tmp_path / "metadata.json",
        text_files=(tmp_path / f"image_{seed}_text.png",) if with_text else (),
    )


def _patch_batch(
    monkeypatch: pytest.MonkeyPatch,
    outcomes_for: Any,
) -> list[list[BatchItem]]:
    """_run_batch を差し替え、渡されたitemsを記録する。"""
    captured: list[list[BatchItem]] = []

    async def fake_run_batch(items: list[BatchItem], settings: Any, timeout: Any) -> Any:
        captured.append(list(items))
        return outcomes_for(items)

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    return captured


def test_runs_each_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_a = _write_spec(tmp_path, "a.yaml")
    spec_b = _write_spec(tmp_path, "b.yaml")
    captured = _patch_batch(
        monkeypatch,
        lambda items: [
            BatchOutcome(item=item, result=_result(tmp_path, 100), error=None) for item in items
        ],
    )

    result = runner.invoke(cli.app, ["batch", str(spec_a), str(spec_b)])

    assert result.exit_code == 0
    assert len(captured[0]) == 2
    assert "成功 2 / 失敗 0" in result.output


def test_reports_composed_text_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """テキスト合成の結果もサマリへ出す。"""
    spec = _write_spec(tmp_path)
    _patch_batch(
        monkeypatch,
        lambda items: [
            BatchOutcome(item=item, result=_result(tmp_path, 100, with_text=True), error=None)
            for item in items
        ],
    )

    result = runner.invoke(cli.app, ["batch", str(spec)])

    assert result.exit_code == 0
    assert "image_100.png" in result.output
    assert "text: " in result.output
    assert "image_100_text.png" in result.output


def test_expands_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _write_spec(tmp_path)
    captured = _patch_batch(
        monkeypatch,
        lambda items: [
            BatchOutcome(item=item, result=_result(tmp_path, 1), error=None) for item in items
        ],
    )

    result = runner.invoke(cli.app, ["batch", str(spec), "--seeds", "1,2,3"])

    assert result.exit_code == 0
    assert [item.spec.generation.seed for item in captured[0]] == [1, 2, 3]
    assert "seed=1" in result.output


def test_reports_failure_and_returns_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1件失敗しても残りは実行し、最後にexit codeで知らせる。"""
    spec = _write_spec(tmp_path)

    def outcomes_for(items: list[BatchItem]) -> list[BatchOutcome]:
        return [
            BatchOutcome(item=items[0], result=_result(tmp_path, 1), error=None),
            BatchOutcome(item=items[1], result=None, error=GenerationFailed("失敗しました")),
        ]

    _patch_batch(monkeypatch, outcomes_for)

    result = runner.invoke(cli.app, ["batch", str(spec), "--seeds", "1,2"])

    assert result.exit_code == 7
    assert "FAILED (exit 7)" in result.output
    assert "成功 1 / 失敗 1" in result.output


def test_rejects_invalid_seeds(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path)

    result = runner.invoke(cli.app, ["batch", str(spec), "--seeds", "1,abc"])

    assert result.exit_code == 2
    assert "--seeds" in result.output


def test_validates_all_specs_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不正なSpecが混ざっていたら、1件も実行せずに落とす。"""
    good = _write_spec(tmp_path, "good.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(VALID_SPEC.replace("seed: 100", "seed: -5"), encoding="utf-8")
    captured = _patch_batch(monkeypatch, lambda items: [])

    result = runner.invoke(cli.app, ["batch", str(good), str(bad)])

    assert result.exit_code == 2
    assert captured == []
