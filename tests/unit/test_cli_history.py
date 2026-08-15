"""`imagegen history` の出力のテスト。

収集の規則は tests/unit/test_history.py が扱う。ここは表示だけを見る。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_imagegen import cli
from agentic_imagegen.domain.results import RunRecord

runner = CliRunner()


def _record(tmp_path: Path, **overrides: Any) -> RunRecord:
    defaults: dict[str, Any] = {
        "directory": tmp_path / "2026-08-15" / "105506_yui_ref",
        "created_at": "2026-08-15T11:00:25+09:00",
        "task": "img2img",
        "model": "hassakuSD15_v13.safetensors",
        "presets": {"style": "sd15-hassaku"},
        "seed": 271828182,
        "width": 512,
        "height": 768,
        "source": None,
        "upscale": 2.0,
        "features": ("reference",),
        "files": (tmp_path / "2026-08-15" / "105506_yui_ref" / "image_0001.png",),
        "workflow": "img2img_vae_hires_model",
    }
    defaults.update(overrides)
    return RunRecord(**defaults)


@pytest.fixture
def stub_collect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_collect(root: Path, **kwargs: Any) -> tuple[RunRecord, ...]:
        seen["root"] = root
        seen.update(kwargs)
        return tuple(seen.get("records", (_record(tmp_path),)))

    monkeypatch.setattr(cli, "collect_history", fake_collect)
    return seen


class TestHistoryCommand:
    def test_shows_time_model_seed_and_path(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["history"])

        assert result.exit_code == 0
        assert "2026-08-15 11:00" in result.stdout
        assert "hassakuSD15_v13.safetensors" in result.stdout
        assert "seed 271828182" in result.stdout
        assert "image_0001.png" in result.stdout

    def test_shows_presets_and_features(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["history"])

        assert "style:sd15-hassaku" in result.stdout
        assert "reference" in result.stdout
        assert "x2.0" in result.stdout

    def test_img2img_shows_the_source_instead_of_the_resolution(
        self, stub_collect: dict[str, Any], tmp_path: Path
    ) -> None:
        """img2imgのサイズは入力画像で決まる。Specの width/height を出すと嘘になる。"""
        stub_collect["records"] = (_record(tmp_path, source="inputs/yui-ref-f.png"),)

        result = runner.invoke(cli.app, ["history"])

        assert "<- inputs/yui-ref-f.png" in result.stdout
        assert "512x768" not in result.stdout

    def test_passes_limit_and_prefix(self, stub_collect: dict[str, Any]) -> None:
        runner.invoke(cli.app, ["history", "--limit", "3", "--prefix", "yui"])

        assert stub_collect["limit"] == 3
        assert stub_collect["prefix"] == "yui"

    def test_empty_history_is_not_an_error(self, stub_collect: dict[str, Any]) -> None:
        """まだ1枚も生成していない状態でも落ちない。"""
        stub_collect["records"] = ()

        result = runner.invoke(cli.app, ["history"])

        assert result.exit_code == 0
        assert "(なし)" in result.stdout

    def test_json_output_is_machine_readable(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["history", "--json"])

        payload = json.loads(result.stdout)
        assert payload[0]["seed"] == 271828182
        assert payload[0]["presets"]["style"] == "sd15-hassaku"
        assert payload[0]["files"][0].endswith("image_0001.png")

    def test_passes_absolute_output_root(self, stub_collect: dict[str, Any]) -> None:
        runner.invoke(cli.app, ["history"])

        assert Path(stub_collect["root"]).is_absolute()
