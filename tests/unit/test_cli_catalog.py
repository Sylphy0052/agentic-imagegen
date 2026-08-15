"""`imagegen catalog` の出力とフォールバックのテスト。

実ComfyUIへは接続せず、`cli.collect_catalog` を差し替えて表示だけを見る。
収集そのものの規則は tests/unit/test_catalog_collect.py が扱う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_imagegen import cli
from agentic_imagegen.domain.results import CatalogSnapshot, HealthStatus

runner = CliRunner()

MODELS = {
    "checkpoints": ("hassakuSD15_v13.safetensors", "meinamix_v12Final.safetensors"),
    "loras": (),
    "controlnets": (),
    "ipadapters": (),
    "clip_visions": (),
    "diffusion_models": (),
    "text_encoders": (),
    "vaes": ("vaeKlF8Anime2_klF8Anime2VAE.safetensors",),
    "upscale_models": (),
    "embeddings": ("negativeXL_D",),
}
PRESETS = {"character": ("kaede",), "scene": (), "style": ("sd15-hassaku",)}


def _snapshot(source: str) -> CatalogSnapshot:
    return CatalogSnapshot(
        source=source,  # type: ignore[arg-type]
        models=MODELS,
        presets=PRESETS,
        fonts=("NotoSansJP.ttf",),
    )


@pytest.fixture
def stub_collect(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """collect_catalog と health 問い合わせを差し替え、渡された引数を記録する。

    既定はComfyUI未起動 (health は None、取得元は filesystem)。
    起動している場合は `seen["source"]` / `seen["status"]` を先に入れて切り替える。
    """
    seen: dict[str, Any] = {}

    async def fake_collect(settings: Any, **kwargs: Any) -> CatalogSnapshot:
        seen.update(kwargs)
        seen["settings"] = settings
        return _snapshot(seen.pop("source", "filesystem"))

    def fake_probe(settings: Any) -> HealthStatus | None:
        return seen.pop("status", None)

    monkeypatch.setattr(cli, "collect_catalog", fake_collect)
    monkeypatch.setattr(cli, "_probe_health", fake_probe)
    return seen


class TestCatalogCommand:
    def test_shows_filesystem_source(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["catalog"])

        assert result.exit_code == 0
        assert "Backend: unavailable (filesystem fallback)" in result.stdout

    def test_lists_every_kind_even_when_empty(self, stub_collect: dict[str, Any]) -> None:
        """未導入の種別も行を出す。無いことが分からないと切り分けに使えない。"""
        result = runner.invoke(cli.app, ["catalog"])

        assert "controlnets (0)" in result.stdout
        assert "(なし)" in result.stdout

    def test_marks_default_checkpoint(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["catalog"])

        assert "hassakuSD15_v13.safetensors  <- 既定 (sd15-hassaku)" in result.stdout

    def test_lists_presets_by_axis(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["catalog"])

        assert "Presets: character 1 / scene 0 / style 1" in result.stdout
        assert "presets/style (1)" in result.stdout

    def test_omits_runtime_lines_when_unreachable(self, stub_collect: dict[str, Any]) -> None:
        """未起動なら実行基盤の行は出さない。空の Devices: は誤読を招く。"""
        result = runner.invoke(cli.app, ["catalog"])

        assert "Devices:" not in result.stdout
        assert "Version:" not in result.stdout

    def test_shows_devices_when_reachable(self, stub_collect: dict[str, Any]) -> None:
        """所要時間がXPUとCPUで一桁違うため、在庫と同じ出力で見えるようにする。"""
        stub_collect["source"] = "api"
        stub_collect["status"] = HealthStatus(
            base_url="http://127.0.0.1:8188",
            comfyui_version="0.3.68",
            devices=("xpu:0",),
        )

        result = runner.invoke(cli.app, ["catalog"])

        assert "Backend: api (http://127.0.0.1:8188)" in result.stdout
        assert "Version: 0.3.68" in result.stdout
        assert "Devices: xpu:0" in result.stdout

    def test_json_includes_runtime(self, stub_collect: dict[str, Any]) -> None:
        stub_collect["source"] = "api"
        stub_collect["status"] = HealthStatus(
            base_url="http://127.0.0.1:8188",
            comfyui_version="0.3.68",
            devices=("xpu:0",),
        )

        payload = json.loads(runner.invoke(cli.app, ["catalog", "--json"]).stdout)

        assert payload["version"] == "0.3.68"
        assert payload["devices"] == ["xpu:0"]

    def test_json_output_is_machine_readable(self, stub_collect: dict[str, Any]) -> None:
        result = runner.invoke(cli.app, ["catalog", "--json"])

        payload = json.loads(result.stdout)
        assert payload["source"] == "filesystem"
        assert payload["devices"] == []
        assert payload["models"]["checkpoints"] == list(MODELS["checkpoints"])
        assert payload["presets"]["style"] == ["sd15-hassaku"]
        assert payload["fonts"] == ["NotoSansJP.ttf"]

    def test_passes_resolved_roots(self, stub_collect: dict[str, Any]) -> None:
        """探索ルートは作業ルート基準の絶対パスで渡す。"""
        runner.invoke(cli.app, ["catalog"])

        assert Path(stub_collect["presets_root"]).is_absolute()
        assert Path(stub_collect["fonts_root"]).is_absolute()
        assert Path(stub_collect["comfyui_home"]).is_absolute()
