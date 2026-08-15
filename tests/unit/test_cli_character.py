"""`imagegen character` のテスト。

台帳を引くだけのコマンドで、ComfyUIへは接続しない。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agentic_imagegen import cli

runner = CliRunner()

HASSAKU = "hassakuSD15_v13.safetensors"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "presets" / "characters").mkdir(parents=True)
    (tmp_path / "presets" / "styles").mkdir()
    (tmp_path / "presets" / "characters" / "anime-girl-blue.yaml").write_text("", encoding="utf-8")
    (tmp_path / "presets" / "styles" / "sd15-hassaku.yaml").write_text("", encoding="utf-8")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "aoi.png").write_bytes(b"png")
    (tmp_path / "registry" / "characters").mkdir(parents=True)
    _write(tmp_path, "aoi")

    monkeypatch.setenv("IMAGEGEN_PRESETS_ROOT", str(tmp_path / "presets"))
    monkeypatch.setenv("IMAGEGEN_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root: Path, name: str, **overrides: object) -> None:
    document: dict[str, object] = {
        "description": "青い髪の少女",
        "preset": "anime-girl-blue",
        "style": "sd15-hassaku",
        "checkpoint": HASSAKU,
        "reference": "inputs/aoi.png",
        "seed": 777001,
    }
    document.update(overrides)
    (root / "registry" / "characters" / f"{name}.yaml").write_text(
        yaml.safe_dump(document, allow_unicode=True), encoding="utf-8"
    )


class TestList:
    def test_lists_registered_characters(self, workspace: Path) -> None:
        _write(workspace, "sora", description="黒髪の少年")

        result = runner.invoke(cli.app, ["character", "list"])

        assert result.exit_code == 0
        assert "aoi" in result.stdout
        assert "sora" in result.stdout

    def test_empty_registry_says_so(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGEGEN_REGISTRY_ROOT", str(tmp_path / "registry"))
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli.app, ["character", "list"])

        assert result.exit_code == 0
        assert "(なし)" in result.stdout

    def test_json_is_machine_readable(self, workspace: Path) -> None:
        result = runner.invoke(cli.app, ["character", "list", "--json"])

        payload = json.loads(result.stdout)
        assert [entry["name"] for entry in payload] == ["aoi"]


class TestShow:
    def test_prints_every_field(self, workspace: Path) -> None:
        result = runner.invoke(cli.app, ["character", "show", "aoi"])

        assert result.exit_code == 0
        assert "anime-girl-blue" in result.stdout
        assert HASSAKU in result.stdout
        assert "inputs/aoi.png" in result.stdout
        assert "777001" in result.stdout
        assert result.stderr == ""

    def test_missing_reference_is_warned(self, workspace: Path) -> None:
        """台帳は古びる。欠けたまま生成すると別人が出てから気づくことになる。"""
        (workspace / "inputs" / "aoi.png").unlink()

        result = runner.invoke(cli.app, ["character", "show", "aoi"])

        assert result.exit_code == 0
        assert "inputs/aoi.png" in result.stderr

    def test_unknown_character_exits_with_two(self, workspace: Path) -> None:
        result = runner.invoke(cli.app, ["character", "show", "unknown"])

        assert result.exit_code == 2

    def test_json_carries_missing(self, workspace: Path) -> None:
        (workspace / "inputs" / "aoi.png").unlink()

        result = runner.invoke(cli.app, ["character", "show", "aoi", "--json"])

        payload = json.loads(result.stdout)
        assert payload["seed"] == 777001
        assert payload["missing"]
