"""`imagegen validate` がstyle presetの助言を出すことのテスト。

助言は検証結果を変えない。exit codeは0のままで、stderrへ出す。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agentic_imagegen import cli

runner = CliRunner()

HASSAKU = "hassakuSD15_v13.safetensors"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    styles = tmp_path / "presets" / "styles"
    styles.mkdir(parents=True)
    (styles / "sd15-hassaku.yaml").write_text(
        yaml.safe_dump({"applies_to": [HASSAKU], "model": {"clip_skip": 2}}), encoding="utf-8"
    )
    monkeypatch.setenv("IMAGEGEN_PRESETS_ROOT", str(tmp_path / "presets"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_spec(
    root: Path, *, style: str | None = None, generation: dict[str, object] | None = None
) -> Path:
    payload: dict[str, object] = {
        "version": "1",
        "task": "txt2img",
        "prompt": {"positive": "1girl"},
        "model": {"checkpoint": HASSAKU},
    }
    if style is not None:
        payload["presets"] = {"style": style}
    if generation is not None:
        payload["generation"] = generation
    path = root / "spec.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestValidateAdvice:
    def test_warns_when_style_preset_is_missing(self, workspace: Path) -> None:
        result = runner.invoke(cli.app, ["validate", str(_write_spec(workspace))])

        assert result.exit_code == 0
        assert result.stdout.startswith("OK")
        assert "sd15-hassaku" in result.stderr

    def test_silent_when_style_preset_matches(self, workspace: Path) -> None:
        path = _write_spec(workspace, style="sd15-hassaku")

        result = runner.invoke(cli.app, ["validate", str(path)])

        assert result.exit_code == 0
        assert result.stderr == ""


class TestValidateEstimate:
    def test_shows_every_device(self, workspace: Path) -> None:
        """validateはComfyUIへ接続しないため、どこで動くかは分からない。"""
        result = runner.invoke(cli.app, ["validate", str(_write_spec(workspace))])

        assert result.exit_code == 0
        assert "Estimate:" in result.stdout
        assert "CUDA" in result.stdout
        assert "XPU" in result.stdout
        assert "CPU" in result.stdout

    def test_shows_only_the_declared_device(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IMAGEGEN_DEVICE があれば、その基盤だけを出す。"""
        monkeypatch.setenv("IMAGEGEN_DEVICE", "cuda")

        result = runner.invoke(cli.app, ["validate", str(_write_spec(workspace))])

        assert result.exit_code == 0
        assert "CUDA" in result.stdout
        assert "XPU" not in result.stdout
        assert "CPU" not in result.stdout

    def test_warns_when_the_estimate_exceeds_the_timeout(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """XPUでもタイムアウトするなら、12分待ってから知るのでは遅い。"""
        monkeypatch.setenv("IMAGEGEN_TIMEOUT", "60")
        path = _write_spec(
            workspace,
            style="sd15-hassaku",
            generation={"width": 1024, "height": 1024, "steps": 40},
        )

        result = runner.invoke(cli.app, ["validate", str(path)])

        assert result.exit_code == 0
        assert "IMAGEGEN_TIMEOUT" in result.stderr

    def test_warning_follows_the_declared_device(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDAと宣言してあれば、XPUなら超える条件でも警告しない。

        この警告が実態と合わないままだと、警告そのものが読まれなくなる。
        """
        monkeypatch.setenv("IMAGEGEN_TIMEOUT", "60")
        generation = {"width": 1024, "height": 1024, "steps": 40}
        path = _write_spec(workspace, style="sd15-hassaku", generation=generation)

        without = runner.invoke(cli.app, ["validate", str(path)])
        monkeypatch.setenv("IMAGEGEN_DEVICE", "cuda")
        with_cuda = runner.invoke(cli.app, ["validate", str(path)])

        assert "IMAGEGEN_TIMEOUT" in without.stderr
        assert with_cuda.stderr == ""

    def test_rejects_an_unknown_device(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """綴り違いを既定へ落とすと、宣言したつもりの基盤と違う見積りが出る。"""
        monkeypatch.setenv("IMAGEGEN_DEVICE", "gpu")

        result = runner.invoke(cli.app, ["validate", str(_write_spec(workspace))])

        assert result.exit_code == 9
        assert "IMAGEGEN_DEVICE" in result.stderr

    def test_silent_when_the_estimate_fits(self, workspace: Path) -> None:
        path = _write_spec(workspace, style="sd15-hassaku")

        result = runner.invoke(cli.app, ["validate", str(path)])

        assert result.stderr == ""
