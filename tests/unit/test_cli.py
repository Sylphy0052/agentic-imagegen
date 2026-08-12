"""CLIの動作とexit codeのテスト。"""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_imagegen import cli
from agentic_imagegen.adapters.comfyui.client import HealthStatus
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import (
    ComfyUIUnavailable,
    GenerationTimeout,
    OutputNotFound,
    WorkflowSubmissionError,
)

runner = CliRunner()

VALID_SPEC = """
version: "1"
task: txt2img
prompt:
  positive: 1girl, blue hair
  negative: low quality
generation:
  width: 512
  height: 768
  steps: 20
  cfg: 5.5
  seed: 4242
  batch_size: 1
model:
  checkpoint: v1-5-pruned-emaonly.safetensors
output:
  prefix: blue_hair
"""


class FakeClient:
    """CLIテスト用のComfyUIClient代替。"""

    def __init__(self, *args: Any, health_error: Exception | None = None, **kwargs: Any) -> None:
        self.health_error = health_error

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def health(self) -> HealthStatus:
        if self.health_error is not None:
            raise self.health_error
        return HealthStatus(
            base_url="http://127.0.0.1:8188",
            comfyui_version="0.3.40",
            devices=("cpu",),
        )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "COMFYUI_BASE_URL",
        "IMAGEGEN_MAX_WIDTH",
        "IMAGEGEN_MAX_HEIGHT",
        "IMAGEGEN_MAX_PIXELS",
        "IMAGEGEN_MAX_BATCH",
        "IMAGEGEN_TIMEOUT",
        "IMAGEGEN_OUTPUT_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_spec(tmp_path: Path, content: str = VALID_SPEC) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(content, encoding="utf-8")
    return path


# --- health -----------------------------------------------------------------


def test_health_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ComfyUIClient", FakeClient)

    result = runner.invoke(cli.app, ["health"])

    assert result.exit_code == 0
    assert "ComfyUI: reachable" in result.output
    assert "http://127.0.0.1:8188" in result.output


def test_health_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(*args: Any, **kwargs: Any) -> FakeClient:
        return FakeClient(health_error=ComfyUIUnavailable("接続できません"))

    monkeypatch.setattr(cli, "ComfyUIClient", factory)

    result = runner.invoke(cli.app, ["health"])

    assert result.exit_code == 3
    assert "unreachable" in result.output


def test_health_invalid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGEN_MAX_WIDTH", "abc")

    result = runner.invoke(cli.app, ["health"])

    assert result.exit_code == 9


# --- validate ---------------------------------------------------------------


def test_validate_ok(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["validate", str(_write_spec(tmp_path))])

    assert result.exit_code == 0
    assert "OK" in result.output
    assert "512x768" in result.output


def test_validate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["validate", str(tmp_path / "absent.yaml")])

    assert result.exit_code == 2


def test_validate_invalid_spec(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, VALID_SPEC.replace("batch_size: 1", "batch_size: 99"))

    result = runner.invoke(cli.app, ["validate", str(spec)])

    assert result.exit_code == 2
    assert "batch_size" in result.output


def test_validate_exceeds_configured_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGEN_MAX_HEIGHT", "512")

    result = runner.invoke(cli.app, ["validate", str(_write_spec(tmp_path))])

    assert result.exit_code == 2
    assert "height" in result.output


# --- generate ---------------------------------------------------------------


def _stub_generate(result: GenerationResult | Exception) -> Any:
    async def _generate(*args: Any, **kwargs: Any) -> GenerationResult:
        if isinstance(result, Exception):
            raise result
        return result

    return _generate


def _result(tmp_path: Path) -> GenerationResult:
    directory = tmp_path / "outputs" / "2026-08-12" / "blue_hair"
    directory.mkdir(parents=True)
    image = directory / "image_0001.png"
    image.write_bytes(b"png")
    metadata = directory / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    return GenerationResult(
        prompt_id="pid-1",
        seed=4242,
        directory=directory,
        files=(image,),
        metadata_path=metadata,
    )


def test_generate_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ComfyUIClient", FakeClient)
    monkeypatch.setattr(cli, "generate", _stub_generate(_result(tmp_path)))

    result = runner.invoke(cli.app, ["generate", str(_write_spec(tmp_path))])

    assert result.exit_code == 0
    assert "image_0001.png" in result.output
    assert "pid-1" in result.output


def test_generate_validates_spec_before_connecting(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, VALID_SPEC.replace("steps: 20", "steps: 500"))

    result = runner.invoke(cli.app, ["generate", str(spec)])

    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ComfyUIUnavailable("接続できません"), 3),
        (WorkflowSubmissionError("拒否されました"), 5),
        (GenerationTimeout("時間切れ"), 6),
        (OutputNotFound("画像がありません"), 8),
    ],
)
def test_generate_error_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: int,
) -> None:
    monkeypatch.setattr(cli, "ComfyUIClient", FakeClient)
    monkeypatch.setattr(cli, "generate", _stub_generate(error))

    result = runner.invoke(cli.app, ["generate", str(_write_spec(tmp_path))])

    assert result.exit_code == expected_code


def test_generate_unexpected_error_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "ComfyUIClient", FakeClient)
    monkeypatch.setattr(cli, "generate", _stub_generate(RuntimeError("boom")))

    result = runner.invoke(cli.app, ["generate", str(_write_spec(tmp_path))])

    assert result.exit_code == 1


def test_generate_timeout_option_is_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _generate(*args: Any, **kwargs: Any) -> GenerationResult:
        captured.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "ComfyUIClient", FakeClient)
    monkeypatch.setattr(cli, "generate", _generate)

    result = runner.invoke(cli.app, ["generate", str(_write_spec(tmp_path)), "--timeout", "42"])

    assert result.exit_code == 0
    assert captured["timeout"] == 42.0


def test_version_command() -> None:
    result = runner.invoke(cli.app, ["version"])

    assert result.exit_code == 0
