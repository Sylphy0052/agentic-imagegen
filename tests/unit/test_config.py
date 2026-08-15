"""Settings (環境変数由来の設定) のテスト。"""

from pathlib import Path

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.errors import InvalidConfiguration

ENV_KEYS = [
    "COMFYUI_BASE_URL",
    "IMAGEGEN_MAX_WIDTH",
    "IMAGEGEN_MAX_HEIGHT",
    "IMAGEGEN_MAX_PIXELS",
    "IMAGEGEN_MAX_BATCH",
    "IMAGEGEN_TIMEOUT",
    "IMAGEGEN_OUTPUT_ROOT",
    "IMAGEGEN_PRESETS_ROOT",
    "IMAGEGEN_MAX_SOURCE_BYTES",
    "IMAGEGEN_MAX_UPSCALED_PIXELS",
    "IMAGEGEN_FONTS_ROOT",
    "IMAGEGEN_REGISTRY_ROOT",
    "COMFYUI_HOME",
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults() -> None:
    settings = Settings.from_env()

    assert settings.comfyui_base_url == "http://127.0.0.1:8188"
    assert settings.max_width == 2048
    assert settings.max_height == 2048
    assert settings.max_pixels == 4194304
    assert settings.max_batch == 4
    assert settings.timeout_seconds == 300
    assert settings.output_root.name == "outputs"
    assert settings.presets_root.name == "presets"
    assert settings.registry_root.name == "registry"
    assert settings.max_source_bytes == 32 * 1024 * 1024
    assert settings.max_upscaled_pixels == 16777216


def test_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://127.0.0.1:9000/")
    monkeypatch.setenv("IMAGEGEN_MAX_WIDTH", "1024")
    monkeypatch.setenv("IMAGEGEN_MAX_HEIGHT", "1024")
    monkeypatch.setenv("IMAGEGEN_MAX_PIXELS", "1048576")
    monkeypatch.setenv("IMAGEGEN_MAX_BATCH", "2")
    monkeypatch.setenv("IMAGEGEN_TIMEOUT", "30")
    monkeypatch.setenv("IMAGEGEN_OUTPUT_ROOT", "tmp-outputs")
    monkeypatch.setenv("IMAGEGEN_PRESETS_ROOT", "tmp-presets")
    monkeypatch.setenv("IMAGEGEN_MAX_SOURCE_BYTES", "1048576")
    monkeypatch.setenv("IMAGEGEN_MAX_UPSCALED_PIXELS", "2097152")

    settings = Settings.from_env()

    assert settings.comfyui_base_url == "http://127.0.0.1:9000"
    assert settings.max_width == 1024
    assert settings.max_height == 1024
    assert settings.max_pixels == 1048576
    assert settings.max_batch == 2
    assert settings.timeout_seconds == 30
    assert settings.output_root.name == "tmp-outputs"
    assert settings.presets_root.name == "tmp-presets"
    assert settings.max_source_bytes == 1048576
    assert settings.max_upscaled_pixels == 2097152


def test_registry_root_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGEN_REGISTRY_ROOT", "tmp-registry")

    assert Settings.from_env().registry_root.name == "tmp-registry"


def test_empty_registry_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGEN_REGISTRY_ROOT", "  ")

    with pytest.raises(InvalidConfiguration, match="IMAGEGEN_REGISTRY_ROOT"):
        Settings.from_env()


def test_empty_presets_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGEN_PRESETS_ROOT", "  ")

    with pytest.raises(InvalidConfiguration, match="IMAGEGEN_PRESETS_ROOT"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("IMAGEGEN_MAX_WIDTH", "abc"),
        ("IMAGEGEN_MAX_WIDTH", "0"),
        ("IMAGEGEN_MAX_HEIGHT", "-1"),
        ("IMAGEGEN_MAX_PIXELS", "0"),
        ("IMAGEGEN_MAX_UPSCALED_PIXELS", "0"),
        ("IMAGEGEN_MAX_BATCH", "0"),
        ("IMAGEGEN_MAX_BATCH", "9999"),
        ("IMAGEGEN_TIMEOUT", "0"),
        ("IMAGEGEN_TIMEOUT", "-5"),
    ],
)
def test_invalid_numeric_env(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(InvalidConfiguration):
        Settings.from_env()


@pytest.mark.parametrize(
    "url",
    ["", "127.0.0.1:8188", "ftp://127.0.0.1:8188", "not a url"],
)
def test_invalid_base_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("COMFYUI_BASE_URL", url)
    with pytest.raises(InvalidConfiguration):
        Settings.from_env()


def test_comfyui_home_defaults_to_expanded_home() -> None:
    """既定の `~/ComfyUI` はホームを展開した絶対パスで持つ。

    展開しないままだと、ComfyUIへ到達できないときのフォールバックで
    `~/ComfyUI` という名前のディレクトリを探しに行く。
    """
    settings = Settings.from_env()

    assert settings.comfyui_home.is_absolute()
    assert settings.comfyui_home.name == "ComfyUI"


def test_comfyui_home_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFYUI_HOME", "/opt/ComfyUI")

    assert Settings.from_env().comfyui_home == Path("/opt/ComfyUI")


def test_empty_comfyui_home_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFYUI_HOME", "   ")

    with pytest.raises(InvalidConfiguration, match="COMFYUI_HOME"):
        Settings.from_env()
