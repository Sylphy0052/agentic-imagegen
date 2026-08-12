"""Settings (環境変数由来の設定) のテスト。"""

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


def test_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://127.0.0.1:9000/")
    monkeypatch.setenv("IMAGEGEN_MAX_WIDTH", "1024")
    monkeypatch.setenv("IMAGEGEN_MAX_HEIGHT", "1024")
    monkeypatch.setenv("IMAGEGEN_MAX_PIXELS", "1048576")
    monkeypatch.setenv("IMAGEGEN_MAX_BATCH", "2")
    monkeypatch.setenv("IMAGEGEN_TIMEOUT", "30")
    monkeypatch.setenv("IMAGEGEN_OUTPUT_ROOT", "tmp-outputs")

    settings = Settings.from_env()

    assert settings.comfyui_base_url == "http://127.0.0.1:9000"
    assert settings.max_width == 1024
    assert settings.max_height == 1024
    assert settings.max_pixels == 1048576
    assert settings.max_batch == 2
    assert settings.timeout_seconds == 30
    assert settings.output_root.name == "tmp-outputs"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("IMAGEGEN_MAX_WIDTH", "abc"),
        ("IMAGEGEN_MAX_WIDTH", "0"),
        ("IMAGEGEN_MAX_HEIGHT", "-1"),
        ("IMAGEGEN_MAX_PIXELS", "0"),
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
