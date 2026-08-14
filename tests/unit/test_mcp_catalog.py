"""MCP toolの列挙系10関数 (list_models等) の中身。

services/mcp_tools.py はComfyUIClientを直接importせず、CatalogBackendFactory
経由でバックエンドを受け取る (Issue #31)。ここではフェイクのバックエンドを
差し替えて、正しい available_* メソッドが呼ばれること、渡した Settings が
そのままファクトリへ渡ることを見る。実ComfyUIへは接続しない。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.services import mcp_tools
from agentic_imagegen.services.catalog import CatalogBackend, CatalogBackendFactory


class _FakeCatalogBackend:
    """CatalogBackendを構造的に満たすフェイク。実ComfyUIへは接続しない。

    どの `available_*` が返す値もコンストラクタ引数で個別に指定できるようにし、
    テスト側で「意図したメソッドが呼ばれたか」まで確認できるようにする。
    """

    def __init__(self, **available: tuple[str, ...]) -> None:
        self._checkpoints = available.get("checkpoints", ())
        self._loras = available.get("loras", ())
        self._controlnets = available.get("controlnets", ())
        self._ipadapters = available.get("ipadapters", ())
        self._clip_visions = available.get("clip_visions", ())
        self._diffusion_models = available.get("diffusion_models", ())
        self._text_encoders = available.get("text_encoders", ())
        self._vaes = available.get("vaes", ())
        self._upscale_models = available.get("upscale_models", ())
        self._embeddings = available.get("embeddings", ())

    async def available_checkpoints(self) -> tuple[str, ...]:
        return self._checkpoints

    async def available_loras(self) -> tuple[str, ...]:
        return self._loras

    async def available_controlnets(self) -> tuple[str, ...]:
        return self._controlnets

    async def available_ipadapters(self) -> tuple[str, ...]:
        return self._ipadapters

    async def available_clip_visions(self) -> tuple[str, ...]:
        return self._clip_visions

    async def available_diffusion_models(self) -> tuple[str, ...]:
        return self._diffusion_models

    async def available_text_encoders(self) -> tuple[str, ...]:
        return self._text_encoders

    async def available_vaes(self) -> tuple[str, ...]:
        return self._vaes

    async def available_upscale_models(self) -> tuple[str, ...]:
        return self._upscale_models

    async def available_embeddings(self) -> tuple[str, ...]:
        return self._embeddings


def _factory_for(
    backend: _FakeCatalogBackend, *, seen_settings: list[Settings]
) -> CatalogBackendFactory:
    """Settingsを受け取ってフェイクを返す、CatalogBackendFactory互換の関数を作る。

    `ComfyUIClient(settings)` と同じく「呼ばれたらasync context managerを返す」
    形にするため、asynccontextmanagerで包む。渡されたsettingsは呼び出し確認用に
    記録する。
    """

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[_FakeCatalogBackend]:
        seen_settings.append(settings)
        yield backend

    return factory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        presets_root=tmp_path / "presets",
    )


#: (呼び出す関数, フェイク側の属性名) の組。10関数それぞれで対応する
#: available_* だけが呼ばれることを確認する。
_LIST_FUNCTIONS: tuple[tuple[Callable[..., Coroutine[None, None, list[str]]], str], ...] = (
    (mcp_tools.list_models, "checkpoints"),
    (mcp_tools.list_loras, "loras"),
    (mcp_tools.list_controlnets, "controlnets"),
    (mcp_tools.list_ipadapters, "ipadapters"),
    (mcp_tools.list_clip_visions, "clip_visions"),
    (mcp_tools.list_diffusion_models, "diffusion_models"),
    (mcp_tools.list_text_encoders, "text_encoders"),
    (mcp_tools.list_vaes, "vaes"),
    (mcp_tools.list_upscale_models, "upscale_models"),
    (mcp_tools.list_embeddings, "embeddings"),
)


def test_list_functions_cover_every_catalog_tool() -> None:
    """列挙系を1つ足したら `_LIST_FUNCTIONS` へも足す。

    このテーブルの取りこぼしは、追加した関数のテストが単に存在しない形で出るため
    テストの成否には現れない。`mcp_tools` が公開している `list_*` と突き合わせて
    追加漏れをここで落とす。`list_workflows` はバックエンドへ接続しないため除く。
    """
    exported = {
        name for name in dir(mcp_tools) if name.startswith("list_") and name != "list_workflows"
    }

    assert {list_fn.__name__ for list_fn, _ in _LIST_FUNCTIONS} == exported


def test_list_functions_cover_every_catalog_backend_method() -> None:
    """`CatalogBackend` の `available_*` が全て、いずれかの列挙系から呼ばれている。

    Protocolへメソッドを足しただけで、それを返すtoolが無い状態を防ぐ。
    テーブルの属性名は `_FakeCatalogBackend` の命名 (available_ を外した形) に合わせる。
    """
    protocol_methods = {name for name in dir(CatalogBackend) if name.startswith("available_")}

    assert {f"available_{attribute}" for _, attribute in _LIST_FUNCTIONS} == protocol_methods


@pytest.mark.parametrize(
    ("list_fn", "attribute"), _LIST_FUNCTIONS, ids=[attr for _, attr in _LIST_FUNCTIONS]
)
async def test_list_function_returns_backend_values_via_factory(
    list_fn: Callable[..., Coroutine[None, None, list[str]]],
    attribute: str,
    settings: Settings,
) -> None:
    """指定したbackend_factoryが開くバックエンドの、対応するavailable_*の値を返す。"""
    seen_settings: list[Settings] = []
    backend = _FakeCatalogBackend(**{attribute: ("a.safetensors", "b.safetensors")})
    factory = _factory_for(backend, seen_settings=seen_settings)

    result = await list_fn(settings, backend_factory=factory)

    assert result == ["a.safetensors", "b.safetensors"]
    # backend_factoryへ渡ったのは呼び出し時に指定したsettingsそのもの
    assert seen_settings == [settings]


@pytest.mark.parametrize(
    ("list_fn", "attribute"), _LIST_FUNCTIONS, ids=[attr for _, attr in _LIST_FUNCTIONS]
)
async def test_list_function_returns_empty_when_backend_has_none(
    list_fn: Callable[..., Coroutine[None, None, list[str]]],
    attribute: str,
    settings: Settings,
) -> None:
    """未配置 (空タプル) のときは空listを返す。"""
    backend = _FakeCatalogBackend()
    factory = _factory_for(backend, seen_settings=[])

    result = await list_fn(settings, backend_factory=factory)

    assert result == []
