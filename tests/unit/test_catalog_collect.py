"""環境と在庫の一括収集 (services.catalog.collect_catalog) のUnit Test。

ComfyUIへ到達できるときはAPI (CatalogBackend) から、到達できないときは
ComfyUIのmodelsディレクトリの直読みから一覧を組み立てる。生成のたびに
ComfyUIを起動・停止する運用のため、探索のためだけに起動を増やさないことが要件。

実ComfyUIへは接続しない。バックエンドはフェイクへ差し替え、
ファイルシステム側は tmp_path へ組み立てる。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agentic_imagegen.config import Settings
from agentic_imagegen.errors import ComfyUIUnavailable
from agentic_imagegen.services.catalog import (
    CATALOG_KINDS,
    CatalogBackend,
    CatalogBackendFactory,
    collect_catalog,
)


class _FakeCatalogBackend:
    """CatalogBackendを構造的に満たすフェイク。"""

    def __init__(self, **available: tuple[str, ...]) -> None:
        self._available = available

    def _get(self, kind: str) -> tuple[str, ...]:
        return self._available.get(kind, ())

    async def available_checkpoints(self) -> tuple[str, ...]:
        return self._get("checkpoints")

    async def available_loras(self) -> tuple[str, ...]:
        return self._get("loras")

    async def available_controlnets(self) -> tuple[str, ...]:
        return self._get("controlnets")

    async def available_ipadapters(self) -> tuple[str, ...]:
        return self._get("ipadapters")

    async def available_clip_visions(self) -> tuple[str, ...]:
        return self._get("clip_visions")

    async def available_diffusion_models(self) -> tuple[str, ...]:
        return self._get("diffusion_models")

    async def available_text_encoders(self) -> tuple[str, ...]:
        return self._get("text_encoders")

    async def available_vaes(self) -> tuple[str, ...]:
        return self._get("vaes")

    async def available_upscale_models(self) -> tuple[str, ...]:
        return self._get("upscale_models")

    async def available_embeddings(self) -> tuple[str, ...]:
        return self._get("embeddings")


def _factory_for(backend: CatalogBackend) -> CatalogBackendFactory:
    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[CatalogBackend]:
        yield backend

    return factory


def _unavailable_factory() -> CatalogBackendFactory:
    """接続時点で ComfyUIUnavailable を送出するファクトリ。"""

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[CatalogBackend]:
        raise ComfyUIUnavailable("ComfyUIへ接続できません: http://127.0.0.1:8188")
        yield  # pragma: no cover - 到達しない

    return factory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=300,
        output_root=tmp_path / "outputs",
    )


@pytest.fixture
def comfyui_home(tmp_path: Path) -> Path:
    """ComfyUIのmodelsディレクトリを模したツリー。"""
    root = tmp_path / "ComfyUI"
    files = {
        "checkpoints": ["b.safetensors", "a.safetensors", "notes.txt"],
        "loras": ["style.safetensors"],
        "vae": ["klF8Anime2.safetensors"],
        "upscale_models": ["RealESRGAN_x4plus_anime_6B.pth"],
        "embeddings": ["negativeXL_D.safetensors"],
    }
    for subdir, names in files.items():
        directory = root / "models" / subdir
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b"")
    return root


@pytest.fixture
def presets_root(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    for kind, names in (
        ("characters", ["kaede"]),
        ("scenes", ["rooftop"]),
        ("styles", ["sd15-hassaku", "anime-soft"]),
    ):
        directory = root / kind
        directory.mkdir(parents=True)
        for name in names:
            (directory / f"{name}.yaml").write_text("description: x\n", encoding="utf-8")
    return root


@pytest.fixture
def fonts_root(tmp_path: Path) -> Path:
    root = tmp_path / "fonts"
    root.mkdir()
    (root / "NotoSansJP.ttf").write_bytes(b"")
    (root / "README.md").write_text("x", encoding="utf-8")
    return root


def _collect(
    settings: Settings,
    factory: CatalogBackendFactory,
    *,
    comfyui_home: Path,
    presets_root: Path,
    fonts_root: Path,
):
    import asyncio

    return asyncio.run(
        collect_catalog(
            settings,
            backend_factory=factory,
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )
    )


class TestKinds:
    def test_kinds_cover_every_backend_method(self) -> None:
        """CatalogBackendの available_* と種別が1対1で対応すること。

        片方だけ増えると、APIとファイルシステムで見える種別が食い違う。
        """
        methods = {name for name in dir(CatalogBackend) if name.startswith("available_")}

        assert {kind.method for kind in CATALOG_KINDS} == methods


class TestApiSource:
    def test_uses_api_when_reachable(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        backend = _FakeCatalogBackend(
            checkpoints=("api-only.safetensors",),
            embeddings=("negativeXL_D",),
        )

        snapshot = _collect(
            settings,
            _factory_for(backend),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.source == "api"
        assert snapshot.models["checkpoints"] == ("api-only.safetensors",)
        assert snapshot.models["embeddings"] == ("negativeXL_D",)
        # APIが空を返す種別は空のまま。ファイルシステムで補わない
        assert snapshot.models["loras"] == ()


class TestFilesystemFallback:
    def test_falls_back_when_unreachable(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.source == "filesystem"
        assert snapshot.models["checkpoints"] == ("a.safetensors", "b.safetensors")
        assert snapshot.models["loras"] == ("style.safetensors",)

    def test_ignores_unrelated_suffixes(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        """モデル以外のファイル (README等) を一覧へ混ぜない。"""
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert "notes.txt" not in snapshot.models["checkpoints"]

    def test_strips_suffix_for_embeddings(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        """embeddingはprompt中の `embedding:<name>` と同じ形 (拡張子なし) で返す。"""
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.models["embeddings"] == ("negativeXL_D",)

    def test_includes_one_level_of_subfolders(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        """ComfyUIはサブフォルダ1階層までを `sub/name` の形で扱う。"""
        nested = comfyui_home / "models" / "loras" / "anime"
        nested.mkdir()
        (nested / "extra.safetensors").write_bytes(b"")

        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.models["loras"] == ("anime/extra.safetensors", "style.safetensors")

    def test_missing_directory_is_empty(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        """未導入の種別 (controlnet等) はエラーにせず空で返す。"""
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.models["controlnets"] == ()

    def test_missing_comfyui_home_is_empty(
        self, settings: Settings, tmp_path: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=tmp_path / "absent",
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.source == "filesystem"
        assert all(names == () for names in snapshot.models.values())


class TestRepositorySources:
    def test_presets_come_from_repository(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        """presetはリポジトリ内にあるため、ComfyUIの状態によらず同じ結果になる。"""
        backend = _FakeCatalogBackend(checkpoints=("x.safetensors",))

        via_api = _collect(
            settings,
            _factory_for(backend),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )
        via_fs = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert via_api.presets == via_fs.presets
        assert via_api.presets["style"] == ("anime-soft", "sd15-hassaku")
        assert via_api.presets["character"] == ("kaede",)

    def test_fonts_list_only_font_files(
        self, settings: Settings, comfyui_home: Path, presets_root: Path, fonts_root: Path
    ) -> None:
        snapshot = _collect(
            settings,
            _unavailable_factory(),
            comfyui_home=comfyui_home,
            presets_root=presets_root,
            fonts_root=fonts_root,
        )

        assert snapshot.fonts == ("NotoSansJP.ttf",)
