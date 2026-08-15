"""どのバックエンドの具象を使うかの選択のテスト。

`IMAGEGEN_BACKEND` と具象クラスの対応を知るのはここ (composition root) だけ、
という約束を守れているかを見る。services / domain 層がadaptersをimportしない
ことは test_service_layer_isolation.py が別に守っている。
"""

from __future__ import annotations

from pathlib import Path

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.adapters.diffusers.backend import DiffusersBackend
from agentic_imagegen.adapters.diffusers.catalog import DiffusersCatalog
from agentic_imagegen.backends import open_catalog_backend, open_generation_backend
from agentic_imagegen.config import BackendName, Settings


def _settings(backend: BackendName) -> Settings:
    return Settings(
        comfyui_base_url="http://127.0.0.1:8188",
        max_width=2048,
        max_height=2048,
        max_pixels=4194304,
        max_batch=4,
        timeout_seconds=30,
        output_root=Path("outputs"),
        backend=backend,
        models_root=Path("/models"),
    )


class TestGenerationBackend:
    def test_comfyui_is_the_default(self) -> None:
        assert isinstance(open_generation_backend(_settings("comfyui")), ComfyUIClient)

    def test_diffusers_is_selected_by_setting(self) -> None:
        assert isinstance(open_generation_backend(_settings("diffusers")), DiffusersBackend)


class TestCatalogBackend:
    def test_comfyui_is_the_default(self) -> None:
        assert isinstance(open_catalog_backend(_settings("comfyui")), ComfyUIClient)

    def test_diffusers_is_selected_by_setting(self) -> None:
        """列挙はHTTPを使わずローカルのファイルを見るため、生成側とは別の具象になる。"""
        assert isinstance(open_catalog_backend(_settings("diffusers")), DiffusersCatalog)
