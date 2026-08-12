"""ComfyUI HTTP APIクライアント。

ComfyUIのエンドポイント仕様とレスポンス形状の知識はこのモジュールに閉じ込め、
上位層へは本アプリケーションの例外型とデータ構造だけを返す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self

import httpx

from agentic_imagegen.config import Settings
from agentic_imagegen.errors import ComfyUIUnavailable

logger: Final = logging.getLogger(__name__)

#: health checkは生成本体より短い時間で打ち切る。
HEALTH_TIMEOUT_SECONDS: Final = 5.0

_CHECKPOINT_LOADER: Final = "CheckpointLoaderSimple"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """ComfyUIの到達確認結果。"""

    base_url: str
    comfyui_version: str | None
    devices: tuple[str, ...]


class ComfyUIClient:
    """ComfyUIサーバとの通信を担う。

    `transport` はテスト用にhttpxのMockTransportを差し込むための拡張点であり、
    通常利用では指定しない。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = settings.comfyui_base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> HealthStatus:
        """ComfyUIへ到達できるかを確認する。到達不能なら ComfyUIUnavailable。"""
        payload = await self._get_json("/system_stats", timeout=HEALTH_TIMEOUT_SECONDS)

        system = payload.get("system")
        version: str | None = None
        if isinstance(system, dict):
            raw_version = system.get("comfyui_version")
            version = raw_version if isinstance(raw_version, str) else None

        devices: tuple[str, ...] = ()
        raw_devices = payload.get("devices")
        if isinstance(raw_devices, list):
            devices = tuple(
                device["name"]
                for device in raw_devices
                if isinstance(device, dict) and isinstance(device.get("name"), str)
            )

        logger.debug("ComfyUI health ok: url=%s version=%s", self._base_url, version)
        return HealthStatus(base_url=self._base_url, comfyui_version=version, devices=devices)

    async def available_checkpoints(self) -> tuple[str, ...]:
        """利用可能なcheckpoint名の一覧を取得する。

        取得できたが形状が想定と違う場合は空タプルを返し、検証をスキップさせる。
        """
        payload = await self._get_json(f"/object_info/{_CHECKPOINT_LOADER}")
        names = _extract_checkpoint_names(payload)
        if not names:
            logger.warning(
                "ComfyUIからcheckpoint一覧を取得できませんでした (レスポンス形式が想定外)"
            )
        return names

    async def _get_json(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        try:
            response = await self._client.get(path, timeout=timeout or self._client.timeout)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIがエラーを返しました (HTTP {exc.response.status_code}): "
                f"{self._base_url}{path}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIへ接続できません: {self._base_url} ({type(exc).__name__})"
            ) from exc
        except ValueError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIのレスポンスを解釈できません: {self._base_url}{path}"
            ) from exc

        if not isinstance(payload, dict):
            raise ComfyUIUnavailable(f"ComfyUIのレスポンス形式が想定外です: {self._base_url}{path}")
        return payload


def _extract_checkpoint_names(payload: dict[str, Any]) -> tuple[str, ...]:
    """object_infoのレスポンスからcheckpoint名の一覧を取り出す。

    ComfyUIの応答は入れ子が深く欠損もありうるため、防御的に取り出す。
    """
    loader = payload.get(_CHECKPOINT_LOADER)
    if not isinstance(loader, dict):
        return ()
    required = loader.get("input", {})
    if not isinstance(required, dict):
        return ()
    fields = required.get("required")
    if not isinstance(fields, dict):
        return ()
    entry = fields.get("ckpt_name")
    if not isinstance(entry, list) or not entry:
        return ()
    candidates = entry[0]
    if not isinstance(candidates, list):
        return ()
    return tuple(name for name in candidates if isinstance(name, str))


__all__ = ["ComfyUIClient", "HealthStatus"]
