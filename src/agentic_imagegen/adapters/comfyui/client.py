"""ComfyUI HTTP APIクライアント。

ComfyUIのエンドポイント仕様とレスポンス形状の知識はこのモジュールに閉じ込め、
上位層へは本アプリケーションの例外型とデータ構造だけを返す。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self
from urllib.parse import urlencode

import httpx
import websockets
from websockets.exceptions import WebSocketException

from agentic_imagegen.config import Settings
from agentic_imagegen.domain.results import HealthStatus, ImageRef
from agentic_imagegen.errors import (
    ComfyUIUnavailable,
    GenerationFailed,
    GenerationTimeout,
    InvalidGenerationSpec,
    OutputNotFound,
    WorkflowSubmissionError,
)

logger: Final = logging.getLogger(__name__)

#: health checkは生成本体より短い時間で打ち切る。
HEALTH_TIMEOUT_SECONDS: Final = 5.0

_CHECKPOINT_LOADER: Final = "CheckpointLoaderSimple"
_LORA_LOADER: Final = "LoraLoader"
_CONTROLNET_LOADER: Final = "ControlNetLoader"
_IPADAPTER_LOADER: Final = "IPAdapterModelLoader"
_CLIP_VISION_LOADER: Final = "CLIPVisionLoader"
_UNET_LOADER: Final = "UNETLoader"
_TEXT_ENCODER_LOADER: Final = "CLIPLoader"
_VAE_LOADER: Final = "VAELoader"
_UPSCALE_MODEL_LOADER: Final = "UpscaleModelLoader"

#: embeddingはobject_info経由のノード選択肢ではなく専用エンドポイントで取得する。
#: CLIPTextEncodeはテキスト中の `embedding:<name>` を実行時に解決するだけで、
#: 選べる候補をobject_infoのノードスキーマとして持たないため。
_EMBEDDINGS_PATH: Final = "/embeddings"

#: アップロードした入力画像に付ける接頭辞。ComfyUIのinput配下で由来を判別できるようにする。
_UPLOAD_PREFIX: Final = "imagegen_"


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
        self._client_id = str(uuid.uuid4())
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(settings.timeout_seconds),
            transport=transport,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client_id(self) -> str:
        """このクライアントの識別子。WebSocket監視とprompt投入で同じ値を使う。"""
        return self._client_id

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
        names = _extract_option_names(payload, node=_CHECKPOINT_LOADER, field="ckpt_name")
        if not names:
            logger.warning(
                "ComfyUIからcheckpoint一覧を取得できませんでした (レスポンス形式が想定外)"
            )
        return names

    async def available_loras(self) -> tuple[str, ...]:
        """利用可能なLoRA名の一覧を取得する。

        LoRAが1つも置かれていない場合も空タプルになる。取得失敗と区別しないのは、
        どちらの場合も「指定できるLoRAがない」という結論が同じため。
        """
        payload = await self._get_json(f"/object_info/{_LORA_LOADER}")
        return _extract_option_names(payload, node=_LORA_LOADER, field="lora_name")

    async def available_controlnets(self) -> tuple[str, ...]:
        """利用可能なControlNetモデル名の一覧を取得する。

        LoRAと同じく、1つも置かれていない場合も空タプルになる。
        """
        payload = await self._get_json(f"/object_info/{_CONTROLNET_LOADER}")
        return _extract_option_names(payload, node=_CONTROLNET_LOADER, field="control_net_name")

    async def available_ipadapters(self) -> tuple[str, ...]:
        """利用可能なIPAdapterモデル名の一覧を取得する。

        IPAdapterはカスタムノード (ComfyUI_IPAdapter_plus) 由来のため、
        未導入ならノード自体が存在せず、応答は空になる。
        """
        payload = await self._get_json(f"/object_info/{_IPADAPTER_LOADER}")
        return _extract_option_names(payload, node=_IPADAPTER_LOADER, field="ipadapter_file")

    async def available_clip_visions(self) -> tuple[str, ...]:
        """利用可能なCLIP Visionモデル名の一覧を取得する。"""
        payload = await self._get_json(f"/object_info/{_CLIP_VISION_LOADER}")
        return _extract_option_names(payload, node=_CLIP_VISION_LOADER, field="clip_name")

    async def available_diffusion_models(self) -> tuple[str, ...]:
        """UNet単体で配布されているモデル名の一覧を取得する。

        DiT系モデル (Anima など) はUNetとtext encoder / VAEが別ファイルのため、
        checkpointとは別のフォルダ (models/diffusion_models/) を見る。
        """
        payload = await self._get_json(f"/object_info/{_UNET_LOADER}")
        return _extract_option_names(payload, node=_UNET_LOADER, field="unet_name")

    async def available_text_encoders(self) -> tuple[str, ...]:
        """単体で配布されているtext encoder名の一覧を取得する。"""
        payload = await self._get_json(f"/object_info/{_TEXT_ENCODER_LOADER}")
        return _extract_option_names(payload, node=_TEXT_ENCODER_LOADER, field="clip_name")

    async def available_vaes(self) -> tuple[str, ...]:
        """単体で配布されているVAE名の一覧を取得する。"""
        payload = await self._get_json(f"/object_info/{_VAE_LOADER}")
        return _extract_option_names(payload, node=_VAE_LOADER, field="vae_name")

    async def available_upscale_models(self) -> tuple[str, ...]:
        """アップスケールモデル (ESRGAN系) 名の一覧を取得する。

        1つも置かれていない場合も空タプルになる。
        """
        payload = await self._get_json(f"/object_info/{_UPSCALE_MODEL_LOADER}")
        return _extract_option_names(payload, node=_UPSCALE_MODEL_LOADER, field="model_name")

    async def available_embeddings(self) -> tuple[str, ...]:
        """利用可能なTextual Inversion embedding名 (拡張子なし) の一覧を取得する。

        ComfyUIの `/embeddings` は拡張子を除いたファイル名を返す
        (`os.path.splitext` で削られる)。prompt中の `embedding:<name>` の
        `<name>` と同じ形なので、そのまま突き合わせに使える。
        """
        names = await self._get_json_list(_EMBEDDINGS_PATH)
        return tuple(name for name in names if isinstance(name, str))

    async def upload_image(self, path: Path) -> str:
        """画像をComfyUIのinputへアップロードし、LoadImageで参照する名前を返す。

        ComfyUIのLoadImageはinput直下のファイルしか候補に出さないため、
        サブフォルダは使わない。名前は内容のダイジェストから決めるので、
        同じ画像を何度指定しても同じ名前に落ち着く。
        """
        try:
            data = path.read_bytes()
        except OSError as exc:
            # 呼び出し元が指定値を添えて投げ直せるよう、ここではファイル名だけを出す
            # (adapterは作業ルートを知らないため、絶対パスを丸められない)
            raise InvalidGenerationSpec(f"入力画像を読み込めません: {path.name}") from exc

        digest = hashlib.sha256(data).hexdigest()[:12]
        name = f"{_UPLOAD_PREFIX}{digest}_{path.name}"
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        try:
            response = await self._client.post(
                "/upload/image",
                files={"image": (name, data, media_type)},
                data={"overwrite": "true"},
            )
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIへ接続できません: {self._base_url} ({type(exc).__name__})"
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise WorkflowSubmissionError(
                f"入力画像のアップロードに失敗しました (status={response.status_code})"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WorkflowSubmissionError(
                "アップロードの応答を解釈できません (JSONではありません)"
            ) from exc

        uploaded = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(uploaded, str) or not uploaded:
            raise WorkflowSubmissionError("アップロードの応答にファイル名が含まれていません")

        logger.debug("uploaded source image: %s -> %s", path, uploaded)
        return uploaded

    async def submit(self, workflow: dict[str, Any]) -> str:
        """Workflowを投入し、prompt_idを返す。"""
        try:
            response = await self._client.post(
                "/prompt", json={"prompt": workflow, "client_id": self._client_id}
            )
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIへ接続できません: {self._base_url} ({type(exc).__name__})"
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise WorkflowSubmissionError(_format_submission_error(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise WorkflowSubmissionError(
                "ComfyUIの応答を解釈できません (JSONではありません)"
            ) from exc

        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise WorkflowSubmissionError(
                f"ComfyUIの応答に prompt_id が含まれていません: {payload!r}"
            )

        logger.info("workflow submitted: prompt_id=%s", prompt_id)
        return prompt_id

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
        use_websocket: bool = True,
    ) -> None:
        """実行完了まで待つ。

        監視方式はこのメソッド内に隠蔽する。WebSocketが使えない環境では
        history のポーリングへ自動的にフォールバックする。
        """
        limit = float(timeout if timeout is not None else self._settings.timeout_seconds)

        if use_websocket:
            try:
                await self._wait_via_websocket(prompt_id, limit)
            except (OSError, WebSocketException) as exc:
                logger.warning(
                    "WebSocket監視を使用できないためポーリングへ切り替えます (%s)",
                    type(exc).__name__,
                )
            else:
                return

        await self._wait_via_polling(prompt_id, limit, poll_interval)

    async def _wait_via_websocket(self, prompt_id: str, timeout: float) -> None:
        """WebSocketで実行完了を検知する。"""
        ws_url = _to_websocket_url(self._base_url, self._client_id)
        try:
            async with asyncio.timeout(timeout), websockets.connect(ws_url) as connection:
                async for raw in connection:
                    if not isinstance(raw, str):
                        continue
                    if _is_completion_message(raw, prompt_id):
                        return
        except TimeoutError as exc:
            raise GenerationTimeout(
                f"生成が制限時間 {timeout} 秒以内に完了しませんでした (prompt_id={prompt_id})"
            ) from exc

    async def _wait_via_polling(self, prompt_id: str, timeout: float, poll_interval: float) -> None:
        """historyのポーリングで実行完了を検知する。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            entry = await self._history_entry(prompt_id)
            if entry is not None:
                _raise_if_execution_failed(entry, prompt_id)
                if _is_completed(entry):
                    return
            if loop.time() >= deadline:
                raise GenerationTimeout(
                    f"生成が制限時間 {timeout} 秒以内に完了しませんでした (prompt_id={prompt_id})"
                )
            await asyncio.sleep(poll_interval)

    async def fetch_outputs(self, prompt_id: str) -> tuple[ImageRef, ...]:
        """生成された画像への参照を取得する。"""
        entry = await self._history_entry(prompt_id)
        if entry is None:
            raise OutputNotFound(f"履歴に prompt_id={prompt_id} の記録がありません")

        _raise_if_execution_failed(entry, prompt_id)

        images = _extract_images(entry)
        if not images:
            raise OutputNotFound(f"生成結果に画像が含まれていません (prompt_id={prompt_id})")
        return images

    async def download(self, ref: ImageRef) -> bytes:
        """ComfyUIのoutputから画像データを取得する。"""
        params = {
            "filename": ref.filename,
            "subfolder": ref.subfolder,
            "type": ref.type,
        }
        try:
            response = await self._client.get("/view", params=params)
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(
                f"ComfyUIへ接続できません: {self._base_url} ({type(exc).__name__})"
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise OutputNotFound(
                f"画像を取得できません (HTTP {response.status_code}): {ref.filename}"
            )
        return response.content

    async def _history_entry(self, prompt_id: str) -> dict[str, Any] | None:
        payload = await self._get_json(f"/history/{prompt_id}")
        entry = payload.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def _get_json(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        payload = await self._get_raw(path, timeout=timeout)
        if not isinstance(payload, dict):
            raise ComfyUIUnavailable(f"ComfyUIのレスポンス形式が想定外です: {self._base_url}{path}")
        return payload

    async def _get_json_list(self, path: str, *, timeout: float | None = None) -> list[Any]:
        payload = await self._get_raw(path, timeout=timeout)
        if not isinstance(payload, list):
            raise ComfyUIUnavailable(f"ComfyUIのレスポンス形式が想定外です: {self._base_url}{path}")
        return payload

    async def _get_raw(self, path: str, *, timeout: float | None = None) -> Any:
        try:
            response = await self._client.get(path, timeout=timeout or self._client.timeout)
            response.raise_for_status()
            return response.json()
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


def _extract_option_names(payload: dict[str, Any], *, node: str, field: str) -> tuple[str, ...]:
    """object_infoのレスポンスから、あるノードの選択肢一覧を取り出す。

    ComfyUIの応答は入れ子が深く欠損もありうるため、防御的に取り出す。
    選択肢の並びには2つの形があり、どちらで来るかはノードの定義側で決まる。

    - 旧来のノード (nodes.py 由来): ``[["a.safetensors", "b.safetensors"]]``
    - 新しい定義APIのノード (comfy_extras 由来): ``["COMBO", {"options": [...]}]``

    UpscaleModelLoader は後者で返る。前者だけを読むと選択肢を取りこぼす。
    """
    loader = payload.get(node)
    if not isinstance(loader, dict):
        return ()
    required = loader.get("input", {})
    if not isinstance(required, dict):
        return ()
    fields = required.get("required")
    if not isinstance(fields, dict):
        return ()
    entry = fields.get(field)
    if not isinstance(entry, list) or not entry:
        return ()
    candidates = entry[0]
    if isinstance(candidates, list):
        return tuple(name for name in candidates if isinstance(name, str))
    if len(entry) > 1 and isinstance(entry[1], dict):
        options = entry[1].get("options")
        if isinstance(options, list):
            return tuple(name for name in options if isinstance(name, str))
    return ()


def _to_websocket_url(base_url: str, client_id: str) -> str:
    """HTTPのベースURLからWebSocket監視用のURLを組み立てる。"""
    scheme, _, rest = base_url.partition("://")
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{rest}/ws?{urlencode({'clientId': client_id})}"


def _is_completion_message(raw: str, prompt_id: str) -> bool:
    """WebSocketメッセージが対象promptの完了通知かを判定する。

    ComfyUIは実行終了時に node=None の executing メッセージを送る。
    """
    try:
        message = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(message, dict):
        return False

    data = message.get("data")
    if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
        return False

    if message.get("type") == "execution_error":
        raise GenerationFailed(_execution_error_message(data, prompt_id))
    return message.get("type") == "executing" and data.get("node") is None


def _is_completed(entry: dict[str, Any]) -> bool:
    status = entry.get("status")
    if not isinstance(status, dict):
        return False
    return status.get("completed") is True or status.get("status_str") == "success"


def _raise_if_execution_failed(entry: dict[str, Any], prompt_id: str) -> None:
    status = entry.get("status")
    if not isinstance(status, dict) or status.get("status_str") != "error":
        return

    detail = "詳細不明"
    messages = status.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "execution_error"
                and isinstance(item[1], dict)
            ):
                detail = _execution_error_message(item[1], prompt_id)
                break
    raise GenerationFailed(f"ComfyUIでの実行が失敗しました (prompt_id={prompt_id}): {detail}")


def _execution_error_message(data: dict[str, Any], prompt_id: str) -> str:
    message = data.get("exception_message")
    node = data.get("node_type") or data.get("node_id")
    if isinstance(message, str) and message:
        return f"{message} (node={node})" if node else message
    return f"実行エラー (prompt_id={prompt_id})"


def _extract_images(entry: dict[str, Any]) -> tuple[ImageRef, ...]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return ()

    images: list[ImageRef] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for image in node_output.get("images", []):
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if not isinstance(filename, str):
                continue
            images.append(
                ImageRef(
                    filename=filename,
                    subfolder=str(image.get("subfolder", "")),
                    type=str(image.get("type", "output")),
                )
            )
    return tuple(images)


def _format_submission_error(response: httpx.Response) -> str:
    """ComfyUIの拒否レスポンスから、原因を特定できるメッセージを組み立てる。"""
    base = f"ComfyUIがWorkflowを受け付けませんでした (HTTP {response.status_code})"
    try:
        payload = response.json()
    except ValueError:
        return base
    if not isinstance(payload, dict):
        return base

    details: list[str] = []
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            details.append(message)

    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id, node_error in node_errors.items():
            if not isinstance(node_error, dict):
                continue
            for item in node_error.get("errors", []):
                if isinstance(item, dict) and isinstance(item.get("message"), str):
                    details.append(f"node {node_id}: {item['message']}")

    return f"{base}: {' / '.join(details)}" if details else base


__all__ = ["ComfyUIClient"]
