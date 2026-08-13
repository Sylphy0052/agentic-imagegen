"""ComfyUI Clientのテスト。実ComfyUIへは接続せずMockTransportで代替する。"""

from typing import Any

import httpx
import pytest

from agentic_imagegen.adapters.comfyui.client import ComfyUIClient
from agentic_imagegen.config import Settings
from agentic_imagegen.errors import ComfyUIUnavailable

SYSTEM_STATS: dict[str, Any] = {
    "system": {
        "os": "posix",
        "comfyui_version": "0.3.40",
        "python_version": "3.12.13",
    },
    "devices": [{"name": "cpu", "type": "cpu", "vram_total": 0}],
}

OBJECT_INFO: dict[str, Any] = {
    "CheckpointLoaderSimple": {
        "input": {
            "required": {
                "ckpt_name": [["v1-5-pruned-emaonly.safetensors", "sd15/anything.safetensors"], {}]
            }
        }
    }
}


def _settings(**overrides: Any) -> Settings:
    from pathlib import Path

    defaults: dict[str, Any] = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "max_width": 2048,
        "max_height": 2048,
        "max_pixels": 4194304,
        "max_batch": 4,
        "timeout_seconds": 5,
        "output_root": Path("outputs"),
    }
    return Settings(**{**defaults, **overrides})


def _client(handler: Any, **settings_overrides: Any) -> ComfyUIClient:
    return ComfyUIClient(_settings(**settings_overrides), transport=httpx.MockTransport(handler))


async def test_health_returns_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=SYSTEM_STATS)

    async with _client(handler) as client:
        status = await client.health()

    assert status.base_url == "http://127.0.0.1:8188"
    assert status.comfyui_version == "0.3.40"
    assert status.devices == ("cpu",)
    assert str(requests[0].url) == "http://127.0.0.1:8188/system_stats"


async def test_health_tolerates_missing_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        status = await client.health()

    assert status.comfyui_version is None
    assert status.devices == ()


async def test_health_connection_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable, match="接続できません"):
            await client.health()


async def test_health_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.health()


@pytest.mark.parametrize("status_code", [404, 500, 503])
async def test_health_http_error_raises(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="error")

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable, match=str(status_code)):
            await client.health()


async def test_health_invalid_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.health()


async def test_base_url_trailing_slash_is_normalized() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=SYSTEM_STATS)

    async with _client(handler, comfyui_base_url="http://127.0.0.1:9000") as client:
        await client.health()

    assert seen == ["http://127.0.0.1:9000/system_stats"]


async def test_available_checkpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CheckpointLoaderSimple"
        return httpx.Response(200, json=OBJECT_INFO)

    async with _client(handler) as client:
        checkpoints = await client.available_checkpoints()

    assert checkpoints == (
        "v1-5-pruned-emaonly.safetensors",
        "sd15/anything.safetensors",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"CheckpointLoaderSimple": {}},
        {"CheckpointLoaderSimple": {"input": {"required": {}}}},
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": []}}}},
        {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": ["x"]}}}},
    ],
)
async def test_available_checkpoints_unexpected_shape_returns_empty(
    payload: dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        assert await client.available_checkpoints() == ()


async def test_available_checkpoints_unreachable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.available_checkpoints()


CONTROLNET_OBJECT_INFO: dict[str, Any] = {
    "ControlNetLoader": {
        "input": {
            "required": {
                "control_net_name": [
                    ["control_v11p_sd15_canny_fp16.safetensors", "control_v11f1p_sd15_depth.pth"],
                    {},
                ]
            }
        }
    }
}


async def test_available_controlnets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/ControlNetLoader"
        return httpx.Response(200, json=CONTROLNET_OBJECT_INFO)

    async with _client(handler) as client:
        names = await client.available_controlnets()

    assert names == (
        "control_v11p_sd15_canny_fp16.safetensors",
        "control_v11f1p_sd15_depth.pth",
    )


async def test_available_controlnets_empty_when_none_installed() -> None:
    """ControlNetモデルが1つも置かれていない場合も空タプルになる。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ControlNetLoader": {"input": {"required": {}}}})

    async with _client(handler) as client:
        assert await client.available_controlnets() == ()


IPADAPTER_OBJECT_INFO = {
    "IPAdapterModelLoader": {
        "input": {
            "required": {
                "ipadapter_file": [
                    ["ip-adapter-plus_sd15.safetensors", "ip-adapter_sd15_light_v11.bin"],
                    {},
                ]
            }
        }
    }
}

CLIP_VISION_OBJECT_INFO = {
    "CLIPVisionLoader": {
        "input": {"required": {"clip_name": [["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"], {}]}}
    }
}


async def test_available_ipadapters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/IPAdapterModelLoader"
        return httpx.Response(200, json=IPADAPTER_OBJECT_INFO)

    async with _client(handler) as client:
        names = await client.available_ipadapters()

    assert names == (
        "ip-adapter-plus_sd15.safetensors",
        "ip-adapter_sd15_light_v11.bin",
    )


async def test_available_ipadapters_empty_when_custom_node_absent() -> None:
    """IPAdapterはカスタムノード由来のため、未導入ならノード自体が存在しない。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        assert await client.available_ipadapters() == ()


async def test_available_clip_visions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CLIPVisionLoader"
        return httpx.Response(200, json=CLIP_VISION_OBJECT_INFO)

    async with _client(handler) as client:
        assert await client.available_clip_visions() == (
            "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        )


DIFFUSION_MODEL_OBJECT_INFO = {
    "UNETLoader": {
        "input": {
            "required": {
                "unet_name": [["hassakuAnima_v13_int8.safetensors"], {}],
                "weight_dtype": [["default", "fp8_e4m3fn"], {}],
            }
        }
    }
}

TEXT_ENCODER_OBJECT_INFO = {
    "CLIPLoader": {
        "input": {
            "required": {
                "clip_name": [["qwen_3_06b_base.safetensors"], {}],
                "type": [["stable_diffusion", "flux2"], {}],
            }
        }
    }
}

VAE_OBJECT_INFO = {
    "VAELoader": {"input": {"required": {"vae_name": [["qwen_image_vae.safetensors"], {}]}}}
}

#: UpscaleModelLoaderは新しいノード定義API由来で、選択肢がCOMBO形式で返る
#: (実機のComfyUIで確認済み)。旧来のローダとは形が違う。
UPSCALE_MODEL_OBJECT_INFO = {
    "UpscaleModelLoader": {
        "input": {
            "required": {
                "model_name": [
                    "COMBO",
                    {"multiselect": False, "options": ["RealESRGAN_x4plus_anime_6B.pth"]},
                ]
            }
        }
    }
}


async def test_available_diffusion_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/UNETLoader"
        return httpx.Response(200, json=DIFFUSION_MODEL_OBJECT_INFO)

    async with _client(handler) as client:
        assert await client.available_diffusion_models() == ("hassakuAnima_v13_int8.safetensors",)


async def test_available_text_encoders() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CLIPLoader"
        return httpx.Response(200, json=TEXT_ENCODER_OBJECT_INFO)

    async with _client(handler) as client:
        assert await client.available_text_encoders() == ("qwen_3_06b_base.safetensors",)


async def test_available_vaes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/VAELoader"
        return httpx.Response(200, json=VAE_OBJECT_INFO)

    async with _client(handler) as client:
        assert await client.available_vaes() == ("qwen_image_vae.safetensors",)


async def test_available_upscale_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/UpscaleModelLoader"
        return httpx.Response(200, json=UPSCALE_MODEL_OBJECT_INFO)

    async with _client(handler) as client:
        assert await client.available_upscale_models() == ("RealESRGAN_x4plus_anime_6B.pth",)


async def test_available_upscale_models_empty_when_none_placed() -> None:
    """1つも置かれていない環境でもエラーにせず空で返す。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        assert await client.available_upscale_models() == ()


@pytest.mark.parametrize(
    "payload",
    [
        # COMBO形式だがメタデータが無い
        {"UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO"]}}}},
        # メタデータが辞書でない
        {"UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", "x"]}}}},
        # optionsがリストでない
        {
            "UpscaleModelLoader": {
                "input": {"required": {"model_name": ["COMBO", {"options": None}]}}
            }
        },
        {
            "UpscaleModelLoader": {
                "input": {"required": {"model_name": ["COMBO", {"options": {"a": 1}}]}}
            }
        },
        # COMBO以外の型がoptionsという名前のメタデータを持っていても選択肢とは見ない
        {
            "UpscaleModelLoader": {
                "input": {"required": {"model_name": ["STRING", {"options": ["x.pth"]}]}}
            }
        },
    ],
)
async def test_available_upscale_models_unexpected_shape_returns_empty(
    payload: dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        assert await client.available_upscale_models() == ()


async def test_available_upscale_models_skips_non_string_options() -> None:
    """optionsに文字列以外が混ざっていても落とさず、文字列だけを返す。"""
    payload = {
        "UpscaleModelLoader": {
            "input": {
                "required": {"model_name": ["COMBO", {"options": ["a.pth", 1, None, "b.pth"]}]}
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        assert await client.available_upscale_models() == ("a.pth", "b.pth")


async def test_available_upscale_models_with_combo_options_empty() -> None:
    """モデルを置いていない環境ではoptionsが空で返る。"""
    payload = {
        "UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", {"options": []}]}}}
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        assert await client.available_upscale_models() == ()


async def test_available_diffusion_models_empty_when_none_placed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with _client(handler) as client:
        assert await client.available_diffusion_models() == ()


async def test_available_embeddings() -> None:
    """`/embeddings` はobject_info経由ではなく専用エンドポイントで、拡張子を含まない。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        return httpx.Response(200, json=["easynegative", "badhandv4"])

    async with _client(handler) as client:
        assert await client.available_embeddings() == ("easynegative", "badhandv4")


async def test_available_embeddings_empty_when_none_placed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        assert await client.available_embeddings() == ()


async def test_available_embeddings_unreachable_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.available_embeddings()


async def test_available_embeddings_unexpected_shape_raises() -> None:
    """object_info系のようにdictを返す想定外レスポンスも検出する。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    async with _client(handler) as client:
        with pytest.raises(ComfyUIUnavailable):
            await client.available_embeddings()
