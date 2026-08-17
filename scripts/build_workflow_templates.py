"""Workflowテンプレートの派生形を機械的に組み立てる。

`workflows/txt2img.json` を唯一の手書きベースとし、そこから
img2img / LoRA / hires fix の組み合わせを生成する。

    txt2img.json  (ベース: 人間がComfyUI GUIで作成しAPI形式で書き出したもの)
      ├─ img2img            EmptyLatentImage -> LoadImage + VAEEncode
      ├─ *_lora             CheckpointLoader の後に LoraLoader を3段
      ├─ *_hires            KSampler の後に LatentUpscaleBy + 2段目 KSampler
      ├─ *_vae              CheckpointLoaderのVAE出力 (2番) を参照する全ノードを
      │                     VAELoader へ差し替え (checkpoint系のみ、DiT系は対象外)
      └─ txt2img_unet       CheckpointLoader -> UNETLoader + CLIPLoader + VAELoader

組み合わせが増えても手で書かないのは、ノード参照を間違えても
「形は正しいまま意味だけ壊れる」ためである (実際に一度踏んでいる)。
生成後に参照整合性と、既存ノードを潰していないことを検査する。

生成しうるテンプレート名と、それを構成する軸の並びは
`agentic_imagegen.workflows.axes` が一元管理する。`build_all()` はその列挙結果を
辿り、軸ごとのグラフ合成関数 (`AXIS_GRAPH_BUILDERS`) を引いて順に適用するだけである。
軸を1本足すときにここで書くのは、そのグラフ合成関数と `AXIS_GRAPH_BUILDERS` への
登録だけでよい (テンプレート名の列挙自体はaxes側で決まる)。

使い方:

    uv run python scripts/build_workflow_templates.py            # 生成
    uv run python scripts/build_workflow_templates.py --check    # 差分がないか確認するだけ
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_imagegen.workflows.axes import (
    AXIS_BETA57,
    AXIS_CONTROLNET,
    AXIS_CONTROLNET_RAW,
    AXIS_HIRES,
    AXIS_HIRES_MODEL,
    AXIS_IPADAPTER,
    AXIS_LORA,
    AXIS_UNET,
    AXIS_VAE,
    Task,
    axes_in_build_order,
    iter_template_specs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
BASE_NAME = "txt2img"

# ベーステンプレートのノードID
CHECKPOINT = "4"
KSAMPLER = "3"
POSITIVE = "6"
NEGATIVE = "7"
VAE_DECODE = "8"
EMPTY_LATENT = "5"

# 派生で使うノードID。用途ごとに帯を分け、衝突しないようにする
LORA_IDS = ("10", "11", "12")
IMG2IMG_LOAD = "10"
IMG2IMG_VAE_ENCODE = "11"
IMG2IMG_LORA_IDS = ("20", "21", "22")
HIRES_UPSCALE = "30"
HIRES_KSAMPLER = "31"
HIRES_MODEL_DECODE = "32"
HIRES_MODEL_LOADER = "33"
HIRES_MODEL_UPSCALE = "34"
HIRES_MODEL_RESIZE = "35"
HIRES_MODEL_ENCODE = "36"
CONTROL_LOAD_IMAGE = "40"
CONTROL_PREPROCESSOR = "41"
CONTROL_LOADER = "42"
CONTROL_APPLY = "43"
REFERENCE_LOAD_IMAGE = "50"
REFERENCE_LOADER = "51"
REFERENCE_CLIP_VISION = "52"
REFERENCE_APPLY = "53"
UNET_LOADER = "60"
UNET_CLIP_LOADER = "61"
UNET_VAE_LOADER = "62"
CLIP_SKIP = "70"
EXTERNAL_VAE_LOADER = "80"
# beta57 (KSampler -> SamplerCustomAdvanced) の1段目
BETA57_NOISE = "90"
BETA57_SAMPLER_SELECT = "91"
BETA57_SIGMAS = "92"
BETA57_GUIDER = "93"
BETA57_SAMPLER = "94"
# beta57 の2段目 (hires fix の描き足し)。denoise を SplitSigmasDenoise で表す
BETA57_HIRES_NOISE = "95"
BETA57_HIRES_SAMPLER_SELECT = "96"
BETA57_HIRES_SIGMAS = "97"
BETA57_HIRES_SPLIT = "98"
BETA57_HIRES_GUIDER = "99"
BETA57_HIRES_SAMPLER = "100"

#: 配布元が beta57 と呼ぶノイズスケジュールの実体。beta分布のalpha / beta。
#: RES4LYFの beta57 と同じ値で、ComfyUI標準の BetaSamplingScheduler で表せる。
BETA57_ALPHA = 0.5
BETA57_BETA = 0.7

DEFAULT_LORA = "add_detail.safetensors"
DEFAULT_SOURCE_IMAGE = "example.png"
DEFAULT_CONTROLNET = "control_v11p_sd15_canny_fp16.safetensors"
DEFAULT_IPADAPTER = "ip-adapter-plus_sd15.safetensors"
DEFAULT_CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
DEFAULT_UPSCALE_MODEL = "RealESRGAN_x4plus_anime_6B.pth"
DEFAULT_UNET = "hassakuAnima_v13_int8.safetensors"
DEFAULT_TEXT_ENCODER = "qwen_3_06b_base.safetensors"
DEFAULT_VAE = "qwen_image_vae.safetensors"
#: checkpoint系で外部VAEへ差し替える際の既定値。SD1.5汎用のVAEを使う。
DEFAULT_EXTERNAL_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"

#: 出力スロット数。範囲外の参照を検出するために使う
OUTPUT_COUNTS = {
    "CheckpointLoaderSimple": 3,
    "LoadImage": 2,
    "LoraLoader": 2,
    "VAEEncode": 1,
    "LatentUpscaleBy": 1,
    "KSampler": 1,
    "CLIPTextEncode": 1,
    "VAEDecode": 1,
    "EmptyLatentImage": 1,
    "SaveImage": 0,
    "Canny": 1,
    "ControlNetLoader": 1,
    "ControlNetApplyAdvanced": 2,
    "IPAdapterModelLoader": 1,
    "CLIPVisionLoader": 1,
    "UNETLoader": 1,
    "CLIPLoader": 1,
    "VAELoader": 1,
    "IPAdapterAdvanced": 1,
    "CLIPSetLastLayer": 1,
    "UpscaleModelLoader": 1,
    "ImageUpscaleWithModel": 1,
    "ImageScaleBy": 1,
    "RandomNoise": 1,
    "KSamplerSelect": 1,
    "BetaSamplingScheduler": 1,
    "SplitSigmasDenoise": 2,
    "CFGGuider": 1,
    "SamplerCustomAdvanced": 2,
}

Graph = dict[str, dict[str, Any]]


def to_img2img(graph: Graph) -> Graph:
    """EmptyLatentImage を LoadImage + VAEEncode へ置き換える。"""
    graph = copy.deepcopy(graph)
    del graph[EMPTY_LATENT]

    graph[IMG2IMG_LOAD] = {
        "class_type": "LoadImage",
        "inputs": {"image": DEFAULT_SOURCE_IMAGE},
    }
    graph[IMG2IMG_VAE_ENCODE] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": [IMG2IMG_LOAD, 0], "vae": [CHECKPOINT, 2]},
    }
    graph[KSAMPLER]["inputs"]["latent_image"] = [IMG2IMG_VAE_ENCODE, 0]
    # denoise は img2img で意味を持つ。Specから注入されるが既定は控えめにしておく
    graph[KSAMPLER]["inputs"]["denoise"] = 0.6
    return graph


def to_separate_loaders(graph: Graph) -> Graph:
    """CheckpointLoaderSimple を UNETLoader + CLIPLoader + VAELoader へ置き換える。

    DiT系のモデル (Anima など) はUNet単体で配布され、text encoderとVAEを同梱しない。
    1ファイルから MODEL / CLIP / VAE の3つを取り出す前提が崩れるため、
    ローダーそのものを分ける。
    """
    graph = copy.deepcopy(graph)
    del graph[CHECKPOINT]

    graph[UNET_LOADER] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": DEFAULT_UNET, "weight_dtype": "default"},
    }
    graph[UNET_CLIP_LOADER] = {
        "class_type": "CLIPLoader",
        # typeはstate dictからComfyUIが判別するため、既定値のままでAnimaも読める
        "inputs": {"clip_name": DEFAULT_TEXT_ENCODER, "type": "stable_diffusion"},
    }
    graph[UNET_VAE_LOADER] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": DEFAULT_VAE},
    }

    graph[KSAMPLER]["inputs"]["model"] = [UNET_LOADER, 0]
    # DiT系のtext encoderはCLIPではなくQwen3のため、CLIPSetLastLayerを通さない。
    # stop_at_clip_layer=-1でも素通しにならず条件付けが壊れ、出力が単色や人型の
    # 崩れた塊になる (2026-08-16に実機で確認)。CLIPTextEncodeはCLIPLoaderへ直結する
    del graph[CLIP_SKIP]
    for node in graph.values():
        if node["inputs"].get("clip") == [CLIP_SKIP, 0]:
            node["inputs"]["clip"] = [UNET_CLIP_LOADER, 0]
    graph[VAE_DECODE]["inputs"]["vae"] = [UNET_VAE_LOADER, 0]
    if IMG2IMG_VAE_ENCODE in graph:
        # img2imgでは入力画像をVAEEncodeする側もCheckpointLoaderのVAEを見ている
        graph[IMG2IMG_VAE_ENCODE]["inputs"]["vae"] = [UNET_VAE_LOADER, 0]

    # 既定値もDiT系へ寄せる (Specから注入されるが、テンプレート単体で見たときに
    # SD1.5向けの値が残っていると誤解を招く)
    graph[KSAMPLER]["inputs"]["steps"] = 30
    graph[KSAMPLER]["inputs"]["cfg"] = 4.0
    graph[KSAMPLER]["inputs"]["scheduler"] = "simple"
    if EMPTY_LATENT in graph:
        # img2imgは入力画像の解像度をそのまま使うためEmptyLatentImage自体が無い
        graph[EMPTY_LATENT]["inputs"]["width"] = 1024
        graph[EMPTY_LATENT]["inputs"]["height"] = 1024
    return graph


def with_lora_chain(graph: Graph, node_ids: tuple[str, ...]) -> Graph:
    """MODELの供給元の後に LoraLoader を直列で挟む。

    checkpoint系は MODEL / CLIP とも CheckpointLoader の出力 (0 / 1) から来るが、
    DiT系は UNETLoader と CLIPLoader に分かれ、どちらも出力0から来る。
    CLIP側の受け手も違う (checkpoint系は CLIPSetLastLayer、DiT系は
    CLIPTextEncode 2つ。DiT系はQwen3のため CLIPSetLastLayer を通さない)。
    """
    graph = copy.deepcopy(graph)
    for node_id in node_ids:
        if node_id in graph:
            raise ValueError(f"LoRA用のノードID {node_id} が既に使われている")

    separate = UNET_LOADER in graph
    model_source = [UNET_LOADER, 0] if separate else [CHECKPOINT, 0]
    clip_source = [UNET_CLIP_LOADER, 0] if separate else [CHECKPOINT, 1]

    for node_id in node_ids:
        graph[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": DEFAULT_LORA,
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": model_source,
                "clip": clip_source,
            },
        }
        model_source = [node_id, 0]
        clip_source = [node_id, 1]

    last = node_ids[-1]
    graph[KSAMPLER]["inputs"]["model"] = [last, 0]
    if separate:
        # CLIPTextEncodeがCLIPLoaderへ直結しているため、そこを最終段へ向け直す。
        # 1段目のLoraLoader自身もCLIPLoaderを見ているため、チェーンは対象外にする
        for node_id, node in graph.items():
            if node_id in node_ids:
                continue
            if node["inputs"].get("clip") == [UNET_CLIP_LOADER, 0]:
                node["inputs"]["clip"] = [last, 1]
    else:
        # CLIPTextEncodeは変わらずCLIPSetLastLayer経由。CLIPSetLastLayerの供給元を
        # LoRAチェーンの最終段へ差し替える (LoRA適用後のCLIPに対して層を打ち切るため)
        graph[CLIP_SKIP]["inputs"]["clip"] = [last, 1]
    # VAE は LoraLoader を通らないため元のローダー直結のまま
    return graph


def with_hires_fix(graph: Graph) -> Graph:
    """KSampler の後に LatentUpscaleBy と2段目の KSampler を挟む。"""
    graph = copy.deepcopy(graph)
    for node_id in (HIRES_UPSCALE, HIRES_KSAMPLER):
        if node_id in graph:
            raise ValueError(f"hires用のノードID {node_id} が既に使われている")

    first = graph[KSAMPLER]["inputs"]
    graph[HIRES_UPSCALE] = {
        "class_type": "LatentUpscaleBy",
        "inputs": {
            "samples": [KSAMPLER, 0],
            "upscale_method": "nearest-exact",
            "scale_by": 1.5,
        },
    }
    graph[HIRES_KSAMPLER] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": first["seed"],
            "steps": first["steps"],
            "cfg": first["cfg"],
            "sampler_name": first["sampler_name"],
            "scheduler": first["scheduler"],
            # 拡大後は元の絵を保ちつつ描き足すため、denoiseを下げる
            "denoise": 0.5,
            "model": first["model"],
            "positive": first["positive"],
            "negative": first["negative"],
            "latent_image": [HIRES_UPSCALE, 0],
        },
    }
    graph[VAE_DECODE]["inputs"]["samples"] = [HIRES_KSAMPLER, 0]
    return graph


def with_hires_model_fix(graph: Graph) -> Graph:
    """KSampler の後にアップスケールモデルでの拡大と2段目の KSampler を挟む。

    latent拡大 (with_hires_fix) との違いは拡大の場所だけで、一度pixelへ戻す。

        KSampler -> VAEDecode -> ImageUpscaleWithModel -> ImageScaleBy
                 -> VAEEncode -> 2段目のKSampler -> 既存のVAEDecode

    ImageScaleBy を必ず挟むのは、アップスケールモデルの倍率が固定 (4x など) で
    要求された倍率と一致しないためである。等倍のときは scale_by へ1.0が入る。

    VAEDecode / VAEEncode の VAE は既存の VAEDecode と同じ供給元から取る。
    DiT系 (VAELoader を分けた構成) でも取り違えないようにするため。
    """
    graph = copy.deepcopy(graph)
    added = (
        HIRES_MODEL_DECODE,
        HIRES_MODEL_LOADER,
        HIRES_MODEL_UPSCALE,
        HIRES_MODEL_RESIZE,
        HIRES_MODEL_ENCODE,
        HIRES_KSAMPLER,
    )
    for node_id in added:
        if node_id in graph:
            raise ValueError(f"hires (model) 用のノードID {node_id} が既に使われている")

    first = graph[KSAMPLER]["inputs"]
    vae = graph[VAE_DECODE]["inputs"]["vae"]

    graph[HIRES_MODEL_DECODE] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [KSAMPLER, 0], "vae": vae},
    }
    graph[HIRES_MODEL_LOADER] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": DEFAULT_UPSCALE_MODEL},
    }
    graph[HIRES_MODEL_UPSCALE] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": [HIRES_MODEL_LOADER, 0],
            "image": [HIRES_MODEL_DECODE, 0],
        },
    }
    graph[HIRES_MODEL_RESIZE] = {
        "class_type": "ImageScaleBy",
        "inputs": {
            "image": [HIRES_MODEL_UPSCALE, 0],
            "upscale_method": "nearest-exact",
            # 4xのモデルで2倍が欲しい場合の既定値。Specから注入される
            "scale_by": 0.5,
        },
    }
    graph[HIRES_MODEL_ENCODE] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": [HIRES_MODEL_RESIZE, 0], "vae": vae},
    }
    graph[HIRES_KSAMPLER] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": first["seed"],
            "steps": first["steps"],
            "cfg": first["cfg"],
            "sampler_name": first["sampler_name"],
            "scheduler": first["scheduler"],
            # 拡大の時点で線が補間されるため、latent拡大より低めでよい
            "denoise": 0.4,
            "model": first["model"],
            "positive": first["positive"],
            "negative": first["negative"],
            "latent_image": [HIRES_MODEL_ENCODE, 0],
        },
    }
    graph[VAE_DECODE]["inputs"]["samples"] = [HIRES_KSAMPLER, 0]
    return graph


def _build_controlnet(graph: Graph, *, preprocess: bool) -> Graph:
    """CLIPTextEncode と KSampler の間に ControlNet を挟む。

    `preprocess` が True なら control画像を Canny で線画へ変換してから
    ControlNetApplyAdvanced へ渡し、False なら LoadImage の出力をそのまま渡す
    (前処理済みの制御画像を使う経路)。

    ApplyAdvanced は positive / negative の両方を返すため、KSamplerの
    2つの入力をどちらもここから受け直す。
    """
    graph = copy.deepcopy(graph)
    used_ids = [CONTROL_LOAD_IMAGE, CONTROL_LOADER, CONTROL_APPLY]
    if preprocess:
        used_ids.append(CONTROL_PREPROCESSOR)
    for node_id in used_ids:
        if node_id in graph:
            raise ValueError(f"ControlNet用のノードID {node_id} が既に使われている")

    graph[CONTROL_LOAD_IMAGE] = {
        "class_type": "LoadImage",
        "inputs": {"image": DEFAULT_SOURCE_IMAGE},
    }
    if preprocess:
        graph[CONTROL_PREPROCESSOR] = {
            "class_type": "Canny",
            "inputs": {
                "image": [CONTROL_LOAD_IMAGE, 0],
                "low_threshold": 0.4,
                "high_threshold": 0.8,
            },
        }
    graph[CONTROL_LOADER] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": DEFAULT_CONTROLNET},
    }
    graph[CONTROL_APPLY] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": graph[KSAMPLER]["inputs"]["positive"],
            "negative": graph[KSAMPLER]["inputs"]["negative"],
            "control_net": [CONTROL_LOADER, 0],
            "image": [CONTROL_PREPROCESSOR if preprocess else CONTROL_LOAD_IMAGE, 0],
            "strength": 1.0,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    graph[KSAMPLER]["inputs"]["positive"] = [CONTROL_APPLY, 0]
    graph[KSAMPLER]["inputs"]["negative"] = [CONTROL_APPLY, 1]
    return graph


def with_controlnet(graph: Graph) -> Graph:
    """control画像を Canny で線画へ変換してから ControlNet へ渡す。"""
    return _build_controlnet(graph, preprocess=True)


def with_controlnet_raw(graph: Graph) -> Graph:
    """前処理済みの control画像をそのまま ControlNet へ渡す。"""
    return _build_controlnet(graph, preprocess=False)


def with_ipadapter(graph: Graph) -> Graph:
    """KSamplerが受け取るMODELを IPAdapterAdvanced 経由に差し替える。

    参照画像はCLIP Visionで特徴量へ落としてからモデルへ適用する。
    ControlNetがpositive / negativeを差し替えるのに対し、こちらはmodelだけを
    差し替えるため、両方を同時にかけても干渉しない。
    """
    graph = copy.deepcopy(graph)
    for node_id in (
        REFERENCE_LOAD_IMAGE,
        REFERENCE_LOADER,
        REFERENCE_CLIP_VISION,
        REFERENCE_APPLY,
    ):
        if node_id in graph:
            raise ValueError(f"IPAdapter用のノードID {node_id} が既に使われている")

    graph[REFERENCE_LOAD_IMAGE] = {
        "class_type": "LoadImage",
        "inputs": {"image": DEFAULT_SOURCE_IMAGE},
    }
    graph[REFERENCE_LOADER] = {
        "class_type": "IPAdapterModelLoader",
        "inputs": {"ipadapter_file": DEFAULT_IPADAPTER},
    }
    graph[REFERENCE_CLIP_VISION] = {
        "class_type": "CLIPVisionLoader",
        "inputs": {"clip_name": DEFAULT_CLIP_VISION},
    }
    graph[REFERENCE_APPLY] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
            # LoRAチェーンがある場合はその末尾を受ける
            "model": graph[KSAMPLER]["inputs"]["model"],
            "ipadapter": [REFERENCE_LOADER, 0],
            "image": [REFERENCE_LOAD_IMAGE, 0],
            "clip_vision": [REFERENCE_CLIP_VISION, 0],
            "weight": 1.0,
            "weight_type": "linear",
            # 参照画像は1枚のみ扱うため、埋め込みの合成方法は既定のままにする
            "combine_embeds": "concat",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
        },
    }
    graph[KSAMPLER]["inputs"]["model"] = [REFERENCE_APPLY, 0]
    return graph


def with_external_vae(graph: Graph) -> Graph:
    """CheckpointLoaderSimple の VAE 出力を参照している全ノードを VAELoader へ差し替える。

    checkpoint同梱のVAEではなく、色褪せ・眠い線を避けるために使う外部VAE
    (vae-ft-mse-840000 / klF8Anime2VAE など) を使う版を作る。

    差し替え対象を `VAEDecode` / `VAEEncode` と決め打ちせず、`[CHECKPOINT, 2]`
    (CheckpointLoaderSimpleのVAE出力) を参照している入力を機械的に走査するのは、
    `with_hires_model_fix` を先にかけたグラフでは増えたVAEDecode / VAEEncodeも
    同じ参照を持っており、決め打ちだと拾い漏れるため。

    DiT系 (`to_separate_loaders` 済みのグラフ) は CheckpointLoaderSimple 自体が
    存在せず、この関数の対象外 (既に独自のVAELoaderルートを持つ)。
    """
    graph = copy.deepcopy(graph)
    if EXTERNAL_VAE_LOADER in graph:
        raise ValueError(f"外部VAE用のノードID {EXTERNAL_VAE_LOADER} が既に使われている")
    if CHECKPOINT not in graph:
        raise ValueError("with_external_vae は CheckpointLoaderSimple を持つグラフにのみ適用できる")

    graph[EXTERNAL_VAE_LOADER] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": DEFAULT_EXTERNAL_VAE},
    }
    for node in graph.values():
        for key, value in node["inputs"].items():
            if value == [CHECKPOINT, 2]:
                node["inputs"][key] = [EXTERNAL_VAE_LOADER, 0]
    return graph


def _beta57_stage(
    graph: Graph,
    *,
    ksampler_id: str,
    noise_id: str,
    sampler_select_id: str,
    sigmas_id: str,
    guider_id: str,
    sampler_id: str,
    split_id: str | None,
) -> None:
    """1つの KSampler を SamplerCustomAdvanced 一式へ置き換える (graphを破壊的に更新)。

    KSampler が1ノードで担っていたものを、ComfyUI標準のノードへ分解する。

        RandomNoise            <- seed
        KSamplerSelect         <- sampler_name
        BetaSamplingScheduler  <- steps / alpha=0.5 / beta=0.7 (= beta57)
        CFGGuider              <- cfg / positive / negative / model
        SamplerCustomAdvanced  <- 上の4つを束ねて sampling する

    `split_id` を渡すとhires fixの2段目として組み立て、`SplitSigmasDenoise` を
    挟んで KSampler の `denoise` に相当する部分だけを取り出す。
    """
    previous = graph[ksampler_id]["inputs"]
    for node_id in (noise_id, sampler_select_id, sigmas_id, guider_id, sampler_id, split_id):
        if node_id is not None and node_id in graph:
            raise ValueError(f"beta57用のノードID {node_id} が既に使われている")

    graph[noise_id] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": previous["seed"]},
    }
    graph[sampler_select_id] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": previous["sampler_name"]},
    }
    graph[sigmas_id] = {
        "class_type": "BetaSamplingScheduler",
        "inputs": {
            "model": previous["model"],
            "steps": previous["steps"],
            "alpha": BETA57_ALPHA,
            "beta": BETA57_BETA,
        },
    }
    graph[guider_id] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": previous["model"],
            "positive": previous["positive"],
            "negative": previous["negative"],
            "cfg": previous["cfg"],
        },
    }
    sigmas_source = [sigmas_id, 0]
    if split_id is not None:
        graph[split_id] = {
            "class_type": "SplitSigmasDenoise",
            "inputs": {"sigmas": [sigmas_id, 0], "denoise": previous["denoise"]},
        }
        # 出力は 0=high_sigmas / 1=low_sigmas。KSamplerのdenoiseに当たるのは後半側
        sigmas_source = [split_id, 1]
    graph[sampler_id] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": [noise_id, 0],
            "guider": [guider_id, 0],
            "sampler": [sampler_select_id, 0],
            "sigmas": sigmas_source,
            "latent_image": previous["latent_image"],
        },
    }
    del graph[ksampler_id]

    # KSamplerの出力を見ていたノードを、置き換え後のサンプラーへ向け直す。
    # SamplerCustomAdvancedの出力は 0=output / 1=denoised_output で、0を使う
    for node in graph.values():
        for key, value in node["inputs"].items():
            if value == [ksampler_id, 0]:
                node["inputs"][key] = [sampler_id, 0]


def with_beta57_sampling(graph: Graph) -> Graph:
    """グラフ中の全 KSampler を beta57 スケジュールの SamplerCustomAdvanced へ置き換える。

    KSamplerの `scheduler` 欄には beta分布のalpha / betaを渡す口が無く、
    ComfyUI標準の `beta` (alpha=0.6 / beta=0.6) しか選べない。配布元が
    beta57 と呼ぶ alpha=0.5 / beta=0.7 を使うには `BetaSamplingScheduler` を
    持つグラフが要るため、テンプレートを分ける。

    hires fixが増やす2段目のKSamplerも対象に含めるため、この軸は他の軸を
    全て適用し終えた後にかける (`axes.AXIS_BUILD_ORDER` の末尾)。
    """
    graph = copy.deepcopy(graph)
    if KSAMPLER not in graph:
        raise ValueError("with_beta57_sampling は KSampler を持つグラフにのみ適用できる")

    # 2段目から先に置き換える。1段目を先に消すと、2段目が参照している
    # [KSAMPLER, 0] (hires fixの入口) の張り替え対象が二重になる
    if HIRES_KSAMPLER in graph:
        _beta57_stage(
            graph,
            ksampler_id=HIRES_KSAMPLER,
            noise_id=BETA57_HIRES_NOISE,
            sampler_select_id=BETA57_HIRES_SAMPLER_SELECT,
            sigmas_id=BETA57_HIRES_SIGMAS,
            guider_id=BETA57_HIRES_GUIDER,
            sampler_id=BETA57_HIRES_SAMPLER,
            split_id=BETA57_HIRES_SPLIT,
        )
    _beta57_stage(
        graph,
        ksampler_id=KSAMPLER,
        noise_id=BETA57_NOISE,
        sampler_select_id=BETA57_SAMPLER_SELECT,
        sigmas_id=BETA57_SIGMAS,
        guider_id=BETA57_GUIDER,
        sampler_id=BETA57_SAMPLER,
        split_id=None,
    )
    return graph


#: LoRAのノードID帯はtaskごとに空き番が違う (img2imgは10/11をLoadImageと
#: VAEEncodeで使っている)ため、他の軸と違いtask別のテーブルを引く。
_LORA_IDS_BY_TASK: dict[Task, tuple[str, ...]] = {
    "txt2img": LORA_IDS,
    "img2img": IMG2IMG_LORA_IDS,
}

#: 軸ごとのグラフ合成関数。`build_all()` が `iter_template_specs()` の列挙から
#: 軸の並びを受け取り、ここを引いて順に適用する。`lora` だけはtaskごとに
#: ノードID帯が違うため、ここには登録せず `build_all()` 側で個別に呼ぶ。
#: 軸を1本足したら、対応するグラフ合成関数をここへ登録する。
AXIS_GRAPH_BUILDERS: dict[str, Callable[[Graph], Graph]] = {
    AXIS_UNET: to_separate_loaders,
    AXIS_VAE: with_external_vae,
    AXIS_HIRES: with_hires_fix,
    AXIS_HIRES_MODEL: with_hires_model_fix,
    AXIS_CONTROLNET: with_controlnet,
    AXIS_CONTROLNET_RAW: with_controlnet_raw,
    AXIS_IPADAPTER: with_ipadapter,
    AXIS_BETA57: with_beta57_sampling,
}


def build_all(base: Graph) -> dict[str, Graph]:
    """ベースから全テンプレートを組み立てる。

    生成するテンプレート名と、それを構成する軸の並びは `iter_template_specs()`
    (agentic_imagegen.workflows.axes) が一元管理する。合成順は
    `axes_in_build_order()` に従う。テンプレート名の接尾辞順 (`_vae` はunetの直後)
    とは vae の適用順だけが異なることに注意する (vaeは常に最後に適用する。
    hires fix が増やすVAEDecode / VAEEncodeのVAE参照もまとめて拾うため)。
    """
    task_bases: dict[Task, Graph] = {
        "txt2img": copy.deepcopy(base),
        "img2img": to_img2img(base),
    }

    templates: dict[str, Graph] = {}
    for spec in iter_template_specs():
        # 軸ごとの合成関数はいずれも先頭で deepcopy するが、軸が1つも無い場合
        # (img2img) はベースがそのまま成果物になる。ここで複製しておかないと
        # task_bases と成果物が同じオブジェクトを指す
        graph = copy.deepcopy(task_bases[spec.task])
        for axis in axes_in_build_order(spec.axes):
            if axis == AXIS_LORA:
                graph = with_lora_chain(graph, _LORA_IDS_BY_TASK[spec.task])
            else:
                graph = AXIS_GRAPH_BUILDERS[axis](graph)
        templates[spec.name] = graph
    return templates


def verify(name: str, base: Graph, graph: Graph) -> None:
    """参照整合性と、ベースのノードを潰していないことを検査する。"""
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)):
                continue
            ref, slot = value
            if ref not in graph:
                raise ValueError(f"{name}: {node_id}.{key} が存在しないノード {ref} を参照")
            count = OUTPUT_COUNTS.get(graph[ref]["class_type"], 1)
            if slot >= count:
                raise ValueError(f"{name}: {node_id}.{key} の出力スロット {slot} が範囲外")

    for node_id, node in base.items():
        if node_id == EMPTY_LATENT and EMPTY_LATENT not in graph:
            continue  # img2img系では意図的に外している
        if node_id == CHECKPOINT and CHECKPOINT not in graph:
            continue  # unet系ではローダーを3つに分けている
        if node_id == KSAMPLER and KSAMPLER not in graph:
            continue  # beta57系ではKSamplerをSamplerCustomAdvanced一式へ置き換えている
        if node_id == CLIP_SKIP and CLIP_SKIP not in graph:
            continue  # unet系ではCLIPSetLastLayerを通さない (条件付けが壊れるため)
        if node_id not in graph:
            raise ValueError(f"{name}: ベースのノード {node_id} が消えている")
        if graph[node_id]["class_type"] != node["class_type"]:
            raise ValueError(
                f"{name}: ベースのノード {node_id} が "
                f"{node['class_type']} から {graph[node_id]['class_type']} へ変わっている"
            )


def dump(graph: Graph) -> str:
    ordered = {key: graph[key] for key in sorted(graph, key=int)}
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="書き出さず、差分の有無だけ確認する")
    args = parser.parse_args()

    base = json.loads((WORKFLOWS_DIR / f"{BASE_NAME}.json").read_text(encoding="utf-8"))
    templates = build_all(base)

    stale: list[str] = []
    for name, graph in templates.items():
        verify(name, base, graph)
        path = WORKFLOWS_DIR / f"{name}.json"
        content = dump(graph)
        if args.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != content:
                stale.append(name)
            continue
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    if args.check:
        if stale:
            print("差分があります:", ", ".join(stale), file=sys.stderr)
            return 1
        print("すべて最新です")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
