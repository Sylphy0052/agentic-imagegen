"""Workflowテンプレートの派生形を機械的に組み立てる。

`workflows/txt2img.json` を唯一の手書きベースとし、そこから
img2img / LoRA / hires fix の組み合わせを生成する。

    txt2img.json  (ベース: 人間がComfyUI GUIで作成しAPI形式で書き出したもの)
      ├─ img2img            EmptyLatentImage -> LoadImage + VAEEncode
      ├─ *_lora             CheckpointLoader の後に LoraLoader を3段
      └─ *_hires            KSampler の後に LatentUpscaleBy + 2段目 KSampler

組み合わせが増えても手で書かないのは、ノード参照を間違えても
「形は正しいまま意味だけ壊れる」ためである (実際に一度踏んでいる)。
生成後に参照整合性と、既存ノードを潰していないことを検査する。

使い方:

    uv run python scripts/build_workflow_templates.py            # 生成
    uv run python scripts/build_workflow_templates.py --check    # 差分がないか確認するだけ
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

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

DEFAULT_LORA = "add_detail.safetensors"
DEFAULT_SOURCE_IMAGE = "example.png"

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


def with_lora_chain(graph: Graph, node_ids: tuple[str, ...]) -> Graph:
    """CheckpointLoader の後に LoraLoader を直列で挟む。"""
    graph = copy.deepcopy(graph)
    for node_id in node_ids:
        if node_id in graph:
            raise ValueError(f"LoRA用のノードID {node_id} が既に使われている")

    upstream = CHECKPOINT
    for node_id in node_ids:
        graph[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": DEFAULT_LORA,
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": [upstream, 0],
                "clip": [upstream, 1],
            },
        }
        upstream = node_id

    last = node_ids[-1]
    graph[KSAMPLER]["inputs"]["model"] = [last, 0]
    graph[POSITIVE]["inputs"]["clip"] = [last, 1]
    graph[NEGATIVE]["inputs"]["clip"] = [last, 1]
    # VAE は LoraLoader を通らないため CheckpointLoader 直結のまま
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


def build_all(base: Graph) -> dict[str, Graph]:
    """ベースから全テンプレートを組み立てる。"""
    txt2img = copy.deepcopy(base)
    img2img = to_img2img(base)

    # txt2img自身は手書きのベースなので生成対象に含めない
    return {
        "txt2img_lora": with_lora_chain(txt2img, LORA_IDS),
        "txt2img_hires": with_hires_fix(txt2img),
        "txt2img_lora_hires": with_hires_fix(with_lora_chain(txt2img, LORA_IDS)),
        "img2img": img2img,
        "img2img_lora": with_lora_chain(img2img, IMG2IMG_LORA_IDS),
        "img2img_hires": with_hires_fix(img2img),
        "img2img_lora_hires": with_hires_fix(with_lora_chain(img2img, IMG2IMG_LORA_IDS)),
    }


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
