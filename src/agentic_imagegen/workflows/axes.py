"""Workflowテンプレートを構成する軸の定義。

テンプレートは `unet` / `beta57` / `vae` / `lora` / `hires` / `hires_model` /
`controlnet` / `ipadapter` という直交する軸の組み合わせで決まる。この列挙結果を次の3か所が
共有することで、軸を1本足したときに書く場所を1か所に留める。

- `workflows/injector.py` の `resolve_workflow_name()` — Specのどのフィールドで
  軸を選ぶかを判定し、`suffix_for()` でテンプレート名を組み立てる
- `scripts/build_workflow_templates.py` の `build_all()` — `iter_template_specs()`
  が返す軸の並びから、軸ごとのグラフ合成関数を引いて順に適用する
- `adapters/comfyui/workflow.py` — 同様に軸ごとのbinding合成関数を引いて組み立てる

このモジュール自体はComfyUIのNode IDや、Specの具体的なフィールド構造を知らない。
持っているのは「軸の名前・接尾辞の並び順・組み立ての適用順・排他関係」という
命名とポリシーだけである。

軸を1本足す手順:

    1. 軸のキーをここへ定数として足し、`AXIS_ORDER` (テンプレート名の接尾辞順) と
       `AXIS_BUILD_ORDER` (合成の適用順、通常は `AXIS_ORDER` と同じでよい) へ
       追加する。位置は `resolve_workflow_name()` の接尾辞組み立てと一致させること
    2. 既存の軸と同時に指定できないなら `_EXCLUSIVE_PAIRS` へ追記する
    3. `workflows/injector.py` の `resolve_workflow_name()` へ、Specのどのフィールドで
       この軸を選ぶかの分岐を足す (Specの構造依存のため、ここは機械化していない)
    4. `scripts/build_workflow_templates.py` へ軸のグラフ合成関数を書き、
       `AXIS_GRAPH_BUILDERS` へ登録する
    5. `adapters/comfyui/workflow.py` へ軸のbinding合成関数を書き、
       `AXIS_BINDING_BUILDERS` へ登録する
    6. `uv run python scripts/build_workflow_templates.py` でテンプレートJSONを書き出す

これで3か所とも `iter_template_specs()` の結果を辿るだけになり、軸を1本足しても
組み合わせの数だけ手で宣言を書き足す必要がなくなる。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final, Literal

#: 論理的なタスク名。テンプレート名の先頭に来る。
Task = Literal["txt2img", "img2img"]

#: UNet / CLIP / VAE を別々に読む形式 (DiT系)。checkpoint系のローダーを丸ごと差し替える。
AXIS_UNET: Final = "unet"
#: checkpoint同梱ではなく外部VAE (VAELoader) を使う指定。
AXIS_VAE: Final = "vae"
#: LoRAを1件以上適用する指定。
AXIS_LORA: Final = "lora"
#: hires fix (latent拡大)。
AXIS_HIRES: Final = "hires"
#: hires fix (アップスケールモデルでの拡大)。`hires` と同じ軸の別の値。
AXIS_HIRES_MODEL: Final = "hires_model"
#: ControlNetで構図を指定する (制御画像をCannyで線画へ変換してから渡す)。
AXIS_CONTROLNET: Final = "controlnet"
#: ControlNetで構図を指定する (前処理済みの制御画像をそのまま渡す)。
#: `controlnet` と同じ軸の別の値で、Cannyノードの有無だけが違う。
AXIS_CONTROLNET_RAW: Final = "controlnet_raw"
#: IPAdapterで参照画像の特徴を寄せる。
AXIS_IPADAPTER: Final = "ipadapter"
#: KSamplerを SamplerCustomAdvanced + BetaSamplingScheduler (alpha=0.5 / beta=0.7) へ
#: 置き換え、配布元が beta57 と呼ぶノイズスケジュールで sampling する。
#: KSamplerのscheduler欄からは選べないため、テンプレートを分ける必要がある。
AXIS_BETA57: Final = "beta57"

#: テンプレート名の接尾辞の並び順。`resolve_workflow_name()` の接尾辞組み立てと
#: 完全に一致させること (例: txt2img_vae_lora_hires_controlnet_ipadapter)。
AXIS_ORDER: Final[tuple[str, ...]] = (
    AXIS_UNET,
    AXIS_BETA57,
    AXIS_VAE,
    AXIS_LORA,
    AXIS_HIRES,
    AXIS_HIRES_MODEL,
    AXIS_CONTROLNET,
    AXIS_CONTROLNET_RAW,
    AXIS_IPADAPTER,
)

#: グラフ / binding を組み立てる際の適用順。`AXIS_ORDER` と基本は同じだが、
#: `vae` (外部VAEへの差し替え) だけは他の軸を組み立て終えた後に適用する
#: (hires fix が増やすVAEDecode / VAEEncodeのVAE参照もまとめて拾うため)。
#: 名前の接尾辞位置と、グラフ組み立ての適用順が異なる軸が2本ある。
#:
#: - `vae` (外部VAEへの差し替え): 名前はunetの直後、適用は最後
#: - `beta57` (KSamplerの置き換え): 名前はunetの直後、適用は最後
#:
#: `beta57` を最後にするのは、hires fixが増やす2段目のKSamplerも一緒に
#: 置き換えるためである。先に適用すると、後から足された2段目だけがKSamplerのまま残る。
AXIS_BUILD_ORDER: Final[tuple[str, ...]] = (
    AXIS_UNET,
    AXIS_LORA,
    AXIS_HIRES,
    AXIS_HIRES_MODEL,
    AXIS_CONTROLNET,
    AXIS_CONTROLNET_RAW,
    AXIS_IPADAPTER,
    AXIS_VAE,
    AXIS_BETA57,
)

#: 同時に指定できない軸の組。
#: - unet (DiT系) は vae / lora / controlnet / controlnet_raw / ipadapter と排他
#:   (control / reference は Specのバリデーションでも拒否している。
#:   hires / hires_model とは併用できる)
#: - hires と hires_model は同じ軸の2値であるため互いに排他
#: - controlnet と controlnet_raw も同じ軸の2値であるため互いに排他
#: - ipadapter は hires / hires_model と併用しない
#:   (Specのバリデーションが upscale と reference の同時指定を拒否している。Issue #38)
_EXCLUSIVE_PAIRS: Final[frozenset[frozenset[str]]] = frozenset(
    frozenset(pair)
    for pair in (
        (AXIS_UNET, AXIS_VAE),
        (AXIS_UNET, AXIS_LORA),
        (AXIS_UNET, AXIS_CONTROLNET),
        (AXIS_UNET, AXIS_CONTROLNET_RAW),
        (AXIS_UNET, AXIS_IPADAPTER),
        (AXIS_HIRES, AXIS_HIRES_MODEL),
        (AXIS_HIRES, AXIS_IPADAPTER),
        (AXIS_HIRES_MODEL, AXIS_IPADAPTER),
        (AXIS_CONTROLNET, AXIS_CONTROLNET_RAW),
    )
)

#: その軸を選ぶために別の軸が要る、という依存関係。
#: - beta57 はDiT系 (unet) だけを対象にする。checkpoint系でも同じ置き換えは可能だが、
#:   SD1.5 / SDXLでの有効性を確かめておらず、テンプレート数が倍に増えるため広げない
_REQUIRED_AXES: Final[dict[str, frozenset[str]]] = {
    AXIS_BETA57: frozenset({AXIS_UNET}),
}


def _is_valid_combination(axes: tuple[str, ...]) -> bool:
    if any(frozenset(pair) in _EXCLUSIVE_PAIRS for pair in combinations(axes, 2)):
        return False
    present = frozenset(axes)
    return all(required <= present for axis, required in _REQUIRED_AXES.items() if axis in present)


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """生成しうる1テンプレートの名前と、それを構成する軸の並び。

    `axes` は `AXIS_ORDER` の部分列 (テンプレート名の接尾辞順)。グラフ / binding の
    組み立て順が欲しい場合は `axes_in_build_order()` で並べ替える。
    """

    name: str
    task: Task
    axes: tuple[str, ...]


def suffix_for(present_axes: tuple[str, ...]) -> str:
    """軸の並びからテンプレート名の接尾辞を組み立てる。"""
    return "".join(f"_{axis}" for axis in present_axes)


def axes_in_build_order(present_axes: tuple[str, ...]) -> tuple[str, ...]:
    """`AXIS_ORDER` (名前の接尾辞順) で並んだ軸を、`AXIS_BUILD_ORDER` (合成の適用順) へ並べ替える。"""
    order_index = {axis: index for index, axis in enumerate(AXIS_BUILD_ORDER)}
    return tuple(sorted(present_axes, key=lambda axis: order_index[axis]))


def iter_template_specs() -> tuple[TemplateSpec, ...]:
    """生成しうる全テンプレートを、軸の組み合わせとして列挙する。

    `txt2img` (軸なし) は手書きのベースであるため、この列挙には含めない。
    `img2img` (軸なし) は生成対象であるため含める。
    """
    specs: list[TemplateSpec] = []
    for task in ("txt2img", "img2img"):
        for size in range(len(AXIS_ORDER) + 1):
            for combo in combinations(AXIS_ORDER, size):
                if task == "txt2img" and size == 0:
                    continue
                if not _is_valid_combination(combo):
                    continue
                specs.append(TemplateSpec(name=f"{task}{suffix_for(combo)}", task=task, axes=combo))
    return tuple(specs)


#: `iter_template_specs()` の結果に手書きベースの `txt2img` を加えた、
#: 実際に許可すべきテンプレート名の全体。`ALLOWED_WORKFLOWS` の集合と一致させる。
ALL_TEMPLATE_NAMES: Final[frozenset[str]] = frozenset(
    {"txt2img", *(spec.name for spec in iter_template_specs())}
)


__all__ = [
    "ALL_TEMPLATE_NAMES",
    "AXIS_BETA57",
    "AXIS_BUILD_ORDER",
    "AXIS_CONTROLNET",
    "AXIS_CONTROLNET_RAW",
    "AXIS_HIRES",
    "AXIS_HIRES_MODEL",
    "AXIS_IPADAPTER",
    "AXIS_LORA",
    "AXIS_ORDER",
    "AXIS_UNET",
    "AXIS_VAE",
    "Task",
    "TemplateSpec",
    "axes_in_build_order",
    "iter_template_specs",
    "suffix_for",
]
