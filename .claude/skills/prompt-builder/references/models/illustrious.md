# Illustrious系 (novaAnimeXL_ilV190など)

Illustrious XLを土台にしたfine-tune。ベースはSDXLだが、品質タグの語彙と
タグ記法がSDXL 1.0系のfine-tuneと割れるため分けて扱う。
Illustrious由来ではないSDXL (Animagine XL / ShiratakiMix XL / AnythingXL) は
[sdxl.md](sdxl.md) を参照する。
系統によらない原則は [common.md](../common.md) を参照する。

| 項目 | 目安 |
| --- | --- |
| トークン上限 | 248 |
| 解像度 | 1024x1024相当の画素数。縦長は832x1216 / 1024x1536 |
| cfg | 7 (実用域4.5-7。7.5を超えると彩度が飽和し、3未満は色が抜ける) |
| steps | 30 (実用域20-30) |
| sampler / scheduler | `euler_ancestral` / `normal` |
| 品質タグ | `masterpiece, best quality, absurdres, highres` |

- **品質タグは先頭、構図のmodifierは末尾へ置く。** 後方のタグほど効果が薄まるため、
  重要な要素ほど前に置く
- **`score_9` のようなPony系の記法は使わない。** Illustriousは対応しておらず、
  `masterpiece, best quality` 系の品質タグを使う
- **タグはDanbooruに実在する表記を使う。** Illustriousの学習はDanbooruのタグ体系に
  厳密に沿っているため、表記が外れると効きが落ちる。キャラクタ名もDanbooruの表記順に従う
  (確認手順は [common.md](../common.md#タグの実在を確認する))
- **v2.0以降は自然文とタグの併用に対応する。** それ以前はタグ主体で書く
- **配布元はclip skip 2を推奨する。** `sdxl-illustrious` が持っているため、
  このpresetを使うならSpec側へ書かなくてよい
  ([model.clip_skip](../../../../../docs/spec-reference.md#modelclip_skip))

## モデルごとの推奨設定

| モデル | sampler / scheduler | cfg | steps | style preset |
| --- | --- | --- | --- | --- |
| novaAnimeXL | `euler_ancestral` / `normal` | 7 | 30 | `sdxl-illustrious` |
| hassakuXL | `euler_ancestral` / `normal` | 7 | 30 | `sdxl-illustrious` |
| waiNSFWIllustrious | `euler_ancestral` / `normal` | 7 | 30 | `sdxl-illustrious` |

- ComfyUIへ実在するIllustrious系のcheckpointは `novaAnimeXL_ilV190.safetensors`。
  hassakuXL / waiNSFWIllustriousは未配置のため、使う前に
  `~/ComfyUI/models/checkpoints/` へ置く
- **SD1.5へ蒸留した `waiIllustriousSD15_v1` は別扱い。** タグ記法はこの系統に合わせるが、
  トークン上限とcfgの実用域はSD1.5のものになる ([sd15.md](sd15.md))
- **Illustrious系は線が立つ。** `edge_stats.py` の高周波比はSD1.5系のアニメ調が
  0.04-0.10なのに対し0.15前後まで上がる。破綻の判定に使うときは系統ごとに基準を分ける

## hires fix

値の扱いはSDXLと共通のため [sdxl.md](sdxl.md#hires-fix) を参照する。

構成例は [specs/examples/txt2img_sdxl.yaml](../../../../../specs/examples/txt2img_sdxl.yaml)、
preset本体は [presets/styles/sdxl-illustrious.yaml](../../../../../presets/styles/sdxl-illustrious.yaml)
にある。

## 参考

- [Arctenox's Simple Prompt Guide for Illustrious](https://civitai.com/articles/23210/arctenoxs-simple-prompt-guide-for-illustrious)
- [Comprehensive Guide of Illustrious XL](https://tensor.art/articles/831123524065191393)
