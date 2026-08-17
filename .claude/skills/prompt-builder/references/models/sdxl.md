# SDXL系 (AnythingXL_xlなど、Illustrious由来を除く)

Stable Diffusion XL 1.0をベースにしたfine-tune。`model.checkpoint` にファイル名を書く。
Illustriousを土台にしたモデル (novaAnimeXL / hassakuXL / waiNSFWIllustrious) は
記法と品質タグの語彙が別系統のため [illustrious.md](illustrious.md) を参照する。
系統によらない原則は [common.md](../common.md) を参照する。

| 項目 | 目安 |
| --- | --- |
| トークン上限 | 248 |
| 解像度 | 1024x1024相当の画素数。縦長は832x1216 / 1024x1536 |
| cfg | 4.5-7 (7.5を超えると彩度が飽和し、3未満は色が抜ける) |
| steps | 20-30 |
| sampler / scheduler | `euler_ancestral` / `normal` |

- **品質タグは先頭、構図のmodifierは末尾へ置く。** 後方のタグほど効果が薄まるため、
  重要な要素ほど前に置く
- **`score_9` のようなPony系の記法は使わない。** `masterpiece, best quality` 系か、
  モデルが学習した品質ラベルを使う
- **タグはDanbooruに実在する表記を使う。** 学習データが少ないタグはLoRAなしでは効かない。
  キャラクタ名もDanbooruの表記順に従う
  (確認手順は [common.md](../common.md#タグの実在を確認する))
- **clip skipは配布元が値を示していない。** Animagine XL 4.0とShiratakiMix XLは
  HuggingFace / civitaiのいずれにも記載がなく (2026-08-17に確認)、
  `sdxl-animagine` / `sdxl-shiratakimix` も持たない。既定のまま使う。
  clip skip 2はIllustrious系の推奨であり、この系統へ広げない
  ([illustrious.md](illustrious.md))

## モデルごとの推奨設定

同じSDXLでも、fine-tuneの系統ごとに品質タグの語彙とサンプラー設定が割れる。
style presetを系統ごとに分けているのはこのため。

| モデル | sampler / scheduler | cfg | steps | 品質タグの語彙 | style preset |
| --- | --- | --- | --- | --- | --- |
| Animagine XL 4.0 | `euler_ancestral` / `normal` | 4-7 (5を推奨) | 25-28 (28を推奨) | `masterpiece, high score, great score, absurdres` | `sdxl-animagine` |
| AnythingXL | `euler_ancestral` / `normal` | 5-7 | 25-30 | Illustrious系と同じ | `sdxl-illustrious` |
| ShiratakiMix XL | `dpmpp_3m_sde` / `karras` | 7.5 (3-8) | 20以上 | Illustrious系と同じ | `sdxl-shiratakimix` |

- **Animagine XLの品質タグは他系統へ流用しない。** `high score` / `great score` は
  Animagineの学習語彙で、Illustrious系では効かない。逆も同じ。
  どちらもDanbooruタグではなく学習時に付与された品質ラベルのため、
  `post_count` では判定しない
- **AnythingXLとShiratakiMix XLはIllustrious系の品質タグ語彙で書く。**
  ベースはSDXL 1.0だが、`masterpiece, best quality, absurdres, highres` が実用的で、
  style presetもIllustrious系のものを共有する ([illustrious.md](illustrious.md))
- **ShiratakiMix XLだけサンプラーの系統が違う。** `euler_ancestral`でも生成できるが、
  配布元のサンプルはDPM++系 + karrasで作られている
- **SDE系サンプラーはstepsを削ると破綻する。** `dpmpp_3m_sde` + `karras`を
  steps 8で流すと収束せず、ほぼ真っ白な画像になる (2026-08-13にXPUで確認)。
  steps 24では正常に生成できる。動作確認のためにstepsを落とす場合は
  `sdxl-illustrious` (`euler_ancestral`) を使う
- ComfyUIへ実在するSDXL checkpointは `animagineXL40_v40.safetensors` と
  `shiratakimixXL_v20.safetensors` (2026-08-17に配置)。`AnythingXL_xl.safetensors`
  は現在の環境には無いため、使うなら `~/ComfyUI/models/checkpoints/` へ置く
- **`animagineXL40_v40.safetensors` は現状まともに生成できない。** 解像度・サンプラー・
  preset・外部VAEのいずれを振っても格子状の抽象画にしかならない。ファイル破損・
  fp16のNaN・text encoderの欠落はいずれも切り分け済みで、原因は未特定
  ([Issue #135](https://github.com/Sylphy0052/agentic-imagegen/issues/135))。
  Animagine XL系を使うなら、この件が片付くまでは `shiratakimixXL_v20` か
  `novaAnimeXL_ilV190` (`sdxl-illustrious`) を選ぶ

## hires fix

832x1216で構図を作り、`upscale.scale: 1.5`で1248x1824へ引き上げる。
Illustrious系も同じ値でよい。

- `denoise`は0.35-0.5がSDXLで扱いやすい。SD1.5系より低めの値で足りる
- `upscale.steps`は1段目の1/3程度 (steps 30なら10)
- **実運用の定番である1024x1536の2倍 (2048x3072) は既定の上限を超える。**
  `IMAGEGEN_MAX_HEIGHT` (2048) と`IMAGEGEN_MAX_PIXELS` (4194304) の両方に当たるため、
  通すには環境変数を引き上げる
- **配布元が推奨する`R-ESRGAN 4x+Anime6B`は`upscale.model`で使える。**
  `RealESRGAN_x4plus_anime_6B.pth` を指定する
- **`sdxlVAE`のような外部VAEは`model.vae`で差し替えられる。** ただし配置済みのVAEは
  SD1.5向けのため、SDXL向けのVAEを使うなら`~/ComfyUI/models/vae/`へ置いてから指定する

SDXLはSD1.5の3-4倍の計算量になる。CPU推論では実用的な時間で終わらないため、
XPU ([xpu-setup.md](../../../../../docs/xpu-setup.md)) を用意してから使う。

構成例は [specs/examples/txt2img_sdxl.yaml](../../../../../specs/examples/txt2img_sdxl.yaml)、
preset本体は [presets/styles/sdxl-animagine.yaml](../../../../../presets/styles/sdxl-animagine.yaml)
にある。**サンプルのcheckpointとstyle presetはIllustrious系のため、
この系統で使うときは両方を差し替える。** Specの骨格は同じでよい。
