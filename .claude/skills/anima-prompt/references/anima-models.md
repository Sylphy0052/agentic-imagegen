# Anima系モデル別の設定

配置済みのAnima系モデルと、配布元が推奨する設定をまとめる。
共通の記法とタグ順序は [SKILL.md](../SKILL.md) にある。
ここはモデルごとに違う部分だけを扱う。

## 共通の前提

Anima系はUNet単体で配布され、text encoderとVAEを同梱しない。3つを揃えて指定する。

| 役割 | ファイル | 置き場 |
| --- | --- | --- |
| text encoder | `qwen_3_06b_base.safetensors` (1.11GB) | `models/text_encoders/` |
| VAE | `qwen_image_vae.safetensors` (242MB) | `models/vae/` |

WAI-ANIMAとMiaoMiao Haremは配布ページに専用のtext encoder
(`waiANIMA_v10Base10_txt.safetensors` / `miaomiaoHarem_aniAnimeColoring10_txt.safetensors`、
いずれも1.11GB) を同梱している。**同一かどうかは未検証。**
共用のtext encoderで出力が崩れた場合は、専用のものを落として差し替えて比べる。

## 配置済みのモデル

| 通称 | `model.unet` | サイズ | 系統 |
| --- | --- | --- | --- |
| Hassaku (Anima) v1.3 | `hassakuAnima_v13_int8.safetensors` | 2.1GB (int8) | Base系fine-tune |
| WAI-ANIMA v1.0 | `waiANIMA_v10Base10.safetensors` | 3.9GB (fp16) | Base 1.0系fine-tune |
| MiaoMiao Harem Ani | `miaomiaoHarem_aniAnimeColoring10.safetensors` | 3.9GB (bf16) | Anima 1.2系fine-tune |
| CottonAnima base1 | `cottonanima_base1.safetensors` | 3.9GB (fp16) | Base v1.0 + 画風LoRAのマージ |

int8版はComfyUI独自の量子化形式 (`comfy_quant`) で、`UNETLoader` の
`weight_dtype=default` でそのまま読める。

**メモリに余裕がない環境ではHassakuのint8を既定にする。** 3.9GB版はtext encoderとVAEを
合わせて5.3GBを要求する。複数モデルを続けて試すときは、1モデルごとにComfyUIを起動して停止する
(手順は [CLAUDE.md](../../../../CLAUDE.md) の「ComfyUIを常駐させたまま生成を繰り返さない」)。

## モデル別の推奨設定

配布元の記載をそのまま写したもの。**実測による裏取りは未実施。**

| 通称 | sampler | scheduler | cfg | steps | 解像度 |
| --- | --- | --- | --- | --- | --- |
| Hassaku (Anima) v1.3 | `er_sde` | `simple` | 3.5 (3-6) | 25 (20-50) | 832x1216 |
| WAI-ANIMA v1.0 | `euler_ancestral` | `normal` | 4-5 | 20-30 | 1024x1344 |
| MiaoMiao Harem Ani | `euler` / `euler_ancestral` | `normal` | 4-5 | 30 | 記載なし |
| CottonAnima base1 | `er_sde` | (記載なし。`simple`) | 5 | 24-30 | 記載なし |

### beta57はschedulerではない

**モデル配布元が推奨する `beta57` はKSamplerのschedulerではない。**
`beta` schedulerのalpha=0.5 / beta=0.7を指す通称であり、指定するには `BetaSamplingScheduler`
ノードを持つWorkflowが要る。本リポジトリのテンプレートはKSamplerベースのため使えない。
`simple` を使う。指定できるschedulerの一覧は
[docs/spec-reference.md](../../../../docs/spec-reference.md#generation) を参照。

## モデル別の品質タグ

**ここがモデル間で最も食い違う。** そのまま流用しない。

| 通称 | positive接頭 | negative |
| --- | --- | --- |
| Hassaku (Anima) v1.3 | `masterpiece, best quality, score_7` | `worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts` |
| WAI-ANIMA v1.0 | `masterpiece, best quality, score_7` | 上記 + `artist name, lowres, censor` |
| MiaoMiao Harem Ani | `best quality, score_7, score_9, sensitive, very aesthetic, ultra detailed, fair skin, high contrast` | 上記 + `shiny skin` |
| CottonAnima base1 | `masterpiece, highres, absurdres, newest` | 上記 + `sepia, username, watermark, chibi, source_pony, source_furry, source_cartoon, monochrome, greyscale, 3d, realistic, backlighting` |

- **CottonAnimaはscoreタグを使わない。** 画風LoRAをマージした固定画風のモデルで、
  推奨接頭は `masterpiece, highres, absurdres, newest`。
  Aesthetic系と同じ扱いにして `score_*` をpositive / negativeの双方から外す
- **MiaoMiaoだけratingが `sensitive`。** 配布元の既定がそうなっている。
  `safe` にしたい場合は明示的に置き換える
- MiaoMiaoの `shiny skin` はこのモデル固有のnegative。肌のテカりを止めるためのもの

## 得手不得手

**未検証。** 4モデルを同一条件 (同一プロンプト・同一seed・同一sampler設定) で
比較した結果をここへ書く。比較の手順は
[imagegen skillのablation](../../imagegen/references/ablation.md) に従い、1回につき1軸だけ振る。

現時点で配布元の記載から分かっているのは次だけ。

- **CottonAnima** — 画風LoRAをマージ済み。画風の一貫性が高い代わりに多様性が低い
- **MiaoMiao Harem Ani** — 塗り (Anime Coloring) を狙ったfine-tune。`fair skin` / `high contrast` を
  推奨接頭に含み、肌のテカりをnegativeで抑える設計
- **WAI-ANIMA** — hires fixは1.5倍まで、denoise 0.35-0.5を推奨
- **Hassaku (Anima)** — int8版があり最も軽い。既定に据える

## 出典

- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) — 記法・タグ順序・品質タグ体系の一次情報
- [Hassaku (Anima)](https://civitai.com/models/2641326/hassaku-anima)
- [WAI-ANIMA](https://civitai.com/models/2544636/wai-anima)
- [MiaoMiao Harem](https://civitai.com/models/934764/miaomiao-harem)
- [CottonAnima](https://civitai.com/models/2382223/cottonanima)
