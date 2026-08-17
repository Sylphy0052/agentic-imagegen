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

### beta57の使い方

**`beta57` はKSamplerのscheduler欄からは選べないが、Specでは指定できる。**
beta分布の alpha=0.5 / beta=0.7 を指す通称で、KSamplerが選べる `beta` は
ComfyUI既定の alpha=0.6 / beta=0.6 に固定されている。

Specへ `scheduler: beta57` と書くと、`BetaSamplingScheduler` を持つDiT系専用の
テンプレート (`*_unet_beta57*`) へ自動で切り替わる。RES4LYFなどのカスタムノードは要らない
(ComfyUI標準の `BetaSamplingScheduler` が alpha / beta を受け付ける)。

```yaml
generation:
  sampler: er_sde
  scheduler: beta57
```

- **checkpoint系 (SD1.5 / SDXL) では指定できない。** 同じ置き換えは技術的に可能だが、
  有効性を確かめておらずテンプレートを用意していない。指定するとSpecの検証で拒否する
- hires fixと併用できる (`*_unet_beta57_hires`)。2段目も同じスケジュールで走る
- 低ノイズ側のstepへ配分が寄るため、背景・テクスチャ・肌の情報量が増える。
  配布元は「写実寄り・絵画寄りの質感」に効くと書いている

指定できるschedulerの一覧は
[docs/spec-reference.md](../../../../docs/spec-reference.md#generation) を参照。

### sampler / schedulerの選び方

配布元とコミュニティの記載を整理したもの。**いずれも実測による裏取りは未実施。**

| sampler | 傾向 |
| --- | --- |
| `er_sde` | 中庸。フラットな色と締まった線。汎用の既定に据える |
| `euler_ancestral` | 線が柔らかく細くなる。cfgを上げても崩れにくい。3D寄りの背景には向かない |
| `dpmpp_2m_sde_gpu` | `er_sde` に近いが多様性が出る |
| `heunpp2` | `beta57` と組むと発色が濃く、まとまりが良い |
| `uni_pc` + `ddim_uniform` | 速く安定するという報告がある |

避けるものも記載がある。

- **CFG++系 (`*_cfg_pp`) をcfg 4-5で使わない。** 過剰に処理されたノイズの多い出力になる。
  使うならcfg 1.5以下
- **`res_multistep` + `beta` はドット状のアーティファクトが出ることがある**
- `ipndm_v` は出力が安定しないという報告がある

schedulerは `simple` / `normal` / `sgm_uniform` が無難で、情報量を上げたいときに `beta57`。
`kl_optimal` は3D寄りのテクスチャに向くという報告がある一方、
不安定という報告もあり評価が割れている。

### hires fixの目安

配布元とコミュニティの記載による。**実測による裏取りは未実施。**

| 項目 | 値 |
| --- | --- |
| 倍率 | 1.5倍まで。それ以上は1段で上げず、1.5倍 -> 目標解像度の2段に分ける |
| denoise | 0.25-0.3 (アニメ調)。3D寄りなら0.5 |
| steps | 15-20 |

Specの既定 (`denoise` 0.5) はSD1.5系に合わせた値で、Anima系には強すぎる。
DiT系でhires fixを使うときは `generation.upscale.denoise` を明示して下げる。

```yaml
generation:
  upscale:
    scale: 1.5
    denoise: 0.3
    steps: 16
```

WAI-ANIMAだけは配布元がhires fixのdenoise 0.35-0.5を推奨しており、他より高い。

### LoRA

**Anima系はLoRAと併用できる** (Issue #39)。`model.loras` へ書く形はSD1.5 / SDXLと同じ。

```yaml
model:
  unet: hassakuAnima_v13_int8.safetensors
  clip: qwen_3_06b_base.safetensors
  vae: qwen_image_vae.safetensors
  loras:
    - name: anima_context_detailer_base10.safetensors
      strength_model: 1.0
```

- **baseModelがAnimaのLoRAだけが当たる。** SD1.5 / SDXL向けのLoRAを指定しても
  キーが一致せず、エラーも警告も出ないまま何も起きない。
  civitaiでは `baseModel=Anima` で絞り込める
- Anima向けのLoRAはUNet側のキーしか持たないものが多い。その場合 `strength_clip` は
  効かない (指定してもエラーにはしない)
- hires fix・beta57とも併用できる

## 品質タグ・rating・yearの体系

品質タグには2つの系統がある。混ぜて盛らない。

| 系統 | 語彙 |
| --- | --- |
| 人間評価ベース | `masterpiece` / `best quality` / `good quality` / `normal quality` / `low quality` / `worst quality` |
| aestheticモデルベース | `score_9` - `score_1` |

モデルの系統によってscoreタグの扱いが変わる。どの系統にどう振り分けるかは
[SKILL.mdの手順1](../SKILL.md) にある。Aesthetic系は学習時にcaptionから品質タグを
除いてあるため、`score_9` を重ねるとかえって過剰な「AI絵らしさ」に寄る。

ratingは `safe` / `sensitive` / `nsfw` / `explicit` の4つ。
**通常のイラストでは `safe` を明示する。** 短く曖昧なプロンプトでは意図しない出力になりやすい。

yearは `year 2025` のような明示指定と、`newest` / `recent` / `mid` / `early` / `old` の
相対指定がある。絵柄の年代を効かせたいときに入れる。

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

## 選ぶ順序

モデルを決めていない、または出力が要求に合わずに振り直すときは、この順で試す。

1. **Hassaku (Anima) v1.3** — 既定。int8で最も軽く、光の作り方と線の細さが扱いやすい
2. **WAI-ANIMA v1.0** — 背景ごと見せたいとき。人物が引きになる分、場面の情報量が出る
3. **CottonAnima base1** — 画風を揃えた連作にするとき。1枚ごとに絵柄を変えたい用途には向かない
4. **MiaoMiao Harem Ani** — 塗りの階調が要るとき。ただし構図の指示から外れやすい

同一条件で出した4枚を比べて決めた順序で、各モデルの傾向は下の「得手不得手」にある。
題材や要求が変われば順序も変わるため、固定の優劣ではなく試す順の既定として扱う。

## 得手不得手

同一のSpec (同一プロンプト・同一seed 545078971・640x896・style presetはモデルごとの推奨設定)
で4モデルを1枚ずつ出した観察。**n=1のため傾向の目安であり、確定した特性ではない。**
題材は「図書館の本棚の間に立つ制服の少女、全身」。
比較の手順は [imagegen skillのablation](../../imagegen/references/ablation.md) に従い、
1回につき1軸だけ振る。

| 通称 | 色調 | 塗り・線 | 構図の取り方 | 背景の密度 |
| --- | --- | --- | --- | --- |
| Hassaku (Anima) v1.3 | 暖色寄り。逆光と差し込む光を作る | 陰影が柔らかく線が細い | 指示どおり正面。人物が大きい | 高い。棚の本に色数が出る |
| WAI-ANIMA v1.0 | 寒色寄りで彩度が低い | フラットで均一 | 指示どおり正面。人物が小さく引く | 高い。天井や照明まで描く |
| MiaoMiao Harem Ani | 明暗差が大きい | 肌の階調が滑らか | 指示に反して斜めに立つ | 低い。暗く落として奥をぼかす |
| CottonAnima base1 | シアンに寄る。床まで色が乗る | 線が太く輪郭がはっきり | 指示どおり正面 | 中程度。全体が明るくフラット |

- **MiaoMiaoは構図の指示から外れやすい。** `looking at viewer` と自然文で正面を指定しても
  斜め立ちになった。`high contrast` を推奨接頭に含む設計と合わせて、
  絵作りを優先する性格と考えられる。構図を固定したい用途では他を選ぶ
- **CottonAnimaは画風が全体を支配する。** 画風LoRAをマージ済みのため、
  背景の床や空気の色まで一定のシアン寄りに引き寄せられる。画風を揃えたい連作に向き、
  1枚ごとに絵柄を変えたい用途には向かない
- **WAI-ANIMAは引きの構図で背景を作り込む。** 全身指定で人物が最も小さくなった。
  背景ごと見せたい場面に向く
- **Hassakuは光を作る。** 窓からの光を拾って逆光や差し込みを描く。int8版で最も軽く、
  既定に据えて困らない

配布元の記載から分かっているのは次。

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
- [About Anima + settings](https://civitai.com/articles/30357/about-anima-settings) — sampler / scheduler / hires fixの目安
- [Optimal sampler / scheduler settings for Anima](https://huggingface.co/circlestone-labs/Anima/discussions/165) — 組み合わせの比較とhires fixの上限
- [beta57とは何か](https://note.com/hkmclab/n/n10e506f4553d) — beta57の由来と推奨の組み合わせ
