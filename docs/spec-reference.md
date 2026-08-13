# GenerationSpecリファレンス

`GenerationSpec` (YAML) の全フィールドの仕様。**値域・既定値・組み合わせ規則の一次情報はこの文書**で、
`README.md` / `CLAUDE.md` / `.claude/skills/imagegen/SKILL.md` からはここを参照する。

定義の実体は [`src/agentic_imagegen/domain/models.py`](../src/agentic_imagegen/domain/models.py)。
未知のキーは受け付けない。書いたのに効かない状態を作らないため、効かない組み合わせは
黙って無視せずその場で拒否する。

## 目次

- [全体像](#全体像)
- [トップレベル](#トップレベル)
- [prompt](#prompt)
- [presets](#presets)
- [generation](#generation)
- [model](#model)
- [source (img2img)](#source-img2img)
- [control (ControlNet)](#control-controlnet)
- [reference (IPAdapter)](#reference-ipadapter)
- [text (テキスト合成)](#text-テキスト合成)
- [output](#output) / [metadata.json](#metadatajson)
- [組み合わせの可否](#組み合わせの可否)
- [Workflowテンプレートの決まり方](#workflowテンプレートの決まり方)
- [環境変数による上限](#環境変数による上限)

## 全体像

最小のSpec。

```yaml
version: "1"
task: txt2img

prompt:
  positive: 1girl, blue hair, anime illustration

model:
  checkpoint: meinamix_v12Final.safetensors
```

指定できるブロックは次の10個。

```text
version   task   presets   prompt   generation   model
source    control   reference   text   output
```

## トップレベル

| キー | 型 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `version` | `"1"` | `"1"` | Specのバージョン。現在は`"1"`のみ |
| `task` | `txt2img` / `img2img` | `txt2img` | 生成の種類 |
| `presets` | mapping | `{}` | 適用するpresetの参照 |
| `prompt` | mapping | 必須 (presetで補える) | プロンプト |
| `generation` | mapping | 既定値で補完 | 生成パラメータ |
| `model` | mapping | 必須 | 使用するモデル |
| `source` | mapping | `null` | img2imgの入力画像。`task: img2img`のときのみ |
| `control` | mapping | `null` | ControlNetの指定 |
| `reference` | mapping | `null` | IPAdapterの指定 |
| `text` | mapping | `null` | 生成後に合成するテキスト |
| `output` | mapping | 既定値で補完 | 出力先 |

## prompt

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `positive` | string | 必須 | 1文字以上 |
| `negative` | string | `""` | - |

`presets`で`positive`が埋まる場合はSpec本体の`prompt`を省略できる。
presetとSpec本体の連結規則は [presets](#presets) を参照。

モデル系統ごとのプロンプトの書き方 (タグ語彙・語順・重み付け・品質タグの記法) は
[prompting-guide.md](prompting-guide.md) を参照。

### Textual Inversion embedding

`positive` / `negative` の中に `embedding:<name>` と書くと、ComfyUIが
`models/embeddings/` 配下の該当ファイルをCLIPTextEncode側で解決する
(`easynegative` / `badhandv4` のような手・解剖の破綻を抑える定型embeddingなど)。

```yaml
prompt:
  negative: >
    embedding:easynegative, worst quality, low quality
```

- **`embedding:` の直前には空白が要る。** ComfyUIは空白の直後にある `embedding:` しか
  embeddingとして解決しない。`1girl,embedding:easynegative` のように空白が無いと
  ただの語として扱われ、警告すら出ずに無視される。この書き方は
  `InvalidGenerationSpec` (exit code 2) で止める
- **`<name>` の拡張子は付けても付けなくてもよい。** `list_embeddings` は拡張子を
  落とした名前を返すが、ComfyUIは `easynegative.safetensors` のような指定も解決する
- **未配置のembeddingを指定すると生成前に `InvalidGenerationSpec` (exit code 2) で止まる。**
  ComfyUI自身は未配置のembeddingを見つけても例外にせず、警告ログを残して黙って
  無視するだけ (生成は成功するがembeddingは効かない)。それでは気づけないため、
  本リポジトリ側で生成直前に検出する
- **このチェックは `imagegen generate` / MCPの `generate_image` / `generate_batch` でのみ行う。**
  `imagegen validate` と MCPの `validate_generation` はComfyUIへ接続しないため、
  embeddingの実在チェックはできない (スキーマと上限値だけを見る)
- **プレフィックス無しの素の単語 (`easynegative` など) は対象外。** ComfyUI自体が
  `embedding:` プレフィックスの無い語をembeddingとして解決しない仕様のため、
  ここでもそのまま普通の単語 (トークン) として扱い、警告も拒否もしない。
  意図してembeddingを効かせたい場合は必ず `embedding:` を付ける
- 配置済みのembedding名は `list_embeddings` (MCP) で確認する。CLIからは
  `~/ComfyUI/models/embeddings/` を直接確認する
- checkpointの世代 (SD1.5 / SDXL) とembeddingの対応関係は検証しない (ComfyUI側の責務)。
  詳細は [prompting-guide.md](prompting-guide.md) を参照

## presets

繰り返し使う指定を3つの軸へまとめ、名前で参照する。1軸につき1つまで。

| キー | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定 |

preset名は英数字で始まる`[A-Za-z0-9._-]`のみ。探索ルートは`IMAGEGEN_PRESETS_ROOT` (既定`presets`)。

```yaml
presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: anime-soft
```

preset本体は`description` / `prompt` / `generation`を持つ。

```yaml
# presets/styles/anime-soft.yaml
description: 柔らかい光のアニメ調。SD1.5想定

prompt:
  positive: anime illustration, soft lighting, masterpiece, best quality
  negative: low quality, worst quality, blurry

generation:
  sampler: dpmpp_2m
  scheduler: karras
```

解決規則:

- **prompt**: `character` -> `scene` -> `style` -> Spec本体 の順にカンマ連結し、重複トークンは
  最初の1つを残して除去する (大文字小文字と連続空白は無視)。`negative`も同じ
- **generation**: presetの指定を取り込んだうえでSpec本体の指定を優先する
  (優先順位はspec > style > scene > character)
- 適用したpreset名は解決後のSpecに残り、`metadata.json`にも記録される

軸の責務を混ぜない。解像度とseedは再現性に直結するためpresetには書かず、Spec側で指定する。
品質タグとサンプラー設定はモデルの学習内容に依存するため、style presetを別の系統へ流用しない。

SD1.5系は配置済みのcheckpointごとに1つ用意してある。SDXL系はfine-tuneの系統ごと。

| style preset | 対象 | sampler / scheduler | cfg | steps |
| --- | --- | --- | --- | --- |
| `sd15-meinamix` | `meinamix_v12Final` | `dpmpp_2m` / `karras` | 6.5 | 30 |
| `sd15-counterfeit` | `counterfeitV30_v30` | `dpmpp_2m` / `karras` | 9.0 | 25 |
| `sd15-aom3` | `abyssorangemix3AOM3_aom3a1b` | `dpmpp_sde` / `karras` | 7.0 | 25 |
| `sd15-anylora` | `anyloraCheckpoint_bakedvaeBlessedFp16` | `dpmpp_2m` / `karras` | 7.0 | 25 |
| `sd15-cetusmix` | `cetusMix_Whalefall2` | `dpmpp_2m` / `karras` | 6.0 | 24 |
| `sd15-darksushi` | `darkSushiMixMix_225D` | `dpmpp_sde` / `karras` | 7.5 | 30 |
| `sd15-hassaku` | `hassakuSD15_v13` | `ddim` / `normal` | 8.0 | 20 |
| `sd15-chilloutmix` | `chilloutmix_NiPrunedFp16Fix` (写実寄り) | `dpmpp_sde` / `karras` | 7.0 | 20 |
| `sdxl-illustrious` | SDXL (Illustrious系 / AnythingXL) | `euler_ancestral` / `normal` | 7.0 | 30 |
| `sdxl-animagine` | SDXL (Animagine XL系) | `euler_ancestral` / `normal` | 6.0 | 25 |
| `sdxl-shiratakimix` | SDXL (ShiratakiMix XL系) | `dpmpp_3m_sde` / `karras` | 7.5 | 28 |
| `anima-base` | DiT系 (Anima) | `er_sde` / `simple` | 4.5 | 32 |
| `anime-soft` | SD1.5汎用 (下描き) | `dpmpp_2m` / `karras` | 5.5 | 20 |
| `anime-detailed` | SD1.5汎用 (仕上げ) | `dpmpp_2m` / `karras` | 7.0 | 30 |

`sd15-*` はcheckpointと1対1で対応する。checkpointを決めていれば、それに合うものを選べば
sampler / scheduler / cfg / stepsをSpec側で書き直す必要はない。
値の根拠と各モデルの傾向は
[prompting-guide.md](prompting-guide.md#配置済みのsd15系モデル) を参照。

`anime-soft` / `anime-detailed` はcheckpointを決めていない段階の汎用preset。
負荷で選ぶ (`anime-soft`が下描き、`anime-detailed`が仕上げ)。

SDXL向けの3つはcheckpointのfine-tune系統で選ぶ。系統ごとの違いは
[prompting-guide.md](prompting-guide.md#モデルごとの推奨設定) を参照。
`sdxl-animagine` / `sdxl-shiratakimix` に対応するcheckpointは未配置。

## generation

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `width` | int | 512 | 64-8192、8の倍数。`IMAGEGEN_MAX_WIDTH`も超えられない |
| `height` | int | 512 | 64-8192、8の倍数。`IMAGEGEN_MAX_HEIGHT`も超えられない |
| `steps` | int | 20 | 1-100 |
| `cfg` | float | 7.0 | 0-30 |
| `seed` | int | -1 | -1、または0以上`2**63-1`以下。-1は実行時に乱数へ解決する |
| `batch_size` | int | 1 | 1-4。`IMAGEGEN_MAX_BATCH`も超えられない |
| `sampler` | enum | `euler` | 下記44種 |
| `scheduler` | enum | `normal` | 下記9種 |
| `upscale` | mapping | `null` | hires fix。指定すると2段階生成になる |

`sampler`に指定できる値:

```text
euler  euler_cfg_pp  euler_ancestral  euler_ancestral_cfg_pp
heun  heunpp2  exp_heun_2_x0  exp_heun_2_x0_sde
dpm_2  dpm_2_ancestral  lms  dpm_fast  dpm_adaptive
dpmpp_2s_ancestral  dpmpp_2s_ancestral_cfg_pp  dpmpp_sde  dpmpp_sde_gpu
dpmpp_2m  dpmpp_2m_cfg_pp  dpmpp_2m_sde  dpmpp_2m_sde_gpu
dpmpp_2m_sde_heun  dpmpp_2m_sde_heun_gpu  dpmpp_3m_sde  dpmpp_3m_sde_gpu
ddpm  lcm  ipndm  ipndm_v  deis
res_multistep  res_multistep_cfg_pp  res_multistep_ancestral  res_multistep_ancestral_cfg_pp
gradient_estimation  gradient_estimation_cfg_pp  er_sde
seeds_2  seeds_3  sa_solver  sa_solver_pece  ddim  uni_pc  uni_pc_bh2
```

`scheduler`に指定できる値:

```text
normal  karras  exponential  sgm_uniform  simple  ddim_uniform  beta
linear_quadratic  kl_optimal
```

ComfyUIのKSamplerが受け付けるものに揃えてある。配布元が`beta57`のような通称で
推奨している場合は [prompting-guide.md](prompting-guide.md) を参照する。

CPU推論では負荷が所要時間へ直接跳ね返るため、既定は控えめにする。

| 項目 | 推奨 |
| --- | --- |
| 解像度 | 512x512 / 512x768 |
| steps | 20前後 |
| cfg | 5.0-8.0 |
| batch_size | 1 |

実測の所要時間と`IMAGEGEN_TIMEOUT`の目安は [xpu-setup.md](xpu-setup.md) を参照。

`img2img`では`width` / `height`を指定できない (入力画像のサイズをそのまま使うため)。
`batch_size`も1のみ。

### generation.upscale (hires fix)

1段目の結果を拡大し、2段目のKSamplerで描き足す。拡大の仕方は2通りある。

- `model`未指定: latentのまま拡大する。追加のモデルが要らない
- `model`指定: 一度pixelへ戻し、アップスケールモデル (ESRGAN系) で拡大してから戻す

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `scale` | float | 1.5 | 1.0より大きく4.0以下 |
| `denoise` | float | 0.5 | 0.0-1.0。低いほど元の絵を保つ。0.3-0.5が扱いやすい |
| `steps` | int | 1段目と同じ | 1-100 |
| `method` | enum | `nearest-exact` | 下の対応表を参照 |
| `model` | string | `null` | `~/ComfyUI/models/upscale_models/`のファイル名。拡張子は`.pth` / `.safetensors` |
| `model_scale` | float | 4.0 | 1.0-8.0。`model`と一緒にのみ指定できる |

`method`は拡大の経路で選べる値が違う。ComfyUI側のノードに無い値は投入前に拒否する。

| 値 | latent拡大 (`model`未指定) | モデル拡大 (`model`指定) |
| --- | --- | --- |
| `nearest-exact` / `bilinear` / `area` / `bicubic` | 使える | 使える |
| `bislerp` | 使える | 使えない |
| `lanczos` | 使えない | 使える |

```yaml
generation:
  width: 512
  height: 768
  steps: 20
  upscale:
    scale: 1.5
    denoise: 0.45
    steps: 8
```

```yaml
generation:
  width: 512
  height: 512
  steps: 14
  upscale:
    model: RealESRGAN_x4plus_anime_6B.pth
    scale: 2.0
    denoise: 0.35
    steps: 8
```

- 指定するとテンプレートが`*_hires` (モデル拡大なら`*_hires_model`) へ自動的に切り替わる
- **生成時間は倍以上になる。** 2段目は拡大後の解像度で走るため、1stepあたりの時間も伸びる
- 2段目のseedは1段目と同じ値を使う (変えると元の絵から離れるため)
- 最初から大きい解像度で生成するより、hires fixの方が構図が破綻しにくい
- 滑らかさを求める場合は`method: bislerp`を試す (latent拡大のみ)
- `control`と併用できる (`*_hires_controlnet` / `*_hires_model_controlnet`)。
  ControlNetが効くのは1段目だけ。`reference`とは併用できない
  ([組み合わせの可否](#組み合わせの可否))

#### アップスケールモデルを使う場合

- **`model_scale`にはモデルの固有倍率を書く。** 出力サイズはモデル側で決まるため、
  `scale`へ合わせるにはこの値が要る。`RealESRGAN_x4plus_anime_6B.pth`なら4.0。
  拡大後に`scale / model_scale`倍へ縮小する
- `scale`が`model_scale`を超える指定は拒否する (モデルの出力より大きくは引き伸ばさない)
- **拡大の時点で線が補間されるため、`denoise`はlatent拡大より低めで足りる。** 0.3-0.4が目安。
  latent拡大を同じ`denoise`で使うと、拡大の粗さを2段目が補いきれず絵が崩れることがある
- **拡大後の解像度にも上限がある。** `IMAGEGEN_MAX_UPSCALED_PIXELS` (既定16777216) と
  突き合わせるのは最終解像度ではなく途中のピークで、モデル拡大では
  `width x model_scale` x `height x model_scale` になる
- pixelへ戻す往復 (VAEDecode / VAEEncode) が挟まるぶん所要時間が伸びる。
  実測は [xpu-setup.md](xpu-setup.md#所要時間とタイムアウトの目安) を参照
- モデルは`~/ComfyUI/models/upscale_models/`へ置く。配置済みの名前はMCPの`list_upscale_models`で確認する

## model

指定の仕方は2通りあり、どちらか一方だけを使う。両方書くと拒否される。

- `checkpoint`: UNet / CLIP / VAEを1ファイルに含む従来形式 (SD1.5 / SDXL系)
- `unet` + `clip` + `vae`: 3つを別々に読む形式 (AnimaなどのDiT系)

| キー | 型 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `checkpoint` | string | `null` | `~/ComfyUI/models/checkpoints/`のファイル名 |
| `unet` | string | `null` | `~/ComfyUI/models/diffusion_models/`のファイル名 |
| `clip` | string | `null` | `~/ComfyUI/models/text_encoders/`のファイル名 |
| `vae` | string | `null` | `~/ComfyUI/models/vae/`のファイル名 |
| `loras` | list | `[]` | 適用するLoRA。同時3件まで |
| `clip_skip` | int | `null` | CLIPの最終層を何層手前で打ち切るか。1-12 |

配置済みのcheckpointごとの絵柄とcfg / stepsの目安は
[prompting-guide.md](prompting-guide.md#配置済みのsd15系モデル)にある。

`checkpoint` / `unet` / `clip` / `vae`の拡張子は`.safetensors` / `.ckpt`。
モデル名は共通して次を拒否する。

- 絶対パス (`/`始まり)、ホームディレクトリ参照 (`~`始まり)、バックスラッシュ、制御文字
- 上位ディレクトリ参照 (`..`)、空のセグメント
- サブフォルダ2階層以上 (1階層までは許可)

### model.loras

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `name` | string | 必須 | `~/ComfyUI/models/loras/`のファイル名。拡張子は`.safetensors` / `.pt` / `.ckpt` |
| `strength_model` | float | 1.0 | -10.0-10.0 |
| `strength_clip` | float | 1.0 | -10.0-10.0 |

```yaml
model:
  checkpoint: meinamix_v12Final.safetensors
  loras:
    - name: add_detail.safetensors
      strength_model: 0.8
      strength_clip: 0.8
```

- 同時3件まで。テンプレート側の`LoraLoader`の段数と一致させている
- 同じLoRAの重複指定は拒否する。二重に積むと意図しない強度になるため
- 指定するとテンプレートが`*_lora`へ自動的に切り替わる
- 配置先は`~/ComfyUI/models/loras/`。実在しない名前を指定するとComfyUI側で拒否される

### model.clip_skip

SD1.5系のアニメ調モデルには、CLIPの最終層を1つ手前で打ち切るclip skip 2を前提に
学習しているものが多い (NovelAI系の系譜)。配布元が推奨値を明記していることが多く、
そこに合わせて指定する。

```yaml
model:
  checkpoint: meinamix_v12Final.safetensors
  clip_skip: 2
```

- 値域は1-12。ComfyUIの`CLIPSetLastLayer`は`stop_at_clip_layer`として-1から-24を
  受け付けるが、実用は1-2程度のため`clip_skip`の値をそのまま`-clip_skip`へ変換して渡す
- **未指定時はComfyUI既定 (clip skip 1相当、`stop_at_clip_layer: -1`) のまま動く。**
  既存のSpecの出力に影響しない
- `model.loras`と併用する場合、打ち切りはLoRA適用後のCLIPに対して効く
  (`LoraLoader`の後段に挿入されるため)
- **DiT系 (`unet` + `clip` + `vae`) とは併用できない。** text encoderの構造が異なるため

### DiT系モデル (Anima)

DiT系モデルはUNet単体で配布され、text encoderとVAEを同梱しない。3つを揃えて指定する。

```yaml
generation:
  width: 832        # Animaは1024x1024前後が前提。832x1216が扱いやすい
  height: 1216
  steps: 32         # 配布元の推奨は30-50。下げすぎると線が甘くなる
  cfg: 4.0          # SD1.5系より低め。推奨は4-5で、5を超えると崩れやすい
  sampler: er_sde
  scheduler: simple

model:
  unet: hassakuAnima_v13_int8.safetensors
  clip: qwen_3_06b_base.safetensors
  vae: qwen_image_vae.safetensors
```

- テンプレートは`<task>_unet`へ切り替わる (`txt2img_unet` / `img2img_unet`)
- **img2img / hires fixとは併用できる。** `generation.upscale`を足すと`<task>_unet_hires`になる
- **LoRA / ControlNet / IPAdapterとは併用できない。** `control_v11p_sd15_*`も
  `ip-adapter-plus_sd15`もSD1.5向けで、DiT系のUNetへは適用できない。
  LoRAも同様に、SD1.5 / SDXL向けのものは当てられない
- text encoderとVAEはUNetと対応関係がある。AnimaならQwen3-0.6BとQwen-Image VAE
- モデル名はMCPの`list_diffusion_models` / `list_text_encoders` / `list_vaes`で確認する

入手手順は [comfyui-setup.md](comfyui-setup.md)、併用範囲を広げる条件は
[Issue #39](https://github.com/Sylphy0052/agentic-imagegen/issues/39) を参照。

## source (img2img)

`task: img2img`のときのみ指定する。`txt2img`で書くと拒否される。

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `image` | string | 必須 | リポジトリ配下の相対パス。拡張子は`.png` / `.jpg` / `.jpeg` / `.webp` |
| `denoise` | float | 0.6 | 0.0-1.0。0に近いほど入力画像を保ち、1に近いほど描き直す |

```yaml
task: img2img

source:
  image: inputs/reference.png
  denoise: 0.55
```

- 入力画像は生成前にComfyUIへ自動でアップロードされる。`~/ComfyUI/input/`へ手で置く必要はない
- **解像度は入力画像のサイズをそのまま使う。** `width` / `height`を書くと拒否される
- `batch_size`は1のみ。LoRAは併用できる
- 入力画像は`inputs/`へ置く (git管理外)。上限サイズは`IMAGEGEN_MAX_SOURCE_BYTES` (既定32MiB)

## control (ControlNet)

参考画像からCannyで線画を取り、その構図を保ったまま生成する。txt2img / img2imgの両方で使える。

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `image` | string | 必須 | リポジトリ配下の相対パス。拡張子は`source`と同じ |
| `model` | string | 必須 | `~/ComfyUI/models/controlnet/`のファイル名。拡張子は`.safetensors` / `.pth` / `.ckpt` |
| `strength` | float | 1.0 | 0.0-10.0。効かせる強さ |
| `start_percent` | float | 0.0 | 0.0-1.0。効かせ始める進行度 |
| `end_percent` | float | 1.0 | 0.0-1.0。効かせ終える進行度。構図だけ借りるなら下げる |
| `low_threshold` | float | 0.4 | 0.01-0.99。Cannyの閾値。低いほど細かい線を拾う |
| `high_threshold` | float | 0.8 | 0.01-0.99 |

`low_threshold < high_threshold`、`start_percent < end_percent`を満たさないと拒否される。

```yaml
control:
  image: inputs/pose.png
  model: control_v11p_sd15_canny_fp16.safetensors
  strength: 0.9
  low_threshold: 0.3
  high_threshold: 0.7
```

- 指定するとテンプレートが`*_controlnet`へ自動的に切り替わる
- control画像は生成前にComfyUIへ自動でアップロードされる
- **前処理はCannyのみ。** pose / depthはpreprocessorのカスタムノードが要るため未対応
  ([Issue #37](https://github.com/Sylphy0052/agentic-imagegen/issues/37))
- 線が強く出すぎる場合は`low_threshold`を上げて細かい線を捨てるか、`strength`を下げる。
  写真やイラストをそのまま渡すと輪郭を拾いすぎ、元絵のエッジが残ったような絵になりやすい
- `generation.upscale` (hires fix) と併用できる。ControlNetが効くのは1段目だけで、
  `start_percent` / `end_percent`は1段目の進行度に対する指定として読む

## reference (IPAdapter)

参照画像をCLIP Visionで読み、その特徴 (人物の顔立ち・服装・画風) を効かせたまま生成する。
プロンプトだけでは揺れる要素を固定できる。txt2img / img2imgの両方で使える。

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `image` | string | 必須 | リポジトリ配下の相対パス。拡張子は`source`と同じ |
| `model` | string | 必須 | `~/ComfyUI/models/ipadapter/`のファイル名。拡張子は`.safetensors` / `.bin` / `.pth` / `.ckpt` |
| `clip_vision` | string | 必須 | `~/ComfyUI/models/clip_vision/`のファイル名。拡張子は`.safetensors` / `.bin` / `.pt` / `.ckpt` |
| `weight` | float | 1.0 | 0.0-3.0。効かせる強さ |
| `weight_type` | enum | `linear` | 下記15種 |
| `start_percent` | float | 0.0 | 0.0-1.0 |
| `end_percent` | float | 1.0 | 0.0-1.0 |

`weight_type`に指定できる値:

```text
linear  ease in  ease out  ease in-out  reverse in-out
weak input  weak output  weak middle  strong middle
style transfer  composition  strong style transfer
style and composition  style transfer precise  composition precise
```

```yaml
reference:
  image: inputs/character.png
  model: ip-adapter-plus_sd15.safetensors
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
  weight: 0.8
  weight_type: linear
```

- 指定するとテンプレートが`*_ipadapter`へ自動的に切り替わる
- **[ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) の導入が要る。**
  未導入だとノードが無く投入が拒否される (exit code 5)
- モデルとCLIP Visionは対応関係がある。`ip-adapter-plus_sd15`にはViT-Hを使う
- `weight`は0.6-0.9が扱いやすい。1.0を超えると参照画像へ寄りすぎ、プロンプトが効かなくなる
- **背景まで参照画像に引きずられる場合は`weight_type: style transfer`を使う。**
  既定の`linear`は参照画像を背景ごと読むため、プロンプトで別の場所を指定しても元絵の背景が出る。
  weightを下げても背景が変わる前に服装や顔立ちが崩れるだけで、切り分けは`weight_type`で行う
- ControlNetと併用できる。構図をControlNet、特徴をIPAdapterが担う

同一キャラクタを別の構図で出す手順は
[character-consistency.md](../.claude/skills/imagegen/references/character-consistency.md) を参照。

## text (テキスト合成)

SD1.5 / SDXL系のモデルは日本語をほぼ描けない。読める文字が必要な場合は生成に任せず、
生成後にPillowで合成する。生成そのものの挙動は変わらない。

`layers`は指定順に描画し、後のものが上へ重なる。最大10件。

| キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- |
| `content` | string | 必須 | 1-500文字。改行のみ制御文字として許可 |
| `font` | string | 必須 | `fonts/`配下のファイル名。拡張子は`.ttf` / `.otf` / `.ttc` |
| `font_index` | int | 0 | 0-64。`.ttc` (コレクション) 内の書体を選ぶ |
| `size` | int | 必須 | 1-512 |
| `color` | string | `#ffffff` | `#rgb` / `#rrggbb` / `#rrggbbaa`。色名は不可 |
| `anchor` | enum | `center` | 9分割の基準位置 (下記) |
| `offset` | [int, int] | `[0, 0]` | anchorからのずれ (px)。各-8192-8192 |
| `max_width` | float | `null` | 折り返し幅。1.0以下は画像幅に対する比率、1.0超はpx |
| `line_spacing` | float | 1.2 | 0.5-5.0 |
| `align` | enum | `center` | `left` / `center` / `right` |
| `opacity` | float | 1.0 | 0.0-1.0。レイヤ全体の不透明度 |
| `rotation` | float | 0.0 | -180.0-180.0。度、反時計回り |
| `direction` | enum | `horizontal` | `horizontal` / `vertical` (縦書き) |
| `stroke` | mapping | `null` | 縁取り |
| `shadow` | mapping | `null` | 影 |
| `box` | mapping | `null` | 文字の背後へ敷く矩形 |

`anchor`に指定できる値:

```text
top-left      top-center     top-right
middle-left   center         middle-right
bottom-left   bottom-center  bottom-right
```

### text.layers[].stroke / shadow / box

| ブロック | キー | 型 | 既定値 | 値域 |
| --- | --- | --- | --- | --- |
| `stroke` | `width` | int | 2 | 1-64 |
| `stroke` | `color` | string | `#000000` | 色指定と同じ |
| `shadow` | `offset` | [int, int] | `[4, 4]` | 各-8192-8192 |
| `shadow` | `blur` | float | 4.0 | 0.0-64.0 |
| `shadow` | `color` | string | `#000000` | 色指定と同じ |
| `shadow` | `opacity` | float | 0.5 | 0.0-1.0 |
| `box` | `color` | string | `#000000` | 色指定と同じ |
| `box` | `opacity` | float | 0.5 | 0.0-1.0 |
| `box` | `padding` | [int, int] | `[16, 16]` | (横, 縦) 各0-512 |
| `box` | `radius` | int | 0 | 0-512。角丸の半径 |

1レイヤの描画順はbox -> shadow -> stroke + text。`rotation`と`opacity`はこの3つを描いたあとに
レイヤ全体へ掛かる。

```yaml
text:
  layers:
    - content: 夜の街
      font: NotoSansJP-Bold.ttf
      size: 72
      color: "#ffffff"
      anchor: top-center
      offset: [0, 48]
      max_width: 0.8
      stroke:
        width: 4
        color: "#101020"
      box:
        color: "#000000"
        opacity: 0.55
        padding: [24, 16]
        radius: 12
```

- 生成そのままの画像は残り、合成結果は`image_0001_text.png`として別に出力される
- フォントは`fonts/`へ置く (git管理外)。置き方は [fonts-setup.md](fonts-setup.md)。
  探索ルートは`IMAGEGEN_FONTS_ROOT`
- **見つからないフォントは別の書体へ代替せず失敗する** (exit code 10)。
  意図しない書体で出力されるより、その場で止める方が扱いやすいため
- ネガティブプロンプトへ`text, watermark`を入れておくと、モデルが描く崩れた文字を減らせる
- **ルビ・縦中横・縦書き時の句読点の位置補正は未対応**
  ([Issue #40](https://github.com/Sylphy0052/agentic-imagegen/issues/40))

生成済みの画像へ後から合成する場合は`compose`を使う。入力画像は変更しない。
テキスト定義のYAMLは`text`ブロックだけを書いてもよいし、生成に使ったSpecをそのまま渡してもよい
(`text`セクションだけを読む)。

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml
uv run imagegen compose inputs/base.png specs/generated/caption.yaml -o outputs/caption.png
```

日本語を直接描けるモデル (Qwen-Image) の評価と導入条件は
[plan/phase5-japanese-text.md](plan/phase5-japanese-text.md) を参照。
現在の実行環境ではメモリが足りず動かせないため、合成方式を既定とする。

## output

| キー | 型 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `directory` | string | `IMAGEGEN_OUTPUT_ROOT` (既定`outputs`) | 出力ルート |
| `prefix` | string | `imagegen` | 出力ディレクトリ名の接頭辞。英数字始まりの`[A-Za-z0-9._-]`のみ |

出力先が作業ルートの外を指す場合は拒否する。
実際の出力は`<directory>/<YYYY-MM-DD>/<prefix>/`配下で、同じ日に同じprefixで再実行した場合は
連番ディレクトリを作り既存の結果を上書きしない。

```text
outputs/
└── 2026-08-12/
    └── blue_hair/
        ├── image_0001.png
        ├── image_0001_text.png   # text を指定した場合のみ
        └── metadata.json
```

### metadata.json

生成結果と同じディレクトリへ出力する。再現に必要な情報をここへ集約する。

| キー | 内容 |
| --- | --- |
| `prompt_id` | ComfyUI側の実行ID |
| `workflow` | 使用したworkflow名 |
| `workflow_hash` | Workflowテンプレートのダイジェスト (`sha256:...`) |
| `created_at` | 生成時刻 (タイムゾーン付き) |
| `resolved_seed` | 実際に使われたseed |
| `backend` | 実行基盤 (`comfyui_version` / `devices`)。取得に失敗した場合は`null` |
| `spec` | preset展開後のSpec全体。適用したpreset名も含む |
| `outputs` | 出力ファイル名 |
| `text` | テキスト合成の結果 (解決したフォントの実パスと合成後のファイル名)。合成しなかった場合は`null` |

- `seed`に`-1`を指定した場合、実際に使われた値は`resolved_seed`に入る。
  同じ画を再現したい場合はその値をSpecへ書き戻す
- `workflow_hash`は正規化したJSONから取るため、インデントや鍵の順序が変わっただけでは動かない。
  同じSpecで結果が変わったときに、テンプレート自体が変わったのかを切り分けられる。
  実行基盤が変わったのかは`backend`と併せて見る
- `batch_size`が1より大きくテキスト合成の一部だけが失敗した場合、`text`にはそれまでに成功した
  分の`outputs` / `fonts`と、失敗理由を示す`error`が入る。1件も成功しなかった場合のみ`text`は`null`

## 組み合わせの可否

| | LoRA | img2img | hires fix | ControlNet | IPAdapter | text | clip_skip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LoRA** | - | 可 | 可 | 可 | 可 | 可 | 可 |
| **img2img** | 可 | - | 可 | 可 | 可 | 可 | 可 |
| **hires fix** | 可 | 可 | - | 可 | **不可** | 可 | 可 |
| **ControlNet** | 可 | 可 | 可 | - | 可 | 可 | 可 |
| **IPAdapter** | 可 | 可 | **不可** | 可 | - | 可 | 可 |
| **DiT系 (unet/clip/vae)** | **不可** | 可 | 可 | **不可** | **不可** | 可 | **不可** |
| **clip_skip** | 可 | 可 | 可 | 可 | 可 | 可 | - |

`text`は生成後の後処理のため、どの構成とも併用できる。

**hires fixとControlNetを併用した場合、ControlNetが効くのは1段目だけになる。**
構図は1段目で決まるため、2段目は拡大後の解像度で描き足すことに徹する。2段目にも効かせると
拡大後の解像度でControlNetの推論が追加で走り、得られるものに対して所要時間が伸びすぎる。
`control.end_percent`は1段目の進行度に対する指定として読む。

不可の組み合わせを指定した場合はSpecの検証時 (exit code 2) に拒否する。理由と着手条件は
[Issue #38](https://github.com/Sylphy0052/agentic-imagegen/issues/38) (hires fixとIPAdapterの併用) と
[Issue #39](https://github.com/Sylphy0052/agentic-imagegen/issues/39)
(DiT系とLoRA / ControlNet / IPAdapterの併用) にまとめてある。

## Workflowテンプレートの決まり方

Specの内容から自動的に決まる。`uv run imagegen validate`の`Workflow:`行で確認できる。

`<task>`へ次の順で接尾辞を足す。

| 順 | 条件 | 接尾辞 |
| --- | --- | --- |
| 1 | `model.unet`を指定 | `_unet` |
| 1 | `model.loras`が空でない (`model.unet`が無い場合) | `_lora` |
| 2 | `generation.upscale`を指定 | `_hires` (`upscale.model`を指定した場合は`_hires_model`) |
| 3 | `control`を指定 | `_controlnet` |
| 4 | `reference`を指定 | `_ipadapter` |

`_unet`と`_lora`は同じ位置に入り、両立しない (DiT系とLoRAの併用は拒否される)。

例: `task: txt2img`にLoRAとControlNetを指定すると`txt2img_lora_controlnet`。
`task: img2img`にDiT系モデルとhires fixを指定すると`img2img_unet_hires`。
hires fixで`upscale.model`まで指定すると`img2img_unet_hires_model`。

テンプレートの一覧と各構成のノード内訳、作り直しの手順は
[workflows/README.md](../workflows/README.md) を参照。
読み込み時にノードID・class_type・必要な入力キー・ノード間の接続を検証し、
1つでも想定と違えば注入せずに失敗する (exit code 4)。

## 環境変数による上限

このリファレンスに書いた値域はハード制約で、実運用上の上限は環境変数で別途課す。
解像度の上限はベース解像度 (`IMAGEGEN_MAX_PIXELS`) とhires fixの拡大後
(`IMAGEGEN_MAX_UPSCALED_PIXELS`) で別に持つ。前者は1段目の生成負荷を抑えるもの、
後者は拡大時のピークメモリを抑えるもので、目的が違うため同じ値では縛らない。
両方を満たさない指定は拒否される (exit code 2、環境変数の設定値自体が不正なら9)。

変数名・既定値・用途の一覧は [CLAUDE.mdの「環境変数」](../CLAUDE.md#環境変数) を参照。
