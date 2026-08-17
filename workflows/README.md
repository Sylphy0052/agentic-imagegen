# Workflowテンプレート

ComfyUIで実行するWorkflowを **API形式JSON** で置く場所。

## 重要な前提

- Workflowは人間がComfyUI GUIで作成する。LLMにノードや接続を組み立てさせない。
- コードから差し替えてよいのは、許可済みパラメータのみ。
  - `positive_prompt` / `negative_prompt` / `checkpoint` / `seed` / `steps` / `cfg`
  - `sampler` / `scheduler` / `width` / `height` / `batch_size` / `filename_prefix`
  - LoRA用テンプレートでは `lora_name` / `strength_model` / `strength_clip`
- 実行を許可するworkflowはallowlistで管理する。定義は
  [src/agentic_imagegen/workflows/injector.py](../src/agentic_imagegen/workflows/injector.py)
  の `ALLOWED_WORKFLOWS` を参照。

## 同梱テンプレート

| ファイル | 用途 |
| --- | --- |
| `txt2img.json` | text-to-image (ComfyUI標準のtxt2imgグラフと同じノード構成) |
| `txt2img_lora.json` | LoRA付きtext-to-image (`txt2img` に `LoraLoader` を3段挟んだ構成) |
| `img2img.json` | image-to-image (`EmptyLatentImage` を `LoadImage` + `VAEEncode` へ置き換えた構成) |
| `img2img_lora.json` | LoRA付きimage-to-image (`img2img` に `LoraLoader` を3段挟んだ構成) |
| `txt2img_hires.json` | hires fix付きtext-to-image |
| `txt2img_lora_hires.json` | LoRA + hires fix |
| `img2img_hires.json` | hires fix付きimage-to-image |
| `img2img_lora_hires.json` | LoRA + hires fix (img2img) |
| `txt2img_controlnet.json` | ControlNet付きtext-to-image |
| `txt2img_controlnet_raw.json` | ControlNet付きtext-to-image / 制御画像を前処理せずそのまま渡す |
| `txt2img_lora_controlnet.json` | LoRA + ControlNet |
| `txt2img_lora_controlnet_raw.json` | LoRA + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_controlnet.json` | ControlNet付きimage-to-image |
| `img2img_controlnet_raw.json` | ControlNet付きimage-to-image / 制御画像を前処理せずそのまま渡す |
| `img2img_lora_controlnet.json` | LoRA + ControlNet (img2img) |
| `img2img_lora_controlnet_raw.json` | LoRA + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_hires_controlnet.json` | hires fix + ControlNet (ControlNetが効くのは1段目だけ) |
| `txt2img_hires_controlnet_raw.json` | hires fix + ControlNet (ControlNetが効くのは1段目だけ) / 制御画像を前処理せずそのまま渡す |
| `txt2img_lora_hires_controlnet.json` | LoRA + hires fix + ControlNet |
| `txt2img_lora_hires_controlnet_raw.json` | LoRA + hires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_hires_controlnet.json` | hires fix + ControlNet (img2img) |
| `img2img_hires_controlnet_raw.json` | hires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_lora_hires_controlnet.json` | LoRA + hires fix + ControlNet (img2img) |
| `img2img_lora_hires_controlnet_raw.json` | LoRA + hires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_ipadapter.json` | IPAdapter付きtext-to-image |
| `txt2img_lora_ipadapter.json` | LoRA + IPAdapter |
| `img2img_ipadapter.json` | IPAdapter付きimage-to-image |
| `img2img_lora_ipadapter.json` | LoRA + IPAdapter (img2img) |
| `txt2img_controlnet_ipadapter.json` | ControlNet + IPAdapter |
| `txt2img_controlnet_raw_ipadapter.json` | ControlNet + IPAdapter / 制御画像を前処理せずそのまま渡す |
| `txt2img_lora_controlnet_ipadapter.json` | LoRA + ControlNet + IPAdapter |
| `txt2img_lora_controlnet_raw_ipadapter.json` | LoRA + ControlNet + IPAdapter / 制御画像を前処理せずそのまま渡す |
| `img2img_controlnet_ipadapter.json` | ControlNet + IPAdapter (img2img) |
| `img2img_controlnet_raw_ipadapter.json` | ControlNet + IPAdapter (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_lora_controlnet_ipadapter.json` | LoRA + ControlNet + IPAdapter (img2img) |
| `img2img_lora_controlnet_raw_ipadapter.json` | LoRA + ControlNet + IPAdapter (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_unet.json` | UNet / text encoder / VAEを別々に読むtext-to-image (DiT系) |
| `txt2img_unet_hires.json` | DiT系 + hires fix |
| `img2img_unet.json` | DiT系のimage-to-image |
| `img2img_unet_hires.json` | DiT系 + hires fix (img2img) |
| `txt2img_unet_beta57.json` | DiT系 / beta57 スケジュール (KSamplerではなくSamplerCustomAdvanced) |
| `txt2img_unet_beta57_hires.json` | DiT系 / beta57 + hires fix |
| `txt2img_unet_beta57_hires_model.json` | DiT系 / beta57 + モデル拡大のhires fix |
| `img2img_unet_beta57.json` | DiT系 / beta57 (img2img) |
| `img2img_unet_beta57_hires.json` | DiT系 / beta57 + hires fix (img2img) |
| `img2img_unet_beta57_hires_model.json` | DiT系 / beta57 + モデル拡大のhires fix (img2img) |
| `txt2img_unet_lora.json` | DiT系 + LoRA |
| `txt2img_unet_lora_hires.json` | DiT系 + LoRA + hires fix |
| `txt2img_unet_lora_hires_model.json` | DiT系 + LoRA + モデル拡大のhires fix |
| `img2img_unet_lora.json` | DiT系 + LoRA (img2img) |
| `img2img_unet_lora_hires.json` | DiT系 + LoRA + hires fix (img2img) |
| `img2img_unet_lora_hires_model.json` | DiT系 + LoRA + モデル拡大のhires fix (img2img) |
| `txt2img_unet_beta57_lora.json` | DiT系 / beta57 + LoRA |
| `txt2img_unet_beta57_lora_hires.json` | DiT系 / beta57 + LoRA + hires fix |
| `txt2img_unet_beta57_lora_hires_model.json` | DiT系 / beta57 + LoRA + モデル拡大のhires fix |
| `img2img_unet_beta57_lora.json` | DiT系 / beta57 + LoRA (img2img) |
| `img2img_unet_beta57_lora_hires.json` | DiT系 / beta57 + LoRA + hires fix (img2img) |
| `img2img_unet_beta57_lora_hires_model.json` | DiT系 / beta57 + LoRA + モデル拡大のhires fix (img2img) |
| `txt2img_hires_model.json` | アップスケールモデルで拡大するhires fix |
| `txt2img_lora_hires_model.json` | LoRA + モデル拡大のhires fix |
| `img2img_hires_model.json` | モデル拡大のhires fix (img2img) |
| `img2img_lora_hires_model.json` | LoRA + モデル拡大のhires fix (img2img) |
| `txt2img_unet_hires_model.json` | DiT系 + モデル拡大のhires fix |
| `img2img_unet_hires_model.json` | DiT系 + モデル拡大のhires fix (img2img) |
| `txt2img_hires_model_controlnet.json` | モデル拡大のhires fix + ControlNet |
| `txt2img_hires_model_controlnet_raw.json` | モデル拡大のhires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `txt2img_lora_hires_model_controlnet.json` | LoRA + モデル拡大のhires fix + ControlNet |
| `txt2img_lora_hires_model_controlnet_raw.json` | LoRA + モデル拡大のhires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_hires_model_controlnet.json` | モデル拡大のhires fix + ControlNet (img2img) |
| `img2img_hires_model_controlnet_raw.json` | モデル拡大のhires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_lora_hires_model_controlnet.json` | LoRA + モデル拡大のhires fix + ControlNet (img2img) |
| `img2img_lora_hires_model_controlnet_raw.json` | LoRA + モデル拡大のhires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae.json` | 外部VAEへ差し替えたtext-to-image (`CheckpointLoaderSimple`のVAE出力を`VAELoader`へ差し替えた構成) |
| `txt2img_vae_lora.json` | 外部VAE + LoRA |
| `img2img_vae.json` | 外部VAEへ差し替えたimage-to-image |
| `img2img_vae_lora.json` | 外部VAE + LoRA (img2img) |
| `txt2img_vae_hires.json` | 外部VAE + hires fix |
| `txt2img_vae_lora_hires.json` | 外部VAE + LoRA + hires fix |
| `img2img_vae_hires.json` | 外部VAE + hires fix (img2img) |
| `img2img_vae_lora_hires.json` | 外部VAE + LoRA + hires fix (img2img) |
| `txt2img_vae_hires_model.json` | 外部VAE + モデル拡大のhires fix |
| `txt2img_vae_lora_hires_model.json` | 外部VAE + LoRA + モデル拡大のhires fix |
| `img2img_vae_hires_model.json` | 外部VAE + モデル拡大のhires fix (img2img) |
| `img2img_vae_lora_hires_model.json` | 外部VAE + LoRA + モデル拡大のhires fix (img2img) |
| `txt2img_vae_controlnet.json` | 外部VAE + ControlNet |
| `txt2img_vae_controlnet_raw.json` | 外部VAE + ControlNet / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_lora_controlnet.json` | 外部VAE + LoRA + ControlNet |
| `txt2img_vae_lora_controlnet_raw.json` | 外部VAE + LoRA + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_controlnet.json` | 外部VAE + ControlNet (img2img) |
| `img2img_vae_controlnet_raw.json` | 外部VAE + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_lora_controlnet.json` | 外部VAE + LoRA + ControlNet (img2img) |
| `img2img_vae_lora_controlnet_raw.json` | 外部VAE + LoRA + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_hires_controlnet.json` | 外部VAE + hires fix + ControlNet |
| `txt2img_vae_hires_controlnet_raw.json` | 外部VAE + hires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_lora_hires_controlnet.json` | 外部VAE + LoRA + hires fix + ControlNet |
| `txt2img_vae_lora_hires_controlnet_raw.json` | 外部VAE + LoRA + hires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_hires_controlnet.json` | 外部VAE + hires fix + ControlNet (img2img) |
| `img2img_vae_hires_controlnet_raw.json` | 外部VAE + hires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_lora_hires_controlnet.json` | 外部VAE + LoRA + hires fix + ControlNet (img2img) |
| `img2img_vae_lora_hires_controlnet_raw.json` | 外部VAE + LoRA + hires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_hires_model_controlnet.json` | 外部VAE + モデル拡大のhires fix + ControlNet |
| `txt2img_vae_hires_model_controlnet_raw.json` | 外部VAE + モデル拡大のhires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_lora_hires_model_controlnet.json` | 外部VAE + LoRA + モデル拡大のhires fix + ControlNet |
| `txt2img_vae_lora_hires_model_controlnet_raw.json` | 外部VAE + LoRA + モデル拡大のhires fix + ControlNet / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_hires_model_controlnet.json` | 外部VAE + モデル拡大のhires fix + ControlNet (img2img) |
| `img2img_vae_hires_model_controlnet_raw.json` | 外部VAE + モデル拡大のhires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_lora_hires_model_controlnet.json` | 外部VAE + LoRA + モデル拡大のhires fix + ControlNet (img2img) |
| `img2img_vae_lora_hires_model_controlnet_raw.json` | 外部VAE + LoRA + モデル拡大のhires fix + ControlNet (img2img) / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_ipadapter.json` | 外部VAE + IPAdapter |
| `txt2img_vae_lora_ipadapter.json` | 外部VAE + LoRA + IPAdapter |
| `img2img_vae_ipadapter.json` | 外部VAE + IPAdapter (img2img) |
| `img2img_vae_lora_ipadapter.json` | 外部VAE + LoRA + IPAdapter (img2img) |
| `txt2img_vae_controlnet_ipadapter.json` | 外部VAE + ControlNet + IPAdapter |
| `txt2img_vae_controlnet_raw_ipadapter.json` | 外部VAE + ControlNet + IPAdapter / 制御画像を前処理せずそのまま渡す |
| `txt2img_vae_lora_controlnet_ipadapter.json` | 外部VAE + LoRA + ControlNet + IPAdapter |
| `txt2img_vae_lora_controlnet_raw_ipadapter.json` | 外部VAE + LoRA + ControlNet + IPAdapter / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_controlnet_ipadapter.json` | 外部VAE + ControlNet + IPAdapter (img2img) |
| `img2img_vae_controlnet_raw_ipadapter.json` | 外部VAE + ControlNet + IPAdapter (img2img) / 制御画像を前処理せずそのまま渡す |
| `img2img_vae_lora_controlnet_ipadapter.json` | 外部VAE + LoRA + ControlNet + IPAdapter (img2img) |
| `img2img_vae_lora_controlnet_raw_ipadapter.json` | 外部VAE + LoRA + ControlNet + IPAdapter (img2img) / 制御画像を前処理せずそのまま渡す |

どれを使うかは `task` と `model.loras` / `generation.upscale` / `control` / `reference` の有無、
`model.unet` の指定で自動的に決まる。定義は
[src/agentic_imagegen/workflows/injector.py](../src/agentic_imagegen/workflows/injector.py)
の `resolve_workflow_name` を参照。

`txt2img.json` はComfyUI標準のデフォルトグラフと同じノードID構成を採用している。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 3 | `KSampler` | seed / steps / cfg / sampler_name / scheduler |
| 4 | `CheckpointLoaderSimple` | ckpt_name |
| 5 | `EmptyLatentImage` | width / height / batch_size |
| 6 | `CLIPTextEncode` | positive prompt |
| 7 | `CLIPTextEncode` | negative prompt |
| 8 | `VAEDecode` | - |
| 9 | `SaveImage` | filename_prefix |
| 70 | `CLIPSetLastLayer` | stop_at_clip_layer (clip skip) |

`CLIPTextEncode` (6, 7) の `clip` 入力はCLIPの供給元へ直結せず、必ず70
(`CLIPSetLastLayer`) 経由で受ける。70の `clip` 入力は素の構成では
`CheckpointLoaderSimple` (4のスロット1) から受け、`stop_at_clip_layer` の既定値は
`-1` (ComfyUI既定と同値の素通し)。`model.clip_skip` を指定しない限り出力は
このノードが無かった頃と完全に一致する。全29テンプレートへ無条件に挿入してあり、
派生テンプレートを増やしていない ([Issue #60](https://github.com/Sylphy0052/agentic-imagegen/issues/60))。

`txt2img_lora.json` は上記に `LoraLoader` を3段挟んだもの。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 10 | `LoraLoader` | 1本目 (`CheckpointLoaderSimple` からMODEL / CLIPを受ける) |
| 11 | `LoraLoader` | 2本目 (10から受ける) |
| 12 | `LoraLoader` | 3本目 (11から受ける) |

12のMODELが `KSampler.model` へ、CLIPが70 (`CLIPSetLastLayer`) の `clip` へ繋がる
(LoRA適用後のCLIPに対して層を打ち切るため、`CLIPTextEncode` へは直結しない)。
`VAEDecode.vae` は `LoraLoader` がVAEを出さないため `CheckpointLoaderSimple` 直結のまま。

指定されたLoRAは先頭のスロットから順に割り当てる。余ったスロットは
`strength_model` / `strength_clip` を0にして無効化する。ComfyUIは `lora_name` に
実在するファイル名を要求するため空にはできず、直前のLoRA名を使い回す。

`img2img.json` は `txt2img.json` の `EmptyLatentImage` (5) を外し、代わりに次を持つ。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 10 | `LoadImage` | image (ComfyUIのinput配下の名前) |
| 11 | `VAEEncode` | pixelsは10から、vaeは `CheckpointLoaderSimple` から受ける |

`KSampler.latent_image` は11から受ける。`denoise` はimg2imgで意味を持つため注入対象に含む。

`LoadImage` が参照できるのはComfyUIの `input/` 直下だけで、サブフォルダに置いたファイルは
候補に現れない。入力画像は生成前に `POST /upload/image` で直下へ送っており、名前は
`imagegen_<内容ダイジェスト>_<元のファイル名>` になる。同じ画像なら同じ名前へ落ち着く。

`img2img_lora.json` は `img2img.json` へ `LoraLoader` を3段挟んだもの。
**ノードIDは20-22を使う。** `txt2img_lora.json` と同じ10-12にすると
`LoadImage` (10) と `VAEEncode` (11) を上書きしてしまうため。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 20 | `LoraLoader` | 1本目 (`CheckpointLoaderSimple` からMODEL / CLIPを受ける) |
| 21 | `LoraLoader` | 2本目 (20から受ける) |
| 22 | `LoraLoader` | 3本目 (21から受ける) |

`LoraLoader` はVAEを出さないため、`VAEEncode.vae` と `VAEDecode.vae` は
`CheckpointLoaderSimple` 直結のままにする。

## 派生テンプレートは合成スクリプトで生成する

`txt2img.json` だけが手書きのベースで、残りは
[scripts/build_workflow_templates.py](../scripts/build_workflow_templates.py) が生成する。

```bash
uv run python scripts/build_workflow_templates.py          # 生成
uv run python scripts/build_workflow_templates.py --check  # 差分がないか確認するだけ
```

```text
txt2img.json  (手書きベース)
  ├─ img2img            EmptyLatentImage -> LoadImage + VAEEncode
  ├─ *_lora             CheckpointLoader の後に LoraLoader を3段
  ├─ *_hires            KSampler の後に LatentUpscaleBy + 2段目 KSampler
  ├─ *_hires_model      KSampler の後に VAEDecode + アップスケールモデル + VAEEncode
  │                     + 2段目 KSampler (pixel空間で拡大する版)
  ├─ *_controlnet       CLIPTextEncode と KSampler の間に ControlNet
  ├─ *_ipadapter        MODEL の経路に IPAdapterAdvanced
  ├─ *_vae              CheckpointLoaderのVAE出力 ([node, 2]) を参照する全ノードを
  │                     VAELoader へ差し替え (checkpoint系のみ、他の軸をかけた後に適用)
  ├─ txt2img_unet       CheckpointLoader -> UNETLoader + CLIPLoader + VAELoader
  └─ *_beta57           KSampler -> RandomNoise + KSamplerSelect
                        + BetaSamplingScheduler + CFGGuider + SamplerCustomAdvanced
                        (DiT系のみ、全ての軸をかけた後に適用)
```

`*_ipadapter` はKSamplerのMODEL入力だけを差し替えるため、positive / negativeを
差し替える `*_controlnet` と同時にかけられる。

組み合わせが70種を超えても手で書かないのは、ノード参照を間違えたときに
**形は正しいまま意味だけ壊れる**ためである (実際に一度踏んでいる)。
スクリプトは生成後に次を検査する。

- 存在しないノードIDや範囲外の出力スロットを参照していないか
- ベースのノードを潰していないか (class_typeが変わっていないか)

### 軸を1本足すとき

生成しうるテンプレート名と、それを構成する軸の並びは
[src/agentic_imagegen/workflows/axes.py](../src/agentic_imagegen/workflows/axes.py) が
一元管理する。合成スクリプト・binding定義 (`adapters/comfyui/workflow.py`)・
allowlist (`workflows/injector.py`) の3か所はいずれもこの列挙を辿るだけなので、
軸を足しても組み合わせの数だけ宣言を書き足す必要はない。
具体的な手順は `axes.py` のモジュールdocstringにある。

ノードIDは用途ごとに帯を分けている。

| 帯 | 用途 |
| --- | --- |
| 3-9 | ベース (txt2img標準グラフ) |
| 10-12 | txt2img系のLoRA / img2imgのLoadImage・VAEEncode |
| 20-22 | img2img系のLoRA |
| 30-31 | hires fix (LatentUpscaleBy / 2段目KSampler) |
| 32-36 | モデル拡大のhires fix (VAEDecode / UpscaleModelLoader / ImageUpscaleWithModel / ImageScaleBy / VAEEncode) |
| 40-43 | ControlNet (LoadImage / Canny / ControlNetLoader / ControlNetApplyAdvanced)。`_controlnet_raw`はCanny (41) を持たず、`ControlNetApplyAdvanced`が`LoadImage`を直接読む |
| 50-53 | IPAdapter (LoadImage / IPAdapterModelLoader / CLIPVisionLoader / IPAdapterAdvanced) |
| 60-62 | DiT系のローダー分割 (UNETLoader / CLIPLoader / VAELoader) |
| 70 | clip skip (CLIPSetLastLayer、全テンプレート共通で1個) |
| 80 | 外部VAE (VAELoader、checkpoint系のみ) |
| 90-94 | beta57の1段目 (RandomNoise / KSamplerSelect / BetaSamplingScheduler / CFGGuider / SamplerCustomAdvanced) |
| 95-100 | beta57の2段目 (hires fixの描き足し。98が `SplitSigmasDenoise`) |

hires fixとControlNetの組み合わせ (`*_hires_controlnet`) は、hiresを重ねてからControlNetを
かける順で合成している。この順にすると `ControlNetApplyAdvanced` が差し替えるのは1段目の
KSamplerだけになり、2段目は素の `CLIPTextEncode` を受けたまま残る。構図は1段目で決まるため、
2段目は拡大後の解像度で描き足すことに徹する。

`*_hires_model` は latent のまま拡大する代わりに、一度pixelへ戻して
`ImageUpscaleWithModel` で拡大し、`ImageScaleBy` で要求された倍率へ合わせてから
`VAEEncode` で戻す。`ImageScaleBy` を必ず挟むのは、アップスケールモデルの倍率が
固定 (4xなど) で、Specの `generation.upscale.scale` と一致しないためである。
増やした `VAEDecode` / `VAEEncode` は最終段の `VAEDecode` と同じVAEを見る
(DiT系では `VAELoader` から受ける)。

hires fixとIPAdapterの組み合わせは作っていない (Specの検証で同時指定を拒否している)。
[Issue #38](https://github.com/Sylphy0052/agentic-imagegen/issues/38) を参照。

`*_ipadapter` は [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
のノードを使う。未導入のComfyUIでは投入が拒否される。

`*_unet` はDiT系モデル (Animaなど) 向けで、`CheckpointLoaderSimple` を3つのローダーへ
置き換える。UNet単体で配布されるモデルはtext encoderとVAEを同梱しておらず、1ファイルから
MODEL / CLIP / VAEを取り出す前提が崩れるため。

ローダーを分けてから他の派生をかける順で合成する。逆順にすると、後から足した2段目の
KSamplerが `CheckpointLoaderSimple` を見たまま残る。img2img では入力画像を `VAEEncode` する
側も `VAELoader` から受け直す。

LoRA / ControlNet / IPAdapterとの組み合わせは作っていない (Specの検証で併用を拒否している)。
`control_v11p_sd15_*` も `ip-adapter-plus_sd15` もSD1.5向けで、DiT系のUNetへは適用できない。

`*_beta57` はDiT系向けで、`KSampler` 1ノードが担っていた sampling を
ComfyUI標準の5ノードへ分解する。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 90 | `RandomNoise` | noise_seed |
| 91 | `KSamplerSelect` | sampler_name |
| 92 | `BetaSamplingScheduler` | steps / alpha=0.5 / beta=0.7 |
| 93 | `CFGGuider` | cfg / positive / negative / model |
| 94 | `SamplerCustomAdvanced` | 上の4つを束ねてsamplingする |

分けるのは、Anima系の配布元が推奨する `beta57` (beta分布の alpha=0.5 / beta=0.7) を
KSamplerの `scheduler` 欄から選べないためである。KSamplerが選べる `beta` は
ComfyUI既定の alpha=0.6 / beta=0.6 で固定されている。

hires fixの2段目 (`*_beta57_hires` / `*_beta57_hires_model`) は95-100を使い、
`SplitSigmasDenoise` (98) を挟んでKSamplerの `denoise` に相当する区間を取り出す。
KSamplerは `denoise` < 1 のとき `int(steps / denoise)` 手のスケジュールを引いて
後ろ `steps + 1` 個を使う実装のため、`BetaSamplingScheduler` へ渡すstep数も
同じ式で割り戻す (注入側の `_sigma_steps_for_denoise`)。

**他の軸を全て適用し終えた後にかける。** 先にかけると、hires fixが後から足す
2段目のKSamplerだけが置き換わらずに残る。

`*_vae` はcheckpoint系向けで、checkpoint同梱のVAEではなく外部VAE
(`vae-ft-mse-840000` / `klF8Anime2VAE` など、色褪せ・眠い線を避けるために
差し替えるのが通例) を使う版。`CheckpointLoaderSimple` を残したまま、そのVAE出力
(`[node, 2]`) を参照している全ノードを機械的に走査して `VAELoader` へ差し替える
(特定のノードIDを決め打ちしない)。決め打ちしないのは、`*_hires_model` のように
他の軸で `VAEDecode` / `VAEEncode` が増えるテンプレートでも取りこぼさないため。

**checkpoint系32件それぞれのローダー段の合成の最後にかける。** LoRA / hires fix /
ControlNet / IPAdapterを組み合わせ終えたグラフへ最後に `with_external_vae` を
かけることで、他の軸が増やしたVAEDecode / VAEEncodeの参照も一緒に拾える。
DiT系 (`*_unet`) は既に独自の `VAELoader` ルートを持つため対象外
([Issue #57](https://github.com/Sylphy0052/agentic-imagegen/issues/57))。

## GUIから書き出す手順

ComfyUI環境に合わせて作り直す場合の手順。

1. ComfyUIを起動し、txt2imgのWorkflowを組む (またはデフォルトグラフを読み込む)。
2. 設定画面で開発者向けオプション (Enable Dev mode Options) を有効にする。
3. `Save (API Format)` でJSONを書き出す。**通常の `Save` で保存したJSONは形式が異なり使用できない。**
4. 書き出したJSONを `workflows/txt2img.json` として置く。

## 既存テンプレートへノードを足す場合

GUIから書き出すのが原則だが、既存テンプレートに定型のノードを挟むだけであれば
機械的に組み立ててもよい。ただし次を必ず満たすこと。

1. `GET /object_info/<class_type>` で入力キーと出力スロットの並びを確認する
   (例: `CheckpointLoaderSimple` は0=MODEL / 1=CLIP / 2=VAE、`LoraLoader` は0=MODEL / 1=CLIP)
2. 既存テンプレートを読み込んで接続だけを差し替える。手書きしない
3. 生成後にノード参照の整合性を検査する (存在しないノードID・範囲外の出力スロットがないか)
4. **既存ノードを上書きしていないことを検査する。** 追加するノードIDが既存と衝突すると、
   参照の形は正しいまま意味だけが壊れる。`img2img_lora.json` を作った際、
   `txt2img_lora` と同じ10-12を使って `LoadImage` と `VAEEncode` を潰す事故が実際に起きた
5. **実機のComfyUIへ投入し、生成が成功することを確認する**

`txt2img_lora.json` と `img2img.json` はこの手順で作成し、実機 (ComfyUI 0.32.0 / Intel XPU) で
生成成功を確認している。ノードや接続を実行時に組み立てる設計は引き続き採らない。

## 構造検証について

読み込み時に、以下を検証してから注入する。1つでも一致しなければ `WorkflowValidationError` で即座に失敗する。

- 対象ノードIDが存在するか
- そのノードの `class_type` が想定と一致するか
- 注入に必要な `inputs` のキーが存在するか
- KSamplerの `positive` / `negative` / `latent_image` / `model` の接続先が想定ノードか

positiveとnegativeの接続が入れ替わっているようなケースも、この検証で検出される。
ノードIDを変えた場合は
[src/agentic_imagegen/adapters/comfyui/workflow.py](../src/agentic_imagegen/adapters/comfyui/workflow.py)
の `TXT2IMG_BINDING` を合わせて更新する。
