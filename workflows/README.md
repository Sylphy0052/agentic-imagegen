# Workflowテンプレート

ComfyUIで実行するWorkflowを **API形式JSON** で置く場所。

## 重要な前提

- Workflowは人間がComfyUI GUIで作成する。LLMにノードや接続を組み立てさせない。
- コードから差し替えてよいのは、許可済みパラメータのみ。
  - `positive_prompt` / `negative_prompt` / `checkpoint` / `seed` / `steps` / `cfg`
  - `sampler` / `scheduler` / `width` / `height` / `batch_size` / `filename_prefix`
  - LoRA用テンプレートでは `lora_name` / `strength_model` / `strength_clip`
- 実行を許可するworkflowは allowlist で管理する。定義は
  [src/agentic_imagegen/workflows/injector.py](../src/agentic_imagegen/workflows/injector.py)
  の `ALLOWED_WORKFLOWS` を参照。

## 同梱テンプレート

| ファイル | 用途 |
| --- | --- |
| `txt2img.json` | text-to-image (ComfyUI標準のtxt2imgグラフと同じノード構成) |
| `txt2img_lora.json` | LoRA付きtext-to-image (`txt2img` に `LoraLoader` を3段挟んだ構成) |
| `img2img.json` | image-to-image (`EmptyLatentImage` を `LoadImage` + `VAEEncode` へ置き換えた構成) |
| `img2img_lora.json` | LoRA付き image-to-image (`img2img` に `LoraLoader` を3段挟んだ構成) |
| `txt2img_hires.json` | hires fix 付き text-to-image |
| `txt2img_lora_hires.json` | LoRA + hires fix |
| `img2img_hires.json` | hires fix 付き image-to-image |
| `img2img_lora_hires.json` | LoRA + hires fix (img2img) |
| `txt2img_controlnet.json` | ControlNet付き text-to-image |
| `txt2img_lora_controlnet.json` | LoRA + ControlNet |
| `img2img_controlnet.json` | ControlNet付き image-to-image |
| `img2img_lora_controlnet.json` | LoRA + ControlNet (img2img) |
| `txt2img_ipadapter.json` | IPAdapter付き text-to-image |
| `txt2img_lora_ipadapter.json` | LoRA + IPAdapter |
| `img2img_ipadapter.json` | IPAdapter付き image-to-image |
| `img2img_lora_ipadapter.json` | LoRA + IPAdapter (img2img) |
| `txt2img_controlnet_ipadapter.json` | ControlNet + IPAdapter |
| `txt2img_lora_controlnet_ipadapter.json` | LoRA + ControlNet + IPAdapter |
| `img2img_controlnet_ipadapter.json` | ControlNet + IPAdapter (img2img) |
| `img2img_lora_controlnet_ipadapter.json` | LoRA + ControlNet + IPAdapter (img2img) |

どれを使うかは `task` と `model.loras` の有無で自動的に決まる。定義は
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

`txt2img_lora.json` は上記に `LoraLoader` を3段挟んだもの。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 10 | `LoraLoader` | 1本目 (`CheckpointLoaderSimple` から MODEL / CLIP を受ける) |
| 11 | `LoraLoader` | 2本目 (10から受ける) |
| 12 | `LoraLoader` | 3本目 (11から受ける) |

12の MODEL が `KSampler.model` へ、CLIP が `CLIPTextEncode` 2つの `clip` へ繋がる。
`VAEDecode.vae` は `LoraLoader` がVAEを出さないため `CheckpointLoaderSimple` 直結のまま。

指定されたLoRAは先頭のスロットから順に割り当てる。余ったスロットは
`strength_model` / `strength_clip` を0にして無効化する。ComfyUIは `lora_name` に
実在するファイル名を要求するため空にはできず、直前のLoRA名を使い回す。

`img2img.json` は `txt2img.json` の `EmptyLatentImage` (5) を外し、代わりに次を持つ。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 10 | `LoadImage` | image (ComfyUIのinput配下の名前) |
| 11 | `VAEEncode` | pixels は10から、vae は `CheckpointLoaderSimple` から受ける |

`KSampler.latent_image` は11から受ける。`denoise` はimg2imgで意味を持つため注入対象に含む。

`LoadImage` が参照できるのはComfyUIの `input/` 直下だけで、サブフォルダに置いたファイルは
候補に現れない。入力画像は生成前に `POST /upload/image` で直下へ送っており、名前は
`imagegen_<内容ダイジェスト>_<元のファイル名>` になる。同じ画像なら同じ名前へ落ち着く。

`img2img_lora.json` は `img2img.json` へ `LoraLoader` を3段挟んだもの。
**ノードIDは20-22を使う。** `txt2img_lora.json` と同じ10-12にすると
`LoadImage` (10) と `VAEEncode` (11) を上書きしてしまうため。

| ノードID | class_type | 役割 |
| --- | --- | --- |
| 20 | `LoraLoader` | 1本目 (`CheckpointLoaderSimple` から MODEL / CLIP を受ける) |
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
  ├─ *_controlnet       CLIPTextEncode と KSampler の間に ControlNet
  └─ *_ipadapter        MODEL の経路に IPAdapterAdvanced
```

`*_ipadapter` はKSamplerのMODEL入力だけを差し替えるため、positive / negative を
差し替える `*_controlnet` と同時にかけられる。

組み合わせが20種になっても手で書かないのは、ノード参照を間違えたときに
**形は正しいまま意味だけ壊れる**ためである (実際に一度踏んでいる)。
スクリプトは生成後に次を検査する。

- 存在しないノードIDや範囲外の出力スロットを参照していないか
- ベースのノードを潰していないか (class_typeが変わっていないか)

ノードIDは用途ごとに帯を分けている。

| 帯 | 用途 |
| --- | --- |
| 3-9 | ベース (txt2img標準グラフ) |
| 10-12 | txt2img系のLoRA / img2imgの LoadImage・VAEEncode |
| 20-22 | img2img系のLoRA |
| 30-31 | hires fix (LatentUpscaleBy / 2段目KSampler) |
| 40-43 | ControlNet (LoadImage / Canny / ControlNetLoader / ControlNetApplyAdvanced) |
| 50-53 | IPAdapter (LoadImage / IPAdapterModelLoader / CLIPVisionLoader / IPAdapterAdvanced) |

hires fix と ControlNet / IPAdapter の組み合わせは作っていない。両方かけると生成時間が
現実的でなく、必要になってから足せばよい (Specの検証で同時指定を拒否している)。

`*_ipadapter` は [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
のノードを使う。未導入のComfyUIでは投入が拒否される。

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
   (例: `CheckpointLoaderSimple` は 0=MODEL / 1=CLIP / 2=VAE、`LoraLoader` は 0=MODEL / 1=CLIP)
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

positive と negative の接続が入れ替わっているようなケースも、この検証で検出される。
ノードIDを変えた場合は
[src/agentic_imagegen/adapters/comfyui/workflow.py](../src/agentic_imagegen/adapters/comfyui/workflow.py)
の `TXT2IMG_BINDING` を合わせて更新する。
