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

どちらを使うかは `model.loras` の有無で自動的に決まる。定義は
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
4. **実機のComfyUIへ投入し、生成が成功することを確認する**

`txt2img_lora.json` はこの手順で作成し、実機 (ComfyUI 0.32.0 / Intel XPU) で
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
