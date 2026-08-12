# Workflowテンプレート

ComfyUIで実行するWorkflowを **API形式JSON** で置く場所。

## 重要な前提

- Workflowは人間がComfyUI GUIで作成する。LLMにノードや接続を組み立てさせない。
- コードから差し替えてよいのは、許可済みパラメータのみ。
  - `positive_prompt` / `negative_prompt` / `checkpoint` / `seed` / `steps` / `cfg`
  - `sampler` / `scheduler` / `width` / `height` / `batch_size` / `filename_prefix`
- 実行を許可するworkflowは allowlist で管理する (Phase 1は `txt2img` のみ)。
  定義は [src/agentic_imagegen/workflows/injector.py](../src/agentic_imagegen/workflows/injector.py) を参照。

## 同梱テンプレート

| ファイル | 用途 |
| --- | --- |
| `txt2img.json` | text-to-image (ComfyUI標準のtxt2imgグラフと同じノード構成) |

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

## GUIから書き出す手順

ComfyUI環境に合わせて作り直す場合の手順。

1. ComfyUIを起動し、txt2imgのWorkflowを組む (またはデフォルトグラフを読み込む)。
2. 設定画面で開発者向けオプション (Enable Dev mode Options) を有効にする。
3. `Save (API Format)` でJSONを書き出す。**通常の `Save` で保存したJSONは形式が異なり使用できない。**
4. 書き出したJSONを `workflows/txt2img.json` として置く。

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
