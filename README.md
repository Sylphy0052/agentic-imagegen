# agentic-imagegen

AIコーディングエージェント (Claude Code など) から、ComfyUIを介してStable Diffusion系モデルによる画像生成を実行するための基盤。

```text
Claude Code -> GenerationSpec -> Python CLI (imagegen) -> ComfyUI API -> 画像生成
```

エージェントにComfyUIのWorkflowを組み立てさせるのではなく、
**人間が作ったWorkflowテンプレートの、許可されたパラメータだけを差し替える**設計を採る。
エージェントとの境界は `GenerationSpec` (YAML) に固定し、
将来のMCP Server化やバックエンド追加に備えて層を分離している。

## ステータス

Phase 1 (txt2img) は完了。現在はPhase 2 (preset / LoRA / img2img) を実装中。

| フェーズ | 内容 | 状態 |
| --- | --- | --- |
| Phase 1 | txt2img、CLI、ComfyUI連携 | 完了 ([#1](https://github.com/Sylphy0052/agentic-imagegen/issues/1)) |
| Phase 2 | Preset / LoRA / img2img / Claude Code Skill | 進行中 ([#3](https://github.com/Sylphy0052/agentic-imagegen/issues/3)) |
| Phase 3 | MCP Server (Claude Code / Codex 双方対応) | 未着手 ([#4](https://github.com/Sylphy0052/agentic-imagegen/issues/4)) |
| Phase 4 | ControlNet / IPAdapter / Batch / Upscaling | 未着手 ([#5](https://github.com/Sylphy0052/agentic-imagegen/issues/5)) |

設計は [docs/plan/phase1.md](docs/plan/phase1.md) を参照。

## 必要環境

- Python 3.12以上
- [uv](https://docs.astral.sh/uv/)
- ComfyUI (別途セットアップ)

GPUはIntel Arc内蔵GPU (XPU) での動作を確認している。NVIDIA GPUがなくてもCPU推論で動くが、
XPUのほうがおよそ5倍速い。

| 実行基盤 | SD1.5 / 512x768 / 20 steps |
| --- | --- |
| Intel XPU | 約135秒 |
| CPU | 約12分 |

## セットアップ

```bash
git clone git@github.com:Sylphy0052/agentic-imagegen.git
cd agentic-imagegen
uv sync
```

## ComfyUIの起動

- Intel GPU (XPU) を使う場合: [docs/xpu-setup.md](docs/xpu-setup.md)
- CPU推論で動かす場合: [docs/comfyui-setup.md](docs/comfyui-setup.md)

要点のみ:

```bash
cd ~/ComfyUI
./.venv/bin/python main.py --listen 127.0.0.1 --port 8188   # XPU (自動検出)
./.venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8188  # CPU推論を強制
```

`0.0.0.0` での公開は想定していない。ループバック限定で使う。

## Workflowの準備

同梱テンプレート:

| ファイル | 用途 |
| --- | --- |
| `workflows/txt2img.json` | text-to-image |
| `workflows/txt2img_lora.json` | LoRA付き text-to-image (`LoraLoader` 3段) |
| `workflows/img2img.json` | image-to-image (`LoadImage` + `VAEEncode`) |

どれを使うかは `task` と `model.loras` の有無で自動的に決まる。通常は差し替え不要。
自環境に合わせて作り直す場合は [workflows/README.md](workflows/README.md) の手順
(API形式での書き出し) に従う。

読み込み時にノードID・class_type・必要な入力キー・ノード間の接続を検証し、
1つでも想定と違えば注入せずに失敗する。

## 使い方

### 到達確認

```bash
uv run imagegen health
```

```text
ComfyUI: reachable
URL: http://127.0.0.1:8188
Version: 0.32.0
Devices: xpu:0 Intel(R) Graphics [0x7d55]
```

`Devices:` が `xpu:0` ならIntel GPU、`cpu` ならCPU推論で動いている。

### Specの検証

ComfyUIへは接続せず、Specだけを検証する。

```bash
uv run imagegen validate specs/examples/txt2img.yaml
```

```text
OK
Spec: specs/examples/txt2img.yaml
Workflow: txt2img
Resolution: 512x768 (batch 1)
Checkpoint: v1-5-pruned-emaonly.safetensors
```

presetやLoRAを指定している場合は、適用内容と実際に使われるテンプレート名も表示される。

```text
Workflow: txt2img_lora
Checkpoint: meinamix_v12Final.safetensors
LoRA: add_detail.safetensors (model=0.8, clip=0.8)
Presets: character=anime-girl-blue, style=anime-soft
```

### 生成

```bash
uv run imagegen generate specs/examples/txt2img.yaml
```

```text
prompt_id: 5f2c...
seed: 883021
directory: /path/to/outputs/2026-08-12/blue_hair
/path/to/outputs/2026-08-12/blue_hair/image_0001.png
metadata: /path/to/outputs/2026-08-12/blue_hair/metadata.json
```

タイムアウトを個別指定する場合:

```bash
uv run imagegen generate specs/examples/txt2img.yaml --timeout 600
```

失敗時は原因ごとに異なるexit codeを返す (一覧は [CLAUDE.md](CLAUDE.md) を参照)。

### 一括生成

複数のSpecをまとめて実行する。seed掃引もできる。

```bash
uv run imagegen batch specs/generated/a.yaml specs/generated/b.yaml
uv run imagegen batch specs/generated/a.yaml --seeds 111,222,333
```

```text
[1/2] specs/generated/a.yaml (seed=111)
  -> /path/to/outputs/2026-08-12/sample/image_0001.png
[2/2] specs/generated/a.yaml (seed=222)
  -> /path/to/outputs/2026-08-12/sample-2/image_0001.png
成功 2 / 失敗 0
```

**1件失敗しても残りは続ける。** 途中で止まるとどこまで進んだのか分からなくなるため。
全件終わったあとにサマリを出し、失敗があれば最初の失敗のexit codeで終了する。

Specの検証は実行前に全件まとめて行う。不正なSpecが混ざっていた場合は1件も生成しない。

## GenerationSpec

```yaml
version: "1"
task: txt2img

presets:              # 省略可。character / scene / style を名前で参照する
  character: anime-girl-blue
  style: anime-soft

prompt:
  positive: 1girl, blue hair, blue eyes, anime illustration, full body
  negative: low quality, blurry, bad anatomy

generation:
  width: 512
  height: 768
  steps: 20
  cfg: 5.5
  seed: -1          # -1 は実行時にランダムへ解決し、実際の値をmetadataへ記録する
  batch_size: 1
  sampler: euler
  scheduler: normal

model:
  checkpoint: v1-5-pruned-emaonly.safetensors
  loras:            # 省略可。同時3件まで
    - name: add_detail.safetensors
      strength_model: 0.8
      strength_clip: 0.8

output:
  directory: outputs  # 省略時は IMAGEGEN_OUTPUT_ROOT
  prefix: blue_hair
```

検証内容:

- 解像度は64-8192かつ8の倍数、steps 1-100、cfg 0-30、batch_size 1-4
- 環境変数による上限 (`IMAGEGEN_MAX_*`) を超える指定は拒否する
- checkpoint名・LoRA名はPath Traversal・絶対パス・想定外の拡張子を拒否する
- LoRAは同時3件まで、同じLoRAの重複指定は拒否、strengthは±10.0まで
- preset名は英数字始まりの `[A-Za-z0-9._-]` のみ
- 出力先が作業ルートの外を指す場合は拒否する

## Preset

繰り返し使う指定を3つの軸にまとめ、Specから名前で参照する。

| 軸 | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定 |

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

- **prompt** は `character` -> `scene` -> `style` -> Spec本体 の順にカンマ連結し、
  重複トークンは最初の1つを残して除去する (大文字小文字と連続空白は無視)。negativeも同じ
- **generation** はpresetの指定を取り込んだうえで、Spec本体の指定を優先する
  (spec > style > scene > character)
- 適用したpreset名は解決後のSpecに残り、`metadata.json` にも記録される

## ControlNet (構図を指定する)

参考画像から線画 (Canny) を取り、その構図を保ったまま生成する。

```yaml
control:
  image: inputs/pose.png
  model: control_v11p_sd15_canny_fp16.safetensors
  strength: 0.9
  low_threshold: 0.3
  high_threshold: 0.7
```

```text
$ uv run imagegen validate specs/generated/controlnet-check.yaml
Workflow: txt2img_controlnet
ControlNet: inputs/pose.png (model=control_v11p_sd15_canny_fp16.safetensors, strength=0.9)
```

- txt2img / img2img のどちらでも使える。LoRAとも併用できる
- control画像は生成前にComfyUIへ自動でアップロードされる
- ControlNetモデルは `~/ComfyUI/models/controlnet/` へ置く
- **前処理は Canny のみ。** pose / depth はpreprocessorのカスタムノードが要るため未対応
- `upscale` との同時指定は未対応

線が強く出すぎる場合は `low_threshold` を上げるか `strength` を下げる。

## hires fix (解像度を上げる)

1段目の結果をlatentのまま拡大し、2段目のKSamplerで描き足す。アップスケールモデルは不要。

```yaml
generation:
  width: 512
  height: 512
  steps: 8
  upscale:
    scale: 1.5        # 1.0より大きく4.0以下
    denoise: 0.45     # 低いほど元の絵を保つ
    steps: 6          # 省略時は1段目と同じ
    method: nearest-exact
```

指定するとテンプレートが `*_hires` へ自動的に切り替わる。
2段目のseedは1段目と同じ値を使う (変えると元の絵から離れるため)。

実測 (Intel XPU / SD1.5 / 512x512 -> 768x768): 43.7秒。
**生成時間は倍以上になる。** 2段目は拡大後の解像度で走る。

## img2img

既存の画像を入力にして描き直す。

```yaml
version: "1"
task: img2img

source:
  image: inputs/reference.png   # リポジトリ配下に置く (git管理外)
  denoise: 0.55                 # 0に近いほど入力を保ち、1に近いほど描き直す

prompt:
  positive: 1girl, blue hair, rooftop, night, city lights

model:
  checkpoint: meinamix_v12Final.safetensors
```

- 入力画像は生成前にComfyUIへ自動でアップロードされる (`~/ComfyUI/input/` へ手で置く必要はない)
- **解像度は入力画像のサイズをそのまま使う。** `width` / `height` を書くと拒否される
- `batch_size` は1のみ。LoRAは併用できる (`img2img_lora` テンプレートへ切り替わる)
- 拡張子は `.png` / `.jpg` / `.jpeg` / `.webp`、上限は `IMAGEGEN_MAX_SOURCE_BYTES` (既定32MiB)

## 出力

```text
outputs/
└── 2026-08-12/
    └── blue_hair/
        ├── image_0001.png
        └── metadata.json
```

`metadata.json` の内容:

| キー | 内容 |
| --- | --- |
| `prompt_id` | ComfyUI側の実行ID |
| `workflow` | 実際に使ったテンプレート名 |
| `workflow_hash` | テンプレートのダイジェスト (`sha256:...`) |
| `created_at` | 生成時刻 (タイムゾーン付き) |
| `resolved_seed` | 実際に使われたseed |
| `backend` | 実行基盤 (`comfyui_version` / `devices`) |
| `spec` | preset展開後のSpec全体 |
| `outputs` | 出力ファイル名 |

同じ日に同じprefixで再実行した場合は連番ディレクトリを作り、既存の結果を上書きしない。

同じSpecなのに結果が変わった場合は `workflow_hash` と `backend` を前回と比べると、
テンプレートが変わったのか実行基盤が変わったのかを切り分けられる。

## 設定

| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI接続先 |
| `IMAGEGEN_MAX_WIDTH` | 2048 | 幅の上限 |
| `IMAGEGEN_MAX_HEIGHT` | 2048 | 高さの上限 |
| `IMAGEGEN_MAX_PIXELS` | 4194304 | 総pixel数の上限 (batch込み) |
| `IMAGEGEN_MAX_BATCH` | 4 | batch_sizeの上限 |
| `IMAGEGEN_TIMEOUT` | 300 | 生成のタイムアウト秒 |
| `IMAGEGEN_OUTPUT_ROOT` | `outputs` | 出力ルート |
| `IMAGEGEN_PRESETS_ROOT` | `presets` | presetの探索ルート |
| `IMAGEGEN_MAX_SOURCE_BYTES` | 33554432 | img2imgの入力画像の上限バイト数 |

秘密情報は扱わないため、環境変数ファイルは必須ではない。

## テスト

```bash
uv run pytest                 # Unit Test のみ (ComfyUI不要)
uv run pytest --cov           # カバレッジ付き
uv run pytest -m integration  # ComfyUI起動時のみ
```

Unit Testは実ComfyUIへ接続しない。Integration TestはComfyUI未起動時にskipされる。
特定のcheckpointで実行する場合は `IMAGEGEN_TEST_CHECKPOINT` を指定する。

品質ゲート:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Claude Codeからの利用

このリポジトリでClaude Codeに画像生成を依頼すると、
[.claude/skills/imagegen/SKILL.md](.claude/skills/imagegen/SKILL.md) の手順に従って
Spec作成から生成まで自動で実行される。

```text
512x768で青髪・青い瞳のアニメキャラクターを1枚生成して
```

実行される流れ:

1. `imagegen health` で実行基盤を確認
2. 使えるcheckpoint / LoRA / presetを確認
3. GenerationSpecを作成し `specs/generated/` へ保存
4. `uv run imagegen validate` で検証
5. `uv run imagegen generate` で生成
6. 出力パスとseedを返す

「同じキャラで別の構図」のような依頼では、character presetを再利用して
scene だけを差し替える。失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](.claude/skills/imagegen/references/troubleshooting.md)
にexit codeごとの手順がある。

Claude Codeが守るルール (Workflowを勝手に書き換えない、validationを迂回しないなど) は
[CLAUDE.md](CLAUDE.md) に定義している。

## MCP Server

Claude Code / Codex の双方から同じ基盤を使える。手順は
[docs/mcp-setup.md](docs/mcp-setup.md) を参照。

```bash
uv run imagegen-mcp   # stdioで待ち受ける (通常はクライアントが子プロセスとして起動する)
```

| tool | 用途 |
| --- | --- |
| `validate_generation` | GenerationSpecを検証する (生成はしない) |
| `generate_image` | 生成を開始し `job_id` を返す (完了は待たない) |
| `get_generation_status` | 生成の状態と結果 (出力パス / seed / exit_code) を返す |
| `generate_batch` | 複数のSpecをまとめて生成し `job_id` を返す (seed掃引にも対応) |
| `get_batch_status` | 一括生成の状態と1件ごとの結果を返す |
| `list_models` | 利用可能なcheckpoint名 |
| `list_loras` | 利用可能なLoRA名 |
| `list_controlnets` | 利用可能なControlNetモデル名 |
| `list_workflows` | 実行を許可しているWorkflowテンプレート名 |

生成は数十秒から数分かかるため、`generate_image` は完了を待たずに `job_id` を返し、
`get_generation_status` で結果を受け取る。失敗時はCLIと同じ exit code を返す。

ControlNet・hires fix・LoRA・img2imgはいずれもSpecの内容で決まるため、MCP側に専用の
パラメータは無い。CLIで書けるSpecはそのままMCPでも使える。

MCP層は薄いアダプタで、検証も生成もCLIと同じ Service / Domain を通る。
MCP経由で検証を迂回できる経路は作っていない。

## アーキテクチャ

```text
CLI (typer)                 入出力とexit codeへの変換
    |
Service (generation)        ユースケースの組み立て、presetの解決
    |
Domain (models / policy)    GenerationSpecと検証規則。外部依存なし
    |
Workflows (injector)        テンプレートの読み込みとallowlist
    |
Adapters (comfyui)          HTTP / WebSocket / ComfyUI固有のJSON形状
```

ComfyUI依存は `adapters/comfyui/` に閉じ込めてあり、
バックエンドを追加する際もDomain / Service層の変更を必要としない。
実際、CPU推論からIntel XPUへ切り替えた際も `src/` の変更は不要だった。

## ライセンス

[LICENSE](LICENSE) を参照。
