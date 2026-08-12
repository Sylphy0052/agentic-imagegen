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

Phase 1 (txt2imgのみ) を実装中。進捗は [Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1)、
設計は [docs/plan/phase1.md](docs/plan/phase1.md) を参照。

## 必要環境

- Python 3.12以上
- [uv](https://docs.astral.sh/uv/)
- ComfyUI (別途セットアップ。Phase 1はWSL上のCPU推論を前提とする)

## セットアップ

```bash
git clone git@github.com:Sylphy0052/agentic-imagegen.git
cd agentic-imagegen
uv sync
```

## ComfyUIの起動

詳細な手順は [docs/comfyui-setup.md](docs/comfyui-setup.md) を参照。要点のみ:

```bash
cd ~/ComfyUI
uv run python main.py --cpu --listen 127.0.0.1 --port 8188
```

`0.0.0.0` での公開は想定していない。Phase 1はループバック限定で使う。

## Workflowの準備

`workflows/txt2img.json` を同梱済み。ComfyUI標準のtxt2imgグラフと同じノード構成のため、
通常は差し替え不要。自環境に合わせて作り直す場合は
[workflows/README.md](workflows/README.md) の手順 (API形式での書き出し) に従う。

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
Version: 0.3.40
Devices: cpu
```

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

## GenerationSpec

```yaml
version: "1"
task: txt2img

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

output:
  directory: outputs  # 省略時は IMAGEGEN_OUTPUT_ROOT
  prefix: blue_hair
```

検証内容:

- 解像度は64-8192かつ8の倍数、steps 1-100、cfg 0-30、batch_size 1-4
- 環境変数による上限 (`IMAGEGEN_MAX_*`) を超える指定は拒否する
- checkpoint名はPath Traversal・絶対パス・想定外の拡張子を拒否する
- 出力先が作業ルートの外を指す場合は拒否する

## 出力

```text
outputs/
└── 2026-08-12/
    └── blue_hair/
        ├── image_0001.png
        └── metadata.json
```

`metadata.json` には `prompt_id` / `workflow` / `created_at` / `resolved_seed` / `spec` / `outputs` を記録する。
同じ日に同じprefixで再実行した場合は連番ディレクトリを作り、既存の結果を上書きしない。

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

秘密情報は扱わないため、環境変数ファイルは必須ではない。

## テスト

```bash
uv run pytest                 # Unit Test のみ (ComfyUI不要)
uv run pytest --cov           # カバレッジ付き
uv run pytest -m integration  # ComfyUI起動時のみ
```

Unit Testは実ComfyUIへ接続しない。Integration TestはComfyUI未起動時にskipされる。

品質ゲート:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Claude Codeからの利用

このリポジトリでClaude Codeに画像生成を依頼すると、[CLAUDE.md](CLAUDE.md) の手順に従って
Spec作成から生成まで自動で実行される。

```text
512x768で青髪・青い瞳のアニメキャラクターを1枚生成して
```

実行される流れ:

1. GenerationSpecを作成し `specs/generated/` へ保存
2. `uv run imagegen validate` で検証
3. `uv run imagegen generate` で生成
4. 出力パスとseedを返す

Claude Codeが守るルール (Workflowを書き換えない、validationを迂回しないなど) は
[CLAUDE.md](CLAUDE.md) に定義している。

## アーキテクチャ

```text
CLI (typer)                 入出力とexit codeへの変換
    |
Service (generation)        ユースケースの組み立て
    |
Domain (models / policy)    GenerationSpecと検証規則。外部依存なし
    |
Workflows (injector)        テンプレートの読み込みとallowlist
    |
Adapters (comfyui)          HTTP / WebSocket / ComfyUI固有のJSON形状
```

ComfyUI依存は `adapters/comfyui/` に閉じ込めてあり、
バックエンドを追加する際もDomain / Service層の変更を必要としない。

## ライセンス

[LICENSE](LICENSE) を参照。
