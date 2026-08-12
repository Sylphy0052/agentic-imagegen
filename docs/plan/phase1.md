# agentic-imagegen Phase 1 実装計画

- 対象: Phase 1 (Claude Code -> GenerationSpec -> Python CLI -> ComfyUI API -> 画像生成)
- 作成日: 2026-08-12
- ステータス: 実装準備完了 (Step 1 着手)
- 関連Issue: [#1 Roadmap](https://github.com/Sylphy0052/agentic-imagegen/issues/1) / [#2 推論高速化検証](https://github.com/Sylphy0052/agentic-imagegen/issues/2)

---

## 1. 目的とゴール

Claude Codeから自然言語で画像生成を指示し、ComfyUI GUIを一切操作せずにPNG出力まで到達できる最小構成を作る。

```text
自然言語要求
    -> GenerationSpec (YAML)
    -> specs/generated/*.yaml
    -> imagegen generate <spec>
    -> ComfyUI API
    -> Stable Diffusion 推論
    -> PNG保存
    -> 生成結果パスを返す
```

Phase 1の完了条件は「一気通貫で実行できること」であり、画像品質の最適化は対象外とする。

---

## 2. 環境調査結果 (Ground Truth)

2026-08-12時点で実機調査した結果を記録する。設計判断の前提となるため、変化した場合はこの節を更新する。

| 項目 | 実測値 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (WSL2, kernel 6.18.33.2-microsoft-standard-WSL2) |
| CPU | Intel Core Ultra 7 165H (22スレッド) |
| RAM | WSL割当 15GB (空き約8GB) |
| GPU | Intel Arc Graphics (Core Ultra 7 165H内蔵iGPU / Xe-LPG)。NVIDIA GPUなし |
| WSL GPU | `/dev/dxg` あり、`libd3d12.so` / `libdxcore.so` あり。Intel Level Zeroランタイムは未インストール |
| ディスク | WSL側 928GB空き / C: 211GB空き |
| Python | システム 3.10.12。uv管理の 3.12.13 / 3.13.14 が利用可能 |
| uv | 0.12.2 |
| ComfyUI | 未インストール (WSL側・Windows側いずれにも存在せず)。127.0.0.1:8188 および WSLホストIP:8188 へ到達不可 |
| リポジトリ | `LICENSE` のみ。実装コードなし |
| リモート | `github.com:Sylphy0052/agentic-imagegen` (gh CLI認証済み) |

重要な帰結:

- CUDAが使えないため、一般的な「ComfyUI + NVIDIA GPU」のセットアップ手順はそのまま適用できない。
- ComfyUIが未インストールのため、Integration TestとE2Eは環境構築後にしか実行できない。Unit Testは環境非依存で先行実装する。

---

## 3. ComfyUI実行環境の方針

決定内容は次のとおり。

### 採用: CPU推論 (Phase 1本編)

- WSL上にComfyUIをclone、`torch` はCPU版のみ導入する。追加ドライバ不要で確実に動作する。
- モデルはSD1.5系に固定する。SDXLはCPUでは1枚5-15分となりPhase 1の反復速度に見合わない。
- 想定所要時間 (SD1.5 / 512x512 / 20 steps): 約1-2.5分/枚。Integration Test設定 (512x512 / steps 2-4) では約10-20秒。

### 後続: Intel XPU (Level Zero) による高速化

- 別Issueで扱う。Intel aptリポジトリ追加、`libze-intel-gpu` 等の導入、XPU版 `torch` への差し替えが必要。
- 期待効果はCPU比で2-4倍程度。Xe-LPG 8コアかつRAM共有帯域律速のため劇的な改善は見込まない。
- WSL2でのiGPU passthroughはドライバ版依存で不安定なため、Phase 1のクリティカルパスからは外す。

### 代替案として保持: Windows側ComfyUI + WSLから接続

- Windows版ComfyUIはDirectML経由でArcを利用できる。
- ただし `--listen` とファイアウォール設定が必要で、Phase 1の「127.0.0.1限定」設計原則と衝突する。
- Intel XPUが不調だった場合の退避先として、上記の高速化Issue内に代替案として記載する。

Adapter層でComfyUI依存を隔離しているため、後からバックエンドの実行基盤を差し替えても Core / CLI 層は変更不要である。

---

## 4. スコープ

### Phase 1で実装する

- GenerationSpec (Pydanticモデル) とバリデーション
- Workflowテンプレート読み込みとパラメータ注入 (txt2imgのみ)
- ComfyUI Adapter (health check / submit / 実行監視 / 出力取得)
- CLI (`imagegen generate` / `validate` / `health`)
- 出力の保存とmetadata記録
- Unit Test / Integration Test (marker分離)
- CLAUDE.md / README / サンプルSpec / Workflow配置手順

### Phase 1で実装しない (Non-Goals)

MCP Server、Codex専用integration、img2img、ControlNet、IPAdapter、LoRA自動選択、LLMによるWorkflow生成、画像品質自動評価、複数ComfyUI Server、ジョブキュー管理、Web UI、認証、クラウド実行、動画生成。

ただし、いずれも後から追加しやすい層構造を維持する。

---

## 5. アーキテクチャ

```text
CLI (typer)                 <- 入出力とexit code
    |
Service (generation)        <- ユースケース組み立て
    |
Domain (models / errors)    <- GenerationSpec と検証規則。外部依存を持たない
    |
Workflows (injector)        <- テンプレート読込・注入・構造検証
    |
Adapters (comfyui)          <- HTTP / WebSocket / ComfyUI固有のJSON形状
```

境界の原則:

- Domain層はComfyUIのNode IDやHTTP仕様を一切知らない。
- Adapter層のみがComfyUI固有の知識を持つ。
- Node IDとclass_typeのマッピングは1ファイルに閉じ込め、コード全体に散在させない。
- CLIはMCP導入後も残す (ローカルデバッグ / CI / 障害切り分け用)。

### ディレクトリ構成

```text
agentic-imagegen/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── workflows/
│   └── txt2img.json
├── presets/
│   └── anime.yaml
├── specs/
│   ├── examples/
│   │   └── txt2img.yaml
│   └── generated/
├── outputs/
├── docs/
│   ├── plan/phase1.md
│   └── comfyui-setup.md
├── src/agentic_imagegen/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── errors.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── generation.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   └── injector.py
│   └── adapters/comfyui/
│       ├── __init__.py
│       ├── client.py
│       └── workflow.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   └── txt2img.json
    ├── unit/
    │   ├── test_models.py
    │   ├── test_workflow_injector.py
    │   ├── test_config.py
    │   └── test_cli.py
    └── integration/
        └── test_comfyui.py
```

---

## 6. 主要インターフェース設計

### 6.1 GenerationSpec

Pydantic v2で定義する。Phase 1はtxt2imgのみ。

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
  seed: -1
  batch_size: 1
  sampler: euler
  scheduler: normal

model:
  checkpoint: v1-5-pruned-emaonly.safetensors

output:
  directory: outputs
  prefix: blue_hair
```

モデル構成:

- `GenerationSpec` : version / task / prompt / generation / model / output
- `PromptSpec` : positive (必須) / negative (既定は空文字)
- `GenerationParams` : width / height / steps / cfg / seed / batch_size / sampler / scheduler
- `ModelSpec` : checkpoint
- `OutputSpec` : directory / prefix

すべて `extra="forbid"` とし、未知キーは実行前に弾く。

### 6.2 バリデーション仕様

2段構えとする。

1. モデル定義上のハード制約 (Pydantic `Field`)
   - width / height: 64以上8192以下、8の倍数
   - steps: 1以上100以下
   - cfg: 0以上30以下
   - batch_size: 1以上4以下
   - seed: -1 (ランダム) または 0以上 2^63-1 以下
   - sampler / scheduler: 既知値のリテラル集合
2. 設定由来のポリシー制約 (`config.Settings` から注入)
   - `IMAGEGEN_MAX_WIDTH` (既定 2048)
   - `IMAGEGEN_MAX_HEIGHT` (既定 2048)
   - `IMAGEGEN_MAX_PIXELS` (既定 4194304)
   - `IMAGEGEN_MAX_BATCH` (既定 4)
   - 違反時は `InvalidGenerationSpec` を送出する

セキュリティ関連の検証:

- checkpoint: `..` セグメント禁止、絶対パス禁止、バックスラッシュ禁止、拡張子は `.safetensors` / `.ckpt` のみ許可。ComfyUIのサブフォルダ指定を許すため `/` 区切りは1階層まで許可する。
- checkpoint存在確認: ComfyUI到達可能時は `GET /object_info/CheckpointLoaderSimple` から利用可能一覧を取得して照合する。到達不能時はスキップし警告ログを出す (Phase 1ではallowlist未使用)。
- output.directory: 解決後のパスがリポジトリルート配下であることを検証する。外部への書き出しは禁止。
- 実行可能なworkflowは `txt2img` のみ。ユーザー入力から任意のworkflow JSONを実行させない。

### 6.3 Workflow Injection

```python
@dataclass(frozen=True)
class NodeRef:
    node_id: str
    class_type: str

TXT2IMG_NODES: dict[str, NodeRef] = {
    "checkpoint":      NodeRef("4", "CheckpointLoaderSimple"),
    "positive_prompt": NodeRef("6", "CLIPTextEncode"),
    "negative_prompt": NodeRef("7", "CLIPTextEncode"),
    "latent":          NodeRef("5", "EmptyLatentImage"),
    "ksampler":        NodeRef("3", "KSampler"),
    "save_image":      NodeRef("9", "SaveImage"),
}
```

- Node IDはComfyUI標準txt2imgテンプレート (API形式) の既定IDに合わせる。
- 注入前に、各NodeRefについて「対象node_idが存在するか」「class_typeが一致するか」「必要な入力キーが存在するか」を検証する。1つでも不一致なら `WorkflowValidationError` でfail-fastし、誤ったノードへの注入を防ぐ。
- 注入対象は positive_prompt / negative_prompt / checkpoint / seed / steps / cfg / sampler / scheduler / width / height / batch_size / filename_prefix に限定する。
- 注入は元テンプレートを破壊しない (deep copyしてから書き換える)。
- seedが -1 の場合は実行時に乱数へ解決し、解決後の値をmetadataへ記録する。

### 6.4 ComfyUI Client

- ベースURL: `COMFYUI_BASE_URL` (既定 `http://127.0.0.1:8188`)
- 責務: health check / workflow submission / 実行監視 / ステータス取得 / 出力取得 / エラー変換

実行フロー:

```text
health check (GET /system_stats)
    -> POST /prompt {"prompt": <workflow>, "client_id": <uuid>}
    -> prompt_id 取得
    -> WebSocket /ws?clientId=<uuid> で実行監視
    -> 完了検知
    -> GET /history/<prompt_id> で出力確認
    -> GET /view?filename=&subfolder=&type=output で画像取得
```

- 監視方式はClient内部に隠蔽する。WebSocketが利用できない場合は `/history/<prompt_id>` のポーリングへ自動フォールバックする。呼び出し側は方式を意識しない。
- タイムアウトは `IMAGEGEN_TIMEOUT` (既定300秒) で設定可能。Integration Testでは短縮する。
- ComfyUI固有のエラーレスポンスは、後述の例外型へ変換してから上位へ返す。

### 6.5 出力とmetadata

```text
outputs/
└── 2026-08-12/
    └── blue_hair/
        ├── image_0001.png
        └── metadata.json
```

ComfyUIのoutputディレクトリだけに依存せず、プロジェクト側にコピーして追跡可能にする。

```json
{
  "prompt_id": "...",
  "workflow": "txt2img",
  "created_at": "2026-08-12T12:00:00+09:00",
  "resolved_seed": 123456789,
  "spec": {},
  "outputs": ["image_0001.png"]
}
```

Phase 1では過剰実装を避け、上記項目のみ記録する。

### 6.6 エラー型とexit code

```text
ImageGenError (基底)
├── InvalidGenerationSpec
├── ComfyUIUnavailable
├── WorkflowValidationError
├── WorkflowSubmissionError
├── GenerationTimeout
├── GenerationFailed
└── OutputNotFound
```

| exit code | 条件 |
| --- | --- |
| 0 | 成功 |
| 1 | 想定外の内部エラー |
| 2 | InvalidGenerationSpec |
| 3 | ComfyUIUnavailable |
| 4 | WorkflowValidationError |
| 5 | WorkflowSubmissionError |
| 6 | GenerationTimeout |
| 7 | GenerationFailed |
| 8 | OutputNotFound |

CLIには原因が特定できる短いメッセージを表示し、内部トレースバックをそのまま大量表示しない (詳細は `--verbose` 時のみ)。

### 6.7 CLI

```bash
uv run imagegen health
uv run imagegen validate specs/examples/txt2img.yaml
uv run imagegen generate specs/examples/txt2img.yaml
```

`health` の期待出力:

```text
ComfyUI: reachable
URL: http://127.0.0.1:8188
```

### 6.8 Configuration

| 環境変数 | 既定値 | 用途 |
| --- | --- | --- |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI接続先 |
| `IMAGEGEN_MAX_WIDTH` | 2048 | 幅上限 |
| `IMAGEGEN_MAX_HEIGHT` | 2048 | 高さ上限 |
| `IMAGEGEN_MAX_PIXELS` | 4194304 | 総ピクセル数上限 |
| `IMAGEGEN_MAX_BATCH` | 4 | batch_size上限 |
| `IMAGEGEN_TIMEOUT` | 300 | 生成タイムアウト秒 |
| `IMAGEGEN_OUTPUT_ROOT` | `outputs` | 出力ルート |

秘密情報は扱わないため環境変数ファイルは必須としない。設定は `config.Settings` に集約し、ハードコードを避ける。

---

## 7. 技術スタック

- Python >= 3.12 (uv管理の3.12.13を使用)
- 依存: `pydantic` / `PyYAML` / `httpx` / `websockets` / `typer`
- 開発依存: `pytest` / `pytest-asyncio` / `pytest-cov` / `ruff` / `mypy`
- 依存は必要最小限にとどめる。Phase 1でDIコンテナ等のフレームワークは導入しない。
- 型ヒントは原則必須。`Any` の濫用を避ける。

---

## 8. 実装順序と受入基準

| Step | 内容 | 受入基準 |
| --- | --- | --- |
| 1 | プロジェクト初期化 (pyproject / src layout / ruff / mypy / pytest / .gitignore / ディレクトリscaffold) | `uv sync` 成功、`uv run ruff check .` / `uv run mypy src` / `uv run pytest` が空実行で通過 |
| 2 | GenerationSpec + バリデーション | 正常系・異常系のUnit Testが通過 (width/height/steps/cfg/batch_size/checkpoint traversal) |
| 3 | Workflow loader / injector | 注入テストと構造不一致時のfail-fastテストが通過 |
| 4 | ComfyUI Client (health checkまで) | `imagegen health` が到達可否を正しく判定。未起動時に `ComfyUIUnavailable` |
| 5 | submission + 実行監視 + 出力取得 | モックサーバに対する submit -> 完了検知 -> 画像取得 が通過。タイムアウト動作を検証 |
| 6 | CLI (validate / health / generate) | 3コマンドが動作し、失敗時に規定のexit codeを返す |
| 7 | ComfyUI環境構築 + Integration Test | ComfyUI起動、SD1.5配置、`uv run pytest -m integration` が通過 |
| 8 | CLAUDE.md | 画像生成要求時の手順と禁止事項を記載 |
| 9 | README + docs/comfyui-setup.md | セットアップからClaude Codeでの利用までを記載 |
| 10 | Claude CodeからのE2E実行 | 自然言語指示からPNG出力まで一気通貫で成功 |

Step 1-6と8-9はComfyUI不要で先行実装できる。Step 7で環境構築を行い、Step 10で通しの確認をする。

---

## 9. テスト戦略

TDD寄りで進め、実装と並行してテストを書く。

### Unit Test (GPU / ComfyUI不要)

- `test_valid_generation_spec`
- `test_invalid_width` / `test_invalid_height`
- `test_invalid_batch_size` / `test_invalid_steps` / `test_invalid_cfg`
- `test_checkpoint_path_traversal_rejected`
- `test_output_directory_escape_rejected`
- `test_positive_prompt_injection` / `test_negative_prompt_injection`
- `test_seed_injection` / `test_resolution_injection` / `test_batch_size_injection` / `test_checkpoint_injection`
- `test_unknown_workflow_structure_fails`
- `test_cli_exit_codes`

Unit Testから実ComfyUIへは通信しない。ComfyUI応答は `httpx.MockTransport` とフィクスチャJSONで代替する。

### Integration Test (`-m integration`)

ComfyUI起動時のみ実行する。通常の `uv run pytest` ではskipされる。

- GPU負荷を最小化する設定: width 512 / height 512 / steps 2-4 / batch_size 1
- 確認項目: 接続可能 / submit成功 / prompt_id取得 / 正常終了 / 画像output存在

```bash
uv run pytest            # Unit のみ
uv run pytest -m integration
```

---

## 10. 品質基準

- `uv run pytest` 通過
- `uv run ruff check .` 警告ゼロ
- `uv run mypy src` 通過
- `uv run pytest --cov` でカバレッジ80%以上
- 巨大な単一モジュールを作らない。DRYを守りつつ過剰な抽象化はしない。

優先順位: 1. 正しく動く 2. テスト可能 3. 障害を切り分けやすい 4. 型安全 5. 将来MCP化しやすい 6. コード量を増やしすぎない。

---

## 11. リスクと対応

| リスク | 影響 | 対応 |
| --- | --- | --- |
| CPU推論が遅くE2Eの反復が困難 | Step 10の確認コスト増 | SD1.5固定、E2Eは低stepsで確認。品質評価はPhase 1対象外 |
| Workflow JSONのNode IDが環境依存 | 注入先の不一致 | class_typeとinputsキーを検証しfail-fast。手順書でAPI形式の書き出しを明示 |
| WebSocket監視が不安定 | 完了検知に失敗 | historyポーリングへ自動フォールバック。タイムアウトで確実に打ち切る |
| Intel XPU導入がWSLで不安定 | 高速化が頓挫 | Phase 1のクリティカルパスから外し別Issue化。Windows側ComfyUIを代替案として保持 |
| RAM 15GBでのCPU推論 | OOMの可能性 | SD1.5 + 512x512 + batch_size 1を既定とし、上限をポリシーで強制 |

---

## 12. 将来拡張 (参考)

- Phase 2: Claude Code Skill / Preset system / LoRA / Character preset / Scene preset / Metadata強化 / Reference image / img2img
- Phase 3: MCP Server (`generate_image` / `validate_generation` / `list_models` / `list_loras` / `list_workflows` / `get_generation_status`)
- Phase 4: ControlNet / IPAdapter / Character consistency / Batch generation / Upscaling / Backend追加

Phase 1時点では、MCPのためだけの抽象層は作らない。
