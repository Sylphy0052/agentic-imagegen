# agentic-imagegen

AIコーディングエージェント (Claude Codeなど) から、ComfyUIを介してStable Diffusion系モデルによる画像生成を実行するための基盤。

```text
Claude Code -> GenerationSpec -> Python CLI (imagegen) -> ComfyUI API -> 画像生成
```

エージェントにComfyUIのWorkflowを組み立てさせるのではなく、
**人間が作ったWorkflowテンプレートの、許可されたパラメータだけを差し替える**設計を採る。
エージェントとの境界は `GenerationSpec` (YAML) に固定し、
将来のMCP Server化やバックエンド追加に備えて層を分離している。

## ステータス

txt2imgから始め、preset / LoRA / img2img、MCP Server、ControlNet / IPAdapter / hires fix /
batch、日本語テキスト合成、DiT系モデル (Anima) 対応まで一通り実装済み。

| 機能 | 内容 |
| --- | --- |
| txt2img / img2img | CLI、ComfyUI連携、preset (character / scene / style) |
| LoRA | 複数のLoRAを重ねて適用 |
| MCP Server | Claude Code / Codex双方対応 |
| ControlNet / IPAdapter | 構図・特徴の引き継ぎ、Character consistency |
| hires fix / batch | latentアップスケール、複数枚一括生成 |
| 日本語テキスト合成 | 生成後にPillowで合成 (`text` / `compose`) |
| DiT系モデル (Anima) | UNet / CLIP / VAEを個別指定するローダに対応 |

ここまでの経緯と実機確認の結果は
[Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1) (Roadmap、完了済み) を
参照。未着手の拡張はopenなIssueで個別に管理している。
最初の設計は [docs/plan/phase1.md](docs/plan/phase1.md) を参照。

## ドキュメントの構成

| 文書 | 何が書いてあるか |
| --- | --- |
| README.md (この文書) | プロジェクトの紹介、セットアップ、CLIの使い方 |
| [docs/spec-reference.md](docs/spec-reference.md) | GenerationSpecの全フィールド仕様。値域・既定値・組み合わせ規則・metadata.json |
| [CLAUDE.md](CLAUDE.md) | Claude Codeがこのリポジトリを操作するときのルール。環境変数・exit code |
| [.claude/skills/imagegen/SKILL.md](.claude/skills/imagegen/SKILL.md) | 画像生成要求を受けてから結果を返すまでの手順 |
| [docs/diffusers-backend.md](docs/diffusers-backend.md) | ComfyUIを使わずdiffusersで直接生成する場合の設定と対応範囲 |
| [docs/](docs/) | 環境構築 ([cuda](docs/cuda-setup.md) / [xpu](docs/xpu-setup.md) / [cpu](docs/comfyui-setup.md) / [mcp](docs/mcp-setup.md) / [fonts](docs/fonts-setup.md)) とモデル別の [プロンプト指針](docs/prompting-guide.md) |
| [workflows/README.md](workflows/README.md) | Workflowテンプレートの一覧と作り方 |

## 必要環境

- Python 3.12以上
- [uv](https://docs.astral.sh/uv/)
- ComfyUI (別途セットアップ)

NVIDIA GPU (CUDA) / Intel Arc内蔵GPU (XPU) / CPUの3構成で動作を確認している。
SD1.5 / 512x768 / 20 stepsの実測はCUDAで約4秒、XPUで約135秒、CPUで約12分。
条件別の実測値は
[docs/xpu-setup.mdの「所要時間とタイムアウトの目安」](docs/xpu-setup.md#所要時間とタイムアウトの目安)。

## セットアップ

```bash
git clone git@github.com:Sylphy0052/agentic-imagegen.git
cd agentic-imagegen
uv sync
```

## ComfyUIの起動

- NVIDIA GPU (CUDA) を使う場合: [docs/cuda-setup.md](docs/cuda-setup.md)
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

同梱テンプレートは `workflows/` にあり、通常は差し替え不要。実際に使うテンプレートは
Specの内容から自動的に決まる (決まり方は
[Workflowテンプレートの決まり方](docs/spec-reference.md#workflowテンプレートの決まり方))。

テンプレート一覧と各構成のノード内訳は [workflows/README.md](workflows/README.md) を参照。
自環境に合わせて作り直す場合も同ファイルの手順 (API形式での書き出し) に従う。

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

### 使えるモデルとpresetの一覧

実行基盤と、checkpoint / LoRA / ControlNet / IPAdapter / VAE / アップスケールモデル /
embedding、リポジトリ内のpreset・フォントをまとめて出す。

```bash
uv run imagegen catalog
```

```text
Backend: unavailable (filesystem fallback)
ComfyUI home: /home/user/ComfyUI
Presets: character 3 / scene 5 / style 16

presets/style (16)
  anima-base
  ...

checkpoints (14)
  hassakuSD15_v13.safetensors  <- 既定 (sd15-hassaku)
  ...
```

`Devices:` と `Version:` はComfyUIが起動しているときだけ出る。
`Backend:` が取得元。ComfyUIが起動していれば `api` (実際に読み込める名前) を、
起動していなければ `COMFYUI_HOME` 配下の `models/` を直接見た結果を出す。
探索のためだけにComfyUIを起動しなくてよい。ただしIPAdapterのようにカスタムノードへ
依存する種別は、ファイルが置いてあっても実際に使えるとは限らない。正確に見るなら
ComfyUIを起動して実行する。

```bash
scripts/comfyui-session.sh catalog
```

`--json` で機械可読な形も出せる。

### 直近の生成を引く

`outputs/` の `metadata.json` を読んで、新しい順に出す。ComfyUIへは接続しない。

```bash
uv run imagegen history --limit 5
uv run imagegen history --prefix yui
uv run imagegen history --json
```

```text
2026-08-15 11:00  img2img  seed 271828182
  hassakuSD15_v13.safetensors  <- inputs/yui-ref-f.png  x1.5
  outputs/2026-08-15/105506_yui_ref_f_up/image_0001.png
```

「さっきの子で別の場面を」のような依頼で、基準画像と使ったseedを記録から引くために使う。
キャラクタを台帳へ登録してあるなら、次の `character` のほうが確実に引ける。

### キャラクタ台帳

同じキャラクタを別の場面で出すには、character preset・style preset・checkpoint・
基準画像・seedの5つが揃っている必要がある。`registry/characters/<name>.yaml` へ書いておくと
名前で引ける。ComfyUIへは接続しない。

```yaml
# registry/characters/aoi.yaml
description: 青い髪の少女
preset: anime-girl-blue                 # presets/characters/ のファイル名
style: sd15-hassaku                     # presets/styles/ のファイル名
checkpoint: hassakuSD15_v13.safetensors # 基準画像を作ったcheckpoint
reference: inputs/aoi.png               # IPAdapterへ渡す基準画像
seed: 777001                            # 基準画像を作ったときのseed
notes: 制服は赤いタイ
```

```bash
uv run imagegen character list
uv run imagegen character show aoi
uv run imagegen character show aoi --json
```

```text
Character: aoi
Description: 青い髪の少女
Preset: anime-girl-blue
Style: sd15-hassaku
Checkpoint: hassakuSD15_v13.safetensors
Reference: inputs/aoi.png
Seed: 777001
```

台帳は生成には関与しない。Specへ書き写す値を1か所へまとめるだけで、
`preset` / `style` / `reference` が実在するかは引くたびに確かめる。欠けていれば
標準エラーへ `warning:` が出る (exit codeは0のまま)。置き場は
`IMAGEGEN_REGISTRY_ROOT` で差し替えられる。

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
Checkpoint: meinamix_v12Final.safetensors
Estimate: XPU 約1分 / CPU 約12分
```

preset・LoRA・ControlNet・IPAdapter・テキスト合成を指定している場合は、
適用内容と実際に使われるテンプレート名も表示される。

`Estimate:` は解像度・steps・batch・hires fixから出した概算。どちらの実行基盤で動くかは
接続しないと分からないため両方出す (img2imgは入力画像のサイズで生成するため出さない)。

検証を通ったうえで気になる点があると、標準エラーへ `warning:` が出る。
exit codeは0のままで、生成もできる。

```text
warning: style presetを指定していません。clip skipと外部VAE、そのcheckpointで詰めた
sampler / scheduler / cfg / stepsはstyle presetが持ちます。hassakuSD15_v13.safetensors には
sd15-hassaku が対応します
```

出るのは次の3つ。

- style presetを指定していない
- 指定したstyle presetが`model.checkpoint`とは別のcheckpoint向け
- XPUでも見積りが`IMAGEGEN_TIMEOUT`を超える

```text
Workflow: txt2img_lora
Checkpoint: meinamix_v12Final.safetensors
LoRA: add_detail.safetensors (model=0.8, clip=0.8)
Presets: character=anime-girl-blue, style=anime-soft
```

### 生成

```bash
scripts/comfyui-session.sh generate specs/examples/txt2img.yaml
```

`scripts/comfyui-session.sh` はComfyUIを起動し、生成し、必ず停止する
(失敗しても停止する)。ComfyUIを常駐させたままモデルを何本も切り替えると
XPUのアロケータが断片化し、空き容量が十分でも数十MiBの確保に失敗するようになるため、
1回の生成ごとにプロセスを立て直す。既に起動しているComfyUIがある場合はそれを使い、
停止もしない。`batch` / `validate` / `health` も同じように渡せる。

ComfyUIを自分で起動しているなら、CLIを直接呼んでもよい。

```bash
uv run imagegen generate specs/examples/txt2img.yaml
```

```text
prompt_id: 5f2c...
seed: 883021
directory: /path/to/outputs/2026-08-12/143052_blue_hair
/path/to/outputs/2026-08-12/143052_blue_hair/image_0001.png
metadata: /path/to/outputs/2026-08-12/143052_blue_hair/metadata.json
```

タイムアウトを個別指定する場合:

```bash
uv run imagegen generate specs/examples/txt2img.yaml --timeout 600
```

失敗時は原因ごとに異なるexit codeを返す
(一覧は [CLAUDE.mdの「exit code」](CLAUDE.md#exit-code))。

### 一括生成

複数のSpecをまとめて実行する。seed掃引もできる。

```bash
scripts/comfyui-session.sh batch specs/generated/a.yaml specs/generated/b.yaml
scripts/comfyui-session.sh batch specs/generated/a.yaml --seeds 111,222,333
```

```text
[1/2] specs/generated/a.yaml (seed=111)
  -> /path/to/outputs/2026-08-12/143052_sample/image_0001.png
[2/2] specs/generated/a.yaml (seed=222)
  -> /path/to/outputs/2026-08-12/143118_sample/image_0001.png
成功 2 / 失敗 0
```

**1件失敗しても残りは続ける。** 途中で止まるとどこまで進んだのか分からなくなるため。
全件終わったあとにサマリを出し、失敗があれば最初の失敗のexit codeで終了する。

Specの検証は実行前に全件まとめて行う。不正なSpecが混ざっていた場合は1件も生成しない。

### テキストの後合成

生成済みの画像へ日本語を合成する。入力画像は変更しない。

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml
uv run imagegen compose inputs/base.png specs/generated/caption.yaml -o outputs/caption.png
```

## GenerationSpec

エージェントとの境界となるYAML。最小構成は次のとおり。

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

model:
  checkpoint: meinamix_v12Final.safetensors

output:
  prefix: blue_hair
```

追加のブロックを書くと機能が有効になり、使うWorkflowテンプレートも自動的に切り替わる。

| ブロック | 何ができるか | 仕様 |
| --- | --- | --- |
| `presets` | character / scene / styleを名前で参照して再利用する | [presets](docs/spec-reference.md#presets) |
| `model.loras` | LoRAを重ねて適用する | [model.loras](docs/spec-reference.md#modelloras) |
| `model.unet` / `clip` / `vae` | DiT系モデル (Anima) を3ローダ構成で使う | [DiT系モデル](docs/spec-reference.md#dit系モデル-anima) |
| `source` | 既存画像を入力にして描き直す (img2img) | [source](docs/spec-reference.md#source-img2img) |
| `control` | 参考画像から線画 (Canny) を取り構図を保つ。前処理済みの画像もそのまま渡せる | [control](docs/spec-reference.md#control-controlnet) |
| `reference` | 参照画像の顔立ち・服装・画風を引き継ぐ (IPAdapter) | [reference](docs/spec-reference.md#reference-ipadapter) |
| `generation.upscale` | latentのまま拡大し2段目で描き足す (hires fix) | [generation.upscale](docs/spec-reference.md#generationupscale-hires-fix) |
| `text` | 生成後に日本語テキストを合成する | [text](docs/spec-reference.md#text-テキスト合成) |

**全フィールドの値域・既定値・組み合わせ規則は
[docs/spec-reference.md](docs/spec-reference.md) を参照。**
指定しても効かない項目は黙って無視せず、Specの検証時に拒否する。
IPAdapterには [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) の
導入が要る。それ以外はComfyUI本体のノードだけで動く。

## 出力

```text
outputs/
└── 2026-08-12/
    └── blue_hair/
        ├── image_0001.png
        ├── image_0001_text.png   # text を指定した場合のみ
        └── metadata.json
```

同じ日に同じprefixで再実行した場合は連番ディレクトリを作り、既存の結果を上書きしない。
`metadata.json` には再現に必要な情報 (`resolved_seed` / `workflow_hash` / preset展開後のSpecなど) を
集約する。キーの一覧は [metadata.json](docs/spec-reference.md#metadatajson) を参照。

## 設定

接続先と上限値は環境変数で変えられる。一覧は
[CLAUDE.mdの「環境変数」](CLAUDE.md#環境変数) を参照。
秘密情報は扱わないため、環境変数ファイルは必須ではない。

`IMAGEGEN_BACKEND=diffusers` を指定すると、ComfyUIを立てずに
diffusersでプロセス内推論する。Specの書き方もCLIの使い方も変わらないが、
ControlNet / IPAdapter / hires fix などには対応していない。
設定と対応範囲は [docs/diffusers-backend.md](docs/diffusers-backend.md) を参照。

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

1. `imagegen catalog` で実行基盤と使えるモデル / presetを確認
2. GenerationSpecを作成し `specs/generated/` へ保存
3. `uv run imagegen validate` で検証
4. `uv run imagegen generate` で生成
5. 出力パスとseedを返す

「同じキャラで別の構図」のような依頼では、基準画像を1枚作って `reference` に指定したまま
scene presetだけを差し替える (手順:
[character-consistency.md](.claude/skills/imagegen/references/character-consistency.md))。
失敗時の切り分けは
[troubleshooting.md](.claude/skills/imagegen/references/troubleshooting.md)
にexit codeごとの手順がある。

Claude Codeが守るルール (Workflowを勝手に書き換えない、validationを迂回しないなど) は
[CLAUDE.md](CLAUDE.md) に定義している。

## MCP Server

Claude Code / Codexの双方から同じ基盤を使える。手順とtoolの一覧は
[docs/mcp-setup.md](docs/mcp-setup.md) を参照 (一次情報はそちら)。

```bash
uv run imagegen-mcp   # stdioで待ち受ける (通常はクライアントが子プロセスとして起動する)
```

生成系・確認系・一覧系あわせて14個あり、生成は数十秒から数分かかるため、`generate_image` は
完了を待たずに `job_id` を返し、`get_generation_status` で結果を受け取る。
失敗時はCLIと同じexit codeを返す。

ControlNet・IPAdapter・hires fix・LoRA・img2imgはいずれもSpecの内容で決まるため、MCP側に専用の
パラメータは無い。CLIで書けるSpecはそのままMCPでも使える。

MCP層は薄いアダプタで、検証も生成もCLIと同じService / Domainを通る。
MCP経由で検証を迂回できる経路は作っていない。

## アーキテクチャ

```text
CLI (typer) / MCP Server    入出力とexit codeへの変換
    |
Service (generation)        ユースケースの組み立て、presetの解決
    |                       見えるのは GenerationBackend / CatalogBackend の2つのProtocolだけ
Domain (models / policy)    GenerationSpecと検証規則。外部依存なし
    |
backends.py                 IMAGEGEN_BACKEND から具象を選ぶ
    |
    +-- Adapters (comfyui)    Workflowテンプレート / HTTP / WebSocket / ComfyUI固有のJSON形状
    +-- Adapters (diffusers)  プロセス内推論。Pipelineの読み込みとscheduler対応
```

バックエンド依存は `adapters/<name>/` に閉じ込めてあり、
2つ目のバックエンド (diffusers) を足した際もDomain / Service層の変更は不要だった。
CPU推論からIntel XPUへ切り替えた際も `src/` の変更は要らなかった。

## ライセンス

[LICENSE](LICENSE) を参照。
