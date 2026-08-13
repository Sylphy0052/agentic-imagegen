# MCP Serverの接続手順

`agentic-imagegen` をMCP Serverとして起動し、Claude Code / Codexの双方から
同じ生成基盤を使うための手順。

CLIは引き続き残る。MCP経由で不具合が出たときに、CLIで層を切り分けられる状態を保つため。

## 前提

- `uv sync` 済み (開発環境では `mcp` がdev依存に入っている)
- ComfyUIが起動していること ([xpu-setup.md](xpu-setup.md) / [comfyui-setup.md](comfyui-setup.md))

MCP依存だけを入れる場合はextrasを使う。

```bash
uv pip install -e '.[mcp]'
```

## 起動確認

```bash
uv run imagegen-mcp
```

stdioで待ち受けるため、単体で実行しても何も表示されずに待機する (正常)。
`Ctrl-C` で終了する。実際にはクライアントが子プロセスとして起動する。

## 提供するtool

| tool | 用途 | ComfyUIへの接続 |
| --- | --- | --- |
| `validate_generation` | GenerationSpecを検証する。画像は生成しない | 不要 |
| `generate_image` | 生成を開始し、`job_id` を返す | 必要 |
| `get_generation_status` | 生成の状態と結果を返す | 不要 |
| `generate_batch` | 複数のSpecをまとめて生成し、`job_id` を返す | 必要 |
| `get_batch_status` | 一括生成の状態と結果を返す | 不要 |
| `list_models` | 利用可能なcheckpoint名を返す | 必要 |
| `list_loras` | 利用可能なLoRA名を返す | 必要 |
| `list_controlnets` | 利用可能なControlNetモデル名を返す | 必要 |
| `list_ipadapters` | 利用可能なIPAdapterモデル名を返す | 必要 |
| `list_clip_visions` | 利用可能なCLIP Visionモデル名を返す | 必要 |
| `list_diffusion_models` | 利用可能なUNet単体のモデル名を返す (DiT系) | 必要 |
| `list_text_encoders` | 利用可能なtext encoder名を返す (DiT系) | 必要 |
| `list_vaes` | 利用可能なVAE名を返す (DiT系) | 必要 |
| `list_workflows` | 実行を許可しているWorkflowテンプレート名を返す | 不要 |

`validate_generation` はpresetを展開したうえで、選択されるテンプレート・解像度・LoRA構成・
ControlNet (`control`)、IPAdapter (`reference`)、hires fix (`generation.upscale`) の設定を返す。
不正なSpecでもエラーにはせず `valid: false` と理由を返すので、生成前の確認に使える。

### CLIとの機能差

Specに書ける項目はCLIとMCPで同じものが使える。ControlNet・IPAdapter・hires fix・LoRA・img2imgは
いずれもSpecの内容で決まるため、MCP側に専用のパラメータは無い。使うWorkflowテンプレートも
Specから自動的に決まる。

```json
{
  "task": "txt2img",
  "prompt": {"positive": "1girl, blue hair"},
  "generation": {"width": 512, "height": 512, "steps": 20, "seed": 1234},
  "model": {"checkpoint": "meinamix_v12Final.safetensors"},
  "control": {
    "image": "inputs/pose.png",
    "model": "control_v11p_sd15_canny_fp16.safetensors",
    "strength": 0.8
  }
}
```

ControlNetモデルは `list_controlnets` で実在するものを確認してから指定する。
`control` の画像パスは作業ルートからの相対で書く (CLIと同じ規則)。

IPAdapterを使う場合は `reference` を書く。モデル名は `list_ipadapters` / `list_clip_visions`
で確認する。`list_ipadapters` が空ならカスタムノードが未導入で、`reference` は使えない。

```json
{
  "reference": {
    "image": "inputs/character.png",
    "model": "ip-adapter-plus_sd15.safetensors",
    "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "weight": 0.8
  }
}
```

### 生成の流れ

生成は数十秒から数分かかるため、`generate_image` は完了を待たずに `job_id` を返す。

```text
generate_image(spec)
  -> {"job_id": "3f1c...", "status": "running", "workflow": "txt2img_lora"}

get_generation_status(job_id)
  -> {"status": "running", ...}
  -> {"status": "completed",
      "seed": 24680,
      "files": ["outputs/2026-08-12/sample/image_0001.png"],
      "metadata_path": "outputs/2026-08-12/sample/metadata.json",
      "error": null, "exit_code": null}
```

失敗した場合は `status: failed` とともに理由と `exit_code` を返す。
exit codeはCLIと同じ体系 (2: Specが不正 / 3: ComfyUIへ到達できない / 6: タイムアウト /
7: ComfyUI側で失敗 など)。一覧は
[CLAUDE.mdの「exit code」](../CLAUDE.md#exit-code) を参照。

パスは作業ルートからの相対で返す。絶対パスは実行環境の構成を露出するため返さない。

ジョブの状態はサーバープロセスのメモリにのみ持つ。サーバーを再起動すると
`job_id` は失われるが、生成物と `metadata.json` はディスクに残る。

### 一括生成の流れ

同じSpecでseedを変えて何枚か出したい場合や、複数のSpecを流したい場合は `generate_batch` を使う。
`seeds` を指定すると、Specごとに各seedを当てたものへ展開する (Spec1件 + seed3つなら3枚)。

```text
generate_batch(specs=[specA, specB], seeds=[111, 222])
  -> {"job_id": "bd4a...", "status": "running", "total": 4,
      "items": [{"label": "spec[0] (seed=111)", "workflow": "txt2img"}, ...]}

get_batch_status(job_id)
  -> {"status": "running", "total": null, "items": []}
  -> {"status": "completed", "total": 4, "succeeded": 3, "failed": 1,
      "items": [{"label": "spec[0] (seed=111)", "status": "completed",
                 "seed": 111, "files": [...], "error": null, "exit_code": null},
                {"label": "spec[1] (seed=222)", "status": "failed",
                 "seed": null, "files": [], "error": "...", "exit_code": 6}]}
```

- 検証は投入前に全件行う。1件でも不正なら1件も生成しない。このときtool呼び出し自体がエラーになる
- 生成は順に実行され、1件失敗しても残りは続く。そのため失敗が混ざっていても
  ジョブ全体の `status` は `completed` になる。内訳は `succeeded` / `failed` と `items` で判断する
- Specがファイルとして存在しないため、`label` には受け取った並びの位置 (`spec[0]`) が入る
- 実行中は件数がまだ確定した結果として無いため `total` は `null` になる。投入時の戻り値に入っている

## Claude Codeから使う

**リポジトリに [.mcp.json](../.mcp.json) を同梱しているため、設定は不要。**
このリポジトリをClaude Codeで開くと検出される。

```json
{
  "mcpServers": {
    "agentic-imagegen": {
      "command": "uv",
      "args": ["run", "imagegen-mcp"]
    }
  }
}
```

パスを書いていないのは、Claude Codeがプロジェクトルートをカレントディレクトリとして
サーバーを起動するため。環境に依存しないのでそのままコミットできる。

初回は承認が要る。承認前は次のように表示される。

```bash
$ claude mcp list
agentic-imagegen: uv run imagegen-mcp - ⏸ Pending approval (run `claude` to approve)
```

`claude` を起動して承認すると使えるようになる。

別ディレクトリから使う場合はコマンドで登録する。

```bash
claude mcp add agentic-imagegen -- uv run --directory /path/to/agentic-imagegen imagegen-mcp
```

## Codexから使う

```bash
codex mcp add agentic-imagegen -- uv run --directory /path/to/agentic-imagegen imagegen-mcp
```

`~/.codex/config.toml` へ次の形で書き込まれる。手で編集してもよい。

```toml
[mcp_servers.agentic-imagegen]
command = "uv"
args = ["run", "--directory", "/path/to/agentic-imagegen", "imagegen-mcp"]
```

登録を確認する。

```bash
codex mcp get agentic-imagegen
codex mcp list
```

外すときは `codex mcp remove agentic-imagegen`。

### 非対話実行 (`codex exec`) では使えない

`codex exec` からMCP toolを呼ぶと、承認プロンプトを出せないため必ずキャンセルされる。

```text
mcp: agentic-imagegen/list_workflows started
mcp: agentic-imagegen/list_workflows (failed)
user cancelled MCP tool call
```

これはCodex側の既知の制限で、`approval_policy` や `default_tools_approval_mode` を
変えても回避できない ([openai/codex#24135](https://github.com/openai/codex/issues/24135) /
[#16685](https://github.com/openai/codex/issues/16685))。
サーバー自体は起動しており、toolも認識されている (上記ログの `started` がその証拠)。

**対話セッション (`codex`) で承認すれば使える。**
自動化したい場合の回避策は `--dangerously-bypass-approvals-and-sandbox` のみだが、
サンドボックスごと無効化するため常用しない。

`uv run --directory` はカレントディレクトリも変更するため、作業ルートは
リポジトリルートに解決される。`IMAGEGEN_PROJECT_ROOT` の指定は不要。

```toml
# 作業ルートを明示したい場合のみ
[mcp_servers.agentic-imagegen.env]
IMAGEGEN_PROJECT_ROOT = "/path/to/agentic-imagegen"
```

どちらのクライアントもstdioで同じサーバーを起動するため、サーバー側に
クライアント固有の分岐は持たせていない。

## 作業ルートについて

出力先と入力画像はリポジトリ配下へ解決される。その基準となる作業ルートは、
既定ではサーバープロセスのカレントディレクトリになる。

クライアントが任意のディレクトリからサーバーを起動するため、
**`IMAGEGEN_PROJECT_ROOT` を明示することを推奨する。**

## 環境変数

CLIと共通。詳細は [README](../README.md) を参照。

| 変数 | 用途 |
| --- | --- |
| `COMFYUI_BASE_URL` | ComfyUI接続先 |
| `IMAGEGEN_PROJECT_ROOT` | 作業ルート (MCP専用。未指定ならcwd) |
| `IMAGEGEN_PRESETS_ROOT` | presetの探索ルート |
| `IMAGEGEN_MAX_*` | 各種上限 |

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| toolが表示されない | クライアント側でサーバーの起動に失敗している。`uv run imagegen-mcp` を手で実行してエラーを確認する |
| Claude Codeで `Pending approval` のまま | `.mcp.json` は検出できている。`claude` を起動して承認する |
| Codexで `user cancelled MCP tool call` | `codex exec` (非対話) の既知の制限。対話セッション (`codex`) で承認して使う。詳細は下記 |
| `list_models` が失敗する | ComfyUIが未起動。`uv run imagegen health` で確認する |
| presetが見つからない | 作業ルートがずれている。`IMAGEGEN_PROJECT_ROOT` を明示する |
| 出力先がおかしい | 同上。出力は作業ルート配下へ解決される |
