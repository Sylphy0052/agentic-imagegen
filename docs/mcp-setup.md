# MCP Serverの接続手順

`agentic-imagegen` をMCP Serverとして起動し、Claude Code / Codex の双方から
同じ生成基盤を使うための手順。

CLIは引き続き残る。MCP経由で不具合が出たときに、CLIで層を切り分けられる状態を保つため。

## 前提

- `uv sync` 済み (開発環境では `mcp` が dev 依存に入っている)
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
| `list_models` | 利用可能なcheckpoint名を返す | 必要 |
| `list_loras` | 利用可能なLoRA名を返す | 必要 |
| `list_workflows` | 実行を許可しているWorkflowテンプレート名を返す | 不要 |

`validate_generation` はpresetを展開したうえで、選択されるテンプレート・解像度・LoRA構成を返す。
不正なSpecでもエラーにはせず `valid: false` と理由を返すので、生成前の確認に使える。

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
7: ComfyUI側で失敗 など)。一覧は [CLAUDE.md](../CLAUDE.md) を参照。

パスは作業ルートからの相対で返す。絶対パスは実行環境の構成を露出するため返さない。

ジョブの状態はサーバープロセスのメモリにのみ持つ。サーバーを再起動すると
`job_id` は失われるが、生成物と `metadata.json` はディスクに残る。

## Claude Code から使う

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

## Codex から使う

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
