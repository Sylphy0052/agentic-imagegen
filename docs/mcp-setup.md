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

プロジェクトルートに `.mcp.json` を置く。

```json
{
  "mcpServers": {
    "agentic-imagegen": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/agentic-imagegen", "imagegen-mcp"],
      "env": {
        "IMAGEGEN_PROJECT_ROOT": "/path/to/agentic-imagegen"
      }
    }
  }
}
```

コマンドで登録することもできる。

```bash
claude mcp add agentic-imagegen -- uv run --directory /path/to/agentic-imagegen imagegen-mcp
```

## Codex から使う

`~/.codex/config.toml` (プロジェクト単位なら `.codex/config.toml`) へ追記する。

```toml
[mcp_servers.agentic-imagegen]
command = "uv"
args = ["run", "--directory", "/path/to/agentic-imagegen", "imagegen-mcp"]

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
| `list_models` が失敗する | ComfyUIが未起動。`uv run imagegen health` で確認する |
| presetが見つからない | 作業ルートがずれている。`IMAGEGEN_PROJECT_ROOT` を明示する |
| 出力先がおかしい | 同上。出力は作業ルート配下へ解決される |
