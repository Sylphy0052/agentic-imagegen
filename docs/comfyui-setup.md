# ComfyUIセットアップ手順 (WSL / CPU推論)

本プロジェクトのIntegration TestとE2Eを実行するための、ComfyUI導入手順。

## 前提

Phase 1では **WSL上でCPU推論** を使う。理由は次のとおり。

- 開発機にNVIDIA GPUがなく、GPUはIntel Arc Graphics (Core Ultra 7 165H内蔵iGPU) のみ
- したがってCUDA前提の一般的な手順は使えない
- CPU推論は追加ドライバ不要で確実に動作し、Phase 1のゴール (一気通貫の動作確認) には十分

Intel XPUによる高速化とWindows側ComfyUIの代替案は
[Issue #2](https://github.com/Sylphy0052/agentic-imagegen/issues/2) で別途扱う。

想定所要時間 (SD1.5 / 512x512):

| 用途 | 設定 | 目安 |
| --- | --- | --- |
| Integration Test | steps 2-4 | 10-20秒 |
| 通常の生成 | steps 20 | 1-2.5分 |

SDXLはCPUでは1枚5-15分かかるため、Phase 1ではSD1.5系を使う。

## 1. ComfyUIの取得

本リポジトリの外に置く (このリポジトリに取り込まない)。

```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ~/ComfyUI
```

## 2. Python環境とCPU版PyTorch

```bash
cd ~/ComfyUI
uv venv --python 3.12
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

`--index-url` にCPU版のインデックスを指定するのが要点。指定しないとCUDA版が入り、
NVIDIA GPUのない環境では無駄に大きなダウンロードになる。

## 3. checkpointの配置

SD1.5系のcheckpointを `models/checkpoints/` へ置く。

```bash
cd ~/ComfyUI/models/checkpoints
# 例: Stable Diffusion v1.5 (約4GB)
curl -L -o v1-5-pruned-emaonly.safetensors \
  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

配置したファイル名を、GenerationSpecの `model.checkpoint` に指定する。
ファイル名の指定にはPath Traversal対策の検証がかかるため、
サブフォルダは1階層まで、拡張子は `.safetensors` / `.ckpt` のみ許可される。

## 4. 起動

```bash
cd ~/ComfyUI
uv run python main.py --cpu --listen 127.0.0.1 --port 8188
```

- `--cpu` : CPU推論を強制する
- `--listen 127.0.0.1` : ループバックのみで待ち受ける。**`0.0.0.0` で公開しない**

起動後、別のシェルから到達確認する。

```bash
cd <このリポジトリ>
uv run imagegen health
```

期待する出力:

```text
ComfyUI: reachable
URL: http://127.0.0.1:8188
```

接続先を変える場合は `COMFYUI_BASE_URL` を設定する。

## 5. Workflowテンプレート

本リポジトリには `workflows/txt2img.json` を同梱済み。
ComfyUI標準のtxt2imgグラフと同じノード構成のため、通常は差し替え不要。

自環境に合わせて作り直す場合の手順とノード構成は
[workflows/README.md](../workflows/README.md) を参照。

## 6. Integration Testの実行

```bash
uv run pytest -m integration
```

- ComfyUIが起動していない場合は失敗ではなくskipし、理由を表示する
- 待ち時間は `IMAGEGEN_TIMEOUT` で調整できる (既定180秒)

```bash
IMAGEGEN_TIMEOUT=300 uv run pytest -m integration
```

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `imagegen health` が unreachable | ComfyUIが未起動、またはポートが違う。`COMFYUI_BASE_URL` を確認する |
| checkpointが見つからないと言われる | `models/checkpoints/` 配下のファイル名と `model.checkpoint` の指定が一致しているか確認する |
| 生成がタイムアウトする | CPU推論では時間がかかる。`IMAGEGEN_TIMEOUT` を伸ばすか `steps` を下げる |
| メモリ不足で落ちる | `batch_size` を1に、解像度を512x512に下げる。WSLの割当メモリも確認する |
| WorkflowValidationError が出る | テンプレートのノード構成が想定と違う。`workflows/README.md` の手順で書き出し直す |
