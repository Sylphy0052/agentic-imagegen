# ComfyUIセットアップ手順 (WSL / CPU推論)

本プロジェクトのIntegration TestとE2Eを実行するための、ComfyUI導入手順。
2026-08-12にこの手順で実際に構築し、動作を確認している。

## 前提

Phase 1では **WSL上でCPU推論** を使う。理由は次のとおり。

- 開発機にNVIDIA GPUがなく、GPUはIntel Arc Graphics (Core Ultra 7 165H内蔵iGPU) のみ
- したがってCUDA前提の一般的な手順は使えない
- CPU推論は追加ドライバ不要で確実に動作し、Phase 1のゴール (一気通貫の動作確認) には十分

Intel XPUによる高速化とWindows側ComfyUIの代替案は
[Issue #2](https://github.com/Sylphy0052/agentic-imagegen/issues/2) で別途扱う。

CPU推論の所要時間 (Core Ultra 7 165H / 22スレッド / WSL2での実測):

| モデル | 解像度 | steps | 実測 |
| --- | --- | --- | --- |
| MeinaMix V12 (SD1.5) | 512x512 | 2 | Integration Test 4件で計91.7秒 (モデルロード込み) |
| MeinaMix V12 (SD1.5) | 512x768 | 20 | **約12分** (36秒/step) |
| SDXL / Illustrious系 | 1024x1024 | 25 | 未実測。上記から数十分規模と見込まれる |

CPU推論は1stepあたり数十秒かかる。反復作業では steps を下げるか解像度を落とす。
SDXL / Illustrious系は仕上がり確認用と位置づけ、常用しない。

## 1. ComfyUIの取得

本リポジトリの外に置く (このリポジトリには取り込まない)。

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
```

## 2. Python環境とCPU版PyTorch

```bash
cd ~/ComfyUI
uv venv --python 3.12
uv pip install --python ~/ComfyUI/.venv/bin/python \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ~/ComfyUI/.venv/bin/python -r ~/ComfyUI/requirements.txt
```

`--index-url` にCPU版のインデックスを指定するのが要点。指定しないとCUDA版が入り、
NVIDIA GPUのない環境では数GBの無駄なダウンロードになる。

構築実績: ComfyUI 0.32.0 / torch 2.13.0+cpu / Python 3.12.13。

## 3. checkpointの配置

checkpointは `~/ComfyUI/models/checkpoints/` へ置く。
配置したファイル名を、GenerationSpecの `model.checkpoint` に指定する。

ファイル名にはPath Traversal対策の検証がかかる。サブフォルダは1階層まで、
拡張子は `.safetensors` / `.ckpt` のみ許可される。

### Hugging Faceから取得する場合 (認証不要)

```bash
cd ~/ComfyUI/models/checkpoints
curl -L -o v1-5-pruned-emaonly.safetensors \
  https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
```

### civitaiから取得する場合 (APIキーが必要)

civitaiのモデルダウンロードは未認証だと403になる。
civitai → Account settings → API Keys でキーを発行し、
**リポジトリの外**に保存する。

```bash
mkdir -p ~/.config/civitai
printf '%s' '<発行したAPIキー>' > ~/.config/civitai/token
chmod 600 ~/.config/civitai/token
```

APIキーはコマンドライン引数に渡さない (`ps` から見えるため)。ファイルから読んでヘッダに載せる。

```bash
TOKEN="$(tr -d '\r\n' < ~/.config/civitai/token)"
curl -L --fail -C - -H "Authorization: Bearer ${TOKEN}" \
  -o ~/ComfyUI/models/checkpoints/meinamix_v12Final.safetensors \
  https://civitai.com/api/download/models/948574
```

本プロジェクトで動作確認に使っているモデル:

| モデル | baseModel | ファイル | 用途 |
| --- | --- | --- | --- |
| [MeinaMix V12](https://civitai.com/models/7240) | SD 1.5 | `meinamix_v12Final.safetensors` (2.0GB) | Integration Test / E2Eの標準 |
| [Nova Anime XL IL v19.0](https://civitai.com/models/376130) | Illustrious (SDXL系) | `novaAnimeXL_ilV190.safetensors` (6.5GB) | 仕上がり確認用 |
| [Hassaku (Anima) v1.3](https://civitai.com/models/2641326) | Anima (DiT) | `hassakuAnima_v13_int8.safetensors` (2.1GB) | 未対応。[Issue #6](https://github.com/Sylphy0052/agentic-imagegen/issues/6) で調査中 |

Hassaku (Anima) はSDXL系ではなくDiT系のアーキテクチャであり、
同梱の `workflows/txt2img.json` (CheckpointLoaderSimple + KSampler構成) では動かない可能性が高い。

## 4. 起動

```bash
cd ~/ComfyUI
./.venv/bin/python main.py --cpu --listen 127.0.0.1 --port 8188
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
Version: 0.32.0
Devices: cpu
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
- checkpointはComfyUIが実際に持っているものから選ぶ。SD1.5系を優先する
- 特定のcheckpointで実行したい場合は `IMAGEGEN_TEST_CHECKPOINT` を指定する
- 待ち時間は `IMAGEGEN_TIMEOUT` で調整できる (既定180秒)

```bash
IMAGEGEN_TEST_CHECKPOINT=meinamix_v12Final.safetensors \
IMAGEGEN_TIMEOUT=300 uv run pytest -m integration
```

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `imagegen health` が unreachable | ComfyUIが未起動、またはポートが違う。`COMFYUI_BASE_URL` を確認する |
| checkpointが見つからないと言われる | `models/checkpoints/` 配下のファイル名と `model.checkpoint` の指定が一致しているか確認する |
| civitaiのダウンロードが403 | APIキーが未設定。`Authorization: Bearer <key>` ヘッダを付ける |
| 生成がタイムアウトする | CPU推論では時間がかかる。`IMAGEGEN_TIMEOUT` を伸ばすか `steps` を下げる。SDXL系は特に遅い |
| メモリ不足で落ちる | `batch_size` を1に、解像度を512x512に下げる。WSLの割当メモリも確認する |
| WorkflowValidationError が出る | テンプレートのノード構成が想定と違う。`workflows/README.md` の手順で書き出し直す |
