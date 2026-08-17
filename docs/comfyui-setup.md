# ComfyUIセットアップ手順 (WSL / CPU推論)

本プロジェクトのIntegration TestとE2Eを実行するための、ComfyUI導入手順。
2026-08-12にこの手順で実際に構築し、動作を確認している。

## 前提

本手順は **WSL上でCPU推論** を動かす場合のもの。GPUが使えるなら、下記のほうが速い。

| 環境 | 手順 | SD1.5 / 512x768 / 20 stepsの実測 |
| --- | --- | --- |
| NVIDIA GPU (CUDA) | [cuda-setup.md](cuda-setup.md) | 約4秒 |
| Intel GPU (XPU) | [xpu-setup.md](xpu-setup.md) | 約135秒 |
| CPU (本手順) | この文書 | 約12分 |

ComfyUIの取得・モデルの配置・Integration Testの実行はどの環境でも共通のため、
**この文書をそれらの一次情報とする。** CUDA / XPUの文書には固有の差分だけを書く。

CPU推論を選ぶ理由は次のとおり。

- 最初の開発機にNVIDIA GPUがなく、GPUはIntel Arc Graphics (Core Ultra 7 165H内蔵iGPU) のみ
  だった。したがってCUDA前提の一般的な手順は使えなかった
- CPU推論は追加ドライバ不要で確実に動作し、Phase 1のゴール (一気通貫の動作確認) には十分

**2026-08-12追記: Intel XPU (内蔵Arc GPU) での実行が可能になった。** 手順は
[xpu-setup.md](xpu-setup.md) を参照。CPU推論よりおよそ5倍速いため、通常はそちらを使う。
本ドキュメントのCPU手順は、GPUが使えない環境でのフォールバックとして維持する。
経緯は [Issue #2](https://github.com/Sylphy0052/agentic-imagegen/issues/2)。

**2026-08-17追記: NVIDIA GPU (CUDA) での実行手順を足した。** 手順は
[cuda-setup.md](cuda-setup.md) を参照。3つの中で最も速い。

CPU推論はSD1.5 / 512x768 / 20 stepsで**約12分**かかる (Core Ultra 7 165H / 22スレッド / WSL2)。
XPUとの比較を含む実測値の一次情報は
[xpu-setup.mdの「所要時間とタイムアウトの目安」](xpu-setup.md#所要時間とタイムアウトの目安)。

1stepあたり数十秒かかるため、反復作業ではstepsを下げるか解像度を落とす。
SDXL / Illustrious系 (1024x1024 / 25 steps) は未実測だが数十分規模と見込まれる。
仕上がり確認用と位置づけ、常用しない。

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

同梱のサンプルSpec (`specs/examples/`) は `meinamix_v12Final.safetensors` を指している。
サンプルをそのまま動かすなら下のcivitaiの手順で入れる。別のcheckpointを使う場合は
Spec側の `model.checkpoint` を実際のファイル名へ書き換える。

### Hugging Faceから取得する場合 (認証不要)

```bash
cd ~/ComfyUI/models/checkpoints
curl -L -o v1-5-pruned-emaonly.safetensors \
  https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
```

### civitaiから取得する場合 (APIキーが必要)

civitaiのモデルダウンロードは未認証だと403になる。
civitai → Account settings → API Keysでキーを発行し、
**リポジトリの外**に保存する。

```bash
mkdir -p ~/.config/civitai
printf '%s' '<発行したAPIキー>' > ~/.config/civitai/token
chmod 600 ~/.config/civitai/token
```

**`Authorization: Bearer` ヘッダでは通らない (2026-08-17時点)。** civitaiは
ダウンロードをR2 / S3へ302で飛ばすが、curlの `-L` はリダイレクト先まで
Authorizationヘッダを持ち回るため、ストレージ側が
`Missing x-amz-content-sha256` を返して **HTTP 400** になる。
`token=` のクエリで渡すとこれを踏まない。

APIキーはコマンドライン引数に渡さない。`/proc/<pid>/cmdline` は他ユーザーからも読めるため、
`curl "...token=${TOKEN}"` と書くとダウンロード中ずっと見える状態になる
(`TOKEN="$(cat ...)"` とファイルから読んでも、展開後の値が引数に載るので同じこと)。
URLを組み立てた設定ファイルをstdinからcurlへ渡せば、キーは引数にも一時ファイルにも出ない。

```bash
URL="https://civitai.com/api/download/models/948574"
sed "s|^|url = \"${URL}?token=|; s|\$|\"|" ~/.config/civitai/token \
  | curl -K - -L --fail -C - \
      -o ~/ComfyUI/models/checkpoints/meinamix_v12Final.safetensors
```

`--url-query "token@<file>"` でも同じことができるが、curl 7.87以降が要る
(Ubuntu 22.04同梱のcurlは7.81のため使えない)。

### 同名ファイルが複数あるバージョンを落とす場合

1つのバージョンに同じファイル名の候補が複数ぶら下がっていることがある。
その場合は `fileId` まで指定しないと、意図しないほうが落ちる。

```bash
curl -s https://civitai.com/api/v1/model-versions/3195694 \
  | python3 -c "import json,sys; [print(f['id'], f['name'], f['metadata']) for f in json.load(sys.stdin)['files']]"
```

```text
3076709 hassakuAnima_v13.safetensors {'format': 'SafeTensor', 'fp': 'bf16'}
3076685 hassakuAnima_v13.safetensors {'format': 'SafeTensor', 'fp': 'int8'}
```

```bash
URL="https://civitai.com/api/download/models/3195694?fileId=3076685"
sed "s|^|url = \"${URL}\&token=|; s|\$|\"|" ~/.config/civitai/token \
  | curl -K - -L --fail -C - \
      -o ~/ComfyUI/models/diffusion_models/hassakuAnima_v13_int8.safetensors
```

落とし終えたらサイズを照合する。認証や `fileId` を誤ると、モデルではなく
数KBのHTMLが `.safetensors` という名前で保存されることがある。
期待するサイズは `https://civitai.com/api/v1/model-versions/<versionId>` の
`files[].sizeKB` で引ける。

本プロジェクトで動作確認に使っているモデル:

| モデル | baseModel | ファイル | 用途 |
| --- | --- | --- | --- |
| [MeinaMix V12](https://civitai.com/models/7240) | SD 1.5 | `meinamix_v12Final.safetensors` (2.0GB) | Integration Test / E2Eの標準 |
| [Nova Anime XL IL v19.0](https://civitai.com/models/376130) | Illustrious (SDXL系) | `novaAnimeXL_ilV190.safetensors` (6.5GB) | 仕上がり確認用 |
| [Hassaku (Anima) v1.3](https://civitai.com/models/2641326) | Anima (DiT) | `hassakuAnima_v13_int8.safetensors` (2.1GB) | `models/diffusion_models/` へ置く。下記参照 |

Hassaku (Anima) はSDXL系ではなくDiT系のアーキテクチャで、同梱の `workflows/txt2img.json`
(CheckpointLoaderSimple + KSampler構成) では動かない。専用の `txt2img_unet` を使う。

### DiT系モデル (Anima) を置く場合

Anima系のモデルはUNet単体で配布され、text encoderとVAEを同梱していない。
`models/checkpoints/` へ置いて `CheckpointLoaderSimple` で読ませると、CLIPとVAEが
`None` のままになり `clip input is invalid: None` で失敗する。3つを別々に置く。

| ファイル | 置き場 | サイズ | 入手元 |
| --- | --- | --- | --- |
| `hassakuAnima_v13_int8.safetensors` | `models/diffusion_models/` | 2.10GB | [civitai](https://civitai.com/models/2641326) (APIキー要) |
| `qwen_3_06b_base.safetensors` | `models/text_encoders/` | 1.19GB | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) |
| `qwen_image_vae.safetensors` | `models/vae/` | 254MB | 同上 |

```bash
cd ~/ComfyUI/models/text_encoders
curl -L -o qwen_3_06b_base.safetensors \
  https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors

cd ~/ComfyUI/models/vae
curl -L -o qwen_image_vae.safetensors \
  https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors
```

int8量子化版はComfyUI独自の量子化形式 (`comfy_quant`) を使っており、
`UNETLoader` の `weight_dtype=default` でそのまま読める。bf16版 (3.90GB) は不要。

Spec側の書き方は
[spec-reference.mdの「DiT系モデル (Anima)」](spec-reference.md#dit系モデル-anima) を参照。

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
- 待ち時間は `IMAGEGEN_TIMEOUT` で調整できる (既定300秒)
- ControlNetモデルやDiT系の UNet / text encoder / VAE を置いていない環境では、
  それらを使うケースだけがskipされる。全体は失敗しない

txt2img / hires fix / ControlNet / 両者の併用 / img2img / DiT系 (単体とimg2img+hires fix)
を通しで確認する。XPU環境での所要時間は全10件で約90秒。

```bash
IMAGEGEN_TEST_CHECKPOINT=meinamix_v12Final.safetensors \
IMAGEGEN_TIMEOUT=300 uv run pytest -m integration
```

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `imagegen health` がunreachable | ComfyUIが未起動、またはポートが違う。`COMFYUI_BASE_URL` を確認する |
| checkpointが見つからないと言われる | `models/checkpoints/` 配下のファイル名と `model.checkpoint` の指定が一致しているか確認する |
| civitaiのダウンロードが403 | APIキーが未設定。`Authorization: Bearer <key>` ヘッダを付ける |
| 生成がタイムアウトする | CPU推論では時間がかかる。`IMAGEGEN_TIMEOUT` を伸ばすか `steps` を下げる。SDXL系は特に遅い |
| メモリ不足で落ちる | `batch_size` を1に、解像度を512x512に下げる。WSLの割当メモリも確認する |
| WorkflowValidationErrorが出る | テンプレートのノード構成が想定と違う。`workflows/README.md` の手順で書き出し直す |
