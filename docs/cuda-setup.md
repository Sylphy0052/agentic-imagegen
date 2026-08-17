# NVIDIA GPU (CUDA) でComfyUIを動かす手順 (WSL2)

WSL2からNVIDIA GPUを使ってComfyUIを動かす手順。2026-08-17に実機で構築し動作を確認している。

Intel XPUの手順は [xpu-setup.md](xpu-setup.md)、CPU推論は [comfyui-setup.md](comfyui-setup.md)
を参照。ComfyUIの取得・モデルの配置・Integration Testの実行は3つの環境で共通のため、
[comfyui-setup.md](comfyui-setup.md) を一次情報とし、ここではCUDA固有の差分だけを書く。

## 結論

WSL2ではWindows側のNVIDIAドライバがそのまま見えるため、WSL内へドライバを入れる必要はない。
Intel XPUと違いcompute runtimeの導入も要らず、**PyTorchをCUDA版で入れるだけで動く**。

3つの環境の中では最も速い。SD1.5 / 512x768 / 20 stepsがXPUで約135秒、CPUで約12分に対し、
CUDAでは約4秒で終わる。

## 前提

| 項目 | 実機の値 |
| --- | --- |
| OS | WSL2 / kernel 6.18.33.2-microsoft-standard-WSL2 |
| CPU | Intel Core i7-14700 (28スレッド) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER 16GB |
| ドライバ | 591.86 (Windows側) / CUDA 13.1 |
| WSLメモリ | 15GB |

WSL内から次が通ることを先に確認する。通らない場合はWindows側のドライバを更新する。

```bash
nvidia-smi
```

`/dev/dri` が無くても問題ない (Intel XPUで使うデバイスファイルであり、CUDAには要らない)。

## 1. ComfyUIの取得とPython環境

取得手順は [comfyui-setup.mdの「1. ComfyUIの取得」](comfyui-setup.md#1-comfyuiの取得) と同じ。

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI
uv venv --python 3.12
```

## 2. PyTorchをCUDA版で入れる

```bash
uv pip install --python ~/ComfyUI/.venv/bin/python \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/ComfyUI/.venv/bin/python -r ~/ComfyUI/requirements.txt
```

**`--index-url` に入れるCUDAバージョンは、ドライバが報告する値より上げない。**
`nvidia-smi` の `CUDA Version` が13.1なら `cu130` を選ぶ。
利用できるインデックスは <https://download.pytorch.org/whl/> に並んでいる。

構築実績: ComfyUI 0.33.0 / torch 2.13.0+cu130 / Python 3.12.12。

導入後にCUDAを掴んでいることを確認する。

```bash
~/ComfyUI/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```text
2.13.0+cu130 True
```

## 3. モデルの配置

[comfyui-setup.mdの「3. checkpointの配置」](comfyui-setup.md#3-checkpointの配置) と同じ。
CUDA固有の差は無い。

## 4. 起動

```bash
cd ~/ComfyUI
./.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

**`--cpu` や `--gpu-only` は付けない。** ComfyUIは起動時にデバイスを自動判定するため、
CUDA版のtorchが入っていればそのままGPUを使う。
本リポジトリの `scripts/comfyui-session.sh` も同じ引数で起動するため、
CUDA環境のための変更は要らない。

別のシェルから到達確認する。

```bash
uv run imagegen health
```

期待する出力:

```text
comfyui: reachable
URL: http://127.0.0.1:8188
Version: 0.33.0
Devices: cuda:0 NVIDIA GeForce RTX 4070 Ti SUPER : cudaMallocAsync
```

`Devices:` が `cpu` になっている場合はCPU版のtorchが入っている。
`--reinstall` を付けて2の手順をやり直す (同じバージョン番号の `+cpu` が入っていると
`+cu130` へ入れ替わらずスキップされる)。

## 所要時間

**実測値の一次情報は
[xpu-setup.mdの「所要時間とタイムアウトの目安」](xpu-setup.md#所要時間とタイムアウトの目安)。**
CUDAの行もそこにまとめてある。

既定の `IMAGEGEN_TIMEOUT` (300秒) で足りなかった条件は今のところ無い。

`imagegen validate` が出す `Estimate:` 行はXPUとCPUの係数から起こしたもので、
**CUDA環境では大幅に過大に出る** (Anima 832x1216 / 32 stepsで「XPU 約11分」と出るが実測10秒)。
CUDAの係数を持たせる件は
[Issue #130](https://github.com/Sylphy0052/agentic-imagegen/issues/130) で扱う。

## VRAMの目安

RTX 4070 Ti SUPER (16GB) での実測。

| 条件 | VRAM使用 |
| --- | --- |
| Anima (DiT系) / 832x1216 / 32 steps | 約5.4GB |

ComfyUIは空きVRAMに応じて自動でオフロードするため、VRAMが少ない環境でも動く。
足りない場合は `--lowvram` を付ける。
