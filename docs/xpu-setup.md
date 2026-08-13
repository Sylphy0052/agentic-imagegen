# Intel XPU (iGPU) でComfyUIを動かす手順 (WSL2)

NVIDIA GPUを持たない開発機で、Intel Arc内蔵GPU (Core Ultra 7 165H / Xe-LPG) を
PyTorchのXPUバックエンド経由で使うための手順。2026-08-12に実機で構築し動作を確認している。

CPU推論の手順は [comfyui-setup.md](comfyui-setup.md) を参照。
背景と検証経緯は [Issue #2](https://github.com/Sylphy0052/agentic-imagegen/issues/2)。

## 結論

WSL2からIntel iGPUを使える。ただし **compute runtimeのバージョンが古いと `torch.xpu.is_available()` がsegfaultする**。
Ubuntu 22.04 (jammy) の場合、Intelリポジトリの `client` コンポーネントでは足りず、`unified` を使う必要がある。

## 前提

| 項目 | 実機の値 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (WSL2) / kernel 6.18.33.2-microsoft-standard-WSL2 |
| CPU / GPU | Intel Core Ultra 7 165H / 内蔵Arc Graphics (`0x7d55`, Xe-LPG) |
| WSL GPU | `/dev/dxg` あり |

事前調査では否定的な情報が多かったが (WSL2でのArc検出失敗報告、Meteor Lakeのpassthrough不具合)、
本機の構成では下記手順で動作した。

## 1. Intel GPU compute runtimeの導入

Intelのリポジトリを追加し、`unified` コンポーネントから導入する。

```bash
curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
sudo apt install -y libze1 libze-intel-gpu1 intel-opencl-icd intel-ocloc clinfo
```

**`client` ではなく `unified` を指定するのが要点。** jammyの `client` コンポーネントは
`libze-intel-gpu1` が24.39.31294 (2024年10月版) 止まりで、PyTorch 2.13のXPUバックエンドから
呼ぶとデバイス列挙時にsegfaultする。`unified` には25.18.33578があり、これで解消する。

導入後のバージョン (動作確認時):

| パッケージ | バージョン |
| --- | --- |
| `libze-intel-gpu1` | 25.18.33578.15-1146~22.04 |
| `intel-opencl-icd` | 25.18.33578.15-1146~22.04 |
| `libze1` | 1.21.9.0-1136~22.04 |
| `intel-ocloc` | 25.18.33578.15-1146~22.04 |

`intel-level-zero-gpu` (1.3系の旧命名) は入れない。`libze-intel-gpu1` と競合しうる。

### runtimeの確認

```bash
clinfo -l
```

```text
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics [0x7d55]
```

`/usr/lib/wsl/lib` にIntel由来のライブラリは置かれない (`libd3d12.so` などのみ)。
WSL2ではcompute runtimeをWSL内へaptで入れる方式であり、NVIDIAのようにWindowsドライバから
`.so` がマップされるわけではない。

## 2. PyTorchをXPU版へ差し替える

```bash
uv pip install --python ~/ComfyUI/.venv/bin/python \
  --index-url https://download.pytorch.org/whl/xpu --reinstall \
  torch torchvision torchaudio
```

**`--reinstall` が必要。** 同じバージョン番号のCPU版 (`2.13.0+cpu`) が入っていると、
`+xpu` へ入れ替わらずスキップされる。

torchと一緒に `intel-sycl-rt` / `intel-cmplr-lib-rt` / `intel-opencl-rt` / `intel-pti` /
`tcmlib` / `umf` / `pytorch-triton-xpu` が入る。これらはpip側のSYCLランタイムで、
手順1で入れたシステム側のcompute runtimeと組み合わせて動作する。

### PyTorchの確認

```bash
~/ComfyUI/.venv/bin/python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_properties(0).name)"
```

```text
True Intel(R) Graphics [0x7d55]
```

`UserWarning: Can't initialize Level Zero Sysman` が出るが、これは電力・温度などの
監視APIが使えないだけで、推論には影響しない。

## 3. ComfyUIをXPUで起動する

CPU推論のときに付けていた `--cpu` を外す。ComfyUIはXPUを自動検出する。

```bash
cd ~/ComfyUI
./.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

起動ログにデバイスが出る。

```text
[INFO] Device: xpu:0 Intel(R) Graphics [0x7d55]
[INFO] model weight dtype torch.float16, manual cast: None
```

CLI側からも確認できる。

```bash
uv run imagegen health
```

```text
ComfyUI: reachable
URL: http://127.0.0.1:8188
Version: 0.32.0
Devices: xpu:0 Intel(R) Graphics [0x7d55]
```

## 4. CPU推論との比較 (2026-08-12実測)

同一マシン、同一checkpoint (MeinaMix V12 / SD1.5)、同一seedでの比較。

| 条件 | CPU | XPU | 比 |
| --- | --- | --- | --- |
| 512x768 / steps 20 / batch 1 | 約720秒 (36秒/step) | **135.3秒** | 約5.3倍 |
| Integration Test 4件 (512x512 / steps 2) | 91.7秒 | **17.0秒** | 約5.4倍 |

サンプリング中のstep単体では2.7-3.0秒/stepであり、CPUの36秒/stepに対して約12倍速い。
全体が5倍程度に留まるのは、モデルのロードとVAEデコード、および初回のカーネルコンパイルが
支配的になるため。1step目だけは17秒前後かかり、以降2.7-3.0秒へ落ち着く。

生成結果は、同一seedでもCPUとbit単位では一致しない (XPUはfp16、CPUはfp32で動作するため)。
実際に同一Specから生成した画像を比較したところ、構図・配色は同一で、品質上の破綻もなかった。

`torch.xpu` 単体の健全性も確認済み。512x512のmatmulでCPU結果と `allclose` (atol=1e-3) が成立し、
fp16の相対誤差は4.2e-4だった。

### 所要時間とタイムアウトの目安

**実測値はこの節を一次情報とする。** 他の文書は代表値1行とこの節への参照に留める。

| 条件 | 実行基盤 | 実測 | `IMAGEGEN_TIMEOUT` の目安 |
| --- | --- | --- | --- |
| SD1.5 / 512x768 / 20 steps | Intel XPU | 約135秒 (初回はモデルロード込み) | 300 |
| SD1.5 / 512x768 / 20 steps | CPU | 約12分 (36秒/step) | 1200 |
| SD1.5 / 512x512 -> 768x768 (hires fix) | Intel XPU | 43.7秒 | 300 |

ControlNet / IPAdapterを使うと1-2割、hires fixを使うと倍以上に伸びる。
SDXL / Illustrious系 (`novaAnimeXL_ilV190.safetensors`) はさらに遅く、常用しない。
再計測した場合はこの表を直し、参照側の代表値と食い違っていないかだけを確認する。

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `torch.xpu.is_available()` でsegfault (rc=139) | compute runtimeが古い。`unified` コンポーネントから25.18以上を入れる |
| `torch.__version__` が `+cpu` のまま | `--reinstall` を付けずに入れ直した。同一バージョン番号だとスキップされる |
| `clinfo -l` でデバイスが出ない | Windows側のIntelグラフィックスドライバを更新する。`/dev/dxg` の有無も確認する |
| `Can't initialize Level Zero Sysman` | 無害。Sysman (監視API) が使えないだけで推論には影響しない |
| ComfyUIが `Device: cpu` で起動する | `--cpu` を外し忘れている。またはvenvのtorchが `+cpu` のまま |
