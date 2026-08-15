# diffusersバックエンド

ComfyUIを立てずに、同じGenerationSpecからdiffusersで直接生成するためのバックエンド。
`IMAGEGEN_BACKEND=diffusers` で切り替える (既定はComfyUI)。

CLI / MCP Server / GenerationSpecの書き方は共通で、切り替えても変わらない。
Domain / Service層はどちらのバックエンドで動いているかを知らない。

## 何のためにあるか

- ComfyUIのプロセス管理・Workflowテンプレートの用意なしに動かしたい場合
- CIやサーバー上で、GUIを持たない環境から生成したい場合
- Backend抽象が本当に成立しているかを、2つ目の実装で検証するため
  ([Issue #31](https://github.com/Sylphy0052/agentic-imagegen/issues/31))

常用するバックエンドはComfyUIのまま。diffusers側は機能の対応範囲が狭い (後述)。

## セットアップ

torch / diffusers は既定ではインストールしない。extraで明示的に入れる。

```bash
uv sync --extra diffusers
```

モデルはComfyUIのディレクトリ構成をそのまま読む。`IMAGEGEN_MODELS_ROOT` に
その親ディレクトリを指定する。

```bash
export IMAGEGEN_BACKEND=diffusers
export IMAGEGEN_MODELS_ROOT=~/ComfyUI/models
```

| Specのフィールド | 探す場所 |
| --- | --- |
| `model.checkpoint` | `<IMAGEGEN_MODELS_ROOT>/checkpoints/` |
| `model.loras[].name` | `<IMAGEGEN_MODELS_ROOT>/loras/` |

`IMAGEGEN_MODELS_ROOT` の外を指すパス (`../` を含む名前) は拒否する。

到達確認:

```bash
uv run imagegen health
# diffusers: reachable
# URL: in-process
# Devices: xpu (torch 2.13.0+xpu)
```

`Devices:` が `xpu` ならIntel GPU、`cuda` ならNVIDIA GPU、`cpu` ならCPU推論。
選択はこの順の自動判定で、指定する手段は用意していない。

## 初回はネットワークが要る

単一ファイルのcheckpoint (`.safetensors`) には、tokenizerやscheduler設定などの
モデル本体以外の構成が入っていない。diffusersはそれをHugging Face Hubから取得する
(SD1.5系なら `stable-diffusion-v1-5/stable-diffusion-v1-5` の設定ファイル群)。
取得後は `~/.cache/huggingface/` に残るため、2回目以降はオフラインで動く。
重み自体はダウンロードせず、ローカルのcheckpointから読む。

## 対応している機能

| 機能 | 対応 |
| --- | --- |
| txt2img | あり |
| img2img (`source`) | あり |
| LoRA (`model.loras`) | UNet側だけを持つものに限る ([後述](#loraはunet側だけ)) |
| SD1.5系 / SDXL系のcheckpoint | あり (safetensorsのキーから自動判別) |
| clip skip (`model.clip_skip`) | あり |
| sampler / scheduler | あり ([対応表](#samplerscheduler-の対応)) |
| batch_size | あり |
| テキスト合成 (`text`) | あり (生成後の処理のためバックエンド非依存) |

## 対応していない機能

指定すると生成前に拒否し、ComfyUIで実行するよう促す。

| 機能 | 理由 |
| --- | --- |
| ControlNet (`control`) | 未実装 |
| IPAdapter (`reference`) | 未実装 |
| hires fix (`generation.upscale`) | 未実装 |
| 外部VAE (`model.vae`) | 未実装 |
| DiT系モデル (`model.unet` / `clip`) | 未実装 |
| Textual Inversion (`embedding:` 記法) | 未実装 |
| text encoder側を持つLoRA | diffusers 0.39が読めない ([後述](#loraはunet側だけ)) |

## LoRAはUNet側だけ

kohya形式のLoRAは、UNet側 (`lora_unet_*`) とtext encoder側 (`lora_te_*`) の
重みを持つ。diffusers 0.39はtext encoder側を変換しきれず、読み込みの途中で
`IndexError` になる (UNet側だけなら読める)。

UNet側だけを当てて続けることもできるが、それでは指定したLoRAとは違うものが
当たった状態になるため、**text encoder側を持つLoRAは生成前に拒否する**。
配布されているLoRAの多くは両方を持つため、実質的にはComfyUIバックエンドで
実行することになる。

強度は `strength_model` だけを見る。diffusersのadapter weightは1つしか無いため、
`strength_clip` を別の値にしても無視する (警告を出す)。

## sampler/scheduler の対応

Specの `generation.sampler` はdiffusersのSchedulerクラスへ、
`generation.scheduler` はsigmaの取り方へ対応させている。

| Specのsampler | diffusersのScheduler |
| --- | --- |
| `euler` | `EulerDiscreteScheduler` |
| `euler_ancestral` | `EulerAncestralDiscreteScheduler` |
| `heun` | `HeunDiscreteScheduler` |
| `dpm_2` | `KDPM2DiscreteScheduler` |
| `dpm_2_ancestral` | `KDPM2AncestralDiscreteScheduler` |
| `lms` | `LMSDiscreteScheduler` |
| `dpmpp_2s_ancestral` | `DPMSolverSinglestepScheduler` |
| `dpmpp_sde` | `DPMSolverSDEScheduler` |
| `dpmpp_2m` | `DPMSolverMultistepScheduler` |
| `dpmpp_2m_sde` | `DPMSolverMultistepScheduler` (`algorithm_type=sde-dpmsolver++`) |
| `dpmpp_3m_sde` | 同上 + `solver_order=3` |
| `deis` | `DEISMultistepScheduler` |
| `uni_pc` | `UniPCMultistepScheduler` |
| `uni_pc_bh2` | `UniPCMultistepScheduler` (`solver_type=bh2`) |
| `ddim` | `DDIMScheduler` |
| `ddpm` | `DDPMScheduler` |
| `lcm` | `LCMScheduler` |

`scheduler` が `karras` / `exponential` / `beta` の場合、対応するsigmaオプションを渡す。
ただしオプションを受け付けないSchedulerクラスがあり
(`EulerAncestralDiscreteScheduler` / `LMSDiscreteScheduler` / `DPMSolverSDEScheduler` /
`DDIMScheduler` / `DDPMScheduler` / `LCMScheduler`)、その組み合わせは生成前に拒否する。
黙って `normal` へ倒すと、書いた指定が効かないまま絵だけが変わる。
`normal` を指定するか、sampler側を変える。

上の表に無いsamplerと、`sgm_uniform` / `simple` / `ddim_uniform` / `linear_quadratic` /
`kl_optimal` のschedulerも同じく生成前に拒否する。

## ComfyUIとの違い

同じSpecでも同じ絵にはならない。以下が主な原因。

- 乱数の消費のしかたが違う (ComfyUIはノイズをCPUで生成する場合がある)
- プロンプトの重み付け (`(word:1.2)`) の解釈が違う。diffusers側は素の
  CLIP tokenizerを使うため、A1111 / ComfyUI式の重み記法は効かない
- 77トークンを超えるプロンプトの扱いが違う (diffusers側は切り詰める)
- sigmaの刻み方が一部のSchedulerで一致しない

seedを固定すればdiffusersバックエンド内での再現性はある。
バックエンドをまたいだ再現性は期待できない。

## 実装の位置

| パス | 役割 |
| --- | --- |
| `src/agentic_imagegen/adapters/diffusers/backend.py` | 生成本体。パイプラインの読み込みとキャッシュ |
| `src/agentic_imagegen/adapters/diffusers/schedulers.py` | sampler / scheduler の対応表 |
| `src/agentic_imagegen/adapters/diffusers/models.py` | checkpointのパス解決とSDXL判定 |
| `src/agentic_imagegen/adapters/diffusers/catalog.py` | ローカルにあるモデルの列挙 (MCPの `list_*`) |
| `src/agentic_imagegen/backends.py` | `IMAGEGEN_BACKEND` から具象を選ぶ |

torchとdiffusersのimportは関数内に置いてある。extraを入れていない環境でも
`import agentic_imagegen` が失敗しないようにするため。

## テスト

```bash
IMAGEGEN_MODELS_ROOT=~/ComfyUI/models uv run pytest -m integration tests/integration/test_diffusers.py
```

extraが未インストールの場合、`IMAGEGEN_MODELS_ROOT` が未設定の場合、
使うcheckpointが無い場合はいずれもskipする。
