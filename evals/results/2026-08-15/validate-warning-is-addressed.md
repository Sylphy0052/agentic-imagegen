# validate-warning-is-addressed

- skill: `imagegen`
- query: hassakuで青い髪の少女を1024x1024、steps 40で生成して
- 所要: 0.90 USD / 79秒

## 応答

## 判断

**要求の問題: hassakuはSD1.5系。1024x1024直接生成は構図破綻 (多頭・二重胴)。**

根拠:

- `docs/spec-reference.md` の [generation.upscale](docs/spec-reference.md) — 「最初から大きい解像度で生成するより、hires fixの方が構図が破綻しにくい」
- `docs/prompting-guide.md` の「hires fixの値」— 512x768 + `scale: 2.0` がSD1.5系の定番
- 所要時間 (`docs/xpu-setup.md` 「所要時間とタイムアウトの目安」): 1024x1024直接/40 steps → 1.049Mpx × 40 × 9秒 ≈ **378秒**。既定 `IMAGEGEN_TIMEOUT` 300超過 → validate warning。hires fix分割なら 94秒 + 113秒 ≈ **207秒**で収まる

→ **最終1024x1024はhires fixで到達**。steps 40は要求どおり1段目へ。

## 参照した文書

- `.claude/skills/imagegen/SKILL.md` (手順1-7)
- `CLAUDE.md` (既定checkpoint、禁止事項、環境変数)
- `docs/spec-reference.md` (presets / generation / generation.upscale / 組み合わせの可否 / 環境変数による上限)
- `docs/prompting-guide.md` (配置済みのSD1.5系モデル、hires fixの値)
- `docs/xpu-setup.md` (所要時間とタイムアウトの目安)
- `presets/characters/anime-girl-blue.yaml`、`presets/styles/sd15-hassaku.yaml` (実在確認済)

## 使うpreset

- character: **anime-girl-blue** (青髪・青目。`1girl, solo, blue hair, blue eyes, long hair`)
- style: **sd15-hassaku** (`applies_to: hassakuSD15_v13.safetensors`。sampler/scheduler/cfg/clip_skip 2/外部VAE `vaeKlF8Anime2_klF8Anime2VAE.safetensors` を持つ → Spec側は `model.checkpoint` だけ)
- scene: 指定なし → 省略

## Spec (保存先想定: `specs/generated/blue-hair-girl-1024.yaml`)

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  style: sd15-hassaku

generation:
  width: 512
  height: 512
  steps: 40
  seed: -1
  upscale:
    scale: 2.0
    denoise: 0.5
    steps: 12

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: blue_hair_girl_1024
```

- workflowは `txt2img_vae_hires` へ自動切替 (style presetの外部VAE + upscale)
- 上限確認: base 262144px / 拡大後 1048576px → `IMAGEGEN_MAX_PIXELS` 4194304・`IMAGEGEN_MAX_UPSCALED_PIXELS` 16777216 の内側
- cfg 7.0 / `dpmpp_2m_sde` + `exponential` / clip_skip 2 はpreset由来。steps 40はSDE系の下限20を上回る → 問題なし

## 実行するはずのコマンド

```bash
uv run imagegen catalog
```

```bash
uv run imagegen validate specs/generated/blue-hair-girl-1024.yaml
```

```bash
scripts/comfyui-session.sh generate specs/generated/blue-hair-girl-1024.yaml
```

`uv run imagegen generate` 直叩き禁止 (XPUアロケータ断片化 → exit 7)。

## 代替 (要求どおり1024x1024を1段で出す場合)

`generation` を `width: 1024 / height: 1024`、`upscale` 削除。ただし構図破綻リスクとvalidateの `Estimate:` タイムアウト警告 (378秒 > 300) を受け入れる形。CPU推論なら約10倍 → `IMAGEGEN_TIMEOUT` 引き上げ必須。
