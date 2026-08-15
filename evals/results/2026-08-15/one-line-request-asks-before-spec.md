# one-line-request-asks-before-spec

- skill: `imagegen`
- query: 猫の画像を作って
- 所要: 0.48 USD / 40秒

## 応答

**判断: 生成前に質問。** SKILL.md手順3「一言だけの要求なら、Specを作る前に確認する」に該当 (「猫の画像を作って」= 例示そのもの)。モデル・解像度・画風が要求文から一切決まらない → 推測で埋めない。対話不可 → 質問提示で停止。

## 参照文書

- `.claude/skills/imagegen/SKILL.md` (手順3「一言だけの要求なら、Specを作る前に確認する」/ 手順1 `catalog` / 手順2 preset選択)
- `CLAUDE.md` (「画像生成要求を受けたときの手順」「style presetはcheckpointごとに用意する」)
- `docs/spec-reference.md` (フィールド仕様、未読 — Spec確定時に参照)
- `docs/prompting-guide.md#配置済みのsd15系モデル` (checkpoint選択肢の傾向、未読)

## 実行するはずのコマンド (未実行)

在庫と実行基盤確認 (手順1):

```bash
uv run imagegen catalog
```

回答後、Spec保存 → validate → generate:

```bash
uv run imagegen validate specs/generated/cat.yaml
scripts/comfyui-session.sh generate specs/generated/cat.yaml
```

## 質問 (一問一答、1問目)

**Q1. モデル (checkpoint) どれにする？**

- **A (推奨): `hassakuSD15_v13.safetensors` + style preset `sd15-hassaku`** — CLAUDE.md規定の既定。9種のSD1.5系を同一条件比較した結果。checkpoint未定時の標準
- **B: `anime-soft` (汎用style preset、steps 20)** — 下描き向け。負荷下げて速く出す場合
- **C: SDXL系 (`sdxl-illustrious` 等)** — 高精細だがCPU/XPU推論で所要時間が跳ねる

Q2 (解像度: 正方形512x512 / 縦長512x768 / 横長768x512)、Q3 (画風preset) は Q1回答後に出す。

## 無回答時の既定 (推奨案A)

`catalog` 未実行のためcheckpoint実在は未確認。以下は推奨案での想定Spec (ファイル未保存)。

```yaml
version: "1"
task: txt2img

presets:
  style: sd15-hassaku

prompt:
  positive: >
    1cat, solo, cute cat, sitting, looking at viewer, detailed fur, simple background
  negative: >
    text, watermark, bad anatomy, extra limbs

generation:
  width: 512
  height: 512
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: cat
```

使用preset名: **`sd15-hassaku`** (style軸のみ)。character / scene presetは猫に該当するものが無いため未指定 → `prompt.positive` へ直書き。
