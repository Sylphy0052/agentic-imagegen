# style-preset-carries-clip-skip-and-vae

- skill: `imagegen`
- query: hassakuで女の子の立ち絵を1枚出して
- 所要: 0.49 USD / 43秒

## 応答

## 判断

質問不要。要求で3項目中2つ確定 (checkpoint=hassaku、画風=対応style preset)、残る解像度も「立ち絵」→全身縦長512x768で確定。SKILL.md手順3の「発動しない場面」該当。

## 参照文書

- `CLAUDE.md` — 既定checkpoint、preset軸の責務、生成は`scripts/comfyui-session.sh`経由
- `.claude/skills/imagegen/SKILL.md` — 手順1-7、ヒアリング発動条件
- `docs/spec-reference.md` — Specフィールド値域・既定値
- `docs/prompting-guide.md` — SD1.5系のタグ記法 (danbooruタグ主体、75トークン)

## preset

- style: **`sd15-hassaku`** (checkpoint `hassakuSD15_v13.safetensors` と1対1。品質タグ・sampler/scheduler/cfg/steps・clip skip・外部VAEを持つ → Spec側は`model.checkpoint`のみ)
- character: 使わない (特定キャラ指定なし)
- scene: 使わない (立ち絵=無地背景 → `prompt.positive`へ直書き)

## 実行するはずのコマンド

```bash
# 1. 在庫確認 (hassakuSD15_v13.safetensors と sd15-hassaku の実在、Devices: xpu:0 か cpu か)
uv run imagegen catalog

# 2. 検証 (Workflow行/Presets行/Estimate行、warning無しを確認)
uv run imagegen validate specs/generated/hassaku-standing-girl.yaml

# 3. 生成 (ComfyUI起動→生成→停止)
scripts/comfyui-session.sh generate specs/generated/hassaku-standing-girl.yaml
```

## Spec (保存先想定: `specs/generated/hassaku-standing-girl.yaml`)

```yaml
version: "1"
task: txt2img

presets:
  style: sd15-hassaku

prompt:
  positive: >
    1girl, solo, full body, standing, cowboy shot to full body,
    looking at viewer, simple background, white background
  negative: >
    cropped, out of frame, text, watermark

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: hassaku_standing_girl
```

`steps`/`cfg`/`sampler`/`scheduler`/`clip skip`/外部VAEは`sd15-hassaku`が持つ → Spec側未記載。

## 報告予定

exit code 0 + 出力ファイル実在確認後、画像パスと`metadata.json`の`resolved_seed`を返す。
