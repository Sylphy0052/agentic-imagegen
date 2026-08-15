# sd15-avoids-score-tags

- skill: `prompt-builder`
- query: hassaku (SD1.5) で高品質な女の子のプロンプトを作って
- 所要: 0.66 USD / 74秒

## 応答

## 参照した文書

- `CLAUDE.md`(プロジェクト)
- `.claude/skills/prompt-builder/SKILL.md`
- `docs/prompting-guide.md` — 「全モデル共通の原則」「SD1.5系」「配置済みのSD1.5系モデル」「タグの実在を確認する」「タグをブロックで組む」
- `presets/styles/sd15-hassaku.yaml`
- `presets/characters/anime-girl-blue.yaml`
- `specs/examples/txt2img_preset_lora.yaml`

## 確定分

- 系統: SD1.5。checkpoint `hassakuSD15_v13.safetensors`
- style preset: **`sd15-hassaku`**(`applies_to` が当該checkpoint。品質タグ `masterpiece, best quality, ultra-detailed, highres` + `dpmpp_2m_sde`/`exponential`/cfg 7.0/steps 30 + `clip_skip: 2` + 外部VAE `vaeKlF8Anime2_klF8Anime2VAE.safetensors` を保持)
- 「高品質」= style preset側の品質タグで充足。Spec本体へ品質タグ重複記載 不要
- トークン: 骨格のみで10前後。SD1.5上限75に対し余裕大

## 実行するはずのコマンド(未実行)

タグ実在確認:

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, looking at viewer"
```

Spec確定後の検証:

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

## 現時点で確定できるSpec(骨格、外見未定)

```yaml
version: "1"
task: txt2img

presets:
  # character / scene は未確定 (下記の質問待ち)
  style: sd15-hassaku

prompt:
  positive: >
    1girl, solo, looking at viewer

generation:
  width: 512
  height: 768
  seed: -1
  batch_size: 1

model:
  # clip_skip と vae は sd15-hassaku preset が持つため書かない
  checkpoint: hassakuSD15_v13.safetensors

output:
  directory: outputs
  prefix: hassaku_girl
```

## 質問(1問)

外見が未指定 → 推測で髪色・服装を仮置きしない。どれにするか。

- **A(推奨)**: 既存の `anime-girl-blue`(青髪・青目・ロング)を character preset に使う — 検証済みpresetで即出せる、hassakuの明るくコントラストの強い塗りとも噛み合う
- **B**: 新規character presetを起こす(髪色/目色/髪型/服装を指定してもらう)
- **C**: character presetを使わず `1girl, solo` のみ(外見はseed任せ)

答えが決まれば、tagcheck結果・置換表・最終プロンプト・scene軸の要否まで詰める。
