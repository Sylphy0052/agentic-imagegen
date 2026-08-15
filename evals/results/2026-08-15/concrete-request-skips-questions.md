# concrete-request-skips-questions

- skill: `imagegen`
- query: anime-girl-blue と rooftop-sunset のpresetで、512x768の縦長を1枚出して
- 所要: 0.55 USD / 42秒

## 応答

presetは3軸とも実在確認済み。style未指定 → checkpoint未決 → 既定 `hassakuSD15_v13.safetensors` + `sd15-hassaku` を適用。要求文でcharacter/scene/解像度が確定済 → ヒアリング不発動。

## 参照文書

- `.claude/skills/imagegen/SKILL.md` (手順1-7)
- `CLAUDE.md` (既定checkpoint、禁止事項、所要時間)
- `presets/characters/anime-girl-blue.yaml`
- `presets/scenes/rooftop-sunset.yaml`
- `presets/styles/sd15-hassaku.yaml`

## preset

- character: `anime-girl-blue` (青髪青目)
- scene: `rooftop-sunset` (夕暮れ屋上、上半身)
- style: `sd15-hassaku` (既定。`applies_to: hassakuSD15_v13.safetensors`。sampler `dpmpp_2m_sde` / scheduler `exponential` / cfg 7.0 / steps 30 / clip_skip 2 / 外部VAE `vaeKlF8Anime2` を保持 → Spec側へは書かない)

## Spec

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: sd15-hassaku

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: blue_hair_rooftop
```

保存先想定: `specs/generated/blue-hair-rooftop.yaml` (今回未保存)。

## 実行するはずのコマンド

```bash
uv run imagegen catalog
```

```bash
uv run imagegen validate specs/generated/blue-hair-rooftop.yaml
```

```bash
scripts/comfyui-session.sh generate specs/generated/blue-hair-rooftop.yaml
```

## 判断メモ

- 生成は`scripts/comfyui-session.sh`経由固定。`uv run imagegen generate`直叩き禁止(XPU断片化)
- steps 30 + SDE系 → styleコメント通りsteps下げない。所要時間はXPUで135秒(20 steps基準)より伸びる → `validate`の`Estimate:`行を見てから流す
- 1枚 → batch不要、`--seeds`不要
- 報告時は出力パスと`metadata.json`の`resolved_seed`を返す
