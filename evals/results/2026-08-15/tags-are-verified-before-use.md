# tags-are-verified-before-use

- skill: `prompt-builder`
- query: 和風の巫女服で神社の境内にいる女の子のプロンプトを作って
- 所要: 0.83 USD / 117秒

## 応答

読んだ文書: `docs/prompting-guide.md`、`.claude/skills/prompt-builder/SKILL.md`、`references/tag-replacements.md`、`presets/styles/sd15-hassaku.yaml`、`presets/characters/anime-girl-blue.yaml`、`presets/scenes/rooftop-sunset.yaml`、`CLAUDE.md`

## 1. モデル系統

checkpoint未指定 → **SD1.5系**、`hassakuSD15_v13.safetensors` + style preset **`sd15-hassaku`**（`docs/prompting-guide.md#既定のcheckpointを決める` の既定）。系統を決めないと品質タグ・sampler設定が確定しない → 既定が明文化済のため質問せず採用。

sampler/scheduler/cfg/steps/clip_skip/外部VAEは全て`sd15-hassaku`が持つ → Spec側は`model.checkpoint`だけ。

## 2. 実行するはずのコマンド（未実行）

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, miko, red hakama, white kimono, wide sleeves, black hair, long hair, looking at viewer, standing, cowboy shot, shrine, torii, stone lantern, day, sunlight"
```

```bash
uv run imagegen validate specs/generated/miko-shrine.yaml
```

## 3. 最終プロンプト

positive（ブロック順: 大枠 → 外見 → 服装 → 構図 → 背景。品質タグは`sd15-hassaku`が末尾へ連結）

```text
1girl, solo,
black hair, long hair,
miko, red hakama, white kimono, wide sleeves,
looking at viewer, standing, cowboy shot,
shrine, torii, stone lantern, day, sunlight
```

negative: **Spec側に書かない**。`sd15-hassaku`のnegative（`embedding:negativeXL_D`他18語）で足りる。色崩れ・丈の押さえは先回りで並べず、出た症状を見てから足す（`prompting-guide.md#指定した色と丈を出す`）。

タグ16個 + 品質タグ4語 = 20。SD1.5上限75トークンに対し約45トークン、余裕あり。

## 4. Spec（本文のみ、未保存）

```yaml
task: txt2img

presets:
  style: sd15-hassaku

model:
  checkpoint: hassakuSD15_v13.safetensors

prompt:
  positive: >
    1girl, solo,
    black hair, long hair,
    miko, red hakama, white kimono, wide sleeves,
    looking at viewer, standing, cowboy shot,
    shrine, torii, stone lantern, day, sunlight

generation:
  width: 512
  height: 768
```

seedは未指定（初回は振らせ、気に入ったseedを`metadata.json`から拾って固定）。解像度・seedはpresetへ書かない規則どおりSpec側。

## 5. 組み立て時の判断

- `japanese clothes` 不採用 — `miko`が上位を含意、重複
- 色は色+アイテムを1タグへ（`red hakama` / `white kimono`）。重みは先回りで振らない（素の語で1枚 → 外れた箇所だけ重み）
- `simple background` 不採用 — 具体的な場所タグ`shrine`と競合
- 白は`white kimono`の1箇所のみ。色語の分散を回避
- `sunlight` はscene側の照明タグ。styleへ入れない
- 複合表現なし（`shrine at daytime` 等へ分解不要な粒度で書いた）

## 6. 残った判断・未確認

- **全16タグ実在未確認。** tagcheck.py未実行のため`post_count`を1件も取っていない。実在すると推測で報告しない。特に`shrine` / `stone lantern` / `white kimono` / `wide sleeves` / `red hakama` / `cowboy shot` は実行後に置換要否を再判定
- `black hair, long hair` は要求に無い仮置き。SD1.5は髪色未指定だと揺れるため入れた。不要なら外す
- 構図は`cowboy shot`（膝上）。袴の裾まで見せるなら`full body`へ替える。ただし顔の描き込みは落ちる
- テキスト・文字要素なし → `text`系はstyle presetのnegativeで既にカバー
