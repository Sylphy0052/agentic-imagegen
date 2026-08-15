# color-and-length-are-written-per-garment

- skill: `prompt-builder`
- query: 白いシャツと赤いロングスカートの子のプロンプトを作って
- 所要: 0.90 USD / 105秒

## 応答

skill: **prompt-builder** (生成なし要求→imagegenでなくこちら)。

## 前提

- モデル系統: **SD1.5**。要求に指定なし→CLAUDE.md既定`hassakuSD15_v13.safetensors`+style preset **`sd15-hassaku`**採用。別系統(SDXL/Anima)なら品質タグとサンプラー丸ごと組み直し
- 未指定項目(髪・場所・構図)は勝手に盛らない。丈が見えないと赤ロングスカートが効かない→`full body`のみ追加

## 実行するはずのコマンド (未実行)

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, white shirt, red long skirt, full body, standing, looking at viewer"
```

キャッシュ`.cache/tagcheck.json`の在庫は`1girl` 8,284,673 / `solo` 6,960,789 / `oversized` 0 のみ。**残り5タグ(`white shirt` `red long skirt` `full body` `standing` `looking at viewer`)はpost_count未確認**。推測で「実在する」と報告しない。

## 置換・判断

- `red long skirt` — 色とアイテムを1タグへ結合(`red, long skirt`と分けると赤が画面全体へ回る)
- **重みは付けない** — 先回りで振ると重み同士が競合。素の語で1枚出し、外れた箇所だけ後から`(…:1.2-1.4)`
- 丈はnegativeで固定 → `miniskirt, short skirt`追加
- 色語重複なし(白1・赤1)→侵食対策不要
- トークン約18。SD1.5上限75に余裕

## 最終プロンプト

positive (ブロック順: 大枠 / 服装 / 構図。品質タグは`sd15-hassaku`が末尾へ付ける)

```text
1girl, solo,
white shirt, red long skirt,
full body, standing, looking at viewer
```

negative (`sd15-hassaku`の定型に上乗せする差分のみ)

```text
miniskirt, short skirt
```

## YAML

character preset案 (`presets/characters/white-shirt-red-longskirt.yaml`、未保存)

```yaml
description: 白シャツと赤いロングスカートの少女

prompt:
  positive: >
    1girl, solo,
    white shirt, red long skirt

  negative: >
    miniskirt, short skirt
```

Spec案 (`specs/generated/white-shirt-red-longskirt.yaml`、未保存)

```yaml
version: "1"
task: txt2img

presets:
  character: white-shirt-red-longskirt
  style: sd15-hassaku

prompt:
  positive: >
    full body, standing, looking at viewer

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: white_shirt_red_longskirt
```

`clip_skip`/`vae`/`sampler`/`cfg`/`steps`は`sd15-hassaku`が持つ→Spec側不要。

## 残った判断

- 未確認5タグ。tagcheck実行後、0件なら置換必須
- 系統をSD1.5と**推定**で確定。checkpoint指定あれば組み直し
- 色指定2箇所→生成時は1枚判定不可。`batch --seeds`でseed4本、命中本数で比較
- 既存`sd15-hassaku`のnegativeに`extra fingers`(0)・`super deformed`(0)、positiveに`ultra-detailed`(0)。それぞれ`extra digits`/`chibi`/`absurdres`が実タグ。運用中設定→変更提案に留める

参照: `CLAUDE.md` / `.claude/skills/prompt-builder/SKILL.md` / `.claude/skills/imagegen/SKILL.md` / `docs/prompting-guide.md`(タグの実在を確認する・タグをブロックで組む・指定した色と丈を出す・配置済みのSD1.5系モデル) / `references/tag-replacements.md` / `presets/styles/sd15-hassaku.yaml`
