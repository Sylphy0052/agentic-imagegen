# tags-are-verified-before-use

- skill: `prompt-builder`
- query: 和風の巫女服で神社の境内にいる女の子のプロンプトを作って
- 所要: 0.95 USD / 122秒
- 許可したコマンド: Bash(python3 .claude/skills/prompt-builder/scripts/tagcheck.py:*)

## 応答

## 前提 (モデル系統)

checkpointの指定がなかったため、`CLAUDE.md`と`docs/prompting-guide.md`の「既定のcheckpointを決める」に従い **SD1.5系 / `hassakuSD15_v13.safetensors` + style preset `sd15-hassaku`** を前提に組みました。SDXL系やAnima系で使うなら品質タグとサンプラー設定が変わるので言ってください。

## 参照した文書

- `.claude/skills/prompt-builder/SKILL.md` (手順)
- `docs/prompting-guide.md` (SD1.5系の記法・トークン上限・タグの実在確認・ブロック分解・色の書き方・既定のcheckpoint)
- `presets/styles/sd15-hassaku.yaml` (品質タグ・negative・サンプラー設定・clip skip・外部VAEの既存値)
- `presets/characters/anime-girl-blue.yaml` (characterpresetの書式)
- `CLAUDE.md` (軸の責務、解像度とseedはSpec側)

## 実行したコマンドと結果

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, long hair, black hair, hair ribbon, miko, red hakama, white kimono, wide sleeves, hakama skirt, japanese clothes, nontraditional miko, light smile, closed mouth, standing, looking at viewer, cowboy shot, shrine, torii, stone lantern, outdoors, day, tree, dappled sunlight, shrine grounds"
```

25件中24件が実在、要対応1件 (`shrine_grounds`)。主要な件数は `miko` 22,821 / `red_hakama` 19,899 / `white_kimono` 56,991 / `torii` 16,559 / `shrine` 6,251 / `stone_lantern` 2,527 / `dappled_sunlight` 14,963。

## 置換・削除

| 元のタグ | post_count | 対応 |
| --- | ---: | --- |
| `shrine grounds` | 0 (存在しない) | 削除。`shrine` + `torii` + `stone lantern` で境内を表現 |
| `japanese clothes` | 524,084 | 削除。`miko` / `white kimono` / `red hakama` の上位タグで重複 |
| `nontraditional miko` | 35,893 | 削除。アレンジ巫女服のタグで、「和風の巫女服」の要求と競合する |
| `hakama skirt` | 46,778 | 削除。`red hakama` と役割が重なるため服装ブロックを厚くしない |
| `tree` | 230,084 | 削除。`dappled sunlight` が木漏れ日として同じ絵を作る |

`shrine` (6,251) と `stone_lantern` (2,527) は1,000件超のため残しましたが、件数は多くないので効きは弱めです。

## 最終プロンプト

positive (character -> scene -> style の連結順、品質タグは末尾):

```text
1girl, solo, long hair, black hair, hair ribbon, miko, white kimono, red hakama, wide sleeves, light smile, closed mouth,
cowboy shot, standing, looking at viewer, shrine, torii, stone lantern, outdoors, day, dappled sunlight,
masterpiece, best quality, ultra-detailed, highres
```

negative (末尾は `sd15-hassaku` が持つ既存値):

```text
bad anatomy, bad hands, extra digits,
embedding:negativeXL_D, nsfw, super deformed, worst quality, low quality, normal quality, lowres, blurry, jpeg artifacts, watermark, text, logo, signature, speech bubble, 3d, photorealistic, harsh lighting, lens flare, film grain, realistic, monochrome, censored, multiple views
```

色は `white kimono` / `red hakama` / `black hair` と色とアイテムを1タグにまとめ、重みは振っていません (`docs/prompting-guide.md`「指定した色と丈を出す」の、まず素の語で出して外れた箇所だけ重みを足す手順に従う)。足元は `cowboy shot` で写らないため `zouri` / `tabi` は書いていません。

## preset化する場合の振り分け

preset名: character `anime-girl-miko` / scene `shrine-grounds-daylight` / style は既存の `sd15-hassaku` をそのまま使う。

```yaml
# presets/characters/anime-girl-miko.yaml
description: 黒髪ロングの巫女服アニメ調女性キャラクタ

prompt:
  positive: >
    1girl, solo, long hair, black hair, hair ribbon, miko, white kimono,
    red hakama, wide sleeves, light smile, closed mouth

  negative: >
    bad anatomy, bad hands, extra digits
```

```yaml
# presets/scenes/shrine-grounds-daylight.yaml
description: 昼の神社の境内。鳥居と石灯籠、木漏れ日

prompt:
  positive: >
    cowboy shot, standing, looking at viewer, shrine, torii, stone lantern,
    outdoors, day, dappled sunlight
```

Spec (ファイルへは保存していません):

```yaml
task: txt2img

presets:
  character: anime-girl-miko
  scene: shrine-grounds-daylight
  style: sd15-hassaku

model:
  checkpoint: hassakuSD15_v13.safetensors

generation:
  width: 512
  height: 768
  seed: 20260815
```

解像度とseedはpresetへ書かずSpec側に置いています。sampler / scheduler / cfg / steps / clip skip / 外部VAEは `sd15-hassaku` が持つため書いていません。

preset化した場合に実行するコマンド (未実行):

```bash
uv run imagegen validate specs/generated/miko-shrine.yaml
```

## 残った判断

- checkpointは指定がなかったため既定 (`hassakuSD15_v13` + `sd15-hassaku`) を仮置きしています。別のcheckpointなら品質タグとサンプラー設定を選び直します
- タグ24個で、英語タグ平均2-3トークンから概算すると55-65トークン。SD1.5の上限75に対して余裕は小さめです。要素を足すなら背景ブロック (`stone lantern` / `dappled sunlight`) から削ります
- `shrine` / `stone_lantern` は件数が少なく効きが弱い可能性があります。境内が出ない場合は `torii` を前方へ寄せるか重み `(torii:1.2)` を検討します
- 既存の `sd15-hassaku` のnegativeにある `embedding:negativeXL_D` はSDXL向けのembedding名に見えます。世代が違うと効かない可能性がありますが、実運用の設定なので今回は触っていません (実ファイルの有無は未確認)
