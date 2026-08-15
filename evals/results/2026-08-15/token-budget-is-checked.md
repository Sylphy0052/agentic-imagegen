# token-budget-is-checked

- skill: `prompt-builder`
- query: 夕焼けの海辺で犬と散歩する少女、風になびく髪、逆光、細かい砂の質感、遠くにヨット、雲の階調、レンズフレア、映画的な色調で、できるだけ細かく描写したプロンプトにして
- 所要: 1.35 USD / 265秒

## 応答

## 参照文書

- `.claude/skills/imagegen/SKILL.md`
- `.claude/skills/prompt-builder/SKILL.md`
- `docs/prompting-guide.md`
- `docs/spec-reference.md` (presets 連結規則 L157-163)
- `presets/styles/sd15-hassaku.yaml` / `sdxl-illustrious.yaml` / `sd15-counterfeit.yaml` / `anime-detailed.yaml`
- `presets/scenes/rooftop-sunset.yaml` (scene の書式参考)
- `CLAUDE.md`

## 判断の要点: 既定checkpointを外す

要求に「逆光」「レンズフレア」「映画的な色調」あり。既定 `sd15-hassaku` の negative:

```
... photorealistic, harsh lighting, lens flare, film grain, realistic, ...
```

`lens flare` / `harsh lighting` / `film grain` / `realistic` が要求と正面衝突。preset連結は重複トークン除去のみ→positiveに `lens flare` 書いてもnegative側は消えず打ち消し合う (`prompt-builder` 手順6)。`sdxl-illustrious` も同じ語を持つ→同罪。

該当語を持たない既存styleは `sd15-counterfeit` / `anime-detailed` / `sd15-*`(counterfeit系negative)。うち **`sd15-counterfeit`** を採用。

- negativeに `lens flare` / `film grain` / `harsh lighting` なし
- 「背景と色彩の描き込みが厚い」(prompting-guide L63) → 砂・雲・遠景ヨットの要求に合致
- cfg 9.0/steps 25で追従強い
- `applies_to: counterfeitV30_v30.safetensors` → checkpointも合わせる。ズレると `validate` が warning

**採用preset名**: `sd15-counterfeit` (既存) + 新規 `girl-dogwalk-windswept` (character) + 新規 `beach-sunset-backlit` (scene)。
**不採用**: `sd15-hassaku`(既定だがnegative衝突)、`sdxl-illustrious`(同衝突)、`anime-detailed`(汎用でcfg 7.0、counterfeit推奨下限8を割る)。

## 「できるだけ細かく」の制約

SD1.5=75トークン上限。組んだ結果は約48トークン→要求要素は全部載る。ただし「砂の質感」「雲の階調」級の微細指定はSD1.5のtext encoderでは効きが薄い→`generation.upscale` (hires fix x2.0) で描き込みを稼ぐ。

248トークン欲しいならSDXL (`novaAnimeXL_ilV190`) だが `sdxl-illustrious` のnegativeから `lens flare, film grain, realistic` を抜いた派生style preset新規作成が前提。XPU必須、所要時間3-4倍。

## 実行するはずのコマンド (未実行)

在庫・実行基盤確認:

```bash
uv run imagegen catalog
```

タグ実在確認 (positiveのみ):

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, long hair, floating hair, dog, walking, leash, beach, sunset, ocean, sand, footprints, wind, backlighting, sunlight, lens flare, cloud, cloudy sky, sailboat, horizon, scenery, wide shot, from side"
```

検証:

```bash
uv run imagegen validate specs/generated/beach-sunset-dogwalk.yaml
```

生成:

```bash
scripts/comfyui-session.sh generate specs/generated/beach-sunset-dogwalk.yaml
```

**未確認タグ**: `floating hair` / `footprints` / `sailboat` / `cloudy sky` / `backlighting` / `scenery` / `wide shot` / `from side` / `leash` は tagcheck 未実行のため実在未確認。`post_count` 0 なら置換または削除が要る。推測で「実在する」とは報告しない。

なお「映画的な色調」に対応するDanbooruタグは無いと判断し `cinematic` 系の語は入れていない。代わりに `backlighting` + `lens flare` + `sunlight` + `sunset` の光学表現で置換 (prompting-guide L141「対応するタグが無い形容は消す」)。「細かい砂の質感」の"質感"、「雲の階調」の"階調"も同様に語としては落とし、`sand` / `footprints` / `cloud` / `cloudy sky` へ分解した。

## Spec (保存せず本文提示)

`specs/generated/beach-sunset-dogwalk.yaml` 相当:

```yaml
version: "1"
task: txt2img

presets:
  character: girl-dogwalk-windswept
  scene: beach-sunset-backlit
  style: sd15-counterfeit

generation:
  width: 768
  height: 512
  seed: -1
  upscale:
    scale: 2.0
    denoise: 0.55
    steps: 8
    model: RealESRGAN_x4plus_anime_6B.pth

model:
  checkpoint: counterfeitV30_v30.safetensors

output:
  prefix: beach_sunset_dogwalk
```

`sampler` / `scheduler` / `cfg` / `steps` / `clip_skip` / `vae` は `sd15-counterfeit` が持つため Spec 側に書かない。

## 新規preset (保存せず本文提示)

`presets/characters/girl-dogwalk-windswept.yaml`:

```yaml
# 犬を連れて歩く少女。髪が風になびく状態までを人物側の責務とする。
# 場所・時間帯・照明は scene 側へ置く。

description: 犬を連れ、髪を風になびかせて歩く少女

prompt:
  positive: >
    1girl, solo, long hair, floating hair, dog, walking, leash
```

`presets/scenes/beach-sunset-backlit.yaml`:

```yaml
# 夕焼けの海辺。逆光とレンズフレアを含む照明もここへ置く
# (照明はシーンの一部として scene 側で指定する方針に従う)。
# 人物の外見は character 側の責務とする。

description: 夕焼けの海辺、逆光とレンズフレア、遠景にヨット

prompt:
  positive: >
    beach, sunset, ocean, sand, footprints, wind, backlighting, sunlight,
    lens flare, cloud, cloudy sky, sailboat, horizon, scenery, wide shot,
    from side
```

## 残った判断

- **negative衝突の回避はcheckpoint選択で行った**。`sd15-hassaku` / `sdxl-illustrious` のnegativeは実運用設定なので勝手に書き換えない (`prompt-builder`「利用者が実運用している設定は尊重する」)。hassakuで出したい場合は該当4語を抜いた派生style presetを別名で作る提案になる
- **タグ実在は未確認**。上記コマンド実行が前提
- **所要時間がタイムアウトを超える可能性**。768x512/steps 25 + hires x2.0 は既定 `IMAGEGEN_TIMEOUT` 300秒を超える見込み。`validate` の `Estimate:` 行を見てから引き上げる
- **解像度は横長768x512を選択**。海辺・遠景ヨット・雲の階調が主要素のため。人物を大きく出したいなら512x768へ変える判断が残る
- **seedは1本**。色・丈の指定がないため4本振りルールは適用外。逆光とフレアの出方は揺れるので、外れたら `batch --seeds` で振る
