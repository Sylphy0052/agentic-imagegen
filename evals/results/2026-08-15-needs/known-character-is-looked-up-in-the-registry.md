# known-character-is-looked-up-in-the-registry

- skill: `imagegen`
- query: aoiで図書館の絵を作って
- 所要: 0.65 USD / 87秒
- 許可したコマンド: Bash(python3 evals/bin/imagegen_ro.py:*)

## 応答

aoiは台帳にいましたが、基準画像が欠落しています。判断の材料と、そこまでの成果物を出します。

## 実行したコマンド (許可済み)

```bash
python3 evals/bin/imagegen_ro.py character list
python3 evals/bin/imagegen_ro.py catalog
python3 evals/bin/imagegen_ro.py character show aoi
```

## 台帳から引いた内容

`character show aoi` の結果:

| 項目 | 値 |
| --- | --- |
| preset | `anime-girl-blue` |
| style | `sd15-hassaku` |
| checkpoint | `hassakuSD15_v13.safetensors` |
| reference | `inputs/aoi.png` |
| seed | 545078971 |

```
warning: 台帳が指す参照先が見つかりません: inputs/aoi.png
```

**基準画像が無いため `reference` (IPAdapter) は書けません。** 書いても生成時に入力画像を解決できず失敗します。したがって顔立ちまでは固定できず、引き継げるのは character preset に書かれた言葉の特徴 (青髪・青目・ロング) と、checkpoint / style による画風までです。

なお `catalog` の `Backend:` は `unavailable (filesystem fallback)` でした。IPAdapterを実際に使う段になったら `scripts/comfyui-session.sh catalog` で在庫を見直す必要があります。

## Spec (referenceなしの案)

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  scene: library-daylight
  style: sd15-hassaku

model:
  checkpoint: hassakuSD15_v13.safetensors

generation:
  width: 512
  height: 768
  seed: -1

output:
  prefix: aoi_library
```

保存先は `specs/generated/aoi-library.yaml` を想定しています (今回は保存していません)。

解像度は `library-daylight` が上半身・立ち構図のため縦長512x768にしました。checkpoint・style・画風は台帳で確定しているので、そこは確認していません。

## 実行するはずのコマンド (未実行)

```bash
uv run imagegen validate specs/generated/aoi-library.yaml
scripts/comfyui-session.sh generate specs/generated/aoi-library.yaml
```

## 参照した文書とpreset

- 文書: `.claude/skills/imagegen/SKILL.md`、`.claude/skills/imagegen/references/character-consistency.md`
- preset: `presets/characters/anime-girl-blue.yaml`、`presets/scenes/library-daylight.yaml`、`presets/styles/sd15-hassaku.yaml` (名前のみ参照、中身は未読)
- 台帳: `registry/characters/aoi.yaml` (`character show aoi` 経由)

## 確認したいこと

顔立ちの引き継ぎをどう扱いますか。基準画像が無いため、ここで結果と手数が変わります。

- **A. 基準画像なしで、上記Specのまま図書館の絵を1枚出す** — 推奨。1回の生成で済み、髪色・目の色・画風は台帳どおりに揃います (顔立ちは過去のaoiと一致しません)
- **B. 先に基準画像を1枚作り直して台帳へ登録し、IPAdapterで図書館を出す** — 生成2回。顔立ちまで揃いますが、そもそもの「過去のaoiの顔」は復元できず、新しく決め直すことになります
- **C. 台帳のseed 545078971 を固定して1枚出す** — sceneが変わるぶんプロンプトが変わるため、seedを揃えても同じ顔になる保証はありません
