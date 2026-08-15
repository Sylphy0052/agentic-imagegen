# incompatible-combination-is-refused

- skill: `imagegen`
- query: 参照画像の顔立ちを保ったまま、hires fixで2倍に上げて出して
- 所要: 0.55 USD / 47秒

## 応答

## 参照文書

- `CLAUDE.md`(併用可否・exit code・所要時間)
- `.claude/skills/imagegen/SKILL.md`(手順1-7、「キャラクタの引き継ぎと解像度アップを両方求められた場合は2段に分ける」)
- `.claude/skills/imagegen/references/character-consistency.md`(「解像度を上げる (2段運用)」節、weight_type実測表)
- `docs/spec-reference.md#組み合わせの可否`(未読、参照予定)

## 判断

**IPAdapter (`reference`) + hires fix (`generation.upscale`) は併用不可** → 1回のSpecで出せない。2段運用。

1. 1段目: `reference` で顔立ち固定、512x768、`upscale` 無し
2. 2段目: 1段目出力を`inputs/`へ、`task: img2img` + `upscale.scale: 2.0`、`reference` 無し
   - 2段目 denoise 0.35-0.4 → 顔立ちは1段目の絵から引き継がれる
   - 色修正を2段目でやらない(denoise 0.4で塗り替わらない実測あり)

## 実行するはずのコマンド

```bash
uv run imagegen catalog
uv run imagegen character list
uv run imagegen character show <name>        # 台帳にいる場合
uv run imagegen history --prefix <name> --limit 1   # 台帳に無い場合
uv run imagegen validate specs/generated/keep-face-base.yaml
scripts/comfyui-session.sh generate specs/generated/keep-face-base.yaml
cp outputs/<日付>/<時刻>_keep_face_base/image_0001.png inputs/keep-face-base.png
uv run imagegen validate specs/generated/keep-face-up.yaml
scripts/comfyui-session.sh generate specs/generated/keep-face-up.yaml
```

## Spec (1段目・IPAdapter)

```yaml
version: "1"
task: txt2img

presets:
  character: <台帳の preset>
  scene: <差し替える場面>
  style: sd15-hassaku

model:
  checkpoint: hassakuSD15_v13.safetensors

generation:
  width: 512
  height: 768
  seed: 777101

reference:
  image: inputs/<基準画像>.png
  model: ip-adapter-plus_sd15.safetensors
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
  weight: 0.8
  weight_type: style transfer   # linearだと背景まで基準画像に引きずられる

output:
  prefix: keep_face_base
```

## Spec (2段目・hires fix 2倍)

```yaml
version: "1"
task: img2img

presets:
  style: sd15-hassaku          # 1段目と揃える

model:
  checkpoint: hassakuSD15_v13.safetensors   # 1段目と同一

source:
  image: inputs/keep-face-base.png
  denoise: 0.35

generation:
  seed: 777101                 # 1段目と同じ
  upscale:
    scale: 2.0
    denoise: 0.4
    steps: 12
    method: lanczos

output:
  prefix: keep_face_up
```

`img2img` は `width`/`height` 書かない(入力画像サイズ使用、書くと拒否)。

## preset名

- style: `sd15-hassaku`(checkpoint `hassakuSD15_v13.safetensors` と1対1。未指定だとvalidateが警告)
- character / scene: 参照画像が特定できていないため未確定

## 質問

**参照画像はどれか。** 記憶で当てない規約のため、実体を引いてから1段目を組む。

- **A** キャラクタ台帳にいる → 名前を教えてほしい。`imagegen character show <name>` で preset・style・checkpoint・基準画像・seed が全部出る(推奨: 基準画像とcheckpointが1回で揃い、別人化リスクが最小)
- **B** `inputs/` に既に置いてあるファイル → パスを教えてほしい。checkpoint と style preset は別途確認が要る
- **C** 直近の生成の出力を使う → `imagegen history --limit 5` で候補を出して選んでもらう

あわせて、2段目の**場面を変えるか**(同じ絵を大きくするだけか、scene差し替えか)も後で確認する。
