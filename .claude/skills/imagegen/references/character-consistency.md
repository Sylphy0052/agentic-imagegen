# 同一キャラクタを別の構図で出す

「さっきの子で、今度は図書館の絵を」と言われたときの手順。

presetだけでは顔立ちまでは固定できない。同じcharacter presetを使っても、
seedが違えば別人になる。特徴を引き継ぐには参照画像 (IPAdapter) を使う。

## 何がどこまで固定できるか

| 手段 | 固定できるもの | 変えられるもの |
| --- | --- | --- |
| seedを固定する | 絵そのもの | 何も変えられない (promptを変えると崩れる) |
| character preset | 髪色・目の色などの言葉で書ける特徴 | 構図・画風・細部 |
| IPAdapter (`reference`) | 顔立ち・服装・色調 | 構図・背景 (ただし後述の `weight_type` が要る) |
| キャラクタLoRA | 顔立ち・服装 (学習済みなら最も強い) | 構図・背景 |
| ControlNet (`control`) | 構図・ポーズ | 人物の外見 |

キャラクタLoRAがあるならそれが最も安定する。手元に無い場合は
character preset + IPAdapterで代替する。

## 手順

### 1. 基準になる1枚を作る

character / scene / style presetを指定し、**seedを固定して**生成する。
これが以降の参照画像になるため、seedは `-1` にせず具体的な値を書く。

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: anime-soft

model:
  checkpoint: meinamix_v12Final.safetensors

generation:
  width: 512
  height: 768
  seed: 777001

output:
  prefix: consistency_base
```

顔がはっきり写っている絵を選ぶ。IPAdapterはCLIP Visionで画像全体を読むため、
顔が小さい全身絵や後ろ姿を参照にすると特徴が拾えない。

### 2. 基準画像を `inputs/` へ置く

どの出力が基準の1枚かは `history` で引く。ディレクトリ名を目で探さない。

```bash
uv run imagegen history --prefix consistency_base --limit 1
cp <出力パス> inputs/character.png
```

セッションを跨いだあとも同じ。`--prefix` にキャラの名前を入れて、
直近でそのキャラを出した生成のパスとseedを引く。

### 2.5. 台帳へ登録する

以降も同じキャラを出すなら、ここで `registry/characters/<name>.yaml` へ書く。
`history` は生成の記録であり、どの1枚を基準に選んだかまでは残らない。

```yaml
# registry/characters/aoi.yaml
description: 青い髪の少女
preset: anime-girl-blue
style: sd15-hassaku
checkpoint: hassakuSD15_v13.safetensors
reference: inputs/character.png
seed: 777001
notes: 制服は赤いタイ
```

次からは名前で引ける。基準画像やpresetが消えていれば `warning:` が出る。

```bash
uv run imagegen character show aoi
```

**checkpointとstyle presetも台帳へ書く。** 顔立ちは画風とcheckpointにも依存するため、
基準画像だけを引き継いでcheckpointを変えると別人に見える。

### 3. scene presetだけ差し替えて生成する

`character` presetはそのまま残し、`reference` に基準画像を指定する。

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue    # そのまま残す
  scene: library-daylight       # ここだけ差し替える
  style: anime-soft

model:
  checkpoint: meinamix_v12Final.safetensors

generation:
  width: 512
  height: 768
  seed: 777002                  # 変えてよい

reference:
  image: inputs/character.png
  model: ip-adapter-plus_sd15.safetensors
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
  weight: 0.8
  weight_type: style transfer  # これが無いと背景まで基準画像に引きずられる

output:
  prefix: consistency_library
```

**`weight_type: style transfer` を必ず入れる。** IPAdapterは参照画像を背景ごと
CLIP Visionで読むため、既定の `linear` ではscene presetを差し替えても背景が
基準画像のままになる (実測: 屋上夕景の基準画像から「昼の図書館」を指定しても、
夕焼けの屋上が出る)。`style transfer` は人物の特徴だけを取り、背景と構図を
promptへ委ねる。

**checkpointは基準画像と同じものを使う。** 変えると画風ごと変わり、
IPAdapterをかけても別人に見える。

### 4. 構図を厳密に決めたい場合はControlNetを足す

ポーズや構図の参考画像があるなら `control` を併用する。
構図をControlNet、顔立ちをIPAdapterが担う。

```yaml
control:
  image: inputs/pose.png
  model: control_v11p_sd15_canny_fp16.safetensors
  strength: 0.8

reference:
  image: inputs/character.png
  model: ip-adapter-plus_sd15.safetensors
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
  weight: 0.8
  weight_type: style transfer
```

ControlNetが構図を決めるため、この場合scene presetは使わなくてよい。

構図は参考画像どおりになるが、**背景の色調はIPAdapter側へ引かれる**。
実測では `dark background` とpromptへ書いても、基準画像の夕景の色が出た。
背景色まで指定したい場合は `weight` を下げるか、背景の無い参照画像を使う。

## weightとweight_typeの決め方

同じ基準画像 (屋上夕景) から「昼の図書館」を指定して比べた実測。

| 設定 | 結果 |
| --- | --- |
| `weight: 0.8` (既定の `linear`) | 顔立ち・服装は保たれるが、背景が夕焼けの屋上のまま。sceneが効かない |
| `weight: 0.5` (既定の `linear`) | 背景は半分だけ図書館になる。服装のディテールが崩れる (赤いタイが紺のリボンへ変わった) |
| `weight: 0.8` + `weight_type: style transfer` | 背景は図書館、顔立ち・髪・制服の色は維持。**これを使う** |
| `weight: 0.7` / `0.85` (既定の `linear`) | 背景が基準画像のままなのに加え、構図まで引かれて煽りになり、足先が画面外へ出た |

weightを下げて背景を振り切ろうとすると、先に服装や顔立ちが崩れる。
背景を変えたいときはweightではなく `weight_type` で切り分ける。

`style transfer` でも顔立ちが揺れる場合だけ `weight` を0.9-1.0へ上げる。
1.0を超えるとpromptが効かなくなる。

## 解像度を上げる (2段運用)

**IPAdapterとhires fixは併用できない** ([組み合わせの可否](../../../../docs/spec-reference.md#組み合わせの可否))。
解像度を上げるには2段に分ける。

1. IPAdapterを効かせて512x768を生成する (`upscale` は書かない)
2. その出力を `inputs/` へ置き、`img2img` + `upscale` で仕上げる (`reference` は書かない)

```bash
scripts/comfyui-session.sh generate specs/generated/scene-base.yaml
cp outputs/<日付>/<時刻>_scene_base/image_0001.png inputs/scene-base.png
scripts/comfyui-session.sh generate specs/generated/scene-up.yaml
```

2段目のSpecは1段目とプロンプト・checkpoint・VAE・clip skipを揃え、
`source.denoise` を0.4前後にする。顔立ちと色は1段目の絵から引き継がれるため、
2段目でIPAdapterを使わなくても崩れない。

```yaml
task: img2img

source:
  image: inputs/scene-base.png
  denoise: 0.4

generation:
  seed: <1段目と同じ値>
  upscale:
    model: RealESRGAN_x4plus_anime_6B.pth
    model_scale: 4.0
    scale: 1.5
    denoise: 0.4
    steps: 20
    method: lanczos
```

- `upscale.scale` は1.0より大きい値しか指定できない。拡大を最小にしたい場合は1.5にする
- **色の修正を2段目でやらない。** `denoise` 0.4では既に塗られている色が塗り替わらず、
  2段目のプロンプトへ重みを足しても効かない (実測: 紺になった靴を
  `(white loafers:1.3)` へ上げても紺のままだった)。色は1段目で直す

## 色が指定どおりに出ないとき

IPAdapterを使うと、参照画像の色調とプロンプトの重み括弧が競合する。

- **複数箇所へ同時に重みを振らない。** 4箇所へ重みを付けた版では
  `(white sailor uniform:1.3)` と書いた上衣が紺になった。同じseedで重みを全部外した版は
  上衣が白で出た
- **外れた1箇所だけへ重みを足す。** 靴だけ `(white loafers:1.2)` にした版は、
  上衣・スカーフ・スカート・靴下・靴のすべてが指定どおりになった
- 順序は「素の語で1枚 -> 外れた箇所を見る -> その箇所だけ重みを足して撃ち直す」。
  書き方の実測値は
  [指定した色と丈を出す](../../../../docs/prompting-guide.md#指定した色と丈を出す) を参照

## うまくいかないとき

- **別人になる**: 参照画像の顔が小さすぎないか確認する。上半身の絵を基準にする
- **背景が元絵のままになる**: `weight_type: style transfer` を入れる。
  weightを下げるのは逆効果で、背景が変わる前に服装が崩れる
- **参照画像が効いていない**: `metadata.json` の `workflow` が `*_ipadapter` に
  なっているか確認する。`reference` の書き忘れではテンプレートが切り替わらない
- **画風が変わった**: checkpointかstyle presetが基準画像と違う

## 複数枚まとめて出す

構図違いを何枚か出す場合は、Specを人数分作って `batch` で流す。

```bash
scripts/comfyui-session.sh batch \
  specs/generated/consistency-library.yaml \
  specs/generated/consistency-night.yaml
```

1件失敗しても残りは続く。生成時間は枚数分かかるため、枚数を増やす前に
1枚で `weight` と `weight_type` を決めておく。
