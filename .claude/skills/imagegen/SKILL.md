---
name: imagegen
description: "自然言語の画像生成要求をGenerationSpecへ落とし込み、preset選択・validate・generateまで実行して出力パスとseedを返す。生成済み画像へのテキスト合成・拡大、過去の生成やキャラクタ台帳の照会も扱う。ComfyUI未起動時やエラー時は原因ごとの切り分けまで担う。Use when: 「〇〇な画像を生成して」「画像を作って」「イラストを生成」「同じキャラで別の構図」「presetを追加して」「どのモデルがいいか比べて」「設定を振って比較して」「この画像に文字を入れて」「この画像を大きくして」「さっきの子の設定を教えて」「どのモデルが入ってる？」、/imagegen。"
allowed-tools: Read, Write, Bash, Glob, Grep, AskUserQuestion
argument-hint: "[生成したい画像の説明]"
---

# 画像生成スキル

自然言語の要求をGenerationSpecへ落とし込み、ComfyUI経由で画像を生成する。

Workflowテンプレートは人間がComfyUI GUIで作成したものだけを使う。
ノードや接続を組み立てる設計は採らない。

**Specのフィールド仕様 (値域・既定値・組み合わせ規則) は
[docs/spec-reference.md](../../../docs/spec-reference.md) を一次情報とする。**
このSKILLは要求を受けてから結果を返すまでの手順を扱う。

## 要求の種類から入口を決める

全部が生成の依頼ではない。生成しない要求で手順1から読み始めない。

| 要求 | 行き先 |
| --- | --- |
| 「〇〇な画像を生成して」「イラストを作って」 | [手順](#手順) の1から順に |
| 「さっきの子で別の場面を」 | [手順](#手順) の1 (台帳・記録から引く) |
| 「この画像に文字を入れて」「キャプションを付けて」 | [生成済み画像へテキストだけ入れる](#生成済み画像へテキストだけ入れる) |
| 「この画像を大きくして」「高解像度にして」 | [生成済み画像を大きくする](#生成済み画像を大きくする) |
| 「前回どう作ったっけ」「あの子の設定は」 | [過去の生成とキャラクタを照会する](#過去の生成とキャラクタを照会する) |
| 「どのモデルが入ってる」「LoRAの一覧」 | [手順](#手順) の1 (`catalog`) |
| 「どれがいいか比べて」「設定を振って」 | [references/ablation.md](references/ablation.md) |

## 手順

### 1. 実行基盤と在庫を確かめる

存在しないcheckpointを指定しない。実在するものだけを使う。実行基盤と、
checkpoint / LoRA / ControlNet / IPAdapter / CLIP Vision / DiT系の3点 / VAE /
アップスケールモデル / embedding / preset / フォントを一度に出す。

```bash
uv run imagegen catalog
```

- `Devices:` が `xpu:0` ならIntel GPU、`cpu` ならCPU推論。所要時間が一桁変わる。
  ComfyUIが起動しているときだけ出る
- `Backend:` が取得元。`api` はComfyUIが実際に読み込める名前、
  `filesystem` は未起動時のディレクトリ直読み。後者はカスタムノード由来の
  種別 (IPAdapter) が実際に使えるかまでは分からないため、それらを使う要求では
  `scripts/comfyui-session.sh catalog` で見直す
- checkpointの既定 (`hassakuSD15_v13.safetensors`) には対応するstyle presetが付く
- 種別が `(なし)` なら未配置。ControlNet / IPAdapterはカスタムノードごと未導入の疑いがある
- フォントが `(なし)` でテキスト合成を求められている場合は
  [docs/fonts-setup.md](../../../docs/fonts-setup.md) の手順を案内する

どのcheckpointがどういう絵柄で、cfgとstepsの実用域がどこかは
[配置済みのSD1.5系モデル](../prompt-builder/references/models/sd15.md#配置済みのsd15系モデル)にまとめてある。

生成は `scripts/comfyui-session.sh` 経由で行うため、事前の手動起動は要らない
(スクリプトが起動し、生成し、停止する)。手で立ち上げてデバッグしたいときだけ次を使う
(起動しない場合は環境に応じて [docs/cuda-setup.md](../../../docs/cuda-setup.md) /
[docs/xpu-setup.md](../../../docs/xpu-setup.md) /
[docs/comfyui-setup.md](../../../docs/comfyui-setup.md) を案内する)。

```bash
cd ~/ComfyUI && ./.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

手で起動したComfyUIが動いている間は `scripts/comfyui-session.sh` はそれを使い、
停止もしない。

**「さっきの子」「前回の設定で」と言われたら記録から引く。** 推測で当てない。

台帳に載っているキャラなら名前で引ける。preset・style・checkpoint・基準画像・seedが
そのまま出るため、Specへ書き写すだけでよい。

```bash
uv run imagegen character list
uv run imagegen character show yui
```

台帳に無い場合は生成の記録から引く。

```bash
uv run imagegen history --limit 5
uv run imagegen history --prefix yui
```

出力パス・実際に使われたseed・checkpoint・presetがそのまま出る。
基準画像はここに出たパスを `inputs/` へ置いて使う。
以降も同じキャラを出しそうなら、そのときに台帳へ登録する
(書き方は [character-consistency.md](references/character-consistency.md))。

### 2. presetを選ぶか、新しく作る

要求に合うpresetがあれば名前で参照する。軸は3つで、1軸につき1つまで。

| 軸 | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定・clip skip・外部VAE |

新しく作るときは軸の責務を混ぜない。解像度とseedは再現性に直結するため
presetには書かず、Spec側で指定する。

プロンプトの組み立て (ブロックの分け方、Danbooruタグの実在確認、トークン数の詰め方) は
[prompt-builder skill](../prompt-builder/SKILL.md) の手順に従う。

```yaml
# presets/characters/<name>.yaml
description: 一行でどんなキャラクタか

prompt:
  positive: >
    1girl, solo, ...
  negative: >
    bad anatomy, ...
```

style presetはcheckpointごとに選ぶ。品質タグとサンプラー設定は流用しない。
SD1.5系は `sd15-<通称>` がcheckpointと1対1で対応し、そのモデルの推奨
sampler / scheduler / cfg / stepsを持つ。SDXL系はfine-tuneの系統で選ぶ
(Illustrious系とAnythingXLは `sdxl-illustrious`、Animagine XL系は `sdxl-animagine`、
ShiratakiMix XL系は `sdxl-shiratakimix`)。Anima系は `anima-base`。
checkpointを決めていない段階は `hassakuSD15_v13.safetensors` + `sd15-hassaku` を既定にする
(9種のSD1.5系を同一条件で比較した結果)。clip skipと外部VAEはstyle presetが持つため、
Spec側に書くのは `model.checkpoint` だけでよい。
負荷を下げたいときだけ汎用の `anime-soft` (steps 20) へ落とす。
どのcheckpointにどのpresetを使うかは `catalog` の `presets/style` と
[配置済みのSD1.5系モデル](../prompt-builder/references/models/sd15.md#配置済みのsd15系モデル)の表で確認する。
一覧と連結・優先順位の規則は
[presets](../../../docs/spec-reference.md#presets) を参照。

同じキャラクタで別の構図を求められた場合は、character presetを再利用して
sceneだけ差し替える。プロンプトを一から書き直さない。
顔立ちまで固定したい場合は [references/character-consistency.md](references/character-consistency.md)
の手順で基準画像を作り `reference` に指定する。

### 3. 一言だけの要求なら、Specを作る前に確認する

「猫の画像を作って」のように要求が一言で、モデル・解像度・画風が読み取れない場合は、
推測で埋めずに **AskUserQuestion** で聞く。解像度やcheckpointは出来上がりと所要時間を
決めてしまうため、後から直すと生成をやり直すことになる。

- **発動する場面** — 要求が一言レベルで、下の項目が要求文から決まらないとき
- **発動しない場面** — 要求に画風・被写体・用途が具体的に書かれているとき、
  または既存のSpecやpresetを名指しで再利用するとき。この場合は聞かずに進める
- **聞き方** — 一問一答にする。複数の項目をまとめて1回で聞かない。
  選択肢は推奨案を先頭に置き、代替案を2つ以上並べる
- **答えが返らない場合** — 推奨案として提示した値で進め、
  何を既定にしたかを結果の報告に添える

聞く項目は次の3つに絞る。seed / steps / cfg / samplerはstyle presetと既定値で埋まるため聞かない。

| 項目 | 選択肢の作り方 |
| --- | --- |
| モデル (checkpoint) | `catalog` の `checkpoints` にあるものから、要求に近い系統を推奨案にする。傾向は [配置済みのSD1.5系モデル](../prompt-builder/references/models/sd15.md#配置済みのsd15系モデル) を見て書く |
| 解像度・アスペクト | 縦長 512x768 / 正方形 512x512 / 横長 768x512 のように、CPU・XPUで現実的な範囲から出す。SDXL系を選んだ場合は1024x1024相当で出す |
| 画風 (style preset) | `catalog` の `presets/style` にあるものから出す。checkpointが決まっていれば対応する `sd15-*` / `sdxl-*` / `anima-base` を推奨案にする |

要求から明らかな項目は聞かない。3項目のうち2つが要求文で決まっているなら、
残る1つだけを聞く。

**選択肢に別の軸のものを混ぜない。** checkpointを聞く問いの選択肢はcheckpointだけにする
(`anime-soft` はstyle presetであってcheckpointではない)。軸が混ざると、答えても
何が決まったのか分からなくなる。checkpointが決まればstyle presetは対応するものへ絞れるため、
画風の問いは選択肢が2つ以上残るときだけ出す。

### 4. Specを作る

`specs/generated/<内容が分かる名前>.yaml` へ保存する (git管理外)。

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: anime-soft

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: meinamix_v12Final.safetensors

output:
  prefix: blue_hair_rooftop
```

presetで足りない要素だけ `prompt.positive` に足す。preset側と重複するトークンは
自動で除去されるので、重複を気にして削る必要はない。

**プロンプトの書き方はモデル系統ごとに違う。** SD1.5はdanbooruタグ主体で75トークンまで、
SDXL / Illustrious系は品質タグを先頭に置き `score_9` 系の記法は使わない、Animaはタグと
自然文のどちらでもよく絵師タグに `@` を前置する、といった差がある。語順・重み付けの効き方・
品質タグの記法は [references/models/](../prompt-builder/references/models/) を参照する。

要求に応じて足すブロックは次のとおり。**値域と既定値は書かず、参照先で確かめる。**

| 要求 | 足すブロック | 参照 |
| --- | --- | --- |
| 絵柄や細部を寄せたい | `model.loras` | [model.loras](../../../docs/spec-reference.md#modelloras) |
| 既存画像を描き直したい | `task: img2img` と `source` | [source](../../../docs/spec-reference.md#source-img2img) |
| 参考画像の構図を保ちたい | `control` | [control](../../../docs/spec-reference.md#control-controlnet) |
| 参照画像の顔立ちや服装を保ちたい | `reference` | [reference](../../../docs/spec-reference.md#reference-ipadapter) |
| 解像度を上げたい | `generation.upscale` | [generation.upscale](../../../docs/spec-reference.md#generationupscale-hires-fix) |
| 読める日本語を入れたい | `text.layers` | [text](../../../docs/spec-reference.md#text-テキスト合成) |
| AnimaなどのDiT系モデルを使いたい | `model.unet` / `clip` / `vae` | [DiT系モデル](../../../docs/spec-reference.md#dit系モデル-anima) |

判断に効く点だけをここに残す。

- **img2imgでは `width` / `height` を書かない。** 入力画像のサイズをそのまま使うため拒否される
- **ControlNetでリポジトリ内で行える前処理はCannyのみ。** 線が強く出すぎる場合は
  `low_threshold` を上げるか `strength` を下げる。骨格図・深度図など前処理済みの画像は
  `preprocessor: none` でそのまま渡す (`model` はその制御画像に合ったものを指定する)
- **IPAdapterで背景まで引きずられる場合は `weight_type: style transfer`。**
  weightを下げても背景が変わる前に顔立ちが崩れるだけで、切り分けは `weight_type` で行う
- **読める文字が要求されたらプロンプトへ書かず `text` で合成する。**
  SD1.5 / SDXL系のモデルは日本語をほぼ描けない。ネガティブプロンプトへ `text, watermark` を
  入れておくと、モデルが描く崩れた文字を減らせる
- **併用できない組み合わせがある。** hires fixとIPAdapter、DiT系モデルと
  LoRA / ControlNet / IPAdapterは指定するとその場で拒否される。
  hires fixとControlNetは併用できる (ControlNetが効くのは1段目だけ)。
  DiT系モデルはimg2img / hires fixと併用できる。
  一覧は [組み合わせの可否](../../../docs/spec-reference.md#組み合わせの可否)
- 入力画像・参照画像・control画像は `inputs/` へ置く。生成前にComfyUIへ自動でアップロードされるため、
  `~/ComfyUI/input/` へ手で置かない

### 5. validateする

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

`Workflow:` 行で実際に使われるテンプレートを、`Presets:` 行で意図したpresetが並んでいるかを
確認する。`text` を書いた場合は `Text:` 行にレイヤ数とフォント名が出る。
`Estimate:` 行に所要時間の概算が出る (XPUとCPUの両方。img2imgは出ない)。
検証を緩めて通すことはしない。

`warning:` が出たら生成の前に対処する。exit codeは0でも、そのまま流すと
絵柄が静かに変わるか、タイムアウトで捨てることになる。

| warning | 対処 |
| --- | --- |
| style presetを指定していない | 名指しされたpresetを `presets.style` へ書く |
| style presetが別のcheckpoint向け | checkpointに対応するpresetへ変える |
| 見積りが `IMAGEGEN_TIMEOUT` を超える | steps・解像度・batch_sizeを下げる |

### 6. generateする

```bash
scripts/comfyui-session.sh generate specs/generated/<name>.yaml
```

所要時間はSD1.5 / 512x768 / 20 stepsで **CUDA約4秒、XPU約135秒、CPU約12分**。
`IMAGEGEN_TIMEOUT` はCUDA / XPUなら300、CPUなら1200を目安にする。
`validate` の `Estimate:` は `IMAGEGEN_DEVICE` が未設定だと3基盤を併記し、
タイムアウトの警告はXPUを物差しにする。CUDA環境でその警告を根拠に
stepsや解像度を落とさない。条件別の実測値は
[docs/xpu-setup.mdの「所要時間とタイムアウトの目安」](../../../docs/xpu-setup.md#所要時間とタイムアウトの目安)
を参照。長くかかる場合はバックグラウンド実行にして、完了を待ってから報告する。

seedを変えて何枚か出す場合や、複数のSpecを流す場合は `batch` を使う。

```bash
scripts/comfyui-session.sh batch specs/generated/<name>.yaml --seeds 111,222,333
scripts/comfyui-session.sh batch specs/generated/a.yaml specs/generated/b.yaml
```

1件失敗しても残りは続き、最後にサマリが出る。Specの検証は実行前に全件行うため、
不正なSpecが混ざっていたら1件も生成しない。枚数分だけ時間がかかるので、
`steps` と解像度を落としてから使う。

**キャラクタの引き継ぎと解像度アップを両方求められた場合は2段に分ける。**
IPAdapter (`reference`) とhires fix (`generation.upscale`) は併用できないため、
1段目でIPAdapterを効かせて512x768を出し、その画像を `inputs/` へ置いて2段目の
`img2img` + `upscale` で仕上げる。手順は
[references/character-consistency.md](references/character-consistency.md) を参照。

**服の色や靴下の丈を指定された場合は、1枚で判定せずseedを4本振る。**
色の命中はseedごとに揺れるため、1枚だけ見て「効いた」と判断すると次の生成で崩れる。
書き方の対策と実測値は
[指定した色と丈を出す](../prompt-builder/references/common.md#指定した色と丈を出す) を参照。

**「どれがいいか比べて」と言われた場合は1回につき1軸だけ振る。**
seedとプロンプトを固定し、条件が本当に揃っているかを `metadata.json` で確かめる。
手順と、結論をどこへ書くかは [references/ablation.md](references/ablation.md) を参照。

### 7. 結果を報告する

exit codeが0であること、出力ファイルが存在することを確認したうえで、
**生成された画像のパスとseed** を伝える。テキストを合成した場合は、
合成後のファイル (`*_text.png`) のパスを伝える。

seedに `-1` を指定した場合は実際に使われた値が `metadata.json` の `resolved_seed` に記録される。
同じ画を再現したい場合は、その値をSpecへ書き戻す。

seedを振って比べた場合は、**どのseedを採ったか、指定した要素が何本中何本で命中したか**を
併せて伝える。命中しなかった要素は黙って落とさず、外れた内容 (色が違う、丈が違う) を報告する。

## 生成以外の要求

### 生成済み画像へテキストだけ入れる

「この画像に文字を入れて」と言われた場合は生成し直さない。`compose` は
入力画像を変更せず、テキストを重ねた別ファイル (`*_text.png`) を書き出す。

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml
```

ComfyUIへは接続しないため `scripts/comfyui-session.sh` を経由しなくてよい。
Specは `text.layers` だけを持つものを書く (書き方は
[text](../../../docs/spec-reference.md#text-テキスト合成))。
フォントが未配置なら `catalog` の `fonts` が `(なし)` になる。

### 生成済み画像を大きくする

「この画像を大きくして」と言われた場合はimg2imgで作り直す。生成済み画像だけを
拡大するコマンドは持たない (アップスケールは生成パイプラインの中にある)。

1. 対象の画像を `inputs/` へ置く
2. `task: img2img` と `source.denoise` を低め (0.3-0.45) にして元の絵を保つ
3. `generation.upscale.scale` で倍率を指定する。線を補間したいなら
   `generation.upscale.model` にアップスケールモデルを指定する

```yaml
task: img2img
source:
  image: inputs/base.png
  denoise: 0.35
generation:
  upscale:
    scale: 2.0
    denoise: 0.4
    steps: 12
```

**元の絵を作ったcheckpointとstyle presetを揃える。** 変えると拡大のついでに
絵柄まで変わる。分からない場合は次の照会で引く。
拡大後の総pixel数には上限がある (`IMAGEGEN_MAX_UPSCALED_PIXELS`)。
所要時間は倍以上に伸びるため、`validate` の `Estimate:` を見てから流す。

### 過去の生成とキャラクタを照会する

生成を伴わない照会だけの要求。ComfyUIへは接続しない。

```bash
uv run imagegen character list          # 台帳にいるキャラクタ
uv run imagegen character show yui      # preset・checkpoint・基準画像・seed
uv run imagegen history --limit 5       # 直近の生成
uv run imagegen history --prefix yui    # 出力ディレクトリ名で絞る
```

答えるだけで済む要求に対して、確認なく生成を始めない。

## 失敗したとき

exit codeで原因を切り分ける。code別の対処は
[references/troubleshooting.md](references/troubleshooting.md)、
codeと例外の対応は [CLAUDE.mdの「exit code」](../../../CLAUDE.md#exit-code) を参照。

よく出るもの:

- **2** Specが不正。値域か、併用できない組み合わせを指定している
- **3** ComfyUIへ到達できない。`uv run imagegen health` で確認する
- **5** Workflowの投入が拒否された。ControlNet / IPAdapterのカスタムノードが未導入の疑い
- **6** タイムアウト。`IMAGEGEN_TIMEOUT` が実行基盤に対して短い
- **10** テキスト合成に失敗。フォントが `fonts/` に無い (別の書体へ代替はしない)

## してはいけないこと

- `workflows/*.json` を書き換える。Workflowは人間がGUIで作成しAPI形式で書き出す
- ComfyUI workflowをその場で組み立てる
- ComfyUIに存在しないcheckpointを指定する
- `validate` を飛ばす、または検証を緩めて通す
- 巨大解像度・大量batchを実行する
- ComfyUI APIへCLIを迂回して直接curlする
