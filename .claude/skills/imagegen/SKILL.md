---
name: imagegen
description: "自然言語の画像生成要求をGenerationSpecへ落とし込み、preset選択・validate・generateまで実行して出力パスとseedを返す。ComfyUI未起動時やエラー時は原因ごとの切り分けまで担う。Use when: 「〇〇な画像を生成して」「画像を作って」「イラストを生成」「同じキャラで別の構図」「presetを追加して」、/imagegen。"
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

## 手順

### 1. ComfyUIの状態を確認する

```bash
uv run imagegen health
```

`Devices:` が `xpu:0` ならIntel GPU、`cpu` ならCPU推論。所要時間が大きく変わるため必ず見る。

未起動なら次で起動し、[docs/xpu-setup.md](../../../docs/xpu-setup.md) を案内する。

```bash
cd ~/ComfyUI && ./.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

### 2. 使えるcheckpointとpresetを確かめる

存在しないcheckpointを指定しない。実在するものだけを使う。
どのcheckpointがどういう絵柄で、cfgとstepsの実用域がどこかは
[配置済みのSD1.5系モデル](../../../docs/prompting-guide.md#配置済みのsd15系モデル)にまとめてある。

```bash
ls ~/ComfyUI/models/checkpoints/
ls ~/ComfyUI/models/loras/
ls presets/characters presets/scenes presets/styles
```

DiT系モデル (Animaなど) はUNet単体で配布され、checkpointとは別の場所に置く。

```bash
ls ~/ComfyUI/models/diffusion_models/ ~/ComfyUI/models/text_encoders/ ~/ComfyUI/models/vae/
```

ControlNet / IPAdapterを使う場合は `~/ComfyUI/models/controlnet/`、
`~/ComfyUI/models/ipadapter/`、`~/ComfyUI/models/clip_vision/` も見る。
テキストを合成する場合は `ls fonts/` で実在するフォント名を確認する
(空なら [docs/fonts-setup.md](../../../docs/fonts-setup.md) の手順を案内する)。

### 3. presetを選ぶか、新しく作る

要求に合うpresetがあれば名前で参照する。軸は3つで、1軸につき1つまで。

| 軸 | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定 |

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
(9種のSD1.5系を同一条件で比較した結果。あわせて `model.clip_skip: 2` と
`model.vae: vaeKlF8Anime2_klF8Anime2VAE.safetensors` を書く)。
負荷を下げたいときだけ汎用の `anime-soft` (steps 20) へ落とす。
どのcheckpointにどのpresetを使うかは `ls presets/styles` と
[配置済みのSD1.5系モデル](../../../docs/prompting-guide.md#配置済みのsd15系モデル)の表で確認する。
一覧と連結・優先順位の規則は
[presets](../../../docs/spec-reference.md#presets) を参照。

同じキャラクタで別の構図を求められた場合は、character presetを再利用して
sceneだけ差し替える。プロンプトを一から書き直さない。
顔立ちまで固定したい場合は [references/character-consistency.md](references/character-consistency.md)
の手順で基準画像を作り `reference` に指定する。

### 4. 一言だけの要求なら、Specを作る前に確認する

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
| モデル (checkpoint) | `ls ~/ComfyUI/models/checkpoints/` で実在するものから、要求に近い系統を推奨案にする。傾向は [配置済みのSD1.5系モデル](../../../docs/prompting-guide.md#配置済みのsd15系モデル) を見て書く |
| 解像度・アスペクト | 縦長 512x768 / 正方形 512x512 / 横長 768x512 のように、CPU・XPUで現実的な範囲から出す。SDXL系を選んだ場合は1024x1024相当で出す |
| 画風 (style preset) | `ls presets/styles` の実在するものから出す。checkpointが決まっていれば対応する `sd15-*` / `sdxl-*` / `anima-base` を推奨案にする |

要求から明らかな項目は聞かない。3項目のうち2つが要求文で決まっているなら、
残る1つだけを聞く。

### 5. Specを作る

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
品質タグの記法は [docs/prompting-guide.md](../../../docs/prompting-guide.md) を参照する。

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

### 6. validateする

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

`Workflow:` 行で実際に使われるテンプレートを、`Presets:` 行で意図したpresetが並んでいるかを
確認する。`text` を書いた場合は `Text:` 行にレイヤ数とフォント名が出る。
検証を緩めて通すことはしない。

### 7. generateする

```bash
uv run imagegen generate specs/generated/<name>.yaml
```

所要時間はSD1.5 / 512x768 / 20 stepsで **XPU約135秒、CPU約12分**。
`IMAGEGEN_TIMEOUT` はXPUなら300、CPUなら1200を目安にする。条件別の実測値は
[docs/xpu-setup.mdの「所要時間とタイムアウトの目安」](../../../docs/xpu-setup.md#所要時間とタイムアウトの目安)
を参照。長くかかる場合はバックグラウンド実行にして、完了を待ってから報告する。

seedを変えて何枚か出す場合や、複数のSpecを流す場合は `batch` を使う。

```bash
uv run imagegen batch specs/generated/<name>.yaml --seeds 111,222,333
uv run imagegen batch specs/generated/a.yaml specs/generated/b.yaml
```

1件失敗しても残りは続き、最後にサマリが出る。Specの検証は実行前に全件行うため、
不正なSpecが混ざっていたら1件も生成しない。枚数分だけ時間がかかるので、
`steps` と解像度を落としてから使う。

生成済みの画像へ後からテキストだけ入れる場合は `compose` を使う。入力画像は変更しない。

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml
```

### 8. 結果を報告する

exit codeが0であること、出力ファイルが存在することを確認したうえで、
**生成された画像のパスとseed** を伝える。テキストを合成した場合は、
合成後のファイル (`*_text.png`) のパスを伝える。

seedに `-1` を指定した場合は実際に使われた値が `metadata.json` の `resolved_seed` に記録される。
同じ画を再現したい場合は、その値をSpecへ書き戻す。

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
