---
name: imagegen
description: "自然言語の画像生成要求をGenerationSpecへ落とし込み、preset選択・validate・generateまで実行して出力パスとseedを返す。ComfyUI未起動時やエラー時は原因ごとの切り分けまで担う。Use when: 「〇〇な画像を生成して」「画像を作って」「イラストを生成」「同じキャラで別の構図」「presetを追加して」、/imagegen。"
allowed-tools: Read, Write, Bash, Glob, Grep
argument-hint: "[生成したい画像の説明]"
---

# 画像生成スキル

自然言語の要求を GenerationSpec へ落とし込み、ComfyUI経由で画像を生成する。

Workflowテンプレートは人間がComfyUI GUIで作成したものだけを使う。
ノードや接続を組み立てる設計は採らない。

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

```bash
ls ~/ComfyUI/models/checkpoints/
ls ~/ComfyUI/models/loras/
ls presets/characters presets/scenes presets/styles
```

### 3. presetを選ぶか、新しく作る

要求に合う preset があれば名前で参照する。軸は3つで、1軸につき1つまで。

| 軸 | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定 |

新しく作るときは軸の責務を混ぜない。解像度とseedは再現性に直結するため
presetには書かず、Spec側で指定する。

```yaml
# presets/characters/<name>.yaml
description: 一行でどんなキャラクタか

prompt:
  positive: >
    1girl, solo, ...
  negative: >
    bad anatomy, ...
```

同じキャラクタで別の構図を求められた場合は、character presetを再利用して
scene だけ差し替える。プロンプトを一から書き直さない。

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

LoRAを使う場合は `model.loras` に足す (同時3件まで)。

```yaml
model:
  checkpoint: meinamix_v12Final.safetensors
  loras:
    - name: add_detail.safetensors
      strength_model: 0.8
      strength_clip: 0.8
```

指定するとWorkflowテンプレートが `txt2img_lora` へ自動的に切り替わる。
使えるLoRAは `ls ~/ComfyUI/models/loras/` で確認する。強度は省略時1.0。

### 既存画像を描き直す場合 (img2img)

```yaml
task: img2img

source:
  image: inputs/reference.png   # リポジトリ配下に置く
  denoise: 0.55                 # 0に近いほど入力を保ち、1に近いほど描き直す
```

- 入力画像は生成前にComfyUIへ自動でアップロードされる。手で `~/ComfyUI/input/` へ置かない
- **解像度は入力画像のサイズをそのまま使う。** `width` / `height` を書くと拒否される
- `batch_size` は1のみ。LoRAは併用できる (`img2img_lora` テンプレートへ切り替わる)
- 入力画像は `inputs/` へ置く。既存の生成結果を使う場合はそこからコピーする

パラメータの目安 (CPU推論では時間が跳ね返るため控えめにする):

| 項目 | 推奨 | 上限 |
| --- | --- | --- |
| 解像度 | 512x512 / 512x768 | `IMAGEGEN_MAX_WIDTH` / `IMAGEGEN_MAX_HEIGHT` (既定2048) |
| steps | 20前後 | 100 |
| cfg | 5.0-8.0 | 30 |
| batch_size | 1 | 4 |

### 5. validateする

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

`Presets:` 行に意図したpresetが並んでいるか確認する。検証を緩めて通すことはしない。

### 6. generateする

```bash
uv run imagegen generate specs/generated/<name>.yaml
```

所要時間の目安 (SD1.5 / 512x768 / 20 steps):

| 実行基盤 | 実測 | `IMAGEGEN_TIMEOUT` の目安 |
| --- | --- | --- |
| Intel XPU | 約135秒 (初回はモデルロード込み) | 300 |
| CPU | 約12分 | 1200 |

長くかかる場合はバックグラウンド実行にして、完了を待ってから報告する。

### 7. 結果を報告する

exit codeが0であること、出力ファイルが存在することを確認したうえで、
**生成された画像のパスとseed** を伝える。

seedに `-1` を指定した場合は実際に使われた値が `metadata.json` に記録される。
同じ画を再現したい場合は、その値をSpecへ書き戻す。

## 失敗したとき

exit codeで原因を切り分ける。詳細と対処は
[references/troubleshooting.md](references/troubleshooting.md) を参照。

| code | 意味 |
| --- | --- |
| 2 | Specが不正 |
| 3 | ComfyUIへ到達できない |
| 4 | Workflowテンプレートが不正 |
| 6 | 生成がタイムアウトした |
| 7 | ComfyUI側で実行が失敗した |

## してはいけないこと

- `workflows/*.json` を書き換える。Workflowは人間がGUIで作成しAPI形式で書き出す
- ComfyUI workflowをその場で組み立てる
- ComfyUIに存在しないcheckpointを指定する
- `validate` を飛ばす、または検証を緩めて通す
- 巨大解像度・大量batchを実行する
- ComfyUI APIへCLIを迂回して直接curlする
