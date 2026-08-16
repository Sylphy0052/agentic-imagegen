---
name: anima-prompt
description: "Anima系 (DiT + Qwen3 text encoder) 向けのプロンプトを、タグ行と自然文のハイブリッド構造へ組み立てる。モデルごとに品質タグ・rating・サンプラー設定を切り替え、タグの実在を確認して3軸のpresetへ振り分ける。生成は行わない。Use when: 「Animaでプロンプトを作って」「hassakuAnima / WAI-ANIMA / MiaoMiao / CottonAnimaで描きたい」「SDXL向けのプロンプトをAnima用に直して」「Animaのプロンプトが効かない」、/anima-prompt。"
allowed-tools: Read, Write, Bash, Glob, Grep
argument-hint: "[組み立てたい要求、または既存のプロンプト]"
---

# Anima系プロンプト組み立てスキル

Anima系モデル向けのプロンプト文字列を組み立て、検証する。画像は生成しない。

**Animaは「SDXLの新しいcheckpoint」ではない。** 約20億パラメータの独自アーキテクチャで、
text encoderはCLIPではなくQwen3-0.6B、VAEはQwen-Image VAE。
タグを並べるほど良くなるSD1.5 / SDXLの流儀をそのまま持ち込むと効かない。

| 系統 | 使うskill |
| --- | --- |
| SD1.5 / SDXL / Illustrious系 | [prompt-builder](../prompt-builder/SKILL.md) |
| Anima系 (`model.unet` + `clip` + `vae` の3点指定) | このskill |

生成まで求められている場合は [imagegen skill](../imagegen/SKILL.md) が本体で、
このskillはその中でプロンプトを決める部分だけを担う。

**モデルごとの値域・推奨設定・品質タグの実体は
[references/anima-models.md](references/anima-models.md) を一次情報とする。**
このSKILLは適用の手順を扱う。同じ内容をここへ転記しない。
タグの実在確認の閾値など系統によらない判定基準は
[prompt-builder skillのreferences/common.md](../prompt-builder/references/common.md) にある。

## 貫く原則

> **タグで語彙 (Vocabulary) を指定し、自然文で意味 (Semantics) を指定する。**

タグは「何が写っているか」を決める。自然文は「それらがどう配置され、どう作用しているか」を決める。
どちらか一方だけで組まない。Animaはタグ・自然文・その混在の3形式すべてで学習されている。

## 手順

### 1. モデルを確定する

品質タグの語彙とサンプラー設定がモデルで変わる。先に決める。
配置済みのモデルと個別の推奨設定は
[references/anima-models.md](references/anima-models.md) を見る。

決まっていない場合は `hassakuAnima_v13_int8.safetensors` を既定にする
(int8で2.1GB。他の3つは3.9GBあり、メモリの余裕がない環境では読み込みで詰まる)。

要求に合わなくて振り直すときは **Hassaku -> WAI-ANIMA -> CottonAnima -> MiaoMiao** の順で試す。
同一条件で出した4枚を比べて決めた優先順で、根拠は
[references/anima-models.md](references/anima-models.md#選ぶ順序) にある。

系統によって品質タグの扱いが変わる点だけ、ここで押さえる。

| 系統 | 品質タグ |
| --- | --- |
| Base系 | `masterpiece, best quality, score_7` を付ける |
| Aesthetic系 / score非推奨のfine-tune (CottonAnimaなど) | `score_*` を**positive / negativeの双方から外す** |
| Turbo系 | 品質タグは最小限。cfg 1 / steps 8-12 |

Aesthetic系は学習時にcaptionから品質タグを除いてある。`score_9` を重ねると
かえって過剰な「AI絵らしさ」に寄る。

### 2. 中間表現へ分解する

要求を次の枠へ割り当ててから文字列にする。枠を飛ばして直接書かない。

| 枠 | 入るもの | 行き先 |
| --- | --- | --- |
| quality / year / rating | `masterpiece` `best quality` `score_7` `newest` `safe` | タグ行の先頭 |
| count | `1girl` `1boy` `2girls` | タグ行 |
| character / series | キャラクタ名、作品名 | タグ行 |
| artist | `@artist name` | タグ行 |
| appearance | 髪型・髪色・瞳・体格 | タグ行 |
| outfit | 服・靴・靴下 | タグ行 |
| expression / pose | 表情・姿勢 | タグ行 |
| camera | `upper body` `cowboy shot` `from below` | タグ行 |
| environment / lighting | 場所・時間帯・光 | タグ行 |
| relation | 誰が何にどう触れ、光がどこから当たるか | **自然文** |

- **「知っていること」ではなく「見えるもの」だけを書く。** 足が写らない構図で靴を書かない
- **全属性をタグ化しない。** Animaはtag dropoutで学習されているため、
  関連タグを網羅しても効果が上がらず、むしろ指定が薄まる
- **キャラクタ名だけに頼らない。** 名前と併せて髪・瞳・服の基本を書く。
  複数キャラでは名前だけを並べると属性が混ざる

### 3. タグ行を組む

順序は公式の推奨に従う。ブロック内の並びは自由。

```text
[quality / meta / year / rating] [1girl / 1boy / 1other] [character] [series] [artist] [general tags]
```

- **小文字とスペースで書く。** アンダースコアを使うのは `score_7` のようなscoreタグだけ
- **絵師タグは `@` を前置する** (`@artist name`)。付けないと効きが極端に落ちる
- **DanbooruとGelbooruで名前が違うタグはGelbooru側を採る**
- **重み付けはSDXLより強い値が要る。** `(chibi:1.1)` では動かない。`(chibi:2)` の桁で書く
- rating (`safe` / `sensitive` / `nsfw` / `explicit`) を明示する。
  短く曖昧なプロンプトでは意図しない出力になりやすい
- year (`newest` / `recent` / `year 2025` など) は絵柄の年代を効かせたいときに入れる

### 4. 自然文を書く

タグ行のあとに続ける。**最低2文。** 1文だけでは効きが安定しない。

書くのは関係と作用に限る。タグで済むことを文で繰り返さない。

- 誰がどこに、何に対してどう位置しているか
- 手や視線がどこへ向いているか
- 光源の向きと当たる面
- 前景・後景に何が流れているか

```text
A girl is sitting sideways on a chair next to the classroom window.
She rests one elbow on the windowsill and quietly watches the sunset outside.
Warm orange light illuminates the side of her face.
```

通常の英文として書く (文頭は大文字、文末はピリオド)。

### 5. タグの実在を確認する

タグ行の一般タグをまとめてスクリプトへ渡す。1タグずつ手で `curl` を組まない。

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, aqua hair, bob cut"
```

判定の読み方と閾値は
[prompt-builder skillの手順3](../prompt-builder/SKILL.md) と
[タグの実在を確認する](../prompt-builder/references/common.md#タグの実在を確認する)に従う。
Anima固有の注意は次の3つ。

- **確認するのはタグ行だけでよい。** 自然文とscoreタグ・ratingタグ・yearタグは対象外
- **スクリプトはDanbooruを引く。** Gelbooru側の表記を採ったタグは0件で返ることがある。
  その場合は0件をもって捨てず、Gelbooruでの実在を確かめたうえで残すかどうかを決め、
  **確かめていないなら「未確認」と伝える**
- **引いていない `post_count` を書かない。** スクリプトを走らせていないタグへ件数を添えない

### 6. negativeを決める

**短いbaselineから始める。** SD1.5時代の巨大テンプレートを最初から積まない。
過度な制約は自由度を落とす。

Base系の起点は公式のこれ。

```text
worst quality, low quality, score_1, score_2, score_3,
artist name, blurry, jpeg artifacts, chromatic aberration
```

score非推奨のモデルでは `score_1, score_2, score_3` を外す。
`bad anatomy` / `extra fingers` の類は、**実際に破綻した枚を見てから足す。**

### 7. 生成設定を添える

プロンプトと設定は対で決まる。値の実体は
[references/anima-models.md](references/anima-models.md) と
[docs/spec-reference.md](../../../docs/spec-reference.md#dit系モデル-anima)。

- **cfgをSDXLの感覚で上げない。** 4-5が公式推奨。上げると彩度とエッジが破綻する
- steps 30-50。20台へ落とすと線が甘くなる
- 解像度は512x512-1536x1536。832x1216が扱いやすい
- sampler `er_sde` / scheduler `simple` を既定にする。
  柔らかい線は `euler_ancestral`、多様性は `dpmpp_2m_sde_gpu`
- `beta57` はschedulerではない。指定できない
  ([references/anima-models.md](references/anima-models.md#beta57はschedulerではない))

### 8. presetへ振り分ける

3軸への割り当ては [prompt-builder skillの手順7](../prompt-builder/SKILL.md) と同じ。
Anima固有の制約だけ挙げる。

- **style presetはモデルごとに用意する。** 品質タグとサンプラー設定は学習内容に依存する。
  `applies_to` に対象の `unet` ファイル名を書き、取り違えを `validate` に警告させる
- **style presetに `model.clip_skip` を書かない。** DiT系はclip skipと併用できない
- **VAEをpresetへ書かない。** `model.unet` / `clip` / `vae` は3点セットでSpec側の必須項目
- 自然文はcharacter presetへ入れない。character / scene / styleを連結したあとに続ける形で、
  Spec側の `prompt.positive` の末尾へ置く

新しく作ったpresetはSpecから参照して `validate` が通ることを確認する。

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

### 9. 結果を提示する

1. **タグ行と自然文** — 分けて示す
2. **置換表** — 元のタグ / `post_count` / 置換先。削除した語は理由を添える。未確認のものは未確認と書く
3. **生成設定** — sampler / scheduler / cfg / steps / 解像度と、その根拠にしたモデル
4. **残った判断** — 品質タグの系統をどちらで扱ったか、Gelbooru差異で判断を留保したタグ

## SDXL / Illustriousから移すときに直すもの

| SDXL / Illustrious | Anima |
| --- | --- |
| `long_hair` | `long hair` (スペース区切り) |
| 絵師名をそのまま | `@artist name` |
| cfg 5-8 | cfg 4-5 |
| `(tag:1.1)` | `(tag:2)` の桁 |
| 品質タグを盛る | 最小限。Aesthetic系ではscoreを外す |
| 長大なnegative | 短いbaselineから始める |
| キャラ名だけ | キャラ名 + 外見 |
| 全属性をタグ化 | 重要なタグだけ + 自然文 |

## 関連

- [references/anima-models.md](references/anima-models.md) — モデル別の推奨設定と得手不得手
- [prompt-builder skillのreferences/common.md](../prompt-builder/references/common.md) — 系統によらない原則とタグの実在確認の基準
- [docs/spec-reference.md](../../../docs/spec-reference.md) — presetとSpecのフィールド仕様
- [prompt-builder skill](../prompt-builder/SKILL.md) — SD1.5 / SDXL系のプロンプトとタグ検証スクリプト
- [imagegen skill](../imagegen/SKILL.md) — Specへ落として生成まで行う手順
