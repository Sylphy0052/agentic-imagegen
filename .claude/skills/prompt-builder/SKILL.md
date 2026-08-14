---
name: prompt-builder
description: "自然言語の要求や既存のプロンプトからDanbooruタグ主体のプロンプトを組み立て、タグの実在・ブロック順・トークン数を検証してpresetの3軸へ振り分ける。生成は行わない。Use when: 「プロンプトを作って」「Danbooruタグで書き直して」「このプロンプトを最適化して」「A1111の設定をpreset化して」「タグが効いているか確認して」、/prompt-builder。"
allowed-tools: Read, Write, Bash, Glob, Grep
argument-hint: "[組み立てたい要求、または既存のプロンプト]"
---

# プロンプト組み立てスキル

プロンプト文字列そのものを組み立て、検証する。画像は生成しない。

生成まで求められている場合は [imagegen skill](../imagegen/SKILL.md) が本体で、
このskillはその中でプロンプトを決める部分だけを担う。

**記法・値域・判定基準の実体は
[docs/prompting-guide.md](../../../docs/prompting-guide.md) を一次情報とする。**
このSKILLは適用の手順を扱う。同じ内容をここへ転記しない。

## 手順

### 1. モデル系統を確定する

記法・トークン上限・重み付けの効き方がすべて系統で変わる。先に決める。

| 系統 | 判断材料 | 参照 |
| --- | --- | --- |
| SD1.5系 | `model.checkpoint` がSD1.5のマージ (`meinamix` / `counterfeit` など) | [SD1.5系](../../../docs/prompting-guide.md#sd15系-meinamix_v12finalなど) |
| SDXL / Illustrious系 | `model.checkpoint` がSDXL系 (`novaAnimeXL` / `AnythingXL` など) | [SDXL / Illustrious系](../../../docs/prompting-guide.md#sdxl--illustrious系-novaanimexl_ilv190など) |
| Anima系 (DiT) | `model.unet` / `clip` / `vae` の3点指定 | [Anima系](../../../docs/prompting-guide.md#anima系-hassakuanima_v13などdit--qwen3-06b) |

Specが手元にあればそこから読む。無く、要求からも決まらない場合は聞く。
系統を決めずに組み立てると、品質タグの語彙とサンプラー設定を丸ごと作り直すことになる。

### 2. ブロックへ分解する

要求の語をブロックへ割り当てる。ブロックの区切りとpresetの軸への対応は
[タグをブロックで組む](../../../docs/prompting-guide.md#タグをブロックで組む)に従う。

- **「知っていること」ではなく「見えるもの」だけを書く。** 足が写らない構図で靴のタグを書かない
- **1ブロックを厚くしすぎない。** 服装だけ10語あるなら、その絵で見える範囲を疑う

### 3. タグの実在を確認する

同梱スクリプトへまとめて渡す。1タグずつ手で `curl` を組まない。

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, oversized, athletic"
```

```text
1girl	8,276,580	実在する
solo	6,954,502	実在する
oversized	0	存在しない (置換または削除)
athletic	0	存在しない (置換または削除)

4件を確認、うち要対応 2件
```

件数は実行時点の値のため、この例と一致しなくてよい。

3列目が判定で、次の5種類のいずれかになる。

| 表示 | 次の手 |
| --- | --- |
| 実在する | そのまま使う |
| 件数が少なく効きにくい (置換を検討) | より一般的なタグへ寄せる |
| 存在しない (置換または削除) | 置換先を探す。無ければ削除する |
| 品質ラベル / 学習時点にのみ存在 | 残す |
| 確認できなかった | APIへ到達できなかった。下記のとおり利用者へ伝える |

- **確認するのはpositiveだけでよい。** 理由と判定の閾値、`post_count` 0でも残す例外は
  [タグの実在を確認する](../../../docs/prompting-guide.md#タグの実在を確認する)を一次情報とする
- **「確認できなかった」が出たタグは、確認できなかったことを利用者へ伝える。**
  推測で「実在する」と報告しない。スクリプトは到達失敗も要対応として数える

### 4. 置換または削除する

| 状態 | 対応 |
| --- | --- |
| 存在しない | 近いタグへ置換する。見つからなければ削除する |
| 件数が数百以下 | 効きにくい。より一般的なタグへ寄せる |
| 複合表現 | 実在するタグへ分解する (`city street at night` -> `street` + `night`) |
| 表記違い | Danbooruの表記へ合わせる (`high res` -> `highres`、`bookshelves` -> `bookshelf`) |

置換の実績は [references/tag-replacements.md](references/tag-replacements.md) にまとめてある。
同じ語が繰り返し出てくるため、まずここを見る。

**削除した語は利用者へ必ず報告する。** 黙って落とすと、効いていないことに気づけない。

### 5. トークン数を確認する

系統ごとの上限は手順1で開いた [prompting-guide.md](../../../docs/prompting-guide.md) の
表で確認する。超えた分は効きが落ちる。

- **タグ30個前後でSD1.5の上限に達する** (英語のタグは平均2-3トークン)
- 超えている場合は後方のブロック (背景 -> 構図 -> 表情) から削る。
  前方のトークンほど強く効くため、主題は最後まで残す
- Anima系は自然文も混ぜられるため、この制約は緩い

### 6. 競合と重複を洗う

- **相反する指定を残さない。** `covered eyes` (両目) と `hair over one eye` (片目)、
  `simple background` と具体的な場所タグ
- **上位タグを重ねない。** `black hoodie` があれば `hoodie` は要らない
- **positiveとnegativeで打ち消していないか見る。** positiveに `forest`、
  negativeに `trees` を入れると互いに効かなくなる
- **同じ色語が3箇所以上に出ていないか数える。** その色が画面全体へ回り、
  別のアイテムへ指定した色を押しのける。上位のタグ1つへ集約する

### 6-2. 色と丈を指定している場合の書き方へ直す

服の色や靴下の丈を指定した要求では、書き方だけで命中率が変わる。
手順は [指定した色と丈を出す](../../../docs/prompting-guide.md#指定した色と丈を出す)
を一次情報とする。要点だけ挙げると、色とアイテムを1タグにまとめて重みを付け
(`(navy pleated skirt:1.3)`)、丈はnegative (`thighhighs, over-knee socks`) で押さえる。

色の命中はseedごとに揺れるため、**プロンプトの良し悪しを1枚で判定しない。**
生成まで行う場合はseedを4本振り、何本命中したかで書き方を比べる
(生成の手順は [imagegen skill](../imagegen/SKILL.md))。

### 7. preset化する場合は軸へ振り分ける

手順2のブロックをそのまま3軸へ流す。

```yaml
# presets/characters/<name>.yaml
description: <一行で分かる説明>

prompt:
  positive: >
    <大枠>, <外見>, <服装>, <表情>

  negative: >
    bad anatomy, bad hands, extra digits
```

- **解像度とseedをpresetへ書かない。** 再現性に直結するためSpec側で指定する
- **styleにはcfg / steps / sampler / schedulerと品質タグを置く。** モデル系統ごとに分ける
- **sceneに人物の外見を混ぜない。** 使い回せなくなる
- 連結順は `character` -> `scene` -> `style`。品質タグは末尾へ回る
  (詳細は [presets](../../../docs/spec-reference.md#presets))
- **preset名は内容が分かるものにする** (`anime-girl-trailrun` / `studio-white-dynamic`)

新しく作ったpresetは、Specから参照して `validate` が通ることを確認する。

```bash
uv run imagegen validate specs/generated/<name>.yaml
```

### 8. 結果を提示する

次の3つを出す。

1. **置換表** — 元のタグ / `post_count` / 置換先。削除した語は理由を添える
2. **最終プロンプト** — positiveとnegativeをブロックの改行で区切って示す
3. **残った判断** — トークン数が上限に近い、系統を推測で決めた、APIへ到達できなかった等

## 判断に迷ったときの原則

- [全モデル共通の原則](../../../docs/prompting-guide.md#全モデル共通の原則)に従う
- **利用者が実運用している設定は尊重する。** 存在しないタグの是正と明白な重複の解消に留め、
  好みの領域 (negativeの語数、画風の語彙) を勝手に変えない。削るべきと考える場合は
  理由を添えて提案する

## 関連

- [docs/prompting-guide.md](../../../docs/prompting-guide.md) — 記法と判定基準の一次情報
- [docs/spec-reference.md](../../../docs/spec-reference.md) — presetとSpecのフィールド仕様
- [imagegen skill](../imagegen/SKILL.md) — Specへ落として生成まで行う手順
