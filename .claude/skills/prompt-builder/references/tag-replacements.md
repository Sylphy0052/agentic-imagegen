# タグ置換の実績

`post_count` はDanbooru tags APIでの実測値 (2026-08時点)。
判定基準は [docs/prompting-guide.md](../../../../docs/prompting-guide.md#タグの実在を確認する) を一次情報とする。

同じ語が繰り返し登場するため、確認する前にまずこの表を見る。
ここに無い語は [scripts/tagcheck.py](../scripts/tagcheck.py) で確認し、判明したらここへ足す。

## 人物・体型

| 使いたい表現 | よくある誤り | post_count | 正しいタグ | post_count |
| --- | --- | --- | --- | --- |
| 成人男性 | `20years` | 23 | `mature male` | 44,694 |
| 引き締まった体 | `athletic` | 0 | `toned` | 59,618 |
| 筋肉質な女性 | - | - | `muscular female` | 37,494 |
| 顔に落ちる影 | `shadow face` | - | `shaded face` | 80,589 |
| 薄い笑み | `slight smile` | 0 | `light smile` | 107,422 |
| 前髪 | - | 0 | `bangs` (学習時点にのみ存在。古い世代のモデルでは残す) | 0 |

## 服装

| 使いたい表現 | よくある誤り | post_count | 正しいタグ | post_count |
| --- | --- | --- | --- | --- |
| ゆったりした服 | `oversized` | 0 | `oversized clothes` | 18,215 |
| パーカー | `hooded` | 0 | `hoodie` / `hood up` | 195,469 / 78,844 |
| ランニング用の短パン | `running shorts` | 0 | `short shorts` | 191,258 |
| レギンス | `trail leggings` | 存在せず | `leggings` | 23,955 |
| スニーカー | `trail running shoes` | - | `sneakers` | 95,293 |
| リュック | `small backpack` | - | `backpack` | 98,803 |
| スポーツウェア | - | - | `sportswear` | 33,212 |

上位タグと重ねない。`black hoodie` 28,917 があれば `hoodie` は要らない。

## 構図・背景

| 使いたい表現 | よくある誤り | post_count | 正しいタグ | post_count |
| --- | --- | --- | --- | --- |
| 夜の街路 | `city street at night` | - | `street` + `night` | 11,445 / 168,193 |
| 濡れた路面の反射 | `rain reflection` | - | `rain` + `puddle` + `reflection` | 47,970 / 7,971 / 55,653 |
| 昼 | `daylight` | 0 | `day` | 450,845 |
| 本棚 | `bookshelves` | - | `bookshelf` | 27,687 |
| 背景なし | `no background` | 0 | `simple background` + `white background` | 2,802,355 / 2,295,874 |
| 動きのあるポーズ | `energetic` | 0 | `dynamic pose` | 3,799 |
| 中央に配置 | `centered` | 0 | 該当タグなし (構図タグで表す) | - |

## 画質・破綻 (negative)

| 使いたい表現 | よくある誤り | post_count | 正しいタグ | post_count |
| --- | --- | --- | --- | --- |
| 指の破綻 | `extra fingers` | 0 | `extra digits` | 525 |
| 手の破綻 | `bad hand` | 0 | `bad hands` | 3,330 |
| 頭身の低い絵柄 | `super deformed` | 0 | `chibi` | 372,093 |
| 低解像度 | `loweres` (typo) | - | `lowres` | 112,068 |
| 解剖の破綻 | `deformed anatomy` / `deformed fingers` | 0 | `bad anatomy` | 17,879 |

negativeはpositiveほど学習タグ語彙に縛られない。
`photorealistic` 1,742 のように件数が少ない語も使ってよい。

## 品質・解像度

| 使いたい表現 | よくある誤り | post_count | 正しいタグ | post_count |
| --- | --- | --- | --- | --- |
| 高精細 | `ultra-detailed` / `detailed` | 0 | `absurdres` | 2,918,852 |
| 高解像度 | `high res` | - | `highres` | 7,987,544 |
| アニメ調の塗り | `anime illustration` / `illustration` | 0 | `anime coloring` | 54,394 |

## post_countで判定しないもの

学習時に付与された品質ラベル。Danbooruタグではないため件数は0だが効く。

- NAI系 (SD1.5 / SDXLのアニメ系マージ): `masterpiece` / `best quality` / `high quality` /
  `normal quality` / `low quality` / `worst quality`
- Animagine XL 4.0: `masterpiece` / `high score` / `great score`
- Anima系: `score_1` から `score_9`、`safe` / `sensitive` / `nsfw` の安全度タグ

系統をまたいで流用しない。Animagineの `high score` はIllustrious系では効かない。

## 対応するタグが無い語

削除する。雰囲気を足す働きもしない。

`melancholic` / `mysterious` / `cheerful` / `energetic` / `centered` / `detailed face` /
`soft lighting` / `harsh lighting` / `fantasy costume` / `nsfw` (SD1.5 / SDXL系)

照明を指定したい場合は実在するタグへ寄せる (`backlighting` 45,101 / `sunlight` 101,449 /
`light particles` 80,097 / `light rays` 32,968)。
ただし照明はシーンの一部のため、style presetではなくscene側へ置く。
