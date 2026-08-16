# 全モデル共通の原則

系統によらず当てはまる原則と、プロンプトを組むときの共通手順をまとめる。
系統ごとのトークン上限・値域・記法は [models/](models/) の4本を参照する。

## 並べ方と削り方

- **前方のトークンほど強く効く。** 主題 -> 属性 -> 構図 -> 品質タグ の順に並べる
- **重み付けは `(tag:1.3)`。** SD1.5 / SDXL系は0.7-1.5が実用域。それを超えると色が焼き付き、
  構図が破綻する。効きはSD1.5系が最も強く、SDXL以降は逓減する。
  同じ破綻が出続けるときは0.2-0.3刻みで上げる
- **品質タグは3-4個で打ち止め。** 積み増しても品質は上がらず、主題のトークンが希釈されるだけ
- **プロンプトは削る方が改善する。** 半分に削って壊れた箇所が、実際に効いていたトークン
- **negativeは最小から始め、症状を見てから足す。** SD1.5時代の長大なnegativeを貼ると
  色が抜けて眠い絵になる。washed-out / paleはnegativeが過剰なサインで、半分に削ると戻る
- **negativeにpositiveの要素を書かない。** positiveに `forest`、negativeに `trees` を入れると
  互いに打ち消し合う
- **seedを固定したままpromptとcfgを詰め、最後にseedを振って構図を探す。**
  seedと生成パラメータを同時に動かすと、何が効いたのか判別できない

## Textual Inversion embedding

`easynegative` / `badhandv4` のような定型embeddingを使う場合は、必ず
`embedding:<name>` とプレフィックスを付けて書く (`embedding:easynegative`)。
プレフィックスの無い素の単語はComfyUI側でもembeddingとして解決されず、
ただの単語として扱われる (警告も拒否もされない)。

`embedding:<name>` と書いた場合は、生成前に該当ファイルが
`~/ComfyUI/models/embeddings/` に実在するかを自動で検証する。未配置なら
`imagegen generate` / MCPの `generate_image` が実行前に拒否する
(検証の仕組みと`imagegen validate`との違いは
[spec-reference.md](../../../../docs/spec-reference.md#prompt) を参照)。

## タグの実在を確認する

存在しないタグは学習語彙に対応する概念を持たない。SD1.5は75トークンしか使えないため、
効かないタグを並べるとそれだけ主題のトークンが希釈される。
思いついたタグは書く前にDanbooruで実在を確認する。

```bash
curl -s 'https://danbooru.donmai.us/tags.json?search%5Bname%5D=oversized_clothes&limit=1' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[0]["post_count"] if d else "NOT_FOUND")'
```

複数のタグをまとめて確認する場合は
[.claude/skills/prompt-builder/scripts/tagcheck.py](../scripts/tagcheck.py)
を使う。プロンプトをそのまま渡せる。

```bash
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "1girl, solo, oversized"
```

- **確認はアンダースコア表記で行う。** Danbooruのタグ名は `hair_over_one_eye` の形で登録されている。
  プロンプトへ書くときはスペース区切りでよい (CLIPはどちらも同じに解釈する)
- **`post_count` が0なら存在しないタグと判断する。** 1,000件未満は学習への寄与が小さく、
  より一般的なタグへの置換を検討する (`tagcheck.py` の判定もこの閾値による)
- **置換先が見つからない語は消す。** `mysterious` や `melancholic` のような形容は
  対応するタグが無く、雰囲気を足す働きもしない

`post_count` で判定してはいけない例外が2つある。

- **品質ラベルはDanbooruタグではない。** `masterpiece` / `best quality` / `worst quality` /
  `low quality` はいずれも `post_count` 0だが、anime系checkpointが学習時に
  aesthetic scoreから付与したラベルであり、効く
- **学習時点との差がある。** `bangs` は現在のDanbooruでは整理されて0だが、
  SD1.5系checkpointの学習データ (2022-2023時点) には存在した。世代の古いモデルでは残す

negativeはpositiveほど学習タグ語彙に縛られない。CLIPの一般語彙としても効くため、
`photorealistic` のように件数が少ない語を使ってよい。厳密に確認するのはpositive側。

## タグをブロックで組む

タグを役割ごとにまとめて並べると、書き換える箇所と使い回せる箇所が分かれる。
presetの軸もこの区切りに合わせて切る。

| ブロック | 例 | 置き場所 |
| --- | --- | --- |
| 品質 | `masterpiece, best quality, absurdres, highres` | style |
| 大枠 | `1boy, solo, male focus, mature male` | character |
| 外見 | `black hair, short hair, hair over one eye, bangs` | character |
| 服装 | `black hoodie, hood up, oversized clothes, long sleeves` | character |
| 表情 | `pale skin, shaded face, light smile` | character |
| 構図 | `full body, standing, looking at viewer` | scene |
| 背景 | `simple background, white background` | scene |

複合表現はタグとして成立しない。`city street at night` は `street` と `night` へ、
`rain reflection` は `rain` / `puddle` / `reflection` へ分解する。

相反するタグと重複するタグも洗う。`covered eyes` (両目) と `hair over one eye` (片目) は
競合し、`black hoodie` があれば `hoodie` は要らない。

A1111 / Forgeでは品質タグを先頭へ置く書き方が通例だが、presetは
`character` -> `scene` -> `style` の順に連結するため、品質タグは末尾へ回る。
前方のトークンほど強く効く以上、主題が先に来るこの順の方が意図どおりに効く。
品質タグを前へ出したい場合はstyleではなくSpec本体の`prompt.positive`へ書く
(連結順は [spec-reference.md](../../../../docs/spec-reference.md#presets) を参照)。

実際の構成は [presets/styles/anime-detailed.yaml](../../../../presets/styles/anime-detailed.yaml) と
[presets/characters/anime-boy-hooded.yaml](../../../../presets/characters/anime-boy-hooded.yaml)、
組み合わせた例は
[specs/examples/txt2img_hires.yaml](../../../../specs/examples/txt2img_hires.yaml) にある。

## 指定した色と丈を出す

「白のセーラー服に紺のプリーツスカート」のように服の色を指定すると、色が隣のアイテムへ
流れる。SD1.5系は色語をアイテムへ結び付けきれないため、書き方で押さえる。
以下は hassakuSD15_v13 / 512x768 -> hires x2.0 / seed 4本で確認した結果
(2026-08-15)。

- **色とアイテムを1タグにまとめる。** `navy, pleated skirt` ではなく
  `navy pleated skirt` と書く。色を独立したタグに置くと、その色が画面全体へ回る
- **重みは先回りで振らない。** まず全部を素の語で1枚出し、**外れた箇所だけ**へ重みを足して
  撃ち直す。最初から複数箇所へ重みを振ると、重み同士が競合して指定していない色が勝つ
  (実測: 4箇所へ同時に重みを付けたところ、`(white sailor uniform:1.3)` と書いた上衣が
  紺になった。靴1箇所だけを `(white loafers:1.2)` にした版は全項目が命中した)
- **重みは1.2-1.4に収める。** 1.5を超えると形のほうが崩れる
- **同じ色語を何度も書かない。** 白を4箇所 (`white sailor uniform` / `white shirt` /
  `white knee socks` / `white loafers`) に書くと白が過剰になり、他のアイテムの色を侵食する。
  上位のタグ1つへ集約するか、重みの弱いものから色を落とす。
  重複を減らすと、水色の髪が紺へ寄る崩れと水色のスカーフが濃紺化する崩れは消えた。
  代わりに白の指定が弱まり、seed 4本のうち1本で白のセーラー服が紺になった。
  色語を減らすほど良いのではなく、減らしたうえで外れた箇所だけを重みで補う
- **誤って出る色をnegativeへ名指しする。** `white skirt, light blue skirt,
  dark blue neckerchief, dark blue hair` のように、実際に出てしまった色をそのまま書く。
  「起きうる誤り」を先回りで並べるのではなく、生成した絵を見てから足す
- **丈はnegativeで決める。** `white knee socks` と書いても太もも丈になるため、
  `thighhighs, over-knee socks` をnegativeへ入れる。この2語で膝下丈に固定できた

**1枚で判定しない。** 色の命中はseedごとに揺れる。seedを4本振って
`batch --seeds` で流し、何本命中したかで書き方を比べる。
上記の対策をすべて入れても4本全部は揃わない。実測では靴下の丈と靴の色は4/4、
上衣・スカート・スカーフの色は各3/4で、外れるseedが要素ごとに違った。
確実に固定したい場合はプロンプトではなくキャラクタLoRAかIPAdapterを使う
([character-consistency.md](../../imagegen/references/character-consistency.md))。

**IPAdapterと併用する場合は重みの扱いがさらに厳しくなる。** 参照画像の色調と重み括弧が
競合するため、複数箇所へ重みを振ると素の語で書いたときより命中が落ちる。
IPAdapterを使うときこそ「素の語で1枚 -> 外れた1箇所だけ重み」の順で進める。
色を直すのは1段目 (IPAdapterを効かせる側) で行う。2段目のimg2imgは `denoise` 0.4では
既に塗られた色を塗り替えないため、2段目のプロンプトを直しても色は変わらない。

## 参考

- [Stable Diffusion prompt: a definitive guide](https://stable-diffusion-art.com/prompt-guide/)

