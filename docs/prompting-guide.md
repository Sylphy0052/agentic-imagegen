# プロンプトとWorkflowのベストプラクティス

対応モデルごとのプロンプト記法と、ComfyUI workflowテンプレートの扱い方をまとめる。
Specの書き方そのものは [spec-reference.md](spec-reference.md)、失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](../.claude/skills/imagegen/references/troubleshooting.md)
を参照する。

## 全モデル共通の原則

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

### Textual Inversion embedding

`easynegative` / `badhandv4` のような定型embeddingを使う場合は、必ず
`embedding:<name>` とプレフィックスを付けて書く (`embedding:easynegative`)。
プレフィックスの無い素の単語はComfyUI側でもembeddingとして解決されず、
ただの単語として扱われる (警告も拒否もされない)。

`embedding:<name>` と書いた場合は、生成前に該当ファイルが
`~/ComfyUI/models/embeddings/` に実在するかを自動で検証する。未配置なら
`imagegen generate` / MCPの `generate_image` が実行前に拒否する
(検証の仕組みと`imagegen validate`との違いは
[spec-reference.md](spec-reference.md#prompt) を参照)。

## SD1.5系 (meinamix_v12Finalなど)

| 項目 | 目安 |
| --- | --- |
| トークン上限 | 75 (CLIP ViT-L/14の77から開始/終了トークンを引いた数) |
| cfg | 7.0前後 (実用域5.0-8.0) |
| steps | 20前後 |
| sampler / scheduler | `dpmpp_2m` / `karras` |

- **danbooruタグを主体にする。** `1girl` `solo` `looking at viewer` のような学習時のタグ語彙へ寄せる
- **「知っていること」ではなく「見えるもの」をタグにする。** 足が写らない構図で靴のタグを書かない
- text encoderが単純なため、プロンプト追従をcfgに依存する。指示が効かないときは
  語順の見直しを先に行い、cfgを上げるのは最後にする
- negativeは `worst quality, low quality, lowres, blurry, bad anatomy, text, watermark,
  signature` のような10語前後の定型から始め、出た症状に応じて足す。
  15語を大きく超えると色が抜け始める
- **embeddingはcheckpointの世代に固定される。** SD1.5向けのembeddingはSDXLでは機能しない

### 配置済みのSD1.5系モデル

`~/ComfyUI/models/checkpoints/` にあるSD1.5系のcheckpointと、配布元・利用者が挙げている
推奨設定。Specでは `model.checkpoint` にファイル名をそのまま書く。
どれもdanbooruタグ主体で書く点は共通で、差が出るのはcfgの実用域と塗りの傾向。

| checkpoint | 傾向 | sampler / scheduler | steps | cfg | style preset |
| --- | --- | --- | --- | --- | --- |
| `meinamix_v12Final.safetensors` | アニメ調。プロンプトが短くてもまとまる | `dpmpp_2m` / `karras` | 20-60 | 4-9 | `sd15-meinamix` |
| `counterfeitV30_v30.safetensors` | アニメ調。背景と色彩の描き込みが厚い | `dpmpp_2m` / `karras` | 20-30 | 8-10 | `sd15-counterfeit` |
| `abyssorangemix3AOM3_aom3a1b.safetensors` | アニメ調。イラスト寄りの塗り | `dpmpp_sde` / `karras` | 20-30 | 6以上 | `sd15-aom3` |
| `anyloraCheckpoint_bakedvaeBlessedFp16.safetensors` | ニュートラルなアニメ調。LoRAの土台向け | `dpmpp_2m` / `karras` | 20-30 | 7前後 | `sd15-anylora` |
| `cetusMix_Whalefall2.safetensors` | フラットなアニメ調。人物と背景の分離が良い | `dpmpp_2m` / `karras` | 20以上 | 4-8 | `sd15-cetusmix` |
| `darkSushiMixMix_225D.safetensors` | 2.25D (2Dと2.5Dの中間) | `dpmpp_sde` / `karras` | 20-60 | 7.5 | `sd15-darksushi` |
| `hassakuSD15_v13.safetensors` | 明るくコントラストの強いアニメ調 | `ddim` / `normal` | 20 | 8 | `sd15-hassaku` |
| `chilloutmix_NiPrunedFp16Fix.safetensors` | 写実寄り。人物の肌と質感に振れる | `dpmpp_sde` / `karras` | 20前後 | 7前後 | `sd15-chilloutmix` |
| `perfectdeliberate_v20.safetensors` | アニメ調。厚めの塗り。高めの解像度が前提 | `dpmpp_2m` / `karras` | 20-50 | 5-8 | `sd15-perfectdeliberate` |
| `waiIllustriousSD15_v1.safetensors` | WAI-illustrious-SDXLの蒸留版。Illustrious系のタグ記法で書く | `dpmpp_2m` / `karras` | 20-30 | 5-7 | `sd15-wai-illustrious` |

style presetはこの表の値を持っているため、checkpointに合うものを選べば
sampler / scheduler / cfg / stepsをSpec側で書き直す必要はない。
ただし `sd15-meinamix` と `sd15-anylora` の2つだけは表の値ではなく、同じcheckpointを
A1111で運用したときの実績設定 (`dpmpp_2m_sde` / `exponential` / cfg 7.0 / steps 30) を持つ。
経緯は後述の[A1111から設定を移す](#a1111から設定を移す)を参照。

- **cfgの実用域はモデルごとに違う。** `counterfeitV30` の8-10と `cetusMix` の4-8では、
  同じ7でも意味が変わる。別のcheckpointのstyle presetを流用するときはcfgとstepsを見直す
- **`chilloutmix` だけ写実寄り。** アニメ調のstyle preset (`anime-soft` / `anime-detailed` /
  他の `sd15-*`) を当てると品質タグが打ち消し合う。`sd15-chilloutmix` を使う
- **`anylora` はLoRAを載せる土台としてニュートラルに作られている。** 単体で使うより
  `model.loras` と組み合わせる方が本来の用途。`sd15-anylora` はLoRAの画風と競合しないよう
  `anime coloring` を入れていない
- **`waiIllustriousSD15_v1` は配布元がsampler / cfg / stepsの推奨を出していない。**
  表の値は蒸留元のIllustrious系に倣った暫定。品質タグの記法もIllustrious系に合わせ、
  Pony系の `score_9` 記法は使わない
- **checkpointを決めていない段階は `hassakuSD15_v13` + `sd15-hassaku` を使う。**
  後述の[既定のcheckpointを決める](#既定のcheckpointを決める)を参照。
  `anime-soft` / `anime-detailed` は負荷を下げたいときの汎用preset
  (`anime-soft` が下描き、`anime-detailed` が仕上げ) として残してある
- **`AnythingXL_xl.safetensors` はSDXL系。** 同じ `checkpoints/` に置かれているが、
  SD1.5向けのstyle presetと設定を流用しない。目安は後述の
  [SDXL / Illustrious系](#sdxl--illustrious系-novaanimexl_ilv190など)を参照

配布元の推奨のうち、Spec側で明示しないと既定値のままになるものが3つある。

- **clip skipは大半のモデルが2を推奨する。** 既定は未指定 (1相当) のため、
  `model.clip_skip: 2` と書く
  ([model.clip_skip](spec-reference.md#modelclip_skip))。
  1のままでも破綻はしないが、配布元のサンプル画像へ絵柄を寄せたい場合は差が出る
- **外部VAEの差し替えを前提にするモデルが多い。** `model.vae` へファイル名を書くと
  checkpoint同梱のVAEの代わりに使う ([model](spec-reference.md#model))。
  配置済みは `vaeKlF8Anime2_klF8Anime2VAE.safetensors` / `kl-f8-anime2.ckpt` /
  `vae-ft-mse-840000-ema-pruned.safetensors` で、実在するものは `list_vaes` で確認する。
  アニメ調のSD1.5系には `vaeKlF8Anime2_klF8Anime2VAE.safetensors` を使う
  (`kl-f8-anime2.ckpt` は同名を騙る別ファイル。後述の[A1111から設定を移す](#a1111から設定を移す)を参照)。
  `anylora` はVAEを焼き込み済みのため差し替えなくても動くが、
  焼き込み済みのVAEと外部VAEでは彩度と線の締まりが変わる
- **hires fixのアップスケーラは `R-ESRGAN 4x+Anime6B` が定番。**
  `generation.upscale.model` へ `RealESRGAN_x4plus_anime_6B.pth` を指定する
  ([generation.upscale](spec-reference.md#generationupscale-hires-fix))

### タグの実在を確認する

存在しないタグは学習語彙に対応する概念を持たない。SD1.5は75トークンしか使えないため、
効かないタグを並べるとそれだけ主題のトークンが希釈される。
思いついたタグは書く前にDanbooruで実在を確認する。

```bash
curl -s 'https://danbooru.donmai.us/tags.json?search%5Bname%5D=oversized_clothes&limit=1' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[0]["post_count"] if d else "NOT_FOUND")'
```

複数のタグをまとめて確認する場合は
[.claude/skills/prompt-builder/scripts/tagcheck.py](../.claude/skills/prompt-builder/scripts/tagcheck.py)
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

### タグをブロックで組む

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
(連結順は [spec-reference.md](spec-reference.md#presets) を参照)。

実際の構成は [presets/styles/anime-detailed.yaml](../presets/styles/anime-detailed.yaml) と
[presets/characters/anime-boy-hooded.yaml](../presets/characters/anime-boy-hooded.yaml)、
組み合わせた例は
[specs/examples/txt2img_hires.yaml](../specs/examples/txt2img_hires.yaml) にある。

### 指定した色と丈を出す

「白のセーラー服に紺のプリーツスカート」のように服の色を指定すると、色が隣のアイテムへ
流れる。SD1.5系は色語をアイテムへ結び付けきれないため、書き方で押さえる。
以下は hassakuSD15_v13 / 512x768 -> hires x2.0 / seed 4本で確認した結果
(2026-08-15)。

- **色とアイテムを1タグにまとめ、重みを付ける。** `navy, pleated skirt` ではなく
  `(navy pleated skirt:1.3)` と書く。色を独立したタグに置くと、その色が画面全体へ回る
- **重みは1.2-1.4に収める。** 1.5を超えると形のほうが崩れる。押さえたい順に強くする
  (最も流れやすいものを1.4、確実に出ているものは無指定でよい)
- **同じ色語を何度も書かない。** 白を4箇所 (`white sailor uniform` / `white shirt` /
  `white knee socks` / `white loafers`) に書くと白が過剰になり、他のアイテムの色を侵食する。
  上位のタグ1つへ集約するか、重みの弱いものから色を落とす。
  重複を減らすと、水色の髪が紺へ寄る崩れと水色のスカーフが濃紺化する崩れは消えた。
  代わりに白の指定が弱まり、seed 4本のうち1本で白のセーラー服が紺になった。
  色語を減らすほど良いのではなく、押さえたい色ごとに重みで序列を付ける
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
([character-consistency.md](../.claude/skills/imagegen/references/character-consistency.md))。

### hires fixの値

512x768で構図を作り、`upscale.scale: 2.0`で1024x1536へ引き上げるのがSD1.5系の定番。

- `denoise`は0.5-0.65あたりが2段目で描き足す量として扱いやすい。
  上げるほど元の構図から離れ、下げるほど拡大しただけの絵に近づく
- `upscale.steps`は1段目の1/3程度 (steps 30なら10) から始める
- 2段目のcfgとsamplerは1段目と同じ値を使う。片方だけ変える手段は用意していない
- `upscale.model`を書くとlatent拡大ではなくアップスケールモデル (ESRGAN系) で拡大する。
  `RealESRGAN_x4plus_anime_6B.pth` が配置済み。省略するとlatent拡大になる
- 外部VAE (`model.vae`) とclip skip (`model.clip_skip`) はhires fixと併用できる
  (前掲の[配置済みのSD1.5系モデル](#配置済みのsd15系モデル)を参照)

## SDXL / Illustrious系 (novaAnimeXL_ilV190など)

| 項目 | 目安 |
| --- | --- |
| トークン上限 | 248 |
| 解像度 | 1024x1024相当の画素数。縦長は832x1216 / 1024x1536 |
| cfg | 4.5-7 (7.5を超えると彩度が飽和し、3未満は色が抜ける) |
| steps | 20-30 |
| sampler / scheduler | `euler_ancestral` / `normal` |

- **品質タグは先頭、構図のmodifierは末尾へ置く。** 後方のタグほど効果が薄まるため、
  重要な要素ほど前に置く
- **`score_9` のようなPony系の記法は使わない。** Illustriousは対応しておらず、
  `masterpiece, best quality` 系の品質タグを使う
- **タグはDanbooruに実在する表記を使う。** 学習データが少ないタグはLoRAなしでは効かない。
  キャラクタ名もDanbooruの表記順に従う
  (確認手順は [タグの実在を確認する](#タグの実在を確認する))
- v2.0以降は自然文とタグの併用に対応する
- **配布元はclip skip 2を推奨する。** 既定は未指定 (1相当) のため、
  `model.clip_skip: 2` と書く ([model.clip_skip](spec-reference.md#modelclip_skip))

### モデルごとの推奨設定

同じSDXLでも、fine-tuneの系統ごとに品質タグの語彙とサンプラー設定が割れる。
style presetを系統ごとに分けているのはこのため。

| モデル | sampler / scheduler | cfg | steps | 品質タグの語彙 | style preset |
| --- | --- | --- | --- | --- | --- |
| Illustrious系 (novaAnimeXL / hassakuXL / waiNSFWIllustrious) | `euler_ancestral` / `normal` | 7 | 30 | `masterpiece, best quality, absurdres, highres` | `sdxl-illustrious` |
| Animagine XL 4.0 | `euler_ancestral` / `normal` | 5-6 | 25 | `masterpiece, high score, great score, absurdres` | `sdxl-animagine` |
| AnythingXL | `euler_ancestral` / `normal` | 5-7 | 25-30 | Illustrious系と同じ | `sdxl-illustrious` |
| ShiratakiMix XL | `dpmpp_3m_sde` / `karras` | 7.5 (3-8) | 20以上 | Illustrious系と同じ | `sdxl-shiratakimix` |

- **Animagine XLの品質タグは他系統へ流用しない。** `high score` / `great score` は
  Animagineの学習語彙で、Illustrious系では効かない。逆も同じ。
  どちらもDanbooruタグではなく学習時に付与された品質ラベルのため、
  `post_count` では判定しない
- **ShiratakiMix XLだけサンプラーの系統が違う。** `euler_ancestral`でも生成できるが、
  配布元のサンプルはDPM++系 + karrasで作られている
- **SDE系サンプラーはstepsを削ると破綻する。** `dpmpp_3m_sde` + `karras`を
  steps 8で流すと収束せず、ほぼ真っ白な画像になる (2026-08-13にXPUで確認)。
  steps 24では正常に生成できる。動作確認のためにstepsを落とす場合は
  `sdxl-illustrious` (`euler_ancestral`) を使う
- ComfyUIへ実在するSDXL checkpointは`novaAnimeXL_ilV190.safetensors`と
  `AnythingXL_xl.safetensors`。animagineXL / hassakuXL / shiratakimixXL /
  waiNSFWIllustriousは未配置のため、使う前に
  `~/ComfyUI/models/checkpoints/` へ置く

### SDXLでのhires fix

832x1216で構図を作り、`upscale.scale: 1.5`で1248x1824へ引き上げる。

- `denoise`は0.35-0.5がSDXLで扱いやすい。SD1.5系より低めの値で足りる
- `upscale.steps`は1段目の1/3程度 (steps 30なら10)
- **実運用の定番である1024x1536の2倍 (2048x3072) は既定の上限を超える。**
  `IMAGEGEN_MAX_HEIGHT` (2048) と`IMAGEGEN_MAX_PIXELS` (4194304) の両方に当たるため、
  通すには環境変数を引き上げる
- **配布元が推奨する`R-ESRGAN 4x+Anime6B`は`upscale.model`で使える。**
  `RealESRGAN_x4plus_anime_6B.pth` を指定する
- **`sdxlVAE`のような外部VAEは`model.vae`で差し替えられる。** ただし配置済みのVAEは
  SD1.5向けのため、SDXL向けのVAEを使うなら`~/ComfyUI/models/vae/`へ置いてから指定する

SDXLはSD1.5の3-4倍の計算量になる。CPU推論では実用的な時間で終わらないため、
XPU ([xpu-setup.md](xpu-setup.md)) を用意してから使う。

構成例は [specs/examples/txt2img_sdxl.yaml](../specs/examples/txt2img_sdxl.yaml)、
preset本体は [presets/styles/sdxl-illustrious.yaml](../presets/styles/sdxl-illustrious.yaml)
にある。

## Anima系 (hassakuAnima_v13など、DiT + Qwen3-0.6B)

| 項目 | 目安 |
| --- | --- |
| 解像度 | 512x512 - 1536x1536 (832x1216が扱いやすい) |
| cfg | 4-5 |
| steps | 30-50 |
| sampler | `er_sde` (フラットでシャープ) / `euler_ancestral` (柔らかい線) / `dpmpp_2m_sde_gpu` (多様性) |
| scheduler | `simple` |

- **danbooruタグ・自然文・その混在をすべて受け付ける。** 自然文で書く場合は最低2文書く
- **タグの並び順:** quality / meta / year / safety -> 人数 -> キャラクタ -> シリーズ -> 絵師
  -> 一般タグ
- **小文字とスペースで書く。** アンダースコアを使うのはscoreタグ (`score_7`) だけ
- **絵師タグは `@` を前置する** (`@artist_name`)。付けないとほとんど効かない
- 推奨するpositiveの接頭: `masterpiece, best quality, score_7, safe,`
- 推奨するnegative: `worst quality, low quality, score_1, score_2, score_3, artist name,
  blurry, jpeg artifacts, chromatic aberration`
- **重み付けはSDXL系より強い値が要る。** `(chibi:2)` のような指定でようやく効く
- tag dropoutで学習されているため、関連タグを網羅する必要はない
- **SD1.5 / SDXL向けのTextual Inversion embeddingを流用できるとは限らない。**
  text encoderがCLIPではなくQwen3-0.6Bのため、`embedding:<name>` の実在チェック自体は
  他のモデルと同様に働くが、埋め込みベクトルの互換性はComfyUI側の責務であり未検証

**モデル配布元が推奨する `beta57` はKSamplerのschedulerではない。**
`beta` schedulerのalpha=0.5 / beta=0.7を指す通称であり、指定するには `BetaSamplingScheduler`
ノードを持つWorkflowが要る。本リポジトリのテンプレートはKSamplerベースのため使えない。
`simple` を使う。指定できるschedulerの一覧は
[docs/spec-reference.md](spec-reference.md#generation) を参照。

Anima向けのstyle presetは [presets/styles/anima-base.yaml](../presets/styles/anima-base.yaml)
にある。

## 既定のcheckpointを決める

checkpointを指定されていないときは `hassakuSD15_v13.safetensors` + `sd15-hassaku` を使う。

配置済みのSD1.5系9種を、同一プロンプト・同一seed (545078971)・同一設定
(`dpmpp_2m_sde` / `exponential` / cfg 7.0 / steps 30 / 512x768 -> hires x2.0 /
clip skip 2 / 外部VAE) で1枚ずつ生成して比べた結果による (2026-08-15)。

| checkpoint | 高周波比 | 所見 |
| --- | ---: | --- |
| `hassakuSD15_v13` | 0.068 | 指定した服装・小物への追従が最も良い。破綻なし |
| `anyloraCheckpoint_bakedvaeBlessedFp16` | 0.058 | 最も平滑。LoRAの土台向けで単体では平坦 |
| `meinamix_v12Final` | 0.062 | 破綻はないが暗く沈みやすい |
| `abyssorangemix3AOM3_aom3a1b` | 0.065 | 彩度が高い |
| `perfectdeliberate_v20` | 0.074 | 厚塗り |
| `darkSushiMixMix_225D` | 0.077 | 2.25D |
| `counterfeitV30_v30` | 0.084 | 明るい。背景の描き込みが厚い |
| `cetusMix_Whalefall2` | 0.088 | 最も線が立つ |
| `chilloutmix_NiPrunedFp16Fix` | 0.044 | 写実寄り。アニメ品質タグと打ち消し合う |

高周波比はエッジ強度が閾値を超えた画素の割合で、破綻の検出に使う目安。
正常な範囲は0.04-0.09で、これを大きく超えるものは絵が壊れている
(VAE不整合を起こした `waiIllustriousSD15_v1` は0.296だった)。
数値は破綻の有無しか見ていないため、順位付けには使わない。採用の決め手は
「指定した服装・小物がそのまま出るか」で、そこに最も忠実だったのがhassakuだった。

`sd15-hassaku` は配布元推奨の `ddim` / `normal` / steps 20 / cfg 8 ではなく、
この比較で使った設定を持つ。既定として使う場合もSpec側に次の2つを書く。

```yaml
model:
  checkpoint: hassakuSD15_v13.safetensors
  clip_skip: 2
  vae: vaeKlF8Anime2_klF8Anime2VAE.safetensors
```

`waiIllustriousSD15_v1` だけは `model.vae` を書いてはいけない。このcheckpointは
SDXL系のVAEを内蔵しており、SD1.5用の外部VAEへ差し替えると出力が極彩色のノイズになる。

## A1111から設定を移す

A1111 (Stable Diffusion web UI) で気に入った絵ができているなら、その設定をSpecへ写せば
同じ絵柄を再現できる。実際に4枚の生成画像から設定を復元し、絵柄が一致するところまで
確認した手順を残す。

### 生成画像から設定を読む

A1111はPNGのテキストチャンクへ生成パラメータを丸ごと書き込む。

```bash
python3 -c "from PIL import Image; print(Image.open('inputs/ref.png').info['parameters'])"
```

読めるのは positive / negative / Steps / Sampler / Schedule type / CFG scale / Seed / Size /
Model hash / Model / VAE hash / VAE / Denoising strength / Clip skip / Hires系の各値、
それにA1111のVersion。ここに無い項目は既定値だったということ。

- **LoRA・ADetailer・ControlNetを使っていれば必ず記録される。** 記載が無ければ使っていない
- **`RNG` と `ENSD` が無ければ既定値。** 設定を変えた場合だけ書き込まれる
- **Hires系の値 (`Hires upscale` / `Hires steps` / `Hires upscaler`) が無ければhires fixは未使用**

### モデルの同一性をハッシュで確かめる

同じファイル名でも中身が違うことがある。A1111が記録する `Model hash` / `VAE hash` は
AutoV2 (ファイル全体のsha256の先頭10桁) のため、手元のファイルと直接照合できる。

```bash
sha256sum ~/ComfyUI/models/vae/*.safetensors | cut -c1-10
```

一致しなければ別のファイル。civitaiは
`https://civitai.com/api/v1/model-versions/by-hash/<sha256先頭10桁>` でハッシュから
元のファイルを引けるため、そこから正しいものを落とす。

実例として、手元の `kl-f8-anime2.ckpt` (`df3c506e51`) は参考画像が使っていた
`vaeKlF8Anime2_klF8Anime2VAE.safetensors` (`b8821a5d58`) とは別のファイルだった。
名前が同じ系統でも中身が違い、これが絵柄の差の主因だった。

### Specへ写す

sampler / scheduler / cfg / steps とプロンプトはstyle presetへ入れられる。
残りの3つはpresetに書く場所が無いため、Spec側に書く。

| A1111の項目 | Specの書き場所 |
| --- | --- |
| Sampler / Schedule type / CFG scale / Steps | style presetの `generation` (またはSpecの `generation`) |
| Clip skip | `model.clip_skip` |
| VAE | `model.vae` |
| Hires upscaler / upscale / steps / Denoising strength | `generation.upscale` |

`model.clip_skip` を省くと1相当になり、2を前提にしたアニメ系モデルでは絵柄が変わる。
書き方をまとめた例は
[specs/examples/txt2img_a1111_compat.yaml](../specs/examples/txt2img_a1111_compat.yaml) にある。

Textual Inversionのnegative embedding (`negativeXL_D` など) は
プロンプト中へ `embedding:negativeXL_D` と書く
(前述の[Textual Inversion embedding](#textual-inversion-embedding)を参照)。

### 揃うものと揃わないもの

上をすべて合わせると絵柄・塗り・線は一致する。構図とポーズは同じseedでも一致しない。

- **初期ノイズの生成器が違う。** A1111はCPU、ComfyUIはGPUでノイズを作るため、
  同じseedでもノイズのパターン自体が別物になる
- **プロンプトの重み付けの正規化方式が違う。** `(word:1.2)` の効き方が完全には一致しない
- **75トークンを超えた分の連結方法が違う。** 長いプロンプトほど差が出る

構図まで含めて同じ絵が要るなら、A1111側の出力をそのまま使う。
ComfyUI側では絵柄を合わせたうえでseedを振り直し、好みの構図を選び直す。

## ComfyUI workflowのベストプラクティス

`workflows/*.json` の扱いは [workflows/README.md](../workflows/README.md) が一次情報。
ここでは一般則と、本リポジトリでの担保状況を対応させる。

- **API形式で保存する。** GUIの通常のSaveではなく「Save (API Format)」を使う。
  座標・色・グループ・ノードサイズといったUI用のmetadataを落とした形式でないと投入できない
- **`control_after_generate` を残さない。** `randomize` が残っていると実行ごとにseedが変わり、
  再現できなくなる。本リポジトリの同梱テンプレートには含まれていない
- **workflow JSONをバージョン管理下に置き、実行時は入力値だけ差し替える。**
  実行時にグラフを組み立てない (本リポジトリの設計方針と同じ)
- **workflowと実行環境をセットで固定する。** ComfyUI本体のcommit、custom nodeのリリース、
  checkpointのハッシュまで含めて1つの成果物として扱う。ComfyUI Managerが既定で最新版を
  取りにいくため、放置するとworkflowが壊れる
- **custom nodeは必要最小限に絞る。** 各パックがPython依存を持ち込み、衝突と起動遅延を招く。
  IPAdapterのようにノードが無いと投入が拒否される依存は、導入条件として明記する
- **GUIで編集するときはgroupとrerouteで整理する。** API形式には残らないが、
  テンプレートを人間が保守する以上、原本の可読性が変更コストを決める
- **テンプレートの変更を検出できるようにする。** 本リポジトリでは正規化JSONの
  `workflow_hash` を `metadata.json` へ記録しており、同じSpecで結果が変わったときに
  テンプレート側の変更かどうかを切り分けられる

## 参考

- [Stable Diffusion prompt: a definitive guide](https://stable-diffusion-art.com/prompt-guide/)
- [MeinaMix (Civitai)](https://civitai.com/models/7240)
- [Counterfeit-V3.0 (Civitai)](https://civitai.com/models/4468/counterfeit-v30)
- [AbyssOrangeMix3 (CivArchive)](https://civarchive.com/models/9942)
- [AnyLoRA - Checkpoint (Civitai)](https://civitai.com/models/23900/anylora-checkpoint)
- [Cetus-Mix (Civitai)](https://civitai.com/models/6755/cetus-mix)
- [Dark Sushi Mix (Civitai)](https://civitai.com/models/24779/dark-sushi-mix-mix)
- [Hassaku (SD1.5) (Civitai)](https://civitai.com/models/2583)
- [Arctenox's Simple Prompt Guide for Illustrious](https://civitai.com/articles/23210/arctenoxs-simple-prompt-guide-for-illustrious)
- [Comprehensive Guide of Illustrious XL](https://tensor.art/articles/831123524065191393)
- [circlestone-labs/Anima (model card)](https://huggingface.co/circlestone-labs/Anima)
- [Anima Base v1 ComfyUI workflow example](https://docs.comfy.org/tutorials/image/anima/anima)
- [Workflow API Format (ComfyUI docs)](https://docs.comfy.org/development/api-development/workflow-api-format)
- [ComfyUI API: The Complete Developer's Guide](https://www.runflow.io/blog/comfyui-api-developer-guide)
- [ComfyUI custom nodes: Manager, Nodes 2.0, prod](https://www.runflow.io/blog/comfyui-custom-nodes)
