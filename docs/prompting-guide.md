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

style presetはこの表の値を持っているため、checkpointに合うものを選べば
sampler / scheduler / cfg / stepsをSpec側で書き直す必要はない。

- **cfgの実用域はモデルごとに違う。** `counterfeitV30` の8-10と `cetusMix` の4-8では、
  同じ7でも意味が変わる。別のcheckpointのstyle presetを流用するときはcfgとstepsを見直す
- **`chilloutmix` だけ写実寄り。** アニメ調のstyle preset (`anime-soft` / `anime-detailed` /
  他の `sd15-*`) を当てると品質タグが打ち消し合う。`sd15-chilloutmix` を使う
- **`anylora` はLoRAを載せる土台としてニュートラルに作られている。** 単体で使うより
  `model.loras` と組み合わせる方が本来の用途。`sd15-anylora` はLoRAの画風と競合しないよう
  `anime coloring` を入れていない
- **checkpointを決めていない段階では `anime-soft` / `anime-detailed` を使う。**
  負荷で選ぶ汎用preset (`anime-soft` が下描き、`anime-detailed` が仕上げ) として残してある
- **`AnythingXL_xl.safetensors` はSDXL系。** 同じ `checkpoints/` に置かれているが、
  SD1.5向けのstyle presetと設定を流用しない。目安は後述の
  [SDXL / Illustrious系](#sdxl--illustrious系-novaanimexl_ilv190など)を参照

配布元の推奨のうち、現状の実装では指定できないものが3つある。

- **clip skipは大半のモデルが2を推奨する。** 現行のテンプレートは1相当で固定
  ([Issue #60](https://github.com/Sylphy0052/agentic-imagegen/issues/60))。
  1で運用する分には現状の出力と一致するが、配布元のサンプル画像へ絵柄を寄せたい場合は差が出る
- **外部VAEの差し替えを前提にするモデルが多い**
  (`kl-f8-anime2` / `vae-ft-mse-840000-ema` / `Pastel-Waifu-Diffusion`)。
  現行未対応 ([Issue #57](https://github.com/Sylphy0052/agentic-imagegen/issues/57))。
  `anylora` はVAEを焼き込み済みのため差し替え不要
- **hires fixのアップスケーラは `R-ESRGAN 4x+Anime6B` が定番**
  ([Issue #58](https://github.com/Sylphy0052/agentic-imagegen/issues/58))

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

### hires fixの値

512x768で構図を作り、`upscale.scale: 2.0`で1024x1536へ引き上げるのがSD1.5系の定番。

- `denoise`は0.5-0.65あたりが2段目で描き足す量として扱いやすい。
  上げるほど元の構図から離れ、下げるほど拡大しただけの絵に近づく
- `upscale.steps`は1段目の1/3程度 (steps 30なら10) から始める
- 2段目のcfgとsamplerは1段目と同じ値を使う。片方だけ変える手段は用意していない
- アップスケーラは使えずlatent拡大だけになる。外部VAEとclip skipも指定できない
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
- **配布元はclip skip 2を推奨するが指定できない**
  ([Issue #60](https://github.com/Sylphy0052/agentic-imagegen/issues/60))。
  既定は1相当のため、clip skip 1で運用している場合との差は出ない

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
- **配布元が推奨する`R-ESRGAN 4x+Anime6B`は使えない**
  (latent拡大のみ。[Issue #58](https://github.com/Sylphy0052/agentic-imagegen/issues/58))
- **`sdxlVAE`のような外部VAEへの差し替えも未対応**
  ([Issue #57](https://github.com/Sylphy0052/agentic-imagegen/issues/57))。
  checkpoint同梱のVAEを使う

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

**モデル配布元が推奨する `beta57` はKSamplerのschedulerではない。**
`beta` schedulerのalpha=0.5 / beta=0.7を指す通称であり、指定するには `BetaSamplingScheduler`
ノードを持つWorkflowが要る。本リポジトリのテンプレートはKSamplerベースのため使えない。
`simple` を使う。指定できるschedulerの一覧は
[docs/spec-reference.md](spec-reference.md#generation) を参照。

Anima向けのstyle presetは [presets/styles/anima-base.yaml](../presets/styles/anima-base.yaml)
にある。

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
