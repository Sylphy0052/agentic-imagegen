# プロンプトとWorkflowのベストプラクティス

対応モデルごとのプロンプト記法と、ComfyUI workflowテンプレートの扱い方をまとめる。
Specの書き方そのものは [spec-reference.md](spec-reference.md)、失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](../.claude/skills/imagegen/references/troubleshooting.md)
を参照する。

## 全モデル共通の原則

- **前方のトークンほど強く効く。** 主題 -> 属性 -> 構図 -> 品質タグ の順に並べる
- **重み付けは `(tag:1.3)`。** SD1.5 / SDXL系は0.7-1.5が実用域。それを超えると色が焼き付き、
  構図が破綻する
- **品質タグは3-4個で打ち止め。** 積み増しても品質は上がらず、主題のトークンが希釈されるだけ
- **プロンプトは削る方が改善する。** 半分に削って壊れた箇所が、実際に効いていたトークン
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
- negativeは15-30語程度が通例。`worst quality, low quality, blurry, bad anatomy, text,
  watermark, signature` のような定型から始める
- **embeddingはcheckpointの世代に固定される。** SD1.5向けのembeddingはSDXLでは機能しない

## SDXL / Illustrious系 (novaAnimeXL_ilV190など)

| 項目 | 目安 |
| --- | --- |
| トークン上限 | 248 |
| cfg | 3-6 (6を超えると過処理になりやすい) |
| steps | 20-32 |
| sampler / scheduler | `euler` / `normal` |

- **品質タグは先頭、構図のmodifierは末尾へ置く。** 後方のタグほど効果が薄まるため、
  重要な要素ほど前に置く
- **`score_9` のようなPony系の記法は使わない。** Illustriousは対応しておらず、
  `masterpiece, best quality` 系の品質タグを使う
- **タグはDanbooruに実在する表記を使う。** 学習データが少ないタグはLoRAなしでは効かない。
  キャラクタ名もDanbooruの表記順に従う
- v2.0以降は自然文とタグの併用に対応する

## Anima系 (hassakuAnima_v13など、DiT + Qwen3-0.6B)

| 項目 | 目安 |
| --- | --- |
| 解像度 | 512x512 - 1536x1536 (832x1216が扱いやすい) |
| cfg | 4-5 |
| steps | 30-50 |
| sampler | `er_sde` (フラットでシャープ) / `euler_ancestral` (柔らかい線) / `dpmpp_2m_sde` (多様性) |
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

モデル配布元が推奨する `beta57` schedulerは本CLIのallowlistに無い。`simple` を使う。
Anima向けのstyle presetは [presets/styles/anima-base.yaml](../presets/styles/anima-base.yaml)
にある。

## ComfyUI workflowのベストプラクティス

`workflows/*.json` の扱いは [workflows/README.md](../workflows/README.md) が一次情報。
ここでは一般則と、本リポジトリでの担保状況を対応させる。

- **API形式で保存する。** GUIの通常のSaveではなく「Save (API Format)」を使う。
  座標・色・グループ・ノードサイズといったUI用のmetadataを落とした形式でないと投入できない
- **`control_after_generate` を残さない。** `randomize` が残っていると実行ごとにseedが変わり、
  再現できなくなる。本リポジトリの22テンプレートには含まれていない
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
- [Arctenox's Simple Prompt Guide for Illustrious](https://civitai.com/articles/23210/arctenoxs-simple-prompt-guide-for-illustrious)
- [Comprehensive Guide of Illustrious XL](https://tensor.art/articles/831123524065191393)
- [circlestone-labs/Anima (model card)](https://huggingface.co/circlestone-labs/Anima)
- [Anima Base v1 ComfyUI workflow example](https://docs.comfy.org/tutorials/image/anima/anima)
- [Workflow API Format (ComfyUI docs)](https://docs.comfy.org/development/api-development/workflow-api-format)
- [ComfyUI API: The Complete Developer's Guide](https://www.runflow.io/blog/comfyui-api-developer-guide)
- [ComfyUI custom nodes: Manager, Nodes 2.0, prod](https://www.runflow.io/blog/comfyui-custom-nodes)
