# プロンプトとWorkflowのベストプラクティス

モデル系統ごとのプロンプト記法は
[prompt-builder skill](../.claude/skills/prompt-builder/SKILL.md) の references 配下が
一次情報になっている。この文書はその索引と、ComfyUI workflowテンプレートの扱い方を持つ。
Specの書き方そのものは [spec-reference.md](spec-reference.md)、失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](../.claude/skills/imagegen/references/troubleshooting.md)
を参照する。

## モデル系統ごとのプロンプト記法

トークン上限・cfg / stepsの実用域・品質タグの語彙・タグ記法は系統ごとに違う。
`model.checkpoint` (DiT系は `model.unet`) から系統を決めて、対応する1本を読む。

| 系統 | 該当するモデル | 参照 |
| --- | --- | --- |
| SD1.5系 | `meinamix` / `counterfeit` / `hassakuSD15` / `waiIllustriousSD15` など | [models/sd15.md](../.claude/skills/prompt-builder/references/models/sd15.md) |
| SDXL系 (Illustrious由来を除く) | `AnythingXL` / `animagineXL` / `shiratakimixXL` | [models/sdxl.md](../.claude/skills/prompt-builder/references/models/sdxl.md) |
| Illustrious系 | `novaAnimeXL` / `hassakuXL` / `waiNSFWIllustrious` | [models/illustrious.md](../.claude/skills/prompt-builder/references/models/illustrious.md) |
| Anima系 (DiT) | `hassakuAnima` など、`model.unet` / `clip` / `vae` の3点指定 | [anima-prompt skill](../.claude/skills/anima-prompt/SKILL.md) |

系統によらない内容は次の3本にある。

| 内容 | 参照 |
| --- | --- |
| 全モデル共通の原則、Textual Inversion embedding、タグの実在確認、ブロックの組み方、指定した色と丈の出し方 | [references/common.md](../.claude/skills/prompt-builder/references/common.md) |
| A1111 (Stable Diffusion web UI) の設定をSpecへ写す手順 | [references/a1111-migration.md](../.claude/skills/prompt-builder/references/a1111-migration.md) |
| 存在しないタグの置換実績 | [references/tag-replacements.md](../.claude/skills/prompt-builder/references/tag-replacements.md) |

配置済みのcheckpointの一覧と、checkpointを指定されなかったときの既定は
[models/sd15.md](../.claude/skills/prompt-builder/references/models/sd15.md) にある。

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

- [Workflow API Format (ComfyUI docs)](https://docs.comfy.org/development/api-development/workflow-api-format)
- [ComfyUI API: The Complete Developer's Guide](https://www.runflow.io/blog/comfyui-api-developer-guide)
- [ComfyUI custom nodes: Manager, Nodes 2.0, prod](https://www.runflow.io/blog/comfyui-custom-nodes)
