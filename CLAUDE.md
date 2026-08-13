# CLAUDE.md

Claude Codeがこのリポジトリを操作するときのルール。

## このプロジェクトは何か

AIコーディングエージェントから、ComfyUI経由でStable Diffusion系モデルによる画像生成を実行する基盤。

```text
Claude Code -> GenerationSpec -> Python CLI (imagegen) -> ComfyUI API -> 画像生成
```

txt2img / preset / LoRA / img2img / MCP Server / ControlNet / IPAdapter / hires fix / batch /
日本語テキスト合成 / DiT系モデル (Anima) 対応まで実装済み。
ここまでの経緯と実機確認の結果は
[Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1) (Roadmap、完了済み)
の棚卸しにまとまっている。未着手の拡張はopenなIssueで個別に管理し、それぞれ着手条件を
明文化してある。最初の設計は [docs/plan/phase1.md](docs/plan/phase1.md) を参照。

## 文書の役割分担

同じことを複数の文書へ書かない。実体は1箇所に置き、他からは参照する。

| 文書 | 何を書くか |
| --- | --- |
| **CLAUDE.md** (この文書) | リポジトリを操作するときのルール。禁止事項・設計原則・ディレクトリ・環境変数・exit code |
| [docs/spec-reference.md](docs/spec-reference.md) | GenerationSpecの全フィールド仕様。値域・既定値・組み合わせ規則・metadata.json |
| [.claude/skills/imagegen/SKILL.md](.claude/skills/imagegen/SKILL.md) | 画像生成要求を受けたときの手順 |
| [README.md](README.md) | プロジェクトの紹介、セットアップ、CLIの使い方 |
| [docs/](docs/) | 環境構築とモデル別の運用知識 |

パラメータを1つ足したときに直すのは、実装と `docs/spec-reference.md` だけで済む形を保つ。

## 画像生成要求を受けたときの手順

詳細な手順とpresetの選び方は [.claude/skills/imagegen/SKILL.md](.claude/skills/imagegen/SKILL.md)
にある。失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](.claude/skills/imagegen/references/troubleshooting.md)
を参照する。

ユーザーから「〇〇な画像を生成して」と指示された場合、次の順で実行する。

1. **GenerationSpecを作る** — 自然言語の要求をSpecの各フィールドへ落とし込む
   (フィールドの仕様は [docs/spec-reference.md](docs/spec-reference.md))
2. **`specs/generated/` へ保存する** — ファイル名は内容が分かるものにする (例: `specs/generated/blue-hair-girl.yaml`)
3. **validateを実行する**

   ```bash
   uv run imagegen validate specs/generated/<name>.yaml
   ```

4. **generateを実行する**

   ```bash
   uv run imagegen generate specs/generated/<name>.yaml
   ```

5. **結果を確認する** — exit codeが0であること、出力ファイルが存在すること
6. **output pathをユーザーへ返す** — 生成された画像のパスとseedを伝える

ComfyUIが起動していない場合は `uv run imagegen health` で状態を確認し、
[docs/comfyui-setup.md](docs/comfyui-setup.md) の手順を案内する。

Specの書き方はサンプルを参照する (`specs/examples/`)。

| ファイル | 内容 |
| --- | --- |
| [txt2img.yaml](specs/examples/txt2img.yaml) | 最小構成のtxt2img |
| [txt2img_preset_lora.yaml](specs/examples/txt2img_preset_lora.yaml) | preset・LoRAを使う場合 |
| [img2img.yaml](specs/examples/img2img.yaml) | 既存画像を入力にするimg2img |
| [txt2img_hires.yaml](specs/examples/txt2img_hires.yaml) | preset・hires fixで解像度を上げる場合 |
| [txt2img_anima.yaml](specs/examples/txt2img_anima.yaml) | DiT系モデル (Anima) を使う場合 |
| [text_overlay.yaml](specs/examples/text_overlay.yaml) | 生成後に日本語テキストを合成する場合 |

モデルごとにプロンプトの書き方が違う (タグ語彙・語順・重み付けの効き方・品質タグの記法)。
どのモデルで何を書くかは [docs/prompting-guide.md](docs/prompting-guide.md) を参照する。

## 使える機能と参照先

値域・既定値・注意書きの実体はすべて
[docs/spec-reference.md](docs/spec-reference.md) にある。ここは索引に留める。

| 機能 | Specの書き方 | 参照 |
| --- | --- | --- |
| preset (character / scene / style) | `presets:` に軸ごと1つまで | [presets](docs/spec-reference.md#presets) |
| LoRA | `model.loras` に列挙する | [model.loras](docs/spec-reference.md#modelloras) |
| img2img | `task: img2img` と `source.image` | [source](docs/spec-reference.md#source-img2img) |
| ControlNet (構図の指定) | `control.image` と `control.model` | [control](docs/spec-reference.md#control-controlnet) |
| IPAdapter (特徴の引き継ぎ) | `reference.image` / `model` / `clip_vision` | [reference](docs/spec-reference.md#reference-ipadapter) |
| hires fix (解像度を上げる) | `generation.upscale.scale` | [generation.upscale](docs/spec-reference.md#generationupscale-hires-fix) |
| 日本語テキスト合成 | `text.layers` に重ねる文字を並べる | [text](docs/spec-reference.md#text-テキスト合成) |
| DiT系モデル (Anima) | `model.unet` / `clip` / `vae` の3点 | [DiT系モデル](docs/spec-reference.md#dit系モデル-anima) |

判断が要る箇所だけをここに書く。

- **軸の責務を混ぜない。** 解像度とseedは再現性に直結するためpresetには書かず、Spec側で指定する
- **style presetはモデル系統ごとに用意する。** `anime-soft` / `anime-detailed` はSD1.5向け、
  `anima-base` はAnima向け。品質タグとサンプラー設定はモデルの学習内容に依存するため流用しない。
  SD1.5向けの2つは負荷で使い分ける (`anime-soft` はsteps 20の下描き向け、
  `anime-detailed` はsteps 30・品質タグ厚めの仕上げ向け)
- **併用できない組み合わせがある。** hires fixとIPAdapter、DiT系モデルと
  LoRA / ControlNet / IPAdapterは指定するとその場で拒否される。
  hires fixとControlNetは併用できる (ControlNetが効くのは1段目だけ)。
  DiT系モデルはimg2img / hires fixと併用できる。
  一覧は [組み合わせの可否](docs/spec-reference.md#組み合わせの可否)
- **読める日本語が要求されたら生成に任せず `text` で合成する。**
  SD1.5 / SDXL系のモデルは日本語をほぼ描けない
- **「さっきの子で別の場面を」と言われたら基準画像を作り `reference` に指定する。**
  presetだけでは顔立ちまでは固定できない。手順は
  [.claude/skills/imagegen/references/character-consistency.md](.claude/skills/imagegen/references/character-consistency.md)

## 複数枚をまとめて生成する

同じSpecでseedを変えて何枚か出したい場合や、複数のSpecを流したい場合は `batch` を使う。

```bash
uv run imagegen batch specs/generated/a.yaml --seeds 111,222,333
uv run imagegen batch specs/generated/a.yaml specs/generated/b.yaml
```

- 1件失敗しても残りは続き、最後にサマリが出る
- Specの検証は実行前に全件行う。不正なSpecが混ざっていたら1件も生成しない
- CPU推論では枚数分だけ時間がかかる。`steps` と解像度を落としてから使う

## 禁止事項

- **`workflows/*.json` を勝手に書き換えない。** Workflowは人間がComfyUI GUIで作成しAPI形式で書き出す。
  既存テンプレートへ定型のノードを挟むだけの場合に限り、ユーザーの指示があれば機械的に組み立ててよいが、
  `object_info` での仕様確認・参照整合性の検査・実機での生成成功確認をすべて満たすこと
  (手順: [workflows/README.md](workflows/README.md))
- **ComfyUI workflowを実行時に組み立てない。** テンプレートは静的ファイルとして固定する
- **未知のcheckpointを勝手に使用しない。** ComfyUIに実在するファイル名だけを指定する
- **validationを迂回しない。** `validate` をスキップしたり、検証を緩めて通したりしない
- **巨大解像度・大量batchを実行しない。** CPU推論のため負荷が直接時間に跳ね返る
- **ComfyUI APIへCLI/Coreを迂回して直接curlしない。** 障害切り分けが崩れる

## 設計上守ること

- **GenerationSpecが内部API契約。** Claude Code固有形式やComfyUI固有JSONを層間インターフェースにしない
- **ComfyUI依存は `src/agentic_imagegen/adapters/comfyui/` に閉じ込める。** Domain / Service層はNode IDやHTTP仕様を知らない
- **Node IDとclass_typeのマッピングは1か所に集約する。** 定義は `adapters/comfyui/workflow.py` の `TXT2IMG_BINDING`
- **CLIはMCP導入後も残す。** ローカルデバッグ・CI・Integration Test・障害切り分けに使う
- **想像上の共通化を先行させない。** Backend抽象は2つ目のバックエンドを足す時点で確定させる
  ([Issue #31](https://github.com/Sylphy0052/agentic-imagegen/issues/31))

## 所要時間に注意する

CPU推論のため、生成パラメータの負荷が所要時間へ直接跳ね返る。
SD1.5 / 512x768 / 20 stepsの実測は **XPUで約135秒、CPUで約12分**。
条件別の実測値と `IMAGEGEN_TIMEOUT` の目安は
[docs/xpu-setup.mdの「所要時間とタイムアウトの目安」](docs/xpu-setup.md#所要時間とタイムアウトの目安)
を一次情報とする。

XPUが使える環境ではそちらを使う (手順: [docs/xpu-setup.md](docs/xpu-setup.md))。
`uv run imagegen health` の `Devices:` が `xpu:0` ならXPUで動いている。

## 開発時のルール

### 品質ゲート (commit前に全て通す)

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

カバレッジは80%以上を維持する (`uv run pytest --cov`)。

### テスト

- **Unit Testから実ComfyUIへ接続しない。** `httpx.MockTransport` とフィクスチャで代替する
- ComfyUIが必要なテストには `@pytest.mark.integration` を付ける。通常の `uv run pytest` ではskipされる
- Integration Testは低負荷設定 (512x512以下 / steps 2-4 / batch_size 1) にする。
  必要なモデルがComfyUIに無いケース (ControlNet / DiT系) は失敗ではなくskipする

### 実装方針

- 型ヒントは原則必須。`Any` の濫用を避ける
- 新機能はTDD (RED -> GREEN -> REFACTOR) で進める
- 巨大な単一モジュールを作らない。過剰な抽象化もしない

### ドキュメント

- パラメータの値域・既定値を足したときは `docs/spec-reference.md` を直す。
  README / CLAUDE.md / SKILL.mdへ同じ内容を転記しない
- 日本語の漢字・カナとASCIIの間に空白を入れない (OK: `Claude Code入門` / NG: `Claude Code 入門`)
- 絵文字を使わない

## ディレクトリ

| パス | 役割 |
| --- | --- |
| `src/agentic_imagegen/domain/` | GenerationSpec、検証規則、結果の型。外部依存を持たない |
| `src/agentic_imagegen/services/` | ユースケースの組み立て |
| `src/agentic_imagegen/workflows/` | Workflowテンプレートの読み込みとallowlist |
| `src/agentic_imagegen/adapters/comfyui/` | ComfyUI固有のHTTP / WebSocket / JSON形状 |
| `workflows/` | API形式のWorkflowテンプレート (人間が作成) |
| `presets/characters/` | キャラクタpreset (外見的特徴) |
| `presets/scenes/` | シーンpreset (場所・時間帯・構図) |
| `presets/styles/` | 画風preset (画風・品質タグ・サンプラー設定) |
| `specs/examples/` | サンプルSpec |
| `specs/generated/` | Claude Codeが生成したSpec (git管理外) |
| `inputs/` | img2imgの入力画像 (git管理外) |
| `fonts/` | テキスト合成に使うフォント (git管理外) |
| `outputs/` | 生成結果 (git管理外) |

## 環境変数

| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI接続先 |
| `IMAGEGEN_MAX_WIDTH` | 2048 | 幅の上限 |
| `IMAGEGEN_MAX_HEIGHT` | 2048 | 高さの上限 |
| `IMAGEGEN_MAX_PIXELS` | 4194304 | 総pixel数の上限 (batch込み) |
| `IMAGEGEN_MAX_BATCH` | 4 | batch_sizeの上限 (4が上限。超える値を設定すると起動時にエラーになる) |
| `IMAGEGEN_TIMEOUT` | 300 | 生成のタイムアウト秒 |
| `IMAGEGEN_OUTPUT_ROOT` | `outputs` | 出力ルート |
| `IMAGEGEN_PRESETS_ROOT` | `presets` | presetの探索ルート |
| `IMAGEGEN_MAX_SOURCE_BYTES` | 33554432 | img2imgの入力画像の上限バイト数 |
| `IMAGEGEN_FONTS_ROOT` | `fonts` | テキスト合成に使うフォントの探索ルート |

秘密情報は扱わないため、環境変数ファイルは必須ではない。

## exit code

失敗時は原因ごとに異なるexit codeを返す。自動化する場合はこれで分岐する。
code別の切り分け手順は
[.claude/skills/imagegen/references/troubleshooting.md](.claude/skills/imagegen/references/troubleshooting.md)
にある。

| code | 例外 | 意味 |
| --- | --- | --- |
| 0 | - | 成功 |
| 1 | - | 想定外の内部エラー |
| 2 | `InvalidGenerationSpec` | Specが不正 |
| 3 | `ComfyUIUnavailable` | ComfyUIへ到達できない |
| 4 | `WorkflowValidationError` | Workflowテンプレートが不正 |
| 5 | `WorkflowSubmissionError` | Workflowの投入が拒否された |
| 6 | `GenerationTimeout` | 生成がタイムアウトした |
| 7 | `GenerationFailed` | ComfyUI側で実行が失敗した |
| 8 | `OutputNotFound` | 出力画像が見つからない |
| 9 | `InvalidConfiguration` | 環境変数の設定値が不正 |
| 10 | `TextCompositionError` | テキスト合成に失敗した |
