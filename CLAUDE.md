# CLAUDE.md

Claude Codeがこのリポジトリを操作するときのルール。

## このプロジェクトは何か

AIコーディングエージェントから、ComfyUI経由でStable Diffusion系モデルによる画像生成を実行する基盤。

```text
Claude Code -> GenerationSpec -> Python CLI (imagegen) -> ComfyUI API -> 画像生成
```

Phase 1 (txt2img) は完了。現在はPhase 2 (preset / LoRA / img2img) を実装中。
Phase 1の設計は [docs/plan/phase1.md](docs/plan/phase1.md)、
進捗は [Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1) と
[Issue #3](https://github.com/Sylphy0052/agentic-imagegen/issues/3) を参照。

## 画像生成要求を受けたときの手順

詳細な手順とpresetの選び方は [.claude/skills/imagegen/SKILL.md](.claude/skills/imagegen/SKILL.md)
にある。失敗時の切り分けは
[.claude/skills/imagegen/references/troubleshooting.md](.claude/skills/imagegen/references/troubleshooting.md)
を参照する。

ユーザーから「〇〇な画像を生成して」と指示された場合、次の順で実行する。

1. **GenerationSpecを作る** — 自然言語の要求をSpecの各フィールドへ落とし込む
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

Specの書き方は [specs/examples/txt2img.yaml](specs/examples/txt2img.yaml)、
preset・LoRAを使う場合は
[specs/examples/txt2img_preset_lora.yaml](specs/examples/txt2img_preset_lora.yaml) を参照。

## Presetを使う

繰り返し使う指定は preset にまとめてSpecから名前で参照する。軸は3つで、
1軸につき1つまで指定できる。

| 軸 | 置き場 | 書く内容 |
| --- | --- | --- |
| `character` | `presets/characters/<name>.yaml` | 人物の外見的特徴 |
| `scene` | `presets/scenes/<name>.yaml` | 場所・時間帯・構図 |
| `style` | `presets/styles/<name>.yaml` | 画風・品質タグ・サンプラー設定 |

```yaml
presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: anime-soft
```

解決規則は次のとおり。

- **prompt**: `character` -> `scene` -> `style` -> Spec本体 の順にカンマ連結し、
  重複トークンは最初の1つを残して除去する (大文字小文字と余分な空白は無視)。negativeも同じ
- **generation**: presetの指定を取り込んだうえで、Spec本体の指定を優先する
  (優先順位は spec > style > scene > character)
- 適用したpreset名は解決後のSpecに残り、`metadata.json` にも記録される

新しいpresetを作るときは軸の責務を混ぜない。解像度とseedは再現性に直結するため
presetには書かず、Spec側で指定する。

## LoRAを使う

`model.loras` に指定する。同時に3件まで。

```yaml
model:
  checkpoint: meinamix_v12Final.safetensors
  loras:
    - name: add_detail.safetensors
      strength_model: 0.8
      strength_clip: 0.8
```

- LoRAを指定すると、Workflowテンプレートが `txt2img_lora` へ自動的に切り替わる
  (`uv run imagegen validate` の `Workflow:` 行で確認できる)
- `strength_model` / `strength_clip` は省略時1.0、範囲は ±10.0
- 同じLoRAを重複指定できない。二重に積むと意図しない強度になるため
- 拡張子は `.safetensors` / `.pt` / `.ckpt`
- 配置先は `~/ComfyUI/models/loras/`。実在しない名前を指定するとComfyUI側で拒否される

## img2imgを使う

`task: img2img` と `source` を指定する。入力画像はリポジトリ配下に置く。

```yaml
task: img2img

source:
  image: inputs/reference.png
  denoise: 0.55            # 0に近いほど入力画像を保ち、1に近いほど描き直す
```

- 入力画像は生成前にComfyUIへ自動でアップロードされる。`~/ComfyUI/input/` へ手で置く必要はない
- **解像度は入力画像のサイズをそのまま使う。** `width` / `height` を書くと拒否される
  (書いたのに効かない状態を作らないため)
- `batch_size` は1のみ。LoRAは併用できる (`img2img_lora` テンプレートへ切り替わる)
- 入力画像は `inputs/` へ置く (git管理外)。拡張子は `.png` / `.jpg` / `.jpeg` / `.webp`
- 上限サイズは `IMAGEGEN_MAX_SOURCE_BYTES` (既定32MiB)

## 構図を指定する (ControlNet)

参考画像から線画 (Canny) を取り、その構図を保ったまま生成する。

```yaml
control:
  image: inputs/pose.png                              # リポジトリ配下に置く
  model: control_v11p_sd15_canny_fp16.safetensors     # ~/ComfyUI/models/controlnet/
  strength: 0.9        # 効かせる強さ (0.0-10.0)
  start_percent: 0.0   # 効かせ始める進行度
  end_percent: 1.0     # 効かせ終える進行度。構図だけ借りるなら下げる
  low_threshold: 0.3   # Cannyの閾値。低いほど細かい線を拾う
  high_threshold: 0.7
```

- 指定するとテンプレートが `*_controlnet` へ自動的に切り替わる。txt2img / img2img の両方で使える
- control画像は生成前にComfyUIへ自動でアップロードされる
- **前処理は Canny のみ。** pose / depth はpreprocessorのカスタムノードが要るため未対応
- `upscale` との同時指定は未対応 (両方かけると生成時間が現実的でないため)

線が強く出すぎる場合は `low_threshold` を上げて細かい線を捨てるか、`strength` を下げる。
写真やイラストをそのまま渡すと輪郭を拾いすぎ、元絵のエッジが残ったような絵になりやすい。

## 参照画像から特徴を引き継ぐ (IPAdapter)

参照画像をCLIP Visionで読み、その特徴 (人物の顔立ち・服装・画風) を効かせたまま生成する。
プロンプトだけでは揺れる要素を固定できるため、同一キャラクタを別の構図で出すときに使う。

```yaml
reference:
  image: inputs/character.png                            # リポジトリ配下に置く
  model: ip-adapter-plus_sd15.safetensors                # ~/ComfyUI/models/ipadapter/
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors  # ~/ComfyUI/models/clip_vision/
  weight: 0.8          # 効かせる強さ (0.0-3.0)
  weight_type: linear  # 効かせ方 (style transfer / composition など15種)
  start_percent: 0.0   # 効かせ始める進行度
  end_percent: 1.0     # 効かせ終える進行度
```

- 指定するとテンプレートが `*_ipadapter` へ自動的に切り替わる。txt2img / img2img の両方で使える
- 参照画像は生成前にComfyUIへ自動でアップロードされる
- **ComfyUI_IPAdapter_plus (カスタムノード) が要る。** 未導入だとノードが無く投入が拒否される
- ControlNetと併用できる (`*_controlnet_ipadapter`)。構図をControlNet、特徴をIPAdapterが担う
- `upscale` との同時指定は未対応 (ControlNetと同じ理由)
- モデルとCLIP Visionは対応関係がある。`ip-adapter-plus_sd15` には ViT-H を使う

`weight` は0.6-0.9が扱いやすい。1.0を超えると参照画像へ寄りすぎ、プロンプトが効かなくなる。
顔立ちだけ借りて服装や背景はプロンプトへ従わせたい場合は `weight_type: style transfer` を使う。

## 解像度を上げる (hires fix)

`generation.upscale` を指定すると、1段目の結果をlatentのまま拡大し、2段目のKSamplerで
描き足す。アップスケールモデルは不要。

```yaml
generation:
  width: 512
  height: 768
  steps: 20
  upscale:
    scale: 1.5        # 1.0より大きく4.0以下
    denoise: 0.45     # 低いほど元の絵を保つ。0.3-0.5が扱いやすい
    steps: 8          # 省略時は1段目と同じ
```

- 指定するとテンプレートが `*_hires` へ自動的に切り替わる
- **生成時間は倍以上になる。** 2段目は拡大後の解像度で走るため、1stepあたりの時間も伸びる
- 2段目のseedは1段目と同じ値を使う (変えると元の絵から離れる)
- 最初から大きい解像度で生成するより、hires fix の方が構図が破綻しにくい

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
- **Phase 1でMCPのためだけの抽象層を作らない。**

## 生成パラメータの目安

CPU推論のため、既定は控えめにする。

| 項目 | 推奨 | 上限 |
| --- | --- | --- |
| 解像度 | 512x512 / 512x768 | `IMAGEGEN_MAX_WIDTH` / `IMAGEGEN_MAX_HEIGHT` (既定2048) |
| steps | 20前後 | 100 |
| cfg | 5.0-8.0 | 30 |
| batch_size | 1 | 4 |

**所要時間に注意する。** SD1.5 / 512x768 / 20 steps の実測は次のとおり。

| 実行基盤 | 実測 |
| --- | --- |
| Intel XPU (内蔵Arc GPU) | 約135秒 |
| CPU | 約12分 |

XPUが使える環境ではそちらを使う (手順: [docs/xpu-setup.md](docs/xpu-setup.md))。
`uv run imagegen health` の `Devices:` が `xpu:0` ならXPUで動いている。
生成時は `IMAGEGEN_TIMEOUT` を十分に取る (XPUなら300、CPUなら1200が目安)。
SDXL / Illustrious系 (`novaAnimeXL_ilV190.safetensors`) はさらに遅く、常用しない。

seedに `-1` を指定するとランダムな値へ解決され、実際に使われた値が `metadata.json` に記録される。
同じ画を再現したい場合は、その値をSpecへ書き戻す。

## metadata.json

生成結果と同じディレクトリへ出力する。再現に必要な情報をここへ集約する。

| キー | 内容 |
| --- | --- |
| `prompt_id` | ComfyUI側の実行ID |
| `workflow` | 使用したworkflow名 |
| `workflow_hash` | Workflowテンプレートのダイジェスト (`sha256:...`) |
| `created_at` | 生成時刻 (タイムゾーン付き) |
| `resolved_seed` | 実際に使われたseed |
| `backend` | 実行基盤 (`comfyui_version` / `devices`)。取得に失敗した場合は `null` |
| `spec` | preset展開後のSpec全体。適用したpreset名も含む |
| `outputs` | 出力ファイル名 |

`workflow_hash` は正規化したJSONから取るため、インデントや鍵の順序が変わっただけでは動かない。
同じSpecで結果が変わったときに、テンプレート自体が変わったのかを切り分けられる。

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
- Integration Testは低負荷設定 (512x512 / steps 2-4 / batch_size 1) にする

### 実装方針

- 型ヒントは原則必須。`Any` の濫用を避ける
- 新機能はTDD (RED -> GREEN -> REFACTOR) で進める
- 巨大な単一モジュールを作らない。過剰な抽象化もしない

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
| `outputs/` | 生成結果 (git管理外) |

## 環境変数

| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | ComfyUI接続先 |
| `IMAGEGEN_MAX_WIDTH` | 2048 | 幅の上限 |
| `IMAGEGEN_MAX_HEIGHT` | 2048 | 高さの上限 |
| `IMAGEGEN_MAX_PIXELS` | 4194304 | 総pixel数の上限 (batch込み) |
| `IMAGEGEN_MAX_BATCH` | 4 | batch_sizeの上限 |
| `IMAGEGEN_TIMEOUT` | 300 | 生成のタイムアウト秒 |
| `IMAGEGEN_OUTPUT_ROOT` | `outputs` | 出力ルート |
| `IMAGEGEN_PRESETS_ROOT` | `presets` | presetの探索ルート |
| `IMAGEGEN_MAX_SOURCE_BYTES` | 33554432 | img2imgの入力画像の上限バイト数 |

## exit code

失敗時は原因ごとに異なるexit codeを返す。自動化する場合はこれで分岐する。

| code | 意味 |
| --- | --- |
| 0 | 成功 |
| 1 | 想定外の内部エラー |
| 2 | Specが不正 (`InvalidGenerationSpec`) |
| 3 | ComfyUIへ到達できない (`ComfyUIUnavailable`) |
| 4 | Workflowテンプレートが不正 (`WorkflowValidationError`) |
| 5 | Workflowの投入が拒否された (`WorkflowSubmissionError`) |
| 6 | 生成がタイムアウトした (`GenerationTimeout`) |
| 7 | ComfyUI側で実行が失敗した (`GenerationFailed`) |
| 8 | 出力画像が見つからない (`OutputNotFound`) |
| 9 | 環境変数の設定値が不正 (`InvalidConfiguration`) |
