# Phase 5: 日本語テキスト描画

画像内へ日本語の文字を入れるための方針と設計。

## 背景

Stable Diffusion 1.5 / SDXL 系のモデルは日本語の文字をほぼ描けない。看板・タイトル・
台詞のように読める文字が必要な用途では、現在のパイプラインで要求を満たせない。

解決策は2つある。

1. 日本語のテキスト描画に対応したモデル (Qwen-Image 系) を使う
2. 背景だけを既存モデルで生成し、日本語は生成後に画像処理で合成する

調査の結果、**1は現在の実行環境では動かせない**。したがって2を実装し、1は環境が整った
段階で着手できるよう設計だけ残す。

## Qwen-Image の評価

### モデルの構成

Qwen-Image は 20B パラメータの MMDiT で、ComfyUI はネイティブ対応している。ただし
既存テンプレートのような単一 checkpoint ではなく、3つのファイルを個別に読み込む。

| 役割 | ノード | 配置先 |
| --- | --- | --- |
| 拡散モデル本体 | `UNETLoader` (GGUF時は `UnetLoaderGGUF`) | `models/diffusion_models/` |
| テキストエンコーダ (Qwen2.5-VL-7B) | `CLIPLoader` (type=`qwen_image`) | `models/text_encoders/` |
| VAE | `VAELoader` | `models/vae/` |

latent は `EmptySD3LatentImage`、サンプリングは cfg 2.5 前後・euler/simple・20 steps が
目安。Qwen-Image-Lightning LoRA を併用すると 8 steps まで落とせる。

### 必要メモリ

| 構成 | 拡散モデル | テキストエンコーダ | 合計の目安 |
| --- | --- | --- | --- |
| fp8 | 約20GB | 約9GB | 約29GB |
| GGUF Q4_K_M | 約13.1GB | 約5GB (Q4) | 約18GB |
| GGUF Q3_K_S | 約10GB | 約5GB (Q4) | 約15GB |
| GGUF Q2_K | 約8GB | 約5GB (Q4) | 約13GB |

出典は ComfyUI Wiki の Qwen-Image ガイドおよび Unsloth の Qwen-Image-2512 ドキュメント。
「合計の目安」は VRAM と RAM を合わせた使用可能メモリに対する要求量である。

### 現環境との突き合わせ

計測時点 (2026-08-12) の実行環境は次のとおり。

| 項目 | 実測 |
| --- | --- |
| WSL2 へ割り当てられた RAM | 13GB (空き4GB) |
| swap | 8GB (5GB 使用済み) |
| GPU | Intel 内蔵 Arc (専用 VRAM を持たず共有メモリで動作) |
| SD1.5 512x768 20 steps の所要時間 | XPU で約135秒 |

最軽量の Q2_K 構成でも要求は約13GB で、割当 RAM とほぼ同値となり空きメモリでは
まったく足りない。swap へ落ちた状態で 20B のモデルを回すことになり、SD1.5 で
135秒かかる環境である以上、1枚あたり数十分から数時間の水準になる。実用しうる
速度ではない。

### 導入の条件

次のいずれかを満たした時点で再評価する。

- WSL2 へ 32GB 以上の RAM を割り当てられること (物理 48GB 以上が目安)
- VRAM 16GB 以上の外部 GPU を利用できること
- クラウド GPU 上の ComfyUI へ接続する構成へ移行すること

### 導入時に必要になる変更

現在の設計は「1つの checkpoint を `CheckpointLoaderSimple` で読む」ことを前提に
`ModelSpec` と Workflow テンプレートを組んでいる。Qwen 系を扱うには次が要る。

- `ModelSpec` にモデル種別の概念を導入する。`checkpoint` 単一指定に加えて、
  `diffusion_model` / `text_encoder` / `vae` を個別に指定できる形へ拡張する。
  既存 Spec を壊さないため、種別は指定内容から判別できるようにする
- Workflow テンプレートを新規に用意する (`qwen_txt2img.json` ほか)。テンプレートは
  人間が ComfyUI GUI で作成し API 形式で書き出す規約であるため、自動生成はしない
- `workflows/injector.py` のバインディングに Qwen 用の定義を追加する。
  Node ID と class_type の対応は `adapters/comfyui/workflow.py` へ集約する規約を維持する
- 生成パラメータの既定値がモデル系列で異なる (cfg 2.5 / steps 8-20)。Spec 側で
  明示指定する運用とし、暗黙のモデル別デフォルトは持たない

この節は将来の作業の入口としてのみ残す。Phase 5 の実装対象には含めない。

## テキスト合成の設計

### 方針

生成した画像に対し、Pillow で日本語テキストを描画する後処理を加える。文字の内容・
書体・位置を完全に制御できるため、看板やタイトルのように「読めること」が要件になる
用途では生成モデルに任せるより確実である。

利用経路は2つ用意する。内部の合成処理は共通のサービスとして1つだけ持つ。

- `imagegen generate` — Spec に `text` があれば生成後に続けて合成する
- `imagegen compose` — 既存の画像ファイルへ後から合成する

### Spec の拡張

`GenerationSpec` へ `text` セクションを追加する。省略時は合成を行わない。

```yaml
text:
  layers:
    - content: "秋葉原駅"
      font: NotoSansJP-Bold.ttf
      size: 64
      color: "#ffffff"
      anchor: bottom-center
      offset: [0, -48]
      max_width: 0.8
      line_spacing: 1.2
      align: center
      opacity: 1.0
      rotation: -5.0
      direction: horizontal
      stroke:
        width: 3
        color: "#000000"
      shadow:
        offset: [4, 4]
        blur: 6
        color: "#000000"
        opacity: 0.5
      box:
        color: "#000000"
        opacity: 0.6
        padding: [16, 24]
        radius: 12
```

各フィールドの意味は次のとおり。

| フィールド | 既定値 | 内容 |
| --- | --- | --- |
| `content` | 必須 | 描画する文字列。改行を含められる |
| `font` | 必須 | `fonts/` 配下のファイル名 |
| `size` | 必須 | 字送りの基準となるピクセル数 |
| `color` | `#ffffff` | 文字色。`#rgb` / `#rrggbb` / `#rrggbbaa` |
| `anchor` | `center` | 9分割の基準位置 |
| `offset` | `[0, 0]` | `anchor` からのピクセル単位のずれ |
| `max_width` | なし | 折り返し幅。1.0以下は画像幅に対する比率、1.0超はピクセル |
| `line_spacing` | `1.2` | 行送りの倍率 |
| `align` | `center` | 複数行の揃え (`left` / `center` / `right`) |
| `opacity` | `1.0` | レイヤ全体の不透明度 |
| `rotation` | `0.0` | 回転角 (度、反時計回り) |
| `direction` | `horizontal` | `horizontal` / `vertical` |
| `stroke` | なし | 縁取り |
| `shadow` | なし | 影 |
| `box` | なし | 文字の背後へ敷く矩形 |

制約は次のとおり。

- レイヤは最大10件。順に描画し、後のものが上へ重なる
- `content` は1レイヤあたり最大500文字
- `size` は 1 以上 512 以下
- `rotation` は -180.0 以上 180.0 以下
- `offset` (レイヤ本体・影とも) は各成分 `-MAX_DIMENSION` 以上 `MAX_DIMENSION` 以下
  (解像度のハード上限と同じ8192)
- `max_width` の上限も `MAX_DIMENSION` (8192)。比率指定 (1.0以下) には影響しない
- `direction: vertical` は Pillow が縦書きを持たないため1文字ずつ配置して実現する。
  句読点や小書き文字の位置補正、ルビ、縦中横は対象外とする

### フォントの扱い

`fonts/` をフォントのルートとし、git 管理外に置く。Spec からはファイル名で参照する。
checkpoint や LoRA と同じ規則であり、Spec が環境依存の絶対パスを含まない状態を保てる。

- 探索ルートは環境変数 `IMAGEGEN_FONTS_ROOT` で変更できる (既定 `fonts`)
- 受け付ける拡張子は `.ttf` / `.otf` / `.ttc`
- ルート外を指す指定 (絶対パス、`..`、`~`) は `domain.models` の検証で拒否する。
  実体解決とルート外への脱出検証は `domain.policy` が担う。既存の
  `_validate_model_filename` / `resolve_source_image` と同じ構造にする
- `.ttc` はコレクション内の索引を `font_index` で指定できる (既定 0)
- 該当ファイルが無い場合は候補を列挙したエラーを返す。別のフォントへ暗黙に
  フォールバックしない。意図しない書体で出力されるより失敗させる方が扱いやすい

セットアップ手順はドキュメントで案内する。現在の WSL 環境には IPAGothic があり、
Windows 側には BIZ UDGothic と游ゴシックがある。再配布可能なものとしては
Noto Sans JP を推奨する。

### 合成サービス

`src/agentic_imagegen/services/compose.py` を新設する。

```
compose_text(
    *, image: Path, spec: TextSpec, fonts_root: Path, output: Path, max_pixels: int | None = None
) -> ComposeResult
```

- 入力画像は開いた後 RGBA へ変換する
- レイヤごとに透明な画像へ描画し、`rotation` と `opacity` を適用してから重ねる。
  こうすると回転が他のレイヤへ影響しない
- 影は描画後に `ImageFilter.GaussianBlur` をかけてから本体の下へ敷く
- 保存時は元画像のモードへ戻す。JPEG のようにアルファを持てない形式では RGB へ落とす

ドメインの型は `domain/models.py` へ `TextSpec` / `TextLayer` / `StrokeSpec` /
`ShadowSpec` / `BoxSpec` として置く。既存の `_StrictModel` を継承し、未知フィールドは
拒否する。

`services/generation.py` は生成完了後にこのサービスを呼ぶ。合成の有無で生成そのものの
挙動は変えない。

### 出力

合成前の画像を消さずに残す。文字だけ差し替えて作り直せる状態を保つため。

```
outputs/2026-08-12/night_city/
  image_0001.png           # 生成そのままの画像
  image_0001_text.png      # テキスト合成後
  metadata.json
```

`metadata.json` には `text` を追加し、解決したフォントの実パスと合成後のファイル名を
記録する。レイヤ定義そのものは `spec` の中に含まれるため重複させない。同じ Spec で結果が変わったときにフォント差し替えを
切り分けられるようにする。

### compose コマンド

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml -o outputs/caption.png
```

- テキスト定義の YAML は `GenerationSpec` の `text` セクションと同じ構造にする。
  生成用 Spec をそのまま渡した場合も `text` 部分だけを読む
- `-o` を省略した場合は入力と同じディレクトリへ `<元の名前>_text.<元の拡張子>` として書き出す
- 入力画像はリポジトリ配下に限る。上限バイト数は `IMAGEGEN_MAX_SOURCE_BYTES` を流用する
- 既存ファイルは上書きしない

### エラー

exit code 10 を `TextCompositionError` に割り当てる。フォント未検出、画像を開けない、
描画に失敗した場合に返す。Spec の形式不正は従来どおり 2 (`InvalidGenerationSpec`)。

### 依存

`pillow>=11.0` を必須依存へ追加する。テキスト合成は CLI の基本機能として提供するため、
optional にはしない。

## 実装計画

1. `domain/models.py` へ `TextSpec` ほかの型を追加し、検証規則を書く
2. `domain/policy.py` へフォントの実体解決を追加する
3. `config.py` へ `IMAGEGEN_FONTS_ROOT` を追加する
4. `services/compose.py` を実装する (折り返し、アンカー計算、縁取り、影、背景ボックス、
   回転、縦書きの順に進める)
5. `services/generation.py` から合成を呼び、`metadata.json` へ記録する
6. `cli.py` へ `compose` コマンドを追加し、exit code 10 を配線する
7. `services/mcp_tools.py` の Spec 受け渡しを新フィールドへ追随させる
8. ドキュメントを更新する (CLAUDE.md、README、imagegen skill、`fonts/` のセットアップ手順)
9. サンプル Spec を `specs/examples/text_overlay.yaml` として追加する

## 受入基準

- `text` を持つ Spec で `generate` を実行すると、生成画像と合成画像の両方が出力される
- `compose` コマンドで既存画像へ合成でき、元画像は変更されない
- 存在しないフォント名を指定すると exit code 10 で失敗し、`fonts/` 配下の候補が示される
- `max_width` を指定した場合に、指定幅を超えない位置で折り返される
- `direction: vertical` で縦方向に1文字ずつ配置される
- レイヤ11件、`content` 501文字、`size` 513 はいずれも検証で拒否される
- `metadata.json` に適用したテキスト定義と解決済みフォント情報が記録される
- 既存の Spec (text 無し) の挙動と出力ファイル名が変わらない

## テスト方針

- 合成処理は ComfyUI に依存しないため、すべて Unit Test で書ける。生成した画像の
  ピクセルを直接検証する
- テスト用フォントはリポジトリへ含めず、`fonts/` に対象が無い環境ではテストを skip する。
  検証ロジック (パス、上限、色形式) はフォント無しでも動くよう分離する
- アンカー計算・折り返し・色の解釈は純粋関数として切り出し、画像を描かずに検証する
- 回転と影は出力画像の外接矩形および非透明ピクセル数で確認する
- カバレッジ 80% 以上を維持する

## 参考

- ComfyUI Wiki: Qwen-Image Native / GGUF / Nunchaku Workflow Guide
  <https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image>
- Unsloth: How to Run Qwen-Image-2512 Locally in ComfyUI
  <https://unsloth.ai/docs/models/tutorials/qwen-image-2512>
- ComfyUI 公式: Qwen-Image ネイティブワークフローの例
  <https://docs.comfy.org/ja/tutorials/image/qwen/qwen-image>
- ComfyUI で Lightning や Distill を利用した Qwen-Image の高速な生成
  <https://note.com/mayu_hiraizumi/n/n82dc413fc4c8>
