# Phase 5: 日本語テキスト描画

画像内へ日本語の文字を入れるための方針と設計。

**この文書は方式選定の記録であり、現在の実装の仕様書ではない。**
現在の `text` / `compose` の仕様は
[docs/spec-reference.md](../spec-reference.md#text-テキスト合成) の `text` 節を参照。
`docs/plan/` の方針は [README.md](README.md) にある。

- 対象: 画像内への日本語テキスト描画の方式選定
- 作成日: 2026-08-12
- ステータス: 合成方式 (`text` / `compose`) を実装済み。Qwen-Imageは導入条件のみ記録し未着手

## 背景

Stable Diffusion 1.5 / SDXL系のモデルは日本語の文字をほぼ描けない。看板・タイトル・
台詞のように読める文字が必要な用途では、現在のパイプラインで要求を満たせない。

解決策は2つある。

1. 日本語のテキスト描画に対応したモデル (Qwen-Image系) を使う
2. 背景だけを既存モデルで生成し、日本語は生成後に画像処理で合成する

調査の結果、**1は現在の実行環境では動かせない**。したがって2を実装し、1は環境が整った
段階で着手できるよう設計だけ残す。

## Qwen-Imageの評価

### モデルの構成

Qwen-Imageは20BパラメータのMMDiTで、ComfyUIはネイティブ対応している。ただし
既存テンプレートのような単一checkpointではなく、3つのファイルを個別に読み込む。

| 役割 | ノード | 配置先 |
| --- | --- | --- |
| 拡散モデル本体 | `UNETLoader` (GGUF時は `UnetLoaderGGUF`) | `models/diffusion_models/` |
| テキストエンコーダ (Qwen2.5-VL-7B) | `CLIPLoader` (type=`qwen_image`) | `models/text_encoders/` |
| VAE | `VAELoader` | `models/vae/` |

latentは `EmptySD3LatentImage`、サンプリングはcfg 2.5前後・euler/simple・20 stepsが
目安。Qwen-Image-Lightning LoRAを併用すると8 stepsまで落とせる。

### 必要メモリ

| 構成 | 拡散モデル | テキストエンコーダ | 合計の目安 |
| --- | --- | --- | --- |
| fp8 | 約20GB | 約9GB | 約29GB |
| GGUF Q4_K_M | 約13.1GB | 約5GB (Q4) | 約18GB |
| GGUF Q3_K_S | 約10GB | 約5GB (Q4) | 約15GB |
| GGUF Q2_K | 約8GB | 約5GB (Q4) | 約13GB |

出典はComfyUI WikiのQwen-ImageガイドおよびUnslothのQwen-Image-2512ドキュメント。
「合計の目安」はVRAMとRAMを合わせた使用可能メモリに対する要求量である。

### 現環境との突き合わせ

計測時点 (2026-08-12) の実行環境は次のとおり。

| 項目 | 実測 |
| --- | --- |
| WSL2へ割り当てられたRAM | 13GB (空き4GB) |
| swap | 8GB (5GB使用済み) |
| GPU | Intel内蔵Arc (専用VRAMを持たず共有メモリで動作) |
| SD1.5 512x768 20 stepsの所要時間 | XPUで約135秒 |

最軽量のQ2_K構成でも要求は約13GBで、割当RAMとほぼ同値となり空きメモリでは
まったく足りない。swapへ落ちた状態で20Bのモデルを回すことになり、SD1.5で
135秒かかる環境である以上、1枚あたり数十分から数時間の水準になる。実用しうる
速度ではない。

### 導入の条件

次のいずれかを満たした時点で再評価する。

- WSL2へ32GB以上のRAMを割り当てられること (物理48GB以上が目安)
- VRAM 16GB以上の外部GPUを利用できること
- クラウドGPU上のComfyUIへ接続する構成へ移行すること

### 導入時に必要になる変更

執筆時点 (Phase 5) の設計は「1つのcheckpointを `CheckpointLoaderSimple` で読む」ことを
前提に `ModelSpec` とWorkflowテンプレートを組んでいた。Qwen系を扱うには次が要ると
見込んでいた。

- `ModelSpec` にモデル種別の概念を導入する。`checkpoint` 単一指定に加えて、
  `diffusion_model` / `text_encoder` / `vae` を個別に指定できる形へ拡張する。
  既存Specを壊さないため、種別は指定内容から判別できるようにする
- Workflowテンプレートを新規に用意する (`qwen_txt2img.json` ほか)。テンプレートは
  人間がComfyUI GUIで作成しAPI形式で書き出す規約であるため、自動生成はしない
- `workflows/injector.py` のバインディングにQwen用の定義を追加する。
  Node IDとclass_typeの対応は `adapters/comfyui/workflow.py` へ集約する規約を維持する
- 生成パラメータの既定値がモデル系列で異なる (cfg 2.5 / steps 8-20)。Spec側で
  明示指定する運用とし、暗黙のモデル別デフォルトは持たない

**追記 (Anima対応後):** 上記のうち `ModelSpec` の拡張とWorkflowテンプレートの仕組みは、
後続のDiT系モデル (Anima) 対応で実装済みになった。ただし当初の想定と異なり、フィールド名は
`diffusion_model` / `text_encoder` ではなく `unet` / `clip` / `vae` になっている
(`src/agentic_imagegen/domain/models.py` の `ModelSpec`)。`workflows/injector.py` の
`resolve_workflow_name` が `model.uses_separate_loaders` (= `unet` 指定の有無) を見て
`_unet` 接尾辞へ切り替える。テンプレートはimg2img / hires fixとの組み合わせを含め4本ある
(#39、`workflows/README.md` の一覧を参照)。

ただしこれはAnima向けに配線したものであり、そのままQwen-Imageへ転用はできない。
`workflows/txt2img_unet.json` の `CLIPLoader` は `type: stable_diffusion` 固定、latentノードも
`EmptyLatentImage` で、Qwen-Imageが必要とする `type: qwen_image` や `EmptySD3LatentImage`
とは異なる。Qwen固有で残っている作業は次のとおり。

- Qwen用のWorkflowテンプレートを別途用意する (`CLIPLoader type=qwen_image` /
  `EmptySD3LatentImage` を含む構成)
- `workflows/injector.py` にQwen用テンプレートを区別して選択する仕組みを足す
  (現状は `unet` 指定の有無しか見ておらず、どのDiT系モデルかで分岐しないため、
  AnimaとQwenを同時に扱うにはモデル系統の軸が要る)
- `adapters/comfyui/workflow.py` のNode IDとclass_typeの対応をQwen向けに追加する

この節は将来の作業の入口としてのみ残す。Qwen対応自体はPhase 5の実装対象に含めていない。

## テキスト合成の設計

### 方針

生成した画像に対し、Pillowで日本語テキストを描画する後処理を加える。文字の内容・
書体・位置を完全に制御できるため、看板やタイトルのように「読めること」が要件になる
用途では生成モデルに任せるより確実である。

利用経路は2つ用意する。内部の合成処理は共通のサービスとして1つだけ持つ。

- `imagegen generate` — Specに `text` があれば生成後に続けて合成する
- `imagegen compose` — 既存の画像ファイルへ後から合成する

### Specの拡張

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

フィールドの意味・既定値・値域は
[docs/spec-reference.md](../spec-reference.md#text-テキスト合成) の `text` 節を一次情報とする。
この文書では設計判断の理由だけを残す。

- **レイヤを配列にする。** 見出しと本文のように複数の文字列を別々の位置・大きさで置きたい
  要求があるため、1つのSpecへ複数レイヤを重ねられる形にする。指定順に描画し、
  後のものが上へ重なる
- **アンカーとオフセットで位置を決める。** 絶対座標だけだと解像度を変えたときに破綻する。
  9分割のアンカーを基準にし、そこからのずれをピクセルで足す形にする
- **`max_width` は1.0以下を比率として扱う。** 解像度を変えても折り返し幅の見た目が保たれる
- **`direction: vertical` はPillowが縦書きを持たないため1文字ずつ配置して実現する。**
  句読点や小書き文字の位置補正、ルビ、縦中横は対象外とする
  (後続の課題は [Issue #40](https://github.com/Sylphy0052/agentic-imagegen/issues/40))

### フォントの扱い

`fonts/` をフォントのルートとし、git管理外に置く。Specからはファイル名で参照する。
checkpointやLoRAと同じ規則であり、Specが環境依存の絶対パスを含まない状態を保てる。

- 探索ルートは環境変数 `IMAGEGEN_FONTS_ROOT` で変更できる (既定 `fonts`)
- 受け付ける拡張子は `.ttf` / `.otf` / `.ttc`
- ルート外を指す指定 (絶対パス、`..`、`~`) は `domain.models` の検証で拒否する。
  実体解決とルート外への脱出検証は `domain.policy` が担う。既存の
  `_validate_model_filename` / `resolve_source_image` と同じ構造にする
- `.ttc` はコレクション内の索引を `font_index` で指定できる
- 該当ファイルが無い場合は候補を列挙したエラーを返す。別のフォントへ暗黙に
  フォールバックしない。意図しない書体で出力されるより失敗させる方が扱いやすい

セットアップ手順はドキュメントで案内する。現在のWSL環境にはIPAGothicがあり、
Windows側にはBIZ UDGothicと游ゴシックがある。再配布可能なものとしては
Noto Sans JPを推奨する。

### 合成サービス

`src/agentic_imagegen/services/compose.py` を新設する。

```python
compose_text(
    *, image: Path, spec: TextSpec, fonts_root: Path, output: Path, max_pixels: int | None = None
) -> ComposeResult
```

- 入力画像は開いた後RGBAへ変換する
- レイヤごとに透明な画像へ描画し、`rotation` と `opacity` を適用してから重ねる。
  こうすると回転が他のレイヤへ影響しない
- 影は描画後に `ImageFilter.GaussianBlur` をかけてから本体の下へ敷く
- 保存時は元画像のモードへ戻す。JPEGのようにアルファを持てない形式ではRGBへ落とす

ドメインの型は `domain/models.py` へ `TextSpec` / `TextLayer` / `StrokeSpec` /
`ShadowSpec` / `BoxSpec` として置く。既存の `_StrictModel` を継承し、未知フィールドは
拒否する。

`services/generation.py` は生成完了後にこのサービスを呼ぶ。合成の有無で生成そのものの
挙動は変えない。

### 出力

合成前の画像を消さずに残す。文字だけ差し替えて作り直せる状態を保つため。

```text
outputs/2026-08-12/night_city/
  image_0001.png           # 生成そのままの画像
  image_0001_text.png      # テキスト合成後
  metadata.json
```

`metadata.json` には `text` を追加し、解決したフォントの実パスと合成後のファイル名を
記録する。レイヤ定義そのものは `spec` の中に含まれるため重複させない。同じSpecで結果が変わったときにフォント差し替えを
切り分けられるようにする。

### composeコマンド

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml -o outputs/caption.png
```

- テキスト定義のYAMLは `GenerationSpec` の `text` セクションと同じ構造にする。
  生成用Specをそのまま渡した場合も `text` 部分だけを読む
- `-o` を省略した場合は入力と同じディレクトリへ `<元の名前>_text.<元の拡張子>` として書き出す
- 入力画像はリポジトリ配下に限る。上限バイト数は `IMAGEGEN_MAX_SOURCE_BYTES` を流用する
- 既存ファイルは上書きしない

### エラー

exit code 10を `TextCompositionError` に割り当てる。フォント未検出、画像を開けない、
描画に失敗した場合に返す。Specの形式不正は従来どおり2 (`InvalidGenerationSpec`)。

### 依存

`pillow>=11.0` を必須依存へ追加する。テキスト合成はCLIの基本機能として提供するため、
optionalにはしない。

## 実装計画

1. `domain/models.py` へ `TextSpec` ほかの型を追加し、検証規則を書く
2. `domain/policy.py` へフォントの実体解決を追加する
3. `config.py` へ `IMAGEGEN_FONTS_ROOT` を追加する
4. `services/compose.py` を実装する (折り返し、アンカー計算、縁取り、影、背景ボックス、
   回転、縦書きの順に進める)
5. `services/generation.py` から合成を呼び、`metadata.json` へ記録する
6. `cli.py` へ `compose` コマンドを追加し、exit code 10を配線する
7. `services/mcp_tools.py` のSpec受け渡しを新フィールドへ追随させる
8. ドキュメントを更新する (CLAUDE.md、README、imagegen skill、`fonts/` のセットアップ手順)
9. サンプルSpecを `specs/examples/text_overlay.yaml` として追加する

## 受入基準

- `text` を持つSpecで `generate` を実行すると、生成画像と合成画像の両方が出力される
- `compose` コマンドで既存画像へ合成でき、元画像は変更されない
- 存在しないフォント名を指定するとexit code 10で失敗し、`fonts/` 配下の候補が示される
- `max_width` を指定した場合に、指定幅を超えない位置で折り返される
- `direction: vertical` で縦方向に1文字ずつ配置される
- レイヤ11件、`content` 501文字、`size` 513はいずれも検証で拒否される
- `metadata.json` に適用したテキスト定義と解決済みフォント情報が記録される
- 既存のSpec (text無し) の挙動と出力ファイル名が変わらない

## テスト方針

- 合成処理はComfyUIに依存しないため、すべてUnit Testで書ける。生成した画像の
  ピクセルを直接検証する
- テスト用フォントはリポジトリへ含めず、`fonts/` に対象が無い環境ではテストをskipする。
  検証ロジック (パス、上限、色形式) はフォント無しでも動くよう分離する
- アンカー計算・折り返し・色の解釈は純粋関数として切り出し、画像を描かずに検証する
- 回転と影は出力画像の外接矩形および非透明ピクセル数で確認する
- カバレッジ80% 以上を維持する

## 参考

- ComfyUI Wiki: Qwen-Image Native / GGUF / Nunchaku Workflow Guide
  <https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image>
- Unsloth: How to Run Qwen-Image-2512 Locally in ComfyUI
  <https://unsloth.ai/docs/models/tutorials/qwen-image-2512>
- ComfyUI公式: Qwen-Imageネイティブワークフローの例
  <https://docs.comfy.org/ja/tutorials/image/qwen/qwen-image>
- ComfyUIでLightningやDistillを利用したQwen-Imageの高速な生成
  <https://note.com/mayu_hiraizumi/n/n82dc413fc4c8>
