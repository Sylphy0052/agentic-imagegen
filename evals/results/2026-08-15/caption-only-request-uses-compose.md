# caption-only-request-uses-compose

- skill: `imagegen`
- query: inputs/base.png に「夏の記録」って文字を入れて
- 所要: 0.77 USD / 66秒

## 応答

## 判断

要求 = 「生成済み画像へ文字入れ」→ **生成しない**。`compose` 入口。SKILL.mdの要求種別表「この画像に文字を入れて」→ [生成済み画像へテキストだけ入れる](.claude/skills/imagegen/SKILL.md)。ComfyUI不要 → `scripts/comfyui-session.sh` 経由不要。

### 参照文書

- `.claude/skills/imagegen/SKILL.md` — 入口判定、compose手順
- `docs/spec-reference.md` #text (テキスト合成) — layers全キー、stroke/shadow/box、exit 10
- `specs/examples/text_overlay.yaml` — layers記述例
- `docs/fonts-setup.md` — フォント配置 (未読、案内先)
- `src/agentic_imagegen/cli.py:330` — compose引数、既定出力 `<stem>_text<suffix>`

### preset

**使用なし**。composeはcheckpoint/sampler無関係 → character/scene/style 3軸とも不要。

## ブロッカー2件

1. **`inputs/base.png` 不在**。`inputs/` は `.gitkeep` のみ
2. **`fonts/` 空**。`.gitkeep` のみ → `font:` に書ける実ファイル無し。見つからないフォントは代替せず失敗 (exit code 10)。`docs/fonts-setup.md` 手順でNoto Sans JP等を配置要

## 実行するはずのコマンド

```bash
# 1. 在庫確認 (fonts が (なし) でないか)
uv run imagegen catalog

# 2. 合成 (入力画像は不変、outputs は inputs/base_text.png)
uv run imagegen compose inputs/base.png specs/generated/summer-caption.yaml
```

`validate` は生成Spec用 → composeには無い。スキーマ検証は `load_text_spec` が実行時に行う。

## Spec (保存先想定: `specs/generated/summer-caption.yaml`)

フォント名は `NotoSansJP-Bold.ttf` を仮置き。`catalog` の `fonts` に出た実名へ差し替え必須。

```yaml
# 既存画像へ「夏の記録」を合成する。生成は行わない。
version: "1"

text:
  layers:
    - content: 夏の記録
      font: NotoSansJP-Bold.ttf
      size: 72
      color: "#ffffff"
      anchor: top-center
      offset: [0, 48]
      stroke:
        width: 4
        color: "#20303a"
      shadow:
        offset: [0, 6]
        blur: 8
        color: "#000000"
        opacity: 0.6
```

`anchor: top-center` + stroke + shadow は、背景不明の写真へ載せても読める既定として選択 (spec-reference の描画順 box→shadow→stroke+text)。

## 質問

文字の見せ方が要求文から決まらない。以下1問:

**「夏の記録」の配置と装飾、どれにする？**

- **A: 上部中央・大きめ (size 72)・白+濃い縁取り+影** — 見出し扱い。背景が明るくても暗くても読める。上のSpecがこれ (推奨)
- **B: 下部中央・小さめ (size 40)・半透明の黒帯 (`box`) 敷き** — キャプション扱い。被写体を隠しにくい
- **C: 右上・縦書き (`direction: vertical`)・白+縁取り** — 和風の見出し。人物の顔位置次第で `anchor` 調整要

併せて `inputs/base.png` の配置と、`fonts/` へのフォント配置 (`docs/fonts-setup.md`) が要る。
