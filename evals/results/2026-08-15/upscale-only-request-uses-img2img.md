# upscale-only-request-uses-img2img

- skill: `imagegen`
- query: この前作った画像をもっと大きくして
- 所要: 0.49 USD / 44秒

## 応答

判定: 「大きくして」→生成し直しではなく**生成済み画像を大きくする**入口 (img2img + upscale)。専用の拡大コマンドは無い。

参照文書:
- `.claude/skills/imagegen/SKILL.md` — 「生成以外の要求 / 生成済み画像を大きくする」「過去の生成とキャラクタを照会する」
- `CLAUDE.md` — 「使える機能と参照先」「所要時間に注意する」
- `docs/spec-reference.md` — `source` (img2img) / `generation.upscale` (hires fix) / 組み合わせの可否

## 1. 対象特定 (推測禁止、記録から引く)

「この前作った画像」= 記憶で当てない。実行するコマンド:

```bash
uv run imagegen history --limit 5
```

ここで出る **出力パス・resolved_seed・checkpoint・preset** を使う。台帳登録済みキャラなら:

```bash
uv run imagegen character list
uv run imagegen character show <name>
```

対象画像を `inputs/` へ置く (ComfyUIの `input/` へ手置き禁止)。

## 2. Spec (保存せず本文へ)

保存先想定: `specs/generated/upscale-prev.yaml`

```yaml
version: "1"
task: img2img

source:
  image: inputs/prev.png
  denoise: 0.35

generation:
  upscale:
    scale: 2.0
    denoise: 0.4
    steps: 12
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

presets:
  style: sd15-hassaku

output:
  prefix: prev_upscaled
```

preset名: **`sd15-hassaku`** (style軸)、checkpoint `hassakuSD15_v13.safetensors`。
ただしこれは**既定値のプレースホルダ**。手順1の`history`/`character show`で出た元画像のcheckpoint・style presetへ差し替える → 揃えないと拡大ついでに絵柄が変わる。character/scene presetも元画像が使っていればそのまま写す。

Spec上の判断:
- `width`/`height` は書かない (img2imgは入力画像サイズを使う→書くと拒否、exit 2)
- `source.denoise` 0.35 (0.3-0.45域) → 元の絵を保つ
- IPAdapter (`reference`) は書かない。hires fixと併用不可

## 3. validate

```bash
uv run imagegen validate specs/generated/upscale-prev.yaml
```

確認点: `Presets:` に元画像と同じstyle preset、`warning:` ゼロ、拡大後pixel数が `IMAGEGEN_MAX_UPSCALED_PIXELS` (16777216) 内。img2imgは `Estimate:` 出ない → 所要時間は元生成の倍以上と見積る。

## 4. generate

```bash
scripts/comfyui-session.sh generate specs/generated/upscale-prev.yaml
```

直叩き禁止 (XPUアロケータ断片化→exit 7)。

## 質問 (対話不可のためここで終了)

倍率が要求文から決まらない。`history`で対象は引けるが倍率は引けない → 推測仮置きせず確認する。

**倍率をどれにするか**

- **A: 2.0倍** (推奨) — 512x768→1024x1536。上限内、所要時間の伸びが実用域
- **B: 1.5倍** — 768x1152。時間を抑えたいとき
- **C: 2.5-3.0倍** — 元サイズ次第で `IMAGEGEN_MAX_UPSCALED_PIXELS` に当たる。validateで要確認

回答なしならAで進め、既定にした旨を報告へ添える。
