# readable-japanese-uses-text-layer

- skill: `imagegen`
- query: 「本日休業」と読める札を持った女の子の画像を作って
- 所要: 1.17 USD / 157秒

## 応答

## 判断

読める日本語 → 生成に任せず `text` 合成。SD1.5/SDXLは日本語描けない (CLAUDE.md「読める日本語が要求されたら生成に任せず`text`で合成する」)。札は白紙で生成 → 文字は後乗せ。

checkpoint/画風/解像度は要求文に無い → 手順3のヒアリング対象だが、対話不可 → SKILL.md「答えが返らない場合は推奨案で進め、何を既定にしたかを報告に添える」に従い既定で確定。既定 = `hassakuSD15_v13.safetensors` + `sd15-hassaku` + 512x768。

札の位置は生成後まで不明 → **2段構成**。1段目 txt2img (textなし) → 画像確認 → 2段目 `compose` で位置合わせ。1段目のSpecへ`text`を入れると位置ズレのたび再生成 (約200秒) が要る。

## 参照した文書

- `CLAUDE.md` — 手順、既定checkpoint、禁止事項、exit code
- `.claude/skills/imagegen/SKILL.md` — 入口判定、手順1-7、「生成済み画像へテキストだけ入れる」
- `docs/spec-reference.md` — `presets` / `generation` / `model` / `text` (layers・box・direction)、組み合わせの可否
- `docs/prompting-guide.md` — 全モデル共通の原則、SD1.5系 (75トークン・danbooruタグ)、配置済みのSD1.5系モデル、タグをブロックで組む
- `docs/fonts-setup.md` — フォント配置と exit 10
- `presets/styles/sd15-hassaku.yaml` — 実体確認 (negativeに`text, watermark, logo, signature`を既に含む)

## preset

- style: **`sd15-hassaku`** (`applies_to: hassakuSD15_v13.safetensors`。sampler `dpmpp_2m_sde` / scheduler `exponential` / cfg 7.0 / steps 30 / clip_skip 2 / 外部VAE `vaeKlF8Anime2_klF8Anime2VAE.safetensors` を保持 → Spec側は`model.checkpoint`だけ)
- character: **なし** (既存3件 `anime-girl-blue` / `anime-girl-trailrun` / `anime-boy-hooded` は要求と不一致 → Spec本体promptで記述)
- scene: **なし** (既存4件に店先なし → Spec本体promptで記述)

## Spec A — 生成用 `specs/generated/closed-today-sign.yaml`

```yaml
version: "1"
task: txt2img

presets:
  style: sd15-hassaku

prompt:
  # 札は白紙で出す。文字はtext合成で後から乗せるため、文字を示唆する語を入れない。
  # 主題 -> 属性 -> 構図 の順。品質タグはstyle preset側が末尾に付ける。
  positive: >
    1girl, solo, shop clerk, apron, long brown hair, light smile,
    holding sign, white sign, wooden sign, both hands, upper body,
    looking at viewer, storefront, daylight, simple background
  # negativeはstyle presetが text / watermark / logo / signature を持つので追加しない
  # (negativeは最小から始め、症状を見てから足す)

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: closed_today_sign
```

## Spec B — 合成用 `specs/generated/closed-today-caption.yaml`

```yaml
version: "1"

text:
  layers:
    - content: 本日休業
      direction: vertical
      font: ipag.ttf
      size: 44
      color: "#202020"
      anchor: center
      # 札が胸の前に来る想定。1枚目を見てoffset/rotation/sizeを詰める
      offset: [0, 96]
      line_spacing: 1.15
      # 札の面が多少崩れても読めるよう、文字の背後へ紙色の矩形を敷く
      box:
        color: "#f5f0e6"
        opacity: 0.92
        padding: [18, 22]
        radius: 4
```

## 実行するはずのコマンド

```bash
# 1. 在庫と実行基盤 (Devices/Backend、checkpoint、fonts) を確認
uv run imagegen catalog

# 2. 検証 (Workflow行 = txt2img_vae、Presets行 = sd15-hassaku、Estimate行を確認)
uv run imagegen validate specs/generated/closed-today-sign.yaml

# 3. 生成 (comfyui-session.sh経由。uv run imagegen generate は直接叩かない)
scripts/comfyui-session.sh generate specs/generated/closed-today-sign.yaml

# 4. 出力画像を確認後、札の位置へ文字を合成 (ComfyUI不要、元画像は変更されない)
uv run imagegen compose outputs/<YYYY-MM-DD>/<HHMMSS>_closed_today_sign/image_0001.png \
  specs/generated/closed-today-caption.yaml
```

## 注意点

- **worktreeに`fonts/`の実体が無い。** `fonts/` はgit管理外 → 本worktreeには`.gitkeep`のみ。実体 (`ipag.ttf` / `ipagp.ttf`) はメインリポジトリ側 `/home/kfuruhashi/workspace/github/agentic-imagegen/fonts/` にある。このままcomposeするとフォント未発見で **exit 10** (代替書体へフォールバックしない)。`IMAGEGEN_FONTS_ROOT=/home/kfuruhashi/workspace/github/agentic-imagegen/fonts` を渡すか、worktreeの`fonts/`へコピーしてから実行する
- Workflowは`model.vae`をstyle presetが持つため `txt2img_vae` に切り替わる (接尾辞規則)
- 所要時間: 512x768 / steps 30 → XPUで約200秒見込み。既定 `IMAGEGEN_TIMEOUT` 300は余裕が薄い → `validate` の `Estimate:` を見てから流す。CPUなら1200へ上げる
- タグ実在は未確認 (コマンド不可)。実行時は `python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "..."` で `white sign` / `wooden sign` / `shop clerk` を確認する。post_count 1000未満なら `sign` + `holding sign` へ寄せる
- 札を大きく写したいなら `upper body` を `cowboy shot` へ変える案があるが、札の面積が減るため未採用
