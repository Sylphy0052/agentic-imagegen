# img2img-omits-resolution

- skill: `imagegen`
- query: inputs/base.png を下絵にして、水彩っぽく描き直して
- 所要: 0.75 USD / 94秒

## 応答

## 判定

img2img (`inputs/base.png` を下絵、水彩へ描き直し)。**ブロッカー1件あり → 末尾で質問。**

## 参照した文書

- `.claude/skills/imagegen/SKILL.md` — 手順、img2imgの注意 (`width`/`height`書かない)
- `docs/spec-reference.md` — `source` (L453-473、`denoise`既定0.6)、`presets`
- `presets/styles/sd15-hassaku.yaml` — 既定style presetの中身 (sampler/cfg/steps/clip_skip/VAE内蔵)
- `CLAUDE.md` — 既定checkpoint、生成は`comfyui-session.sh`経由
- `.claude/skills/prompt-builder/SKILL.md` — タグ実在確認 (`tagcheck.py`)

## preset名

- style: **`sd15-hassaku`** (checkpoint未決時の既定、`hassakuSD15_v13.safetensors`と1対1)
- character / scene: 使わない (下絵が被写体を決める)
- **水彩style presetは不在**。`presets/styles/` 全16件はアニメ調checkpoint対応 (`sd15-*` / `sdxl-*` / `anima-base` / `anime-soft` / `anime-detailed`)。水彩は`prompt.positive`のタグで寄せる案とした

## 実行するはずのコマンド

```bash
# 1. 在庫と実行基盤
uv run imagegen catalog

# 2. 下絵の出自を引く (元絵のcheckpoint/style/seedへ揃えるため)
uv run imagegen history --limit 5

# 3. タグ実在確認
python3 .claude/skills/prompt-builder/scripts/tagcheck.py --prompt "watercolor, watercolor (medium), traditional media, painterly, muted colors"

# 4. validate
uv run imagegen validate specs/generated/base-watercolor.yaml

# 5. generate (必ずsession経由)
scripts/comfyui-session.sh generate specs/generated/base-watercolor.yaml
```

## Spec案 (保存先は `specs/generated/base-watercolor.yaml`)

```yaml
version: "1"
task: img2img

# 解像度は入力画像のサイズをそのまま使うため width / height は書かない
source:
  image: inputs/base.png
  denoise: 0.55

presets:
  style: sd15-hassaku

prompt:
  positive: >
    watercolor, watercolor (medium), traditional media, painterly,
    soft shading, muted colors, paper texture

generation:
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: base_watercolor
```

判断根拠:
- `width`/`height`/`batch_size`省略 → img2imgで書くと拒否
- `denoise: 0.55` → 既定0.6より下。構図保持しつつ画風だけ載せ替え
- `clip_skip`/外部VAE/sampler/cfg/steps → `sd15-hassaku`が持つ → Spec側に書かない
- hires fix未指定 → 「描き直し」だけの要求。拡大要求なし

## 質問 (これに答えないと進めない)

`inputs/` には `.gitkeep` のみ。**`inputs/base.png` が存在しない。** また下絵をどのcheckpointで作ったかも未確定 (元絵と揃えないと水彩化のついでに絵柄まで変わる)。

まず1点だけ:

**Q. `inputs/base.png` をどう用意するか？**

- **A (推奨): 過去の生成物を使う。** `uv run imagegen history --limit 5` で出た出力パスから1枚選び `inputs/base.png` へ置く。checkpointとstyle presetを元絵へ揃えられるため、水彩化以外の変化が出ない
- **B: 手元の画像を置く。** 置いたパスを教えてもらう。出自不明なら上のSpecどおり `hassakuSD15_v13.safetensors` + `sd15-hassaku` を既定として当てる
- **C: 下絵から作る。** 先にtxt2imgで下絵を1枚生成し、それを入力にして2段目で水彩化する
