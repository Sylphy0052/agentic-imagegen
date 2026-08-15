# same-character-new-scene-uses-reference

- skill: `imagegen`
- query: さっきの子で、今度は図書館にいる場面を出して
- 所要: 0.64 USD / 58秒

## 応答

## 判断

「さっきの子で別の場面」→ 生成手順1から入らず、まず記録から引く。推測でキャラを当てない。
台帳 `registry/characters/` は0件 → `character show` 不可 → `history` で引く。
preset差し替えだけでは顔立ち固定不可 → 基準画像 + IPAdapter (`reference`) が要る。
`weight_type: style transfer` 必須 (無いと背景が前回のまま)。
IPAdapter × hires fix 併用不可 → 今回 `upscale` 書かない。

## 参照文書

- `CLAUDE.md`「さっきの子で別の場面を」項
- `.claude/skills/imagegen/SKILL.md` 手順1/2/4/5/6
- `.claude/skills/imagegen/references/character-consistency.md` (手順2, 2.5, 3、weight/weight_type表)
- `docs/spec-reference.md`「組み合わせの可否」「reference (IPAdapter)」

## 実行するはずのコマンド

```bash
uv run imagegen catalog
uv run imagegen character list
uv run imagegen history --limit 5
```

historyで直近キャラの出力パス・seed・checkpoint・preset確定後:

```bash
cp <historyが返した出力パス> inputs/character.png
uv run imagegen validate specs/generated/consistency-library.yaml
scripts/comfyui-session.sh generate specs/generated/consistency-library.yaml
```

以降も使うなら台帳登録:

```bash
uv run imagegen character show <name>
```

## Spec (保存せず本文提示)

```yaml
version: "1"
task: txt2img

presets:
  character: <historyのpresets.character>   # 前回と同じものを残す
  scene: library-daylight                   # ここだけ差し替え
  style: <historyのpresets.style>           # 変えると別人に見える

model:
  checkpoint: <historyのcheckpoint>         # 基準画像と同一必須

generation:
  width: 512
  height: 768
  seed: -1                                  # 場面違いなので変えてよい

reference:
  image: inputs/character.png
  model: ip-adapter-plus_sd15.safetensors
  clip_vision: CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
  weight: 0.8
  weight_type: style transfer

output:
  prefix: consistency_library
```

## preset名

- character: history依存 (候補は `anime-girl-blue` / `anime-girl-trailrun` / `anime-boy-hooded`)
- scene: `library-daylight` (既存、昼の図書館・本棚背・上半身)
- style: history依存 (checkpointと1対1、例 `sd15-hassaku`)

## 未確定 (history実行で埋まる)

character preset名、style preset名、checkpoint、基準画像パス。台帳0件 → `history --limit 5` 必須。
`catalog` で ipadapter / clip_vision が `(なし)` なら IPAdapter未導入 → 投入拒否 (exit 5) → character preset + seed固定の代替へ落とす。
