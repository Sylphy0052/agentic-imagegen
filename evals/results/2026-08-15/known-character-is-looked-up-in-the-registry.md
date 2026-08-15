# known-character-is-looked-up-in-the-registry

- skill: `imagegen`
- query: aoiで図書館の絵を作って
- 所要: 0.54 USD / 44秒

## 応答

## 確認したこと

- `registry/characters/` = `.gitkeep`のみ → 台帳空。`aoi`未登録
- `presets/characters/` = `anime-boy-hooded` / `anime-girl-trailrun` / `anime-girl-blue`。`aoi`無し
- `outputs/` = `.gitkeep`のみ → 過去生成なし。`history`で引けない

→ `aoi`が誰か解決不能。推測でSpec組まず質問。

## 実行するはずだったコマンド (未実行)

```bash
uv run imagegen catalog
uv run imagegen character list
uv run imagegen character show aoi
uv run imagegen history --prefix aoi
```

## 参照した文書

- `.claude/skills/imagegen/SKILL.md` (手順1 実行基盤と在庫、手順2 preset選択、手順3 一言要求の確認)
- `CLAUDE.md` (既定checkpoint、軸の責務、台帳照会)
- `presets/scenes/library-daylight.yaml`

## 決まっている軸 (aoi判明後にそのまま使う)

- scene preset: **library-daylight** (昼の図書館、本棚背、上半身)
- style preset: **sd15-hassaku** (checkpoint未指定時の既定)
- checkpoint: `hassakuSD15_v13.safetensors`
- character preset: **未定** ← 質問対象

## 質問

`aoi`をどう解決するか。

- **A (推奨)**: 新規character presetとして`presets/characters/aoi.yaml`を作る。外見(髪色・髪型・目・服装)を指定してほしい。台帳も生成実績も空なので、確定情報が他に無い
- **B**: 既存の`anime-girl-blue`を`aoi`として流用する
- **C**: 別の場所(他worktree / 他マシンの`outputs`・`registry`)に記録がある。そのパスを指定してほしい
