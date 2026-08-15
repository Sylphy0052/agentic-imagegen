# generation-goes-through-session-script

- skill: `imagegen`
- query: さっき作ったSpecで生成して
- 所要: 0.49 USD / 48秒

## 応答

## 判断

「さっき作ったSpec」— 本セッションに該当文脈なし。推測でSpec捏造しない (SKILL.md「推測で当てない」/ CLAUDE.md「記憶で補わず引く」)。よってSpec本文は未提示。preset名も未確定 (後述の質問回答後に確定)。

## 実行するはずのコマンド

記録から引く (生成前、ComfyUI不要):

```bash
uv run imagegen history --limit 5
```

```bash
ls -t specs/generated/*.yaml | head -5
```

在庫と実行基盤の確認 (手順1):

```bash
uv run imagegen catalog
```

Spec確定後:

```bash
uv run imagegen validate specs/generated/<name>.yaml
scripts/comfyui-session.sh generate specs/generated/<name>.yaml
```

## 質問 (対話不可 → ここで終了)

どのSpecで生成するか。

- **A (推奨): `imagegen history --limit 1` の直近1件のSpec** — 「さっき」の素直な解釈。実際に使われたseed/checkpoint/presetがmetadata.jsonから取れて再現性が担保される
- **B: `specs/generated/` 配下のファイル名を指定** — 直近生成 != 直近作成Spec のとき (validateだけして未生成 等)
- **C: 台帳のキャラ名で指定 (`imagegen character show <name>`)** — 「さっきの子」を指していて別構図を出す場合

## 参照した文書

- `CLAUDE.md` — 画像生成要求の手順、禁止事項 (`comfyui-session.sh` 経由、validate迂回禁止)
- `.claude/skills/imagegen/SKILL.md` — 要求の種類から入口を決める表、手順1 (記録から引く)、手順3 (曖昧時は推測で埋めない)
- 未参照 (Spec確定後に参照予定): `docs/spec-reference.md`、`docs/prompting-guide.md`

## preset

未確定。A/B/C回答後、`history` / `character show` の出力にあるpresetをそのまま写す。既定にフォールバックする場合は `hassakuSD15_v13.safetensors` + style preset `sd15-hassaku`。
