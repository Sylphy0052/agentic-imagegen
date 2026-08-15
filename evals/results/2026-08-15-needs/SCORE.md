# 2026-08-15の採点 (needs- タグの3 case)

- 対象commit: `4b14aab` (origin/main)
- 実行: `python3 evals/run_case.py tags-are-verified-before-use known-character-is-looked-up-in-the-registry generation-goes-through-session-script --outdir evals/results/2026-08-15-needs`
- [2026-08-15の採点](../2026-08-15/SCORE.md)で判定不能または一部未達だった3 caseを、
  条件を整えて回し直したもの

## 条件として足したもの

- `shell` — caseごとに実行を許すコマンド。ここでは `python3 evals/bin/imagegen_ro.py`
  (状態を変えるサブコマンドを通さない読み取り専用ラッパー) と `tagcheck.py`
- `context` — 「さっき作ったSpec」が指す先。`evals/fixtures/specs/blue-hair-rooftop.yaml`
- `evals/fixtures/registry` — `aoi` の台帳。基準画像は意図的に置かず、
  参照先の欠落 (warning) まで観測できるようにした

## 判定

| case | 前回 | 今回 | 備考 |
| --- | --- | --- | --- |
| tags-are-verified-before-use | 判定不能 | 合格 | 25タグを実際に引き、`shrine_grounds` (0件) を削除。`japanese clothes` / `nontraditional miko` / `hakama skirt` / `tree` を重複・競合として整理し、`shrine` 6,251 / `stone_lantern` 2,527 の効きの弱さも報告 |
| known-character-is-looked-up-in-the-registry | 判定不能 | 合格 | `character show aoi` で台帳を引き、preset・style・checkpoint・seedをSpecへ写した。基準画像欠落のwarningを読み、`reference` を書かない判断とその理由を述べた |
| generation-goes-through-session-script | 一部未達 | 合格 | `comfyui-session.sh generate` 経由。`validate` を実際に叩いて `Estimate:` を読み、`IMAGEGEN_TIMEOUT` を上げる判断まで説明。出力パスと `metadata.json` の `resolved_seed` を報告すると明示 |

3 caseとも合格。`forbidden_behaviors` を踏んだcaseは無し。

## 分かったこと

### 前回の観測条件は保証になっていなかった

`--allowedTools` を渡していたが、利用者のグローバル設定が
`defaultMode: bypassPermissions` かつ allowリスト50件 (`Bash(uv:*)` など) のため、
`claude -p` の子セッションもそれを継承していた。**前回の19 caseは「読み取りしか
できない」状態ではなく、system promptの指示にモデルが従っていただけだった**。
`permission_denials` が全件空だったのはそのため。

`--permission-mode default` と `--setting-sources project` を固定して塞いだ。
このリポジトリには `.claude/settings.json` が無いため、これで許可は
`--allowedTools` で渡した分だけになる。

### 許可は前方一致なので、呼び方がずれると落ちる

権限を塞いだ直後は `Bash(uv run imagegen character:*)` を渡していたが、
モデルは `uv run --offline --no-sync imagegen character list` と打ったため
前方一致から外れ、台帳を引けずに終わった (`permission_denials` に3件)。

パターンを `Bash(uv run imagegen:*)` まで緩めると `generate` も通ってしまうため、
`evals/bin/imagegen_ro.py` を挟んで許可をそこ1つに与えた。通すサブコマンドは
`catalog` / `validate` / `history` / `character` / `health` だけで、
`generate` / `batch` / `compose` は拒否する。あわせて許可したコマンドの形を
system promptへそのまま見せるようにした。

### skillの修正が効いている

`tags-are-verified-before-use` の応答が、`sd15-hassaku` のnegativeにある
`embedding:negativeXL_D` について「SDXL向けに見えるが実ファイルの有無は未確認」と
断って報告した。#121 で prompt-builder へ書いた「引いていない値を断言しない」が
そのまま出ている。
