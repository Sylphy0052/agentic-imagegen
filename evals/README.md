# skillのevals

このリポジトリの価値は、自然言語の要求から妥当なGenerationSpecへ落とせることにある。
その判断は `.claude/skills/` のskillが持っているが、SKILL.mdを直したときに判断が退行しても
`uv run pytest` では何も落ちない。ここはその穴を埋めるための手動評価の台帳。

実際の退行例:

- style presetがclip skipと外部VAEを持てず、注意書きが分散した結果、
  直近9件の生成でpresetが1つも使われていなかった (Issue #98で解消)
- 手順の冒頭が7本の `ls` の列挙で、モデル種別を足すたびにSKILL.mdを直していた (Issue #99で解消)

どちらもSKILL.mdの差分を眺めても気付けず、実行して初めて分かる類だった。

## 何を見るか

**画像は生成しない。** 見るのは判断だけ。

- 作られたSpecの中身 (どのブロックを足したか、何を書かなかったか)
- 叩いたコマンド (`imagegen catalog` か `ls` の列挙か、`comfyui-session.sh` 経由か)
- 質問したかどうか (要求が曖昧なら聞く、具体的なら聞かない)

生成の質そのものはevalsの範囲外とする。CPU / XPU推論は1枚で分単位の時間がかかるため、
毎回回せる形にならない。絵の良し悪しは `outputs/` のmetadataと実物で別途見る。

## 回し方

1. skillを有効にした状態で新しいセッションを開き、`evals.json` の `query` をそのまま投げる
2. 応答と、その過程で作られたSpec・叩かれたコマンドを記録する
3. `expected_behaviors` を1つずつ満たしたか判定する。`forbidden_behaviors` は1つでも
   起きたらそのcaseは不合格
4. 判定結果を日付とcommit hashを添えて残す (置き場所は決めていない。
   PRの本文へ貼るのがいちばん追いやすい)

会話の途中から投げると、それまでの文脈が判断へ影響する。1 caseにつき1セッションで回す。

### skill無しとの比較

skillが効いているかは、skill無しでも同じ結果になるかで見る。判断が退行したかどうかとは
別の観点で、「そのskillの記述が要るのか」を確かめるために使う。

- `--no-skills` のような直接の切り替えは無いので、skillを一時的に読ませない状態
  (別ディレクトリからの実行など) で同じ `query` を投げる
- skill無しでも `expected_behaviors` を満たすcaseは、SKILL.mdから消してよい記述の候補
- skill無しで `forbidden_behaviors` を踏むcaseは、SKILL.mdのその記述が効いている証拠

全caseで比較する必要はない。SKILL.mdを削る判断をするときだけ、対象の記述に対応する
caseで見れば足りる。

## 書式

`evals.json` は次の形。

| フィールド | 内容 |
| --- | --- |
| `id` | kebab-caseの識別子。重複不可 |
| `skill` | `.claude/skills/` 配下のディレクトリ名 |
| `query` | ユーザーの要求文。そのまま投げる |
| `expected_behaviors` | 満たすべき振る舞い。1つずつ判定できる粒度で書く |
| `forbidden_behaviors` | 1つでも起きたら不合格になる振る舞い |
| `presets` | caseが名指しするpreset (`character` / `scene` / `style`)。実在をテストが検査する |
| `references` | caseの根拠になる文書。実在をテストが検査する |
| `tags` | `regression:<Issue番号>` / `routing` / `safety` などの分類 |

`tests/unit/test_skill_evals.py` が検査するのは書式と参照先の実在だけで、
判定そのものは人 (またはエージェント) が行う。

## 増やすとき

退行を1件見つけたら、直すのと同時にcaseを1件足す。
`tags` へ `regression:<Issue番号>` を入れておくと、何を守っているのかが後から追える。
