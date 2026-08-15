# 2026-08-15の採点

- 対象commit: `215bc1c` (origin/main) / 実行したworktreeのHEAD `8f9739a`
- 実行: `python3 evals/run_case.py --all --outdir evals/results/2026-08-15`
- 全19 case、合計 13.88 USD / 約25分。応答は同ディレクトリの `<id>.md`

## 観測条件

`claude -p` を1 caseにつき1回起動した。渡したツールは `Read` / `Glob` / `Grep` / `Skill` だけ。

> **訂正 (同日、[needs- の採点](../2026-08-15-needs/SCORE.md)で判明)。**
> この指定は効いていなかった。利用者のグローバル設定が `defaultMode: bypassPermissions`
> かつallowリスト50件のため、`claude -p` の子セッションもそれを継承していた。
> ここでの19 caseは「読み取りしかできない」状態ではなく、system promptの指示に
> モデルが従っていただけ。`--permission-mode default` と `--setting-sources project` を
> 固定して塞いだのはそのあと。「コマンドは実行せず、実行するはずのコマンドを書き出す」
と指示してあるため、**叩いたコマンドは宣言として観測している**。
コマンドの出力を読んで次を決める類のcaseは、この条件では最後まで見られない。

## 判定

| case | 判定 | 備考 |
| --- | --- | --- |
| readable-japanese-uses-text-layer | 合格 | 生成と合成を2段に分け、`text.layers` へ「本日休業」。`fonts/` が空であることを見つけて exit 10 と `IMAGEGEN_FONTS_ROOT` を案内 |
| same-character-new-scene-uses-reference | 合格 | `history` で引き、`reference` + `weight_type: style transfer`、`scene` だけ `library-daylight` へ差し替え |
| one-line-request-asks-before-spec | 合格 | 一問一答で停止。推奨案は既定のcheckpointとstyle preset |
| concrete-request-skips-questions | 合格 | 質問なし。preset・解像度とも要求どおり |
| style-preset-carries-clip-skip-and-vae | 一部未達 | Specは期待どおり (`model.checkpoint` だけ)。ただしWorkflowが `txt2img_vae` になることを明示せず、「Workflow行を確認する」止まり |
| inventory-check-uses-catalog | 合格 | `catalog` 1回。`Backend:` が `api` か `filesystem` かで値の確からしさを分けた |
| incompatible-combination-is-refused | 合格 | 併用不可を生成前に伝え、2段運用へ分割 |
| img2img-omits-resolution | 合格 | `width`/`height` を書かず `denoise: 0.55`。`inputs/base.png` が無いことを確かめてから止まった |
| generation-goes-through-session-script | 一部未達 | `comfyui-session.sh generate` 経由は満たす。出力パスとseedを報告する話に触れていない |
| dit-model-rejects-lora | 合格 | 併用不可を先に伝え、Anima優先とLoRA優先の選択肢。Animaなら `anima-base` |
| comparison-varies-one-axis | 合格 | 既存の比較結果を先に見つけ、再実行の是非から確認。振る軸はcheckpointのみ、`metadata.json` 照合と `edge_stats.py` も宣言 |
| sd15-avoids-score-tags | 合格 | 品質タグはstyle preset側、danbooruタグ主体、75トークンに対し余裕を明示 |
| tags-are-verified-before-use | 判定不能 | プロンプトは組めているが `tagcheck.py` を実行できない条件のため、置換の判断まで見られない。「16タグとも実在未確認」と明記して完成扱いにはしなかった |
| color-and-length-are-written-per-garment | 合格 | `white shirt` / `red long skirt` と服ごとに1トークン、丈はnegativeで押さえた |
| token-budget-is-checked | 合格 | 約48/75トークンと数え、対応するタグの無い形容 (「映画的な色調」「質感」「階調」) を落として報告。既定style presetのnegativeが要求と衝突することまで見つけてstyle presetを変えた |
| validate-warning-is-addressed | 合格 | 1024x1024直接生成の見積り378秒と `IMAGEGEN_TIMEOUT` 300を突き合わせ、hires fixで到達する形へ組み替えた |
| known-character-is-looked-up-in-the-registry | 判定不能 | `registry/characters/` も `outputs/` も空で `aoi` が解決できない。台帳を引く判断そのものは正しい |
| caption-only-request-uses-compose | 合格 | `compose` へ入り、`text` だけのSpec、出力は `inputs/base_text.png`。`comfyui-session.sh` は不要と明言 |
| upscale-only-request-uses-img2img | 合格 | img2img + `generation.upscale`、`denoise: 0.35`、元のcheckpointとpresetを `history` から引く |

合格15 / 一部未達2 / 判定不能2。`forbidden_behaviors` を踏んだcaseは無し。

一部未達の `generation-goes-through-session-script` と判定不能の2 caseは、条件を整えて
回し直して3件とも合格になった ([needs- の採点](../2026-08-15-needs/SCORE.md))。

## 分かったこと

### caseを直す

- **`upscale-only-request-uses-img2img` の期待が実装と食い違う。** 「`validate` の `Estimate:` を見てから流す」と
  書いてあるが、img2imgでは入力画像のサイズで生成するため `Estimate:` は出さない
  (`services/estimate.py`)。応答はその点を正しく指摘していた。期待の側を直す
- **`known-character-is-looked-up-in-the-registry` は台帳が空だと成立しない。** 台帳へ登録済みという
  前提をcaseに書くか、evalが使うサンプル台帳を置く
- **`generation-goes-through-session-script` は「さっき作ったSpec」の文脈に依存する。** 新しいセッションでは
  対象が引けず、報告の中身まで判定できない。前提をcaseへ書く
- **`tags-are-verified-before-use` はコマンドの実行結果が要る。** `run_case.py` の条件では最後まで見られない。
  対話セッションで回すcaseとして印を付ける

### skillの側で気になった点 (今回は直していない)

- `one-line-request-asks-before-spec` の質問で、checkpointの選択肢へ `anime-soft` (style preset) が
  並んだ。軸が混ざっている。SKILL.md手順3の質問例に、軸ごとに聞くことを書き足す余地がある
- `color-and-length-are-written-per-garment` の応答が、キャッシュに無いタグの `post_count` を
  0と数値付きで書いた。実際に引いていない値を断言している。prompt-builderのSKILL.mdへ
  「引いていないpost_countは書かない」を明記する余地がある

## skill無しとの比較

今回は行っていない。SKILL.mdの記述を削る判断をするときに、対象の記述に対応するcaseだけで見る。
