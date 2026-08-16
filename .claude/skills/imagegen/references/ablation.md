# 条件を1つ振って比べる

「どのcheckpointがいいか」「clip skipは1と2どちらか」「この書き方で色が移らなくなるか」の
ように、**条件を1つだけ変えて結果を比べる**ときの手順。

結論は必ず文書へ残す。比較は時間がかかるため、同じ比較を2度やらないようにする。

## 何を振るか決める

振ってよい軸は1回につき1つだけ。2つ以上動かすと、差がどちらに由来するのか分からなくなる。

| 振る軸 | 固定するもの |
| --- | --- |
| checkpoint | prompt / seed / 解像度 / sampler / scheduler / cfg / steps / clip skip / VAE |
| clip skip | checkpointを含む上記すべて |
| プロンプトの書き方 | checkpoint / seed / 生成パラメータ |
| sampler・scheduler | checkpoint / seed / steps / cfg |

seedは固定する。`-1` のままだと比較にならない。値は何でもよいが、
比較の中では同じ値を使い、結論と一緒に書き残す。

## Specを組む

`specs/generated/` へ、振る軸の値ごとに1ファイル作る。ファイル名へ振った値を入れておくと、
あとで出力と突き合わせやすい (`abl-hassaku.yaml` / `abl-meinamix.yaml`)。

- style presetは比較の対象そのものになる。checkpointを振るなら、
  各checkpointに対応する `sd15-*` を当てるか、逆に全部同じpresetで揃えるかを決めて、
  どちらにしたかを結論へ書く (前者は「そのcheckpointの実力」、後者は「同条件での素の差」を見ている)
- `output.prefix` へ振った値を入れる。出力ディレクトリ名から条件が読める
- 負荷は下げる。1枚あたりの時間が枚数分そのまま伸びる

## 流す

```bash
scripts/comfyui-session.sh batch specs/generated/abl-*.yaml
```

- Specの検証は実行前に全件行われる。1件でも不正なら1件も生成されない
- 1件失敗しても残りは続く。最後にサマリが出る
- **低stepsでSDE系のsampler (`*_sde`) を使わない。** 収束せず絵が破綻するため、
  比較にならない。負荷を下げたいなら `euler` / `euler_ancestral` にする
- 出力を `head` などパイプの読み手が先に閉じるコマンドへ繋がない。
  途中経過が切れて結果を読み違える。全部見たいならファイルへリダイレクトする

## 条件が本当に揃っているか確かめる

思い込みで比べない。`metadata.json` に実際に使われた値が入っている。

```bash
python3 - <<'PY'
import json
from pathlib import Path

for directory in sorted(Path("outputs/<日付>").iterdir()):
    metadata = json.loads((directory / "metadata.json").read_text())
    spec = metadata["spec"]
    generation = spec["generation"]
    print(
        directory.name,
        spec["model"]["checkpoint"],
        generation["seed"],
        generation["steps"],
        generation["cfg"],
        generation["sampler"],
        generation["scheduler"],
    )
PY
```

振った軸以外が1つでも違っていたら、その比較は成立していない。作り直す。

## 破綻を機械で弾く

目で見る前に、壊れている出力を落とす。

```bash
python3 .claude/skills/imagegen/scripts/edge_stats.py outputs/<日付>/<出力ディレクトリ>
```

高周波比が0.2を大きく超えるものはVAE不整合などで絵が壊れている。
正常な範囲と読み方は
[既定のcheckpointを決める](../../prompt-builder/references/models/sd15.md#既定のcheckpointを決める)
を一次情報とする。

**この数値で順位を付けない。** 破綻の有無しか見ていない。
線が立つ絵柄は高く、平滑な絵柄は低く出るだけで、良し悪しとは関係がない。

## 見て決める

残ったものを目で比べる。判断の軸は要求によるが、次を先に決めておくと結論がぶれない。

- 指定した服装・小物・構図がそのまま出るか (指示への追従)
- 破綻が無いか (指の数、顔の崩れ、背景の溶け)
- 絵柄が要求に合っているか

追従が同程度なら、絵柄の好みで決めてよい。決め手を結論へ書く。

## 結論を残す

| 何を決めたか | 書く場所 |
| --- | --- |
| モデルの傾向・推奨設定・プロンプトの書き方 | [references/models/](../../prompt-builder/references/models/) |
| そのcheckpointの推奨 sampler / cfg / steps / clip skip / VAE | `presets/styles/<name>.yaml` |
| 既定として何を使うか | [CLAUDE.md](../../../../CLAUDE.md) の「使える機能と参照先」 |

結論には次を添える。後から追試できない記録は残す意味が薄い。

- 実施日
- 固定した条件 (prompt / seed / 解像度 / sampler / scheduler / cfg / steps)
- 振った軸と値の一覧
- 決め手 (何を見てそう決めたか)

数値を載せる場合は、それを出したコマンドも書く。
