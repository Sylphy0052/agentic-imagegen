# comparison-varies-one-axis

- skill: `imagegen`
- query: SD1.5のcheckpointのうち、どれがいちばん指示どおりの服装を出せるか比べて
- 所要: 0.91 USD / 140秒

## 応答

## 結論先に: この比較、既に実施済み

`docs/prompting-guide.md` の[既定のcheckpointを決める](docs/prompting-guide.md#既定のcheckpointを決める)に2026-08-15実施の結果あり。SD1.5系9種、同一プロンプト・同一seed(545078971)・同一設定(`dpmpp_2m_sde`/`exponential`/cfg 7.0/steps 30/512x768→hires x2.0/clip skip 2/外部VAE)。

> 採用の決め手は「指定した服装・小物がそのまま出るか」で、そこに最も忠実だったのがhassakuだった。

→ **`hassakuSD15_v13.safetensors` + `sd15-hassaku`**。既定checkpointに採用済みの理由がまさに服装追従。

`ablation.md` は「同じ比較を2度やらないように」と規定 → 素直に再実行しない。

## ただし既存結果は服装追従の判定として弱い

`docs/prompting-guide.md` の[指定した色と丈を出す](docs/prompting-guide.md#指定した色と丈を出す)と`SKILL.md`手順6が明記:

> **1枚で判定しない。** 色の命中はseedごとに揺れる。seedを4本振って`batch --seeds`で流し、何本命中したかで書き方を比べる。

既存比較はseed1本/checkpoint。服装追従に特化して取り直す価値はある。ただし9種×4seed=36枚 → XPUでも約2時間。

## 参照した文書

- `.claude/skills/imagegen/SKILL.md` (手順6の「1軸だけ振る」「服の色はseed4本」)
- `.claude/skills/imagegen/references/ablation.md` (振る軸/固定するもの、preset統一の判断、metadata.json照合、edge_stats.py、結論の残し先)
- `docs/prompting-guide.md` — [配置済みのSD1.5系モデル](docs/prompting-guide.md#配置済みのsd15系モデル) / [指定した色と丈を出す](docs/prompting-guide.md#指定した色と丈を出す) / [既定のcheckpointを決める](docs/prompting-guide.md#既定のcheckpointを決める)
- `presets/styles/anime-detailed.yaml`、`presets/styles/sd15-hassaku.yaml`
- `CLAUDE.md` (軸の責務、併用可否、所要時間)

## preset名

- 統一条件用style preset: **`anime-detailed`**(`applies_to`無し→checkpoint取り違え警告が出ない。ただし`clip_skip`/`vae`を持たないためSpec側で明示)
- 各checkpoint対応style preset(実力比較にする場合): `sd15-meinamix` / `sd15-counterfeit` / `sd15-aom3` / `sd15-anylora` / `sd15-cetusmix` / `sd15-darksushi` / `sd15-hassaku` / `sd15-chilloutmix` / `sd15-perfectdeliberate`
- 服装指定を載せるcharacter preset(新規、9 Spec共通化用): `abl-outfit-check`

## Spec案(統一条件版、ファイル未保存)

新規character preset `presets/characters/abl-outfit-check.yaml`:

```yaml
description: checkpoint比較用。色と丈を指定した制服一式

prompt:
  positive: >
    1girl, solo, full body, standing,
    white sailor uniform, navy pleated skirt, light blue neckerchief,
    white knee socks, brown loafers
  negative: >
    thighhighs, over-knee socks
```

書き方は[指定した色と丈を出す](docs/prompting-guide.md#指定した色と丈を出す)準拠 — 色とアイテムを1タグ(`navy pleated skirt`)、重みは振らない(素の語で1周目)、丈はnegative(`thighhighs, over-knee socks`)。

`specs/generated/abl-outfit-hassaku.yaml`(代表):

```yaml
version: "1"
task: txt2img

presets:
  character: abl-outfit-check
  style: anime-detailed

generation:
  width: 512
  height: 768
  seed: -1
  sampler: dpmpp_2m_sde
  scheduler: exponential
  cfg: 7.0
  steps: 30

model:
  checkpoint: hassakuSD15_v13.safetensors
  clip_skip: 2
  vae: vaeKlF8Anime2_klF8Anime2VAE.safetensors

output:
  prefix: abl_outfit_hassaku
```

残り8ファイルは`model.checkpoint`と`output.prefix`だけ差し替え:

- `abl-outfit-meinamix.yaml` — `meinamix_v12Final.safetensors` / `abl_outfit_meinamix`
- `abl-outfit-counterfeit.yaml` — `counterfeitV30_v30.safetensors` / `abl_outfit_counterfeit`
- `abl-outfit-aom3.yaml` — `abyssorangemix3AOM3_aom3a1b.safetensors` / `abl_outfit_aom3`
- `abl-outfit-anylora.yaml` — `anyloraCheckpoint_bakedvaeBlessedFp16.safetensors` / `abl_outfit_anylora`
- `abl-outfit-cetusmix.yaml` — `cetusMix_Whalefall2.safetensors` / `abl_outfit_cetusmix`
- `abl-outfit-darksushi.yaml` — `darkSushiMixMix_225D.safetensors` / `abl_outfit_darksushi`
- `abl-outfit-chilloutmix.yaml` — `chilloutmix_NiPrunedFp16Fix.safetensors` / `abl_outfit_chilloutmix`
- `abl-outfit-perfectdeliberate.yaml` — `perfectdeliberate_v20.safetensors` / `abl_outfit_perfectdeliberate`

`waiIllustriousSD15_v1.safetensors` は**除外**。SDXL系VAE内蔵で`model.vae`を書くと極彩色ノイズになる([既定のcheckpointを決める](docs/prompting-guide.md#既定のcheckpointを決める)末尾)→ 外部VAE統一条件に載らない。既存比較の9種にも入っていない。

## 実行するはずのコマンド

```bash
# 1. 在庫と実行基盤の確認 (checkpoint実在、Devices: xpu:0 か cpu か)
uv run imagegen catalog

# 2. 全件validate
uv run imagegen validate specs/generated/abl-outfit-hassaku.yaml
uv run imagegen validate specs/generated/abl-outfit-meinamix.yaml
uv run imagegen validate specs/generated/abl-outfit-counterfeit.yaml
uv run imagegen validate specs/generated/abl-outfit-aom3.yaml
uv run imagegen validate specs/generated/abl-outfit-anylora.yaml
uv run imagegen validate specs/generated/abl-outfit-cetusmix.yaml
uv run imagegen validate specs/generated/abl-outfit-darksushi.yaml
uv run imagegen validate specs/generated/abl-outfit-chilloutmix.yaml
uv run imagegen validate specs/generated/abl-outfit-perfectdeliberate.yaml

# 3. seed4本固定で流す (36枚)
scripts/comfyui-session.sh batch specs/generated/abl-outfit-*.yaml \
  --seeds 545078971,111,222,333 > /tmp/abl-outfit.log 2>&1

# 4. 条件が揃っているか metadata.json で照合
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

# 5. 破綻を機械で弾く (順位付けには使わない)
python3 .claude/skills/imagegen/scripts/edge_stats.py outputs/<日付>/<出力ディレクトリ>
```

未確認点1つ: `batch` で複数Spec指定と`--seeds`が併用できるか、`SKILL.md`は別々の例しか出していない。併用不可なら「1 Spec × `--seeds` 4本」を9回に分ける。

## 質問

再実行の範囲。**A**推奨。

- **A: 既存結論(hassaku)を採り、再実行しない** — 決め手が既に「指定した服装・小物の追従」そのもの。36枚約2時間を払う前に、hassakuでseed4本振って命中率を測る方が安い(`ablation.md`「同じ比較を2度やらない」)
- **B: 服装特化でSD1.5系8種×seed4本を取り直す** — 上のSpecそのまま。約2時間。統一条件(全部`anime-detailed`+`dpmpp_2m_sde`)で「素の差」を見る
- **C: 上位3種(hassaku / meinamix / counterfeit)×seed4本に絞る** — 12枚、約40分。既存比較で破綻無しだった上位のみ再検証

併せて後で決める点(A/B/C確定後に一問一答で聞く): style presetを`anime-detailed`統一にするか各`sd15-*`にするか(前者は「素の差」、後者は「各モデルの実力」)、服装の指定内容を上のセーラー服一式でよいか。
