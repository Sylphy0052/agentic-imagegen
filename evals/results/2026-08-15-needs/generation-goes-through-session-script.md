# generation-goes-through-session-script

- skill: `imagegen`
- query: さっき作ったSpecで生成して
- 所要: 0.51 USD / 49秒
- 許可したコマンド: Bash(python3 evals/bin/imagegen_ro.py:*)
- 前提: evals/fixtures/specs/blue-hair-rooftop.yaml を作ったところ。presets は character: anime-girl-blue / scene: rooftop-sunset / style: sd15-hassaku、model.checkpoint は hassakuSD15_v13.safetensors、512x768で1枚。validate は通してあるが、生成はまだ行っていない。

## 応答

既存Specの名指し再利用なので、確認は挟まずそのまま流します。

## 参照した文書

- `.claude/skills/imagegen/SKILL.md` (手順1/4/5/6)
- `CLAUDE.md` (画像生成要求を受けたときの手順、exit code、所要時間)
- `docs/spec-reference.md` (presetsの軸、値域) は今回は既存Specの確認のみで追加参照なし

## 使うpreset

`validate` の `Presets:` 行で意図どおり並んでいることを確認済み。

| 軸 | preset |
| --- | --- |
| character | `anime-girl-blue` |
| scene | `rooftop-sunset` |
| style | `sd15-hassaku` (checkpoint `hassakuSD15_v13.safetensors` に1対1で対応) |

`catalog` の実在確認も取れています (checkpoint 14件中に `hassakuSD15_v13.safetensors` あり、style presetに `sd15-hassaku` あり)。

## Spec (evals/fixtures/specs/blue-hair-rooftop.yaml、変更なし)

```yaml
version: "1"
task: txt2img

presets:
  character: anime-girl-blue
  scene: rooftop-sunset
  style: sd15-hassaku

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors

output:
  prefix: blue_hair_rooftop
```

## 実行した読み取り専用コマンド (許可分)

```bash
python3 evals/bin/imagegen_ro.py catalog
python3 evals/bin/imagegen_ro.py validate evals/fixtures/specs/blue-hair-rooftop.yaml
python3 evals/bin/imagegen_ro.py health
```

`validate` は `OK` / `Workflow: txt2img_vae` / `Estimate: XPU 約2分 / CPU 約18分`、warningなし。`health` は `unreachable` (exit 3) ですが、生成は `scripts/comfyui-session.sh` が起動から停止まで面倒を見るため事前起動は不要で、これは阻害要因になりません。

## 生成コマンド (実行していません)

```bash
IMAGEGEN_TIMEOUT=1200 scripts/comfyui-session.sh generate evals/fixtures/specs/blue-hair-rooftop.yaml
```

`IMAGEGEN_TIMEOUT` を既定の300から上げているのは、ComfyUIが停止中で `catalog` の `Backend:` が `filesystem fallback`、`Devices:` が取れずXPU/CPUを確定できないためです。CPU推論だと見積り約18分が既定値を超えてexit 6で捨てることになる一方、XPUなら約2分で終わるのでタイムアウトを長く取っても実害はありません。

生成後は exit code 0と出力ファイルの存在を確認し、出力パスと `metadata.json` の `resolved_seed` (Specは `seed: -1`) を報告します。
