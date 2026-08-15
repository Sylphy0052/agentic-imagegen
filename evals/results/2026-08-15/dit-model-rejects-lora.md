# dit-model-rejects-lora

- skill: `imagegen`
- query: AnimaでLoRAを2つ効かせて出して
- 所要: 0.75 USD / 59秒

## 応答

## 判断: この要求はそのまま実行不可

参照文書:

- `CLAUDE.md`「使える機能と参照先」/「禁止事項」
- `.claude/skills/imagegen/SKILL.md` 手順1・2・4
- `docs/spec-reference.md` の[DiT系モデル (Anima)](docs/spec-reference.md#dit系モデル-anima)、[組み合わせの可否](docs/spec-reference.md#組み合わせの可否)

理由: **DiT系 (`model.unet`+`clip`+`vae`) とLoRAは併用不可**。可否表の DiT系行 × LoRA列 = 不可。`validate` 時点で `InvalidGenerationSpec` (exit code 2) 拒否。SD1.5/SDXL向けLoRAはAnimaのUNetへ当たらない。着手条件は [Issue #39](https://github.com/Sylphy0052/agentic-imagegen/issues/39)。

在庫確認で実行するはずのコマンド (未実行):

```bash
uv run imagegen catalog
```

## 質問 (一問一答、これ1問)

**AnimaとLoRA、どちらを優先?**

- **A (推奨): Anima優先、LoRAなし** — モデル指定は代替不能。LoRAの絵柄寄せはpresetとプロンプトで部分代替可
- **B: LoRA 2つ優先、checkpoint系へ変更** — `hassakuSD15_v13.safetensors` + `sd15-hassaku`。LoRA同時3件まで → 2件可
- **C: 両方1枚ずつ出して比較** — Anima素 vs SD1.5+LoRA2件。ただし別モデル間比較 → 1軸ablationにならない (条件揃わず)

参照preset名: `anima-base` (Anima用)、`sd15-hassaku` (既定checkpoint用)。

## 案別Spec骨子 (被写体未定のため `positive` は仮)

案A (Anima、LoRAなし):

```yaml
version: "1"
task: txt2img

presets:
  style: anima-base

prompt:
  positive: "1girl, solo, <被写体未定>"

generation:
  width: 832
  height: 1216
  seed: -1

model:
  unet: hassakuAnima_v13_int8.safetensors
  clip: qwen_3_06b_base.safetensors
  vae: qwen_image_vae.safetensors

output:
  prefix: anima_base
```

案B (SD1.5、LoRA 2件):

```yaml
version: "1"
task: txt2img

presets:
  style: sd15-hassaku

prompt:
  positive: "1girl, solo, <被写体未定>"

generation:
  width: 512
  height: 768
  seed: -1

model:
  checkpoint: hassakuSD15_v13.safetensors
  loras:
    - name: <catalog で実在確認したLoRA 1>
      strength_model: 0.8
      strength_clip: 0.8
    - name: <catalog で実在確認したLoRA 2>
      strength_model: 0.7
      strength_clip: 0.7

output:
  prefix: sd15_lora2
```

案A選択時のworkflowは `txt2img_unet`、案Bは `txt2img_lora` (接尾辞規則より)。

後続で実行するはずのコマンド (未実行):

```bash
uv run imagegen validate specs/generated/<name>.yaml
scripts/comfyui-session.sh generate specs/generated/<name>.yaml
```

未解決の残件: 被写体・画風の中身が未指定 (A/B決定後に別途1問)。LoRAファイル名も `catalog` 未実行のため未確定。
