# inventory-check-uses-catalog

- skill: `imagegen`
- query: 今使えるcheckpointとLoRAを教えて
- 所要: 0.67 USD / 25秒

## 応答

照会要求 → SKILL.mdの入口表「どのモデルが入ってる」「LoRAの一覧」→ 手順1 (`catalog`)。生成なし → Spec作成不要。

## 実行するはずのコマンド

```bash
uv run imagegen catalog
```

- checkpoint / LoRA / ControlNet / IPAdapter / CLIP Vision / DiT系3点 / VAE / アップスケールモデル / embedding / preset / フォントを一度に出す
- `Backend:` 確認。`api`=ComfyUI実読込名、`filesystem`=未起動時ディレクトリ直読み。今回はcheckpoint/LoRAのみ→`filesystem`でも可。カスタムノード由来種別(IPAdapter)まで問う要求なら `scripts/comfyui-session.sh catalog` で見直し
- `Devices:` が `xpu:0`/`cpu` (ComfyUI起動時のみ表示)

## Spec

生成を伴わない照会 → GenerationSpecは作らない。

## 参照文書

- `.claude/skills/imagegen/SKILL.md` — 「要求の種類から入口を決める」表、手順1「実行基盤と在庫を確かめる」
- `CLAUDE.md` — 禁止事項「未知のcheckpointを勝手に使用しない」
- `docs/prompting-guide.md#配置済みのsd15系モデル` — catalog出力の各checkpointの絵柄・cfg/steps実用域を添える先

## preset名

catalog出力で既定として案内するもの: checkpoint `hassakuSD15_v13.safetensors` + style preset **`sd15-hassaku`**。

質問なし (要求は一意)。
