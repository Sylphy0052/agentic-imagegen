# agentic-imagegen

AIコーディングエージェント (Claude Code など) から、ComfyUIを介してStable Diffusion系モデルによる画像生成を実行するための基盤。

```text
Claude Code -> GenerationSpec -> Python CLI (imagegen) -> ComfyUI API -> 画像生成
```

## ステータス

Phase 1 実装中。進捗は [Roadmap Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1)、設計は [docs/plan/phase1.md](docs/plan/phase1.md) を参照。

現時点で利用できるのはスキャフォールドのみ。

```bash
uv sync
uv run imagegen version
```

## セットアップ (Phase 1 Step 9 で拡充予定)

- 必要環境: Python 3.12以上、uv、ComfyUI (WSL上でCPU推論)
- ComfyUI導入手順: `docs/comfyui-setup.md` (Step 7で作成)

## 開発

```bash
uv sync
uv run pytest                 # Unit Test のみ
uv run pytest -m integration  # ComfyUI起動時のみ
uv run ruff check .
uv run mypy src
```

## ライセンス

[LICENSE](LICENSE) を参照。
