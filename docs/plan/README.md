# docs/planの方針

このディレクトリには **設計判断を独立した文書に残す価値があった機能だけ** の設計文書を置く。
全フェーズ分の設計文書は意図的に作っていない。

## 置いてある文書

| 文書 | 対象 | 位置づけ |
| --- | --- | --- |
| [phase1.md](phase1.md) | Phase 1 (CLI -> ComfyUIの一気通貫) | 最初の設計の記録。実装済み |
| [phase5-japanese-text.md](phase5-japanese-text.md) | 日本語テキスト描画 | 方式選定の記録。合成方式を実装し、Qwen-Imageは導入条件だけ残した |

どちらも **作成時点の記録**であり、現在の実装の仕様書ではない。
現在の仕様は [docs/spec-reference.md](../spec-reference.md) と実装を参照する。

## Phase 2 / 3 / 4の設計文書が無い理由

Phase 2 (preset / LoRA / img2img)、Phase 3 (MCP Server)、
Phase 4 (ControlNet / IPAdapter / batch / hires fix)、DiT系モデル (Anima) 対応については、
**設計判断をIssue本文で管理し、独立した文書は作っていない。**

- いずれもPhase 1で決めた層構造 (Domain / Service / Workflows / Adapters) と
  GenerationSpecの拡張で収まり、方式を比較検討する余地が小さかった
- 一方Phase 1は構造そのものを決める必要があり、Phase 5は
  「モデルを変える」対「生成後に合成する」という方式選定があった。
  この2つだけが独立した文書に見合う

したがって、新しい機能を足すときに毎回ここへ文書を作る必要はない。
**方式の比較検討が要る場合だけ**作り、それ以外はIssue本文
(背景 / スコープ / 仕様 / 受入基準 / 実装計画 / 検証方法) で完結させる。

## 進捗はここで管理しない

各フェーズの完了状況は
[Issue #1](https://github.com/Sylphy0052/agentic-imagegen/issues/1) (Roadmap、完了済み) と
openなIssueが一次情報。この文書と `phase*.md` には進捗を書かない。
