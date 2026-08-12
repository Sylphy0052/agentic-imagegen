"""複数Specの一括実行。

既存の単発生成パスを順に呼ぶだけで、独自のジョブキューは作らない。
並列度はComfyUI側のキューに委ねる。

1件失敗しても残りは続ける。まとめて流したときに途中で止まると、
どこまで進んだのかが分からなくなるため。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.results import GenerationResult

logger: Final = logging.getLogger(__name__)

#: 1件分の生成処理。テストではComfyUIへ接続しないものへ差し替える。
type BatchRunner = Callable[["BatchItem"], Awaitable[GenerationResult]]

#: 進捗の通知。(何件目, 総数, 対象) を受け取る。
type ProgressCallback = Callable[[int, int, "BatchItem"], None]


@dataclass(frozen=True, slots=True)
class BatchItem:
    """一括実行の1件。"""

    #: Specの出どころ。CLIはファイルパス、MCPは受け取った並びの位置を入れる。
    #: ここをパスに固定しないのは、MCP経由ではSpecがファイルとして存在しないため。
    source: str
    spec: GenerationSpec
    #: seed掃引で上書きした場合のみ値が入る。表示の出し分けに使う。
    seed_override: int | None

    @property
    def label(self) -> str:
        """サマリ表示用のラベル。"""
        if self.seed_override is None:
            return self.source
        return f"{self.source} (seed={self.seed_override})"


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """1件分の結果。失敗しても例外にせずここへ入れる。"""

    item: BatchItem
    result: GenerationResult | None
    error: BaseException | None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def expand_seeds(
    pairs: Iterable[tuple[str, GenerationSpec]], *, seeds: Sequence[int] | None
) -> list[BatchItem]:
    """Specとseedの組み合わせを展開する。

    seedsが未指定ならSpecをそのまま1件として扱う。指定された場合は
    Specごとに各seedを当てたものを作る。
    """
    items: list[BatchItem] = []
    for source, spec in pairs:
        if not seeds:
            items.append(BatchItem(source=source, spec=spec, seed_override=None))
            continue
        for seed in seeds:
            generation = spec.generation.model_copy(update={"seed": seed})
            items.append(
                BatchItem(
                    source=source,
                    spec=spec.model_copy(update={"generation": generation}),
                    seed_override=seed,
                )
            )
    return items


async def run_batch(
    items: Sequence[BatchItem],
    *,
    runner: BatchRunner,
    on_progress: ProgressCallback | None = None,
) -> list[BatchOutcome]:
    """各itemを順に実行し、結果をまとめて返す。

    失敗は記録して次へ進む。呼び出し側はまとめてサマリを出せる。
    """
    outcomes: list[BatchOutcome] = []
    total = len(items)

    for index, item in enumerate(items, start=1):
        if on_progress is not None:
            on_progress(index, total, item)
        try:
            result = await runner(item)
        except Exception as exc:
            logger.warning("batch item failed: %s (%s)", item.label, type(exc).__name__)
            outcomes.append(BatchOutcome(item=item, result=None, error=exc))
        else:
            outcomes.append(BatchOutcome(item=item, result=result, error=None))

    return outcomes


__all__ = [
    "BatchItem",
    "BatchOutcome",
    "BatchRunner",
    "ProgressCallback",
    "expand_seeds",
    "run_batch",
]
