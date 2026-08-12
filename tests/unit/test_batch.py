"""複数Specの一括実行。

1件失敗しても残りを続ける。まとめて流すときに、途中で止まると
どこまで進んだのか分からなくなるため。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import ComfyUIUnavailable, GenerationFailed
from agentic_imagegen.services.batch import BatchItem, expand_seeds, run_batch

SPEC: dict[str, Any] = {
    "version": "1",
    "task": "txt2img",
    "prompt": {"positive": "1girl"},
    "generation": {"seed": 100},
    "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
}


def _spec(**overrides: Any) -> GenerationSpec:
    payload = {**SPEC, **overrides}
    return GenerationSpec.model_validate(payload)


def _result(tmp_path: Path, seed: int) -> GenerationResult:
    return GenerationResult(
        prompt_id=f"p-{seed}",
        seed=seed,
        directory=tmp_path,
        files=(tmp_path / f"image_{seed}.png",),
        metadata_path=tmp_path / "metadata.json",
    )


class TestExpandSeeds:
    def test_without_seeds_keeps_spec_as_is(self, tmp_path: Path) -> None:
        items = expand_seeds([(Path("a.yaml"), _spec())], seeds=None)

        assert len(items) == 1
        assert items[0].spec.generation.seed == 100
        assert items[0].seed_override is None

    def test_expands_each_spec_per_seed(self) -> None:
        items = expand_seeds([(Path("a.yaml"), _spec())], seeds=[1, 2, 3])

        assert [item.spec.generation.seed for item in items] == [1, 2, 3]
        assert [item.seed_override for item in items] == [1, 2, 3]

    def test_keeps_other_fields(self) -> None:
        items = expand_seeds([(Path("a.yaml"), _spec())], seeds=[7])

        assert items[0].spec.prompt.positive == "1girl"
        assert items[0].spec.model.checkpoint == "v1-5-pruned-emaonly.safetensors"

    def test_multiple_specs_are_expanded(self) -> None:
        pairs = [(Path("a.yaml"), _spec()), (Path("b.yaml"), _spec())]

        items = expand_seeds(pairs, seeds=[1, 2])

        assert [item.spec_path.name for item in items] == [
            "a.yaml",
            "a.yaml",
            "b.yaml",
            "b.yaml",
        ]


class TestRunBatch:
    async def test_runs_all_items(self, tmp_path: Path) -> None:
        items = [
            BatchItem(spec_path=Path("a.yaml"), spec=_spec(), seed_override=None),
            BatchItem(spec_path=Path("b.yaml"), spec=_spec(), seed_override=None),
        ]
        calls: list[Path] = []

        async def runner(item: BatchItem) -> GenerationResult:
            calls.append(item.spec_path)
            return _result(tmp_path, 1)

        outcomes = await run_batch(items, runner=runner)

        assert len(outcomes) == 2
        assert all(outcome.succeeded for outcome in outcomes)
        assert calls == [Path("a.yaml"), Path("b.yaml")]

    async def test_continues_after_failure(self, tmp_path: Path) -> None:
        """1件失敗しても残りを続ける。"""
        items = [
            BatchItem(spec_path=Path("a.yaml"), spec=_spec(), seed_override=None),
            BatchItem(spec_path=Path("b.yaml"), spec=_spec(), seed_override=None),
            BatchItem(spec_path=Path("c.yaml"), spec=_spec(), seed_override=None),
        ]

        async def runner(item: BatchItem) -> GenerationResult:
            if item.spec_path.name == "b.yaml":
                raise GenerationFailed("ComfyUI側で失敗しました")
            return _result(tmp_path, 1)

        outcomes = await run_batch(items, runner=runner)

        assert [outcome.succeeded for outcome in outcomes] == [True, False, True]
        assert outcomes[1].error is not None
        assert outcomes[1].error.exit_code == 7

    async def test_records_error_per_item(self, tmp_path: Path) -> None:
        items = [BatchItem(spec_path=Path("a.yaml"), spec=_spec(), seed_override=None)]

        async def runner(item: BatchItem) -> GenerationResult:
            raise ComfyUIUnavailable("接続できません")

        outcomes = await run_batch(items, runner=runner)

        assert outcomes[0].succeeded is False
        assert outcomes[0].result is None
        assert outcomes[0].error is not None
        assert outcomes[0].error.exit_code == 3

    async def test_reports_progress(self, tmp_path: Path) -> None:
        """進捗を呼び出し側へ伝えられること。"""
        items = [
            BatchItem(spec_path=Path("a.yaml"), spec=_spec(), seed_override=None),
            BatchItem(spec_path=Path("b.yaml"), spec=_spec(), seed_override=None),
        ]
        progress: list[tuple[int, int]] = []

        async def runner(item: BatchItem) -> GenerationResult:
            return _result(tmp_path, 1)

        await run_batch(
            items,
            runner=runner,
            on_progress=lambda index, total, item: progress.append((index, total)),
        )

        assert progress == [(1, 2), (2, 2)]

    async def test_unexpected_error_is_recorded(self, tmp_path: Path) -> None:
        """ImageGenError以外でもバッチ全体を止めない。"""
        items = [BatchItem(spec_path=Path("a.yaml"), spec=_spec(), seed_override=None)]

        async def runner(item: BatchItem) -> GenerationResult:
            raise RuntimeError("想定外")

        outcomes = await run_batch(items, runner=runner)

        assert outcomes[0].succeeded is False
        assert outcomes[0].error is not None

    async def test_empty_items(self) -> None:
        async def runner(item: BatchItem) -> GenerationResult:  # pragma: no cover
            raise AssertionError("呼ばれないはず")

        assert await run_batch([], runner=runner) == []


def test_batch_item_label() -> None:
    """サマリ表示に使うラベル。seed掃引時はseedも出す。"""
    plain = BatchItem(spec_path=Path("specs/a.yaml"), spec=_spec(), seed_override=None)
    swept = BatchItem(spec_path=Path("specs/a.yaml"), spec=_spec(), seed_override=42)

    assert plain.label == "specs/a.yaml"
    assert swept.label == "specs/a.yaml (seed=42)"
