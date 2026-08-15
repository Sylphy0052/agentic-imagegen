"""過去の生成を `metadata.json` から引く。

生成のたびに `outputs/<日付>/<時刻>_<prefix>/metadata.json` が残る。ここはそれを
読むだけで、ComfyUIへは接続しない。Specも出力もgit管理外なので、セッションを
跨いだあとに「さっきの子」「前回の設定」を辿れる手がかりはこのファイルしかない。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from agentic_imagegen.domain.results import RunRecord

#: 一覧に出す既定の件数。
DEFAULT_LIMIT = 10
#: features へ立てる項目と、Specのどのブロックを見るか。
FEATURE_BLOCKS = ("reference", "control", "text")


def collect_history(
    output_root: Path, *, limit: int = DEFAULT_LIMIT, prefix: str | None = None
) -> tuple[RunRecord, ...]:
    """新しい順に生成結果を返す。読めないものは飛ばす。

    過去の出力は後から直せない。1件壊れていても残りが引けることを優先する。
    """
    records = [
        record
        for record in _iter_records(output_root)
        if prefix is None or prefix in record.directory.name
    ]
    records.sort(key=lambda record: record.created_at, reverse=True)
    return tuple(records[:limit])


def _iter_records(output_root: Path) -> Iterator[RunRecord]:
    if not output_root.is_dir():
        return
    for metadata_path in sorted(output_root.glob("*/*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            yield _to_record(metadata_path.parent, metadata)
        except (KeyError, TypeError, AttributeError):
            continue


def _to_record(directory: Path, metadata: Mapping[str, Any]) -> RunRecord:
    spec = metadata["spec"]
    generation = spec.get("generation") or {}
    model = spec.get("model") or {}
    upscale = generation.get("upscale") or {}
    return RunRecord(
        directory=directory,
        created_at=str(metadata.get("created_at", "")),
        task=str(spec.get("task", "")),
        model=str(model.get("checkpoint") or model.get("unet") or ""),
        presets={axis: name for axis, name in (spec.get("presets") or {}).items() if name},
        seed=int(metadata.get("resolved_seed", generation.get("seed", -1))),
        width=int(generation.get("width") or 0),
        height=int(generation.get("height") or 0),
        source=(spec.get("source") or {}).get("image"),
        upscale=float(upscale["scale"]) if upscale.get("scale") else None,
        features=_features(spec),
        files=tuple(directory / name for name in metadata.get("outputs", ())),
        workflow=str(metadata.get("workflow", "")),
    )


def _features(spec: Mapping[str, Any]) -> tuple[str, ...]:
    found = [block for block in FEATURE_BLOCKS if spec.get(block)]
    if (spec.get("model") or {}).get("loras"):
        found.append("lora")
    return tuple(found)


__all__ = ["DEFAULT_LIMIT", "collect_history"]
