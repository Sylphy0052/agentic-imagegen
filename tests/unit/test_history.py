"""outputs/ から直近の生成を引く collect_history のテスト。

実際の outputs/ には触らない。tmp_path へ metadata.json を並べて完結させる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_imagegen.services.history import collect_history


def _write_run(
    root: Path,
    day: str,
    name: str,
    *,
    created_at: str,
    checkpoint: str | None = "hassakuSD15_v13.safetensors",
    unet: str | None = None,
    presets: dict[str, str] | None = None,
    seed: int = 1234,
    task: str = "txt2img",
    upscale: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    control: dict[str, Any] | None = None,
    text: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    loras: list[dict[str, Any]] | None = None,
    outputs: list[str] | None = None,
) -> Path:
    directory = root / day / name
    directory.mkdir(parents=True)
    metadata = {
        "prompt_id": "p",
        "workflow": "txt2img",
        "created_at": created_at,
        "resolved_seed": seed,
        "outputs": outputs if outputs is not None else ["image_0001.png"],
        "spec": {
            "task": task,
            "presets": presets or {},
            "generation": {"width": 512, "height": 768, "seed": seed, "upscale": upscale},
            "model": {"checkpoint": checkpoint, "unet": unet, "loras": loras or []},
            "source": source,
            "reference": reference,
            "control": control,
            "text": text,
            "output": {"prefix": name.split("_", 1)[-1]},
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


class TestOrderAndLimit:
    def test_newest_first(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "2026-08-14", "100000_old", created_at="2026-08-14T10:00:00+09:00")
        _write_run(tmp_path, "2026-08-15", "090000_new", created_at="2026-08-15T09:00:00+09:00")

        records = collect_history(tmp_path)

        assert [record.directory.name for record in records] == ["090000_new", "100000_old"]

    def test_limit_cuts_the_tail(self, tmp_path: Path) -> None:
        for index in range(5):
            _write_run(
                tmp_path,
                "2026-08-15",
                f"09000{index}_run{index}",
                created_at=f"2026-08-15T09:0{index}:00+09:00",
            )

        assert len(collect_history(tmp_path, limit=2)) == 2

    def test_prefix_filters(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "2026-08-15", "090000_yui_library", created_at="2026-08-15T09:00:00")
        _write_run(tmp_path, "2026-08-15", "091000_cat_street", created_at="2026-08-15T09:10:00")

        records = collect_history(tmp_path, prefix="yui")

        assert [record.directory.name for record in records] == ["090000_yui_library"]


class TestRecordContents:
    def test_reads_model_presets_and_seed(self, tmp_path: Path) -> None:
        _write_run(
            tmp_path,
            "2026-08-15",
            "090000_run",
            created_at="2026-08-15T09:00:00+09:00",
            presets={"style": "sd15-hassaku", "character": "anime-girl-blue"},
            seed=545078971,
        )

        record = collect_history(tmp_path)[0]

        assert record.model == "hassakuSD15_v13.safetensors"
        assert record.seed == 545078971
        assert record.presets == {"style": "sd15-hassaku", "character": "anime-girl-blue"}
        assert record.width == 512
        assert record.height == 768

    def test_uses_unet_for_dit_models(self, tmp_path: Path) -> None:
        """DiT系はcheckpointが無くUNet単体。モデル名の欄を空にしない。"""
        _write_run(
            tmp_path,
            "2026-08-15",
            "090000_anima",
            created_at="2026-08-15T09:00:00",
            checkpoint=None,
            unet="hassakuAnima_v13_int8.safetensors",
        )

        assert collect_history(tmp_path)[0].model == "hassakuAnima_v13_int8.safetensors"

    def test_collects_features(self, tmp_path: Path) -> None:
        """あとで同じ条件を再現するとき、何を使った生成かが一目で要る。"""
        _write_run(
            tmp_path,
            "2026-08-15",
            "090000_full",
            created_at="2026-08-15T09:00:00",
            upscale={"scale": 2.0},
            reference={"image": "inputs/base.png"},
            text={"layers": [{"content": "あ"}]},
            loras=[{"name": "x.safetensors"}],
        )

        record = collect_history(tmp_path)[0]

        assert record.upscale == 2.0
        assert set(record.features) == {"reference", "text", "lora"}

    def test_reads_img2img_source(self, tmp_path: Path) -> None:
        """img2imgは入力画像が分からないと再現できない。"""
        _write_run(
            tmp_path,
            "2026-08-15",
            "090000_up",
            created_at="2026-08-15T09:00:00",
            task="img2img",
            source={"image": "inputs/yui-ref-f.png", "denoise": 0.4},
        )

        assert collect_history(tmp_path)[0].source == "inputs/yui-ref-f.png"

    def test_txt2img_has_no_source(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "2026-08-15", "090000_run", created_at="2026-08-15T09:00:00")

        assert collect_history(tmp_path)[0].source is None

    def test_resolves_output_files(self, tmp_path: Path) -> None:
        """基準画像として inputs/ へ渡せるよう、そのまま使えるパスにする。"""
        directory = _write_run(
            tmp_path, "2026-08-15", "090000_run", created_at="2026-08-15T09:00:00"
        )

        assert collect_history(tmp_path)[0].files == (directory / "image_0001.png",)


class TestRobustness:
    def test_broken_metadata_is_skipped(self, tmp_path: Path) -> None:
        """1件壊れていても残りは引ける。過去の出力は直せない。"""
        _write_run(tmp_path, "2026-08-15", "090000_ok", created_at="2026-08-15T09:00:00")
        broken = tmp_path / "2026-08-15" / "091000_broken"
        broken.mkdir(parents=True)
        (broken / "metadata.json").write_text("{ not json", encoding="utf-8")

        records = collect_history(tmp_path)

        assert [record.directory.name for record in records] == ["090000_ok"]

    def test_directory_without_metadata_is_skipped(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "2026-08-15", "090000_ok", created_at="2026-08-15T09:00:00")
        (tmp_path / "2026-08-15" / "091000_empty").mkdir(parents=True)

        assert len(collect_history(tmp_path)) == 1

    def test_missing_root_is_empty(self, tmp_path: Path) -> None:
        assert collect_history(tmp_path / "nope") == ()
