"""imagegen skillのedge_stats.pyを検証する。

このスクリプトは `src/` の外 (`.claude/skills/`) にあるためパッケージとして
importできない。ファイルパスから直接ロードする。

生成した画像は使わず、性質の分かっている合成画像で指標の振る舞いだけを見る。
"""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / ".claude" / "skills" / "imagegen" / "scripts" / "edge_stats.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("edge_stats", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def edge_stats() -> types.ModuleType:
    return _load_module()


def _flat(path: Path) -> Path:
    Image.new("RGB", (64, 64), (128, 128, 128)).save(path)
    return path


def _checkerboard(path: Path) -> Path:
    image = Image.new("L", (64, 64))
    image.putdata([0 if (x + y) % 2 else 255 for y in range(64) for x in range(64)])
    image.save(path)
    return path


class TestEdgeRatio:
    def test_flat_image_has_no_edges(self, edge_stats: types.ModuleType, tmp_path: Path) -> None:
        assert edge_stats.edge_ratio(_flat(tmp_path / "flat.png")) == 0.0

    def test_checkerboard_is_almost_all_edges(
        self, edge_stats: types.ModuleType, tmp_path: Path
    ) -> None:
        """破綻した画像は画素の大半がエッジになる。0.2を大きく超える側の代表。"""
        assert edge_stats.edge_ratio(_checkerboard(tmp_path / "checker.png")) >= 0.5

    def test_higher_threshold_never_increases_the_ratio(
        self, edge_stats: types.ModuleType, tmp_path: Path
    ) -> None:
        path = _checkerboard(tmp_path / "checker.png")

        loose = edge_stats.edge_ratio(path, threshold=8)
        strict = edge_stats.edge_ratio(path, threshold=200)

        assert strict <= loose

    def test_ratio_is_between_zero_and_one(
        self, edge_stats: types.ModuleType, tmp_path: Path
    ) -> None:
        assert 0.0 <= edge_stats.edge_ratio(_checkerboard(tmp_path / "checker.png")) <= 1.0

    def test_default_threshold_is_documented_in_the_module(
        self, edge_stats: types.ModuleType
    ) -> None:
        """しきい値を変えると値が変わる。既定値は定数として1箇所に置く。"""
        assert edge_stats.DEFAULT_THRESHOLD == 24


class TestCli:
    def test_prints_ratio_per_image(
        self, edge_stats: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _flat(tmp_path / "flat.png")

        assert edge_stats.main([str(path)]) == 0

        out = capsys.readouterr().out
        assert "flat.png" in out
        assert "0.000" in out

    def test_json_output_is_machine_readable(
        self, edge_stats: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _checkerboard(tmp_path / "checker.png")

        assert edge_stats.main([str(path), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["path"].endswith("checker.png")
        assert payload[0]["edge_ratio"] >= 0.5
        assert payload[0]["threshold"] == edge_stats.DEFAULT_THRESHOLD

    def test_accepts_a_directory(
        self, edge_stats: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """生成結果はディレクトリ単位で出るため、そのまま渡せるようにする。"""
        directory = tmp_path / "run"
        directory.mkdir()
        _flat(directory / "image_0001.png")
        _checkerboard(directory / "image_0002.png")

        assert edge_stats.main([str(directory), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 2

    def test_missing_path_is_reported(
        self, edge_stats: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert edge_stats.main([str(tmp_path / "missing.png")]) == 1
        assert "missing.png" in capsys.readouterr().err

    def test_non_image_is_reported(
        self, edge_stats: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = tmp_path / "broken.png"
        broken.write_text("not an image", encoding="utf-8")

        assert edge_stats.main([str(broken)]) == 1
        assert "broken.png" in capsys.readouterr().err
