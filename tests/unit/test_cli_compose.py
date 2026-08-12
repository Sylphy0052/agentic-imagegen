"""compose コマンドの動作とexit codeのテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from agentic_imagegen import cli

runner = CliRunner()

TEXT_ONLY_SPEC = """
layers:
  - content: 秋葉原駅
    font: test.ttf
    size: 32
    color: "#ff0000"
"""

GENERATION_SPEC = """
version: "1"
prompt:
  positive: a street
model:
  checkpoint: v1-5-pruned-emaonly.safetensors
text:
  layers:
    - content: 秋葉原駅
      font: test.ttf
      size: 32
"""

SPEC_WITHOUT_TEXT = """
version: "1"
prompt:
  positive: a street
model:
  checkpoint: v1-5-pruned-emaonly.safetensors
"""


@pytest.fixture
def workspace(tmp_path: Path, fonts_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """作業ルートに入力画像とフォントを揃えたディレクトリ。

    fonts_root fixture が `tmp_path/fonts` を用意するため、既定の探索ルートで引ける。
    """
    assert fonts_root == tmp_path / "fonts"
    (tmp_path / "inputs").mkdir()
    Image.new("RGB", (200, 120), (0, 0, 0)).save(tmp_path / "inputs" / "base.png")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IMAGEGEN_FONTS_ROOT", raising=False)
    return tmp_path


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestComposeCommand:
    def test_writes_to_default_output(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert result.exit_code == 0, result.output
        assert (workspace / "inputs" / "base_text.png").is_file()

    def test_accepts_output_option(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(
            cli.app, ["compose", "inputs/base.png", str(spec), "-o", "outputs/caption.png"]
        )

        assert result.exit_code == 0, result.output
        assert (workspace / "outputs" / "caption.png").is_file()

    def test_prints_output_path(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert "base_text.png" in result.output

    def test_keeps_input_image_unchanged(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)
        original = (workspace / "inputs" / "base.png").read_bytes()

        runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert (workspace / "inputs" / "base.png").read_bytes() == original

    def test_reads_text_section_of_generation_spec(self, workspace: Path) -> None:
        spec = _write(workspace / "spec.yaml", GENERATION_SPEC)

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert result.exit_code == 0, result.output
        assert (workspace / "inputs" / "base_text.png").is_file()


class TestValidateShowsText:
    def test_shows_layer_count_and_fonts(self, workspace: Path) -> None:
        spec = _write(workspace / "spec.yaml", GENERATION_SPEC)

        result = runner.invoke(cli.app, ["validate", str(spec)])

        assert result.exit_code == 0, result.output
        assert "Text: 1 layer(s) (fonts: test.ttf)" in result.output

    def test_hides_line_without_text(self, workspace: Path) -> None:
        spec = _write(workspace / "spec.yaml", SPEC_WITHOUT_TEXT)

        result = runner.invoke(cli.app, ["validate", str(spec)])

        assert result.exit_code == 0, result.output
        assert "Text:" not in result.output


class TestComposeErrors:
    def test_exits_2_when_spec_has_no_text(self, workspace: Path) -> None:
        spec = _write(workspace / "spec.yaml", SPEC_WITHOUT_TEXT)

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert result.exit_code == 2

    def test_exits_2_when_spec_missing(self, workspace: Path) -> None:
        result = runner.invoke(cli.app, ["compose", "inputs/base.png", "absent.yaml"])

        assert result.exit_code == 2

    def test_exits_10_when_font_missing(self, workspace: Path) -> None:
        (workspace / "fonts" / "test.ttf").unlink()
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert result.exit_code == 10

    def test_exits_2_when_image_missing(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(cli.app, ["compose", "inputs/absent.png", str(spec)])

        assert result.exit_code == 2

    def test_exits_2_for_image_outside_root(self, workspace: Path, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.png"
        Image.new("RGB", (10, 10)).save(outside)
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(cli.app, ["compose", str(outside), str(spec)])

        assert result.exit_code == 2

    def test_exits_10_when_output_exists(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)
        (workspace / "inputs" / "base_text.png").write_bytes(b"existing")

        result = runner.invoke(cli.app, ["compose", "inputs/base.png", str(spec)])

        assert result.exit_code == 10
        assert (workspace / "inputs" / "base_text.png").read_bytes() == b"existing"

    def test_exits_2_for_absolute_output_outside_root(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)
        # 既存の test_exits_2_for_image_outside_root と同じ tmp_path.parent (pytestの
        # セッション共有tmpルート) を使うため、ファイル名は衝突しないものにする
        outside = tmp_path.parent / "outside_compose_output.png"

        result = runner.invoke(
            cli.app, ["compose", "inputs/base.png", str(spec), "-o", str(outside)]
        )

        assert result.exit_code == 2
        assert not outside.exists()

    def test_exits_2_for_relative_output_escaping_root(self, workspace: Path) -> None:
        spec = _write(workspace / "text.yaml", TEXT_ONLY_SPEC)

        result = runner.invoke(
            cli.app, ["compose", "inputs/base.png", str(spec), "-o", "../escape.png"]
        )

        assert result.exit_code == 2
        assert not (workspace.parent / "escape.png").exists()
