"""フォントの実体解決と設定の検証。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_imagegen.config import DEFAULT_FONTS_ROOT, Settings
from agentic_imagegen.domain.policy import resolve_font
from agentic_imagegen.errors import InvalidConfiguration, TextCompositionError


@pytest.fixture
def fonts_root(tmp_path: Path) -> Path:
    root = tmp_path / "fonts"
    root.mkdir()
    return root


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")
    return path


class TestResolveFont:
    def test_resolves_font_at_root(self, fonts_root: Path) -> None:
        expected = _touch(fonts_root / "NotoSansJP.ttf")

        assert resolve_font("NotoSansJP.ttf", fonts_root) == expected

    def test_resolves_font_in_subfolder(self, fonts_root: Path) -> None:
        expected = _touch(fonts_root / "noto" / "NotoSansJP.ttf")

        assert resolve_font("noto/NotoSansJP.ttf", fonts_root) == expected

    def test_lists_candidates_when_not_found(self, fonts_root: Path) -> None:
        _touch(fonts_root / "IPAGothic.ttf")
        _touch(fonts_root / "noto" / "NotoSansJP.otf")

        with pytest.raises(TextCompositionError) as excinfo:
            resolve_font("Missing.ttf", fonts_root)

        message = str(excinfo.value)
        assert "Missing.ttf" in message
        assert "IPAGothic.ttf" in message
        assert "noto/NotoSansJP.otf" in message

    def test_shows_root_when_no_candidates(self, fonts_root: Path) -> None:
        with pytest.raises(TextCompositionError, match=str(fonts_root)):
            resolve_font("Missing.ttf", fonts_root)

    def test_fails_when_root_absent(self, tmp_path: Path) -> None:
        with pytest.raises(TextCompositionError):
            resolve_font("NotoSansJP.ttf", tmp_path / "absent")

    def test_rejects_directory(self, fonts_root: Path) -> None:
        (fonts_root / "NotoSansJP.ttf").mkdir()

        with pytest.raises(TextCompositionError):
            resolve_font("NotoSansJP.ttf", fonts_root)

    def test_rejects_symlink_escaping_root(self, fonts_root: Path, tmp_path: Path) -> None:
        outside = _touch(tmp_path / "outside" / "Secret.ttf")
        (fonts_root / "Secret.ttf").symlink_to(outside)

        with pytest.raises(TextCompositionError, match="外"):
            resolve_font("Secret.ttf", fonts_root)

    def test_limits_listed_candidates(self, fonts_root: Path) -> None:
        for index in range(30):
            _touch(fonts_root / f"font{index:02d}.ttf")

        with pytest.raises(TextCompositionError) as excinfo:
            resolve_font("Missing.ttf", fonts_root)

        assert "他" in str(excinfo.value)


class TestFontsRootSetting:
    def test_default_is_fonts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IMAGEGEN_FONTS_ROOT", raising=False)

        assert Settings.from_env().fonts_root == Path(DEFAULT_FONTS_ROOT)

    def test_overridable_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGEGEN_FONTS_ROOT", "assets/fonts")

        assert Settings.from_env().fonts_root == Path("assets/fonts")

    def test_rejects_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGEGEN_FONTS_ROOT", "   ")

        with pytest.raises(InvalidConfiguration, match="IMAGEGEN_FONTS_ROOT"):
            Settings.from_env()


class TestTextCompositionError:
    def test_exit_code_is_10(self) -> None:
        assert TextCompositionError("x").exit_code == 10
