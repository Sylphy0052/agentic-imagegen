"""pytest共通設定。

Integration Testは `-m integration` 指定時のみ実行する。
"""

from collections.abc import Iterable
from pathlib import Path

import pytest

#: テキスト合成のテストで使うTrueTypeフォントの探索先。
#: リポジトリへフォントを同梱しないため、環境にあるものを借りる。
_FONT_SEARCH_ROOTS: tuple[Path, ...] = (
    Path("fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".local/share/fonts",
    Path("/mnt/c/Windows/Fonts"),
)

_FONT_SUFFIXES: frozenset[str] = frozenset({".ttf", ".otf"})


def _find_truetype_font() -> Path | None:
    for root in _FONT_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _FONT_SUFFIXES:
                return path
    return None


@pytest.fixture(scope="session")
def truetype_font_source() -> Path:
    """環境にあるTrueTypeフォント1件。見つからなければテストをskipする。

    描画結果の検証にはグリフの見た目ではなくレイアウトを使うため、
    日本語フォントである必要はない。
    """
    font = _find_truetype_font()
    if font is None:
        pytest.skip("TrueTypeフォントが見つからないため、テキスト合成のテストをskipします")
    return font


@pytest.fixture
def fonts_root(tmp_path: Path, truetype_font_source: Path) -> Path:
    """テスト用のフォントルート。`test.ttf` として1件だけ置く。"""
    root = tmp_path / "fonts"
    root.mkdir()
    (root / "test.ttf").write_bytes(truetype_font_source.read_bytes())
    return root


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """integrationマーカー付きテストは、明示指定がない限りskipする。"""
    if "integration" in (config.getoption("-m") or ""):
        return

    skip_integration = pytest.mark.skip(
        reason="ComfyUI起動が必要。実行するには `uv run pytest -m integration`"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
