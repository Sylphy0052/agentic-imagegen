"""pytest共通設定。

Integration Testは `-m integration` 指定時のみ実行する。
"""

from collections.abc import Iterable
from pathlib import Path

import pytest

from synthetic_font import build_ttf_bytes


@pytest.fixture(scope="session")
def truetype_font_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """テキスト合成のテストで使うTrueTypeフォント1件。

    ホスト環境のフォントを探して借りると、フォント未導入の環境ではテストが
    静かにskipされ「終了コードは0だが実際は検証されていない」状態になる。
    それを避けるため、`fontTools` でその場に最小のTTFを組み立てる
    (実体は tests/synthetic_font.py)。全文字が同じ幅の塗り潰し矩形として
    描かれるため、描画結果の検証にはグリフの見た目ではなくレイアウトを使う
    既存のテストとそのまま整合する。
    """
    font_path = tmp_path_factory.mktemp("synthetic-font") / "block.ttf"
    font_path.write_bytes(build_ttf_bytes())
    return font_path


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
        reason="実機 (ComfyUI起動またはモデル読み込み) が必要。"
        "実行するには `uv run pytest -m integration`"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
