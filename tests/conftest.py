"""pytest共通設定。

Integration Testは `-m integration` 指定時のみ実行する。
"""

from collections.abc import Iterable

import pytest


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
