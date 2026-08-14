"""services層がadapters層に直接依存していないことを保証する。

GitHub Issue #31 (Backend抽象の導入判断) で指摘された層違反の再発防止テスト。
Domain / Service層はComfyUI固有の事情 (HTTPクライアントやWebSocketの形状) を
知らないまま、`services.generation.GenerationBackend` や
`services.catalog.CatalogBackend` のようなProtocol越しにバックエンドを扱う設計に
している。services配下のモジュールが `agentic_imagegen.adapters` を直接importすると
この境界が崩れ、将来2つ目のバックエンドを足すときにservice層の書き換えが必要に
なってしまう。composition root (CLI / MCP Server) だけがadapters層の具象を知って
よい。
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVICES_ROOT: Path = Path(__file__).resolve().parents[2] / "src" / "agentic_imagegen" / "services"

#: このprefixで始まるimportがあればadapters層への直接依存とみなす。
_FORBIDDEN_PREFIX = "agentic_imagegen.adapters"


def _imported_module_names(path: Path) -> set[str]:
    """1ファイル分のimport文から、importしているモジュール名の集合を作る。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_services_do_not_import_adapters() -> None:
    """services配下のどの `.py` も `agentic_imagegen.adapters` をimportしない。

    以前は services/mcp_tools.py だけが ComfyUIClient を直接importしており、
    services/generation.py が GenerationBackend Protocol 越しに徹しているのと
    非対称だった (Issue #31)。列挙系にも CatalogBackend Protocol を揃え、
    ComfyUIの具象生成は composition root である mcp_server.py / cli.py へ
    寄せることでこの非対称を解消した。
    """
    violations: dict[str, set[str]] = {}
    for path in sorted(SERVICES_ROOT.rglob("*.py")):
        imported = {
            name for name in _imported_module_names(path) if name.startswith(_FORBIDDEN_PREFIX)
        }
        if imported:
            violations[path.name] = imported

    assert violations == {}, f"services層からadapters層への直接import: {violations}"
