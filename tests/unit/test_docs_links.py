"""リポジトリ内Markdownの相対リンクが壊れていないことを検証する。

skillは `.claude/skills/imagegen/` から `docs/` を相対参照しており、
ファイル移動で静かに壊れやすい。CIで気づけるようにする。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: 検証対象から外すディレクトリ。生成物・依存・作業用worktreeを含む。
_SKIP_PARTS = frozenset({".git", ".venv", "node_modules", "worktrees", "htmlcov"})

_LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not _SKIP_PARTS & set(path.relative_to(PROJECT_ROOT).parts)
    )


def _relative_links(path: Path) -> list[str]:
    links = _LINK_PATTERN.findall(path.read_text(encoding="utf-8"))
    return [link for link in links if not link.startswith(("http://", "https://", "mailto:", "#"))]


@pytest.mark.parametrize("markdown", _markdown_files(), ids=lambda p: str(p.name))
def test_relative_links_resolve(markdown: Path) -> None:
    broken = [
        link
        for link in _relative_links(markdown)
        if not (markdown.parent / link.split("#")[0]).exists()
    ]

    assert not broken, f"{markdown.relative_to(PROJECT_ROOT)} のリンクが解決できません: {broken}"


def test_markdown_files_are_found() -> None:
    """収集自体が壊れて空パスになっていないことの番人。"""
    assert len(_markdown_files()) >= 5
