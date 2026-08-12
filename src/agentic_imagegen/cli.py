"""imagegen コマンドのエントリポイント。

Phase 1 Step 1 時点ではスキャフォールドのみ。
health / validate / generate は Step 4 以降で追加する。
"""

from __future__ import annotations

import typer

from agentic_imagegen import __version__

app = typer.Typer(
    name="imagegen",
    help="GenerationSpecを入力としてComfyUI経由で画像を生成する。",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """コマンド群のルート。

    Typerはコマンドが1つだけの場合にサブコマンド構造を畳んでしまうため、
    明示的なcallbackを置いて `imagegen <command>` の形を固定する。
    """


@app.command()
def version() -> None:
    """バージョンを表示する。"""
    typer.echo(__version__)


def main() -> None:
    """コンソールスクリプト用のエントリポイント。"""
    app()


if __name__ == "__main__":
    main()
