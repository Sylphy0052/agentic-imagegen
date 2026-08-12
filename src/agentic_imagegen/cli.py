"""imagegen コマンドのエントリポイント。

MCP導入後もローカルデバッグ・CI・障害切り分けのために残す層であり、
暫定実装ではない。ここでは入出力とexit codeへの変換だけを担い、
実処理は services / adapters へ委譲する。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final

import typer

from agentic_imagegen import __version__
from agentic_imagegen.adapters.comfyui.client import ComfyUIClient, HealthStatus
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import validate_against_limits
from agentic_imagegen.domain.results import GenerationResult
from agentic_imagegen.errors import ComfyUIUnavailable, ImageGenError
from agentic_imagegen.services.generation import generate
from agentic_imagegen.services.spec_loader import load_spec

logger: Final = logging.getLogger(__name__)

UNEXPECTED_ERROR_EXIT_CODE: Final = 1

app = typer.Typer(
    name="imagegen",
    help="GenerationSpecを入力としてComfyUI経由で画像を生成する。",
    no_args_is_help=True,
)

SpecArgument = Annotated[
    Path,
    typer.Argument(metavar="SPEC", help="GenerationSpecのYAMLファイル"),
]
VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="詳細ログを表示する"),
]


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


@app.command()
def health(verbose: VerboseOption = False) -> None:
    """ComfyUIへ到達できるかを確認する。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        try:
            status = asyncio.run(_check_health(settings))
        except ComfyUIUnavailable as exc:
            typer.echo("ComfyUI: unreachable", err=True)
            typer.echo(f"URL: {settings.comfyui_base_url}", err=True)
            typer.echo(str(exc), err=True)
            raise typer.Exit(exc.exit_code) from exc

        typer.echo("ComfyUI: reachable")
        typer.echo(f"URL: {status.base_url}")
        if status.comfyui_version:
            typer.echo(f"Version: {status.comfyui_version}")
        if status.devices:
            typer.echo(f"Devices: {', '.join(status.devices)}")


@app.command()
def validate(spec_path: SpecArgument, verbose: VerboseOption = False) -> None:
    """GenerationSpecを検証する。ComfyUIへは接続しない。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        spec = _load_and_validate(spec_path, settings)
        params = spec.generation

        typer.echo("OK")
        typer.echo(f"Spec: {spec_path}")
        typer.echo(f"Workflow: {spec.task}")
        typer.echo(f"Resolution: {params.width}x{params.height} (batch {params.batch_size})")
        typer.echo(f"Checkpoint: {spec.model.checkpoint}")
        if not spec.presets.is_empty():
            applied = ", ".join(
                f"{kind}={name}"
                for kind, name in (
                    ("character", spec.presets.character),
                    ("scene", spec.presets.scene),
                    ("style", spec.presets.style),
                )
                if name is not None
            )
            typer.echo(f"Presets: {applied}")


@app.command(name="generate")
def generate_images(
    spec_path: SpecArgument,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="生成のタイムアウト秒 (既定は IMAGEGEN_TIMEOUT)"),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """GenerationSpecに従って画像を生成する。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        spec = _load_and_validate(spec_path, settings)
        result = asyncio.run(_run_generation(spec, settings, timeout))

        typer.echo(f"prompt_id: {result.prompt_id}")
        typer.echo(f"seed: {result.seed}")
        typer.echo(f"directory: {result.directory}")
        for path in result.files:
            typer.echo(str(path))
        typer.echo(f"metadata: {result.metadata_path}")


@contextmanager
def _handled_errors() -> Iterator[None]:
    """アプリケーション例外をexit codeへ変換する。

    内部トレースバックはそのまま表示せず、原因が特定できるメッセージだけを出す。
    詳細は --verbose 指定時のログで確認する。
    """
    try:
        yield
    except ImageGenError as exc:
        logger.debug("command failed", exc_info=exc)
        typer.echo(str(exc), err=True)
        raise typer.Exit(exc.exit_code) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        logger.debug("unexpected failure", exc_info=exc)
        typer.echo(f"想定外のエラーが発生しました: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(UNEXPECTED_ERROR_EXIT_CODE) from exc


def _load_and_validate(spec_path: Path, settings: Settings) -> GenerationSpec:
    spec = load_spec(spec_path, presets_root=settings.presets_root)
    validate_against_limits(spec, settings)
    return spec


async def _check_health(settings: Settings) -> HealthStatus:
    async with ComfyUIClient(settings) as client:
        return await client.health()


async def _run_generation(
    spec: GenerationSpec, settings: Settings, timeout: float | None
) -> GenerationResult:
    async with ComfyUIClient(settings) as client:
        return await generate(
            spec,
            settings,
            backend=client,
            project_root=Path.cwd(),
            timeout=timeout,
        )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """コンソールスクリプト用のエントリポイント。"""
    app()


if __name__ == "__main__":
    main()
