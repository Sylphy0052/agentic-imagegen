"""imagegen コマンドのエントリポイント。

MCP導入後もローカルデバッグ・CI・障害切り分けのために残す層であり、
暫定実装ではない。ここでは入出力とexit codeへの変換だけを担い、
実処理は services / adapters へ委譲する。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final

import typer

from agentic_imagegen import __version__
from agentic_imagegen.backends import open_catalog_backend, open_generation_backend
from agentic_imagegen.config import Settings
from agentic_imagegen.domain.characters import Character
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.policy import (
    resolve_compose_output,
    resolve_source_image,
    validate_against_limits,
)
from agentic_imagegen.domain.results import (
    CatalogSnapshot,
    GenerationResult,
    HealthStatus,
    RunRecord,
)
from agentic_imagegen.errors import ComfyUIUnavailable, ImageGenError, InvalidGenerationSpec
from agentic_imagegen.services.batch import BatchItem, BatchOutcome, expand_seeds, run_batch
from agentic_imagegen.services.catalog import CATALOG_KINDS, collect_catalog
from agentic_imagegen.services.characters import collect_characters, load_character
from agentic_imagegen.services.compose import compose_text
from agentic_imagegen.services.estimate import estimate_duration, format_duration
from agentic_imagegen.services.generation import TEXT_SUFFIX, generate, resolve_fonts_root
from agentic_imagegen.services.history import DEFAULT_LIMIT, collect_history
from agentic_imagegen.services.preset_advice import style_warnings
from agentic_imagegen.services.spec_loader import load_spec, load_text_spec
from agentic_imagegen.workflows.injector import resolve_workflow_name

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
            typer.echo(f"{settings.backend}: unreachable", err=True)
            typer.echo(f"URL: {settings.comfyui_base_url}", err=True)
            typer.echo(str(exc), err=True)
            raise typer.Exit(exc.exit_code) from exc

        typer.echo(f"{settings.backend}: reachable")
        typer.echo(f"URL: {status.base_url}")
        if status.comfyui_version:
            typer.echo(f"Version: {status.comfyui_version}")
        if status.devices:
            typer.echo(f"Devices: {', '.join(status.devices)}")


@app.command()
def catalog(
    as_json: Annotated[bool, typer.Option("--json", help="機械可読な形で出力する")] = False,
    verbose: VerboseOption = False,
) -> None:
    """使えるモデル・preset・フォントを一覧する。

    ComfyUIへ到達できればそこから、できなければ COMFYUI_HOME 配下の
    modelsディレクトリから読む。探索のためだけにComfyUIを起動しなくてよい。
    """
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        project_root = Path.cwd()
        status = _probe_health(settings)
        snapshot = asyncio.run(
            collect_catalog(
                settings,
                backend_factory=open_catalog_backend,
                comfyui_home=settings.comfyui_home,
                presets_root=_resolve_root(settings.presets_root, project_root),
                fonts_root=resolve_fonts_root(settings, project_root),
            )
        )

        if as_json:
            payload = _catalog_payload(snapshot, settings, status)
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        _print_catalog(snapshot, settings, status)


@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", min=1, help="出す件数")] = DEFAULT_LIMIT,
    prefix: Annotated[
        str | None, typer.Option("--prefix", help="出力ディレクトリ名で絞り込む")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="機械可読な形で出力する")] = False,
    verbose: VerboseOption = False,
) -> None:
    """直近の生成を新しい順に出す。ComfyUIへは接続しない。

    「さっきの子で別の場面を」のような要求で、基準画像と使ったseedを推測ではなく
    記録から引くために使う。
    """
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        root = _resolve_root(settings.output_root, Path.cwd())
        records = collect_history(root, limit=limit, prefix=prefix)

        if as_json:
            payload = [_history_payload(record) for record in records]
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        _print_history(records)


character_app = typer.Typer(help="キャラクタ台帳を引く。ComfyUIへは接続しない。")
app.add_typer(character_app, name="character")


@character_app.command("list")
def character_list(
    as_json: Annotated[bool, typer.Option("--json", help="機械可読な形で出力する")] = False,
    verbose: VerboseOption = False,
) -> None:
    """台帳にあるキャラクタを名前順に出す。"""
    _configure_logging(verbose)

    with _handled_errors():
        characters = _collect_characters()

        if as_json:
            payload = [_character_payload(character) for character in characters]
            typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if not characters:
            typer.echo("(なし)")
            return
        for character in characters:
            summary = character.record.description or "(説明なし)"
            mark = " (参照先に欠落あり)" if character.missing else ""
            typer.echo(f"{character.name}: {summary}{mark}")


@character_app.command("show")
def character_show(
    name: Annotated[str, typer.Argument(help="台帳のファイル名 (拡張子なし)")],
    as_json: Annotated[bool, typer.Option("--json", help="機械可読な形で出力する")] = False,
    verbose: VerboseOption = False,
) -> None:
    """キャラクタ1件を出す。Specへ書き写す値がそのまま並ぶ。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        project_root = Path.cwd()
        character = load_character(
            name,
            registry_root=_resolve_root(settings.registry_root, project_root),
            presets_root=_resolve_root(settings.presets_root, project_root),
            project_root=project_root,
        )

        if as_json:
            typer.echo(json.dumps(_character_payload(character), ensure_ascii=False, indent=2))
            return

        _print_character(character)


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
        # task ではなく実際に使うテンプレート名を出す (LoRA指定で切り替わるため)
        typer.echo(f"Workflow: {resolve_workflow_name(spec)}")
        if spec.source is not None:
            # img2imgは入力画像のサイズをそのまま使うため、解像度は表示しない
            typer.echo(f"Source: {spec.source.image} (denoise {spec.source.denoise})")
        else:
            typer.echo(f"Resolution: {params.width}x{params.height} (batch {params.batch_size})")
        if spec.model.uses_separate_loaders:
            typer.echo(f"UNet: {spec.model.unet}")
            typer.echo(f"CLIP: {spec.model.clip}")
            typer.echo(f"VAE: {spec.model.vae}")
        else:
            typer.echo(f"Checkpoint: {spec.model.checkpoint}")
        if spec.control is not None:
            typer.echo(
                f"ControlNet: {spec.control.image} "
                f"(model={spec.control.model}, strength={spec.control.strength})"
            )
        if spec.reference is not None:
            typer.echo(
                f"IPAdapter: {spec.reference.image} "
                f"(model={spec.reference.model}, weight={spec.reference.weight})"
            )
        for lora in spec.model.loras:
            typer.echo(
                f"LoRA: {lora.name} (model={lora.strength_model}, clip={lora.strength_clip})"
            )
        if spec.text is not None:
            fonts = ", ".join(sorted({layer.font for layer in spec.text.layers}))
            typer.echo(f"Text: {len(spec.text.layers)} layer(s) (fonts: {fonts})")
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

        estimate = estimate_duration(spec)
        if estimate is not None:
            # validateはComfyUIへ接続しないため、どちらの実行基盤で動くかは分からない。
            typer.echo(
                f"Estimate: XPU {format_duration(estimate.xpu_seconds)} / "
                f"CPU {format_duration(estimate.cpu_seconds)}"
            )

        # 助言は検証結果ではない。exit codeは変えず、stdoutの検証結果とも混ぜない。
        advice = style_warnings(
            spec,
            presets_root=_resolve_root(settings.presets_root, Path.cwd()),
            project_root=Path.cwd(),
        )
        if estimate is not None and estimate.xpu_seconds > settings.timeout_seconds:
            advice += (
                f"XPUでも{format_duration(estimate.xpu_seconds)}かかる見込みで、"
                f"IMAGEGEN_TIMEOUT ({settings.timeout_seconds}秒) を超えます。"
                "steps・解像度・batch_sizeを下げるか、IMAGEGEN_TIMEOUTを延ばしてください",
            )
        for warning in advice:
            typer.echo(f"warning: {warning}", err=True)


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
        # 合成結果は生成そのままの画像とは別ファイル。取り違えないよう印を付ける
        for path in result.text_files:
            typer.echo(f"text: {path}")
        typer.echo(f"metadata: {result.metadata_path}")


@app.command(name="compose")
def compose_image(
    image_path: Annotated[
        Path,
        typer.Argument(metavar="IMAGE", help="テキストを合成する画像 (作業ルート配下)"),
    ],
    spec_path: Annotated[
        Path,
        typer.Argument(metavar="SPEC", help="テキスト定義のYAML (生成用Specのtext:でもよい)"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="出力先 (既定は <元の名前>_text.<元の拡張子>)"),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """既存の画像へテキストを合成する。入力画像は変更しない。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        project_root = Path.cwd()
        source = resolve_source_image(
            _relative_to_root(image_path, project_root),
            project_root,
            max_bytes=settings.max_source_bytes,
        )
        spec = load_text_spec(spec_path)
        destination = (
            resolve_compose_output(output, project_root)
            if output is not None
            else source.with_name(f"{source.stem}{TEXT_SUFFIX}{source.suffix}")
        )

        result = compose_text(
            image=source,
            spec=spec,
            fonts_root=resolve_fonts_root(settings, project_root),
            output=destination,
            max_pixels=settings.max_pixels,
            project_root=project_root,
        )
        typer.echo(str(result.output))


def _resolve_root(root: Path, project_root: Path) -> Path:
    """相対指定の探索ルートを作業ルート基準の絶対パスへ解く。"""
    return root if root.is_absolute() else project_root / root


#: checkpointを決めていないときの既定。対応するstyle presetを添えて示す。
DEFAULT_CHECKPOINT: Final = "hassakuSD15_v13.safetensors"
DEFAULT_STYLE_PRESET: Final = "sd15-hassaku"


def _probe_health(settings: Settings) -> HealthStatus | None:
    """catalog の先頭に添える実行基盤の情報。到達できなければ None。

    所要時間がXPUとCPUで一桁違うため、在庫と同じ画面で見えないと選定に使えない。
    """
    try:
        return asyncio.run(_check_health(settings))
    except ComfyUIUnavailable:
        return None


def _catalog_payload(
    snapshot: CatalogSnapshot, settings: Settings, status: HealthStatus | None
) -> dict[str, object]:
    return {
        "source": snapshot.source,
        "base_url": settings.comfyui_base_url,
        "version": status.comfyui_version if status else None,
        "devices": list(status.devices) if status else [],
        "models": {name: list(names) for name, names in snapshot.models.items()},
        "presets": {axis: list(names) for axis, names in snapshot.presets.items()},
        "fonts": list(snapshot.fonts),
    }


def _print_catalog(
    snapshot: CatalogSnapshot, settings: Settings, status: HealthStatus | None
) -> None:
    """人が読む形で一覧を出す。

    先頭に取得元を出す。`filesystem` はComfyUIを起動せずに見た結果で、
    カスタムノード由来の種別が実際に使えるかまでは分からない。
    """
    if snapshot.source == "api":
        typer.echo(f"Backend: api ({settings.comfyui_base_url})")
    else:
        typer.echo("Backend: unavailable (filesystem fallback)")
        typer.echo(f"ComfyUI home: {settings.comfyui_home}")

    if status and status.comfyui_version:
        typer.echo(f"Version: {status.comfyui_version}")
    if status and status.devices:
        typer.echo(f"Devices: {', '.join(status.devices)}")

    presets = " / ".join(f"{axis} {len(names)}" for axis, names in snapshot.presets.items())
    typer.echo(f"Presets: {presets}")

    for axis, names in snapshot.presets.items():
        typer.echo(f"\npresets/{axis} ({len(names)})")
        _echo_names(names)

    for kind in CATALOG_KINDS:
        names = snapshot.models.get(kind.name, ())
        typer.echo(f"\n{kind.name} ({len(names)})")
        _echo_names(names, annotate=kind.name == "checkpoints")

    typer.echo(f"\nfonts ({len(snapshot.fonts)})")
    _echo_names(snapshot.fonts)


def _echo_names(names: tuple[str, ...], *, annotate: bool = False) -> None:
    if not names:
        typer.echo("  (なし)")
        return
    for name in names:
        suffix = ""
        if annotate and name == DEFAULT_CHECKPOINT:
            suffix = f"  <- 既定 ({DEFAULT_STYLE_PRESET})"
        typer.echo(f"  {name}{suffix}")


def _collect_characters() -> tuple[Character, ...]:
    settings = Settings.from_env()
    project_root = Path.cwd()
    return collect_characters(
        registry_root=_resolve_root(settings.registry_root, project_root),
        presets_root=_resolve_root(settings.presets_root, project_root),
        project_root=project_root,
    )


def _character_payload(character: Character) -> dict[str, object]:
    record = character.record
    return {
        "name": character.name,
        "description": record.description,
        "preset": record.preset,
        "style": record.style,
        "checkpoint": record.checkpoint,
        "reference": record.reference,
        "seed": record.seed,
        "notes": record.notes,
        "missing": list(character.missing),
    }


def _print_character(character: Character) -> None:
    record = character.record
    typer.echo(f"Character: {character.name}")
    if record.description:
        typer.echo(f"Description: {record.description}")
    for label, value in (
        ("Preset", record.preset),
        ("Style", record.style),
        ("Checkpoint", record.checkpoint),
        ("Reference", record.reference),
        ("Seed", record.seed),
    ):
        if value is not None:
            typer.echo(f"{label}: {value}")
    if record.notes:
        typer.echo(f"Notes: {record.notes}")
    # 台帳は古びる。欠けたまま生成すると、別人が出てから気づくことになる。
    for item in character.missing:
        typer.echo(f"warning: 台帳が指す参照先が見つかりません: {item}", err=True)


def _history_payload(record: RunRecord) -> dict[str, object]:
    return {
        "directory": str(record.directory),
        "created_at": record.created_at,
        "task": record.task,
        "model": record.model,
        "presets": dict(record.presets),
        "seed": record.seed,
        "width": record.width,
        "height": record.height,
        "source": record.source,
        "upscale": record.upscale,
        "features": list(record.features),
        "files": [str(path) for path in record.files],
        "workflow": record.workflow,
    }


def _print_history(records: tuple[RunRecord, ...]) -> None:
    """1件2行で出す。1行目で何の生成か、2行目でどこにあるかが分かる形にする。"""
    if not records:
        typer.echo("(なし)")
        return

    for record in records:
        details = [record.model or "(モデル不明)"]
        details.extend(f"{axis}:{name}" for axis, name in sorted(record.presets.items()))
        if record.source:
            # img2imgのサイズは入力画像で決まる。Specの width/height は使われない。
            details.append(f"<- {record.source}")
        else:
            details.append(f"{record.width}x{record.height}")
        if record.upscale:
            details.append(f"x{record.upscale}")
        details.extend(record.features)

        typer.echo(f"{_format_time(record.created_at)}  {record.task}  seed {record.seed}")
        typer.echo(f"  {'  '.join(details)}")
        for file in record.files:
            typer.echo(f"  {file}")


def _format_time(created_at: str) -> str:
    """ISO 8601 を分までに切り詰める。秒とタイムゾーンは一覧では読まない。"""
    if len(created_at) >= 16 and created_at[10] == "T":
        return f"{created_at[:10]} {created_at[11:16]}"
    return created_at or "(日時不明)"


def _relative_to_root(path: Path, project_root: Path) -> str:
    """作業ルートからの相対パスへ直す。外を指す場合は拒否する。"""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise InvalidGenerationSpec(
            f"画像は作業ルート配下を指定してください (指定値: {path})"
        ) from exc


@app.command(name="batch")
def batch_generate(
    spec_paths: Annotated[
        list[Path],
        typer.Argument(help="実行するGenerationSpecのパス (複数指定可)"),
    ],
    seeds: Annotated[
        str | None,
        typer.Option("--seeds", help="seed掃引。カンマ区切りで指定する (例: 1,2,3)"),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="1件あたりのタイムアウト秒 (既定は IMAGEGEN_TIMEOUT)"),
    ] = None,
    verbose: VerboseOption = False,
) -> None:
    """複数のSpecをまとめて生成する。1件失敗しても残りは続ける。"""
    _configure_logging(verbose)

    with _handled_errors():
        settings = Settings.from_env()
        parsed_seeds = _parse_seeds(seeds)
        # 検証は全件を実行前に済ませる。途中で不正なSpecに当たって止まらないようにする
        pairs = [(path.as_posix(), _load_and_validate(path, settings)) for path in spec_paths]
        items = expand_seeds(pairs, seeds=parsed_seeds)

        outcomes = asyncio.run(_run_batch(items, settings, timeout))

    _echo_batch_summary(outcomes)

    failures = [outcome for outcome in outcomes if not outcome.succeeded]
    if failures:
        raise typer.Exit(getattr(failures[0].error, "exit_code", 1))


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise InvalidGenerationSpec(
                f"--seeds は整数をカンマ区切りで指定してください (指定値: {raw!r})"
            ) from exc
    if not values:
        raise InvalidGenerationSpec("--seeds に有効な値がありません")
    return values


def _echo_batch_summary(outcomes: list[BatchOutcome]) -> None:
    for index, outcome in enumerate(outcomes, start=1):
        typer.echo(f"[{index}/{len(outcomes)}] {outcome.item.label}")
        if outcome.result is not None:
            for path in outcome.result.files:
                typer.echo(f"  -> {path}")
            for path in outcome.result.text_files:
                typer.echo(f"  -> text: {path}")
        else:
            code = getattr(outcome.error, "exit_code", 1)
            typer.echo(f"  -> FAILED (exit {code}): {outcome.error}")

    succeeded = sum(1 for outcome in outcomes if outcome.succeeded)
    typer.echo(f"成功 {succeeded} / 失敗 {len(outcomes) - succeeded}")


async def _run_batch(
    items: list[BatchItem], settings: Settings, timeout: float | None
) -> list[BatchOutcome]:
    """バッチ全体で1つの接続を使い回す。"""
    async with open_generation_backend(settings) as client:

        async def runner(item: BatchItem) -> GenerationResult:
            return await generate(
                item.spec,
                settings,
                backend=client,
                project_root=Path.cwd(),
                timeout=timeout,
            )

        return await run_batch(items, runner=runner)


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
    spec = load_spec(spec_path, presets_root=settings.presets_root, project_root=Path.cwd())
    validate_against_limits(spec, settings)
    return spec


async def _check_health(settings: Settings) -> HealthStatus:
    async with open_generation_backend(settings) as client:
        return await client.health()


async def _run_generation(
    spec: GenerationSpec, settings: Settings, timeout: float | None
) -> GenerationResult:
    async with open_generation_backend(settings) as client:
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
