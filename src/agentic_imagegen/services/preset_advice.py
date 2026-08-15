"""style presetの選び忘れと流用を検出する。

生成の可否は変えない。style presetを指定しなくても検証は通り、生成も成功する。
落ちるのは clip skip と外部VAE、そしてそのcheckpointで詰めたサンプラー設定で、
結果は「動くが絵柄が静かに変わる」になる。実際に直近9件の生成でstyle presetが
1つも使われていなかったことがあるため、validateの時点で言葉にする。
"""

from __future__ import annotations

from pathlib import Path

from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.domain.presets import PresetDocument, PresetKind
from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.preset_loader import load_preset


def style_warnings(
    spec: GenerationSpec, *, presets_root: Path, project_root: Path | None = None
) -> tuple[str, ...]:
    """style presetの選び方について伝えることがあれば返す。無ければ空。"""
    # checkpoint と unet はどちらか一方が必ずある (GenerationSpecの検証で担保)。
    model = spec.model.checkpoint or spec.model.unet or ""

    documents = _load_style_presets(presets_root, project_root)
    if not documents:
        # presetを1つも読めない環境では、選べと言っても行き先が無い。
        return ()
    matching = sorted(name for name, document in documents.items() if model in document.applies_to)

    if spec.presets.style is None:
        return (_missing_warning(model, matching),)

    chosen = documents.get(spec.presets.style)
    if chosen is None or not chosen.applies_to or model in chosen.applies_to:
        return ()
    return (_mismatch_warning(spec.presets.style, chosen, model, matching),)


def _missing_warning(model: str, matching: list[str]) -> str:
    base = (
        "style presetを指定していません。clip skipと外部VAE、"
        "そのcheckpointで詰めたsampler / scheduler / cfg / stepsはstyle presetが持ちます"
    )
    if matching:
        return f"{base}。{model} には {' / '.join(matching)} が対応します"
    return f"{base}。`imagegen catalog` の presets/style から選んでください"


def _mismatch_warning(name: str, document: PresetDocument, model: str, matching: list[str]) -> str:
    base = (
        f"style preset {name} は {' / '.join(document.applies_to)} 向けです。"
        f"{model} とは品質タグもサンプラー設定も噛み合いません"
    )
    if matching:
        return f"{base} ({' / '.join(matching)} が対応します)"
    return base


def _load_style_presets(presets_root: Path, project_root: Path | None) -> dict[str, PresetDocument]:
    """読めるstyle presetだけを集める。

    1つ壊れていても他のpresetへの助言は出す。validateを止めるのは
    Specが参照しているpresetが壊れているときだけで、それは load_spec の仕事。
    """
    directory = presets_root / PresetKind.STYLE.directory
    if not directory.is_dir():
        return {}

    documents: dict[str, PresetDocument] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            documents[path.stem] = load_preset(
                PresetKind.STYLE, path.stem, root=presets_root, project_root=project_root
            )
        except InvalidGenerationSpec:
            continue
    return documents


__all__ = ["style_warnings"]
