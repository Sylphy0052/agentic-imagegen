"""Workflowテンプレートの読み込み・構造検証・パラメータ注入のテスト。"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agentic_imagegen.adapters.comfyui.workflow import (
    TXT2IMG_BINDING,
    build_workflow,
    resolve_seed,
    validate_structure,
)
from agentic_imagegen.domain.models import GenerationSpec
from agentic_imagegen.errors import WorkflowValidationError
from agentic_imagegen.workflows.injector import (
    WORKFLOWS_DIR,
    load_workflow_template,
    prepare_workflow,
    template_digest,
)


@pytest.fixture
def template() -> dict[str, Any]:
    return load_workflow_template("txt2img")


def _spec(**overrides: Any) -> GenerationSpec:
    payload: dict[str, Any] = {
        "prompt": {"positive": "1girl, blue hair", "negative": "low quality"},
        "model": {"checkpoint": "v1-5-pruned-emaonly.safetensors"},
        "generation": {
            "width": 512,
            "height": 768,
            "steps": 24,
            "cfg": 5.5,
            "seed": 12345,
            "batch_size": 2,
            "sampler": "dpmpp_2m",
            "scheduler": "karras",
        },
        "output": {"directory": "outputs", "prefix": "blue_hair"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(payload.get(section), dict):
            payload[section] = {**payload[section], **values}
        else:
            payload[section] = values
    return GenerationSpec.model_validate(payload)


# --- テンプレート読み込み ---------------------------------------------------


def test_repository_template_is_loadable() -> None:
    workflow = load_workflow_template("txt2img")

    assert workflow["3"]["class_type"] == "KSampler"
    assert (WORKFLOWS_DIR / "txt2img.json").is_file()


@pytest.mark.parametrize("name", ["img2img", "unknown", "TXT2IMG"])
def test_unknown_workflow_name_rejected(name: str) -> None:
    with pytest.raises(WorkflowValidationError, match="許可されていません"):
        load_workflow_template(name)


@pytest.mark.parametrize(
    "name",
    ["../txt2img", "../../etc/passwd", "sub/txt2img", "txt2img.json", "/txt2img"],
)
def test_workflow_name_path_traversal_rejected(name: str) -> None:
    with pytest.raises(WorkflowValidationError):
        load_workflow_template(name)


def test_missing_template_file(tmp_path: Path) -> None:
    with pytest.raises(WorkflowValidationError, match="見つかりません"):
        load_workflow_template("txt2img", workflows_dir=tmp_path)


def test_broken_json(tmp_path: Path) -> None:
    (tmp_path / "txt2img.json").write_text("{ broken", encoding="utf-8")

    with pytest.raises(WorkflowValidationError, match="JSON"):
        load_workflow_template("txt2img", workflows_dir=tmp_path)


# --- 構造検証 (fail-fast) ---------------------------------------------------


def test_valid_structure_passes(template: dict[str, Any]) -> None:
    validate_structure(template, TXT2IMG_BINDING)


def test_missing_node_fails(template: dict[str, Any]) -> None:
    del template["5"]

    with pytest.raises(WorkflowValidationError, match="5"):
        validate_structure(template, TXT2IMG_BINDING)


def test_class_type_mismatch_fails(template: dict[str, Any]) -> None:
    template["5"]["class_type"] = "EmptySD3LatentImage"

    with pytest.raises(WorkflowValidationError, match="class_type"):
        validate_structure(template, TXT2IMG_BINDING)


def test_missing_input_key_fails(template: dict[str, Any]) -> None:
    del template["5"]["inputs"]["batch_size"]

    with pytest.raises(WorkflowValidationError, match="batch_size"):
        validate_structure(template, TXT2IMG_BINDING)


def test_non_mapping_node_fails(template: dict[str, Any]) -> None:
    template["5"] = ["not", "a", "node"]

    with pytest.raises(WorkflowValidationError):
        validate_structure(template, TXT2IMG_BINDING)


def test_prompt_link_mismatch_fails(template: dict[str, Any]) -> None:
    """KSamplerのpositive/negative接続先が想定ノードと違えば拒否する。"""
    template["3"]["inputs"]["positive"] = ["7", 0]
    template["3"]["inputs"]["negative"] = ["6", 0]

    with pytest.raises(WorkflowValidationError, match="positive"):
        validate_structure(template, TXT2IMG_BINDING)


def test_latent_link_mismatch_fails(template: dict[str, Any]) -> None:
    template["3"]["inputs"]["latent_image"] = ["8", 0]

    with pytest.raises(WorkflowValidationError, match="latent_image"):
        validate_structure(template, TXT2IMG_BINDING)


def test_unknown_workflow_structure_fails() -> None:
    """まったく別構造のJSONはfail-fastする。"""
    with pytest.raises(WorkflowValidationError):
        validate_structure({"nodes": [], "links": []}, TXT2IMG_BINDING)


# --- パラメータ注入 ---------------------------------------------------------


def test_positive_prompt_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    assert workflow["6"]["inputs"]["text"] == "1girl, blue hair"


def test_negative_prompt_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    assert workflow["7"]["inputs"]["text"] == "low quality"


def test_checkpoint_injection(template: dict[str, Any]) -> None:
    spec = _spec(model={"checkpoint": "sd15/anything.safetensors"})
    workflow = build_workflow(template, spec, seed=1)
    assert workflow["4"]["inputs"]["ckpt_name"] == "sd15/anything.safetensors"


def test_resolution_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    assert workflow["5"]["inputs"]["width"] == 512
    assert workflow["5"]["inputs"]["height"] == 768


def test_batch_size_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    assert workflow["5"]["inputs"]["batch_size"] == 2


def test_seed_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=987654321)
    assert workflow["3"]["inputs"]["seed"] == 987654321


def test_sampler_parameters_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    inputs = workflow["3"]["inputs"]

    assert inputs["steps"] == 24
    assert inputs["cfg"] == 5.5
    assert inputs["sampler_name"] == "dpmpp_2m"
    assert inputs["scheduler"] == "karras"


def test_filename_prefix_injection(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=1)
    assert workflow["9"]["inputs"]["filename_prefix"] == "blue_hair"


def test_template_is_not_mutated(template: dict[str, Any]) -> None:
    snapshot = copy.deepcopy(template)

    build_workflow(template, _spec(), seed=42)

    assert template == snapshot


def test_links_are_preserved(template: dict[str, Any]) -> None:
    workflow = build_workflow(template, _spec(), seed=42)

    assert workflow["3"]["inputs"]["positive"] == ["6", 0]
    assert workflow["8"]["inputs"]["samples"] == ["3", 0]


def test_build_workflow_rejects_unresolved_seed(template: dict[str, Any]) -> None:
    with pytest.raises(WorkflowValidationError, match="seed"):
        build_workflow(template, _spec(), seed=-1)


def test_build_workflow_validates_structure_first(template: dict[str, Any]) -> None:
    del template["9"]

    with pytest.raises(WorkflowValidationError):
        build_workflow(template, _spec(), seed=1)


# --- seed解決 ---------------------------------------------------------------


def test_resolve_seed_keeps_explicit_value() -> None:
    assert resolve_seed(12345) == 12345
    assert resolve_seed(0) == 0


def test_resolve_seed_randomizes_minus_one() -> None:
    values = {resolve_seed(-1) for _ in range(20)}

    assert all(0 <= value <= 2**63 - 1 for value in values)
    assert len(values) > 1


# --- 高レベルAPI ------------------------------------------------------------


def test_prepare_workflow_returns_workflow_and_seed() -> None:
    prepared = prepare_workflow(_spec())

    assert prepared.workflow["6"]["inputs"]["text"] == "1girl, blue hair"
    assert prepared.seed == 12345


def test_prepare_workflow_resolves_random_seed() -> None:
    prepared = prepare_workflow(_spec(generation={"seed": -1}))

    assert 0 <= prepared.seed <= 2**63 - 1


def test_prepare_workflow_output_is_json_serializable() -> None:
    prepared = prepare_workflow(_spec())

    assert json.loads(json.dumps(prepared.workflow)) == prepared.workflow


def test_prepare_workflow_reports_template_hash() -> None:
    prepared = prepare_workflow(_spec())

    assert prepared.template_hash.startswith("sha256:")
    assert len(prepared.template_hash) == len("sha256:") + 64


def test_template_hash_is_stable_across_formatting() -> None:
    """整形や鍵の順序が変わっただけではダイジェストを動かさない。"""
    template = load_workflow_template("txt2img")
    reordered = dict(reversed(list(template.items())))

    assert template_digest(template) == template_digest(reordered)


def test_template_hash_changes_when_content_changes() -> None:
    template = load_workflow_template("txt2img")
    modified = json.loads(json.dumps(template))
    modified["3"]["inputs"]["steps"] = 999

    assert template_digest(template) != template_digest(modified)
