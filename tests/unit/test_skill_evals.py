"""evals/evals.json の妥当性を機械検査する。

evalsはモデルを動かして採点するものなので、内容の正しさまではテストできない。
ここで防ぐのは「参照先が消えているのに気付かないまま残り続ける」ことだけ。
skillやpresetを消したときにevalsが道連れで腐るのを検出する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_PATH = REPO_ROOT / "evals" / "evals.json"
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
PRESETS_ROOT = REPO_ROOT / "presets"

ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
#: presets/ 直下のディレクトリ名と、Specの `presets:` のキーの対応。
AXIS_DIRECTORIES = {"character": "characters", "scene": "scenes", "style": "styles"}


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return json.loads(EVALS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document["cases"]


class TestDocument:
    def test_readme_exists(self) -> None:
        """判定の付け方が無いとevalsは再現できない。"""
        assert (REPO_ROOT / "evals" / "README.md").is_file()

    def test_has_version(self, document: dict[str, Any]) -> None:
        assert document["version"] == "1"

    def test_covers_every_skill(self, cases: list[dict[str, Any]]) -> None:
        """skillを足したらevalsも足す。片方だけ増える状態を検出する。"""
        skills = {path.name for path in SKILLS_ROOT.iterdir() if (path / "SKILL.md").is_file()}

        assert {case["skill"] for case in cases} == skills


class TestCases:
    def test_ids_are_unique(self, cases: list[dict[str, Any]]) -> None:
        ids = [case["id"] for case in cases]

        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("field", ["id", "skill", "query"])
    def test_required_text_fields_are_filled(self, cases: list[dict[str, Any]], field: str) -> None:
        for case in cases:
            assert case[field].strip(), f"{case.get('id')}: {field} が空"

    def test_ids_are_kebab_case(self, cases: list[dict[str, Any]]) -> None:
        for case in cases:
            assert ID_PATTERN.match(case["id"]), case["id"]

    def test_expected_behaviors_are_listed(self, cases: list[dict[str, Any]]) -> None:
        """満たすべき振る舞いが無いと採点できない。"""
        for case in cases:
            assert case["expected_behaviors"], case["id"]

    def test_skills_exist(self, cases: list[dict[str, Any]]) -> None:
        for case in cases:
            assert (SKILLS_ROOT / case["skill"] / "SKILL.md").is_file(), case["id"]

    def test_referenced_presets_exist(self, cases: list[dict[str, Any]]) -> None:
        """presetを消したりrenameしたら、evalsが指す名前も落ちる。"""
        for case in cases:
            for axis, name in case.get("presets", {}).items():
                directory = AXIS_DIRECTORIES[axis]
                assert (PRESETS_ROOT / directory / f"{name}.yaml").is_file(), (
                    f"{case['id']}: {axis}/{name}"
                )

    def test_referenced_documents_exist(self, cases: list[dict[str, Any]]) -> None:
        for case in cases:
            for reference in case.get("references", []):
                assert (REPO_ROOT / reference).is_file(), f"{case['id']}: {reference}"


class TestRunnerFields:
    """`run_case.py` が読むフィールドと `needs-` タグの対応を保つ。"""

    def test_shell_patterns_name_the_command(self, cases: list[dict[str, Any]]) -> None:
        """`Bash` をそのまま許すと生成まで走る。"""
        for case in cases:
            for pattern in case.get("shell", []):
                assert pattern.startswith("Bash(") and pattern.endswith(")"), (
                    f"{case['id']}: {pattern}"
                )

    def test_shell_is_limited_to_cases_that_need_it(self, cases: list[dict[str, Any]]) -> None:
        for case in cases:
            has_need = any(tag.startswith("needs-") for tag in case.get("tags", []))

            assert bool(case.get("shell")) <= has_need, case["id"]

    def test_context_belongs_to_needs_context_cases(self, cases: list[dict[str, Any]]) -> None:
        """直前のやり取りを前提として渡すのは、それが無いと成立しないcaseだけ。"""
        for case in cases:
            has_context = bool(case.get("context"))

            assert has_context == ("needs-context" in case.get("tags", [])), case["id"]

    def test_registry_fixture_backs_the_needs_fixture_cases(
        self, cases: list[dict[str, Any]]
    ) -> None:
        fixture = REPO_ROOT / "evals" / "fixtures" / "registry" / "characters"
        needs_fixture = [case for case in cases if "needs-fixture" in case.get("tags", [])]

        assert needs_fixture
        assert list(fixture.glob("*.yaml"))
