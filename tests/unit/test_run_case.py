"""evals/run_case.py のコマンド組み立てを検証する。

このスクリプトは `src/` の外にあるためパッケージとしてimportできない。
ファイルパスから直接ロードする。`claude` は起動しない。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "evals" / "run_case.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_case", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_case() -> types.ModuleType:
    return _load_module()


class TestAllowedTools:
    def test_reading_and_skills_are_always_allowed(self, run_case: types.ModuleType) -> None:
        allowed = run_case.allowed_tools({"id": "x"})

        assert allowed == run_case.ALLOWED_TOOLS

    def test_case_shell_patterns_are_appended(self, run_case: types.ModuleType) -> None:
        """コマンドの実行結果が要るcaseだけへ、そのコマンドを名指しで許す。"""
        allowed = run_case.allowed_tools({"id": "x", "shell": ["Bash(uv run imagegen:*)"]})

        assert allowed == f"{run_case.ALLOWED_TOOLS},Bash(uv run imagegen:*)"

    def test_bare_bash_is_rejected(self, run_case: types.ModuleType) -> None:
        """`Bash` をそのまま許すと生成まで走る。名指しのパターンだけを受ける。"""
        with pytest.raises(ValueError, match="Bash"):
            run_case.allowed_tools({"id": "x", "shell": ["Bash"]})


class TestCommand:
    def test_permission_mode_is_pinned(self, run_case: types.ModuleType) -> None:
        """利用者の設定が bypassPermissions だと allowedTools が素通りする。"""
        command = run_case.build_command({"id": "x", "query": "q"})

        assert "--permission-mode" in command
        assert command[command.index("--permission-mode") + 1] == "default"

    def test_user_settings_are_not_read(self, run_case: types.ModuleType) -> None:
        """グローバルのallowリストが効くと、許していないコマンドまで通る。"""
        command = run_case.build_command({"id": "x", "query": "q"})

        assert command[command.index("--setting-sources") + 1] == "project"

    def test_query_is_passed_verbatim(self, run_case: types.ModuleType) -> None:
        command = run_case.build_command({"id": "x", "query": "猫の画像を作って"})

        assert command[command.index("-p") + 1] == "猫の画像を作って"


class TestSystemPrompt:
    def test_constraints_are_included(self, run_case: types.ModuleType) -> None:
        prompt = run_case.system_prompt({"id": "x"})

        assert "コマンドは実行しないでください" in prompt

    def test_shell_case_is_told_to_run_the_allowed_command(
        self, run_case: types.ModuleType
    ) -> None:
        prompt = run_case.system_prompt({"id": "x", "shell": ["Bash(uv run imagegen:*)"]})

        assert "許可されたコマンドは実行してください" in prompt

    def test_allowed_commands_are_shown_verbatim(self, run_case: types.ModuleType) -> None:
        """許可は前方一致。呼び方がずれると拒否されるため、形をそのまま見せる。"""
        prompt = run_case.system_prompt(
            {"id": "x", "shell": ["Bash(python3 evals/bin/imagegen_ro.py:*)"]}
        )

        assert "`python3 evals/bin/imagegen_ro.py`" in prompt

    def test_context_is_presented_as_a_precondition(self, run_case: types.ModuleType) -> None:
        """「さっき作ったSpec」を指せるように、直前のやり取りを前提として渡す。"""
        prompt = run_case.system_prompt({"id": "x", "context": "直前にSpecを1つ作った。"})

        assert "直前にSpecを1つ作った。" in prompt


class TestReadOnlyWrapper:
    """`evals/bin/imagegen_ro.py` は状態を変えるサブコマンドを通さない。"""

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        # 実行するのはこのリポジトリのスクリプトで、引数もテストが与えた定数。
        return subprocess.run(  # noqa: S603
            [sys.executable, str(PROJECT_ROOT / "evals" / "bin" / "imagegen_ro.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    @pytest.mark.parametrize("subcommand", ["generate", "batch", "compose"])
    def test_writing_subcommands_are_refused(self, subcommand: str) -> None:
        result = self._run(subcommand, "spec.yaml")

        assert result.returncode == 2
        assert "通すのは" in result.stderr

    def test_no_subcommand_is_refused(self) -> None:
        assert self._run().returncode == 2

    def test_reading_subcommand_reaches_the_cli(self) -> None:
        result = self._run("character", "list")

        assert result.returncode == 0


class TestEnvironment:
    def test_registry_points_at_the_fixture(self, run_case: types.ModuleType) -> None:
        """台帳が空だと「さっきの子」を引くcaseが判定できない。"""
        environment = run_case.environment()

        assert environment["IMAGEGEN_REGISTRY_ROOT"].endswith("evals/fixtures/registry")

    def test_existing_environment_is_kept(self, run_case: types.ModuleType) -> None:
        environment = run_case.environment()

        assert "PATH" in environment


class TestFixtures:
    def test_registry_fixture_exists(self) -> None:
        path = PROJECT_ROOT / "evals" / "fixtures" / "registry" / "characters" / "aoi.yaml"

        assert path.is_file()

    def test_needs_fixture_case_names_the_fixture_character(self) -> None:
        """フィクスチャの名前とcaseのqueryがずれたら、caseは永遠に判定不能になる。"""
        document: dict[str, Any] = json.loads(
            (PROJECT_ROOT / "evals" / "evals.json").read_text(encoding="utf-8")
        )
        cases = [case for case in document["cases"] if "needs-fixture" in case.get("tags", [])]

        assert cases
        for case in cases:
            assert "aoi" in case["query"]
