"""スキャフォールドのスモークテスト。"""

from typer.testing import CliRunner

from agentic_imagegen import __version__
from agentic_imagegen.cli import app


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
