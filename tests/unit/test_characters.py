"""キャラクタ台帳の読み込みのテスト。

「さっきの子で別の場面を」を再現するには、character preset・style preset・checkpoint・
基準画像・seedの5つが揃っている必要がある。台帳はその5つを1か所に置くだけのもので、
生成の可否には関与しない。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentic_imagegen.domain.characters import Character
from agentic_imagegen.errors import InvalidGenerationSpec
from agentic_imagegen.services.characters import (
    CHARACTERS_DIRECTORY,
    collect_characters,
    load_character,
)

HASSAKU = "hassakuSD15_v13.safetensors"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """台帳・preset・基準画像が揃った作業ルート。"""
    (tmp_path / "presets" / "characters").mkdir(parents=True)
    (tmp_path / "presets" / "styles").mkdir()
    (tmp_path / "presets" / "characters" / "anime-girl-blue.yaml").write_text("", encoding="utf-8")
    (tmp_path / "presets" / "styles" / "sd15-hassaku.yaml").write_text("", encoding="utf-8")

    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "aoi.png").write_bytes(b"png")

    (tmp_path / "registry" / CHARACTERS_DIRECTORY).mkdir(parents=True)
    _write(tmp_path, "aoi")
    return tmp_path


def _write(root: Path, name: str, **overrides: object) -> Path:
    document: dict[str, object] = {
        "description": "青い髪の少女",
        "preset": "anime-girl-blue",
        "style": "sd15-hassaku",
        "checkpoint": HASSAKU,
        "reference": "inputs/aoi.png",
        "seed": 777001,
    }
    document.update(overrides)
    path = root / "registry" / CHARACTERS_DIRECTORY / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def _load(root: Path, name: str = "aoi") -> Character:
    return load_character(
        name,
        registry_root=root / "registry",
        presets_root=root / "presets",
        project_root=root,
    )


class TestLoad:
    def test_reads_every_field(self, workspace: Path) -> None:
        character = _load(workspace)

        assert character.name == "aoi"
        assert character.record.preset == "anime-girl-blue"
        assert character.record.style == "sd15-hassaku"
        assert character.record.checkpoint == HASSAKU
        assert character.record.reference == "inputs/aoi.png"
        assert character.record.seed == 777001

    def test_complete_entry_has_nothing_missing(self, workspace: Path) -> None:
        assert _load(workspace).missing == ()

    def test_unknown_name_is_rejected(self, workspace: Path) -> None:
        with pytest.raises(InvalidGenerationSpec, match="見つかりません"):
            _load(workspace, "unknown")

    def test_path_traversal_is_rejected(self, workspace: Path) -> None:
        """名前はファイル名になる。registry の外を指させない。"""
        with pytest.raises(InvalidGenerationSpec):
            _load(workspace, "../../etc/passwd")

    def test_broken_yaml_is_reported(self, workspace: Path) -> None:
        (workspace / "registry" / CHARACTERS_DIRECTORY / "aoi.yaml").write_text(
            "{ not yaml:", encoding="utf-8"
        )

        with pytest.raises(InvalidGenerationSpec):
            _load(workspace)

    def test_unknown_field_is_rejected(self, workspace: Path) -> None:
        """書き間違えた項目が黙って無視されると、台帳を信じられなくなる。"""
        _write(workspace, "aoi", refernce="inputs/aoi.png")

        with pytest.raises(InvalidGenerationSpec):
            _load(workspace)

    def test_scalar_document_is_rejected(self, workspace: Path) -> None:
        (workspace / "registry" / CHARACTERS_DIRECTORY / "aoi.yaml").write_text(
            "aoi", encoding="utf-8"
        )

        with pytest.raises(InvalidGenerationSpec, match="マッピング"):
            _load(workspace)

    def test_empty_document_is_allowed(self, workspace: Path) -> None:
        """書きかけの台帳でも読めること自体は妨げない。"""
        (workspace / "registry" / CHARACTERS_DIRECTORY / "aoi.yaml").write_text(
            "", encoding="utf-8"
        )

        assert _load(workspace).record.seed is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("preset", "../../etc/passwd"),
            ("style", "../styles/x"),
            ("checkpoint", "/etc/passwd"),
            ("checkpoint", "model.exe"),
            ("reference", "/etc/shadow"),
            ("reference", "inputs/x.txt"),
        ],
    )
    def test_unsafe_values_are_rejected(self, workspace: Path, field: str, value: str) -> None:
        """台帳はファイル名とパスをそのまま持つ。ルート外を指させない。"""
        _write(workspace, "aoi", **{field: value})

        with pytest.raises(InvalidGenerationSpec):
            _load(workspace)

    def test_explicit_nulls_are_allowed(self, workspace: Path) -> None:
        """項目を空で置いた台帳も読める。埋まっている分だけ手掛かりになる。"""
        _write(workspace, "aoi", checkpoint=None, reference=None, preset=None, style=None)

        record = _load(workspace).record

        assert record.checkpoint is None
        assert record.reference is None

    def test_unreadable_entry_is_reported(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """権限やI/Oで読めない台帳を、握りつぶさず原因として返す。"""

        def deny(*args: object, **kwargs: object) -> str:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "read_text", deny)

        with pytest.raises(InvalidGenerationSpec, match="読み込めません"):
            _load(workspace)


class TestMissing:
    def test_absent_preset_is_reported(self, workspace: Path) -> None:
        _write(workspace, "aoi", preset="gone")

        missing = _load(workspace).missing

        assert any("gone" in item for item in missing)

    def test_absent_style_is_reported(self, workspace: Path) -> None:
        _write(workspace, "aoi", style="gone")

        assert any("gone" in item for item in _load(workspace).missing)

    def test_absent_reference_image_is_reported(self, workspace: Path) -> None:
        (workspace / "inputs" / "aoi.png").unlink()

        assert any("inputs/aoi.png" in item for item in _load(workspace).missing)

    def test_missing_does_not_raise(self, workspace: Path) -> None:
        """台帳が古びていても引けること自体は妨げない。"""
        _write(workspace, "aoi", preset="gone", style="gone")

        assert _load(workspace).record.seed == 777001

    def test_reference_outside_the_project_is_rejected(self, workspace: Path) -> None:
        with pytest.raises(InvalidGenerationSpec):
            _write(workspace, "aoi", reference="../secrets.png")
            _load(workspace)


class TestCollect:
    def test_lists_in_name_order(self, workspace: Path) -> None:
        _write(workspace, "sora")

        characters = collect_characters(
            registry_root=workspace / "registry",
            presets_root=workspace / "presets",
            project_root=workspace,
        )

        assert [character.name for character in characters] == ["aoi", "sora"]

    def test_missing_registry_is_empty(self, tmp_path: Path) -> None:
        characters = collect_characters(
            registry_root=tmp_path / "nope",
            presets_root=tmp_path / "presets",
            project_root=tmp_path,
        )

        assert characters == ()

    def test_broken_entry_does_not_hide_the_others(self, workspace: Path) -> None:
        """1件壊れていても、残りは引けたほうが台帳として役に立つ。"""
        _write(workspace, "sora")
        (workspace / "registry" / CHARACTERS_DIRECTORY / "broken.yaml").write_text(
            "{ not yaml:", encoding="utf-8"
        )

        characters = collect_characters(
            registry_root=workspace / "registry",
            presets_root=workspace / "presets",
            project_root=workspace,
        )

        assert [character.name for character in characters] == ["aoi", "sora"]
