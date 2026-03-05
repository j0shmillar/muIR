from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from unpu_bench.config import (
    load_platforms_config,
    validate_core_config,
    CoreConfig,
    ConfigError,
)


def write_platforms(tmp_path: Path, text: str) -> str:
    path = tmp_path / "platforms.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


def test_load_platforms_and_basic_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_platforms(
        tmp_path,
        """
        ai8x:
          depends_on: [onnx]
          flags:
            q_scale:
              type: float
              default: 0.85
        """,
    )
    monkeypatch.chdir(tmp_path)
    platforms = load_platforms_config()
    assert "ai8x" in platforms
    ai8x = platforms["ai8x"]
    assert "q_scale" in ai8x.flags
    assert ai8x.flags["q_scale"].type == "float"


def test_load_platforms_missing_type_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_platforms(
        tmp_path,
        """
        ai8x:
          flags:
            q_scale:
              default: 0.85
        """,
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        load_platforms_config()


def test_validate_core_config_bitwidth_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_platforms(
        tmp_path,
        """
        ai8x:
          flags: {}
        """,
    )
    monkeypatch.chdir(tmp_path)
    platforms = load_platforms_config()
    cfg = CoreConfig(target_format="ai8x", target_hardware="max78000", bit_width=8)
    validate_core_config(cfg, platforms)  # should not raise


def test_validate_core_config_unknown_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_platforms(
        tmp_path,
        """
        ai8x:
          flags: {}
        """,
    )
    monkeypatch.chdir(tmp_path)
    platforms = load_platforms_config()
    cfg = CoreConfig(target_format="does-not-exist", target_hardware="max78000", bit_width=8)
    with pytest.raises(ConfigError):
        validate_core_config(cfg, platforms)


def test_load_formats_schema_and_validate_hardware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_platforms(
        tmp_path,
        """
        formats:
          ai8x:
            bit_widths: [1, 2, 4, 8]
            compatible_hardware: [max78000, max78002]
            flags:
              fifo:
                action: store_true
        hardware:
          max78000: { formats: [ai8x] }
        """,
    )
    monkeypatch.chdir(tmp_path)
    platforms = load_platforms_config()

    validate_core_config(
        CoreConfig(target_format="ai8x", target_hardware="max78000", bit_width=8),
        platforms,
    )

    with pytest.raises(ConfigError):
        validate_core_config(
            CoreConfig(target_format="ai8x", target_hardware="hxwe2", bit_width=8),
            platforms,
        )
