from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

import muir
from unpu_bench.cli import main
from unpu_bench.errors import CompilationError


def _write_tiny_model(path: Path) -> None:
    path.write_text(
        """
import torch
from torch import nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _compile_tiny(tmp_path: Path, backend: str, hardware: str) -> Path:
    model_py = tmp_path / f"tiny_{backend}.py"
    _write_tiny_model(model_py)
    out_dir = tmp_path / f"out_{backend}"
    rc = main(
        [
            "--target-format",
            backend,
            "--target-hardware",
            hardware,
            "--bit-width",
            "8",
            "--model-py",
            str(model_py),
            "--model-class",
            "TinyNet",
            "--input-shape",
            "1 3 8 8",
            "--output-shape",
            "1 4 8 8",
            "--input-names",
            "input",
            "--output-names",
            "output",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    return out_dir / "program.json"


def test_cross_backend_report_from_program_json(tmp_path: Path) -> None:
    p0 = _compile_tiny(tmp_path, backend="tflm", hardware="hxwe2")
    p1 = _compile_tiny(tmp_path, backend="vela", hardware="hxwe2")

    out = muir.compare_runs(
        [p0, p1],
        out_dir=str(tmp_path / "reports"),
        basename="cmp",
    )
    csv_path = Path(out["csv"])
    md_path = Path(out["markdown"])
    assert csv_path.exists()
    assert md_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "backend" in csv_text
    assert "fallback_ratio" in csv_text
    assert "tflm" in csv_text
    assert "vela" in csv_text


def test_hardware_emit_requires_vendor_output_cli(tmp_path: Path) -> None:
    model_py = tmp_path / "tiny_model.py"
    _write_tiny_model(model_py)
    out_dir = tmp_path / "out"

    rc = main(
        [
            "--target-format",
            "tflm",
            "--target-hardware",
            "hxwe2",
            "--bit-width",
            "8",
            "--model-py",
            str(model_py),
            "--model-class",
            "TinyNet",
            "--input-shape",
            "1 3 8 8",
            "--output-shape",
            "1 4 8 8",
            "--input-names",
            "input",
            "--output-names",
            "output",
            "--emit-hardware-artifact",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 1


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):  # type: ignore[override]
        return self.relu(self.conv(x))


def test_hardware_emit_requires_vendor_output_api(tmp_path: Path) -> None:
    with pytest.raises(CompilationError):
        muir.convert(
            _TinyNet(),
            backend="tflm",
            target_hardware="hxwe2",
            out_dir=str(tmp_path / "out"),
            input_shape=(1, 3, 8, 8),
            output_shape=(1, 4, 8, 8),
            emit_hardware_artifact=True,
        )
