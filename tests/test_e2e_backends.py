from pathlib import Path
import json

import pytest
from unpu_bench.cli import main


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


def _run_compile(tmp_path: Path, *, target_format: str, target_hardware: str) -> Path:
    model_py = tmp_path / "tiny_model.py"
    _write_tiny_model(model_py)

    out_dir = tmp_path / f"out_{target_format}"
    rc = main(
        [
            "--target-format",
            target_format,
            "--target-hardware",
            target_hardware,
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
    return out_dir


def _artifact_path(out_dir: Path, backend: str) -> Path:
    return out_dir / backend / f"model.{backend}.ir.json"


def _compiled_path(out_dir: Path, backend: str) -> Path:
    return out_dir / backend / f"model.{backend}.compiled.json"


def _golden_path(backend: str) -> Path:
    return Path(__file__).resolve().parent / "golden" / f"{backend}.tiny.ir.json"


@pytest.mark.parametrize(
    ("target_format", "target_hardware"),
    [
        ("tflm", "hxwe2"),
        ("vela", "hxwe2"),
        ("cvi", "bm1684x"),
        ("eiq", "mcxn947"),
    ],
)
def test_e2e_compile_ir_backends_with_golden(
    tmp_path: Path, target_format: str, target_hardware: str
) -> None:
    out_dir = _run_compile(
        tmp_path, target_format=target_format, target_hardware=target_hardware
    )
    assert (out_dir / "program.json").exists()
    artifact_path = _artifact_path(out_dir, target_format)
    compiled_path = _compiled_path(out_dir, target_format)
    assert artifact_path.exists()
    assert compiled_path.exists()

    got = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = json.loads(_golden_path(target_format).read_text(encoding="utf-8"))
    assert got == expected

    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    artifact_meta = {
        (a["backend"], a["artifact_type"]): a["meta"]
        for a in program.get("backend_artifacts", [])
    }
    ir_meta = artifact_meta[(target_format, "ir_bundle")]
    compiled_meta = artifact_meta[(target_format, "compiled_model")]
    assert ir_meta["vendor_toolchain"] is False
    assert compiled_meta["vendor_toolchain"] is False
    assert ir_meta["execution_engine"] == "unpu_ir_runtime"
    assert compiled_meta["execution_engine"] == "unpu_ir_runtime"
