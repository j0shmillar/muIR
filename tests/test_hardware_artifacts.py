from __future__ import annotations

import json
import os
from pathlib import Path

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


def _write_fake_vela(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--accelerator-config")
p.add_argument("input")
p.add_argument("--output-dir", required=True)
args = p.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
stem = Path(args.input).stem
dst = out / f"{stem}_vela.tflite"
shutil.copy2(args.input, dst)
print("fake vela wrote", dst)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_cvi_tools(root: Path) -> None:
    (root / "model_transform.py").write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model_name")
p.add_argument("--model_def")
p.add_argument("--mlir", required=True)
p.add_argument("--output_names")
args, _ = p.parse_known_args()
Path(args.mlir).write_text("fake-mlir", encoding="utf-8")
print("fake cvi transform wrote", args.mlir)
""",
        encoding="utf-8",
    )
    (root / "run_calibration.py").write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("mlir")
p.add_argument("--dataset")
p.add_argument("--input_num")
p.add_argument("-o", required=True)
args = p.parse_args()
Path(args.o).write_text("fake-table", encoding="utf-8")
print("fake cvi calibration wrote", args.o)
""",
        encoding="utf-8",
    )
    (root / "model_deploy.py").write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mlir", required=True)
p.add_argument("--quantize", required=True)
p.add_argument("--calibration_table", required=True)
p.add_argument("--processor", required=True)
p.add_argument("--tolerance", required=True)
p.add_argument("--model", required=True)
args = p.parse_args()
Path(args.model).write_bytes(b"FAKE_CVIMODEL")
print("fake cvi deploy wrote", args.model)
""",
        encoding="utf-8",
    )
    for name in ["model_transform.py", "run_calibration.py", "model_deploy.py"]:
        (root / name).chmod(0o755)


def _write_fake_neutron(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
args, _ = p.parse_known_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(args.input, args.output)
print("fake neutron wrote", args.output)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_vela_hardware_artifact_emission(tmp_path: Path, monkeypatch) -> None:
    model_py = tmp_path / "tiny_model.py"
    _write_tiny_model(model_py)

    input_tflite = tmp_path / "input.tflite"
    input_tflite.write_bytes(b"FAKE_TFLITE")

    fake_vela = tmp_path / "vela"
    _write_fake_vela(fake_vela)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    out_dir = tmp_path / "out_vela"
    rc = main(
        [
            "--target-format",
            "vela",
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
            "--backend-source-model",
            str(input_tflite),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (out_dir / "vela" / "input_vela.tflite").exists()

    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    hw = [a for a in program.get("backend_artifacts", []) if a.get("artifact_type") == "hardware_model"]
    assert hw
    assert hw[0]["path"].endswith("input_vela.tflite")


def test_cvi_hardware_artifact_emission(tmp_path: Path, monkeypatch) -> None:
    model_py = tmp_path / "tiny_model.py"
    _write_tiny_model(model_py)
    input_onnx = tmp_path / "input.onnx"
    input_onnx.write_bytes(b"FAKE_ONNX")
    data_sample = tmp_path / "sample.npy"
    import numpy as np
    np.save(data_sample, np.random.randn(1, 3, 8, 8).astype(np.float32))

    _write_fake_cvi_tools(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    out_dir = tmp_path / "out_cvi"
    rc = main(
        [
            "--target-format",
            "cvi",
            "--target-hardware",
            "bm1684x",
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
            "--data-sample",
            str(data_sample),
            "--emit-hardware-artifact",
            "--backend-source-model",
            str(input_onnx),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (out_dir / "cvi" / "input.cvimodel").exists()

    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    hw = [a for a in program.get("backend_artifacts", []) if a.get("artifact_type") == "hardware_model"]
    assert hw
    assert hw[0]["path"].endswith("input.cvimodel")


def test_eiq_hardware_artifact_emission(tmp_path: Path, monkeypatch) -> None:
    model_py = tmp_path / "tiny_model.py"
    _write_tiny_model(model_py)
    input_tflite = tmp_path / "input.tflite"
    input_tflite.write_bytes(b"FAKE_TFLITE")

    fake_neutron = tmp_path / "neutron"
    _write_fake_neutron(fake_neutron)
    monkeypatch.setenv("EIQ_NEUTRON_PATH", str(fake_neutron))

    out_dir = tmp_path / "out_eiq"
    rc = main(
        [
            "--target-format",
            "eiq",
            "--target-hardware",
            "mcxn947",
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
            "--backend-source-model",
            str(input_tflite),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert (out_dir / "eiq" / "input_eiq.tflite").exists()

    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    hw = [a for a in program.get("backend_artifacts", []) if a.get("artifact_type") == "hardware_model"]
    assert hw
    assert hw[0]["path"].endswith("input_eiq.tflite")
