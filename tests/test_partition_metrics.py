from __future__ import annotations

import json
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


def test_program_json_contains_partition_metrics(tmp_path: Path) -> None:
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
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0

    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    metrics = program["metadata"]["partition_metrics"]
    assert metrics["heuristic"] == "single_contiguous_core_with_cpu_prefix_suffix"
    assert metrics["offload_topology"] == "prefix_core_suffix"
    assert metrics["topology_valid"] is True
    assert metrics["partition_count"] >= 1
    assert metrics["core_partition_count"] <= 1
    assert metrics["cut_count"] >= 0
    assert metrics["boundary_tensor_count"] >= 0
    assert metrics["cost_proxy"] >= 0
