from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from unpu_bench.ir_version import (
    migrate_program_ir_metadata,
    validate_program_ir_metadata,
)
from unpu_bench.muir import (
    Graph,
    Op,
    Program,
    Tensor,
    TensorType,
    build_program_from_torch,
)
from unpu_bench.passes import (
    run_ir_canonicalization,
    run_ir_rewrite_passes,
    run_quantization_contract_validation,
)
from unpu_bench.semantic_check import run_semantic_check_torch_vs_ir


def _make_identity_program() -> Program:
    tensors = {
        "x": Tensor(
            id="x",
            type=TensorType(shape=[1, 3, 4, 4], dtype="f32", layout="NCHW"),
            role="input",
        ),
        "y": Tensor(
            id="y",
            type=TensorType(shape=[1, 3, 4, 4], dtype="f32", layout="NCHW"),
            producer="op0",
        ),
        "z": Tensor(
            id="z",
            type=TensorType(shape=[1, 3, 4, 4], dtype="f32", layout="NCHW"),
            producer="op1",
            role="output",
        ),
    }
    tensors["x"].consumers = ["op0"]
    tensors["y"].consumers = ["op1"]
    ops = {
        "op0": Op(id="op0", kind="Identity", inputs=["x"], outputs=["y"], attrs={}),
        "op1": Op(id="op1", kind="Relu", inputs=["y"], outputs=["z"], attrs={}),
    }
    g = Graph(
        name="id",
        inputs=["x"],
        outputs=["z"],
        initializers=[],
        tensors=tensors,
        ops=ops,
        op_order=["op0", "op1"],
    )
    return Program(graph=g, partitions=[], metadata={})


def test_rewrite_removes_identity() -> None:
    p = _make_identity_program()
    stats = run_ir_rewrite_passes(p)
    assert stats["identity_removed"] == 1
    assert "op0" not in p.graph.ops
    assert p.graph.ops["op1"].inputs == ["x"]


def test_ir_version_migration_and_validation() -> None:
    p = _make_identity_program()
    migrate_program_ir_metadata(p)
    validate_program_ir_metadata(p)
    assert p.metadata["ir_schema_version"] == 1
    assert "ir_schema_features" in p.metadata


def test_quant_contract_non_strict_warning() -> None:
    p = _make_identity_program()
    run_ir_canonicalization(p)
    report = run_quantization_contract_validation(
        p, backend="tflm", bit_width=8, strict=False
    )
    assert report["status"] in {"ok", "warning"}
    assert "coverage" in report


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):  # type: ignore[override]
        return self.relu(self.conv(x))


def test_semantic_check_torch_vs_ir_passes() -> None:
    model = TinyNet().eval()
    x = torch.randn(1, 3, 8, 8)
    p = build_program_from_torch(
        model,
        x,
        default_backend="tflm",
        target_hardware="hxwe2",
        bit_width=8,
        metadata={},
    )
    run_ir_canonicalization(p)
    out = run_semantic_check_torch_vs_ir(
        program=p, model=model, example_input=x, rtol=1e-3, atol=1e-4
    )
    assert out["status"] == "pass"
