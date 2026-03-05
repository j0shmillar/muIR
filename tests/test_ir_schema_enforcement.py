from pathlib import Path

import pytest

from unpu_bench.capabilities.ir_schema import load_ir_capabilities
from unpu_bench.muir import Graph, Op, Program, Tensor, TensorType
from unpu_bench.passes import run_legality_check


def test_ir_schema_requires_supported_version(tmp_path: Path) -> None:
    caps = tmp_path / "bad_caps.yaml"
    caps.write_text(
        """
fallback_backend: cpu
ops: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_ir_capabilities(caps, backend="tflm")


def test_ir_legality_enforces_layout_and_required_attrs(tmp_path: Path) -> None:
    caps = tmp_path / "caps.yaml"
    caps.write_text(
        """
schema_version: 1
fallback_backend: cpu
ops:
  - op: Conv
    inputs:
      - dtypes: [f32]
        rank: 4
        layouts: [NCHW]
      - dtypes: [f32]
        rank: 4
        layouts: [NCHW]
    outputs:
      - dtypes: [f32]
        rank: 4
        layouts: [NCHW]
    attrs:
      group: {required: true, min: 1}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    g = Graph(
        name="g",
        inputs=["x"],
        outputs=["y"],
        initializers=["w"],
        tensors={
            "x": Tensor(id="x", type=TensorType(shape=[1, 3, 8, 8], dtype="f32", layout="NHWC"), role="input"),
            "w": Tensor(
                id="w",
                type=TensorType(shape=[4, 3, 3, 3], dtype="f32", layout="NCHW"),
                role="initializer",
                is_constant=True,
            ),
            "y": Tensor(id="y", type=TensorType(shape=[1, 4, 8, 8], dtype="f32", layout="NCHW"), role="output"),
        },
        ops={
            "op_0": Op(id="op_0", kind="Conv", inputs=["x", "w"], outputs=["y"], attrs={}),
        },
        op_order=["op_0"],
    )
    p = Program(graph=g, partitions=[])

    run_legality_check(p, backend="tflm", caps_path=caps)
    assert p.graph.ops["op_0"].preferred_backend == "cpu"
