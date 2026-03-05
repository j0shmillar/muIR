from pathlib import Path

from unpu_bench.muir import Graph, Op, Program, Tensor, TensorType
from unpu_bench.passes import run_legality_check, run_partitioning


REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_program_with_ops(kinds: list[str]) -> Program:
    tensors = {
        "x": Tensor(id="x", type=TensorType(shape=[1, 3, 32, 32], dtype="f32", layout="NCHW"), role="input"),
        "w": Tensor(
            id="w",
            type=TensorType(shape=[8, 3, 3, 3], dtype="f32", layout="NCHW"),
            role="initializer",
            is_constant=True,
        ),
        "y": Tensor(id="y", type=TensorType(shape=[1, 8, 30, 30], dtype="f32", layout="NCHW")),
        "z": Tensor(id="z", type=TensorType(shape=[1, 8, 30, 30], dtype="f32", layout="NCHW"), role="output"),
    }

    ops = {}
    op_order = []
    for i, kind in enumerate(kinds):
        op_id = f"op_{i}"
        if kind == "Conv":
            ins = ["x", "w"]
            outs = ["y"]
            attrs = {
                "kernel_shape": [3, 3],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1],
                "dilations": [1, 1],
                "group": 1,
            }
        else:
            ins = ["y"]
            outs = ["z"]
            attrs = {}
        ops[op_id] = Op(id=op_id, kind=kind, inputs=ins, outputs=outs, attrs=attrs)
        op_order.append(op_id)

    g = Graph(
        name="g",
        inputs=["x"],
        outputs=["z"],
        initializers=["w"],
        tensors=tensors,
        ops=ops,
        op_order=op_order,
    )
    return Program(graph=g, partitions=[])


def test_ir_legality_marks_supported_and_unsupported_ops() -> None:
    p = _toy_program_with_ops(["Conv", "Identity"])

    run_legality_check(
        p,
        backend="ai8x",
        caps_path=REPO_ROOT / "unpu_bench" / "capabilities" / "ir_ai8x.yaml",
    )

    assert p.graph.ops["op_0"].preferred_backend == "ai8x"
    assert p.graph.ops["op_1"].preferred_backend == "cpu"


def test_ir_partitioning_produces_cpu_suffix_for_illegal_tail() -> None:
    p = _toy_program_with_ops(["Conv", "Identity"])

    run_legality_check(
        p,
        backend="ai8x",
        caps_path=REPO_ROOT / "unpu_bench" / "capabilities" / "ir_ai8x.yaml",
    )
    run_partitioning(p)

    assert [part.id for part in p.partitions] == ["ai8x_core", "cpu_suffix"]
