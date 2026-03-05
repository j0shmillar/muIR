from unpu_bench.errors import CompilationError
from unpu_bench.muir import Graph, Op, Program, Tensor, TensorType
from unpu_bench.passes import run_ir_validation


def test_ir_validation_rejects_missing_tensor() -> None:
    g = Graph(
        name="g",
        inputs=["x"],
        outputs=["y"],
        initializers=[],
        tensors={
            "x": Tensor(id="x", type=TensorType(shape=[1], dtype="f32"), role="input"),
        },
        ops={
            "op_0": Op(id="op_0", kind="Relu", inputs=["x"], outputs=["y"], attrs={}),
        },
        op_order=["op_0"],
    )
    p = Program(graph=g, partitions=[])

    try:
        run_ir_validation(p)
        raise AssertionError("Expected validation failure")
    except CompilationError as exc:
        assert "graph references missing tensor" in str(exc)
