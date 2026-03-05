from pathlib import Path

import onnx
from onnx import TensorProto, helper

from unpu_bench.muir import build_program_from_onnx


def test_build_program_from_onnx_tracks_roles_and_order(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])

    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [8, 3, 3, 3],
        [0.1] * (8 * 3 * 3 * 3),
    )

    conv = helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["y"],
        name="conv0",
        strides=[1, 1],
    )

    graph = helper.make_graph([conv], "g", [x], [y], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    model_path = tmp_path / "m.onnx"
    onnx.save(model, model_path)

    p = build_program_from_onnx(
        str(model_path),
        default_backend="ai8x",
        target_hardware="max78000",
        bit_width=8,
        metadata={},
    )

    assert p.graph.op_order == ["op_0"]
    assert p.graph.initializers == ["w"]
    assert p.graph.tensors["x"].role == "input"
    assert p.graph.tensors["w"].role == "initializer"
    assert p.graph.tensors["w"].is_constant is True
    assert p.graph.tensors["y"].role == "output"
    assert p.graph.tensors["x"].consumers == ["op_0"]
    assert p.graph.tensors["y"].producer == "op_0"
