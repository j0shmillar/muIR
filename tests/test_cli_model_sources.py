from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from unpu_bench.cli import main


def _write_tiny_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Relu", inputs=["input"], outputs=["output"], name="relu0")
    graph = helper.make_graph([node], "tiny", [x], [y], initializer=[])
    model = helper.make_model(graph, producer_name="unpu-bench-tests")
    onnx.save(model, str(path))


def test_cli_accepts_onnx_source(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _write_tiny_onnx(model_path)
    out_dir = tmp_path / "out_onnx"

    rc = main(
        [
            "--target-format",
            "cvi",
            "--target-hardware",
            "bm1684x",
            "--bit-width",
            "8",
            "--model-onnx",
            str(model_path),
            "--output-names",
            "output",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0
    assert (out_dir / "program.json").exists()
    assert (out_dir / "cvi" / "model.cvi.compiled.json").exists()


def test_cli_accepts_tflite_source(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.tflite"
    model_path.write_bytes(b"FAKE_TFLITE")
    out_dir = tmp_path / "out_tflite"

    rc = main(
        [
            "--target-format",
            "tflm",
            "--target-hardware",
            "hxwe2",
            "--bit-width",
            "8",
            "--model-tflite",
            str(model_path),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0
    assert (out_dir / "program.json").exists()
    assert (out_dir / "tflm" / "model.tflm.compiled.json").exists()
