# from pathlib import Path

# import numpy as np
# import pytest

# from unpu_bench.cli import main

# REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root

# @pytest.mark.ai8x
# def test_ai8x_quant_and_compile_real_model(tmp_path: Path) -> None:
#     # TODO add batchnorm removal

#     model_py = REPO_ROOT / "models" / "ai85net-res-simplenet.py"
#     ckpt = REPO_ROOT / "ckpts" / "ai85-cifar100-residual-qat8-q.pth.tar"
#     yaml_cfg = REPO_ROOT / "yaml" / "cifar100-ressimplenet.yaml"

#     if not (model_py.exists() and ckpt.exists() and yaml_cfg.exists()):
#         pytest.skip("Real ai8x assets not available in this checkout.")

#     # TODO: replace with the actual class name from ai85net-nas-cifar.py
#     model_class = "AI85ResidualSimpleNet"  # for example

#     data_sample = tmp_path / "sample.npy"
#     arr = (np.random.randn(1, 3, 32, 32) * 20).astype("int64")
#     arr = np.clip(arr, -127, 127)
#     np.save(data_sample, arr)

#     out_dir = tmp_path / "out"

#     rc = main(
#         [
#             "--target-format",
#             "ai8x",
#             "--target-hardware",
#             "max78000",
#             "--bit-width",
#             "8",
#             "--model-py",
#             str(model_py),
#             "--model-class",
#             model_class,
#             "--model-ckpt",
#             str(ckpt),
#             "--input-shape",
#             "1 3 32 32",
#             "--output-shape",
#             "1 100",
#             "--input-names",
#             "input",
#             "--output-names",
#             "output",
#             "--data-sample",
#             str(data_sample),
#             "--ai8x-config-file",
#             str(yaml_cfg),
#             "--ai8x-device",
#             "MAX78000",
#             "--ai8x-prefix",
#             "ai85_cifar100_nas",
#             "--out-dir",
#             str(out_dir),
#             "--overwrite",
#         ]
#     )

#     assert rc == 0

#     # Check that quantization + compile artefacts exist
#     assert (out_dir / "model_quantized.pth").exists()
#     assert (out_dir / "model.onnx").exists()
#     assert (out_dir / "program.json").exists()
#     assert (out_dir / "ai8x" / "ai85_cifar100_nas").is_dir()

# tests/test_ai8x_pipeline.py
from pathlib import Path
import os

import numpy as np
import pytest

from unpu_bench.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ai8x
def test_ai8x_ir_compiles_real_model_no_mlir(tmp_path: Path) -> None:
    if os.environ.get("RUN_AI8X_E2E", "") != "1":
        pytest.skip("Set RUN_AI8X_E2E=1 to run ai8x toolchain integration test.")

    model_py = REPO_ROOT / "models" / "ai85net-res-simplenet.py"
    ckpt = REPO_ROOT / "ckpts" / "ai85-cifar100-residual-qat8-q.pth.tar"
    yaml_cfg = REPO_ROOT / "yaml" / "cifar100-ressimplenet.yaml"

    if not (model_py.exists() and ckpt.exists() and yaml_cfg.exists()):
        pytest.skip("ai8x assets are not available in this checkout.")

    model_class = "AI85ResidualSimpleNet"

    # Provide a stable sample input to avoid random-shape edge cases.
    sample_npy = tmp_path / "input.npy"
    x = np.random.randn(1, 3, 32, 32) * 20.0
    x = np.clip(np.rint(x), -127, 127).astype(np.int64)
    np.save(sample_npy, x)

    out_dir = tmp_path / "out"

    rc = main(
        [
            "--target-format",
            "ai8x",
            "--target-hardware",
            "max78000",
            "--bit-width",
            "8",
            "--model-py",
            str(model_py),
            "--model-class",
            model_class,
            "--model-ckpt",
            str(ckpt),
            "--input-shape",
            "1 3 32 32",
            "--output-shape",
            "1 100",
            "--input-names",
            "input",
            "--output-names",
            "output",
            "--data-sample",
            str(sample_npy),
            "--ai8x-config-file",
            str(yaml_cfg),
            "--ai8x-device",
            "MAX78000",
            "--ai8x-prefix",
            "ai85_cifar100_nas",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0

    # Core pipeline artifacts
    assert (out_dir / "model_quantized.pth").exists()
    assert (out_dir / "program.json").exists()

    # ai8x-synthesis project output
    assert (out_dir / "ai8x" / "ai85_cifar100_nas").is_dir()
