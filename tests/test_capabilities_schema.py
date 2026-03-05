from pathlib import Path

from unpu_bench.capabilities.schema import check_op_legality, load_capabilities
from unpu_bench.tosa_ir import TensorSig, TosaOpSig


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_op_variants_are_retained() -> None:
    cap = load_capabilities(
        REPO_ROOT / "unpu_bench" / "capabilities" / "ai8x.yaml",
        backend="ai8x",
    )

    # ai8x capability file intentionally defines multiple legal signatures.
    assert len(cap.ops["tosa.conv2d"]) >= 2
    assert len(cap.ops["tosa.const"]) >= 2


def test_legality_accepts_matching_conv2d_variant() -> None:
    cap = load_capabilities(
        REPO_ROOT / "unpu_bench" / "capabilities" / "ai8x.yaml",
        backend="ai8x",
    )

    op = TosaOpSig(
        op_name="tosa.conv2d",
        operand_names=["%a", "%w", "%b", "%s0", "%s1"],
        result_names=["%r"],
        operands=[
            TensorSig(shape=[1, 32, 32, 3], dtype="f32", raw="tensor<1x32x32x3xf32>"),
            TensorSig(shape=[16, 3, 3, 3], dtype="f32", raw="tensor<16x3x3x3xf32>"),
            TensorSig(shape=[16], dtype="i32", raw="tensor<16xi32>"),
            TensorSig(shape=[1], dtype="i32", raw="tensor<1xi32>"),
            TensorSig(shape=[1], dtype="i32", raw="tensor<1xi32>"),
        ],
        results=[
            TensorSig(shape=[1, 32, 32, 16], dtype="f32", raw="tensor<1x32x32x16xf32>"),
        ],
        attrs={"stride": "[1, 1]", "dilation": "[1, 1]", "pad": "[1, 1, 1, 1]"},
        location=(1, "%r = tosa.conv2d ..."),
    )

    ok, reasons = check_op_legality(op, cap)
    assert ok, reasons
