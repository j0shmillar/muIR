import torch
from torch import nn

from unpu_bench.muir import build_program_from_torch
from unpu_bench.passes import run_ir_canonicalization, run_ir_validation


class DepthwiseTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dw = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.dw(x))


def test_ir_canonicalization_marks_depthwise_and_layouts() -> None:
    model = DepthwiseTiny().eval()
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
    run_ir_validation(p)

    op0 = p.graph.ops["op_0"]
    assert op0.kind == "DepthwiseConv2d"
    assert op0.attrs["kernel_shape"] == [3, 3]
    assert op0.attrs["strides"] == [1, 1]
    assert op0.attrs["dilations"] == [1, 1]
    assert op0.attrs["pads"] == [1, 1, 1, 1]
    assert op0.attrs["group"] == 3
    assert p.graph.tensors["x"].type.layout == "NCHW"
