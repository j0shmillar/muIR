# unpu_bench/ai8x_shim.py
"""
Minimal ai8x API shim for loading ai8x model zoo networks without requiring
the full ai8x Python module.

This is ONLY intended for:
  - Torch model loading
  - ONNX export

and not for correctness/bit-accurate simulation.
"""

import sys
import types
from typing import Any

import torch
from torch import nn

# TODO 
# - only temporary; user should just input ordinary Torch models 

def install_ai8x_shim() -> None:
    """Install a lightweight 'ai8x' module into sys.modules if it's missing."""
    if "ai8x" in sys.modules:
        # Real ai8x is already available, or shim already installed.
        return

    m = types.ModuleType("ai8x")


    class Conv2d(nn.Module):
        """
        Minimal shim for ai8x.Conv2d.

        Accepts ai8x-style arguments but internally just builds a plain nn.Conv2d.
        All ai8x-specific arguments (pooling, activation, wide, weight_bits, etc.)
        are ignored.
        """
        def __init__(  # pylint: disable=too-many-arguments
            self,
            in_channels,
            out_channels,
            kernel_size,
            op: str = "Conv2d",
            pooling=None,
            pool_size=2,
            pool_stride=2,
            pool_dilation=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
            groups=1,
            eps=1e-5,
            momentum=0.05,
        ):
            super().__init__()

            # Normalize kernel_size for nn.Conv2d
            if isinstance(kernel_size, tuple):
                k = kernel_size
            else:
                k = kernel_size

            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=groups,
            )

            # We deliberately ignore pooling, activation, batchnorm etc. here.
            # If you later need closer behavior, you can wrap with nn.MaxPool2d,
            # nn.BatchNorm2d, nn.ReLU, etc.

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.conv(x)

    class FusedConv2dBNReLU(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            bias: bool = False,
            batchnorm: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            # ai8x uses various BN modes; for the shim we just use standard BN if requested
            if batchnorm is None or batchnorm.lower() in ("noaffine", "batchnorm"):
                self.bn = nn.BatchNorm2d(out_channels, affine=True)
            else:
                self.bn = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.conv(x)
            x = self.bn(x)
            return self.act(x)

    class FusedMaxPoolConv2dBNReLU(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            pool_size: int | tuple[int, int] = 2,
            pool_stride: int | tuple[int, int] | None = None,
            bias: bool = False,
            batchnorm: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            # ai8x fused layer does pool then conv+bn+relu; we'll approximate that
            if pool_stride is None:
                pool_stride = pool_size
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            if batchnorm is None or batchnorm.lower() in ("noaffine", "batchnorm"):
                self.bn = nn.BatchNorm2d(out_channels, affine=True)
            else:
                self.bn = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.pool(x)
            x = self.conv(x)
            x = self.bn(x)
            return self.act(x)

    class Linear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            wide: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            # ignore 'wide' and other ai8x-specific flags; just map to nn.Linear
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    # Export symbols on the shim module
    m.Conv2d = Conv2d
    m.FusedConv2dBNReLU = FusedConv2dBNReLU
    m.FusedMaxPoolConv2dBNReLU = FusedMaxPoolConv2dBNReLU
    m.Linear = Linear

    sys.modules["ai8x"] = m