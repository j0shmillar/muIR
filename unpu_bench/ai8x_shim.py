# unpu_bench/ai8x_shim.py
"""
Minimal ai8x API shim / patcher.

Goal:
- Allow importing ai8x-training model zoo networks even when ai8x-training
  is not installed, by providing/patching common symbols used in model code.

This is ONLY intended for model loading + ONNX export (not bit-accurate).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch import nn


def install_ai8x_shim() -> None:
    """
    Ensure that an 'ai8x' module exists and contains a baseline set of symbols
    commonly referenced by ai8x-training model zoo networks.

    If a real ai8x module is present, we *patch in missing attributes* rather
    than replacing it.
    """
    if "ai8x" in sys.modules:
        m = sys.modules["ai8x"]
    else:
        m = types.ModuleType("ai8x")
        sys.modules["ai8x"] = m

    # -----------------------
    # Shim layer definitions
    # -----------------------

    class Conv2d(nn.Module):
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
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=groups,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.conv(x)

    class Add(nn.Module):
        """ai8x-training sometimes uses ai8x.Add() as a module wrapper for elementwise +."""
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return a + b


    class Sub(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return a - b


    class Mul(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return a * b


    class BitwiseOr(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            # Most ai8x-training models won’t hit this at runtime for float inputs; keep it safe.
            return torch.bitwise_or(a.to(torch.int32), b.to(torch.int32)).to(a.dtype)


    class BitwiseXor(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.bitwise_xor(a.to(torch.int32), b.to(torch.int32)).to(a.dtype)


    class Abs(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.abs(x)


    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True, wide: bool = False, **kwargs: Any):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    class FusedConv2dReLU(nn.Module):
        """
        Common in ai8x-training model zoo. Approx: Conv2d -> ReLU.
        """
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            bias: bool = True,
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
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.act(self.conv(x))

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
            # Approximate BN handling.
            if batchnorm is None or str(batchnorm).lower() in ("noaffine", "batchnorm"):
                self.bn = nn.BatchNorm2d(out_channels, affine=True)
            else:
                self.bn = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.act(self.bn(self.conv(x)))

    class FusedMaxPoolConv2dReLU(nn.Module):
        """
        Approx: MaxPool2d -> Conv2d -> ReLU
        """
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int],
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            pool_size: int | tuple[int, int] = 2,
            pool_stride: int | tuple[int, int] | None = None,
            bias: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__()
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
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.act(self.conv(self.pool(x)))

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
            if batchnorm is None or str(batchnorm).lower() in ("noaffine", "batchnorm"):
                self.bn = nn.BatchNorm2d(out_channels, affine=True)
            else:
                self.bn = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.act(self.bn(self.conv(self.pool(x))))

    # --------------------------------
    # Patch missing symbols onto module
    # --------------------------------
    patch_table = {
        "Conv2d": Conv2d,
        "Linear": Linear,

        # elementwise / utils
        "Add": Add,
        "Sub": Sub,
        "Mul": Mul,
        "Abs": Abs,
        "BitwiseOr": BitwiseOr,
        "BitwiseXor": BitwiseXor,

        # fused conv blocks
        "FusedConv2dReLU": FusedConv2dReLU,
        "FusedConv2dBNReLU": FusedConv2dBNReLU,
        "FusedMaxPoolConv2dReLU": FusedMaxPoolConv2dReLU,
        "FusedMaxPoolConv2dBNReLU": FusedMaxPoolConv2dBNReLU,
    }

    for name, obj in patch_table.items():
        if not hasattr(m, name):
            setattr(m, name, obj)
