from __future__ import annotations

import torch
from torch import nn


class DWSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class DSCNNSmall(nn.Module):
    """DS-CNN style tiny model for MCU-scale vision."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, width: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSeparableBlock(width, width),
            DWSeparableBlock(width, width),
            DWSeparableBlock(width, width * 2, stride=2),
            DWSeparableBlock(width * 2, width * 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int = 2) -> None:
        super().__init__()
        mid = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                mid,
                mid,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid,
                bias=False,
            ),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_res else y


class MobileNetV2Tiny(nn.Module):
    """MobileNetV2-like micro variant."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            InvertedResidual(width, width, stride=1, expand=2),
            InvertedResidual(width, width * 2, stride=2, expand=2),
            InvertedResidual(width * 2, width * 2, stride=1, expand=2),
            InvertedResidual(width * 2, width * 3, stride=2, expand=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(width * 3, width * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ident = x if self.down is None else self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + ident)
        return x


class TinyResNet8(nn.Module):
    """ResNet-8 style compact classifier."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(width, width, stride=1)
        self.layer2 = BasicBlock(width, width * 2, stride=2)
        self.layer3 = BasicBlock(width * 2, width * 4, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class TinyConvMixer(nn.Module):
    def __init__(
        self, num_classes: int = 10, in_ch: int = 3, dim: int = 32, depth: int = 4
    ) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)


MODEL_REGISTRY = {
    "dscnn_small": DSCNNSmall,
    "mobilenetv2_tiny": MobileNetV2Tiny,
    "tiny_resnet8": TinyResNet8,
    "tiny_convmixer": TinyConvMixer,
}
