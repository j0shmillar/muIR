from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_ch=in_ch)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.use_res_connect:
            return x + y
        return y


class MobileNetV2(nn.Module):
    def __init__(
        self, num_classes: int = 10, in_ch: int = 3, width_mult: float = 1.0
    ) -> None:
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
            )
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
            )
        )
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return MobileNetV2(num_classes=num_classes, in_ch=in_ch)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSCNN(nn.Module):
    """Reference DS-CNN style model using repeated depthwise-separable blocks."""

    def __init__(self, num_classes: int = 10, in_ch: int = 3, width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConv(width, width, stride=1),
            DepthwiseSeparableConv(width, width * 2, stride=2),
            DepthwiseSeparableConv(width * 2, width * 2, stride=1),
            DepthwiseSeparableConv(width * 2, width * 4, stride=2),
            DepthwiseSeparableConv(width * 4, width * 4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def dscnn(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return DSCNN(num_classes=num_classes, in_ch=in_ch)


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dw(x)
        return self.pw(x)


class ConvMixer(nn.Module):
    """ConvMixer-style reference implementation."""

    def __init__(
        self,
        num_classes: int = 10,
        in_ch: int = 3,
        dim: int = 128,
        depth: int = 8,
        kernel_size: int = 5,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(
            *[ConvMixerBlock(dim=dim, kernel_size=kernel_size) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def convmixer(num_classes: int = 10, in_ch: int = 3) -> nn.Module:
    return ConvMixer(num_classes=num_classes, in_ch=in_ch)


REFERENCE_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "mobilenet_v2": mobilenet_v2,
    "dscnn": dscnn,
    "convmixer": convmixer,
}
