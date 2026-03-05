from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from unpu_bench.backend_runtime import execute_compiled_model, load_compiled_model
from unpu_bench.cli import main


def _write_complex_models(path: Path) -> None:
    path.write_text(
        """
import torch
from torch import nn


class ResidualStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.b1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Conv2d(8, 8, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.act(self.stem(x))
        residual = x
        x = self.act(self.b1(x))
        x = self.b2(x)
        x = self.act(x + residual)
        x = self.pool(x)
        x = self.act(self.head(x))
        return self.gap(x)


class DualPathFuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.b = nn.Conv2d(3, 8, kernel_size=1, padding=0)
        self.mix = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        p = self.act(self.a(x))
        q = self.act(self.b(x))
        x = self.act(p + q)
        x = self.act(self.mix(x))
        x = self.avg(x)
        return self.gap(x)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _load_model_class(module_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location("test_models_mod", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _compile(
    *,
    tmp_path: Path,
    model_py: Path,
    model_class: str,
    target_format: str,
    target_hardware: str,
    model_ckpt: Path,
) -> tuple[Path, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16, dtype=torch.float32)

    out_dir = tmp_path / f"out_{target_format}_{model_class}"
    rc = main(
        [
            "--target-format",
            target_format,
            "--target-hardware",
            target_hardware,
            "--bit-width",
            "8",
            "--model-py",
            str(model_py),
            "--model-class",
            model_class,
            "--model-ckpt",
            str(model_ckpt),
            "--input-shape",
            "1 3 16 16",
            "--output-shape",
            "1 8 1 1",
            "--input-names",
            "input",
            "--output-names",
            "output",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    return out_dir, x


@pytest.mark.parametrize(
    ("target_format", "target_hardware"),
    [
        ("tflm", "hxwe2"),
        ("vela", "hxwe2"),
        ("cvi", "bm1684x"),
        ("eiq", "mcxn947"),
    ],
)
@pytest.mark.parametrize("model_class", ["ResidualStack", "DualPathFuse"])
def test_backend_compiled_model_matches_torch(
    tmp_path: Path,
    target_format: str,
    target_hardware: str,
    model_class: str,
) -> None:
    model_py = tmp_path / "complex_models.py"
    _write_complex_models(model_py)
    cls = _load_model_class(model_py, model_class)
    model = cls().eval()
    ckpt = tmp_path / f"{model_class}.pth"
    torch.save(model.state_dict(), ckpt)

    out_dir, x = _compile(
        tmp_path=tmp_path,
        model_py=model_py,
        model_class=model_class,
        target_format=target_format,
        target_hardware=target_hardware,
        model_ckpt=ckpt,
    )

    with torch.no_grad():
        expected = model(x)

    compiled_path = out_dir / target_format / f"model.{target_format}.compiled.json"
    assert compiled_path.exists()

    compiled = load_compiled_model(compiled_path)
    outputs = execute_compiled_model(compiled, {"x": x})
    assert compiled["graph"]["outputs"], "compiled graph must expose outputs"
    out_id = compiled["graph"]["outputs"][0]
    got = outputs[out_id]

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
