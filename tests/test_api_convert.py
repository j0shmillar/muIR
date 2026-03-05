from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

import muir


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):  # type: ignore[override]
        return self.relu(self.conv(x))


def test_convert_torch_via_top_level_muir(tmp_path: Path) -> None:
    out_dir = tmp_path / "torch_out"
    res = muir.convert(
        _TinyNet(),
        backend="tflm",
        target_hardware="hxwe2",
        out_dir=str(out_dir),
        input_shape=(1, 3, 8, 8),
        output_shape=(1, 4, 8, 8),
    )

    assert res["out_dir"] == str(out_dir)
    assert (out_dir / "program.json").exists()
    assert (out_dir / "tflm" / "model.tflm.compiled.json").exists()


def test_convert_tflite_path(tmp_path: Path) -> None:
    tflite = tmp_path / "tiny.tflite"
    tflite.write_bytes(b"FAKE_TFLITE")
    out_dir = tmp_path / "tflite_out"

    res = muir.convert(
        str(tflite),
        backend="tflm",
        target_hardware="hxwe2",
        out_dir=str(out_dir),
    )

    assert "artifacts" in res
    program = json.loads((out_dir / "program.json").read_text(encoding="utf-8"))
    assert program["metadata"]["frontend"] == "tflite_stub"
