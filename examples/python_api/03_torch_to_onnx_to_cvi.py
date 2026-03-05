from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir
from models.mcu_sota import MobileNetV2Tiny


def main() -> None:
    out_dir = Path("out/examples/03_onnx_to_cvi")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MobileNetV2Tiny(num_classes=10).eval()
    ckpt = Path("ckpts/random_mcu/mobilenetv2_tiny.random.pth")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    dummy = torch.randn(1, 3, 32, 32)
    onnx_path = out_dir / "mobilenetv2_tiny.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    result = muir.convert(
        str(onnx_path),
        backend="cvi",
        target_hardware="bm1684x",
        out_dir=str(out_dir),
        bit_width=8,
        input_names="image",
        output_names="logits",
    )
    print("Artifacts:", [a["path"] for a in result["artifacts"]])
    print("Program:", out_dir / "program.json")


if __name__ == "__main__":
    main()
