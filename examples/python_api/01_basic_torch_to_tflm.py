from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir
from models.reference_impls import REFERENCE_MODEL_REGISTRY


def main() -> None:
    out_dir = Path("out/examples/01_basic_tflm")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "dscnn"
    model = REFERENCE_MODEL_REGISTRY[model_name](num_classes=10).eval()
    ckpt = Path("ckpts/random_reference") / f"{model_name}.random.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    result = muir.convert(
        model,
        backend="tflm",
        target_hardware="hxwe2",
        out_dir=str(out_dir),
        input_shape=(1, 3, 32, 32),
        output_shape=(1, 10),
        input_names="image",
        output_names="logits",
    )
    print("Artifacts:", [a["path"] for a in result["artifacts"]])
    print("Program:", out_dir / "program.json")


if __name__ == "__main__":
    main()
