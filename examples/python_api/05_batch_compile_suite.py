from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir
from models.reference_impls import REFERENCE_MODEL_REGISTRY


RUN_MATRIX = [
    ("dscnn", "tflm", "hxwe2"),
    ("resnet18", "vela", "hxwe2"),
    ("mobilenet_v2", "cvi", "bm1684x"),
    ("convmixer", "eiq", "mcxn947"),
]


def main() -> None:
    root = Path("out/examples/05_batch_suite")
    root.mkdir(parents=True, exist_ok=True)

    for model_name, backend, hw in RUN_MATRIX:
        model = REFERENCE_MODEL_REGISTRY[model_name](num_classes=10).eval()
        ckpt = Path("ckpts/random_reference") / f"{model_name}.random.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        out_dir = root / f"{model_name}_{backend}"
        out_dir.mkdir(parents=True, exist_ok=True)

        res = muir.convert(
            model,
            backend=backend,
            target_hardware=hw,
            out_dir=str(out_dir),
            input_shape=(1, 3, 32, 32),
            output_shape=(1, 10),
            input_names="image",
            output_names="logits",
        )
        print(
            f"[ok] {model_name:16s} -> {backend:4s}: {len(res['artifacts'])} artifacts"
        )

    print("Outputs:", root)


if __name__ == "__main__":
    main()
