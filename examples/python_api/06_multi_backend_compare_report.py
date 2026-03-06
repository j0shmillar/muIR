from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir
from models.reference_impls import REFERENCE_MODEL_REGISTRY


BACKENDS = [
    ("tflm", "hxwe2"),
    ("vela", "hxwe2"),
    ("cvi", "bm1684x"),
    ("eiq", "mcxn947"),
]


def main() -> None:
    root = Path("out/examples/06_multi_backend_compare")
    root.mkdir(parents=True, exist_ok=True)

    model_name = "mobilenet_v2"
    model = REFERENCE_MODEL_REGISTRY[model_name](num_classes=10).eval()

    ckpt = Path("ckpts/random_reference") / f"{model_name}.random.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[info] Loaded checkpoint: {ckpt}")
    else:
        print(f"[warn] Checkpoint not found, using random weights: {ckpt}")

    program_jsons: list[str] = []
    for backend, hw in BACKENDS:
        out_dir = root / backend
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
        pj = str(Path(res["out_dir"]) / "program.json")
        program_jsons.append(pj)
        print(f"[ok] {backend:4s} -> {pj}")

    report = muir.compare_runs(
        program_jsons,
        out_dir=str(root / "reports"),
        basename=f"{model_name}_cross_backend",
    )

    print("\nComparison report:")
    print(f"- CSV: {report['csv']}")
    print(f"- MD:  {report['markdown']}")


if __name__ == "__main__":
    main()
