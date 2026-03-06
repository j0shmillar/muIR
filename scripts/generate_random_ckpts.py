#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mcu_sota import MODEL_REGISTRY as MCU_MODEL_REGISTRY
from models.reference_impls import REFERENCE_MODEL_REGISTRY


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate random checkpoints for MCU-scale PyTorch models."
    )
    p.add_argument(
        "--registry",
        choices=["mcu", "reference"],
        default="reference",
        help="Model registry to use.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("ckpts/random_reference"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-classes", type=int, default=10)
    return p


def main() -> int:
    args = _make_parser().parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_registry = (
        REFERENCE_MODEL_REGISTRY if args.registry == "reference" else MCU_MODEL_REGISTRY
    )

    manifest: dict[str, dict[str, object]] = {}
    for name, cls in model_registry.items():
        model = cls(num_classes=args.num_classes).eval()
        ckpt_path = args.out_dir / f"{name}.random.pth"
        torch.save(model.state_dict(), ckpt_path)
        n_params = sum(p.numel() for p in model.parameters())
        manifest[name] = {
            "checkpoint": str(ckpt_path),
            "num_params": int(n_params),
            "num_classes": args.num_classes,
            "seed": args.seed,
            "registry": args.registry,
        }
        print(f"[ok] {name:20s} -> {ckpt_path} ({n_params} params)")

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
