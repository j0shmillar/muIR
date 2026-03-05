from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir
from models.mcu_sota import MODEL_REGISTRY


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all Python API end-to-end demos with realistic MCU models."
    )
    p.add_argument("--out-root", type=Path, default=Path("out/examples/run_all"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("ckpts/random_mcu"))
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--regen-ckpts", action="store_true")
    return p


def _ensure_ckpts(
    ckpt_dir: Path, *, num_classes: int, seed: int, force: bool
) -> dict[str, str]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    ckpts: dict[str, str] = {}
    manifest: dict[str, dict[str, Any]] = {}

    for model_name, ctor in MODEL_REGISTRY.items():
        path = ckpt_dir / f"{model_name}.random.pth"
        if force or not path.exists():
            model = ctor(num_classes=num_classes).eval()
            torch.save(model.state_dict(), path)
        ckpts[model_name] = str(path)

        model = ctor(num_classes=num_classes).eval()
        n_params = sum(p.numel() for p in model.parameters())
        manifest[model_name] = {
            "checkpoint": str(path),
            "seed": seed,
            "num_classes": num_classes,
            "num_params": int(n_params),
        }

    (ckpt_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return ckpts


def _run_torch_convert(
    *,
    model_name: str,
    backend: str,
    hardware: str,
    out_dir: Path,
    ckpt_path: str,
    num_classes: int,
) -> dict[str, Any]:
    model = MODEL_REGISTRY[model_name](num_classes=num_classes).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    t0 = time.time()
    res = muir.convert(
        model,
        backend=backend,
        target_hardware=hardware,
        out_dir=str(out_dir),
        bit_width=8,
        input_shape=(1, 3, 32, 32),
        output_shape=(1, num_classes),
        input_names="image",
        output_names="logits",
    )
    dt_ms = int((time.time() - t0) * 1000)
    return {
        "case": f"torch_{model_name}_{backend}",
        "backend": backend,
        "hardware": hardware,
        "source": "torch",
        "model": model_name,
        "out_dir": str(out_dir),
        "artifacts": [a["path"] for a in res["artifacts"]],
        "elapsed_ms": dt_ms,
    }


def _run_onnx_case(
    *, out_dir: Path, ckpt_path: str, num_classes: int
) -> dict[str, Any]:
    model = MODEL_REGISTRY["mobilenetv2_tiny"](num_classes=num_classes).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
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
    t0 = time.time()
    res = muir.convert(
        str(onnx_path),
        backend="cvi",
        target_hardware="bm1684x",
        out_dir=str(out_dir),
        bit_width=8,
        input_names="image",
        output_names="logits",
    )
    dt_ms = int((time.time() - t0) * 1000)
    return {
        "case": "onnx_mobilenetv2_tiny_cvi",
        "backend": "cvi",
        "hardware": "bm1684x",
        "source": "onnx",
        "model": "mobilenetv2_tiny",
        "out_dir": str(out_dir),
        "artifacts": [a["path"] for a in res["artifacts"]],
        "elapsed_ms": dt_ms,
    }


def _run_tflite_case(*, out_dir: Path) -> dict[str, Any]:
    tflite_path = out_dir / "placeholder.tflite"
    tflite_path.write_bytes(b"FAKE_TFLITE")
    t0 = time.time()
    res = muir.convert(
        str(tflite_path),
        backend="eiq",
        target_hardware="mcxn947",
        out_dir=str(out_dir),
        bit_width=8,
    )
    dt_ms = int((time.time() - t0) * 1000)
    return {
        "case": "tflite_stub_eiq",
        "backend": "eiq",
        "hardware": "mcxn947",
        "source": "tflite",
        "model": "placeholder",
        "out_dir": str(out_dir),
        "artifacts": [a["path"] for a in res["artifacts"]],
        "elapsed_ms": dt_ms,
    }


def _write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# run_all summary",
        "",
        "| case | source | backend | hardware | artifacts | ms |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['source']} | {r['backend']} | {r['hardware']} | {len(r['artifacts'])} | {r['elapsed_ms']} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    ckpts = _ensure_ckpts(
        args.ckpt_dir,
        num_classes=args.num_classes,
        seed=args.seed,
        force=args.regen_ckpts,
    )
    print(f"[ckpt] using random checkpoints from: {args.ckpt_dir}")

    rows: list[dict[str, Any]] = []
    run_matrix = [
        ("dscnn_small", "tflm", "hxwe2"),
        ("tiny_resnet8", "vela", "hxwe2"),
        ("mobilenetv2_tiny", "cvi", "bm1684x"),
        ("tiny_convmixer", "eiq", "mcxn947"),
    ]
    for model_name, backend, hw in run_matrix:
        run_dir = out_root / f"torch_{model_name}_{backend}"
        run_dir.mkdir(parents=True, exist_ok=True)
        row = _run_torch_convert(
            model_name=model_name,
            backend=backend,
            hardware=hw,
            out_dir=run_dir,
            ckpt_path=ckpts[model_name],
            num_classes=args.num_classes,
        )
        rows.append(row)
        print(f"[ok] {row['case']} ({row['elapsed_ms']} ms)")

    onnx_dir = out_root / "onnx_mobilenetv2_tiny_cvi"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_row = _run_onnx_case(
        out_dir=onnx_dir,
        ckpt_path=ckpts["mobilenetv2_tiny"],
        num_classes=args.num_classes,
    )
    rows.append(onnx_row)
    print(f"[ok] {onnx_row['case']} ({onnx_row['elapsed_ms']} ms)")

    tflite_dir = out_root / "tflite_stub_eiq"
    tflite_dir.mkdir(parents=True, exist_ok=True)
    tflite_row = _run_tflite_case(out_dir=tflite_dir)
    rows.append(tflite_row)
    print(f"[ok] {tflite_row['case']} ({tflite_row['elapsed_ms']} ms)")

    _write_summary(rows, out_root)
    print(f"[done] summary: {out_root / 'summary.md'}")
    print(f"[done] json:    {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
