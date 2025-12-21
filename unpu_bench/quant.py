# unpu_bench/quant.py

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from .bn_fuse import fuse_batchnorm_in_checkpoint
from .errors import CompilationError

if TYPE_CHECKING:
    from .pipeline import CompileConfig  # pragma: no cover

log = logging.getLogger(__name__)


def _import_ai8x_quant_modules():
    """
    Try to import quantize and tornadocnn from a sibling ai8x-synthesis/izer tree.

    Layout expected (from repo root):
        ai8x-synthesis/
          izer/
            __init__.py
            quantize.py
            tornadocnn.py
            ...
    """
    try:
        # First try plain 'izer' if it's already on sys.path
        from izer import quantize as ai8x_quant  # type: ignore[import]
        from izer import tornadocnn as tc  # type: ignore[import]
        return ai8x_quant, tc
    except ImportError:
        pass

    # Add ../ai8x-synthesis to sys.path and retry
    repo_root = Path(__file__).resolve().parents[1]  # .../uNPU-Bench
    ai8x_synth = repo_root / "ai8x-synthesis"
    if ai8x_synth.is_dir() and str(ai8x_synth) not in sys.path:
        sys.path.insert(0, str(ai8x_synth))
        log.debug("Added ai8x-synthesis to sys.path: %s", ai8x_synth)

    try:
        from izer import quantize as ai8x_quant  # type: ignore[import]
        from izer import tornadocnn as tc  # type: ignore[import]
        return ai8x_quant, tc
    except ImportError as exc:  # noqa: BLE001
        raise CompilationError(
            "Failed to import ai8x-synthesis modules (izer.quantize, izer.tornadocnn).\n"
            "Make sure ai8x-synthesis is present as a sibling directory or installed.\n"
            "Expected layout: <repo-root>/ai8x-synthesis/izer/..."
        ) from exc


def run_ai8x_bn_fuse_and_quantize(
    cfg: Any,
    model_ckpt: str,
    out_dir: str | Path,
) -> Path:
    """
    Load a PyTorch checkpoint, fuse BatchNorm into Conv2d, then run ai8x quantization.

    Returns:
        Path to the quantized checkpoint (model_quantized.pth).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_ckpt_path = Path(model_ckpt)
    if not original_ckpt_path.is_file():
        raise CompilationError(f"ai8x: model checkpoint not found: {original_ckpt_path}")

    bn_fused_path = out_dir / "model_bn_fused.pth"
    quantized_path = out_dir / "model_quantized.pth"

    # 1) BN fuse
    log.info("ai8x: loading checkpoint for BN fusion: %s", original_ckpt_path)
    try:
        ckpt = torch.load(original_ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise CompilationError(
            f"ai8x: failed to load checkpoint '{original_ckpt_path}': {exc}"
        ) from exc

    try:
        ckpt = fuse_batchnorm_in_checkpoint(ckpt)
    except Exception as exc:  # noqa: BLE001
        raise CompilationError(f"ai8x: BN fusion failed: {exc}") from exc

    torch.save(ckpt, bn_fused_path)
    log.info("ai8x: saved BN-fused checkpoint to %s", bn_fused_path)

    # 2) Quantization
    ai8x_quant, tc = _import_ai8x_quant_modules()

    device_name = getattr(cfg, "ai8x_device", None) or 85
    device_name = 85
    log.info(
        "ai8x: starting quantization of BN-fused checkpoint for device %s", device_name
    )

    tc.dev = tc.get_device(device_name)

    q_args = argparse.Namespace(
        input=str(bn_fused_path),
        output=str(quantized_path),
        config_file=getattr(cfg, "ai8x_config_file", None),
        device=device_name,
        clip_mode=None,
        qat_weight_bits=None,
        verbose=getattr(cfg, "debug", False),
        scale=None,
        stddev=None,
    )

    try:
        ai8x_quant.convert_checkpoint(q_args.input, q_args.output, q_args)
    except Exception as exc:  # noqa: BLE001
        raise CompilationError(
            f"ai8x convert_checkpoint failed while quantizing '{bn_fused_path}': {exc}"
        ) from exc

    log.info("ai8x: quantization complete. Quantized checkpoint: %s", quantized_path)
    return quantized_path
