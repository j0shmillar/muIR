# unpu_bench/tosa_lowering.py
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from .errors import CompilationError


@dataclass(frozen=True)
class LoweringResult:
    tosa_mlir_path: Path
    toolchain: str          # "torch_mlir_fx" | "torch_mlir_torchscript" | "torch_mlir_compile" | "torch_mlir_opt"
    canonicalized: bool


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise CompilationError(f"Lowering command failed: {' '.join(cmd)}") from exc


def _mlir_to_text(mlir_module) -> str:
    # Different torch-mlir entrypoints return different wrapper types.
    # Prefer operation.get_asm if present, else str().
    op = getattr(mlir_module, "operation", None)
    if op is not None and hasattr(op, "get_asm"):
        return op.get_asm(large_elements_limit=10)
    return str(mlir_module)


def lower_torch_to_tosa_mlir(
    *,
    model: torch.nn.Module,
    example_inputs: Iterable[torch.Tensor],
    out_path: str | Path,
    canonicalize: bool = True,
) -> LoweringResult:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().to("cpu")
    ex_inputs = [t.detach().to("cpu") for t in example_inputs]

    try:
        import torch_mlir  # type: ignore
    except Exception as exc:
        raise CompilationError(
            "torch-mlir is required for Torch->TOSA lowering but is not importable."
        ) from exc
    
    from torch_mlir.fx import export_and_import as fx

    example = ex_inputs[0] if len(ex_inputs) == 1 else tuple(ex_inputs)

    try:
        mlir_module = fx(model, example, output_type="tosa")
    except Exception as exc:
        raise CompilationError(f"FX export_and_import(TOSA) failed: {exc}") from exc

    # Write MLIR text
    op = getattr(mlir_module, "operation", None)
    if op is not None and hasattr(op, "get_asm"):
        mlir_text = op.get_asm(large_elements_limit=10)
    else:
        mlir_text = str(mlir_module)

    out_path.write_text(mlir_text, encoding="utf-8")

    did_canon = False
    if canonicalize:
        out_path = _canonicalize_mlir(out_path)
        did_canon = True

    return LoweringResult(out_path, "torch_mlir_fx", did_canon)


def _canonicalize_mlir(path: Path) -> Path:
    mlir_opt = _which("mlir-opt")
    if not mlir_opt:
        return path

    out = path.with_suffix(".canon.mlir")
    _run([mlir_opt, str(path), "--cse", "--canonicalize", "-o", str(out)])
    return out
