# unpu_bench/repro.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .tosa_ir import TosaOpSig


def write_repro_mlir(op: TosaOpSig, out_dir: str | Path, *, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build function args from operand tensor types (raw).
    args = []
    for i, t in enumerate(op.operands):
        args.append(f"%arg{i}: {t.raw}")

    # Return types from results.
    rets = ", ".join([t.raw for t in op.results]) if op.results else "none"

    # Keep the original line as the body (best-effort; canonicalization should have stabilized it).
    body_line = op.location[1].strip()
    # Ensure indentation and avoid location noise.
    body_line = body_line.replace(" loc(", " ")  # crude

    text = f"""module {{
  func.func @main({", ".join(args)}) -> ({rets}) {{
    // Original op:
    {body_line}
    return
  }}
}}
"""

    p = out_dir / f"{name}.mlir"
    p.write_text(text, encoding="utf-8")
    return p
