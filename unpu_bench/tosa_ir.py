# unpu_bench/tosa_ir.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TensorSig:
    shape: List[int]          # unknown dims are -1
    dtype: str                # "i8", "i32", "f32", ...
    raw: str                  # original MLIR type text


@dataclass(frozen=True)
class TosaOpSig:
    op_name: str
    operand_names: List[str]
    result_names: List[str]
    operands: List[TensorSig]
    results: List[TensorSig]
    attrs: Dict[str, Any]
    location: Tuple[int, str]  # (line_no, full_line)


@dataclass(frozen=True)
class TosaModule:
    path: Path
    ops: List[TosaOpSig]


_TENSOR_RE = re.compile(r"tensor<([^>]+)>")
_SSA_RE = re.compile(r"%[a-zA-Z0-9_.$]+")
_OP_RE = re.compile(r"\b(tosa\.[a-zA-Z0-9_]+)\b")


def _parse_tensor_type(type_text: str) -> Optional[TensorSig]:
    m = _TENSOR_RE.search(type_text)
    if not m:
        return None
    body = m.group(1).strip()
    parts = body.split("x")
    if not parts:
        return None
    dtype = parts[-1].strip()
    dims = parts[:-1]
    shape: List[int] = []
    for d in dims:
        d = d.strip()
        if d == "?":
            shape.append(-1)
        else:
            try:
                shape.append(int(d))
            except ValueError:
                shape.append(-1)
    return TensorSig(shape=shape, dtype=dtype, raw=f"tensor<{body}>")


def _parse_attr_dict(line: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    if "{" not in line or "}" not in line:
        return attrs
    blob = line.split("{", 1)[1].rsplit("}", 1)[0]
    for item in blob.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        attrs[k.strip()] = v.strip()
    return attrs


def load_tosa_module(path: str | Path) -> TosaModule:
    path = Path(path)
    text = path.read_text(encoding="utf-8").splitlines()

    ops: List[TosaOpSig] = []

    for i, line in enumerate(text, start=1):
        if "tosa." not in line:
            continue

        m = _OP_RE.search(line)
        if not m:
            continue
        op_name = m.group(1)

        attrs = _parse_attr_dict(line)

        # Results: before '='
        result_names: List[str] = []
        if "=" in line:
            lhs = line.split("=", 1)[0]
            result_names = _SSA_RE.findall(lhs)

        # Operands: inside (...) if present, else after op for generic form
        operand_names: List[str] = []
        if "(" in line and ")" in line:
            inside = line.split("(", 1)[1].split(")", 1)[0]
            operand_names = _SSA_RE.findall(inside)
        else:
            # e.g. "tosa.add %a, %b : tensor<...>"
            after = line.split(op_name, 1)[1]
            operand_names = _SSA_RE.findall(after)

        operands: List[TensorSig] = []
        results: List[TensorSig] = []

        # Parse MLIR signature if present
        if ":" in line and "->" in line:
            sig = line.split(":", 1)[1]
            left, right = sig.split("->", 1)
            for tm in _TENSOR_RE.finditer(left):
                t = _parse_tensor_type(f"tensor<{tm.group(1)}>")
                if t:
                    operands.append(t)
            for tm in _TENSOR_RE.finditer(right):
                t = _parse_tensor_type(f"tensor<{tm.group(1)}>")
                if t:
                    results.append(t)
        elif ":" in line and "->" not in line:
            t = _parse_tensor_type(line.split(":", 1)[1])
            if t:
                results.append(t)

        ops.append(
            TosaOpSig(
                op_name=op_name,
                operand_names=operand_names,
                result_names=result_names,
                operands=operands,
                results=results,
                attrs=attrs,
                location=(i, line.rstrip("\n")),
            )
        )

    return TosaModule(path=path, ops=ops)
