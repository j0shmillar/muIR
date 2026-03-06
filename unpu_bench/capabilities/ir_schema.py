from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..muir import Op, Program


@dataclass(frozen=True)
class IRTensorConstraint:
    dtypes: List[str]
    rank: Optional[int] = None
    layouts: Optional[List[str]] = None


@dataclass(frozen=True)
class IRAttrConstraint:
    min: Optional[float] = None
    max: Optional[float] = None
    one_of: Optional[List[Any]] = None
    required: bool = False
    length: Optional[int] = None


@dataclass(frozen=True)
class IROpConstraint:
    op: str
    inputs: List[IRTensorConstraint]
    outputs: List[IRTensorConstraint]
    attrs: Dict[str, IRAttrConstraint]
    quantized_only: bool = False


@dataclass(frozen=True)
class IRCapabilityDB:
    schema_version: int
    backend: str
    fallback_backend: str
    ops: Dict[str, List[IROpConstraint]]
    preferred_layouts: List[str]
    quantization_modes: List[str]
    forbidden_partition_boundaries: List[str]
    notes: str | None


def load_ir_capabilities(path: str | Path, *, backend: str) -> IRCapabilityDB:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    schema_version = int(data.get("schema_version", 0) or 0)
    if schema_version != 1:
        raise ValueError(
            f"Unsupported IR capability schema_version={schema_version} in {path}; expected 1."
        )

    fallback = str(data.get("fallback_backend", "cpu"))
    preferred_layouts = [str(x) for x in data.get("preferred_layouts", [])]
    quantization_modes = [str(x) for x in data.get("quantization_modes", [])]
    forbidden_partition_boundaries = [
        str(x) for x in data.get("forbidden_partition_boundaries", [])
    ]
    notes = data.get("notes")

    ops: Dict[str, List[IROpConstraint]] = {}
    for item in data.get("ops", []):
        op = str(item["op"])
        ins = [IRTensorConstraint(**x) for x in item.get("inputs", [])]
        outs = [IRTensorConstraint(**x) for x in item.get("outputs", [])]

        attrs: Dict[str, IRAttrConstraint] = {}
        for k, v in (item.get("attrs") or {}).items():
            attrs[str(k)] = IRAttrConstraint(**v)

        ops.setdefault(op, []).append(
            IROpConstraint(
                op=op,
                inputs=ins,
                outputs=outs,
                attrs=attrs,
                quantized_only=bool(item.get("quantized_only", False)),
            )
        )

    return IRCapabilityDB(
        schema_version=schema_version,
        backend=backend,
        fallback_backend=fallback,
        ops=ops,
        preferred_layouts=preferred_layouts,
        quantization_modes=quantization_modes,
        forbidden_partition_boundaries=forbidden_partition_boundaries,
        notes=str(notes) if notes is not None else None,
    )


def _dtype_ok(dtype: str, c: IRTensorConstraint) -> bool:
    return (not c.dtypes) or (dtype in c.dtypes)


def _rank_ok(rank: int, c: IRTensorConstraint) -> bool:
    return c.rank is None or rank == c.rank


def _layout_ok(layout: Optional[str], c: IRTensorConstraint) -> bool:
    if not c.layouts:
        return True
    if not layout:
        return False
    return layout in c.layouts


def _parse_scalar(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        # For vector attrs we validate per element elsewhere.
        return None
    return None


def check_ir_op_legality(
    op: Op, program: Program, cap: IRCapabilityDB
) -> Tuple[bool, List[str]]:
    variants = cap.ops.get(op.kind)
    if not variants:
        return False, [f"unsupported op: {op.kind}"]

    variant_reasons: List[List[str]] = []

    for c in variants:
        reasons: List[str] = []

        if c.inputs and len(op.inputs) != len(c.inputs):
            reasons.append(
                f"operand arity mismatch: got {len(op.inputs)}, want {len(c.inputs)}"
            )
        else:
            for i, (tid, tc) in enumerate(zip(op.inputs, c.inputs)):
                t = program.graph.tensors.get(tid)
                if t is None:
                    reasons.append(f"operand[{i}] missing tensor: {tid}")
                    continue
                if not _dtype_ok(t.type.dtype, tc):
                    reasons.append(
                        f"operand[{i}] dtype {t.type.dtype} not in {tc.dtypes}"
                    )
                if not _rank_ok(len(t.type.shape), tc):
                    reasons.append(
                        f"operand[{i}] rank {len(t.type.shape)} != {tc.rank}"
                    )
                if not _layout_ok(t.type.layout, tc):
                    reasons.append(
                        f"operand[{i}] layout {t.type.layout} not in {tc.layouts}"
                    )

        if c.outputs and len(op.outputs) != len(c.outputs):
            reasons.append(
                f"result arity mismatch: got {len(op.outputs)}, want {len(c.outputs)}"
            )
        else:
            for i, (tid, tc) in enumerate(zip(op.outputs, c.outputs)):
                t = program.graph.tensors.get(tid)
                if t is None:
                    reasons.append(f"result[{i}] missing tensor: {tid}")
                    continue
                if not _dtype_ok(t.type.dtype, tc):
                    reasons.append(
                        f"result[{i}] dtype {t.type.dtype} not in {tc.dtypes}"
                    )
                if not _rank_ok(len(t.type.shape), tc):
                    reasons.append(f"result[{i}] rank {len(t.type.shape)} != {tc.rank}")
                if not _layout_ok(t.type.layout, tc):
                    reasons.append(
                        f"result[{i}] layout {t.type.layout} not in {tc.layouts}"
                    )

        if c.quantized_only:
            q_ok = True
            for tid in op.inputs + op.outputs:
                t = program.graph.tensors.get(tid)
                if t is None:
                    continue
                if t.quant is None and t.type.dtype not in {"i8", "u8"}:
                    q_ok = False
                    break
            if not q_ok:
                reasons.append(
                    "op requires quantized tensors but quant metadata/dtype is missing"
                )

        for key, constraint in c.attrs.items():
            if key not in op.attrs:
                if constraint.required:
                    reasons.append(f"missing required attr: {key}")
                continue

            raw = op.attrs[key]
            if isinstance(raw, list):
                if constraint.length is not None and len(raw) != constraint.length:
                    reasons.append(
                        f"attr {key} length {len(raw)} != {constraint.length}"
                    )
                values = [v for v in raw if isinstance(v, (int, float))]
            else:
                v = _parse_scalar(raw)
                values = [] if v is None else [v]

            if constraint.one_of is not None:
                seq = raw if isinstance(raw, list) else [raw]
                for v in seq:
                    if v not in constraint.one_of:
                        reasons.append(
                            f"attr {key} value {v} not in {constraint.one_of}"
                        )

            for v in values:
                if constraint.min is not None and v < constraint.min:
                    reasons.append(f"attr {key} value {v} < min {constraint.min}")
                if constraint.max is not None and v > constraint.max:
                    reasons.append(f"attr {key} value {v} > max {constraint.max}")

        if not reasons:
            return True, []
        variant_reasons.append(reasons)

    best = (
        min(variant_reasons, key=len)
        if variant_reasons
        else [f"no legal variant for {op.kind}"]
    )
    return False, best
