# unpu_bench/capabilities/schema.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..tosa_ir import TosaOpSig, TensorSig


@dataclass(frozen=True)
class TensorConstraint:
    dtypes: List[str]                 # e.g. ["i8"]
    rank: Optional[int] = None        # e.g. 4
    layout: Optional[str] = None      # e.g. "NCHW", "NHWC", "OIHW"


@dataclass(frozen=True)
class AttrConstraint:
    """
    Generic numeric range constraint for things like stride/dilation/kernel.
    For multi-dim attrs, we apply to each element.
    """
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass(frozen=True)
class OpConstraint:
    op: str
    inputs: List[TensorConstraint]
    outputs: List[TensorConstraint]
    attrs: Dict[str, AttrConstraint]          # keys are schema-defined attribute names
    accumulator_dtype: Optional[str] = None   # e.g. "i32"
    accumulator_max_abs: Optional[int] = None # e.g. 2**31-1


@dataclass(frozen=True)
class CapabilityDB:
    backend: str
    ops: Dict[str, List[OpConstraint]]


def load_capabilities(path: str | Path, *, backend: str) -> CapabilityDB:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ops: Dict[str, List[OpConstraint]] = {}

    for item in data.get("ops", []):
        op = item["op"]
        ins = [TensorConstraint(**x) for x in item.get("inputs", [])]
        outs = [TensorConstraint(**x) for x in item.get("outputs", [])]

        attrs: Dict[str, AttrConstraint] = {}
        for k, v in (item.get("attrs") or {}).items():
            attrs[k] = AttrConstraint(**v)

        ops.setdefault(op, []).append(
            OpConstraint(
            op=op,
            inputs=ins,
            outputs=outs,
            attrs=attrs,
            accumulator_dtype=(item.get("accumulator") or {}).get("dtype"),
            accumulator_max_abs=(item.get("accumulator") or {}).get("max_abs"),
            )
        )

    return CapabilityDB(backend=backend, ops=ops)


def _dtype_ok(t: TensorSig, c: TensorConstraint) -> bool:
    return (not c.dtypes) or (t.dtype in c.dtypes)


def _rank_ok(t: TensorSig, c: TensorConstraint) -> bool:
    if c.rank is None:
        return True
    return len(t.shape) == c.rank


def check_op_legality(op: TosaOpSig, cap: CapabilityDB) -> Tuple[bool, List[str]]:
    """
    Returns (is_legal, reasons[]).
    Reasons are stable strings you can dump to legality.json/txt and repro metadata.
    """
    variants = cap.ops.get(op.op_name)
    if variants is None:
        return False, [f"unsupported op: {op.op_name}"]

    variant_reasons: List[List[str]] = []
    for c in variants:
        reasons: List[str] = []

        # typed inputs
        if c.inputs and len(op.operands) != len(c.inputs):
            reasons.append(f"operand arity mismatch: got {len(op.operands)}, want {len(c.inputs)}")
        else:
            for i, (ts, tc) in enumerate(zip(op.operands, c.inputs)):
                if not _dtype_ok(ts, tc):
                    reasons.append(f"operand[{i}] dtype {ts.dtype} not in {tc.dtypes}")
                if not _rank_ok(ts, tc):
                    reasons.append(f"operand[{i}] rank {len(ts.shape)} != {tc.rank}")

        # typed outputs
        if c.outputs and len(op.results) != len(c.outputs):
            reasons.append(f"result arity mismatch: got {len(op.results)}, want {len(c.outputs)}")
        else:
            for i, (ts, tc) in enumerate(zip(op.results, c.outputs)):
                if not _dtype_ok(ts, tc):
                    reasons.append(f"result[{i}] dtype {ts.dtype} not in {tc.dtypes}")
                if not _rank_ok(ts, tc):
                    reasons.append(f"result[{i}] rank {len(ts.shape)} != {tc.rank}")

        # attribute constraints (best-effort; attrs are raw strings right now)
        for k, rng in c.attrs.items():
            if k not in op.attrs:
                continue
            raw = str(op.attrs[k])

            vals: List[int] = []
            for tok in raw.replace("[", " ").replace("]", " ").replace(",", " ").split():
                try:
                    vals.append(int(tok))
                except ValueError:
                    pass
            if not vals:
                continue

            for v in vals:
                if rng.min is not None and v < rng.min:
                    reasons.append(f"attr {k} value {v} < min {rng.min}")
                if rng.max is not None and v > rng.max:
                    reasons.append(f"attr {k} value {v} > max {rng.max}")

        if not reasons:
            return True, []
        variant_reasons.append(reasons)

    best = min(variant_reasons, key=len) if variant_reasons else [f"no legal variant for {op.op_name}"]
    return False, best
