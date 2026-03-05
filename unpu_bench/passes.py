from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .errors import CompilationError
from .muir import Program, Partition  # you can keep Program as your “container”
from .capabilities.ir_schema import (
    IRCapabilityDB,
    check_ir_op_legality,
    load_ir_capabilities,
)
from .tosa_ir import TosaModule, TosaOpSig
from .capabilities.schema import CapabilityDB, load_capabilities, check_op_legality
from .repro import write_repro_mlir


# ---------- 0) IR Canonicalization ----------

def _infer_default_layout(shape: List[int]) -> str | None:
    if len(shape) == 4:
        return "NCHW"
    if len(shape) == 3:
        return "CHW"
    if len(shape) == 2:
        return "NC"
    if len(shape) == 1:
        return "C"
    return None


def _as_int_list(value: Any, *, expected_len: int | None = None, default: List[int] | None = None) -> List[int]:
    if isinstance(value, int):
        out = [int(value)]
    elif isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
    elif value is None:
        out = list(default or [])
    else:
        out = list(default or [])

    if expected_len is not None:
        if len(out) == 1 and expected_len > 1:
            out = out * expected_len
        elif len(out) < expected_len:
            out = out + [out[-1] if out else 0] * (expected_len - len(out))
        elif len(out) > expected_len:
            out = out[:expected_len]
    return out


def _broadcastable(a: List[int], b: List[int]) -> bool:
    ra = list(reversed(a))
    rb = list(reversed(b))
    n = max(len(ra), len(rb))
    for i in range(n):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da != db and da != 1 and db != 1:
            return False
    return True


def _canonicalize_op(op_id: str, program: Program) -> None:
    g = program.graph
    op = g.ops[op_id]
    attrs = dict(op.attrs or {})

    # Normalize common attribute aliases.
    if "stride" in attrs and "strides" not in attrs:
        attrs["strides"] = attrs.pop("stride")
    if "dilation" in attrs and "dilations" not in attrs:
        attrs["dilations"] = attrs.pop("dilation")
    if "padding" in attrs and "pads" not in attrs:
        attrs["pads"] = attrs.pop("padding")
    if "pad" in attrs and "pads" not in attrs:
        attrs["pads"] = attrs.pop("pad")
    if "kernel_size" in attrs and "kernel_shape" not in attrs:
        attrs["kernel_shape"] = attrs.pop("kernel_size")
    if "axes" in attrs and "axis" not in attrs:
        attrs["axis"] = attrs.pop("axes")

    if op.kind in {"Conv", "DepthwiseConv2d"}:
        attrs["strides"] = _as_int_list(attrs.get("strides"), expected_len=2, default=[1, 1])
        attrs["dilations"] = _as_int_list(attrs.get("dilations"), expected_len=2, default=[1, 1])
        pads = _as_int_list(attrs.get("pads"), default=[0, 0, 0, 0])
        if len(pads) == 2:
            pads = [pads[0], pads[1], pads[0], pads[1]]
        attrs["pads"] = _as_int_list(pads, expected_len=4, default=[0, 0, 0, 0])
        attrs["kernel_shape"] = _as_int_list(attrs.get("kernel_shape"), expected_len=2, default=[1, 1])
        attrs["group"] = int(attrs.get("group", 1))

        # Infer depthwise from group/input channels where available.
        if op.kind == "Conv" and len(op.inputs) >= 2:
            data_t = g.tensors.get(op.inputs[0])
            if data_t and len(data_t.type.shape) >= 2:
                in_channels = data_t.type.shape[1]
                if in_channels > 1 and int(attrs["group"]) == int(in_channels):
                    op.kind = "DepthwiseConv2d"

    if op.kind in {"MaxPool", "AveragePool"}:
        attrs["kernel_shape"] = _as_int_list(attrs.get("kernel_shape"), expected_len=2, default=[1, 1])
        attrs["strides"] = _as_int_list(attrs.get("strides"), expected_len=2, default=[1, 1])
        pads = _as_int_list(attrs.get("pads"), default=[0, 0, 0, 0])
        if len(pads) == 2:
            pads = [pads[0], pads[1], pads[0], pads[1]]
        attrs["pads"] = _as_int_list(pads, expected_len=4, default=[0, 0, 0, 0])

    if op.kind == "Gemm":
        attrs["alpha"] = float(attrs.get("alpha", 1.0))
        attrs["beta"] = float(attrs.get("beta", 1.0))
        attrs["transA"] = int(attrs.get("transA", 0))
        attrs["transB"] = int(attrs.get("transB", 0))

    if op.kind == "Flatten":
        attrs["axis"] = int(attrs.get("axis", 1))

    if op.kind == "Transpose":
        attrs["perm"] = _as_int_list(attrs.get("perm"), default=[])

    if op.kind in {"Add", "Sub", "Mul", "Div"} and len(op.inputs) == 2:
        lhs = g.tensors.get(op.inputs[0])
        rhs = g.tensors.get(op.inputs[1])
        if lhs and rhs and lhs.type.shape and rhs.type.shape:
            attrs["broadcast"] = lhs.type.shape != rhs.type.shape
            attrs["broadcast_semantics"] = "numpy" if _broadcastable(lhs.type.shape, rhs.type.shape) else "none"

    op.attrs = attrs


def run_ir_canonicalization(program: Program) -> None:
    """Normalize IR to canonical op/tensor semantics before validation and legality."""
    g = program.graph
    for t in g.tensors.values():
        if t.type.layout is None:
            t.type.layout = _infer_default_layout(t.type.shape)
    for op_id in g.op_order:
        _canonicalize_op(op_id, program)
    program.metadata["ir_schema_version"] = 1
    program.metadata["ir_canonicalized"] = True


# ---------- 0) IR Validation ----------

def run_ir_validation(program: Program) -> None:
    """Backend-agnostic structural validation for unified IR."""
    g = program.graph

    # op_order integrity
    seen: set[str] = set()
    for op_id in g.op_order:
        if op_id in seen:
            raise CompilationError(f"IR validation failed: duplicate op id in op_order: {op_id}")
        seen.add(op_id)
        if op_id not in g.ops:
            raise CompilationError(f"IR validation failed: op_order references missing op: {op_id}")

    for op_id in g.ops:
        if op_id not in seen:
            raise CompilationError(f"IR validation failed: op exists but is absent from op_order: {op_id}")

    # Tensor existence for graph interfaces
    for tid in g.inputs + g.outputs + g.initializers:
        if tid not in g.tensors:
            raise CompilationError(f"IR validation failed: graph references missing tensor '{tid}'")

    # Op IO consistency and producer/consumer links
    for op_id in g.op_order:
        op = g.ops[op_id]
        if not op.outputs:
            raise CompilationError(f"IR validation failed: op '{op_id}' has no outputs")

        for tid in op.inputs:
            if tid not in g.tensors:
                raise CompilationError(f"IR validation failed: op '{op_id}' input tensor missing: {tid}")
            t = g.tensors[tid]
            if op_id not in t.consumers:
                raise CompilationError(
                    f"IR validation failed: tensor '{tid}' missing consumer link for op '{op_id}'"
                )

        for tid in op.outputs:
            if tid not in g.tensors:
                raise CompilationError(f"IR validation failed: op '{op_id}' output tensor missing: {tid}")
            t = g.tensors[tid]
            if t.producer != op_id:
                raise CompilationError(
                    f"IR validation failed: tensor '{tid}' producer='{t.producer}' expected '{op_id}'"
                )

    # Graph inputs should not have producer; initializers should be constant.
    for tid in g.inputs:
        t = g.tensors[tid]
        if t.producer is not None:
            raise CompilationError(f"IR validation failed: input tensor '{tid}' has producer '{t.producer}'")
    for tid in g.initializers:
        t = g.tensors[tid]
        if not t.is_constant:
            raise CompilationError(f"IR validation failed: initializer '{tid}' must be marked constant")
    # Canonical tensor semantics: rank-4 tensors should carry a layout.
    for tid, t in g.tensors.items():
        if len(t.type.shape) == 4 and not t.type.layout:
            raise CompilationError(f"IR validation failed: rank-4 tensor '{tid}' missing layout")


# ---------- 1) Legality over unified IR ----------
def run_legality_check(
    program: Program,
    *,
    backend: str = "ai8x",
    caps_path: str | Path | None = None,
) -> None:
    """Annotate IR ops with legal_backends using backend capability schema."""

    if caps_path is None:
        caps_path = (
            Path(__file__).resolve().parents[1]
            / "unpu_bench"
            / "capabilities"
            / f"ir_{backend}.yaml"
        )
    try:
        cap: IRCapabilityDB = load_ir_capabilities(caps_path, backend=backend)
    except ValueError as exc:
        raise CompilationError(str(exc)) from exc
    fallback = cap.fallback_backend or "cpu"

    for op_id in program.graph.op_order:
        op = program.graph.ops[op_id]
        ok, _reasons = check_ir_op_legality(op, program, cap)
        legal = [fallback]
        if ok:
            legal.append(backend)

        op.legal_backends = legal
        op.preferred_backend = backend if ok else fallback


# ---------- 2) Partitioning ----------

def run_partitioning(
    program: Program,
    *,
    backend: str = "ai8x",
    fallback_backend: str = "cpu",
) -> None:
    """Partition the program into up to three segments:
       - optional CPU prefix
       - required <backend> core (if any backend-legal ops exist)
       - optional CPU suffix

    Rules:
      - Any op whose preferred_backend == backend must be in the backend core.
      - Any op whose preferred_backend != backend must be in prefix or suffix.
      - If a fallback op is found *between* two backend ops, we error out.
    """
    ops_order = program.graph.op_order or list(program.graph.ops.keys())
    if not ops_order:
        program.partitions = []
        return

    # Label each op as backend or fallback based on preferred_backend
    kinds: List[str] = []
    for op_id in ops_order:
        op = program.graph.ops[op_id]
        op_backend = op.preferred_backend or fallback_backend
        kinds.append(backend if op_backend == backend else fallback_backend)

    # Find the first and last accelerator ops
    try:
        first_accel = kinds.index(backend)
        last_accel = len(kinds) - 1 - kinds[::-1].index(backend)
    except ValueError:
        # No accelerator ops at all: everything on fallback backend
        program.partitions = [
            Partition(id=f"{fallback_backend}_full", backend=fallback_backend, op_ids=ops_order),
        ]
        return

    # Ensure there are no fallback ops in the core [first_accel, last_accel]
    for idx in range(first_accel, last_accel + 1):
        if kinds[idx] != backend:
            bad_op_id = ops_order[idx]
            bad_op = program.graph.ops[bad_op_id]
            raise CompilationError(
                f"Op '{bad_op_id}' ({bad_op.kind}) is not legal for backend '{backend}' but "
                "appears in the middle of the NPU segment. "
                f"Only {fallback_backend} prefix/suffix around a single {backend} core are supported."
            )

    partitions: List[Partition] = []

    # fallback prefix
    if first_accel > 0:
        prefix_ids = ops_order[:first_accel]
        partitions.append(
            Partition(
                id=f"{fallback_backend}_prefix",
                backend=fallback_backend,
                op_ids=prefix_ids,
            )
        )

    # backend core
    core_ids = ops_order[first_accel : last_accel + 1]
    partitions.append(
        Partition(
            id=f"{backend}_core",
            backend=backend,
            op_ids=core_ids,
        )
    )

    # fallback suffix
    if last_accel < len(ops_order) - 1:
        suffix_ids = ops_order[last_accel + 1 :]
        partitions.append(
            Partition(
                id=f"{fallback_backend}_suffix",
                backend=fallback_backend,
                op_ids=suffix_ids,
            )
        )

    program.partitions = partitions

@dataclass(frozen=True)
class LegalityRecord:
    index: int
    op: str
    legal: bool
    reasons: List[str]
    line_no: int
    line: str
    repro: str | None


def run_tosa_legality_and_partitioning(
    *,
    program: Program,
    tosa: TosaModule,
    caps_path: str | Path,
    backend: str,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap: CapabilityDB = load_capabilities(caps_path, backend=backend)

    records: List[LegalityRecord] = []
    repro_dir = out_dir / "repros"

    for idx, op in enumerate(tosa.ops):
        ok, reasons = check_op_legality(op, cap)
        repro_path = None
        if not ok:
            repro_path = str(write_repro_mlir(op, repro_dir, name=f"op{idx}_{op.op_name.replace('.', '_')}"))

        records.append(LegalityRecord(
            index=idx,
            op=op.op_name,
            legal=ok,
            reasons=reasons,
            line_no=op.location[0],
            line=op.location[1],
            repro=repro_path,
        ))

    # Write legality.json / legality.txt
    legality_json = out_dir / "legality.json"
    legality_txt = out_dir / "legality.txt"

    legality_json.write_text(
        json.dumps([r.__dict__ for r in records], indent=2, sort_keys=False),
        encoding="utf-8",
    )

    lines = []
    for r in records:
        if r.legal:
            continue
        lines.append(f"[{r.index}] {r.op} at {r.line_no}:")
        for reason in r.reasons:
            lines.append(f"  - {reason}")
        if r.repro:
            lines.append(f"  repro: {r.repro}")
        lines.append("")
    legality_txt.write_text("\n".join(lines), encoding="utf-8")

    # Partition: single NPU core segment + CPU prefix/suffix (same rule as before),
    # but now using TOSA legality flags.
    kinds = ["npu" if r.legal else "cpu" for r in records]

    try:
        first_npu = kinds.index("npu")
        last_npu = len(kinds) - 1 - kinds[::-1].index("npu")
    except ValueError:
        program.partitions = [Partition(id="cpu_full", backend="cpu", op_ids=[])]
        program.metadata["tosa_partitions"] = {"cpu_full": [0, len(records)]}
        return

    for i in range(first_npu, last_npu + 1):
        if kinds[i] != "npu":
            raise CompilationError(
                f"TOSA op[{i}] is illegal for {backend} but appears inside the core NPU window.\n"
                f"See {legality_txt} and repros in {repro_dir}."
            )

    parts: List[Partition] = []
    # op_ids is left empty because your Program currently enumerates ONNX-import ops.
    # The SoT partitioning is stored in metadata ranges; you can migrate fully later.
    if first_npu > 0:
        parts.append(Partition(id="cpu_prefix", backend="cpu", op_ids=[]))
    parts.append(Partition(id=f"{backend}_core", backend=backend, op_ids=[]))
    if last_npu < len(records) - 1:
        parts.append(Partition(id="cpu_suffix", backend="cpu", op_ids=[]))

    program.partitions = parts
    program.metadata["tosa_partitions"] = {
        "cpu_prefix": [0, first_npu] if first_npu > 0 else None,
        f"{backend}_core": [first_npu, last_npu + 1],
        "cpu_suffix": [last_npu + 1, len(records)] if last_npu < len(records) - 1 else None,
    }
    program.metadata["tosa_legality"] = str(legality_json)
    program.metadata["tosa_repros_dir"] = str(repro_dir)
