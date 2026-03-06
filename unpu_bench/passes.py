from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .errors import CompilationError
from .muir import Program, Partition
from .capabilities.ir_schema import (
    IRCapabilityDB,
    check_ir_op_legality,
    load_ir_capabilities,
)


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


def _as_int_list(
    value: Any, *, expected_len: int | None = None, default: List[int] | None = None
) -> List[int]:
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
        attrs["strides"] = _as_int_list(
            attrs.get("strides"), expected_len=2, default=[1, 1]
        )
        attrs["dilations"] = _as_int_list(
            attrs.get("dilations"), expected_len=2, default=[1, 1]
        )
        pads = _as_int_list(attrs.get("pads"), default=[0, 0, 0, 0])
        if len(pads) == 2:
            pads = [pads[0], pads[1], pads[0], pads[1]]
        attrs["pads"] = _as_int_list(pads, expected_len=4, default=[0, 0, 0, 0])
        attrs["kernel_shape"] = _as_int_list(
            attrs.get("kernel_shape"), expected_len=2, default=[1, 1]
        )
        attrs["group"] = int(attrs.get("group", 1))

        # Infer depthwise from group/input channels where available.
        if op.kind == "Conv" and len(op.inputs) >= 2:
            data_t = g.tensors.get(op.inputs[0])
            if data_t and len(data_t.type.shape) >= 2:
                in_channels = data_t.type.shape[1]
                if in_channels > 1 and int(attrs["group"]) == int(in_channels):
                    op.kind = "DepthwiseConv2d"

    if op.kind in {"MaxPool", "AveragePool"}:
        attrs["kernel_shape"] = _as_int_list(
            attrs.get("kernel_shape"), expected_len=2, default=[1, 1]
        )
        attrs["strides"] = _as_int_list(
            attrs.get("strides"), expected_len=2, default=[1, 1]
        )
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
            attrs["broadcast_semantics"] = (
                "numpy" if _broadcastable(lhs.type.shape, rhs.type.shape) else "none"
            )

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


def _replace_tensor_uses(program: Program, *, old_tid: str, new_tid: str) -> None:
    g = program.graph
    for op_id in g.op_order:
        op = g.ops[op_id]
        op.inputs = [new_tid if x == old_tid else x for x in op.inputs]
    g.outputs = [new_tid if x == old_tid else x for x in g.outputs]
    if old_tid in g.tensors:
        old_t = g.tensors[old_tid]
        for c in old_t.consumers:
            if c in g.ops and c not in g.tensors[new_tid].consumers:
                g.tensors[new_tid].consumers.append(c)
        if (
            old_t.producer
            and old_t.producer in g.ops
            and g.tensors[new_tid].producer is None
        ):
            g.tensors[new_tid].producer = old_t.producer
        del g.tensors[old_tid]


def run_ir_rewrite_passes(program: Program) -> dict[str, int]:
    """Semantics-preserving IR rewrites that avoid backend-specific assumptions."""
    g = program.graph
    rewrites = {"identity_removed": 0, "relu_chain_collapsed": 0}

    # 1) Remove Identity ops by bypassing input->output.
    for op_id in list(g.op_order):
        op = g.ops.get(op_id)
        if (
            op is None
            or op.kind != "Identity"
            or len(op.inputs) != 1
            or len(op.outputs) != 1
        ):
            continue
        src, dst = op.inputs[0], op.outputs[0]
        if src not in g.tensors:
            continue
        _replace_tensor_uses(program, old_tid=dst, new_tid=src)
        if op_id in g.op_order:
            g.op_order.remove(op_id)
        g.ops.pop(op_id, None)
        if op_id in g.tensors[src].consumers:
            g.tensors[src].consumers.remove(op_id)
        rewrites["identity_removed"] += 1

    # 2) Collapse Relu->Relu chains: relu(relu(x)) == relu(x)
    for op_id in list(g.op_order):
        op = g.ops.get(op_id)
        if op is None or op.kind != "Relu" or len(op.outputs) != 1:
            continue
        mid = op.outputs[0]
        t_mid = g.tensors.get(mid)
        if t_mid is None or len(t_mid.consumers) != 1:
            continue
        next_id = t_mid.consumers[0]
        next_op = g.ops.get(next_id)
        if (
            next_op is None
            or next_op.kind != "Relu"
            or len(next_op.inputs) != 1
            or len(next_op.outputs) != 1
        ):
            continue
        if next_op.inputs[0] != mid:
            continue
        out2 = next_op.outputs[0]
        _replace_tensor_uses(program, old_tid=out2, new_tid=mid)
        if next_id in g.op_order:
            g.op_order.remove(next_id)
        g.ops.pop(next_id, None)
        if next_id in t_mid.consumers:
            t_mid.consumers.remove(next_id)
        rewrites["relu_chain_collapsed"] += 1

    program.metadata["ir_rewrites"] = rewrites
    return rewrites


# ---------- 0) IR Validation ----------


def run_ir_validation(program: Program) -> None:
    """Backend-agnostic structural validation for unified IR."""
    g = program.graph

    # op_order integrity
    seen: set[str] = set()
    for op_id in g.op_order:
        if op_id in seen:
            raise CompilationError(
                f"IR validation failed: duplicate op id in op_order: {op_id}"
            )
        seen.add(op_id)
        if op_id not in g.ops:
            raise CompilationError(
                f"IR validation failed: op_order references missing op: {op_id}"
            )

    for op_id in g.ops:
        if op_id not in seen:
            raise CompilationError(
                f"IR validation failed: op exists but is absent from op_order: {op_id}"
            )

    # Tensor existence for graph interfaces
    for tid in g.inputs + g.outputs + g.initializers:
        if tid not in g.tensors:
            raise CompilationError(
                f"IR validation failed: graph references missing tensor '{tid}'"
            )

    # Op IO consistency and producer/consumer links
    for op_id in g.op_order:
        op = g.ops[op_id]
        if not op.outputs:
            raise CompilationError(f"IR validation failed: op '{op_id}' has no outputs")

        for tid in op.inputs:
            if tid not in g.tensors:
                raise CompilationError(
                    f"IR validation failed: op '{op_id}' input tensor missing: {tid}"
                )
            t = g.tensors[tid]
            if op_id not in t.consumers:
                raise CompilationError(
                    f"IR validation failed: tensor '{tid}' missing consumer link for op '{op_id}'"
                )

        for tid in op.outputs:
            if tid not in g.tensors:
                raise CompilationError(
                    f"IR validation failed: op '{op_id}' output tensor missing: {tid}"
                )
            t = g.tensors[tid]
            if t.producer != op_id:
                raise CompilationError(
                    f"IR validation failed: tensor '{tid}' producer='{t.producer}' expected '{op_id}'"
                )

    # Graph inputs should not have producer; initializers should be constant.
    for tid in g.inputs:
        t = g.tensors[tid]
        if t.producer is not None:
            raise CompilationError(
                f"IR validation failed: input tensor '{tid}' has producer '{t.producer}'"
            )
    for tid in g.initializers:
        t = g.tensors[tid]
        if not t.is_constant:
            raise CompilationError(
                f"IR validation failed: initializer '{tid}' must be marked constant"
            )
    # Canonical tensor semantics: rank-4 tensors should carry a layout.
    for tid, t in g.tensors.items():
        if len(t.type.shape) == 4 and not t.type.layout:
            raise CompilationError(
                f"IR validation failed: rank-4 tensor '{tid}' missing layout"
            )


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
            Partition(
                id=f"{fallback_backend}_full",
                backend=fallback_backend,
                op_ids=ops_order,
            ),
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

    _validate_prefix_core_suffix_topology(
        partitions,
        backend=backend,
        fallback_backend=fallback_backend,
    )
    program.partitions = partitions


def _validate_prefix_core_suffix_topology(
    partitions: List[Partition],
    *,
    backend: str,
    fallback_backend: str,
) -> None:
    """Enforce the only supported offload shape:
    fallback prefix -> optional backend core -> fallback suffix.
    """
    if not partitions:
        return
    states = [p.backend for p in partitions]
    i = 0
    n = len(states)

    while i < n and states[i] == fallback_backend:
        i += 1
    while i < n and states[i] == backend:
        i += 1
    while i < n and states[i] == fallback_backend:
        i += 1

    if i != n or sum(1 for p in partitions if p.backend == backend) > 1:
        raise CompilationError(
            "Invalid partition topology detected. Only fallback prefix/suffix around "
            f"a single contiguous {backend} core are supported."
        )


def compute_partition_metrics(
    program: Program,
    *,
    backend: str,
    fallback_backend: str = "cpu",
) -> Dict[str, Any]:
    """Compute backend-agnostic partition quality metrics."""
    parts = program.partitions or []
    g = program.graph

    op_to_part: Dict[str, int] = {}
    for idx, p in enumerate(parts):
        for op_id in p.op_ids:
            op_to_part[op_id] = idx

    boundary_tensors: set[str] = set()
    for tid, t in g.tensors.items():
        prod = t.producer
        if not prod or prod not in op_to_part:
            continue
        src_idx = op_to_part[prod]
        for c in t.consumers:
            dst_idx = op_to_part.get(c)
            if dst_idx is not None and dst_idx != src_idx:
                boundary_tensors.add(tid)
                break

    backend_parts = [p for p in parts if p.backend == backend]
    backend_ops = sum(len(p.op_ids) for p in backend_parts)
    fallback_ops = sum(len(p.op_ids) for p in parts if p.backend == fallback_backend)
    total_ops = len(g.op_order)
    partition_count = len(parts)
    cut_count = max(0, partition_count - 1)
    topology_valid = True
    try:
        _validate_prefix_core_suffix_topology(
            parts,
            backend=backend,
            fallback_backend=fallback_backend,
        )
    except CompilationError:
        topology_valid = False

    return {
        "heuristic": "single_contiguous_core_with_cpu_prefix_suffix",
        "offload_topology": "prefix_core_suffix",
        "topology_valid": topology_valid,
        "target_backend": backend,
        "fallback_backend": fallback_backend,
        "partition_count": partition_count,
        "core_partition_count": len(backend_parts),
        "cut_count": cut_count,
        "estimated_layout_transitions": cut_count,  # proxy: one transition per partition boundary
        "boundary_tensor_count": len(boundary_tensors),
        "ops_total": total_ops,
        "ops_on_target_backend": backend_ops,
        "ops_on_fallback_backend": fallback_ops,
        "partitions": [
            {"id": p.id, "backend": p.backend, "op_count": len(p.op_ids)} for p in parts
        ],
        # Lightweight cost proxy: fewer cuts/boundaries/fallback ops is better.
        "cost_proxy": (cut_count * 100) + (len(boundary_tensors) * 10) + fallback_ops,
    }


def run_quantization_contract_validation(
    program: Program,
    *,
    backend: str,
    bit_width: int,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and summarize quantization metadata coverage."""
    g = program.graph
    tensors = list(g.tensors.values())
    quantized_tensors = [t for t in tensors if t.quant is not None]
    coverage = (len(quantized_tensors) / len(tensors)) if tensors else 1.0

    contract = {
        "backend": backend,
        "bit_width": int(bit_width),
        "total_tensors": len(tensors),
        "quantized_tensors": len(quantized_tensors),
        "coverage": coverage,
        "strict": strict,
        "status": "ok",
        "notes": [],
    }

    # Backends commonly used with integer quantization.
    quant_expected = bit_width <= 8 and backend in {
        "tflm",
        "vela",
        "cvi",
        "eiq",
        "ai8x",
    }
    if quant_expected and coverage == 0.0:
        msg = (
            "No per-tensor quant metadata found in IR while quantized deployment is expected. "
            "Proceeding in non-strict mode."
        )
        contract["notes"].append(msg)
        contract["status"] = "warning"
        if strict:
            raise CompilationError(msg)

    program.metadata["quantization_contract"] = contract
    return contract
