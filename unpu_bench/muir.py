# unpu_bench/ir.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------- Types / Quantization ----------

@dataclass
class TensorType:
    shape: List[int]
    dtype: str  # "f32", "i8", etc.


@dataclass
class QuantParams:
    scale: float
    zero_point: int
    bit_width: int


@dataclass
class Tensor:
    id: str
    type: TensorType
    quant: Optional[QuantParams] = None


# ---------- Ops / Graph ----------

@dataclass
class Op:
    id: str
    kind: str               # e.g. "Conv", "Relu"
    inputs: List[str]       # tensor ids
    outputs: List[str]      # tensor ids
    attrs: Dict[str, Any]   # op-specific metadata

    # Analysis / lowering info:
    legal_backends: List[str] = field(default_factory=list)
    preferred_backend: Optional[str] = None


@dataclass
class Graph:
    """Single-function graph.

    We keep it simple: a flat, acyclic graph with named tensors and ops.
    """
    name: str
    inputs: List[str]               # tensor ids
    outputs: List[str]              # tensor ids
    tensors: Dict[str, Tensor]
    ops: Dict[str, Op]              # op_id -> Op


# ---------- Partitions / Program ----------

@dataclass
class Partition:
    id: str
    backend: str          # "ai8x", "cpu", "vela", ...
    op_ids: List[str]     # subset of graph.ops keys


@dataclass
class BackendArtifact:
    backend: str          # "ai8x", "cpu"
    artifact_type: str    # "c_project", "onnx", "tflite", ...
    path: str             # relative to program root
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    graph: Graph
    partitions: List[Partition]
    backend_artifacts: List[BackendArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # convenience:
    def add_artifact(self, artifact: BackendArtifact) -> None:
        self.backend_artifacts.append(artifact)


# ---------- Frontend: ONNX -> IR ----------

import os
import onnx


def build_program_from_onnx(
    onnx_path: str,
    *,
    default_backend: str,
    target_hardware: str,
    bit_width: int,
    metadata: Dict[str, Any],
) -> Program:
    """Import an ONNX model into our IR and attach a single initial partition.

    This is the *canonical* starting point: everything else (legality,
    quantization, partitioning, backend compilation) flows from this Program.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(onnx_path)
    g = model.graph

    tensors: Dict[str, Tensor] = {}

    def ttype_from_value_info(vinfo) -> TensorType:
        ttype = vinfo.type.tensor_type
        shape = [int(d.dim_value) for d in ttype.shape.dim]
        elem = ttype.elem_type
        # Map ONNX dtype to a simple string
        onnx_to_ir = {
            onnx.TensorProto.FLOAT: "f32",
            onnx.TensorProto.UINT8: "u8",
            onnx.TensorProto.INT8: "i8",
            onnx.TensorProto.INT32: "i32",
            onnx.TensorProto.INT64: "i64",
        }
        return TensorType(shape=shape, dtype=onnx_to_ir.get(elem, f"onnx_{elem}"))

    # 1) Graph inputs / outputs
    for v in g.input:
        tensors[v.name] = Tensor(
            id=v.name,
            type=ttype_from_value_info(v),
        )
    for v in g.output:
        if v.name not in tensors:
            tensors[v.name] = Tensor(
                id=v.name,
                type=ttype_from_value_info(v),
            )

    # 2) Nodes → Ops + produced tensors
    ops: Dict[str, Op] = {}
    for idx, node in enumerate(g.node):
        op_id = f"op_{idx}"
        # Attributes (primitives only)
        attrs: Dict[str, Any] = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = float(a.f)
            elif a.type == onnx.AttributeProto.INT:
                attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8", errors="ignore")
            elif a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = [int(v) for v in a.ints]
            elif a.type == onnx.AttributeProto.FLOATS:
                attrs[a.name] = [float(v) for v in a.floats]
            # others ignored for now

        # Create tensor shells for outputs if missing
        for out in node.output:
            if out and out not in tensors:
                # Type/shape inference is non-trivial; start unknown.
                tensors[out] = Tensor(
                    id=out,
                    type=TensorType(shape=[], dtype="unknown"),
                )

        ops[op_id] = Op(
            id=op_id,
            kind=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        )

    input_names = [v.name for v in g.input]
    output_names = [v.name for v in g.output]

    graph = Graph(
        name=model.graph.name or "main",
        inputs=input_names,
        outputs=output_names,
        tensors=tensors,
        ops=ops,
    )

    # Initial single partition: everything to default_backend (we'll refine with passes)
    part = Partition(
        id="p0",
        backend=default_backend,
        op_ids=list(ops.keys()),
    )

    return Program(
        graph=graph,
        partitions=[part],
        backend_artifacts=[],
        metadata={
            **metadata,
            "target_hardware": target_hardware,
            "bit_width": bit_width,
        },
    )


# ---------- Serialization ----------

import json


def program_to_json(program: Program) -> Dict[str, Any]:
    """Convert Program to a JSON-serialisable dict."""
    def encode(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (list, dict, str, int, float)) or obj is None:
            return obj
        raise TypeError(f"Cannot JSON-encode {type(obj)}")

    return encode(program)


def write_program_json(program: Program, out_dir: str, filename: str = "program.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    data = program_to_json(program)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path
