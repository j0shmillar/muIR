from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------- Types / Quantization ----------


@dataclass
class TensorType:
    shape: List[int]
    dtype: str
    layout: Optional[str] = None


@dataclass
class QuantParams:
    scale: float
    zero_point: int
    bit_width: int
    axis: Optional[int] = None


@dataclass
class Tensor:
    id: str
    type: TensorType
    quant: Optional[QuantParams] = None
    data: Optional[List[float]] = None
    role: str = "intermediate"  # input | output | initializer | intermediate
    producer: Optional[str] = None
    consumers: List[str] = field(default_factory=list)
    is_constant: bool = False


# ---------- Ops / Graph ----------


@dataclass
class Op:
    id: str
    kind: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    domain: str = ""
    source_name: str = ""

    # Analysis / lowering info:
    legal_backends: List[str] = field(default_factory=list)
    preferred_backend: Optional[str] = None


@dataclass
class Graph:
    """Single-function graph in unified IR."""

    name: str
    inputs: List[str]
    outputs: List[str]
    initializers: List[str]
    tensors: Dict[str, Tensor]
    ops: Dict[str, Op]
    op_order: List[str]


# ---------- Partitions / Program ----------


@dataclass
class Partition:
    id: str
    backend: str
    op_ids: List[str]


@dataclass
class BackendArtifact:
    backend: str
    artifact_type: str
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    graph: Graph
    partitions: List[Partition]
    backend_artifacts: List[BackendArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, artifact: BackendArtifact) -> None:
        self.backend_artifacts.append(artifact)


# ---------- Frontend: ONNX -> IR ----------


import os
from pathlib import Path

import onnx
from onnx import numpy_helper


def _onnx_dtype_to_ir(elem_type: int) -> str:
    onnx_to_ir = {
        onnx.TensorProto.FLOAT: "f32",
        onnx.TensorProto.FLOAT16: "f16",
        onnx.TensorProto.DOUBLE: "f64",
        onnx.TensorProto.BFLOAT16: "bf16",
        onnx.TensorProto.UINT8: "u8",
        onnx.TensorProto.UINT16: "u16",
        onnx.TensorProto.UINT32: "u32",
        onnx.TensorProto.UINT64: "u64",
        onnx.TensorProto.INT8: "i8",
        onnx.TensorProto.INT16: "i16",
        onnx.TensorProto.INT32: "i32",
        onnx.TensorProto.INT64: "i64",
        onnx.TensorProto.BOOL: "bool",
    }
    return onnx_to_ir.get(elem_type, f"onnx_{elem_type}")


def _shape_from_onnx_dims(dims: Any) -> List[int]:
    shape: List[int] = []
    for d in dims:
        if getattr(d, "dim_param", None):
            shape.append(-1)
        else:
            val = int(getattr(d, "dim_value", 0) or 0)
            shape.append(val if val > 0 else -1)
    return shape


def _tensor_type_from_value_info(vinfo: Any) -> TensorType:
    ttype = vinfo.type.tensor_type
    shape = _shape_from_onnx_dims(ttype.shape.dim)
    layout = None
    if len(shape) == 4:
        layout = "NCHW"
    elif len(shape) == 2:
        layout = "NC"
    return TensorType(
        shape=shape,
        dtype=_onnx_dtype_to_ir(ttype.elem_type),
        layout=layout,
    )


def build_program_from_onnx(
    onnx_path: str,
    *,
    default_backend: str,
    target_hardware: str,
    bit_width: int,
    metadata: Dict[str, Any],
) -> Program:
    """Import an ONNX model into unified IR and attach an initial partition."""

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(onnx_path)
    g = model.graph

    tensors: Dict[str, Tensor] = {}

    # Known value infos: graph inputs/outputs + inferred intermediates when present.
    value_infos = list(g.input) + list(g.output) + list(getattr(g, "value_info", []))
    for v in value_infos:
        if v.name and v.name not in tensors:
            tensors[v.name] = Tensor(
                id=v.name,
                type=_tensor_type_from_value_info(v),
                role="intermediate",
            )

    input_names = [v.name for v in g.input]
    output_names = [v.name for v in g.output]
    initializer_names = [init.name for init in g.initializer]

    # Initializers (weights/constants).
    for init in g.initializer:
        if init.name not in tensors:
            arr = numpy_helper.to_array(init)
            tensors[init.name] = Tensor(
                id=init.name,
                type=TensorType(
                    shape=[int(d) for d in init.dims],
                    dtype=_onnx_dtype_to_ir(init.data_type),
                    layout="NCHW" if len(init.dims) == 4 else ("NC" if len(init.dims) == 2 else None),
                ),
                data=arr.astype("float32").reshape(-1).tolist(),
            )
        elif tensors[init.name].data is None:
            arr = numpy_helper.to_array(init)
            tensors[init.name].data = arr.astype("float32").reshape(-1).tolist()
        tensors[init.name].role = "initializer"
        tensors[init.name].is_constant = True

    # Inputs that are not initializers are true user inputs.
    init_set = set(initializer_names)
    for name in input_names:
        if name in tensors and name not in init_set:
            tensors[name].role = "input"

    # Mark graph outputs.
    for name in output_names:
        if name in tensors:
            tensors[name].role = "output"

    ops: Dict[str, Op] = {}
    op_order: List[str] = []

    for idx, node in enumerate(g.node):
        op_id = f"op_{idx}"

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
            else:
                attrs[a.name] = f"onnx_attr_type_{a.type}"

        input_ids = [x for x in node.input if x]
        output_ids = [x for x in node.output if x]

        # Ensure all referenced tensors exist.
        for tid in input_ids + output_ids:
            if tid not in tensors:
                tensors[tid] = Tensor(
                    id=tid,
                    type=TensorType(shape=[], dtype="unknown"),
                )

        # Producer / consumer links.
        for tid in input_ids:
            tensors[tid].consumers.append(op_id)
        for tid in output_ids:
            tensors[tid].producer = op_id

        op = Op(
            id=op_id,
            kind=node.op_type,
            inputs=input_ids,
            outputs=output_ids,
            attrs=attrs,
            domain=node.domain or "",
            source_name=node.name or "",
        )
        ops[op_id] = op
        op_order.append(op_id)

    graph = Graph(
        name=g.name or "main",
        inputs=input_names,
        outputs=output_names,
        initializers=initializer_names,
        tensors=tensors,
        ops=ops,
        op_order=op_order,
    )

    part = Partition(
        id="p0",
        backend=default_backend,
        op_ids=op_order.copy(),
    )

    return Program(
        graph=graph,
        partitions=[part],
        backend_artifacts=[],
        metadata={
            **metadata,
            "target_hardware": target_hardware,
            "bit_width": bit_width,
            "onnx_ir_version": int(model.ir_version),
            "onnx_opsets": {str(x.domain or "ai.onnx"): int(x.version) for x in model.opset_import},
        },
    )


def build_program_from_tflite_stub(
    tflite_path: str,
    *,
    default_backend: str,
    target_hardware: str,
    bit_width: int,
    metadata: Dict[str, Any],
) -> Program:
    """Create a minimal IR program from a TFLite path when full graph import is unavailable."""
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

    graph = Graph(
        name=Path(tflite_path).stem or "tflite_model",
        inputs=[],
        outputs=[],
        initializers=[],
        tensors={},
        ops={},
        op_order=[],
    )
    return Program(
        graph=graph,
        partitions=[],
        backend_artifacts=[],
        metadata={
            **metadata,
            "target_hardware": target_hardware,
            "bit_width": bit_width,
            "frontend": "tflite_stub",
            "tflite_path": os.path.abspath(tflite_path),
        },
    )


def _torch_dtype_to_ir(dtype: Any) -> str:
    import torch

    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float64: "f64",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.uint8: "u8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def _fx_target_to_kind(node: Any, module: Any) -> str:
    import operator
    import torch

    if node.op == "call_module":
        sub = module.get_submodule(str(node.target))
        mname = sub.__class__.__name__
        module_map = {
            "Conv2d": "Conv",
            "ConvTranspose2d": "ConvTranspose",
            "Linear": "Gemm",
            "BatchNorm2d": "BatchNormalization",
            "ReLU": "Relu",
            "LeakyReLU": "LeakyRelu",
            "MaxPool2d": "MaxPool",
            "AvgPool2d": "AveragePool",
            "AdaptiveAvgPool2d": "GlobalAveragePool",
            "Flatten": "Flatten",
            "Identity": "Identity",
        }
        return module_map.get(mname, mname)

    if node.op == "call_function":
        fn = node.target
        fn_map = {
            operator.add: "Add",
            torch.add: "Add",
            operator.sub: "Sub",
            torch.sub: "Sub",
            operator.mul: "Mul",
            torch.mul: "Mul",
            torch.matmul: "MatMul",
            torch.flatten: "Flatten",
            torch.relu: "Relu",
            torch.sigmoid: "Sigmoid",
            torch.tanh: "Tanh",
        }
        return fn_map.get(fn, getattr(fn, "__name__", str(fn)))

    if node.op == "call_method":
        m = str(node.target)
        method_map = {
            "view": "Reshape",
            "reshape": "Reshape",
            "flatten": "Flatten",
            "permute": "Transpose",
            "transpose": "Transpose",
            "contiguous": "Identity",
            "mean": "ReduceMean",
        }
        return method_map.get(m, m)

    return node.op


def _to_int_list(value: Any, *, expected_len: int | None = None) -> List[int]:
    if isinstance(value, int):
        out = [int(value)]
    elif isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
    else:
        out = []
    if expected_len is not None and out:
        if len(out) == 1 and expected_len > 1:
            out = out * expected_len
        elif len(out) < expected_len:
            out = out + [out[-1]] * (expected_len - len(out))
        elif len(out) > expected_len:
            out = out[:expected_len]
    return out


def _extract_module_attrs(sub: Any) -> Dict[str, Any]:
    import torch
    from torch import nn

    attrs: Dict[str, Any] = {}
    if isinstance(sub, nn.Conv2d):
        k = _to_int_list(sub.kernel_size, expected_len=2)
        s = _to_int_list(sub.stride, expected_len=2)
        d = _to_int_list(sub.dilation, expected_len=2)
        p = _to_int_list(sub.padding, expected_len=2)
        pads = [p[0], p[1], p[0], p[1]] if p else [0, 0, 0, 0]
        attrs.update(
            {
                "kernel_shape": k or [1, 1],
                "strides": s or [1, 1],
                "dilations": d or [1, 1],
                "pads": pads,
                "group": int(sub.groups),
                "bias": sub.bias is not None,
            }
        )
    elif isinstance(sub, (nn.MaxPool2d, nn.AvgPool2d)):
        k = _to_int_list(sub.kernel_size, expected_len=2)
        s = _to_int_list(sub.stride, expected_len=2) if sub.stride is not None else k
        p = _to_int_list(sub.padding, expected_len=2)
        pads = [p[0], p[1], p[0], p[1]] if p else [0, 0, 0, 0]
        attrs.update(
            {
                "kernel_shape": k or [1, 1],
                "strides": s or [1, 1],
                "pads": pads,
            }
        )
    elif isinstance(sub, nn.Flatten):
        attrs["axis"] = int(getattr(sub, "start_dim", 1))
    elif isinstance(sub, nn.Linear):
        attrs["transB"] = 1
    elif isinstance(sub, nn.BatchNorm2d):
        attrs["epsilon"] = float(getattr(sub, "eps", 1e-5))
        attrs["momentum"] = float(getattr(sub, "momentum", 0.1) or 0.1)
    elif isinstance(sub, nn.AdaptiveAvgPool2d):
        out_size = getattr(sub, "output_size", None)
        if isinstance(out_size, int):
            attrs["output_size"] = [int(out_size), int(out_size)]
        elif isinstance(out_size, tuple):
            attrs["output_size"] = [int(out_size[0]), int(out_size[1])]
    elif isinstance(sub, nn.LeakyReLU):
        attrs["alpha"] = float(getattr(sub, "negative_slope", 0.01))
    elif isinstance(sub, nn.Hardtanh):
        attrs["min"] = float(getattr(sub, "min_val", -1.0))
        attrs["max"] = float(getattr(sub, "max_val", 1.0))
    elif isinstance(sub, nn.Dropout):
        attrs["ratio"] = float(getattr(sub, "p", 0.5))
    elif isinstance(sub, nn.Upsample):
        sf = getattr(sub, "scale_factor", None)
        if sf is not None:
            if isinstance(sf, (tuple, list)):
                attrs["scales"] = [float(x) for x in sf]
            else:
                attrs["scales"] = [float(sf)]
        mode = getattr(sub, "mode", None)
        if mode:
            attrs["mode"] = str(mode)
    elif isinstance(sub, nn.Identity):
        attrs["noop"] = True
    elif isinstance(sub, torch.nn.Module):
        # Unknown module type, leave attrs empty for now.
        pass
    return attrs


def build_program_from_torch(
    model: Any,
    example_input: Any,
    *,
    default_backend: str,
    target_hardware: str,
    bit_width: int,
    metadata: Dict[str, Any],
) -> Program:
    """Trace a torch.nn.Module into unified IR using torch.fx."""
    import torch
    import torch.fx as fx
    from torch.fx.passes.shape_prop import ShapeProp

    model = model.eval().to("cpu")
    traced = fx.symbolic_trace(model)

    # Best effort shape/dtype propagation.
    try:
        ShapeProp(traced).propagate(example_input)
    except Exception:
        pass

    tensors: Dict[str, Tensor] = {}
    ops: Dict[str, Op] = {}
    op_order: List[str] = []
    initializers: List[str] = []
    node_tensor: Dict[Any, str] = {}
    input_names: List[str] = []
    output_names: List[str] = []

    def ensure_tensor(name: str, *, role: str = "intermediate", producer: str | None = None) -> None:
        if name in tensors:
            return
        tensors[name] = Tensor(
            id=name,
            type=TensorType(shape=[], dtype="unknown"),
            role=role,
            producer=producer,
        )

    def update_tensor_type_from_meta(name: str, node: Any) -> None:
        if name not in tensors:
            ensure_tensor(name)
        tmeta = (getattr(node, "meta", {}) or {}).get("tensor_meta")
        if tmeta is None:
            return
        shape = [int(x) if isinstance(x, int) else -1 for x in getattr(tmeta, "shape", [])]
        dtype = _torch_dtype_to_ir(getattr(tmeta, "dtype", None))
        layout = None
        if len(shape) == 4:
            layout = "NCHW"
        elif len(shape) == 2:
            layout = "NC"
        tensors[name].type = TensorType(shape=shape, dtype=dtype, layout=layout)

    def resolve_node_to_tensor(arg: Any) -> Optional[str]:
        if isinstance(arg, fx.Node):
            return node_tensor.get(arg)
        return None

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            tid = str(node.target)
            node_tensor[node] = tid
            input_names.append(tid)
            ensure_tensor(tid, role="input")
            update_tensor_type_from_meta(tid, node)
            continue

        if node.op == "get_attr":
            tid = str(node.target)
            node_tensor[node] = tid
            ensure_tensor(tid, role="initializer")
            tensors[tid].is_constant = True
            if tid not in initializers:
                initializers.append(tid)
            try:
                attr = traced.get_parameter(tid)
            except Exception:
                try:
                    attr = traced.get_buffer(tid)
                except Exception:
                    attr = None
            if isinstance(attr, torch.Tensor):
                tensors[tid].type = TensorType(
                    shape=[int(x) for x in attr.shape],
                    dtype=_torch_dtype_to_ir(attr.dtype),
                )
            continue

        if node.op == "output":
            args = node.args[0] if node.args else []
            flat = list(args) if isinstance(args, (list, tuple)) else [args]
            for i, a in enumerate(flat):
                tid = resolve_node_to_tensor(a)
                if not tid:
                    tid = f"output_{i}"
                    ensure_tensor(tid, role="output")
                tensors[tid].role = "output"
                output_names.append(tid)
            continue

        op_id = f"op_{len(op_order)}"
        kind = _fx_target_to_kind(node, traced)

        input_ids: List[str] = []
        for arg in fx.node.map_arg(node.args, lambda x: x):
            tid = resolve_node_to_tensor(arg)
            if tid is not None:
                input_ids.append(tid)
        for _k, v in node.kwargs.items():
            tid = resolve_node_to_tensor(v)
            if tid is not None:
                input_ids.append(tid)

        if node.op == "call_module":
            sub = traced.get_submodule(str(node.target))
            prefix = str(node.target)
            for pname, param in sub.named_parameters(recurse=False):
                tid = f"{prefix}.{pname}"
                ensure_tensor(tid, role="initializer")
                tensors[tid].is_constant = True
                tensors[tid].type = TensorType(
                    shape=[int(x) for x in param.shape],
                    dtype=_torch_dtype_to_ir(param.dtype),
                    layout="NCHW" if param.ndim == 4 else ("NC" if param.ndim == 2 else None),
                )
                tensors[tid].data = (
                    param.detach().cpu().to(torch.float32).reshape(-1).tolist()
                )
                if tid not in initializers:
                    initializers.append(tid)
                input_ids.append(tid)
            for bname, buf in sub.named_buffers(recurse=False):
                tid = f"{prefix}.{bname}"
                ensure_tensor(tid, role="initializer")
                tensors[tid].is_constant = True
                tensors[tid].type = TensorType(
                    shape=[int(x) for x in buf.shape],
                    dtype=_torch_dtype_to_ir(buf.dtype),
                )
                tensors[tid].data = (
                    buf.detach().cpu().to(torch.float32).reshape(-1).tolist()
                )
                if tid not in initializers:
                    initializers.append(tid)
                input_ids.append(tid)

        out_tid = f"{op_id}_out0"
        node_tensor[node] = out_tid
        ensure_tensor(out_tid, producer=op_id)
        update_tensor_type_from_meta(out_tid, node)

        attrs: Dict[str, Any] = {}
        if node.op == "call_module":
            sub = traced.get_submodule(str(node.target))
            attrs.update(_extract_module_attrs(sub))
        for k, v in dict(node.kwargs).items():
            if isinstance(v, (int, float, str, bool)):
                attrs[k] = v
            elif isinstance(v, (list, tuple)):
                attrs[k] = [x for x in v if isinstance(x, (int, float, str, bool))]

        op = Op(
            id=op_id,
            kind=kind,
            inputs=input_ids,
            outputs=[out_tid],
            attrs=attrs,
            source_name=str(node.target),
        )
        ops[op_id] = op
        op_order.append(op_id)

        for tid in input_ids:
            ensure_tensor(tid)
            tensors[tid].consumers.append(op_id)
        tensors[out_tid].producer = op_id

    if not output_names and op_order:
        last_out = ops[op_order[-1]].outputs[0]
        tensors[last_out].role = "output"
        output_names = [last_out]

    graph = Graph(
        name=getattr(traced, "__class__", type("x", (), {})).__name__ or "main",
        inputs=input_names,
        outputs=output_names,
        initializers=initializers,
        tensors=tensors,
        ops=ops,
        op_order=op_order,
    )

    return Program(
        graph=graph,
        partitions=[Partition(id="p0", backend=default_backend, op_ids=op_order.copy())],
        backend_artifacts=[],
        metadata={
            **metadata,
            "target_hardware": target_hardware,
            "bit_width": bit_width,
            "frontend": "torch_fx",
        },
    )


# ---------- Serialization ----------


import json


def program_to_json(program: Program) -> Dict[str, Any]:
    """Convert Program to a JSON-serializable dict."""

    def encode(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
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
