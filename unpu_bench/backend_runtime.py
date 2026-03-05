from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .errors import CompilationError


def load_compiled_model(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_torch_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _op_conv(op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    attrs = op.get("attrs", {})
    xs = [env[i] for i in op["inputs"]]
    x = xs[0]
    w = xs[1]
    b = xs[2] if len(xs) > 2 else None
    y = F.conv2d(
        x,
        w,
        bias=b,
        stride=tuple(attrs.get("strides", [1, 1])),
        padding=tuple(attrs.get("pads", [0, 0, 0, 0])[:2]),
        dilation=tuple(attrs.get("dilations", [1, 1])),
        groups=int(attrs.get("group", 1)),
    )
    env[op["outputs"][0]] = y


def _op_pool(op: Dict[str, Any], env: Dict[str, torch.Tensor], avg: bool) -> None:
    attrs = op.get("attrs", {})
    x = env[op["inputs"][0]]
    fn = F.avg_pool2d if avg else F.max_pool2d
    y = fn(
        x,
        kernel_size=tuple(attrs.get("kernel_shape", [1, 1])),
        stride=tuple(attrs.get("strides", [1, 1])),
        padding=tuple(attrs.get("pads", [0, 0, 0, 0])[:2]),
    )
    env[op["outputs"][0]] = y


def execute_compiled_model(compiled: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    env: Dict[str, torch.Tensor] = {}
    tensors = compiled.get("tensors", {})
    constants = compiled.get("constants", {})

    for tid, spec in constants.items():
        shape = list(spec.get("shape", []))
        data = spec.get("data", [])
        env[tid] = torch.tensor(data, dtype=torch.float32).reshape(shape)

    for name, value in inputs.items():
        env[name] = _as_torch_tensor(value).to(torch.float32)

    for op in compiled.get("ops", []):
        kind = op["kind"]
        if kind in {"Conv", "DepthwiseConv2d"}:
            _op_conv(op, env)
            continue
        if kind == "Relu":
            env[op["outputs"][0]] = F.relu(env[op["inputs"][0]])
            continue
        if kind == "LeakyRelu":
            alpha = float(op.get("attrs", {}).get("alpha", 0.01))
            env[op["outputs"][0]] = F.leaky_relu(env[op["inputs"][0]], negative_slope=alpha)
            continue
        if kind == "MaxPool":
            _op_pool(op, env, avg=False)
            continue
        if kind == "AveragePool":
            _op_pool(op, env, avg=True)
            continue
        if kind == "GlobalAveragePool":
            x = env[op["inputs"][0]]
            env[op["outputs"][0]] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            continue
        if kind == "Flatten":
            axis = int(op.get("attrs", {}).get("axis", 1))
            x = env[op["inputs"][0]]
            env[op["outputs"][0]] = torch.flatten(x, start_dim=axis)
            continue
        if kind == "Gemm":
            attrs = op.get("attrs", {})
            a = env[op["inputs"][0]]
            b = env[op["inputs"][1]]
            c = env[op["inputs"][2]] if len(op["inputs"]) > 2 else None
            if int(attrs.get("transA", 0)) == 1:
                a = a.transpose(-1, -2)
            if int(attrs.get("transB", 0)) == 1:
                b = b.transpose(-1, -2)
            y = a.matmul(b)
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            y = y * alpha
            if c is not None:
                y = y + c * beta
            env[op["outputs"][0]] = y
            continue
        if kind == "MatMul":
            a = env[op["inputs"][0]]
            b = env[op["inputs"][1]]
            env[op["outputs"][0]] = a.matmul(b)
            continue
        if kind == "Add":
            env[op["outputs"][0]] = env[op["inputs"][0]] + env[op["inputs"][1]]
            continue
        if kind == "Sub":
            env[op["outputs"][0]] = env[op["inputs"][0]] - env[op["inputs"][1]]
            continue
        if kind == "Mul":
            env[op["outputs"][0]] = env[op["inputs"][0]] * env[op["inputs"][1]]
            continue
        if kind == "Div":
            env[op["outputs"][0]] = env[op["inputs"][0]] / env[op["inputs"][1]]
            continue
        if kind == "Reshape":
            x = env[op["inputs"][0]]
            shape_src = env[op["inputs"][1]].to(torch.int64).reshape(-1).tolist() if len(op["inputs"]) > 1 else []
            if not shape_src:
                raise CompilationError("Reshape op missing shape input")
            env[op["outputs"][0]] = x.reshape(shape_src)
            continue
        if kind == "Transpose":
            perm = op.get("attrs", {}).get("perm", [])
            if not perm:
                raise CompilationError("Transpose op missing perm attr")
            env[op["outputs"][0]] = env[op["inputs"][0]].permute(*perm)
            continue
        if kind == "Identity":
            env[op["outputs"][0]] = env[op["inputs"][0]]
            continue
        raise CompilationError(f"Compiled runtime does not support op kind '{kind}'")

    outputs: Dict[str, torch.Tensor] = {}
    for tid in compiled.get("graph", {}).get("outputs", []):
        if tid not in env:
            raise CompilationError(f"Missing output tensor at runtime: {tid}")
        outputs[tid] = env[tid]
    return outputs
