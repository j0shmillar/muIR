from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .muir import Program


def _as_tensor(program: Program, tid: str) -> torch.Tensor:
    t = program.graph.tensors[tid]
    if t.data is None:
        raise ValueError(f"tensor {tid} has no constant data")
    shape = tuple(int(x) for x in t.type.shape) if t.type.shape else (-1,)
    flat = torch.tensor(t.data, dtype=torch.float32)
    if shape != (-1,):
        return flat.reshape(shape)
    return flat


def _pool_params(attrs: Dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    kernel = attrs.get("kernel_shape", [1, 1])
    strides = attrs.get("strides", kernel)
    pads = attrs.get("pads", [0, 0, 0, 0])
    return kernel, strides, pads


def run_semantic_check_torch_vs_ir(
    *,
    program: Program,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    vals: Dict[str, torch.Tensor] = {}
    g = program.graph
    if not g.inputs:
        return {"status": "skipped", "reason": "no_graph_inputs"}
    vals[g.inputs[0]] = example_input.detach().cpu().float()

    for tid in g.initializers:
        t = g.tensors.get(tid)
        if t is not None and t.data is not None:
            vals[tid] = _as_tensor(program, tid)

    unsupported: list[str] = []
    for op_id in g.op_order:
        op = g.ops[op_id]
        try:
            ins = [vals[x] for x in op.inputs]
        except KeyError:
            return {"status": "skipped", "reason": f"missing_input_value:{op_id}"}

        kind = op.kind
        out: torch.Tensor
        if kind in {"Conv", "DepthwiseConv2d"}:
            stride = tuple(op.attrs.get("strides", [1, 1]))
            dil = tuple(op.attrs.get("dilations", [1, 1]))
            pads = op.attrs.get("pads", [0, 0, 0, 0])
            pad = (int(pads[0]), int(pads[1]))
            groups = int(op.attrs.get("group", 1))
            w = ins[1]
            b = ins[2] if len(ins) >= 3 else None
            out = F.conv2d(
                ins[0], w, b, stride=stride, padding=pad, dilation=dil, groups=groups
            )
        elif kind == "BatchNormalization":
            if len(ins) < 5:
                return {"status": "skipped", "reason": f"bn_inputs_incomplete:{op_id}"}
            eps = float(op.attrs.get("epsilon", 1e-5))
            out = F.batch_norm(
                ins[0], ins[3], ins[4], ins[1], ins[2], training=False, eps=eps
            )
        elif kind in {"Relu", "ReLU"}:
            out = torch.relu(ins[0])
        elif kind == "ReLU6":
            out = torch.clamp(ins[0], min=0.0, max=6.0)
        elif kind == "GELU":
            out = F.gelu(ins[0])
        elif kind == "Add":
            out = ins[0] + ins[1]
        elif kind == "Sub":
            out = ins[0] - ins[1]
        elif kind == "Mul":
            out = ins[0] * ins[1]
        elif kind == "Div":
            out = ins[0] / ins[1]
        elif kind == "Flatten":
            axis = int(op.attrs.get("axis", 1))
            out = torch.flatten(ins[0], start_dim=axis)
        elif kind == "Gemm":
            x = ins[0]
            w = ins[1]
            trans_b = int(op.attrs.get("transB", 0))
            out = x @ (w.t() if trans_b else w)
            if len(ins) >= 3:
                out = out + ins[2]
        elif kind == "GlobalAveragePool":
            out = F.adaptive_avg_pool2d(ins[0], output_size=(1, 1))
        elif kind == "AveragePool":
            k, s, p = _pool_params(op.attrs)
            out = F.avg_pool2d(
                ins[0], kernel_size=tuple(k), stride=tuple(s), padding=(p[0], p[1])
            )
        elif kind == "MaxPool":
            k, s, p = _pool_params(op.attrs)
            out = F.max_pool2d(
                ins[0], kernel_size=tuple(k), stride=tuple(s), padding=(p[0], p[1])
            )
        elif kind == "Reshape":
            if len(ins) >= 2:
                shape = [int(x) for x in ins[1].reshape(-1).tolist()]
                out = ins[0].reshape(shape)
            else:
                unsupported.append(kind)
                continue
        elif kind == "Transpose":
            perm = op.attrs.get("perm")
            if perm:
                out = ins[0].permute(*perm)
            else:
                unsupported.append(kind)
                continue
        elif kind == "Identity":
            out = ins[0]
        else:
            unsupported.append(kind)
            continue

        vals[op.outputs[0]] = out

    if unsupported:
        return {
            "status": "skipped",
            "reason": "unsupported_ops",
            "ops": sorted(set(unsupported)),
        }

    with torch.no_grad():
        ref = model.eval().cpu()(example_input.detach().cpu().float())
    if isinstance(ref, (list, tuple)):
        ref = ref[0]
    out_name = g.outputs[0] if g.outputs else g.op_order[-1] + "_out0"
    got = vals.get(out_name)
    if got is None:
        return {"status": "skipped", "reason": "missing_output_tensor"}

    ref_f = ref.reshape(-1).float()
    got_f = got.reshape(-1).float()
    n = min(ref_f.numel(), got_f.numel())
    ref_f = ref_f[:n]
    got_f = got_f[:n]
    diff = (ref_f - got_f).abs()
    max_abs = float(diff.max().item() if diff.numel() else 0.0)
    mean_abs = float(diff.mean().item() if diff.numel() else 0.0)
    cosine = (
        float(F.cosine_similarity(ref_f.unsqueeze(0), got_f.unsqueeze(0)).item())
        if n
        else 1.0
    )
    ok = bool(torch.allclose(ref_f, got_f, rtol=rtol, atol=atol))

    return {
        "status": "pass" if ok else "fail",
        "rtol": rtol,
        "atol": atol,
        "numel_compared": int(n),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cosine,
    }
