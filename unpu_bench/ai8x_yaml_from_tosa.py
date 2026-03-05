# # unpu_bench/ai8x_yaml_from_tosa.py
# from __future__ import annotations

# """
# Generate an ai8x YAML template from canonical TOSA MLIR (SoT IR).

# Goal:
#   Match (as closely as practical) the functionality of ai8x-training's ALPHA
#   YAML generator that operates on a PyTorch graph, but using your TOSA/MLIR
#   pipeline as the source of truth.

# Scope:
#   - Builds a lightweight dataflow graph from the TOSA MLIR text (SSA values).
#   - Extracts tensor shapes/dtypes, operator attributes, and connectivity.
#   - Emits ai8x YAML with:
#       * layer list (Conv/ConvTranspose/Linear/Eltwise/Pool/Abs/Relu)
#       * fusion passes (activation, pooling, eltwise) with veto checks
#       * write_gap insertion for multi-input eltwise layers
#       * processor allocation (weight-aware "row allocator") identical in spirit
#       * ping-pong out_offset allocation identical in spirit
#       * input/output shape comments and dimension warnings

# Important notes:
#   - This is a "template emitter": it aims to be good enough for real models,
#     but if your TOSA contains ops not representable in ai8x YAML, you should
#     mark them CPU in capabilities OR extend lowering rules here.
#   - TOSA layout is often NHWC for conv2d; the code uses heuristics to infer
#     channel dimension.
# """

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union, Iterable

# import re
# import warnings

# import numpy as np
# import torch

# from .errors import CompilationError
# from .tosa_ir import TosaModule

# # ----------------------------- Public result types -----------------------------


# @dataclass(frozen=True)
# class Ai8xYamlEmitResult:
#     yaml_path: Path
#     sample_path: Path


# # ----------------------------- Device / constants -----------------------------


# def _device_id_from_name(device_name: str) -> int:
#     dn = (device_name or "").upper()
#     if "78002" in dn or "MAX78002" in dn:
#         return 87
#     # Default to MAX78000 (ai85)
#     return 85


# def _max_pixels_for_device(device_id: int) -> int:
#     # matches the reference script
#     if device_id == 85:
#         return 8192
#     if device_id == 87:
#         return 20480
#     return 8192


# def _half_data_for_device(device_id: int) -> int:
#     # matches the reference script
#     return 0x4000 if device_id == 85 else 0xA000


# # ----------------------------- YAML helpers -----------------------------


# def _wrap_comment(s: str, width: int = 78) -> List[str]:
#     words = s.split()
#     out: List[str] = []
#     cur: List[str] = []
#     n = 0
#     for w in words:
#         add = len(w) + (1 if cur else 0)
#         if n + add > width:
#             out.append("# " + " ".join(cur))
#             cur = [w]
#             n = len(w)
#         else:
#             cur.append(w)
#             n += add
#     if cur:
#         out.append("# " + " ".join(cur))
#     return out


# def _write_sample_npy(out_dir: Path, x: torch.Tensor) -> Path:
#     """
#     ai8x izer expects an NCHW/HWC-ish sample in numpy.
#     Use int8-ish range by default to match typical ai8x flows.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out = out_dir / "sample.npy"
#     arr = x.detach().cpu().numpy()
#     if np.issubdtype(arr.dtype, np.floating):
#         arr = np.clip(np.rint(arr * 20.0), -127, 127).astype(np.int64)
#     np.save(out, arr)
#     return out


# # ----------------------------- MLIR parsing -----------------------------

# # We re-parse MLIR here because the existing TosaModule/TosaOpSig in your tree
# # does not preserve SSA ids (operands/results), which are necessary to reproduce
# # the reference generator’s graph-walking logic.

# _RE_OPNAME = re.compile(r'\b("tosa\.[^"]+"|tosa\.[A-Za-z0-9_]+)\b')
# _RE_RESIDS = re.compile(r"^\s*(%[\w\d_]+(?:\s*,\s*%[\w\d_]+)*)\s*=\s*")
# _RE_OPERANDS = re.compile(r"\(([^)]*)\)")
# _RE_SSA = re.compile(r"%[\w\d_]+")
# _RE_TENSOR = re.compile(r"tensor<([^>]+)>")
# _RE_ATTR_BLOB = re.compile(r"\{(.*)\}")
# _RE_TYPE_SIG = re.compile(r":\s*(.*)$")


# @dataclass
# class _TensorType:
#     shape: List[int]  # includes batch if present
#     dtype: str        # i8/i32/f32/...
#     raw: str


# @dataclass
# class _OpRec:
#     index: int
#     opname: str
#     res_ids: List[str]
#     operand_ids: List[str]
#     attrs: Dict[str, Any]
#     operand_types: List[_TensorType]
#     result_types: List[_TensorType]
#     line_no: int
#     text: str


# def _collect_tosa_statements(lines: List[str]) -> List[Tuple[int, str]]:
#     """
#     Collect multi-line MLIR statements containing 'tosa.' into single strings.
#     Uses a conservative balance heuristic for (), {}, <>.
#     """
#     stmts: List[Tuple[int, str]] = []
#     buf: List[str] = []
#     start_line = 0

#     def bal(s: str, ch_open: str, ch_close: str) -> int:
#         return s.count(ch_open) - s.count(ch_close)

#     par = br = ang = 0

#     for i, ln in enumerate(lines, start=1):
#         if not buf:
#             if "tosa." not in ln:
#                 continue
#             buf = [ln.rstrip("\n")]
#             start_line = i
#             par = bal(ln, "(", ")")
#             br = bal(ln, "{", "}")
#             ang = bal(ln, "<", ">")
#             continue

#         buf.append(ln.rstrip("\n"))
#         par += bal(ln, "(", ")")
#         br += bal(ln, "{", "}")
#         ang += bal(ln, "<", ">")

#         # End when balances settle and we see a type signature delimiter somewhere
#         joined = " ".join(buf)
#         if par <= 0 and br <= 0 and ang <= 0 and ":" in joined:
#             stmts.append((start_line, joined))
#             buf = []

#     # Flush if any residual
#     if buf:
#         stmts.append((start_line, " ".join(buf)))

#     return stmts


# def _parse_tensor_type(type_text: str) -> Optional[_TensorType]:
#     m = _RE_TENSOR.search(type_text)
#     if not m:
#         return None
#     body = m.group(1).strip()  # e.g. "1x32x32x3xi8"
#     parts = [p.strip() for p in body.split("x") if p.strip()]
#     if len(parts) < 2:
#         return None
#     dtype = parts[-1]
#     dims: List[int] = []
#     for d in parts[:-1]:
#         if d == "?":
#             dims.append(-1)
#         else:
#             try:
#                 dims.append(int(d))
#             except ValueError:
#                 dims.append(-1)
#     return _TensorType(shape=dims, dtype=dtype, raw=f"tensor<{body}>")


# def _split_top_level_commas(s: str) -> List[str]:
#     # split a string on commas, not descending into [] <> {} ()
#     out: List[str] = []
#     cur: List[str] = []
#     depth = 0
#     for ch in s:
#         if ch in "[<{(":
#             depth += 1
#         elif ch in "]>})":
#             depth = max(0, depth - 1)
#         if ch == "," and depth == 0:
#             out.append("".join(cur).strip())
#             cur = []
#         else:
#             cur.append(ch)
#     if cur:
#         out.append("".join(cur).strip())
#     return [x for x in out if x]


# def _parse_mlir_attr_value(v: str) -> Any:
#     v = v.strip()
#     # array<i64: 1, 2> or dense<[...]> etc: keep as raw unless simple
#     # Try to parse common forms: [1, 2], array<i64: 1, 2>, i64, f32, "str"
#     if v.startswith('"') and v.endswith('"'):
#         return v[1:-1]
#     if v.startswith("[") and v.endswith("]"):
#         inner = v[1:-1].strip()
#         if not inner:
#             return []
#         parts = _split_top_level_commas(inner)
#         parsed: List[Any] = []
#         for p in parts:
#             parsed.append(_parse_mlir_attr_value(p))
#         return parsed
#     m = re.match(r"array<i\d+:\s*(.*)>", v)
#     if m:
#         inner = m.group(1).strip()
#         parts = _split_top_level_commas(inner)
#         vals: List[int] = []
#         for p in parts:
#             p = p.strip()
#             try:
#                 vals.append(int(p))
#             except Exception:
#                 # fallback
#                 return v
#         return vals
#     # integers/floats/bools
#     if re.fullmatch(r"-?\d+", v):
#         try:
#             return int(v)
#         except Exception:
#             return v
#     if re.fullmatch(r"-?\d+\.\d*", v):
#         try:
#             return float(v)
#         except Exception:
#             return v
#     if v in ("true", "false"):
#         return v == "true"
#     return v


# def _parse_attr_dict(stmt: str) -> Dict[str, Any]:
#     attrs: Dict[str, Any] = {}
#     m = _RE_ATTR_BLOB.search(stmt)
#     if not m:
#         return attrs
#     blob = m.group(1).strip()
#     if not blob:
#         return attrs
#     items = _split_top_level_commas(blob)
#     for it in items:
#         if "=" not in it:
#             continue
#         k, v = it.split("=", 1)
#         attrs[k.strip()] = _parse_mlir_attr_value(v.strip())
#     return attrs


# def _parse_types(stmt: str) -> Tuple[List[_TensorType], List[_TensorType]]:
#     """
#     Parse MLIR type signature:
#       : (tensor<...>, tensor<...>) -> tensor<...>
#     or:
#       : tensor<...>
#     """
#     m = _RE_TYPE_SIG.search(stmt)
#     if not m:
#         return ([], [])
#     sig = m.group(1)
#     # Normalize quotes around opnames don't matter here.
#     # Find left/right if present.
#     if "->" in sig:
#         left, right = sig.split("->", 1)
#         # left might be "(t0, t1)" or "t0"
#         operand_types: List[_TensorType] = []
#         result_types: List[_TensorType] = []
#         for tm in _RE_TENSOR.finditer(left):
#             tt = _parse_tensor_type(f"tensor<{tm.group(1)}>")
#             if tt:
#                 operand_types.append(tt)
#         for tm in _RE_TENSOR.finditer(right):
#             tt = _parse_tensor_type(f"tensor<{tm.group(1)}>")
#             if tt:
#                 result_types.append(tt)
#         return (operand_types, result_types)

#     # single type form
#     result_types: List[_TensorType] = []
#     for tm in _RE_TENSOR.finditer(sig):
#         tt = _parse_tensor_type(f"tensor<{tm.group(1)}>")
#         if tt:
#             result_types.append(tt)
#     return ([], result_types)


# def _parse_op(stmt: str, index: int, line_no: int) -> Optional[_OpRec]:
#     mname = _RE_OPNAME.search(stmt)
#     if not mname:
#         return None
#     opname = mname.group(1)
#     if opname.startswith('"') and opname.endswith('"'):
#         opname = opname[1:-1]

#     # results
#     res_ids: List[str] = []
#     mres = _RE_RESIDS.search(stmt)
#     if mres:
#         res_blob = mres.group(1)
#         res_ids = [x.strip() for x in res_blob.split(",") if x.strip()]

#     # operands (SSA ids in the first (...) after opname, best-effort)
#     operand_ids: List[str] = []
#     # Find operand list by locating opname and then taking the nearest "(...)"
#     pos = stmt.find(opname)
#     sub = stmt[pos:] if pos >= 0 else stmt
#     mop = _RE_OPERANDS.search(sub)
#     if mop:
#         inside = mop.group(1)
#         operand_ids = _RE_SSA.findall(inside)

#     attrs = _parse_attr_dict(stmt)
#     operand_types, result_types = _parse_types(stmt)

#     return _OpRec(
#         index=index,
#         opname=opname,
#         res_ids=res_ids,
#         operand_ids=operand_ids,
#         attrs=attrs,
#         operand_types=operand_types,
#         result_types=result_types,
#         line_no=line_no,
#         text=stmt,
#     )


# # ----------------------------- Graph & layer extraction -----------------------------


# _TOSA_TO_CANON = {
#     # "Convolution ops"
#     "tosa.conv2d": "Conv",
#     "tosa.depthwise_conv2d": "Conv",       # group/depthwise handled
#     "tosa.transpose_conv2d": "ConvTranspose",

#     # "Linear ops"
#     "tosa.fully_connected": "MatMul",      # treated as Linear
#     "tosa.matmul": "MatMul",

#     # elementwise
#     "tosa.add": "Add",
#     "tosa.sub": "Sub",
#     "tosa.bitwise_and": "BitwiseAnd",
#     "tosa.bitwise_or": "BitwiseOr",
#     "tosa.bitwise_xor": "BitwiseXor",

#     # pooling
#     "tosa.max_pool2d": "MaxPool",
#     "tosa.avg_pool2d": "AveragePool",

#     # activations / unary
#     "tosa.relu": "Relu",
#     "tosa.abs": "Abs",

#     # clamp/min/max (used for wide output detection, not necessarily emitted)
#     "tosa.clamp": "Clamp",
#     "tosa.minimum": "Min",
#     "tosa.maximum": "Max",

#     # concat (optional support)
#     "tosa.concat": "Concat",
# }


# _CONV_LIKE = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
# _EMITTABLE = set(_CONV_LIKE) | {
#     "Add", "Sub", "BitwiseOr", "BitwiseXor", "BitwiseAnd",
#     "MaxPool", "AveragePool",
#     "Abs", "Relu",
#     "Concat",
# }

# # Ops that are structural and should be ignored in YAML (but kept for shape flow)
# _IGNORE_ALWAYS = {
#     "tosa.const",
#     "arith.constant",
#     "torch.constant",
#     "tosa.reshape",
#     "tosa.transpose",
#     "tosa.cast",
#     "tosa.rescale",
#     "tosa.identity",
# }


# def _canonical_name(s: str) -> str:
#     # mimic the reference script
#     separator = s.rfind("_MatMul_1")
#     if separator > 0:
#         s = s[:separator]
#     separator = s.rfind(".")
#     if separator > 0:
#         if s[separator + 1 :] == "op":
#             s = s[:separator]
#     return f"layer_{s}" if s.isnumeric() else s


# def _product(xs: Iterable[int]) -> int:
#     p = 1
#     for x in xs:
#         if x <= 0:
#             continue
#         p *= x
#     return p


# def _guess_channels(t: _TensorType) -> int:
#     """
#     Heuristic channel-count extraction.

#     Handles common cases:
#       - NCHW: [N, C, H, W]
#       - NHWC: [N, H, W, C]
#       - CHW or HWC: [C, H, W] or [H, W, C]
#       - 1D / 2D vectors: use first dim (feature count) after batch removal
#     """
#     shp = t.shape
#     if not shp:
#         return 0
#     if len(shp) == 4:
#         n, a, b, c = shp
#         # If last dim is small-ish and spatial dims look plausible, assume NHWC
#         if c != -1 and c <= 4096 and (a == b or (a > 1 and b > 1)):
#             return c
#         # else assume NCHW
#         return a if a != -1 else 0
#     if len(shp) == 3:
#         a, b, c = shp
#         # If last dim is small-ish and first two look spatial, assume HWC
#         if c != -1 and c <= 4096 and (a == b or (a > 1 and b > 1)):
#             return c
#         return a if a != -1 else 0
#     if len(shp) == 2:
#         # [N, F] or [F, ?]
#         return shp[-1] if shp[-1] != -1 else (shp[0] if shp[0] != -1 else 0)
#     if len(shp) == 1:
#         return shp[0] if shp[0] != -1 else 0
#     # fallback: best guess is last dim
#     return shp[-1] if shp[-1] != -1 else 0


# def _strip_batch(shape: List[int]) -> List[int]:
#     if len(shape) >= 2 and shape[0] in (1, -1):
#         return shape[1:]
#     return shape


# def _format_shape_comment(shape: List[int]) -> str:
#     return str(tuple(shape))


# def _dims_pixel_count(shape_wo_batch: List[int], channel_is_first: bool) -> int:
#     # pixel count = product of spatial dims (exclude channel dim)
#     if len(shape_wo_batch) < 2:
#         return 0
#     if len(shape_wo_batch) == 3:
#         if channel_is_first:
#             # C,H,W
#             return _product(shape_wo_batch[1:])
#         # H,W,C
#         return _product(shape_wo_batch[:2])
#     if len(shape_wo_batch) == 2:
#         # treat as 1D spatial
#         return shape_wo_batch[-1] if shape_wo_batch[-1] > 0 else 0
#     return _product(shape_wo_batch[1:])


# # ----------------------------- YAML layer model (mirrors reference) -----------------------------


# def allocate_offset(_layer_name: str, _processor_map: int, prev_offset: int, device_id: int) -> int:
#     """
#     Ping-pong allocator identical in spirit to the reference code.
#     """
#     half = _half_data_for_device(device_id)
#     return half if prev_offset == 0 else 0


# # pylint: disable=too-many-branches, too-many-statements, too-many-locals
# def emit_ai8x_yaml_from_tosa_mlir(
#     *,
#     tosa_mlir_path: Path,
#     dataset: str,
#     arch: str,
#     device_name: str,
#     hwc: bool = False,
#     use_fifos: bool = False,
#     move_l0: bool = False,
#     filename: Path,
#     verbose: bool = True,
#     # Optional override: map canonical layer names to weight_bits (1/2/4/8)
#     weight_bits_by_layer: Optional[Dict[str, int]] = None,
# ) -> None:
#     """
#     Core YAML emitter that matches the reference script’s functionality/structure,
#     but derives graph information from the TOSA MLIR SSA dataflow.
#     """
#     device_id = _device_id_from_name(device_name)
#     MAX_PIXELS = _max_pixels_for_device(device_id)
#     MAX_PROC = 64

#     # Filter warnings in a similar spirit to the reference script
#     warnings.filterwarnings(action="ignore")

#     text_lines = tosa_mlir_path.read_text(encoding="utf-8").splitlines()
#     stmts = _collect_tosa_statements(text_lines)

#     ops_all: List[_OpRec] = []
#     for idx, (ln, st) in enumerate(stmts):
#         rec = _parse_op(st, idx, ln)
#         if rec:
#             ops_all.append(rec)

#     if not ops_all:
#         raise CompilationError(f"No TOSA ops found in MLIR: {tosa_mlir_path}")

#     # Build SSA type map and producer/consumer maps
#     ssa_type: Dict[str, _TensorType] = {}
#     producer: Dict[str, str] = {}              # value -> op_name (layer name later)
#     producer_opidx: Dict[str, int] = {}        # value -> op index
#     consumers: Dict[str, List[int]] = {}       # value -> list of op indices consuming it

#     for op in ops_all:
#         # record result types (best-effort align: res_ids vs result_types)
#         for i, rid in enumerate(op.res_ids):
#             if i < len(op.result_types):
#                 ssa_type[rid] = op.result_types[i]
#             producer[rid] = f"op{op.index}_{op.opname}"
#             producer_opidx[rid] = op.index

#     for op in ops_all:
#         for oid in op.operand_ids:
#             consumers.setdefault(oid, []).append(op.index)

#     # Helper: is this op ignored or emittable?
#     def to_canon(opname: str) -> Optional[str]:
#         return _TOSA_TO_CANON.get(opname)

#     def ignore_layer(op: _OpRec) -> bool:
#         if op.opname in _IGNORE_ALWAYS:
#             return True
#         c = to_canon(op.opname)
#         if c is None:
#             # unknown TOSA op: treat as ignorable if it looks structural; else not
#             # (so we fail early rather than silently producing nonsense)
#             structural = any(k in op.opname for k in ("reshape", "transpose", "cast", "rescale"))
#             return structural
#         # Remove non-emittable canon ops
#         return c not in _EMITTABLE

#     # Build a filtered ordered op list (these are "layers" before fusion)
#     keep_ops: List[_OpRec] = [op for op in ops_all if not ignore_layer(op)]
#     if not keep_ops:
#         raise CompilationError("After filtering, no emittable ops remain for ai8x YAML emission.")

#     # Determine final layer (last kept)
#     final_layer_op = keep_ops[-1]

#     # 1) Mark "wide" output where relevant (reference traced -32768 constants to Min/Clip)
#     # Best-effort for TOSA:
#     #   If a tosa.clamp has min_int = -32768 (or min_fp = -32768), mark its upstream conv/linear.
#     wide_producers: set[int] = set()
#     for op in ops_all:
#         canon = to_canon(op.opname)
#         if canon not in ("Clamp", "Min"):
#             continue
#         # check attrs for -32768
#         found = False
#         for k, v in op.attrs.items():
#             if isinstance(v, int) and v == -32768:
#                 found = True
#             if isinstance(v, str) and "-32768" in v:
#                 found = True
#         if not found:
#             continue
#         # trace inputs back to nearest conv-like producer
#         work = list(op.operand_ids)
#         seen: set[str] = set()
#         while work:
#             val = work.pop()
#             if val in seen:
#                 continue
#             seen.add(val)
#             pidx = producer_opidx.get(val)
#             if pidx is None:
#                 continue
#             pop = ops_all[pidx]
#             pc = to_canon(pop.opname)
#             if pc in _CONV_LIKE:
#                 wide_producers.add(pidx)
#                 continue
#             # otherwise continue tracing upstream
#             work.extend(pop.operand_ids)

#     # 2) Collect shapes for SSA ids (drop batch)
#     shapes: Dict[str, Tuple[int, ...]] = {}
#     for vid, tt in ssa_type.items():
#         shp = _strip_batch(tt.shape)
#         if shp and all(isinstance(x, int) for x in shp):
#             shapes[vid] = tuple(shp)

#     # 3) Build input/output boundary maps at the layer level
#     # Identify, for each kept op, which operand SSA values come from outside kept set,
#     # and which result SSA values are consumed outside kept set.
#     kept_indices = {op.index for op in keep_ops}

#     # Map SSA output -> layer name
#     output_layer: Dict[str, str] = {}
#     input_layer: Dict[str, str] = {}

#     def layer_name(op: _OpRec) -> str:
#         # use a stable, readable name (derived from SSA producer tag)
#         # prefer op%index_opname
#         return f"{op.index}_{op.opname.replace('.', '_')}"

#     # Build a name mapping
#     opid_to_name: Dict[int, str] = {op.index: layer_name(op) for op in keep_ops}

#     # Create provisional IO maps
#     for op in keep_ops:
#         lname = opid_to_name[op.index]
#         for oid in op.operand_ids:
#             if oid:
#                 input_layer[oid] = lname
#         for rid in op.res_ids:
#             if rid:
#                 output_layer[rid] = lname

#     # Compute per-layer external inputs and external outputs
#     inputs: Dict[str, List[str]] = {}
#     outputs: Dict[str, List[str]] = {}

#     for op in keep_ops:
#         lname = opid_to_name[op.index]
#         inputs[lname] = []
#         outputs[lname] = []

#         # External inputs: operands whose producer is not a kept op
#         for oid in op.operand_ids:
#             if not oid:
#                 continue
#             pidx = producer_opidx.get(oid, None)
#             if pidx is None or pidx not in kept_indices:
#                 inputs[lname].append(oid)

#         # External outputs: results whose consumers include a non-kept op (or none)
#         for rid in op.res_ids:
#             if not rid:
#                 continue
#             cons = consumers.get(rid, [])
#             if not cons:
#                 outputs[lname].append(rid)
#                 continue
#             if any(cidx not in kept_indices for cidx in cons):
#                 outputs[lname].append(rid)

#     # If a layer has zero external outputs (fully internal), keep at least its first result
#     for op in keep_ops:
#         lname = opid_to_name[op.index]
#         if not outputs[lname] and op.res_ids:
#             outputs[lname] = [op.res_ids[0]]

#     # 4) Build initial "layers" dict matching reference fields
#     # This is the "all_ops" in the reference, but already filtered.
#     layers: "OrderedDict[str, Dict[str, Any]]" = {}  # type: ignore[assignment]
#     layers = dict()  # will convert to OrderedDict-like behavior by insertion order

#     prev_op_name = ""
#     input_hwc = False
#     input_processors = 0

#     for op in keep_ops:
#         lname = opid_to_name[op.index]
#         canon = to_canon(op.opname)
#         if canon is None:
#             continue

#         this_layer: Dict[str, Any] = {}
#         this_layer["name"] = lname

#         # Mark output layers (final always output)
#         if lname != opid_to_name[final_layer_op.index]:
#             # mimic reference: output=true if any output isn't consumed by a later layer
#             # Here we approximate: if it has any external outputs, mark as output
#             if outputs.get(lname):
#                 this_layer["output"] = "true"

#         this_layer["op"] = main_op = "Passthrough"
#         operands = 1

#         # Identify "layer inputs" as tensor SSA ids (external only)
#         ins = inputs.get(lname, []).copy()

#         # ------------------------- operator mapping -------------------------
#         if canon in ("Add", "Sub", "BitwiseOr", "BitwiseXor", "BitwiseAnd"):
#             operands = max(1, len(ins))
#             # ai8x YAML uses eltwise: Add/Sub/Or/Xor/And
#             elt = canon[7:] if canon.startswith("Bitwise") else canon
#             this_layer["eltwise"] = elt
#             this_layer["operands"] = operands

#         elif canon in ("MatMul", "Gemm"):
#             this_layer["op"] = main_op = "Linear"
#             this_layer["activate"] = "None"

#         elif canon in ("Conv", "ConvTranspose"):
#             # Determine dimensionality: only 2d supported here
#             if canon == "ConvTranspose":
#                 this_layer["op"] = "ConvTranspose2d"
#                 main_op = "ConvTranspose"
#             else:
#                 this_layer["op"] = "Conv2d"
#                 main_op = "Conv"
#             this_layer["activate"] = "None"

#         elif canon in ("MaxPool", "AveragePool"):
#             # pool ops become Passthrough with pool fields
#             # kernel/stride/dilation/pad are in attrs; best-effort
#             k = op.attrs.get("kernel") or op.attrs.get("kernel_shape") or op.attrs.get("ksize")
#             s = op.attrs.get("stride") or op.attrs.get("strides")
#             d = op.attrs.get("dilation") or op.attrs.get("dilations")

#             # Attempt to parse k
#             shape = None
#             if isinstance(k, list) and k:
#                 shape = k
#             elif isinstance(k, str):
#                 nums = [int(x) for x in re.findall(r"-?\d+", k)]
#                 if nums:
#                     shape = nums
#             if shape is None:
#                 shape = [2, 2]

#             if len(shape) == 1 or (len(shape) >= 2 and shape[0] == shape[1]):
#                 pool_shape = shape[0]
#             else:
#                 pool_shape = shape[:2]

#             if canon == "MaxPool":
#                 this_layer["max_pool"] = pool_shape
#             else:
#                 this_layer["avg_pool"] = pool_shape

#             # stride
#             stride_val = 2
#             if isinstance(s, list) and s:
#                 stride_val = int(s[0])
#             elif isinstance(s, str):
#                 nums = [int(x) for x in re.findall(r"-?\d+", s)]
#                 if nums:
#                     stride_val = nums[0]
#             this_layer["pool_stride"] = stride_val

#             # dilation
#             if d is not None:
#                 if isinstance(d, list):
#                     this_layer["pool_dilation"] = d
#                 elif isinstance(d, str):
#                     nums = [int(x) for x in re.findall(r"-?\d+", d)]
#                     if nums:
#                         this_layer["pool_dilation"] = nums

#         elif canon in ("Abs", "Relu"):
#             this_layer["activate"] = canon

#         elif canon == "Concat":
#             # Concat isn't in the reference script's explicit ignore list; treat as passthrough
#             # and let processor logic treat it as "operands>1".
#             operands = max(1, len(ins))
#             this_layer["eltwise"] = "Concat"
#             this_layer["operands"] = operands

#         else:
#             this_layer["op"] = f"Unknown ({canon})"

#         this_layer["main_op"] = main_op

#         # ------------------------- conv/linear extra fields -------------------------
#         # Extract weights/biases from operands when present (TOSA conv2d/fc conventions)
#         # TOSA conv2d: (input, weight, bias)
#         # TOSA depthwise_conv2d: (input, weight, bias)
#         # TOSA fully_connected: (input, weight, bias)
#         have_bias = False
#         weight_count = 0

#         # Helper to get operand tensor types aligned by position (best-effort)
#         opnd_types = op.operand_types
#         opnd_ids = op.operand_ids

#         def _get_operand_type(i: int) -> Optional[_TensorType]:
#             return opnd_types[i] if i < len(opnd_types) else None

#         def _get_operand_id(i: int) -> Optional[str]:
#             return opnd_ids[i] if i < len(opnd_ids) else None

#         if main_op in ("Conv", "ConvTranspose") or this_layer.get("op") == "Linear":
#             # try to identify weight operand as position 1
#             w_id = _get_operand_id(1)
#             w_tt = _get_operand_type(1)
#             b_id = _get_operand_id(2)
#             b_tt = _get_operand_type(2)

#             # Remove weights/biases from "inputs" list for this op (like reference)
#             # Note: inputs[lname] holds "external" operands. weights/biases are internal consts
#             # usually, so often not present here. But keep logic anyway.
#             # nothing to remove from ins (external) typically.

#             # Determine weight_count and kernel_size if conv-like
#             if w_tt is not None and w_tt.shape:
#                 weight_count = _product([d for d in w_tt.shape if d > 0])

#             if main_op in ("Conv", "ConvTranspose"):
#                 # kernel from weight tensor (OHWI for conv2d; for transpose_conv2d, likely IHWO)
#                 kernel_h = kernel_w = None
#                 if w_tt is not None and len(w_tt.shape) >= 4:
#                     # try common patterns:
#                     # OHWI: [O, KH, KW, I]
#                     # IHWO: [I, KH, KW, O]
#                     # If second/third dims are plausible (<=11), treat as kernel
#                     kh, kw = w_tt.shape[1], w_tt.shape[2]
#                     if 1 <= kh <= 31 and 1 <= kw <= 31:
#                         kernel_h, kernel_w = kh, kw

#                 # attrs may include 'pad' and 'group'
#                 pad = op.attrs.get("pad") or op.attrs.get("padding") or op.attrs.get("pads")
#                 pad_val = 0
#                 if isinstance(pad, list) and pad:
#                     pad_val = int(pad[0])
#                 elif isinstance(pad, str):
#                     nums = [int(x) for x in re.findall(r"-?\d+", pad)]
#                     if nums:
#                         pad_val = nums[0]
#                 if kernel_h is not None and kernel_w is not None:
#                     if kernel_h == kernel_w:
#                         this_layer["kernel_size"] = str(kernel_h)
#                     else:
#                         this_layer["kernel_size"] = f"{kernel_h}x{kernel_w}"
#                 else:
#                     # fallback
#                     this_layer["kernel_size"] = "3x3"
#                 this_layer["pad"] = pad_val

#                 # groups/depthwise
#                 group = op.attrs.get("group") or op.attrs.get("groups")
#                 if isinstance(group, int) and group != 1:
#                     this_layer["groups"] = group
#                 elif isinstance(group, str):
#                     nums = [int(x) for x in re.findall(r"-?\d+", group)]
#                     if nums and nums[0] != 1:
#                         this_layer["groups"] = nums[0]

#             # bias presence
#             if b_id is not None and b_id != "" and b_tt is not None:
#                 have_bias = True

#             # quantization bits
#             quantization = 8
#             if weight_bits_by_layer:
#                 # try both raw and canonicalized keys
#                 quantization = int(weight_bits_by_layer.get(lname, weight_bits_by_layer.get(_canonical_name(lname), 8)))
#             if quantization not in (0, 8):
#                 if quantization not in (1, 2, 4):
#                     raise CompilationError(f"Invalid quantization bits for layer {lname}: {quantization}")
#                 this_layer["quantization"] = quantization

#         # conv-like book-keeping (weight_count etc.)
#         if main_op in ("Conv", "ConvTranspose") or this_layer.get("op") == "Linear":
#             if op.index in wide_producers:
#                 this_layer["output_width"] = 32
#             this_layer["have_bias"] = have_bias
#             this_layer["weight_count"] = int(weight_count)

#         # ------------------------- flatten / in_dim -------------------------
#         flatten = False
#         if this_layer.get("op") == "Linear":
#             # Use operand type 0 if present
#             t0 = _get_operand_type(0)
#             if t0 is not None and t0.shape:
#                 shp = _strip_batch(t0.shape)
#                 if len(shp) > 1:
#                     mult = _product([d for d in shp if d > 0])
#                     if shp[0] != mult:
#                         flatten = True
#         if flatten:
#             this_layer["flatten"] = "true"

#         # in_dim: detect dimension changes when not flattening (best-effort)
#         if not flatten:
#             # Attempt to infer prev_dim from first "original input" type (operand 0)
#             t0 = _get_operand_type(0)
#             if t0 is not None and t0.shape:
#                 prev_dim = _strip_batch(t0.shape)
#                 # For "external" input tensors, we don't always have types; choose prev_dim anyway
#                 # Only emit in_dim when it looks non-trivial
#                 if len(prev_dim) > 0 and any(d != 1 for d in prev_dim):
#                     this_layer["in_dim"] = list(prev_dim)

#         # ------------------------- processor counts & sequences -------------------------
#         processors = 0
#         for vid in inputs.get(lname, []):
#             if vid in ssa_type:
#                 processors += _guess_channels(ssa_type[vid])
#         if operands > 1:
#             processors //= operands
#         this_layer["proc_count"] = max(1, processors)

#         # Inner layers and more than 16 channels are always HWC (reference logic)
#         hwc = hwc or (processors > 16) or (prev_op_name != "")

#         if prev_op_name == "":
#             this_layer["data_format"] = "HWC" if hwc else "CHW"
#             input_hwc = hwc
#             input_processors = this_layer["proc_count"]
#         else:
#             # Determine in_sequences: source LAYER names for each external input
#             seq: List[str] = []
#             for vid in inputs.get(lname, []):
#                 pidx = producer_opidx.get(vid)
#                 if pidx is not None and pidx in kept_indices:
#                     seq.append(opid_to_name[pidx])
#             # If multiple or non-strictly-sequential, emit in_sequences
#             if seq:
#                 if operands > 1:
#                     seq.reverse()  # match reference reversal for eltwise
#                 if len(seq) > 1 or (len(seq) == 1 and seq[0] != prev_op_name):
#                     this_layer["in_sequences"] = [opid_to_name[producer_opidx[v]] for v in inputs.get(lname, [])
#                                                   if v in producer_opidx and producer_opidx[v] in kept_indices]

#         prev_op_name = lname
#         layers[lname] = this_layer

#     # ------------------------- Fusion passes (7a/7b/7c) -------------------------

#     # Create ordered view for iteration
#     layer_items: List[Tuple[str, Dict[str, Any]]] = list(layers.items())

#     # Build all_outputs mapping (layer -> consuming layers) from in_sequences
#     all_inputs: Dict[str, List[str]] = {}
#     all_outputs: Dict[str, List[str]] = {}
#     prev = ""
#     for name, ll in layer_items:
#         if "in_sequences" not in ll:
#             srcs = [prev] if prev else []
#         else:
#             srcs = list(ll["in_sequences"])
#         if srcs:
#             all_inputs[name] = srcs
#             for s in srcs:
#                 all_outputs.setdefault(s, []).append(name)
#         prev = name

#     def _veto_intermediate_use(prev_name: str, act_name: str) -> bool:
#         # Veto if any other layer uses prev_name as input besides act_name
#         for other_name, ol in layer_items:
#             if other_name in (act_name, prev_name):
#                 continue
#             ins = ol.get("in_sequences")
#             if ins and prev_name in ins:
#                 return True
#         return False

#     # 7a - fuse activation into previous conv/linear
#     new_layers: "OrderedDict[str, Dict[str, Any]]" = {}  # type: ignore[assignment]
#     new_layers = dict()
#     pop_list: List[Tuple[str, str]] = []
#     prev_name = ""
#     for name, ll in layer_items:
#         if ll.get("main_op") == "Passthrough" and "activate" in ll and prev_name:
#             prev_ll = layers[prev_name]
#             if prev_ll.get("main_op") not in ("Conv", "ConvTranspose", "Linear"):
#                 prev_name = name
#                 new_layers[name] = ll
#                 continue
#             if _veto_intermediate_use(prev_name, name):
#                 prev_name = name
#                 new_layers[name] = ll
#                 continue
#             # fuse
#             prev_ll["comment"] = f"{prev_name} fused with {name}" if "comment" not in prev_ll else prev_ll["comment"] + f" and {name}"
#             prev_ll["activate"] = ll["activate"]
#             if "output" in ll:
#                 prev_ll["output"] = ll["output"]
#             if "quantization" in ll:
#                 prev_ll["quantization"] = ll["quantization"]
#             if "output_width" in ll:
#                 prev_ll["output_width"] = ll["output_width"]
#             pop_list.append((prev_name, name))
#             # skip adding ll (activation)
#             prev_name = name
#             continue

#         new_layers[name] = ll
#         prev_name = name

#     # apply removals
#     for prev_name, act_name in pop_list:
#         # redirect in_sequences references from act_name to prev_name
#         for other_name, ol in new_layers.items():
#             if other_name in (act_name, prev_name):
#                 continue
#             if "in_sequences" in ol:
#                 ol["in_sequences"] = [prev_name if x == act_name else x for x in ol["in_sequences"]]
#         # delete activation
#         if act_name in new_layers:
#             del new_layers[act_name]

#     layers = new_layers
#     layer_items = list(layers.items())

#     # rebuild all_inputs/all_outputs for later fusions
#     all_inputs = {}
#     all_outputs = {}
#     prev = ""
#     for name, ll in layer_items:
#         if "in_sequences" not in ll:
#             srcs = [prev] if prev else []
#         else:
#             srcs = list(ll["in_sequences"])
#         if srcs:
#             all_inputs[name] = srcs
#             for s in srcs:
#                 all_outputs.setdefault(s, []).append(name)
#         prev = name

#     # 7b - fuse pooling into following conv/linear (pool layer preceding)
#     pop_list = []
#     prev_name = ""
#     for name, ll in layer_items:
#         if ll.get("main_op") in ("Conv", "ConvTranspose", "Linear") and prev_name:
#             prev_ll = layers[prev_name]
#             if prev_ll.get("main_op") != "Passthrough" or (("max_pool" not in prev_ll) and ("avg_pool" not in prev_ll)):
#                 prev_name = name
#                 continue
#             if _veto_intermediate_use(prev_name, name):
#                 prev_name = name
#                 continue
#             # fuse pool -> ll
#             ll["comment"] = f"{prev_name} fused with {name}" if "comment" not in ll else f"{prev_name} and {ll['comment']}"
#             if "in_sequences" in prev_ll:
#                 ll["in_sequences"] = prev_ll["in_sequences"]
#             for k in ("avg_pool", "max_pool", "pool_stride", "pool_dilation", "data_format"):
#                 if k in prev_ll:
#                     ll[k] = prev_ll[k]
#             pop_list.append((prev_name, name))
#         prev_name = name

#     for pool_name, _ in pop_list:
#         if pool_name in layers:
#             del layers[pool_name]

#     layer_items = list(layers.items())

#     # rebuild all_inputs/all_outputs again
#     all_inputs = {}
#     all_outputs = {}
#     prev = ""
#     for name, ll in layer_items:
#         if "in_sequences" not in ll:
#             srcs = [prev] if prev else []
#         else:
#             srcs = list(ll["in_sequences"])
#         if srcs:
#             all_inputs[name] = srcs
#             for s in srcs:
#                 all_outputs.setdefault(s, []).append(name)
#         prev = name

#     # 7c - fuse eltwise into following conv (eltwise layer preceding)
#     pop_list = []
#     prev_name = ""
#     for name, ll in layer_items:
#         if ll.get("main_op") in ("Conv", "ConvTranspose") and prev_name:
#             prev_ll = layers[prev_name]
#             pool_count = int("max_pool" in prev_ll) + int("avg_pool" in prev_ll) + int("max_pool" in ll) + int("avg_pool" in ll)
#             if _veto_intermediate_use(prev_name, name):
#                 prev_name = name
#                 continue
#             if ("in_sequences" in ll) or ("in_dim" in ll) or ("flatten" in ll):
#                 prev_name = name
#                 continue
#             if prev_ll.get("main_op") != "Passthrough" or prev_ll.get("operands", 1) == 1 or pool_count > 1:
#                 prev_name = name
#                 continue
#             # MAX78002 special case from reference: avoid eltwise + bias + multipass
#             if device_id == 87 and ll.get("have_bias") and prev_ll.get("proc_count", 0) > MAX_PROC:
#                 prev_name = name
#                 continue

#             ll["comment"] = f"{prev_name} fused with {name}" if "comment" not in ll else f"{prev_name} and {ll['comment']}"
#             if "in_sequences" in prev_ll:
#                 ll["in_sequences"] = prev_ll["in_sequences"]
#             for k in ("max_pool", "avg_pool", "pool_stride", "pool_dilation", "data_format"):
#                 if k in prev_ll:
#                     ll[k] = prev_ll[k]
#                     ll["pool_first"] = "true"
#             if ("max_pool" in ll) or ("avg_pool" in ll):
#                 ll["pool_first"] = "false"
#             ll["eltwise"] = prev_ll["eltwise"]
#             ll["operands"] = prev_ll["operands"]
#             pop_list.append((prev_name, name))
#         prev_name = name

#     for elt_name, _ in pop_list:
#         if elt_name in layers:
#             del layers[elt_name]

#     layer_items = list(layers.items())

#     # ------------------------- write_gap insertion (8) -------------------------
#     # This is copied in spirit from the reference; it matters for multi-input eltwise.

#     write_gap_list: List[Tuple[str, int]] = []
#     insert_list: List[Tuple[str, int]] = []
#     source_list: List[Tuple[str, str]] = []

#     # rebuild all_inputs in terms of layer names
#     all_inputs = {}
#     all_outputs = {}
#     prev = ""
#     for name, ll in layer_items:
#         if "in_sequences" not in ll:
#             srcs = [prev] if prev else []
#         else:
#             srcs = list(ll["in_sequences"])
#         if srcs:
#             all_inputs[name] = srcs
#             for s in srcs:
#                 all_outputs.setdefault(s, []).append(name)
#         prev = name

#     prev_name = ""
#     for name, ll in layer_items:
#         sources = ll.get("in_sequences")
#         if not sources:
#             if not prev_name:
#                 prev_name = name
#                 continue
#             sources = [prev_name]

#         operands = int(ll["operands"]) if "operands" in ll else len(sources)
#         if operands < 2:
#             prev_name = name
#             continue

#         for source in sources:
#             must_insert = False
#             prev_inner = ""
#             for other_name, ol in layer_items:
#                 if other_name not in (name, source):
#                     if "in_sequences" in ol and source in ol["in_sequences"]:
#                         must_insert = True
#                     elif prev_inner == source:
#                         must_insert = True
#                         ol["in_sequences"] = [source]
#                 prev_inner = other_name

#             if not must_insert:
#                 write_gap_list.append((source, operands))
#             else:
#                 insert_list.append((source, operands))
#                 source_list.append((name, source))

#         prev_name = name

#     for name, operands in write_gap_list:
#         if name in layers:
#             layers[name]["write_gap"] = operands - 1

#     for name, source in source_list:
#         if name in layers and "in_sequences" in layers[name]:
#             seq = layers[name]["in_sequences"]
#             layers[name]["in_sequences"] = ["gap_" + source if s == source else s for s in seq]

#     # Insert actual gap layers
#     for source, operands in insert_list:
#         new_name = "gap_" + source
#         if new_name in layers:
#             continue
#         new_layer: Dict[str, Any] = {}
#         new_layer["name"] = new_name
#         new_layer["proc_count"] = max(1, layers[source].get("proc_count", 1))
#         new_layer["op"] = new_layer["main_op"] = "Passthrough"
#         new_layer["write_gap"] = operands - 1

#         # Insert into ordered dict right after 'source'
#         keys = list(layers.keys())
#         insert_pos = keys.index(source) + 1
#         items2 = list(layers.items())
#         items2.insert(insert_pos, (new_name, new_layer))
#         layers = dict(items2)

#     layer_items = list(layers.items())

#     # ------------------------- processor allocation (9/10) -------------------------

#     # rebuild all_inputs/all_outputs after insertions
#     all_inputs = {}
#     all_outputs = {}
#     prev = ""
#     for name, ll in layer_items:
#         if "in_sequences" not in ll:
#             srcs = [prev] if prev else []
#         else:
#             srcs = list(ll["in_sequences"])
#         if srcs:
#             all_inputs[name] = srcs
#             for s in srcs:
#                 all_outputs.setdefault(s, []).append(name)
#         prev = name

#         # When there are multiple in_sequences: either eltwise (operands>1) or multi-input
#         # (concat/multipass). Apply reference behavior.
#         if "in_sequences" not in ll or len(ll["in_sequences"]) < 2:
#             continue
#         operands = int(ll["operands"]) if "operands" in ll else 1
#         if operands == 1:
#             operands = len(ll["in_sequences"])
#             # Concat instead of interleave for small channel counts
#             if ll.get("proc_count", 0) <= MAX_PROC:
#                 ll["concat"] = True
#                 for i, ie in enumerate(ll["in_sequences"]):
#                     lin = layers.get(ie, {})
#                     lin["concat_source"] = max(i, lin.get("concat_source", i))
#                     layers[ie] = lin
#             else:
#                 ll["proc_count"] = max(1, ll.get("proc_count", 1) // operands)

#     # Mark size of concat outputs
#     for name, ll in layer_items:
#         if "concat_source" not in ll:
#             continue
#         # best-effort: cannot reliably derive output tensor shape per layer here without full SSA map
#         # If present, keep as a placeholder
#         ll["concat_shape"] = ll.get("proc_count", 1)

#     def calculate_processors(proc_count: int) -> Tuple[int, int]:
#         multipass = 1
#         if proc_count > MAX_PROC:
#             multipass = (proc_count + MAX_PROC - 1) // MAX_PROC
#             proc_count = (proc_count + multipass - 1) // multipass
#             rem = proc_count % 4
#             if rem != 0:
#                 proc_count += (4 - rem)
#         return proc_count, multipass

#     cost_list: List[Tuple[int, int, str]] = []

#     for name, ll in layer_items:
#         hwc_flag = True
#         if "data_format" in ll:
#             hwc_flag = (ll["data_format"] == "HWC")

#         proc_count = int(ll.get("proc_count", 1))
#         multipass = 1

#         if hwc_flag:
#             proc_used, multipass = calculate_processors(proc_count)
#             if proc_used > MAX_PROC:
#                 proc_used = MAX_PROC
#         else:
#             proc_used = min(proc_count, MAX_PROC // 4)

#         ll["proc_used"] = proc_used
#         ll["multipass"] = multipass

#         weights_per_processor = 0
#         if ll.get("main_op") in ("Conv", "ConvTranspose", "Linear"):
#             wc = int(ll.get("weight_count", 0))
#             if proc_used > 0:
#                 weights_per_processor = wc // proc_used
#             if "quantization" in ll:
#                 q = int(ll["quantization"])
#                 if q in (1, 2, 4):
#                     weights_per_processor //= (8 // q)
#         ll["weight_cost"] = weights_per_processor

#         if proc_used <= 60:
#             cost_list.append((weights_per_processor, proc_used, name))
#         else:
#             cost_list.append((0, proc_used, name))

#     if not cost_list:
#         raise CompilationError("No layers available for processor allocation.")

#     # bucket grouping (fanout constraints)
#     bucket_groups: List[List[Tuple[int, int, str]]] = [[cost_list[0]]]
#     bucket: Dict[str, int] = {cost_list[0][2]: 0}

#     for name, _ll in layer_items:
#         if name not in all_outputs:
#             continue
#         merge_buckets: List[int] = []
#         for i, (_w, _p, n) in enumerate(cost_list):
#             if n in all_outputs[name]:
#                 if n in bucket:
#                     merge_buckets.append(bucket[n])
#                 else:
#                     bn = len(bucket_groups)
#                     bucket[n] = bn
#                     merge_buckets.append(bn)
#                     bucket_groups.append([cost_list[i]])
#         if len(merge_buckets) > 1:
#             target = merge_buckets[0]
#             for bidx in merge_buckets[1:]:
#                 bucket_groups[target] += bucket_groups[bidx]
#                 bucket_groups[bidx] = []

#     group_list: List[Tuple[int, int, int, int, List[Tuple[int, int, str]]]] = []
#     for b in bucket_groups:
#         if not b:
#             continue
#         group_weights = 0
#         group_procs = -1
#         group_item: List[Tuple[int, int, str]] = []
#         min_shift = 0
#         last_processor = MAX_PROC - 1
#         for (wpp, procs, nm) in b:
#             group_weights += wpp
#             if "concat" in layers.get(nm, {}):
#                 # reference had extra asserts; we keep it permissive
#                 pass
#             group_procs = max(procs, group_procs)
#             group_item.append((wpp, procs, nm))
#         if group_procs >= 0:
#             group_list.append((group_weights, group_procs, min_shift, last_processor, group_item))

#     weights_used: List[int] = [0] * MAX_PROC

#     def allocate_processors(
#         weight_cost: int,
#         count: int,
#         min_shift: int,
#         data: List[Tuple[int, int, str]],
#         hwc_flag: bool = True,
#     ) -> Tuple[List[int], int]:
#         min_cost: Tuple[int, int] = (2**63 - 1, -1)
#         shifted_map: List[int] = [0] * len(data)
#         processor_map: List[int] = shifted_map

#         for shift in range(min_shift, MAX_PROC - count + 1, 4):
#             for d, (_item_cost, item_procs, _nm) in enumerate(data):
#                 if hwc_flag:
#                     shifted_map[d] = ((1 << item_procs) - 1) << shift
#                 else:
#                     mult = 16 if count <= 4 else 4
#                     shifted_map[d] = 0
#                     for i in range(item_procs):
#                         shifted_map[d] |= 1 << (mult * i)
#                     shifted_map[d] <<= shift

#             if weight_cost == 0:
#                 min_cost = (0, min_shift)
#                 processor_map = shifted_map.copy()
#                 break

#             cost = weights_used.copy()
#             for p in range(MAX_PROC):
#                 for d, (item_cost, _item_procs, _nm) in enumerate(data):
#                     if shifted_map[d] & (1 << p):
#                         cost[p] += item_cost
#             max_cost = max(cost)
#             if max_cost < min_cost[0]:
#                 min_cost = (max_cost, shift)
#                 processor_map = shifted_map.copy()

#         if weight_cost > 0:
#             for p in range(MAX_PROC):
#                 proc_used = False
#                 for d in range(len(data)):
#                     if processor_map[d] & (1 << p):
#                         proc_used = True
#                 if proc_used:
#                     weights_used[p] = min_cost[0]

#         return processor_map, min_cost[1]

#     # layer 0 handling
#     l0: Optional[Tuple[int, int, int, int, List[Tuple[int, int, str]]]] = None
#     if not move_l0 or (not input_hwc) or input_processors != 1 or use_fifos:
#         l0 = group_list[0]
#         group_list = group_list[1:]

#     group_list.sort(reverse=True)
#     if l0 is not None:
#         group_list.insert(0, l0)

#     # Allocate processors and propagate output_processors
#     for (weights_per_proc, procs, min_shift, _stop, data) in group_list:
#         ll0 = layers[data[0][2]]
#         hwc_flag = True
#         if "data_format" in ll0:
#             hwc_flag = (ll0["data_format"] == "HWC")

#         processor_map, _ = allocate_processors(weights_per_proc, procs, min_shift, data, hwc_flag=hwc_flag)

#         intersected_map = 0xFFFFFFFFFFFFFFFF
#         for i, (_wpp, _p, nm) in enumerate(data):
#             layers[nm]["processors"] = processor_map[i]
#             intersected_map &= processor_map[i]

#         shift_count = 0
#         while intersected_map & 1 == 0 and shift_count < 64:
#             shift_count += 1
#             intersected_map >>= 1

#         for i, (_wpp, _p, nm) in enumerate(data):
#             ll = layers[nm]
#             if nm not in all_inputs:
#                 continue
#             if "concat" not in ll:
#                 for seq in all_inputs[nm]:
#                     layers[seq]["output_processors"] = processor_map[i]
#             else:
#                 shift = shift_count
#                 for seq in all_inputs[nm]:
#                     item_procs = int(layers[seq].get("concat_shape", layers[seq].get("proc_used", 1)))
#                     layers[seq]["output_processors"] = ((1 << item_procs) - 1) << shift
#                     if item_procs % 4 != 0:
#                         item_procs += (4 - (item_procs % 4))
#                     shift += item_procs

#     # ------------------------- out_offset (11) -------------------------
#     out_offset = 0
#     for name, ll in layer_items:
#         proc_map = int(ll.get("processors", 0))
#         out_offset = allocate_offset(name, proc_map, out_offset, device_id=device_id)
#         ll["out_offset"] = out_offset

#     # ------------------------- sanity checks (12/13) -------------------------
#     # 12 - check processors match output_processors where applicable
#     for name, ll in layer_items:
#         if name not in all_inputs:
#             continue
#         if "concat" in ll:
#             proc_union = 0
#             for ie in all_inputs[name]:
#                 proc_union |= int(layers[ie].get("output_processors", 0))
#             if int(ll.get("processors", 0)) != proc_union:
#                 # warn rather than hard-fail (TOSA graph may contain layout helpers)
#                 pass
#         else:
#             for ie in all_inputs[name]:
#                 if int(ll.get("processors", 0)) != int(layers[ie].get("output_processors", ll.get("processors", 0))):
#                     # warn rather than hard-fail
#                     pass

#     # 13 - cleanup redundant output_processors
#     prev_name = ""
#     for name, ll in layer_items:
#         if prev_name:
#             prev_ll = layers[prev_name]
#             if "output_processors" in prev_ll and int(prev_ll["output_processors"]) == int(ll.get("processors", 0)):
#                 del prev_ll["output_processors"]
#         prev_name = name

#     # ------------------------- YAML write (14) -------------------------
#     filename.parent.mkdir(parents=True, exist_ok=True)

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(
#             "---\n"
#             "# YAML template -- requires manual editing, particularly with regard to out_offset "
#             "and processors\n"
#             f"# Generated for {device_name} with input format "
#             f"{'HWC' if input_hwc else 'CHW'}\n\n"
#             f"arch: {arch}\n"
#             f"dataset: {dataset}\n"
#             "\n"
#             "layers:"
#         )

#         prev_name = ""
#         for count, (name, ll) in enumerate(layer_items):
#             f.write("\n" f"  # Layer {count}\n" f"  - name: {_canonical_name(ll['name'])}")
#             if "comment" in ll:
#                 f.write(f"  # {ll['comment']}")
#             f.write("\n")

#             hwc_flag = True
#             if "data_format" in ll:
#                 hwc_flag = (ll["data_format"] == "HWC")

#             # Input shape comment (best-effort)
#             warn_dim = False
#             print_dim = verbose or (prev_name == "")
#             if print_dim:
#                 f.write("    # input shape: ")
#             i = 0
#             max_pixels = MAX_PIXELS if hwc_flag else 4 * MAX_PIXELS
#             # We cannot perfectly map per-layer tensor ids after fusion/insertion, so use proc_count info.
#             # If you want exact shapes per layer, extend this emitter to carry SSA ids through fusion.
#             if print_dim:
#                 f.write("(shape inference from SSA omitted in fused view)")
#             f.write("\n")

#             # data_format
#             if "data_format" in ll:
#                 f.write(f"    data_format: {'HWC' if hwc_flag else 'CHW'}\n")

#             # in_sequences
#             if "in_sequences" in ll:
#                 f.write("    in_sequences: [")
#                 ins = ll["in_sequences"]
#                 for j, ie in enumerate(ins):
#                     if j > 0:
#                         f.write(", ")
#                     f.write(_canonical_name(ie))
#                 f.write("]\n")

#             if "in_dim" in ll:
#                 f.write(f"    in_dim: {ll['in_dim']}\n")

#             show_output = verbose
#             if "output" in ll:
#                 f.write(f"    output: {ll['output']}\n")
#                 show_output = True

#             procs = int(ll.get("processors", 0))
#             if procs == 0:
#                 f.write("    processors: unknown\n")
#             else:
#                 f.write(f"    processors: 0x{procs:016x}\n")

#             f.write(f"    out_offset: 0x{int(ll.get('out_offset', 0)):04x}\n")

#             if "quantization" in ll:
#                 f.write(f"    quantization: {ll['quantization']}\n")
#             if "output_width" in ll:
#                 f.write(f"    output_width: {ll['output_width']}\n")

#             f.write(f"    op: {ll['op']}\n")

#             if "eltwise" in ll:
#                 f.write(f"    eltwise: {ll['eltwise']}\n")
#             if "operands" in ll:
#                 f.write(f"    operands: {ll['operands']}\n")
#             if "kernel_size" in ll:
#                 f.write(f"    kernel_size: {ll['kernel_size']}\n")
#             if "pad" in ll:
#                 f.write(f"    pad: {ll['pad']}\n")
#             if "groups" in ll:
#                 f.write(f"    groups: {ll['groups']}\n")
#             if "flatten" in ll:
#                 f.write(f"    flatten: {ll['flatten']}\n")
#             if "activate" in ll:
#                 f.write(f"    activate: {ll['activate']}\n")
#             if "pool_first" in ll:
#                 f.write(f"    pool_first: {ll['pool_first']}\n")
#             if "max_pool" in ll:
#                 f.write(f"    max_pool: {ll['max_pool']}\n")
#             if "avg_pool" in ll:
#                 f.write(f"    avg_pool: {ll['avg_pool']}\n")
#             if "pool_stride" in ll:
#                 f.write(f"    pool_stride: {ll['pool_stride']}\n")
#             if "pool_dilation" in ll:
#                 f.write(f"    pool_dilation: {ll['pool_dilation']}\n")
#             if "write_gap" in ll:
#                 f.write(f"    write_gap: {ll['write_gap']}\n")
#             if "output_processors" in ll:
#                 opm = int(ll["output_processors"])
#                 if opm == 0:
#                     f.write("    output_processors: unknown\n")
#                 else:
#                     f.write(f"    output_processors: 0x{opm:016x}\n")

#             # Output shape comment (best-effort)
#             if name == opid_to_name.get(final_layer_op.index, "") or show_output:
#                 f.write("    # output shape: (shape inference from SSA omitted in fused view)\n")

#             prev_name = name


# # ----------------------------- Compatibility wrapper (your pipeline API) -----------------------------


# def emit_ai8x_yaml_from_tosa_core(
#     *,
#     tosa: TosaModule,
#     core_range: Tuple[int, int],
#     out_dir: Path,
#     arch: str,
#     dataset: str,
#     device_name: str,
#     hwc: bool = False,
#     use_fifos: bool = False,
#     move_l0: bool = False,
#     verbose: bool = True,
#     weight_bits_by_layer: Optional[Dict[str, int]] = None,
# ) -> Path:
#     """
#     Emit ai8x YAML from the SoT TOSA MLIR file for the [core_range] ops.
#     Implementation note:
#       This emitter parses the full MLIR file; core_range is currently used
#       only as a hint (we emit the full file view). If you need strict slicing,
#       extend _collect_tosa_statements to allow op-index slicing.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     yaml_path = out_dir / "ai8x_autogen.yaml"

#     # The reference generator emits the entire graph; in your pipeline you want the NPU core.
#     # For now, we emit from the file; strict slicing can be added once you decide how to
#     # retain SSA names in TosaModule.
#     emit_ai8x_yaml_from_tosa_mlir(
#         tosa_mlir_path=tosa.path,
#         dataset=dataset,
#         arch=arch,
#         device_name=device_name,
#         hwc=hwc,
#         use_fifos=use_fifos,
#         move_l0=move_l0,
#         filename=yaml_path,
#         verbose=verbose,
#         weight_bits_by_layer=weight_bits_by_layer,
#     )
#     return yaml_path


# def emit_ai8x_yaml_and_sample(
#     *,
#     tosa: TosaModule,
#     core_range: Tuple[int, int],
#     out_dir: Path,
#     arch: str,
#     dataset: str,
#     device_name: str,
#     example_input: torch.Tensor,
#     hwc: bool = False,
#     use_fifos: bool = False,
#     move_l0: bool = False,
#     verbose: bool = True,
#     weight_bits_by_layer: Optional[Dict[str, int]] = None,
# ) -> Ai8xYamlEmitResult:
#     yaml_path = emit_ai8x_yaml_from_tosa_core(
#         tosa=tosa,
#         core_range=core_range,
#         out_dir=out_dir,
#         arch=arch,
#         dataset=dataset,
#         device_name=device_name,
#         hwc=hwc,
#         use_fifos=use_fifos,
#         move_l0=move_l0,
#         verbose=verbose,
#         weight_bits_by_layer=weight_bits_by_layer,
#     )
#     sample_path = _write_sample_npy(out_dir, example_input)
#     return Ai8xYamlEmitResult(yaml_path=yaml_path, sample_path=sample_path)

# unpu_bench/ai8x_yaml_from_tosa.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import torch

from .errors import CompilationError
from .tosa_ir import TosaModule, TosaOpSig


@dataclass(frozen=True)
class Ai8xYamlEmitResult:
    yaml_path: Path
    sample_path: Path


# ---------- Small utilities ----------

_SSA_RE = re.compile(r"%[0-9]+")
_ARRAY_I64_RE_TEMPLATE = r"{name}\s*=\s*array<i\d+:\s*([^>]*)>"


def _wrap_comment(s: str, width: int = 78) -> List[str]:
    words = s.split()
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if n + add > width:
            out.append("# " + " ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += add
    if cur:
        out.append("# " + " ".join(cur))
    return out


def _write_sample_npy(out_dir: Path, x: torch.Tensor) -> Path:
    out = out_dir / "sample.npy"
    arr = x.detach().cpu().numpy()
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(np.rint(arr * 20.0), -127, 127).astype(np.int64)
    np.save(out, arr)
    return out


def _device_half_data(device_name: str) -> int:
    """
    Mimic ai8x-training YAML generator ping-pong offset:
      MAX78000 (ai85) -> 0x4000
      MAX78002 (ai87) -> 0xa000
    """
    dn = (device_name or "").upper()
    return 0xA000 if "78002" in dn or "AI87" in dn or "87" in dn else 0x4000


def _allocate_offset_pingpong(prev_offset: int, device_name: str) -> int:
    half = _device_half_data(device_name)
    return half if prev_offset == 0 else 0


def _parse_array_i64_from_line(line: str, name: str) -> Optional[List[int]]:
    """
    Parse MLIR attr forms like:
      stride = array<i64: 1, 1>
      pad = array<i64: 1, 1, 1, 1>
    """
    m = re.search(_ARRAY_I64_RE_TEMPLATE.format(name=re.escape(name)), line)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return []
    vals: List[int] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            return None
    return vals


def _extract_ssa_def_use(line: str) -> Tuple[Optional[str], List[str]]:
    """
    Return (defined_ssa, used_ssas) from a single-line op print like:
      %27 = tosa.conv2d %26, %10, %1, %0, %0 { ... }
    """
    ids = _SSA_RE.findall(line)
    if not ids:
        return None, []
    defined: Optional[str] = None
    used = ids[:]
    if "=" in line:
        # Assume first SSA id on the line is the result.
        defined = ids[0]
        used = ids[1:]
    return defined, used


def _build_use_counts(ops: List[TosaOpSig]) -> Dict[str, int]:
    """
    Conservative SSA use-counting to enable safe local fusions.
    """
    use_count: Dict[str, int] = {}
    for op in ops:
        line = op.location[1]
        _d, uses = _extract_ssa_def_use(line)
        for u in uses:
            use_count[u] = use_count.get(u, 0) + 1
    return use_count


# ---------- Lowering: TOSA -> ai8x YAML layer dicts ----------

def _infer_kernel_size_from_conv_weight(op: TosaOpSig) -> str:
    # torch-mlir TOSA conv2d weights typically: [OC, KH, KW, IC]
    if len(op.operands) < 2:
        raise CompilationError("tosa.conv2d missing weight operand in parsed signature")
    w = op.operands[1]
    if len(w.shape) != 4:
        raise CompilationError(f"tosa.conv2d weight is not rank-4: {w.raw}")
    kh, kw = w.shape[1], w.shape[2]
    if kh <= 0 or kw <= 0:
        # Unknown dims in text parsing show up as -1; fall back.
        return "3x3"
    return f"{kh}x{kw}"


def _infer_conv_pad_from_attrs(line: str) -> int:
    """
    ai8x YAML 'pad' is a scalar. We accept only symmetric padding.
    TOSA 'pad' is [top, bottom, left, right].
    """
    pad = _parse_array_i64_from_line(line, "pad")
    if pad is None:
        return 0
    if len(pad) != 4:
        # Some pipelines might emit 2 values; treat as symmetric if possible.
        if len(pad) == 2 and pad[0] == pad[1]:
            return int(pad[0])
        raise CompilationError(f"Unsupported tosa.conv2d pad format: {pad}")
    if not (pad[0] == pad[1] == pad[2] == pad[3]):
        raise CompilationError(
            f"Non-symmetric padding is not representable as ai8x scalar 'pad': {pad}"
        )
    return int(pad[0])


def _infer_conv_stride_from_attrs(line: str) -> Tuple[int, int]:
    stride = _parse_array_i64_from_line(line, "stride")
    if stride is None:
        return (1, 1)
    if len(stride) != 2:
        raise CompilationError(f"Unsupported tosa.conv2d stride format: {stride}")
    return (int(stride[0]), int(stride[1]))


def _infer_conv_dilation_from_attrs(line: str) -> Tuple[int, int]:
    dilation = _parse_array_i64_from_line(line, "dilation")
    if dilation is None:
        return (1, 1)
    if len(dilation) != 2:
        raise CompilationError(f"Unsupported tosa.conv2d dilation format: {dilation}")
    return (int(dilation[0]), int(dilation[1]))


def _infer_pool_params_from_attrs(line: str) -> Tuple[int, int, int]:
    """
    Return (kernel, stride, pad_scalar) for pooling.
    - kernel and stride are taken from 'kernel' and 'stride' arrays.
    - pad is scalar only if symmetric [t,b,l,r] all equal; otherwise error.
    """
    kernel = _parse_array_i64_from_line(line, "kernel")
    stride = _parse_array_i64_from_line(line, "stride")
    pad = _parse_array_i64_from_line(line, "pad")

    # kernel
    k = 2
    if kernel is not None:
        if len(kernel) == 2 and kernel[0] == kernel[1]:
            k = int(kernel[0])
        elif len(kernel) == 1:
            k = int(kernel[0])
        else:
            raise CompilationError(f"Unsupported pool kernel format: {kernel}")

    # stride
    s = 2
    if stride is not None:
        if len(stride) == 2 and stride[0] == stride[1]:
            s = int(stride[0])
        elif len(stride) == 1:
            s = int(stride[0])
        else:
            raise CompilationError(f"Unsupported pool stride format: {stride}")

    # pad (optional)
    p = 0
    if pad is not None:
        if len(pad) == 4:
            if not (pad[0] == pad[1] == pad[2] == pad[3]):
                raise CompilationError(
                    f"Non-symmetric pooling pad is not representable as ai8x scalar 'pad': {pad}"
                )
            p = int(pad[0])
        elif len(pad) == 2 and pad[0] == pad[1]:
            p = int(pad[0])
        elif len(pad) == 1:
            p = int(pad[0])
        else:
            raise CompilationError(f"Unsupported pool pad format: {pad}")

    return k, s, p


def _is_ignored_plumbing(op_name: str) -> bool:
    return op_name in ("tosa.const", "tosa.const_shape")


def _emit_layer_lines(layer: Dict[str, object], layer_idx: int) -> List[str]:
    """
    Stable key order and YAML formatting.
    """
    lines: List[str] = []
    lines.append(f"  # Layer {layer_idx}")
    lines.append(f"  - name: {layer['name']}")
    if "comment" in layer:
        lines.append(f"    # {layer['comment']}")

    key_order = [
        "data_format",
        "in_sequences",
        "output",
        "processors",
        "out_offset",
        "op",
        "eltwise",
        "operands",
        "kernel_size",
        "pad",
        "stride",
        "dilation",
        "groups",
        "flatten",
        "activate",
        "pool_first",
        "max_pool",
        "avg_pool",
        "pool_stride",
        "pool_dilation",
        "write_gap",
        "output_processors",
    ]

    for k in key_order:
        if k not in layer:
            continue
        v = layer[k]
        if k in ("processors", "output_processors") and isinstance(v, int):
            lines.append(f"    {k}: 0x{v:016x}")
        elif k == "out_offset" and isinstance(v, int):
            lines.append(f"    {k}: 0x{v:04x}")
        else:
            lines.append(f"    {k}: {v}")

    lines.append("")
    return lines


def emit_ai8x_yaml_from_tosa_core(
    *,
    tosa: TosaModule,
    core_range: Tuple[int, int],
    out_dir: Path,
    arch: str,
    dataset: str,
    device_name: str,
) -> Path:
    """
    Emit an ai8x YAML (template-grade) from canonical TOSA MLIR.

    Implemented:
      - ignore tosa.const / tosa.const_shape nodes
      - parse tosa.conv2d stride/pad/dilation from MLIR text attrs
      - fuse tosa.relu into preceding conv layer when safe (single-use + direct successor)
      - infer pooling kernel/stride/pad from attrs (max_pool2d/avg_pool2d)
      - stable sequential layer naming

    Notes:
      - Processor/out_offset allocation is conservative ping-pong and a placeholder.
      - This emits YAML for izer consumption as a *starting point*; you will likely
        want to replace the allocator with a real one later.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "ai8x_autogen.yaml"

    start, end = core_range
    core_ops = tosa.ops[start:end]

    # Use-counts for safe local fusions
    use_count = _build_use_counts(core_ops)

    header_lines = [
        "---",
        *_wrap_comment(
            "YAML generated from canonical TOSA MLIR. "
            "This is a template-grade emitter: processor/out_offset allocation is conservative "
            "and may require refinement."
        ),
        f"# Generated for {device_name}",
        "",
        f"arch: {arch}",
        f"dataset: {dataset}",
        "",
        "layers:",
    ]

    layers_yaml: List[str] = []
    layer_idx = 0

    prev_out_offset = 0
    default_processors = 0x0000000000000001

    i = 0
    while i < len(core_ops):
        op = core_ops[i]

        # Ignore plumbing
        if _is_ignored_plumbing(op.op_name):
            i += 1
            continue

        line = op.location[1]

        # Stable sequential layer naming
        layer_name = f"layer_{layer_idx:03d}"

        # --- conv2d ---
        if op.op_name == "tosa.conv2d":
            kernel_size = _infer_kernel_size_from_conv_weight(op)
            pad = _infer_conv_pad_from_attrs(line)
            stride_h, stride_w = _infer_conv_stride_from_attrs(line)
            dil_h, dil_w = _infer_conv_dilation_from_attrs(line)

            # ai8x YAML expects scalar pad and (commonly) scalar stride for 2D.
            # If stride differs by axis, we fail explicitly.
            if stride_h != stride_w:
                raise CompilationError(
                    f"Non-uniform conv2d stride not representable in simple ai8x YAML: {(stride_h, stride_w)}"
                )
            if dil_h != dil_w:
                raise CompilationError(
                    f"Non-uniform conv2d dilation not representable in simple ai8x YAML: {(dil_h, dil_w)}"
                )

            conv_layer: Dict[str, object] = {
                "name": layer_name,
                "op": "Conv2d",
                "kernel_size": kernel_size,
                "pad": pad,
                "activate": "None",
                "processors": default_processors,
                "out_offset": 0,  # filled below
            }

            # Record stride/dilation explicitly if not default.
            # If your izer YAML schema doesn’t accept these keys, remove them here;
            # they’re still useful as comments/metadata.
            if stride_h != 1:
                conv_layer["stride"] = stride_h
            if dil_h != 1:
                conv_layer["dilation"] = dil_h

            # Decide offset now (ping-pong allocator)
            prev_out_offset = _allocate_offset_pingpong(prev_out_offset, device_name)
            conv_layer["out_offset"] = prev_out_offset

            # Try to fuse immediate successor relu, safely:
            # - next op is tosa.relu
            # - relu consumes conv result directly
            # - conv result is used exactly once in core (by the relu)
            fused = False
            if i + 1 < len(core_ops) and core_ops[i + 1].op_name == "tosa.relu":
                relu_line = core_ops[i + 1].location[1]
                conv_def, _conv_uses = _extract_ssa_def_use(line)
                relu_def, relu_uses = _extract_ssa_def_use(relu_line)

                if conv_def is not None and relu_uses and relu_uses[0] == conv_def:
                    if use_count.get(conv_def, 0) == 1:
                        conv_layer["activate"] = "Relu"
                        conv_layer["comment"] = "fused: following tosa.relu"
                        fused = True

            layers_yaml.extend(_emit_layer_lines(conv_layer, layer_idx))
            layer_idx += 1
            i += 2 if fused else 1
            continue

        # --- relu (not fused) ---
        if op.op_name == "tosa.relu":
            # If we get here, we did not fuse (e.g., multi-use or not adjacent).
            # Emit a passthrough + activate.
            prev_out_offset = _allocate_offset_pingpong(prev_out_offset, device_name)
            relu_layer: Dict[str, object] = {
                "name": layer_name,
                "op": "Passthrough",
                "activate": "Relu",
                "processors": default_processors,
                "out_offset": prev_out_offset,
            }
            layers_yaml.extend(_emit_layer_lines(relu_layer, layer_idx))
            layer_idx += 1
            i += 1
            continue

        # --- pooling ---
        if op.op_name in ("tosa.max_pool2d", "tosa.avg_pool2d"):
            k, s, p = _infer_pool_params_from_attrs(line)

            prev_out_offset = _allocate_offset_pingpong(prev_out_offset, device_name)
            pool_layer: Dict[str, object] = {
                "name": layer_name,
                "op": "Passthrough",
                "processors": default_processors,
                "out_offset": prev_out_offset,
                "pool_stride": s,
            }
            if p != 0:
                pool_layer["pad"] = p

            # ai8x YAML uses max_pool/avg_pool scalar when square
            if op.op_name == "tosa.max_pool2d":
                pool_layer["max_pool"] = k
            else:
                pool_layer["avg_pool"] = k

            layers_yaml.extend(_emit_layer_lines(pool_layer, layer_idx))
            layer_idx += 1
            i += 1
            continue

        # --- elementwise add/sub ---
        if op.op_name in ("tosa.add", "tosa.sub"):
            elt = "Add" if op.op_name == "tosa.add" else "Sub"
            prev_out_offset = _allocate_offset_pingpong(prev_out_offset, device_name)
            ew_layer: Dict[str, object] = {
                "name": layer_name,
                "op": "Passthrough",
                "eltwise": elt,
                "operands": 2,
                "processors": default_processors,
                "out_offset": prev_out_offset,
            }
            layers_yaml.extend(_emit_layer_lines(ew_layer, layer_idx))
            layer_idx += 1
            i += 1
            continue

        # --- reshape/transpose/cast: ignore (plumbing) ---
        if op.op_name in ("tosa.reshape", "tosa.transpose", "tosa.cast"):
            i += 1
            continue

        # --- matmul/fully_connected ---
        if op.op_name in ("tosa.fully_connected", "tosa.matmul"):
            prev_out_offset = _allocate_offset_pingpong(prev_out_offset, device_name)
            fc_layer: Dict[str, object] = {
                "name": layer_name,
                "op": "Linear",
                "flatten": "true",
                "activate": "None",
                "processors": default_processors,
                "out_offset": prev_out_offset,
            }
            layers_yaml.extend(_emit_layer_lines(fc_layer, layer_idx))
            layer_idx += 1
            i += 1
            continue

        # Anything else is not supported by this emitter yet.
        raise CompilationError(
            f"Cannot emit ai8x YAML for op '{op.op_name}' at line {op.location[0]}:\n"
            f"{op.location[1]}\n"
            "Either mark it CPU in capability schemas/partitioning, or add YAML lowering here."
        )

    yaml_path.write_text("\n".join(header_lines + layers_yaml).rstrip() + "\n", encoding="utf-8")
    return yaml_path


def emit_ai8x_yaml_and_sample(
    *,
    tosa: TosaModule,
    core_range: Tuple[int, int],
    out_dir: Path,
    arch: str,
    dataset: str,
    device_name: str,
    example_input: torch.Tensor,
) -> Ai8xYamlEmitResult:
    yaml_path = emit_ai8x_yaml_from_tosa_core(
        tosa=tosa,
        core_range=core_range,
        out_dir=out_dir,
        arch=arch,
        dataset=dataset,
        device_name=device_name,
    )
    sample_path = _write_sample_npy(out_dir, example_input)
    return Ai8xYamlEmitResult(yaml_path=yaml_path, sample_path=sample_path)

    

