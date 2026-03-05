from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import CompilationError
from .muir import BackendArtifact, Program


def _write_backend_ir_bundle(program: Program, out_dir: Path, backend: str, artifact_name: str) -> BackendArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / artifact_name

    payload: dict[str, Any] = {
        "backend": backend,
        "graph": {
            "name": program.graph.name,
            "num_ops": len(program.graph.op_order),
            "num_tensors": len(program.graph.tensors),
            "inputs": program.graph.inputs,
            "outputs": program.graph.outputs,
        },
        "partitions": [
            {
                "id": p.id,
                "backend": p.backend,
                "op_ids": p.op_ids,
            }
            for p in program.partitions
        ],
        "metadata": program.metadata,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return BackendArtifact(
        backend=backend,
        artifact_type="ir_bundle",
        path=str(path.name),
        meta={"format": "json", "ir_native": True},
    )


def _write_backend_compiled_model(program: Program, out_dir: Path, backend: str, artifact_name: str) -> BackendArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / artifact_name

    constants: dict[str, Any] = {}
    for tid in program.graph.initializers:
        t = program.graph.tensors.get(tid)
        if t is None or t.data is None:
            continue
        constants[tid] = {
            "shape": t.type.shape,
            "dtype": t.type.dtype,
            "data": t.data,
        }

    ops = []
    for op_id in program.graph.op_order:
        op = program.graph.ops[op_id]
        ops.append(
            {
                "id": op.id,
                "kind": op.kind,
                "inputs": op.inputs,
                "outputs": op.outputs,
                "attrs": op.attrs,
            }
        )

    payload: dict[str, Any] = {
        "format_version": 1,
        "backend": backend,
        "graph": {
            "name": program.graph.name,
            "inputs": program.graph.inputs,
            "outputs": program.graph.outputs,
        },
        "tensors": {
            tid: {
                "shape": t.type.shape,
                "dtype": t.type.dtype,
                "layout": t.type.layout,
                "role": t.role,
            }
            for tid, t in program.graph.tensors.items()
        },
        "ops": ops,
        "constants": constants,
        "partitions": [
            {
                "id": p.id,
                "backend": p.backend,
                "op_ids": p.op_ids,
            }
            for p in program.partitions
        ],
        "metadata": program.metadata,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return BackendArtifact(
        backend=backend,
        artifact_type="compiled_model",
        path=str(path.name),
        meta={"format": "json", "target_format": backend, "ir_native": True},
    )


def lower_program_for_backend(program: Program, *, target_format: str, out_dir: str) -> list[BackendArtifact]:
    root = Path(out_dir)

    if target_format == "tflm":
        bdir = root / "tflm"
        ir = _write_backend_ir_bundle(program, bdir, "tflm", "model.tflm.ir.json")
        cm = _write_backend_compiled_model(program, bdir, "tflm", "model.tflm.compiled.json")
        ir.path = str(Path("tflm") / ir.path)
        cm.path = str(Path("tflm") / cm.path)
        return [ir, cm]

    if target_format == "vela":
        bdir = root / "vela"
        ir = _write_backend_ir_bundle(program, bdir, "vela", "model.vela.ir.json")
        cm = _write_backend_compiled_model(program, bdir, "vela", "model.vela.compiled.json")
        ir.path = str(Path("vela") / ir.path)
        cm.path = str(Path("vela") / cm.path)
        return [ir, cm]

    if target_format == "cvi":
        bdir = root / "cvi"
        ir = _write_backend_ir_bundle(program, bdir, "cvi", "model.cvi.ir.json")
        cm = _write_backend_compiled_model(program, bdir, "cvi", "model.cvi.compiled.json")
        ir.path = str(Path("cvi") / ir.path)
        cm.path = str(Path("cvi") / cm.path)
        return [ir, cm]

    if target_format == "eiq":
        bdir = root / "eiq"
        ir = _write_backend_ir_bundle(program, bdir, "eiq", "model.eiq.ir.json")
        cm = _write_backend_compiled_model(program, bdir, "eiq", "model.eiq.compiled.json")
        ir.path = str(Path("eiq") / ir.path)
        cm.path = str(Path("eiq") / cm.path)
        return [ir, cm]

    raise CompilationError(f"No IR-native backend lowering implementation for target_format={target_format}")
