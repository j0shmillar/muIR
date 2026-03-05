from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from .backend_hardware import emit_hardware_artifacts
from .backend_lowering import lower_program_for_backend
from .errors import CompilationError
from .muir import (
    build_program_from_onnx,
    build_program_from_tflite_stub,
    build_program_from_torch,
    program_to_json,
    write_program_json,
)
from .passes import (
    run_ir_canonicalization,
    run_ir_validation,
    run_legality_check,
    run_partitioning,
)
from .pipeline import CompileConfig


def _shape_to_str(shape: Iterable[int] | None) -> str:
    if shape is None:
        return ""
    return " ".join(str(int(x)) for x in shape)


def convert(
    model: Any,
    *,
    backend: str,
    target_hardware: str,
    out_dir: str,
    bit_width: int = 8,
    input_shape: Iterable[int] | None = None,
    output_shape: Iterable[int] | None = None,
    input_names: str = "input",
    output_names: str = "output",
    emit_hardware_artifact: bool = False,
    backend_source_model: str | None = None,
    backend_tool_args: list[str] | None = None,
    backend_command: str | None = None,
    backend_output_glob: str | None = None,
    cvi_tolerance: float = 0.99,
    cvi_dynamic: bool = False,
    strict_partition: bool = False,
) -> Dict[str, Any]:
    """Programmatic conversion entrypoint.

    model can be:
    - torch.nn.Module
    - ONNX path (.onnx)
    - TFLite path (.tflite)
    """

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "target_format": backend,
        "pipeline_mode": "ir",
        "input_names": input_names,
        "output_names": output_names,
        "input_shape": _shape_to_str(input_shape),
        "output_shape": _shape_to_str(output_shape),
    }

    if isinstance(model, torch.nn.Module):
        if input_shape is None:
            raise CompilationError("convert(torch_model, ...) requires input_shape.")
        example_input = torch.randn(
            *tuple(int(x) for x in input_shape), dtype=torch.float32
        )
        program = build_program_from_torch(
            model=model.eval(),
            example_input=example_input,
            default_backend=backend,
            target_hardware=target_hardware,
            bit_width=bit_width,
            metadata=metadata,
        )
        source_model_path = backend_source_model
    elif isinstance(model, (str, Path)):
        p = Path(model)
        if not p.exists():
            raise CompilationError(f"Model path not found: {p}")
        if p.suffix.lower() == ".onnx":
            program = build_program_from_onnx(
                str(p),
                default_backend=backend,
                target_hardware=target_hardware,
                bit_width=bit_width,
                metadata=metadata,
            )
            source_model_path = backend_source_model or str(p)
        elif p.suffix.lower() == ".tflite":
            program = build_program_from_tflite_stub(
                str(p),
                default_backend=backend,
                target_hardware=target_hardware,
                bit_width=bit_width,
                metadata=metadata,
            )
            source_model_path = backend_source_model or str(p)
        else:
            raise CompilationError("Path model must be .onnx or .tflite")
    else:
        raise CompilationError("Unsupported model type for convert().")

    run_ir_canonicalization(program)
    run_ir_validation(program)

    ir_caps_path = (
        Path(__file__).resolve().parents[1]
        / "unpu_bench"
        / "capabilities"
        / f"ir_{backend}.yaml"
    )
    if ir_caps_path.exists():
        run_legality_check(program, backend=backend, caps_path=ir_caps_path)
        try:
            run_partitioning(program, backend=backend, fallback_backend="cpu")
        except CompilationError:
            if strict_partition:
                raise
            # Graceful fallback for mixed-legal graphs: keep pipeline runnable by
            # demoting execution to CPU partitions.
            for op_id in program.graph.op_order:
                op = program.graph.ops[op_id]
                op.preferred_backend = "cpu"
                op.legal_backends = ["cpu"]
            run_partitioning(program, backend=backend, fallback_backend="cpu")
            program.metadata["partition_fallback"] = "cpu_full"

    if backend != "ai8x":
        for art in lower_program_for_backend(
            program, target_format=backend, out_dir=str(out_root)
        ):
            program.add_artifact(art)

        if emit_hardware_artifact:
            cfg = CompileConfig(
                target_format=backend,
                target_hardware=target_hardware,
                bit_width=bit_width,
                model_py="",
                model_class="",
                model_ckpt=None,
                model_args={},
                model_onnx=(
                    source_model_path
                    if (source_model_path and str(source_model_path).endswith(".onnx"))
                    else None
                ),
                model_tflite=(
                    source_model_path
                    if (
                        source_model_path and str(source_model_path).endswith(".tflite")
                    )
                    else None
                ),
                input_shape=_shape_to_str(input_shape),
                output_shape=_shape_to_str(output_shape),
                input_names=input_names,
                output_names=output_names,
                data_sample=None,
                out_dir=str(out_root),
                overwrite=True,
                emit_hardware_artifact=True,
                backend_source_model=source_model_path,
                backend_tool_args=backend_tool_args or [],
                backend_command=backend_command,
                backend_output_glob=backend_output_glob,
                cvi_tolerance=cvi_tolerance,
                cvi_dynamic=cvi_dynamic,
            )
            for art in emit_hardware_artifacts(cfg):
                program.add_artifact(art)

    write_program_json(program, str(out_root))
    return {
        "out_dir": str(out_root),
        "program": program_to_json(program),
        "artifacts": [asdict(a) for a in program.backend_artifacts],
    }
