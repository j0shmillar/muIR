from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from .ai8x_shim import install_ai8x_shim
from .backend_hardware import emit_hardware_artifacts
from .backend_lowering import lower_program_for_backend
from .config import PlatformSpec
from .errors import CompilationError
from .ir_version import migrate_program_ir_metadata, validate_program_ir_metadata
from .muir import (
    BackendArtifact,
    Program,
    build_program_from_onnx,
    build_program_from_tflite_stub,
    build_program_from_torch,
    write_program_json,
)
from .passes import (
    compute_partition_metrics,
    run_ir_canonicalization,
    run_ir_rewrite_passes,
    run_legality_check,
    run_partitioning,
    run_quantization_contract_validation,
    run_ir_validation,
)
from .quant import run_ai8x_bn_fuse_and_quantize
from .semantic_check import run_semantic_check_torch_vs_ir

log = logging.getLogger(__name__)


@dataclass
class CompileConfig:
    # Core
    target_format: str
    target_hardware: str
    bit_width: int
    out_dir: str

    # Model
    model_py: str | None = None
    model_class: str | None = None
    model_ckpt: str | None = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    model_onnx: str | None = None
    model_tflite: str | None = None

    # I/O
    input_shape: str = ""
    output_shape: str = ""
    input_names: str = ""
    output_names: str = ""

    # Data / quantization
    data_sample: str | None = None
    overwrite: bool = False

    # ai8x backend config
    ai8x_root: str | None = None
    ai8x_device: str | None = None
    ai8x_config_file: str | None = None
    ai8x_prefix: str = "unpu_model"

    debug: bool = False

    # Optional hardware artifact emission (vendor toolchain outputs)
    emit_hardware_artifact: bool = False
    backend_source_model: str | None = None
    backend_tool_args: list[str] | None = None
    backend_command: str | None = None
    backend_output_glob: str | None = None
    # CVI built-in flags
    cvi_calibration_table: str | None = None
    cvi_tolerance: float = 0.99
    cvi_dynamic: bool = False
    cvi_excepts: str | None = None
    cvi_resize_dims: str | None = None
    cvi_pixel_format: str | None = None
    cvi_test_result: str | None = None
    cvi_keep_aspect_ratio: bool = False
    semantic_check: bool = False
    semantic_strict: bool = False
    semantic_rtol: float = 1e-3
    semantic_atol: float = 1e-4
    quant_contract_strict: bool = False


def compile_model(cfg: CompileConfig, platforms: Dict[str, PlatformSpec]) -> None:
    del platforms  # Platform constraints are validated at CLI-level.

    log.info(
        "Compile start: format=%s hardware=%s bit_width=%d",
        cfg.target_format,
        cfg.target_hardware,
        cfg.bit_width,
    )

    _ensure_out_dir(cfg.out_dir, overwrite=cfg.overwrite)
    _sanity_check_files(cfg)

    source = _resolve_model_source(cfg)
    model = None
    example_input = None
    if source == "torch":
        model = _load_model(cfg, ckpt_override=None)
        example_input = _build_example_input(cfg)
        program = build_program_from_torch(
            model=model,
            example_input=example_input,
            default_backend=cfg.target_format,
            target_hardware=cfg.target_hardware,
            bit_width=cfg.bit_width,
            metadata={
                "input_shape": cfg.input_shape,
                "output_shape": cfg.output_shape,
                "input_names": cfg.input_names,
                "output_names": cfg.output_names,
                "target_format": cfg.target_format,
                "pipeline_mode": "ir",
            },
        )
    elif source == "onnx":
        assert cfg.model_onnx
        program = build_program_from_onnx(
            cfg.model_onnx,
            default_backend=cfg.target_format,
            target_hardware=cfg.target_hardware,
            bit_width=cfg.bit_width,
            metadata={
                "target_format": cfg.target_format,
                "pipeline_mode": "ir",
                "frontend": "onnx",
            },
        )
    else:
        assert cfg.model_tflite
        program = build_program_from_tflite_stub(
            cfg.model_tflite,
            default_backend=cfg.target_format,
            target_hardware=cfg.target_hardware,
            bit_width=cfg.bit_width,
            metadata={
                "target_format": cfg.target_format,
                "pipeline_mode": "ir",
            },
        )

    run_ir_canonicalization(program)
    run_ir_rewrite_passes(program)
    migrate_program_ir_metadata(program)
    validate_program_ir_metadata(program)
    run_ir_validation(program)

    ir_caps_path = (
        Path(__file__).resolve().parents[1]
        / "unpu_bench"
        / "capabilities"
        / f"ir_{cfg.target_format}.yaml"
    )
    if not ir_caps_path.exists():
        raise CompilationError(
            f"Missing IR capability schema for backend '{cfg.target_format}': {ir_caps_path}"
        )
    run_legality_check(program, backend=cfg.target_format, caps_path=ir_caps_path)
    run_partitioning(program, backend=cfg.target_format, fallback_backend="cpu")
    run_quantization_contract_validation(
        program,
        backend=cfg.target_format,
        bit_width=cfg.bit_width,
        strict=cfg.quant_contract_strict,
    )

    if cfg.target_format == "ai8x":
        if source != "torch" or example_input is None:
            raise CompilationError(
                "ai8x backend currently requires torch model source (--model-py/--model-class)."
            )
        _configure_ai8x_inputs(cfg)

    if (
        cfg.semantic_check
        and source == "torch"
        and model is not None
        and example_input is not None
    ):
        sem = run_semantic_check_torch_vs_ir(
            program=program,
            model=model,
            example_input=example_input,
            rtol=cfg.semantic_rtol,
            atol=cfg.semantic_atol,
        )
        program.metadata["semantic_check"] = sem
        if cfg.semantic_strict and sem.get("status") != "pass":
            raise CompilationError(
                f"Semantic check failed/was skipped in strict mode: {sem}"
            )

    quant_ckpt_path: str | None = None
    if cfg.target_format == "ai8x":
        log.info("Running ai8x quantization on checkpoint...")
        quant_ckpt_path = str(
            run_ai8x_bn_fuse_and_quantize(
                cfg,
                model_ckpt=cfg.model_ckpt,
                out_dir=cfg.out_dir,
            )
        )
        log.info("Quantized checkpoint written to %s", quant_ckpt_path)

    if cfg.target_format == "ai8x":
        run_ai8x_backend(program, cfg)
    else:
        artifacts = lower_program_for_backend(
            program,
            target_format=cfg.target_format,
            out_dir=cfg.out_dir,
        )
        for artifact in artifacts:
            program.add_artifact(artifact)
        if cfg.emit_hardware_artifact:
            if not cfg.backend_source_model:
                # default source-model to the frontend artifact when possible
                if cfg.model_tflite:
                    cfg.backend_source_model = cfg.model_tflite
                elif cfg.model_onnx:
                    cfg.backend_source_model = cfg.model_onnx
            hw_arts = emit_hardware_artifacts(cfg)
            if not hw_arts:
                raise CompilationError(
                    "--emit-hardware-artifact was requested, but no hardware artifact was produced."
                )
            for artifact in hw_arts:
                program.add_artifact(artifact)

    program.metadata["partition_metrics"] = compute_partition_metrics(
        program,
        backend=cfg.target_format,
        fallback_backend="cpu",
    )

    program_json_path = write_program_json(program, cfg.out_dir)
    log.info("Compile done.")
    log.info("  Program:  %s", program_json_path)


def _configure_ai8x_inputs(cfg: CompileConfig) -> None:
    if not cfg.ai8x_config_file:
        raise CompilationError("ai8x backend requires --ai8x-config-file.")


def _build_ai8x_argv(
    cfg: CompileConfig, checkpoint_file: str, ai8x_dir: Path
) -> list[str]:
    argv: list[str] = ["ai8x-izer"]

    argv += ["--device", str(cfg.ai8x_device or "MAX78000")]
    if not cfg.ai8x_config_file:
        raise CompilationError(
            "ai8x backend requires a YAML config via --ai8x-config-file."
        )
    argv += ["--config-file", str(cfg.ai8x_config_file)]

    argv += [
        "--checkpoint-file",
        str(checkpoint_file),
        "--embedded-code",
        "--test-dir",
        str(ai8x_dir),
        "--prefix",
        str(cfg.ai8x_prefix or "unpu_model"),
        "--no-version-check",
    ]

    if cfg.data_sample:
        argv += ["--sample-input", str(cfg.data_sample)]

    if cfg.debug:
        argv.append("--display-checkpoint")

    return argv


def run_ai8x_backend(program: Program, cfg: CompileConfig) -> None:
    try:
        izer_mod = importlib.import_module("izer.izer")
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[1]
        ai8x_synth = repo_root / "ai8x-synthesis"
        if ai8x_synth.is_dir() and str(ai8x_synth) not in sys.path:
            sys.path.insert(0, str(ai8x_synth))
        try:
            izer_mod = importlib.import_module("izer.izer")
        except ImportError as inner_exc:
            raise CompilationError(
                "Failed to import ai8x-synthesis 'izer.izer' module.\n"
                "Make sure ai8x-synthesis is installed or available as sibling checkout, e.g.:\n"
                "  cd ../ai8x-synthesis && pip install -e ."
            ) from inner_exc

    out_dir = Path(cfg.out_dir)
    ai8x_dir = out_dir / "ai8x"
    ai8x_dir.mkdir(parents=True, exist_ok=True)

    quant_ckpt = out_dir / "model_quantized.pth"
    if quant_ckpt.exists():
        checkpoint_file = str(quant_ckpt)
    elif cfg.model_ckpt:
        checkpoint_file = cfg.model_ckpt
    else:
        raise CompilationError(
            "ai8x backend requires a quantized or source checkpoint; no ONNX fallback is used."
        )

    argv = _build_ai8x_argv(cfg, checkpoint_file, ai8x_dir)
    log.info("Invoking ai8x-synthesis izer with argv: %s", " ".join(map(str, argv)))

    old_argv = sys.argv
    sys.argv = argv
    try:
        izer_mod.main()
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        if code != 0:
            raise CompilationError(
                f"ai8x-synthesis izer exited with code {code}"
            ) from e
    finally:
        sys.argv = old_argv

    project_dir = ai8x_dir / (cfg.ai8x_prefix or "unpu_model")
    if not project_dir.is_dir():
        raise CompilationError(
            f"ai8x-synthesis did not create expected project directory: {project_dir}"
        )

    rel = os.path.relpath(project_dir, cfg.out_dir)
    program.add_artifact(
        BackendArtifact(
            backend="ai8x",
            artifact_type="c_project",
            path=rel,
            meta={
                "device": cfg.ai8x_device or "MAX78000",
                "source": "ai8x-synthesis",
            },
        )
    )


def _ensure_out_dir(out_dir: str, overwrite: bool) -> None:
    if os.path.exists(out_dir) and os.listdir(out_dir) and not overwrite:
        raise CompilationError(
            f"Output directory '{out_dir}' is not empty; use --overwrite to replace."
        )
    os.makedirs(out_dir, exist_ok=True)


def _sanity_check_files(cfg: CompileConfig) -> None:
    missing: list[str] = []
    source = _resolve_model_source(cfg)
    required_ai8x_yaml = cfg.target_format == "ai8x"
    required_backend_source = (
        cfg.emit_hardware_artifact
        and cfg.target_format in {"tflm", "vela", "cvi", "eiq"}
        and not (cfg.backend_source_model or cfg.model_onnx or cfg.model_tflite)
    )
    for label, path, required in [
        ("model_py", cfg.model_py, source == "torch"),
        ("model_onnx", cfg.model_onnx, source == "onnx"),
        ("model_tflite", cfg.model_tflite, source == "tflite"),
        ("model_ckpt", cfg.model_ckpt, cfg.target_format == "ai8x"),
        ("data_sample", cfg.data_sample, False),
        ("ai8x_config_file", cfg.ai8x_config_file, required_ai8x_yaml),
        ("backend_source_model", cfg.backend_source_model, required_backend_source),
    ]:
        if not path:
            if required:
                missing.append(label)
            continue
        if not os.path.exists(path):
            missing.append(f"{label} ({path})")
    if missing:
        raise CompilationError("Missing required files: " + ", ".join(missing))

    if source == "torch":
        required_text = [
            ("input_shape", cfg.input_shape),
            ("input_names", cfg.input_names),
            ("output_names", cfg.output_names),
        ]
        for label, val in required_text:
            if not val:
                raise CompilationError(
                    f"Missing required argument for torch source: --{label.replace('_', '-')}"
                )


def _resolve_model_source(cfg: CompileConfig) -> str:
    has_torch = bool(cfg.model_py and cfg.model_class)
    has_onnx = bool(cfg.model_onnx)
    has_tflite = bool(cfg.model_tflite)
    count = int(has_torch) + int(has_onnx) + int(has_tflite)
    if count == 0:
        raise CompilationError(
            "No model source provided. Use either --model-py/--model-class, --model-onnx, or --model-tflite."
        )
    if count > 1:
        raise CompilationError(
            "Ambiguous model source: provide exactly one of torch source, --model-onnx, or --model-tflite."
        )
    if has_torch:
        return "torch"
    if has_onnx:
        return "onnx"
    return "tflite"


def _import_module_from_file(path: str):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _load_model(cfg: CompileConfig, ckpt_override: str | None = None) -> nn.Module:
    install_ai8x_shim()
    if not cfg.model_py or not cfg.model_class:
        raise CompilationError(
            "Torch model source requires both --model-py and --model-class."
        )

    module = _import_module_from_file(cfg.model_py)
    if not hasattr(module, cfg.model_class):
        raise CompilationError(
            f"Model class '{cfg.model_class}' not found in {cfg.model_py}"
        )

    cls_or_factory = getattr(module, cfg.model_class)
    model = cls_or_factory(**cfg.model_args)
    if not isinstance(model, nn.Module):
        raise CompilationError("Constructed object is not a torch.nn.Module")

    ckpt_path = ckpt_override or cfg.model_ckpt
    if ckpt_path:
        log.info("Loading checkpoint from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    return model


def _parse_shape(shape_str: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in shape_str.strip().split())
    except ValueError as exc:
        raise CompilationError(
            f"Invalid shape string '{shape_str}'. Expected '1 3 96 96' style."
        ) from exc


def _build_example_input(cfg: CompileConfig) -> torch.Tensor:
    if cfg.data_sample and os.path.exists(cfg.data_sample):
        arr = np.load(cfg.data_sample)
        t = torch.from_numpy(arr)
        return t.float()
    shape = _parse_shape(cfg.input_shape)
    return torch.randn(*shape, dtype=torch.float32)
