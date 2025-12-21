# unpu_bench/pipeline.py
from __future__ import annotations

import importlib.util
import logging
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import PlatformSpec
from .quant import run_ai8x_bn_fuse_and_quantize
from .muir import build_program_from_onnx, write_program_json, Program, BackendArtifact
from .passes import run_legality_check, run_partitioning
from .errors import CompilationError

log = logging.getLogger(__name__)

from .ai8x_shim import install_ai8x_shim  # TODO maybe just install ai8x-training also

import argparse


def _import_ai8x_modules() -> tuple[Any, Any]:
    """
    Import ai8x-synthesis modules.

    Expects ai8x-synthesis to be installed in this env, e.g.
      pip install -e ../ai8x-synthesis
    """
    try:
        # top-level module from ai8x-synthesis/quantize.py
        quantize = importlib.import_module("izer.quantize")
        # package module from ai8x-synthesis/izer/tornadocnn.py
        tc = importlib.import_module("izer.tornadocnn")
        return quantize, tc
    except ImportError as exc:
        raise CompilationError(
            "Failed to import ai8x-synthesis modules (quantize, izer.tornadocnn).\n"
            "Make sure ai8x-synthesis is installed in this Python environment, e.g.:\n"
            "   pip install -e ../ai8x-synthesis"
        ) from exc

@dataclass
class CompileConfig:
    # Core
    target_format: str
    target_hardware: str
    bit_width: int

    # Model
    model_py: str
    model_class: str
    model_ckpt: str | None
    model_args: Dict[str, Any]

    # I/O
    input_shape: str
    output_shape: str
    input_names: str
    output_names: str

    # Data / quantization
    data_sample: str | None

    # Paths
    out_dir: str
    overwrite: bool = False

    # ai8x backend config
    ai8x_root: str | None = None        # path to ai8x-synthesis checkout
    ai8x_device: str | None = None      # "MAX78000" / "MAX78002"
    ai8x_config_file: str | None = None # YAML
    ai8x_prefix: str = "unpu_model"

    debug: bool = False


# TO-ADD in izer.py
        # fext = args.checkpoint_file.rsplit(sep='.', maxsplit=1)[1].lower()
        # if fext == 'onnx':
        #     # ONNX file selected
        #     layers, weights, bias, output_shift, \
        #         input_channels, output_channels = \
        #         onnxcp.load(
        #             args.checkpoint_file,
        #             cfg['arch'],
        #             params['quantization'],
        #             params['bias_quantization'],
        #             params['output_shift'],
        #             params['kernel_size'],
        #             params['operator'],
        #             args.display_checkpoint,
        #             args.no_bias,
        #         )
        # elif fext == 'json':
        #     # IR Program JSON selected
        #     layers, weights, bias, output_shift, \
        #         input_channels, output_channels = \
        #         ircp.load(
        #             args.checkpoint_file,
        #             cfg['arch'],
        #             params['quantization'],
        #             params['bias_quantization'],
        #             params['output_shift'],
        #             params['kernel_size'],
        #             params['operator'],
        #             args.display_checkpoint,
        #             args.no_bias,
        #         )
        # else:
        #     # PyTorch checkpoint file selected
        #     layers, weights, bias, output_shift, \
        #         input_channels, output_channels, final_scale = \
        #         checkpoint.load(...)
            
def compile_model(cfg: CompileConfig, platforms: Dict[str, PlatformSpec]) -> None:
    """Torch → (ai8x quantization) → ONNX → IR → passes → backend.

    For ai8x:
      - Quantize the PyTorch checkpoint using ai8x.quantize.convert_checkpoint
      - Load quantized weights into the Torch model
      - Export quantized model to ONNX
      - Build IR and run legality + partitioning
      - Hand IR (and ONNX) to ai8x backend
    """
    log.info(
        "Compile start: format=%s hardware=%s bit_width=%d",
        cfg.target_format,
        cfg.target_hardware,
        cfg.bit_width,
    )

    _ensure_out_dir(cfg.out_dir, overwrite=cfg.overwrite)
    _sanity_check_files(cfg)

    # 0) Quantization (ai8x only)
    quant_ckpt_path: str | None = None
    if cfg.target_format == "ai8x":
        log.info("Running quantization on checkpoint before ONNX/IR export...")
        quant_ckpt_path = run_ai8x_bn_fuse_and_quantize(
            cfg,
            model_ckpt=cfg.model_ckpt,
            out_dir=cfg.out_dir,
        )
        log.info("Quantized checkpoint written to %s", quant_ckpt_path)

    # 1) Load model (with quantized weights if available)
    model = _load_model(cfg, ckpt_override=quant_ckpt_path)

    # 2) Build example input
    example_input = _build_example_input(cfg) # TODO fix - use 'real' inputs

    # 3) Frontend: Torch → ONNX
    onnx_path = os.path.join(cfg.out_dir, "model.onnx")
    _export_model_to_onnx(cfg, model, example_input, onnx_path)

    # 4) IR import: ONNX → Program
    program = build_program_from_onnx(
        onnx_path=onnx_path,
        default_backend=cfg.target_format,  # usually "ai8x"
        target_hardware=cfg.target_hardware,
        bit_width=cfg.bit_width,
        metadata={
            "input_shape": cfg.input_shape,
            "output_shape": cfg.output_shape,
            "input_names": cfg.input_names,
            "output_names": cfg.output_names,
            "target_format": cfg.target_format,
        },
    )

    # 5) Passes: legality + partitioning
    run_legality_check(program)
    # NOTE: we no longer run a separate IR quantization pass for ai8x here;
    #       real quantization has already been done by ai8x's convert_checkpoint.
    run_partitioning(program)

    print("here1")

    # 6) Backend compilation
    if cfg.target_format == "ai8x":
        run_ai8x_backend(program, cfg, onnx_path)
    else:
        log.warning("No backend implementation for target_format=%s", cfg.target_format)

    print("here2")

    # 7) Serialize Program
    program_json_path = write_program_json(program, cfg.out_dir)

    print("here3")

    log.info("Compile done.")
    log.info("  ONNX:     %s", onnx_path)
    log.info("  Program:  %s", program_json_path)
    for art in program.backend_artifacts:
        log.info("  Artifact: backend=%s type=%s path=%s",
                 art.backend, art.artifact_type, art.path)


# TODO
# - add a small pass that reads layer-wise quant config from the yaml (or from the quantized checkpoint) 
#   and annotates tensors with quantparams to match, w/o changing the actual quantization 
#   (since ai8x has already done it)
# - update to support full range of quant args
# - update to support more than INT8 quantization
# - update to quantize "in memory" i.e. don't resave the model checkpoint post-quantization

def _quantize_checkpoint_ai8x(cfg: CompileConfig) -> str:
    """Quantize the model checkpoint using ai8x's quantize.py (convert_checkpoint).

    Returns:
        Path to the *quantized* checkpoint file.
    """
    if not cfg.model_ckpt:
        raise CompilationError(
            "ai8x quantization requested but --model-ckpt is not provided."
        )
    if not cfg.ai8x_config_file:
        raise CompilationError(
            "ai8x quantization requires --ai8x-config-file (YAML network config)."
        )

    quant_ckpt = os.path.join(cfg.out_dir, "model_quantized.pth")

    # --- import ai8x modules ---
    # TODO move into own function
    try:
        import importlib
        quantize = importlib.import_module("izer.quantize")
        tc = importlib.import_module("izer.tornadocnn")
    except ImportError as exc:
        raise CompilationError(
            "Failed to import ai8x-synthesis modules (quantize, izer.tornadocnn). "
            "Make sure ai8x-synthesis is installed, e.g.:\n"
            "  pip install -e ../ai8x-synthesis"
        ) from exc

    # --- PyTorch 2.6+ safe_globals workaround ---
    import torch

    add_safe_globals = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add_safe_globals is not None:
        try:
            from torch.optim.adam import Adam
            # Allowlist the Adam class so torch.load(...) inside convert_checkpoint succeeds
            add_safe_globals([Adam])
        except Exception as e:  # noqa: BLE001
            # Not fatal: if this fails, we just fall back to default behaviour.
            logging.getLogger(__name__).warning(
                "Failed to add Adam to torch.serialization safe globals: %s", e
            )

    # --- build args for convert_checkpoint ---
    import argparse
    cfg.ai8x_device = 85 # TODO fix
    args = argparse.Namespace(
        config_file=cfg.ai8x_config_file,
        device=cfg.ai8x_device or 85,
        clip_mode=None,        # use MAX_BIT_SHIFT heuristic
        qat_weight_bits=None,
        verbose=cfg.debug,
        scale=None,
        stddev=None,
    )

    # configure device for quantizer
    tc.dev = tc.get_device(args.device)

    try:
        quantize.convert_checkpoint(cfg.model_ckpt, quant_ckpt, args)
    except Exception as exc:  # noqa: BLE001
        raise CompilationError(
            f"ai8x convert_checkpoint failed while quantizing '{cfg.model_ckpt}': {exc}"
        ) from exc

    if not os.path.exists(quant_ckpt):
        raise CompilationError(
            "ai8x quantization reported success but quantized checkpoint file "
            f"'{quant_ckpt}' does not exist."
        )
     
    return quant_ckpt

# ---------- ai8x backend ----------

def _build_ai8x_argv(cfg: CompileConfig, checkpoint_file: str, ai8x_dir: Path) -> list[str]:
    """
    Build a rich argv list for ai8x-synthesis izer, mirroring the Namespace
    we previously hand-constructed.
    """
    argv: list[str] = ["ai8x-izer"]  # argv[0], cosmetic

    argv += [
        "--device",
        str(cfg.ai8x_device or "MAX78000"),
        "--config-file",
        str(cfg.ai8x_config_file),
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

    # Everything is guaranteed to be a string now
    return argv



def run_ai8x_backend(program: Program, cfg: CompileConfig, onnx_path: str) -> None:
    """
    Invoke ai8x-synthesis' izer backend (izer/izer.py) to generate the C project,
    preserving full functionality.

    We:
      - import `izer.izer.main`,
      - build a rich CLI argv from CompileConfig,
      - temporarily replace sys.argv,
      - call main() and map SystemExit into CompilationError,
      - then register the generated project as a BackendArtifact.
    """
    try:
        izer_mod = importlib.import_module("izer.izer")
    except ImportError as exc:
        raise CompilationError(
            "Failed to import ai8x-synthesis 'izer.izer' module.\n"
            "Make sure ai8x-synthesis is installed and importable, e.g.:\n"
            "  cd ../ai8x-synthesis && pip install -e ."
        ) from exc

    out_dir = Path(cfg.out_dir)
    ai8x_dir = out_dir / "ai8x"
    ai8x_dir.mkdir(parents=True, exist_ok=True)

    # Prefer quantized checkpoint, then original ckpt, then ONNX
    quant_ckpt = out_dir / "model_quantized.pth"
    if quant_ckpt.exists():
        checkpoint_file = str(quant_ckpt)
    elif cfg.model_ckpt:
        checkpoint_file = cfg.model_ckpt
    else:
        checkpoint_file = onnx_path  # izer can also consume ONNX

    argv = _build_ai8x_argv(cfg, checkpoint_file, ai8x_dir)
    log.info("Invoking ai8x-synthesis izer with argv: %s", " ".join(map(str, argv)))

    old_argv = sys.argv
    sys.argv = argv
    try:
        izer_mod.main()  # their CLI entrypoint; uses commandline.get_parser() internally
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        if code != 0:
            raise CompilationError(f"ai8x-synthesis izer exited with code {code}") from e
    finally:
        sys.argv = old_argv

    # If we got here, izer completed successfully and should have created a project.
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


# ---------- Helper functions (unchanged from your earlier version) ----------

def _ensure_out_dir(out_dir: str, overwrite: bool) -> None:
    if os.path.exists(out_dir) and os.listdir(out_dir) and not overwrite:
        raise CompilationError(
            f"Output directory '{out_dir}' is not empty; use --overwrite to replace."
        )
    os.makedirs(out_dir, exist_ok=True)


def _sanity_check_files(cfg: CompileConfig) -> None:
    missing: list[str] = []
    for label, path, required in [
        ("model_py", cfg.model_py, True),
        ("model_ckpt", cfg.model_ckpt, False),
        ("data_sample", cfg.data_sample, False),
        ("ai8x_config_file", cfg.ai8x_config_file if cfg.target_format == "ai8x" else None, False),
    ]:
        if not path:
            if required:
                missing.append(label)
            continue
        if not os.path.exists(path):
            missing.append(f"{label} ({path})")
    if missing:
        raise CompilationError("Missing required files: " + ", ".join(missing))


def _import_module_from_file(path: str):
    spec = importlib.util.spec_from_file_location("unpu_model_module", path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["unpu_model_module"] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def _load_model(cfg: CompileConfig, ckpt_override: str | None = None) -> nn.Module:
    # Ensure ai8x is importable for ai8x model zoo nets
    try:
        import ai8x  # type: ignore[import]
        log.debug("Found real ai8x module: %s", ai8x)
    except ImportError:
        install_ai8x_shim()
        log.debug("Installed ai8x shim module for model import.")

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


def _import_module_from_file(path: str):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompilationError(f"Failed to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


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


def _export_model_to_onnx(
    cfg: CompileConfig,
    model: nn.Module,
    example_input: torch.Tensor,
    onnx_path: str,
) -> None:
    model.eval().to("cpu")
    input_names = [s.strip() for s in cfg.input_names.split(",") if s.strip()]
    output_names = [s.strip() for s in cfg.output_names.split(",") if s.strip()]
    if len(input_names) != 1:
        raise CompilationError("Current exporter only supports a single input tensor")
    import torch.onnx

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names or None,
        opset_version=17,
        dynamic_axes=None,
    )
    if not os.path.exists(onnx_path):
        raise CompilationError("ONNX export succeeded but file not found afterward")
