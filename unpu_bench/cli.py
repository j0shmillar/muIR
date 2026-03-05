from __future__ import annotations

import argparse
import logging
import shlex
from typing import Any, Dict, Optional, List
from pathlib import Path

from .config import load_platforms_config, CoreConfig, validate_core_config, ConfigError
from .metadata import write_run_metadata
from .pipeline import CompileConfig, compile_model
from .version import __version__

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unpu-bench",
        description="µNPU model compiler and build tool",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"unpu-bench {__version__}",
    )

    # --- Global options ---
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
    )

    # --- Core compile options ---
    parser.add_argument("--target-format", required=True)
    parser.add_argument("--target-hardware", required=True)
    parser.add_argument("--bit-width", type=int, default=8)

    parser.add_argument("--model-py")
    parser.add_argument("--model-class")
    parser.add_argument("--model-ckpt", dest="model_ckpt")  # note: dash -> underscore
    parser.add_argument("--model-onnx", dest="model_onnx")
    parser.add_argument("--model-tflite", dest="model_tflite")
    parser.add_argument("--model-args", default="{}")  # JSON string (if you want)

    parser.add_argument("--input-shape", default="")
    parser.add_argument("--output-shape", default="")
    parser.add_argument("--input-names", default="")
    parser.add_argument("--output-names", default="")
    parser.add_argument("--data-sample")

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # --- ai8x backend options ---
    ai8x_group = parser.add_argument_group("ai8x backend options")
    ai8x_group.add_argument(
        "--ai8x-config-file",
        type=str,
        help="Path to ai8x YAML network config file (required for ai8x backend).",
    )
    ai8x_group.add_argument(
        "--ai8x-device",
        type=str,
        default="MAX78000",
        help="ai8x device name, e.g. MAX78000 or MAX78002.",
    )
    ai8x_group.add_argument(
        "--ai8x-prefix",
        type=str,
        default="unpu_model",
        help="Prefix/name for ai8x-generated project.",
    )

    # --- Hardware artifact emission ---
    hw_group = parser.add_argument_group("hardware artifact options")
    hw_group.add_argument(
        "--emit-hardware-artifact",
        action="store_true",
        help="Emit backend-native hardware runnable artifact (e.g., Vela-optimized .tflite).",
    )
    hw_group.add_argument(
        "--backend-source-model",
        type=str,
        default=None,
        help="Input model for backend toolchain (e.g., .tflite for Vela, .onnx for external tools).",
    )
    hw_group.add_argument(
        "--backend-tool-args",
        type=str,
        default="",
        help="Extra args for built-in backend tool invocation (shell-style string).",
    )
    hw_group.add_argument(
        "--backend-command",
        type=str,
        default=None,
        help="Custom backend command template. Supports {input}, {out_dir}, {backend}.",
    )
    hw_group.add_argument(
        "--backend-output-glob",
        type=str,
        default=None,
        help="Glob to locate external backend output artifact inside backend out dir.",
    )
    cvi_group = parser.add_argument_group("cvi backend options")
    cvi_group.add_argument("--cvi-calibration-table", type=str, default=None)
    cvi_group.add_argument("--cvi-tolerance", type=float, default=0.99)
    cvi_group.add_argument("--cvi-dynamic", action="store_true")
    cvi_group.add_argument("--cvi-excepts", type=str, default=None)
    cvi_group.add_argument("--cvi-resize-dims", type=str, default=None)
    cvi_group.add_argument("--cvi-pixel-format", type=str, default=None)
    cvi_group.add_argument("--cvi-test-result", type=str, default=None)
    cvi_group.add_argument("--cvi-keep-aspect-ratio", action="store_true")

    return parser


def _parse_model_args(raw: str) -> Dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise ConfigError(f"Failed to parse --model-args as JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("--model-args must be a JSON object/dict.")
    return value


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()

    # handle --help/--version nicely for tests
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code)

    # basic logging setup
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        filename=args.log_file,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Load platforms YAML if you use it (your existing logic)
    platforms_path = Path("platforms.yaml")
    if platforms_path.exists():
        platforms = load_platforms_config(platforms_path)
    else:
        platforms = {}

    try:
        model_args = _parse_model_args(args.model_args)
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return 2

    if platforms:
        try:
            validate_core_config(
                CoreConfig(
                    target_format=args.target_format,
                    target_hardware=args.target_hardware,
                    bit_width=args.bit_width,
                ),
                platforms,
            )
        except ConfigError as exc:
            log.error("Configuration error: %s", exc)
            return 2

    cfg = CompileConfig(
        target_format=args.target_format,
        target_hardware=args.target_hardware,
        bit_width=args.bit_width,
        model_py=args.model_py,
        model_class=args.model_class,
        model_ckpt=args.model_ckpt,
        model_args=model_args,
        model_onnx=args.model_onnx,
        model_tflite=args.model_tflite,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
        input_names=args.input_names,
        output_names=args.output_names,
        data_sample=args.data_sample,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        debug=args.debug,
        # ai8x-specific
        ai8x_root=None,  # or an arg if you want
        ai8x_device=args.ai8x_device,
        ai8x_config_file=args.ai8x_config_file,
        ai8x_prefix=args.ai8x_prefix,
        emit_hardware_artifact=args.emit_hardware_artifact,
        backend_source_model=args.backend_source_model,
        backend_tool_args=(
            shlex.split(args.backend_tool_args) if args.backend_tool_args else []
        ),
        backend_command=args.backend_command,
        backend_output_glob=args.backend_output_glob,
        cvi_calibration_table=args.cvi_calibration_table,
        cvi_tolerance=args.cvi_tolerance,
        cvi_dynamic=args.cvi_dynamic,
        cvi_excepts=args.cvi_excepts,
        cvi_resize_dims=args.cvi_resize_dims,
        cvi_pixel_format=args.cvi_pixel_format,
        cvi_test_result=args.cvi_test_result,
        cvi_keep_aspect_ratio=args.cvi_keep_aspect_ratio,
    )

    try:
        write_run_metadata(args.out_dir, args)
        compile_model(cfg, platforms)
    except Exception as exc:  # noqa: BLE001
        log.error("Compilation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
