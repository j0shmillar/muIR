from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional, List
from pathlib import Path

from .logging_setup import configure_logging
from .config import load_platforms_config, CoreConfig, validate_core_config, ConfigError
from .metadata import write_run_metadata
from .pipeline import CompileConfig, compile_model, CompilationError
from .version import __version__

log = logging.getLogger(__name__)


# unpu_bench/cli.py

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unpu-bench",
        description="µNPU model compiler and build tool",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="unpu-bench 0.1.0",
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

    parser.add_argument("--model-py", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--model-ckpt", dest="model_ckpt")  # note: dash -> underscore
    parser.add_argument("--model-args", default="{}")       # JSON string (if you want)

    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--output-shape", required=True)
    parser.add_argument("--input-names", required=True)
    parser.add_argument("--output-names", required=True)
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

    return parser


def _parse_model_args(raw: str) -> Dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: TRY003
        raise ConfigError(
            f"Failed to parse --model-args as JSON: {exc}"
        ) from exc
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

    # Convert model_args JSON string if you use it
    import json
    try:
        model_args = json.loads(args.model_args)
    except Exception:
        model_args = {}

    cfg = CompileConfig(
        target_format=args.target_format,
        target_hardware=args.target_hardware,
        bit_width=args.bit_width,
        model_py=args.model_py,
        model_class=args.model_class,
        model_ckpt=args.model_ckpt,
        model_args=model_args,
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
    )

    try:
        compile_model(cfg, platforms)
    except Exception as exc:  # noqa: BLE001
        log.error("Compilation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
