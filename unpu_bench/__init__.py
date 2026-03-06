from __future__ import annotations

"""Top-level package for uNPU-Bench Python tooling."""

from .version import __version__
from .logging_setup import configure_logging
from .config import ConfigError, load_platforms_config, validate_core_config
from .metadata import write_run_metadata

__all__ = [
    "__version__",
    "configure_logging",
    "load_platforms_config",
    "validate_core_config",
    "ConfigError",
    "write_run_metadata",
    "CompileConfig",
    "compile_model",
    "convert",
    "compare_runs",
]


def __getattr__(name: str):
    # Avoid importing heavy compile dependencies (torch, backend toolchains)
    # on package import; load them only when requested.
    if name in {"CompileConfig", "compile_model"}:
        from .pipeline import CompileConfig, compile_model

        if name == "CompileConfig":
            return CompileConfig
        return compile_model
    if name in {"convert", "compare_runs"}:
        from .api import compare_runs, convert

        if name == "convert":
            return convert
        return compare_runs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
