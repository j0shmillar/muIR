from __future__ import annotations

"""Top-level package for uNPU-Bench Python tooling."""

from .version import __version__
from .logging_setup import configure_logging
from .config import load_platforms_config, validate_core_config, ConfigError
from .metadata import write_run_metadata
from .pipeline import CompileConfig, compile_model

__all__ = [
    "__version__",
    "configure_logging",
    "load_platforms_config",
    "validate_core_config",
    "ConfigError",
    "write_run_metadata",
    "CompileConfig",
    "compile_model",
]
