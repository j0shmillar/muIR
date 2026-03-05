from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import yaml

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when user configuration or platforms.yaml is invalid."""


@dataclass
class FlagSpec:
    name: str
    type: str  # "int", "float", "str", "bool"
    default: Optional[Any] = None
    help: str | None = None


@dataclass
class PlatformSpec:
    name: str
    depends_on: tuple[str, ...]
    flags: Dict[str, FlagSpec]
    bit_widths: tuple[Any, ...] = ()
    compatible_hardware: tuple[str, ...] = ()


def load_platforms_config(path: str = "platforms.yaml") -> Dict[str, PlatformSpec]:
    """Load and minimally validate platforms.yaml."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing platforms.yaml at {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse platforms.yaml: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("platforms.yaml must contain a mapping at the top level.")

    if "formats" in raw and isinstance(raw.get("formats"), Mapping):
        raw_formats = raw["formats"]
    else:
        # Backward compatibility: format specs at top-level.
        raw_formats = {k: v for k, v in raw.items() if k != "hardware"}

    platforms: Dict[str, PlatformSpec] = {}
    for fmt_name, spec in raw_formats.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"Platform '{fmt_name}' spec must be a mapping.")

        depends_raw = spec.get("depends_on", []) or []
        if not isinstance(depends_raw, (list, tuple)):
            raise ConfigError(f"'depends_on' for '{fmt_name}' must be a list.")
        depends = tuple(str(d) for d in depends_raw)

        flags_raw = spec.get("flags", {}) or {}
        if not isinstance(flags_raw, Mapping):
            raise ConfigError(f"'flags' for '{fmt_name}' must be a mapping.")

        flags: Dict[str, FlagSpec] = {}
        for flag_name, flag_spec in flags_raw.items():
            if not isinstance(flag_spec, Mapping):
                raise ConfigError(f"Flag '{flag_name}' in '{fmt_name}' must be a mapping.")
            type_ = flag_spec.get("type")
            action = flag_spec.get("action")
            if type_ is None and action in {"store_true", "store_false"}:
                type_ = "bool"
            if type_ is None:
                raise ConfigError(
                    f"Flag '{flag_name}' in '{fmt_name}' is missing required key 'type' "
                    "or a supported 'action'."
                )
            if type_ not in {"int", "float", "str", "bool"}:
                raise ConfigError(
                    f"Flag '{flag_name}' in '{fmt_name}' has invalid type '{type_}'. "
                    "Allowed: int, float, str, bool."
                )
            flags[flag_name] = FlagSpec(
                name=flag_name,
                type=str(type_),
                default=flag_spec.get("default"),
                help=flag_spec.get("help"),
            )

        platforms[fmt_name] = PlatformSpec(
            name=fmt_name,
            depends_on=depends,
            flags=flags,
            bit_widths=tuple(spec.get("bit_widths", ()) or ()),
            compatible_hardware=tuple(spec.get("compatible_hardware", ()) or ()),
        )

    log.debug("Loaded %d platforms from platforms.yaml", len(platforms))
    return platforms


@dataclass
class CoreConfig:
    target_format: str
    target_hardware: str
    bit_width: int


# Simple hard-coded rules for now; extend as needed.
_SUPPORTED_TARGETS: Dict[tuple[str, str], tuple[int, ...]] = {
    ("ai8x", "max78000"): (1, 2, 4, 8),
    ("ai8x", "max78002"): (1, 2, 4, 8),
    ("tflm", "hxwe2"): (8,),
    ("vela", "hxwe2"): (8,),
    # Add MCXN947, CVI, etc.
}


def validate_core_config(core: CoreConfig, platforms: Dict[str, PlatformSpec]) -> None:
    """Validate core combination of format/hardware/bit-width using platforms.yaml."""
    if core.target_format not in platforms:
        raise ConfigError(
            f"Unknown target_format '{core.target_format}'. "
            f"Known: {', '.join(sorted(platforms.keys()))}"
        )

    platform = platforms[core.target_format]
    if platform.compatible_hardware and core.target_hardware not in platform.compatible_hardware:
        raise ConfigError(
            f"target_hardware='{core.target_hardware}' is not compatible with "
            f"target_format='{core.target_format}'. Allowed: "
            f"{', '.join(sorted(platform.compatible_hardware))}"
        )

    if platform.bit_widths:
        allowed_from_spec = set(platform.bit_widths)
        if core.bit_width not in allowed_from_spec and str(core.bit_width) not in allowed_from_spec:
            raise ConfigError(
                f"bit_width={core.bit_width} is not supported for {core.target_format}. "
                f"Supported: {sorted(platform.bit_widths, key=str)}"
            )
        return

    key = (core.target_format, core.target_hardware)
    if key not in _SUPPORTED_TARGETS:
        log.warning(
            "No explicit validation rules for (format=%s, hardware=%s); "
            "skipping bit-width validation. Consider adding it to config._SUPPORTED_TARGETS.",
            core.target_format,
            core.target_hardware,
        )
        return

    allowed_bits = _SUPPORTED_TARGETS[key]
    if core.bit_width not in allowed_bits:
        raise ConfigError(
            f"bit_width={core.bit_width} is not supported for "
            f"{core.target_format}/{core.target_hardware}. "
            f"Supported: {sorted(allowed_bits)}"
        )

    log.info(
        "Configuration validated for %s/%s (bit_width=%d)",
        core.target_format,
        core.target_hardware,
        core.bit_width,
    )
