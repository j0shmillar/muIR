from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from .version import __version__

log = logging.getLogger(__name__)


@dataclass
class ToolchainInfo:
    name: str
    version: Optional[str]
    path: Optional[str]


@dataclass
class RunMetadata:
    timestamp_utc: str
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    unpu_bench_version: str
    platforms_yaml_hash: Optional[str]
    args: Dict[str, Any]
    toolchains: Sequence[ToolchainInfo]


def _safe_run(cmd: Sequence[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _hash_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_toolchains() -> Sequence[ToolchainInfo]:
    """Placeholder toolchain detection.

    Extend this to actually detect ai8x-training, Vela, eIQ, etc.
    """
    env_candidates = [
        ("ai8x-training", os.environ.get("AI8X_TRAIN_PATH")),
        ("ai8x-synthesis", os.environ.get("AI8X_SYNTH_PATH")),
        ("vela", os.environ.get("VELA_PATH")),
        ("eiq", os.environ.get("EIQ_PATH")),
    ]

    tools: list[ToolchainInfo] = []
    for name, path in env_candidates:
        if not path:
            continue
        tools.append(ToolchainInfo(name=name, version=None, path=path))
    return tools


def write_run_metadata(out_dir: str, args_namespace: Any) -> str:
    """Write metadata.json into `out_dir` describing this compile run.

    Returns:
        Path to metadata.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    git_commit = _safe_run(["git", "rev-parse", "HEAD"])
    git_status = _safe_run(["git", "status", "--porcelain"])
    git_dirty = bool(git_status) if git_status is not None else None

    platforms_hash = _hash_file("platforms.yaml")

    args_dict = vars(args_namespace) if hasattr(args_namespace, "__dict__") else {}

    meta = RunMetadata(
        timestamp_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        git_commit=git_commit,
        git_dirty=git_dirty,
        unpu_bench_version=__version__,
        platforms_yaml_hash=platforms_hash,
        args=args_dict,
        toolchains=_detect_toolchains(),
    )

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(meta),
                "toolchains": [asdict(t) for t in meta.toolchains],
            },
            f,
            indent=2,
            sort_keys=True,
        )

    log.info("Wrote run metadata to %s", meta_path)
    return meta_path
