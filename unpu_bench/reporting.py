from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import CompilationError


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def summarize_program_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CompilationError(f"program.json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    pm = meta.get("partition_metrics", {})
    backend = (
        pm.get("target_backend")
        or meta.get("target_format")
        or data.get("target_format")
        or "unknown"
    )
    graph = data.get("graph", {})
    op_total = _as_int(pm.get("ops_total", len(graph.get("op_order", []) or [])))
    op_target = _as_int(pm.get("ops_on_target_backend", 0))
    op_fallback = _as_int(pm.get("ops_on_fallback_backend", 0))

    artifacts = data.get("backend_artifacts", [])
    ir_native = [
        a
        for a in artifacts
        if isinstance(a, dict) and a.get("meta", {}).get("vendor_toolchain") is False
    ]
    vendor = [
        a
        for a in artifacts
        if isinstance(a, dict) and a.get("meta", {}).get("vendor_toolchain") is True
    ]

    vendor_paths = sorted(str(a.get("path", "")) for a in vendor if a.get("path"))
    vendor_formats = sorted(
        str(a.get("meta", {}).get("format", ""))
        for a in vendor
        if a.get("meta", {}).get("format")
    )

    return {
        "program_json": str(p),
        "model_name": graph.get("name", p.parent.name),
        "backend": str(backend),
        "topology": str(pm.get("offload_topology", "unknown")),
        "topology_valid": bool(pm.get("topology_valid", False)),
        "ops_total": op_total,
        "ops_target": op_target,
        "ops_fallback": op_fallback,
        "fallback_ratio": (float(op_fallback) / op_total) if op_total > 0 else 0.0,
        "partition_count": _as_int(pm.get("partition_count", 0)),
        "core_partition_count": _as_int(pm.get("core_partition_count", 0)),
        "cut_count": _as_int(pm.get("cut_count", 0)),
        "boundary_tensor_count": _as_int(pm.get("boundary_tensor_count", 0)),
        "cost_proxy": _as_float(pm.get("cost_proxy", 0.0)),
        "ir_native_artifact_count": len(ir_native),
        "vendor_artifact_count": len(vendor),
        "vendor_artifact_formats": ";".join(vendor_formats),
        "vendor_artifact_paths": ";".join(vendor_paths),
    }


def build_cross_backend_report(
    program_jsons: Iterable[str | Path],
) -> list[dict[str, Any]]:
    rows = [summarize_program_json(p) for p in program_jsons]
    rows.sort(key=lambda r: (r["model_name"], r["backend"]))
    return rows


def write_cross_backend_report(
    program_jsons: Iterable[str | Path],
    *,
    out_dir: str | Path,
    basename: str = "cross_backend_report",
) -> dict[str, str]:
    rows = build_cross_backend_report(program_jsons)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{basename}.csv"
    md_path = out / f"{basename}.md"

    if rows:
        fields = list(rows[0].keys())
    else:
        fields = [
            "program_json",
            "model_name",
            "backend",
            "topology",
            "topology_valid",
            "ops_total",
            "ops_target",
            "ops_fallback",
            "fallback_ratio",
            "partition_count",
            "core_partition_count",
            "cut_count",
            "boundary_tensor_count",
            "cost_proxy",
            "ir_native_artifact_count",
            "vendor_artifact_count",
            "vendor_artifact_formats",
            "vendor_artifact_paths",
        ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"csv": str(csv_path), "markdown": str(md_path)}
