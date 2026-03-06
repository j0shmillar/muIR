from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .reporting import write_cross_backend_report


def _collect_program_jsons(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    for p in args.program_json:
        paths.append(str(Path(p)))
    for d in args.program_dir:
        root = Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob(args.glob)):
            if p.is_file() and p.name == "program.json":
                paths.append(str(p))
    return sorted(set(paths))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unpu-bench-report",
        description="Generate cross-backend comparison reports from program.json files.",
    )
    p.add_argument(
        "--program-json",
        action="append",
        default=[],
        help="Path to a program.json (repeatable).",
    )
    p.add_argument(
        "--program-dir",
        action="append",
        default=[],
        help="Directory to recursively scan for program.json files (repeatable).",
    )
    p.add_argument(
        "--glob",
        default="**/program.json",
        help="Glob pattern used under each --program-dir (default: **/program.json).",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for report files."
    )
    p.add_argument(
        "--basename",
        default="cross_backend_report",
        help="Output filename base (without extension).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    program_jsons = _collect_program_jsons(args)
    if not program_jsons:
        raise SystemExit(
            "No program.json inputs found. Use --program-json or --program-dir."
        )

    paths = write_cross_backend_report(
        program_jsons,
        out_dir=args.out_dir,
        basename=args.basename,
    )
    print(f"CSV: {paths['csv']}")
    print(f"Markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
