from __future__ import annotations

from .errors import CompilationError
from .muir import Program

CURRENT_IR_SCHEMA_VERSION = 1
MIN_READER_SCHEMA_VERSION = 1


def migrate_program_ir_metadata(program: Program) -> None:
    meta = program.metadata
    version = int(meta.get("ir_schema_version", 0) or 0)
    if version <= 0:
        meta["ir_schema_version"] = CURRENT_IR_SCHEMA_VERSION
        meta["ir_schema_min_reader_version"] = MIN_READER_SCHEMA_VERSION
        meta["ir_schema_features"] = [
            "canonicalized_layout_attrs",
            "partition_metrics_v1",
            "quant_contract_v1",
        ]
        return

    # Future-proof migration hook.
    if version == 1:
        meta.setdefault("ir_schema_min_reader_version", MIN_READER_SCHEMA_VERSION)
        meta.setdefault(
            "ir_schema_features",
            [
                "canonicalized_layout_attrs",
                "partition_metrics_v1",
                "quant_contract_v1",
            ],
        )


def validate_program_ir_metadata(program: Program) -> None:
    meta = program.metadata
    version = int(meta.get("ir_schema_version", 0) or 0)
    min_reader = int(meta.get("ir_schema_min_reader_version", 0) or 0)
    if version < MIN_READER_SCHEMA_VERSION:
        raise CompilationError(
            f"IR schema version {version} is below minimum supported {MIN_READER_SCHEMA_VERSION}"
        )
    if min_reader > CURRENT_IR_SCHEMA_VERSION:
        raise CompilationError(
            f"IR requires reader version {min_reader}, current reader is {CURRENT_IR_SCHEMA_VERSION}"
        )
