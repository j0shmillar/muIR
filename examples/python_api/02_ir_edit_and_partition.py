from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reference_impls import REFERENCE_MODEL_REGISTRY
from unpu_bench.backend_lowering import lower_program_for_backend
from unpu_bench.errors import CompilationError
from unpu_bench.muir import build_program_from_torch, write_program_json
from unpu_bench.passes import (
    run_ir_canonicalization,
    run_ir_validation,
    run_legality_check,
    run_partitioning,
)


def main() -> None:
    out_dir = Path("out/examples/02_ir_edit_partition")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "resnet18"
    model = REFERENCE_MODEL_REGISTRY[model_name](num_classes=10).eval()
    ckpt = Path("ckpts/random_reference") / f"{model_name}.random.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    x = torch.randn(1, 3, 32, 32)
    program = build_program_from_torch(
        model=model,
        example_input=x,
        default_backend="vela",
        target_hardware="hxwe2",
        bit_width=8,
        metadata={"demo": "ir-edit-partition"},
    )

    run_ir_canonicalization(program)
    run_ir_validation(program)

    caps = Path("unpu_bench/capabilities/ir_vela.yaml")
    run_legality_check(program, backend="vela", caps_path=caps)

    # IR-level edit: force one early op to CPU to demonstrate backend-agnostic partitioning behavior.
    if len(program.graph.op_order) > 2:
        force_id = program.graph.op_order[1]
        program.graph.ops[force_id].preferred_backend = "cpu"
        program.graph.ops[force_id].legal_backends = ["cpu"]
        program.metadata["forced_cpu_op"] = force_id

    try:
        run_partitioning(program, backend="vela", fallback_backend="cpu")
    except CompilationError:
        for op_id in program.graph.op_order:
            op = program.graph.ops[op_id]
            op.preferred_backend = "cpu"
            op.legal_backends = ["cpu"]
        run_partitioning(program, backend="vela", fallback_backend="cpu")
        program.metadata["partition_fallback"] = "cpu_full"

    for artifact in lower_program_for_backend(
        program, target_format="vela", out_dir=str(out_dir)
    ):
        program.add_artifact(artifact)
    write_program_json(program, str(out_dir))

    print("Partitions:", [(p.id, p.backend, len(p.op_ids)) for p in program.partitions])
    print("Program:", out_dir / "program.json")


if __name__ == "__main__":
    main()
