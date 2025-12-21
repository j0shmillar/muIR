# unpu_bench/passes.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .muir import Program, Graph, Op, QuantParams, Tensor, Partition
from .errors import CompilationError


# ---------- 1) Legality for ai8x ----------

# Very rough list; you can refine based on ai8x docs.
AI8X_LEGAL_KINDS = {
    "Conv",
    "ConvTranspose",  # some variants may not be supported
    "Gemm",
    "MatMul",
    "Relu",
    "LeakyRelu",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "BatchNormalization",
    "Add",
    "Mul",
    "Concat",
    "Reshape",
    "Transpose",
    "Flatten",
    "Softmax",  # often done on CPU but ai8x supports small last-layer softmax via C
    "Constant"
}


def run_legality_check(program: Program) -> None:
    """Annotate ops with legal_backends for simple backends (ai8x, cpu)."""
    for op in program.graph.ops.values():
        legal = ["cpu"]  # CPU is always legal fallback

        if op.kind in AI8X_LEGAL_KINDS:
            legal.append("ai8x")

        op.legal_backends = legal
        # default pick: prefer ai8x if legal
        op.preferred_backend = "ai8x" if "ai8x" in legal else "cpu"


# ---------- 2) Partitioning ----------

def run_partitioning(program: Program) -> None:
    """Partition the program into up to three segments:
       - optional CPU prefix
       - required ai8x core (if any ai8x-legal ops exist)
       - optional CPU suffix

    Rules:
      - Any op whose preferred_backend == "ai8x" must be in the ai8x core.
      - Any op whose preferred_backend != "ai8x" must be in prefix or suffix.
      - If a non-ai8x op is found *between* two ai8x ops, we error out.
    """
    ops_order = list(program.graph.ops.keys())
    if not ops_order:
        program.partitions = []
        return

    # Label each op as "ai8x" or "cpu" based on preferred_backend
    kinds: List[str] = []
    for op_id in ops_order:
        op = program.graph.ops[op_id]
        backend = op.preferred_backend or "cpu"
        kinds.append("ai8x" if backend == "ai8x" else "cpu")

    # Find the first and last ai8x ops
    try:
        first_ai8x = kinds.index("ai8x")
        last_ai8x = len(kinds) - 1 - kinds[::-1].index("ai8x")
    except ValueError:
        # No ai8x ops at all: everything on CPU
        program.partitions = [
            Partition(id="cpu_full", backend="cpu", op_ids=ops_order),
        ]
        return

    # Ensure there are no CPU ops in the core [first_ai8x, last_ai8x]
    for idx in range(first_ai8x, last_ai8x + 1):
        if kinds[idx] != "ai8x":
            bad_op_id = ops_order[idx]
            bad_op = program.graph.ops[bad_op_id]
            raise CompilationError(
                f"Op '{bad_op_id}' ({bad_op.kind}) is not ai8x-legal but "
                "appears in the middle of the NPU segment. "
                "Only CPU prefix/suffix around a single ai8x core are supported."
            )

    partitions: List[Partition] = []

    # CPU prefix
    if first_ai8x > 0:
        prefix_ids = ops_order[:first_ai8x]
        partitions.append(
            Partition(
                id="cpu_prefix",
                backend="cpu",
                op_ids=prefix_ids,
            )
        )

    # ai8x core
    core_ids = ops_order[first_ai8x : last_ai8x + 1]
    partitions.append(
        Partition(
            id="ai8x_core",
            backend="ai8x",
            op_ids=core_ids,
        )
    )

    # CPU suffix
    if last_ai8x < len(ops_order) - 1:
        suffix_ids = ops_order[last_ai8x + 1 :]
        partitions.append(
            Partition(
                id="cpu_suffix",
                backend="cpu",
                op_ids=suffix_ids,
            )
        )

    program.partitions = partitions
