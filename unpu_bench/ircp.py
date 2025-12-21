# ircp.py (in ai8x-synthesis repo, e.g. ai8x/ircp.py)
import json
import numpy as np

from . import op as opn
from . import tornadocnn as tc
from .eprint import eprint

import sys


def load(
    ir_program_path,
    unused_arch,
    quantization,
    bias_quantization,
    output_shift,
    kernel_size,
    operator,
    verbose=False,
    no_bias=None,
):
    """
    Load weights and biases from an IR Program JSON (uNPU-Bench MuirProgram),
    and configure:
       - layers
       - weights (list of np.ndarray)
       - bias (list of np.ndarray or None)
       - output_shift (list)
       - input_channels (list)
       - output_channels (list)

    Signature mirrors onnxcp.load so izer can swap it in.
    """
    with open(ir_program_path, "r", encoding="utf-8") as f:
        prog = json.load(f)

    graph = prog["graph"]
    tensors = graph["tensors"]
    ops = graph["ops"]  # dict: op_id -> {kind, inputs, outputs, attrs, ...}

    # For simplicity, assume:
    #   - each ai8x-relevant op is a Conv/MatMul with:
    #       attrs["weights"] : nested list / shape [out_ch, in_ch, kH, kW] or [out, in]
    #       attrs["bias"]    : optional [out_ch]
    #
    # In other words, you'd need your uNPU-Bench IR → ai8x IR lowering to *already*
    # materialize weights/biases as arrays on each op.
    #
    # This code then just re-packs them into the structures ai8x expects.

    weights = []
    bias = []
    input_channels = []
    output_channels = []

    # We maintain the same 'layers' / 'seq' indexing as YAML quantization
    seq = 0
    layers = 0
    errors = False

    for op_id, op in ops.items():
        kind = op["kind"]
        if kind not in ("Conv", "Gemm"):
            # Skip non-parametric ops: activations, reshapes, etc.
            continue

        w_arr = op["attrs"].get("weights", None)
        b_arr = op["attrs"].get("bias", None)

        if w_arr is None:
            eprint(f"IR op {op_id} ({kind}) missing 'weights' in attrs.", exit_code=None)
            errors = True
            continue

        w = np.array(w_arr, dtype=np.int64)  # or your chosen type

        # Derive input/output channels
        if kind == "Gemm":
            # [out, in] or [1, out, in] depending on IR; normalize
            if w.ndim == 2:
                out_ch, in_ch = w.shape
            elif w.ndim == 3:
                out_ch, in_ch = w.shape[1], w.shape[2]
            else:
                eprint(f"Unexpected GEMM weight shape for op {op_id}: {w.shape}", exit_code=None)
                errors = True
                continue
            kH, kW = 1, 1
        else:  # Conv
            # Expect [out_ch, in_ch, kH, kW]
            if w.ndim != 4:
                eprint(f"Unexpected Conv weight shape for op {op_id}: {w.shape}", exit_code=None)
                errors = True
                continue
            out_ch, in_ch, kH, kW = w.shape

        input_channels.append(in_ch)
        output_channels.append(out_ch)

        # Handle kernel_size compatibility check similar to onnxcp
        # (you can reuse that logic, comparing to 'kernel_size[seq]')

        # Bias
        if b_arr is None or (no_bias and seq in no_bias):
            bias.append(None)
        else:
            b = np.array(b_arr, dtype=np.int64)
            bias.append(b)

        # Save weights (possibly reshaped for ai8x)
        weights.append(w)
        layers += 1
        seq += 1

    if errors:
        # match onnxcp behaviour: exit(1) or raise
        sys.exit(1)

    # For now, we leave output_shift untouched; izer/yamlcfg will
    # handle defaulting logic. You could also set them from IR QuantParams.

    if verbose:
        print(f"IRCP: loaded {layers} param layers from {ir_program_path}")

    return layers, weights, bias, output_shift, input_channels, output_channels
