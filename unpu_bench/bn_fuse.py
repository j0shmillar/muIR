# unpu_bench/bn_fuse.py (or whatever filename you used)

from __future__ import annotations

import logging
from typing import Dict, Any

import torch

log = logging.getLogger(__name__)


def fuse_batchnorm_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Fuse BatchNorm parameters into Conv2d weights/biases IN-PLACE.

    Handles ai8x-style checkpoints (layers with '.op' or '.conv2d')
    and the shim-style FusedConv2dBNReLU / FusedMaxPoolConv2dBNReLU
    (layers with '.conv').
    """
    dict_keys = list(state_dict.keys())
    convbn_layers = {
        k.rsplit(".", 3)[0] for k in dict_keys if k.endswith(".bn.running_mean")
    }

    if not convbn_layers:
        log.info("BN fuse: no Conv+BN layers detected in state_dict.")
        return state_dict

    log.info("BN fuse: found %d Conv+BN layers to fuse.", len(convbn_layers))
    fused_count = 0
    skipped_layers: list[str] = []

    for layer in sorted(convbn_layers):
        # ai8x native: '<layer>.op.weight' or '<layer>.conv2d.weight'
        # shim: '<layer>.conv.weight'
        if f"{layer}.op.weight" in state_dict:
            conv_key = f"{layer}.op"
        elif f"{layer}.conv2d.weight" in state_dict:
            conv_key = f"{layer}.conv2d"
        elif f"{layer}.conv.weight" in state_dict:
            conv_key = f"{layer}.conv"
        else:
            log.warning(
                "BN fuse: skipping layer '%s' because no matching conv weight "
                "('<layer>.op.weight' or '<layer>.conv2d.weight' or '<layer>.conv.weight') "
                "was found.",
                layer,
            )
            skipped_layers.append(layer)
            continue

        w_key = f"{conv_key}.weight"
        b_key = f"{conv_key}.bias"

        bn_key = f"{layer}.bn"
        r_mean_key = f"{bn_key}.running_mean"
        r_var_key = f"{bn_key}.running_var"
        beta_key = f"{bn_key}.weight"
        gamma_key = f"{bn_key}.bias"
        batches_key = f"{bn_key}.num_batches_tracked"

        try:
            w = state_dict[w_key]
        except KeyError:
            log.warning(
                "BN fuse: skipping layer '%s' (no conv weight key '%s').",
                layer,
                w_key,
            )
            skipped_layers.append(layer)
            continue

        device = w.device
        b = state_dict.get(b_key, torch.zeros(w.shape[0], device=device))

        if r_mean_key not in state_dict or r_var_key not in state_dict:
            log.warning(
                "BN fuse: missing running stats for layer '%s' "
                "(keys '%s' or '%s'); skipping.",
                layer,
                r_mean_key,
                r_var_key,
            )
            skipped_layers.append(layer)
            continue

        r_mean = state_dict[r_mean_key]
        r_var = state_dict[r_var_key]
        r_std = torch.sqrt(r_var + 1e-20)

        beta = state_dict.get(beta_key, torch.ones(w.shape[0], device=device))
        gamma = state_dict.get(gamma_key, torch.zeros(w.shape[0], device=device))

        # Preserve ai8x’s original scaling trick
        beta = 0.25 * beta
        gamma = 0.25 * gamma

        shape_reshape = (w.shape[0],) + (1,) * (len(w.shape) - 1)
        w_new = w * (beta / r_std).reshape(shape_reshape)
        b_new = (b - r_mean) / r_std * beta + gamma

        state_dict[w_key] = w_new
        state_dict[b_key] = b_new

        for k in (r_mean_key, r_var_key, beta_key, gamma_key, batches_key):
            if k in state_dict:
                del state_dict[k]

        fused_count += 1
        log.debug("BN fuse: fused layer '%s' into '%s'.", layer, conv_key)

    log.info(
        "BN fuse: successfully fused %d layers. Skipped %d layers.",
        fused_count,
        len(skipped_layers),
    )
    if skipped_layers:
        log.debug("BN fuse: skipped layers: %s", ", ".join(skipped_layers))

    return state_dict


def fuse_batchnorm_in_checkpoint(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    if "state_dict" not in checkpoint:
        raise KeyError("BN fuse: checkpoint missing 'state_dict' key.")

    log.info("BN fuse: starting BN fusion inside checkpoint.")
    checkpoint["state_dict"] = fuse_batchnorm_in_state_dict(checkpoint["state_dict"])
    log.info("BN fuse: finished BN fusion inside checkpoint.")
    return checkpoint
