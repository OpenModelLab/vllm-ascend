# SPDX-License-Identifier: Apache-2.0
"""GLM CT INT4/INT8 projection configuration; no checkpoint rewriting."""

import torch


def is_glm_ct_int8_config(config) -> bool:
    """Enable INT8 projections without changing legacy BF16/FP8 checkpoints.

    The supported mixed profile uses per-channel symmetric INT4/INT8 weights
    and dynamic per-token INT8 inputs. Convolutions/recurrent state are separate.
    """
    if config is None or config.get_name() != "compressed-tensors":
        return False
    schemes = list(getattr(config, "target_scheme_map", {}).values())
    if not schemes:
        return False
    for scheme in schemes:
        if scheme is None:
            continue
        weight, activation = scheme.get("weights"), scheme.get("input_activations")
        if weight is None or activation is None:
            return False
        if not (
            weight.type == activation.type == "int"
            and weight.num_bits in (4, 8)
            and weight.strategy == "channel"
            and weight.symmetric
            and not weight.dynamic
            and activation.num_bits == 8
            and activation.strategy == "token"
            and activation.symmetric
            and activation.dynamic is True
        ):
            return False
    return any(scheme is not None for scheme in schemes)


def glm5next_packed_mapping(inherited=None) -> dict[str, list[str]]:
    """Return fresh mappings, preserving inherited vision projections."""
    mapping = {name: list(parts) for name, parts in (inherited or {}).items()}
    mapping.update(
        gate_up_proj=["gate_proj", "up_proj"],
        fused_qkv_a_proj=["q_a_proj", "kv_a_proj_with_mqa"],
        wk_weights_proj=["wk", "weights_proj"],
        in_proj_qkvbfg_a=["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    )
    return mapping


def ct_kpool_selectors(head_dim, hidden_dim, *, device, dtype):
    """Keep CANN's original H, selecting preprojected [K, G, zero padding]."""
    if hidden_dim < 2 * head_dim:
        raise ValueError("CT KPool requires hidden_dim >= 2 * head_dim")
    key = torch.zeros((head_dim, hidden_dim), device=device, dtype=dtype)
    gate = torch.zeros_like(key)
    eye = torch.eye(head_dim, device=device, dtype=dtype)
    key[:, :head_dim] = eye
    gate[:, head_dim : 2 * head_dim] = eye
    return key, gate


def ct_kpool_input(hidden, wk_weights_proj, compress_gate, head_dim):
    """Consume original W8/A8 methods before CANN pooling and cache updates."""
    key = wk_weights_proj(hidden)[0][..., :head_dim]
    gate = compress_gate(hidden)[0]
    if key.shape != gate.shape or hidden.shape[-1] < 2 * head_dim:
        raise ValueError("CT KPool projection shapes do not match")
    projected = hidden.new_zeros(hidden.shape)
    projected[..., :head_dim] = key
    projected[..., head_dim : 2 * head_dim] = gate
    return projected


def restore_ct_kv_b(weight_kn, scale_n, act_dtype, *, column_block_size=256):
    """Restore an explicitly opted-in absorbed KV projection, not W8 GEMM.

    Native CT post-loading has already transposed/sharded the integer matrix.
    Keep the original scales and bound FP32 temporary storage to a column block.
    """
    if weight_kn.dtype != torch.int8 or weight_kn.ndim != 2:
        raise ValueError("CT absorbed KV-b requires INT8 [K,N] after native post-loading")
    if scale_n.ndim != 1 or scale_n.numel() != weight_kn.shape[1] or not scale_n.is_floating_point():
        raise ValueError("CT absorbed KV-b requires one floating-point scale per output channel")
    if act_dtype not in (torch.bfloat16, torch.float16) or column_block_size <= 0:
        raise ValueError("CT absorbed KV-b requires BF16/FP16 output and a positive column block")
    output = torch.empty_like(weight_kn, dtype=act_dtype)
    for start in range(0, weight_kn.shape[1], column_block_size):
        end = min(start + column_block_size, weight_kn.shape[1])
        restored = weight_kn[:, start:end].float() * scale_n[start:end].float()
        output[:, start:end].copy_(restored.to(act_dtype))
    return output
