# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bounded-memory layout conversion for per-channel CT W4A8 experts.

CT stores offset-binary nibbles along K. Ascend consumes signed nibbles along
N. Repack on CPU before the native TP/EP loader; do not requantize the codes.
"""

import logging

import torch

INT4_VALUES_PER_BYTE = 2


def log_loading_memory(layer: torch.nn.Module, stage: str) -> None:
    """Record allocator bytes at startup; CPU tests do not need torch_npu."""
    weight = layer.w13_weight
    if weight.device.type != "npu":
        return
    device = weight.device
    logging.getLogger("vllm").info(
        "CT W4A8 %s: local_experts=%d w13=%s w2=%s allocated=%d reserved=%d peak=%d bytes",
        stage,
        weight.shape[0],
        tuple(weight.shape),
        tuple(layer.w2_weight.shape),
        torch.npu.memory_allocated(device),
        torch.npu.memory_reserved(device),
        torch.npu.max_memory_allocated(device),
    )


def pack_output_int4(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] signed INT4 codes -> [N/2, K] signed-nibble bytes."""
    if codes.dtype != torch.int8 or codes.ndim != 2 or codes.shape[0] % 2:
        raise ValueError("Output packing requires an INT8 matrix with even N")
    return ((codes[0::2] & 0xF) | (codes[1::2] << 4)).contiguous()


def unpack_output_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_output_int4; only materialize one expert at a time."""
    if packed.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("Expected a 2D signed-nibble INT8 tensor")
    low = ((packed & 0xF) ^ 8) - 8
    high = packed >> 4
    return torch.stack((low, high), dim=1).flatten(0, 1)


def process_expert_matrix(
    weight: torch.Tensor, scale: torch.Tensor, *, packed_output: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Ascend [E,K,N/2] bytes and FP32 [E,N] correction bias.

    Keep the original bias formula and operation order. FP32 temporaries are
    bounded to ONE expert, rather than all local experts in a layer. The final
    packed destination coexists with the source of this projection only.
    """
    if weight.dtype != torch.int8 or weight.ndim != 3:
        raise ValueError("Expected INT8 [E,N,K] or packed [E,N/2,K]")
    experts, stored_n, k = weight.shape
    n = stored_n * INT4_VALUES_PER_BYTE if packed_output else stored_n
    if n % INT4_VALUES_PER_BYTE or tuple(scale.shape) != (experts, n, 1):
        raise ValueError(f"Invalid per-channel scale/layout: {weight.shape}, {scale.shape}")
    output = torch.empty((experts, k, n // 2), dtype=torch.int8, device=weight.device)
    bias = torch.empty((experts, n), dtype=torch.float32, device=weight.device)
    for expert in range(experts):
        codes = unpack_output_int4(weight[expert]) if packed_output else weight[expert]
        transposed = codes.t().contiguous()
        # Same expression as the old implementation, with E bounded to one.
        bias[expert].copy_(8 * (transposed.to(torch.float32) * scale[expert].t()).sum(dim=0))
        packed = weight[expert] if packed_output else pack_output_int4(codes)
        output[expert].copy_(packed.t())
        del codes, transposed, packed
    return output, bias
