# SPDX-License-Identifier: Apache-2.0
"""Ascend 950 K-CG execution for per-channel CT INT4 expert checkpoints.

Checkpoint codes and scales are unchanged. Construct C0=4 physical INT4 bytes
on CPU, stage them in ND memory, then copy the complete storage to NZ. Never
write a slice of private NZ with Tensor.copy_: it can insert TransData.
"""

import logging

import torch

from .compressed_tensors_w4a8 import unpack_output_int4

A5_CT_GROUP_SIZE = 256
INT4_CODES_PER_INT32 = 8
A5_INT4_N_ALIGNMENT = 64
FRACTAL_NZ = 29
FRACTAL_ND = 2
A5_NZ_K_BLOCK = 16
A5_NZ_LOGICAL_N_BLOCK = 32  # K-CG consumes INT8 activations, so logical C0=32.
A5_NZ_PACKED_N_BLOCK = A5_NZ_LOGICAL_N_BLOCK // INT4_CODES_PER_INT32
FRACTAL_NZ_C0_32 = 51
FRACTAL_NZ_C0_4 = 53


def pack_a5_nz_codes(codes):
    """CPU [N,K] signed codes -> physical [N/32,K/16,16,4] INT32.

    The 950 K-CG vector kernel selects C0=32 from activation dtype INT8,
    independently of output dtype BF16. Pack each eight consecutive n0 codes.
    C0=16 has the same total bytes and logical shape but a different ordering.
    """
    if codes.device.type != "cpu" or codes.dtype != torch.int8 or codes.ndim != 2:
        raise ValueError("A5 NZ packing requires CPU INT8 [N,K] codes")
    n, k = codes.shape
    if n % A5_NZ_LOGICAL_N_BLOCK or k % A5_NZ_K_BLOCK:
        raise ValueError("A5 NZ packing requires N divisible by 32 and K divisible by 16")
    lo, hi = torch.aminmax(codes)
    if lo < -8 or hi > 7:
        raise ValueError("A5 NZ packing requires signed INT4 codes in [-8,7]")
    lanes = (
        codes.reshape(n // A5_NZ_LOGICAL_N_BLOCK, A5_NZ_LOGICAL_N_BLOCK, k // A5_NZ_K_BLOCK, A5_NZ_K_BLOCK)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    lanes = lanes.reshape(
        n // A5_NZ_LOGICAL_N_BLOCK, k // A5_NZ_K_BLOCK, A5_NZ_K_BLOCK, A5_NZ_PACKED_N_BLOCK, INT4_CODES_PER_INT32
    )
    packed = torch.zeros(lanes.shape[:-1], dtype=torch.int32, device="cpu")
    for lane in range(INT4_CODES_PER_INT32):
        packed.bitwise_or_((lanes[..., lane].to(torch.int32) & 15) << (4 * lane))
    return packed


def a5_ct_probe_reference(codes):
    """Three exact basis rows and a dense integer row, computed before packing."""
    n, k = codes.shape
    probes = torch.zeros((4, k), dtype=torch.int8, device="cpu")
    for row, column in enumerate((0, 15, k // 2)):
        probes[row, column] = 1
    probes[3] = ((torch.arange(k, device="cpu") * 7 + 3) % 15 - 7).to(torch.int8)
    expected = torch.empty((4, n), dtype=torch.bfloat16, device="cpu")
    expected[:3] = codes[:, (0, 15, k // 2)].T.to(torch.bfloat16)
    expected[3] = (codes.to(torch.int32) @ probes[3].to(torch.int32)).to(torch.bfloat16)
    return probes, expected


def verify_a5_ct_projection(weight, antiquant_scale, references, *, backend):
    """Check the assembled NZ tensor through real K-CG, not a shape test.

    All channels of first/middle/last expert slots are probed. Unit scales
    isolate code/layout/kernel errors from checkpoint quantization error.
    """
    experts, k, stored_n = weight.shape
    n = stored_n * INT4_CODES_PER_INT32
    selected = sorted(references)
    x = torch.cat([references[expert][0] for expert in selected]).to(weight.device)
    expected = torch.cat([references[expert][1] for expert in selected])
    counts = torch.zeros(experts, dtype=torch.int64, device="cpu")
    counts[selected] = 4
    result = apply_a5_ct_gmm(
        x,
        torch.ones(x.shape[0], dtype=torch.float32, device=weight.device),
        weight,
        torch.ones((experts, n), dtype=torch.float32, device=weight.device),
        antiquant_scale,
        counts.to(weight.device),
        1,
        backend=backend,
    ).cpu()
    actual_rows, expected_rows = result.reshape(-1, 4, n), expected.reshape(-1, 4, n)
    try:
        torch.testing.assert_close(actual_rows[:, :3], expected_rows[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(actual_rows[:, 3], expected_rows[:, 3], rtol=0.01, atol=1)
    except AssertionError as error:
        raise RuntimeError(
            f"A5 CT INT4 numerical self-check failed for expert slots {selected}, K={k}, N={n}, "
            f"format={int(backend.get_npu_format(weight))}, logical_C0=32/carrier_C0=4; "
            "refusing to serve corrupted projections. " + str(error)
        ) from error
    logging.getLogger("vllm").info(
        "A5 CT INT4 numerical self-check passed: expert_slots=%s K=%d N=%d format=%d",
        selected,
        k,
        n,
        int(backend.get_npu_format(weight)),
    )


def validate_a5_ct_projection(weight, scale, *, packed_output):
    """Validate local TP/EP shapes before replacing any model parameter."""
    if weight.dtype != torch.int8 or weight.ndim != 3:
        raise ValueError("A5 CT W4A8 expects pre-processing INT8 [E,N,K] codes or [E,N/2,K] bytes")
    experts, stored_n, k = weight.shape
    n = stored_n * 2 if packed_output else stored_n
    if experts < 1 or k <= 0 or n <= 0 or k % A5_CT_GROUP_SIZE or n % A5_INT4_N_ALIGNMENT:
        raise ValueError(f"A5 CT W4A8 needs E>0, local K%256=0 and logical N%64=0, got E={experts}, K={k}, N={n}")
    if tuple(scale.shape) != (experts, n, 1) or not scale.is_floating_point():
        raise ValueError(f"Expected raw floating CT scales [E,N,1]={experts, n, 1}, got {scale.dtype} {scale.shape}")
    if scale.device != weight.device:
        raise ValueError("A5 CT weights and scales must be on the same device")
    return experts, n, k


def prepare_a5_ct_projection(weight, scale, *, packed_output, backend):
    """Return INT32 NZ [E,K,N/8], FP32 [E,N], and neutral FP16 [E,K/256,N].

    The native converter is used only for a zero-valued format template. Model
    codes are packed explicitly on CPU; ordinary copy_ only targets ND storage.
    copy_memory_ writes the whole zero-offset NZ allocation once, avoiding both
    TransData and copy_memory_'s padded-slice/storage-offset restrictions.
    """
    experts, n, k = validate_a5_ct_projection(weight, scale, packed_output=packed_output)
    template = backend.npu_format_cast(
        torch.zeros((1, 16, 64), dtype=torch.int32, device=weight.device),
        FRACTAL_NZ,
        # additionalDtype is the ACTIVATION dtype, not the BF16 output dtype.
        customize_dtype=torch.int8,
    )
    if int(backend.get_npu_format(template)) != FRACTAL_NZ_C0_32:
        raise RuntimeError("A5 INT8 K-CG needs NZ_C0_32 before INT4 packing; refusing a C0=16/base-format template")
    template = backend.npu_convert_weight_to_int4pack(template)
    packed_format = int(backend.get_npu_format(template))
    if template.dtype != torch.int32 or tuple(template.shape) != (1, 16, 8) or packed_format != FRACTAL_NZ_C0_4:
        raise RuntimeError("A5 INT8 K-CG needs an INT32 NZ_C0_4 format template; NZ_C0_2 is for A16")
    del template
    output = backend.empty_with_format(
        size=(experts, k, n // INT4_CODES_PER_INT32),
        dtype=torch.int32,
        layout=weight.layout,
        device=weight.device,
        acl_format=packed_format,
    )
    if int(backend.get_npu_format(output)) != packed_format:
        raise RuntimeError("950 packed NZ destination format differs from its template")
    # Generic 5-D empty() infers NCDHW (30), even when internal formats are
    # disabled. Request ND explicitly instead of assuming its format ID is 2.
    stage = backend.empty_with_format(
        size=(experts, n // A5_NZ_LOGICAL_N_BLOCK, k // A5_NZ_K_BLOCK, A5_NZ_K_BLOCK, A5_NZ_PACKED_N_BLOCK),
        dtype=torch.int32,
        layout=weight.layout,
        device=weight.device,
        acl_format=FRACTAL_ND,
    )
    stage_format = int(backend.get_npu_format(stage))
    if stage_format != FRACTAL_ND or not stage.is_contiguous() or stage.storage_offset() != 0:
        raise RuntimeError(
            "A5 INT4 physical-byte staging must use contiguous zero-offset ND storage: "
            f"format={stage_format}, shape={tuple(stage.shape)}, "
            f"stride={stage.stride()}, offset={stage.storage_offset()}"
        )
    if output.storage_offset() != 0 or stage.untyped_storage().nbytes() != stage.numel() * stage.element_size():
        raise RuntimeError("A5 INT4 raw copy requires a zero-offset destination and an unpadded ND staging allocation")
    if output.untyped_storage().nbytes() != stage.untyped_storage().nbytes():
        raise RuntimeError("A5 INT4 NZ and ND physical storage sizes differ; raw copy is unsafe")
    references = {}
    selected = {0, experts // 2, experts - 1}
    for expert in range(experts):
        # Unpack on CPU: no bitwise operation on a private NPU view.
        stored = weight[expert].cpu()
        codes = unpack_output_int4(stored) if packed_output else stored
        if expert in selected:
            references[expert] = a5_ct_probe_reference(codes)
        stage[expert].copy_(pack_a5_nz_codes(codes))
        del stored, codes
    backend.copy_memory_(output, stage, non_blocking=False)
    del stage
    raw_scale = scale.squeeze(-1).to(torch.float32).contiguous()
    antiquant_scale = torch.ones((experts, k // A5_CT_GROUP_SIZE, n), dtype=torch.float16, device=weight.device)
    verify_a5_ct_projection(output, antiquant_scale, references, backend=backend)
    return output, raw_scale, antiquant_scale


def apply_a5_ct_gmm(x, pertoken_scale, weight, scale, antiquant_scale, group_list, group_list_type, *, backend):
    """Run full-width K-CG without legacy encoded scales or assist biases."""
    if x.dtype != torch.int8 or x.ndim != 2:
        raise ValueError(f"A5 CT K-CG needs INT8 [M,K] input, got {x.dtype} {x.shape}")
    if weight.dtype != torch.int32 or weight.ndim != 3 or weight.shape[-2] != x.shape[-1]:
        raise ValueError("A5 CT K-CG needs an INT32 [E,K,N/8] weight matching input K")
    experts, k, stored_n = weight.shape
    n = stored_n * INT4_CODES_PER_INT32
    if scale.dtype != torch.float32 or tuple(scale.shape) != (experts, n):
        raise ValueError("A5 CT K-CG needs FP32 [E,full_N] scales; packed-byte N/2 scales are invalid")
    if antiquant_scale.dtype != torch.float16 or tuple(antiquant_scale.shape) != (experts, k // A5_CT_GROUP_SIZE, n):
        raise ValueError("A5 CT K-CG needs FP16 [E,K/256,full_N] antiquant_scale")
    if group_list.ndim != 1 or group_list.numel() != experts or group_list_type not in (0, 1):
        raise ValueError("A5 CT K-CG expects one group-list entry per local expert, using cumsum or count")
    if pertoken_scale is None or pertoken_scale.numel() != x.shape[0]:
        raise ValueError("A5 CT K-CG expects exactly one activation scale per input token")
    if x.shape[0] == 0:
        return torch.empty((0, n), dtype=torch.bfloat16, device=x.device)
    result = backend.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=[scale],
        antiquant_scale=[antiquant_scale],
        per_token_scale=[pertoken_scale.reshape(-1).to(torch.float32)],
        group_list=group_list.to(torch.int64),
        group_list_type=group_list_type,
        split_item=2,
        group_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    if tuple(result.shape) != (x.shape[0], n):
        raise RuntimeError(f"A5 CT K-CG returned {result.shape}; expected full-width {(x.shape[0], n)}")
    return result
