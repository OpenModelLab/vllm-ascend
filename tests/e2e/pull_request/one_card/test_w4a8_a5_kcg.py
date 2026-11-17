# SPDX-License-Identifier: Apache-2.0
"""NPU acceptance of native 950 packing, expert copies and both CT GMM hooks.

No model checkpoint or distributed initialization required. Unlike CPU tests,
this exercises the actual packed NZ allocation/copies and the CANN kernels.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization.compressed_tensors_w4a8 import pack_output_int4
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD, is_950


@pytest.mark.parametrize(
    "packed,hidden,intermediate,experts",
    [(True, 512, 256, 4), (False, 512, 256, 4), (True, 4096, 2048, 4), (True, 512, 256, 144)],
)
@torch.inference_mode()
def test_real_a5_ct_postload_gmm1_gmm2(packed, hidden, intermediate, experts):
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Ascend 950 required")
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.quant_method = COMPRESSED_TENSORS_METHOD
    method.use_a5_ct_kcg = True
    method.ct_packed_output = packed
    method.use_expert_weight_list = False
    layer = torch.nn.Module()
    original = {}
    generator = torch.Generator().manual_seed(20261116)
    for proj, n, k in (("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate)):
        codes = torch.randint(-8, 8, (experts, n, k), generator=generator, dtype=torch.int8)
        scale = ((torch.arange(experts * n).reshape(experts, n, 1) % 8 + 1) / 1024).bfloat16()
        original[proj] = codes, scale
        checkpoint_weight = torch.stack([pack_output_int4(c) for c in codes]) if packed else codes
        layer.register_parameter(f"{proj}_weight", torch.nn.Parameter(checkpoint_weight.npu(), requires_grad=False))
        layer.register_parameter(f"{proj}_weight_scale", torch.nn.Parameter(scale.npu(), requires_grad=False))
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.int32
    # INT8 activation K-CG reads logical C0=32, packed carrier C0=4.
    # C0=16/2 has identical logical shape/byte count but wrong kernel order.
    assert torch_npu.get_npu_format(layer.w13_weight) == 53
    assert torch_npu.get_npu_format(layer.w2_weight) == 53
    assert tuple(layer.w13_weight.shape) == (experts, hidden, 2 * intermediate // 8)
    assert layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.float32
    assert not hasattr(layer, "w13_scale_bias")
    torch.testing.assert_close(layer.w13_weight_scale.cpu(), original["w13"][1][..., 0].float(), rtol=0, atol=0)
    # Include boundary and interior slots of the 3D private-format destination.
    for first, second in ((0, experts - 1), (1, experts // 2)):
        x = torch.randint(-7, 8, (5, hidden), generator=generator, dtype=torch.int8)
        sx = (torch.arange(5).float() + 1) / 16
        groups = torch.zeros(experts, dtype=torch.int64, device="npu")
        groups[first], groups[second] = 2, 3
        inputs = type("Inputs", (), {})()
        inputs.layer, inputs.group_list, inputs.group_list_type = layer, groups, 1
        inputs.hidden_states, inputs.dynamic_scale = x.npu(), sx.npu().unsqueeze(1)
        selected = [first, first, second, second, second]
        expected1 = torch.cat(
            [
                (
                    (x[row : row + 1].int() @ original["w13"][0][expert].T.int()).float()
                    * original["w13"][1][expert, :, 0].float()
                    * sx[row]
                )
                for row, expert in enumerate(selected)
            ]
        ).bfloat16()
        actual1 = method.apply_gmm1(inputs)
        torch.testing.assert_close(actual1.cpu().float(), expected1.float(), rtol=0.01, atol=0.02)
        gate, up = actual1.chunk(2, -1)
        activated = torch.nn.functional.silu(gate) * up
        q2, sx2 = method.apply_act_quant(inputs, activated)
        q2_cpu, sx2_cpu = q2.cpu(), sx2.cpu().flatten()
        expected2 = torch.cat(
            [
                (
                    (q2_cpu[row : row + 1].int() @ original["w2"][0][expert].T.int()).float()
                    * original["w2"][1][expert, :, 0].float()
                    * sx2_cpu[row]
                )
                for row, expert in enumerate(selected)
            ]
        ).bfloat16()
        actual2 = method.apply_gmm2(inputs, q2, sx2)
        torch.testing.assert_close(actual2.cpu().float(), expected2.float(), rtol=0.01, atol=0.02)


@torch.inference_mode()
def test_a8_c0_32_matches_native_converter_bytes_and_basis_gmm():
    """Independent native A8 converter, exact physical readback, then real GMM."""
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Ascend 950 required")
    from vllm_ascend.quantization.compressed_tensors_w4a8_a5 import (
        a5_ct_probe_reference,
        pack_a5_nz_codes,
        prepare_a5_ct_projection,
        verify_a5_ct_projection,
    )

    generator = torch.Generator(device="cpu").manual_seed(20261117)
    codes = torch.randint(-8, 8, (2, 64, 256), dtype=torch.int8, device="cpu", generator=generator)
    source = codes.transpose(1, 2).int().contiguous().npu()
    nz = torch_npu.npu_format_cast(source, 29, customize_dtype=torch.int8)
    assert torch_npu.get_npu_format(nz) == 51
    native = torch_npu.npu_convert_weight_to_int4pack(nz)
    assert torch_npu.get_npu_format(native) == 53
    physical = torch_npu.empty_with_format(size=(2, 2, 16, 16, 4), dtype=torch.int32, device="npu", acl_format=2)
    assert torch_npu.get_npu_format(physical) == 2
    assert physical.untyped_storage().nbytes() == native.untyped_storage().nbytes()
    torch_npu.copy_memory_(physical, native, non_blocking=False)
    expected = torch.stack([pack_a5_nz_codes(code) for code in codes])
    torch.testing.assert_close(physical.cpu(), expected, rtol=0, atol=0)
    refs = {expert: a5_ct_probe_reference(code) for expert, code in enumerate(codes)}
    anti = torch.ones((2, 1, 64), dtype=torch.float16, device="npu")
    verify_a5_ct_projection(native, anti, refs, backend=torch_npu)
    # This calls the same real postload path and numerical guard as the model.
    prepare_a5_ct_projection(
        codes.npu(),
        torch.ones((2, 64, 1), dtype=torch.bfloat16, device="npu"),
        packed_output=False,
        backend=torch_npu,
    )


@torch.inference_mode()
def test_explicit_nd_storage_roundtrip_with_internal_format_disabled():
    """Pure storage test: no GMM, checkpoint, process group or HCCL."""
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Ascend 950 required")
    previous = torch.npu.config.allow_internal_format
    torch.npu.config.allow_internal_format = False
    try:
        shape = (2, 2, 16, 16, 4)
        values = torch.arange(2 * 2 * 16 * 16 * 4, dtype=torch.int32, device="cpu").reshape(shape)
        # Include negative words and both sign extremes, checking all 32 bits.
        values.flatten()[:4] = torch.tensor([0, -1, -(2**31), 2**31 - 1], dtype=torch.int32, device="cpu")
        stage = torch_npu.empty_with_format(size=shape, dtype=torch.int32, device="npu", acl_format=2)
        output = torch_npu.empty_with_format(size=(2, 256, 8), dtype=torch.int32, device="npu", acl_format=53)
        readback = torch_npu.empty_with_format(size=shape, dtype=torch.int32, device="npu", acl_format=2)
        assert torch_npu.get_npu_format(stage) == torch_npu.get_npu_format(readback) == 2
        assert torch_npu.get_npu_format(output) == 53
        for tensor in (stage, output, readback):
            assert tensor.storage_offset() == 0
            assert tensor.is_contiguous()
            assert tensor.untyped_storage().nbytes() == values.numel() * values.element_size()
        stage.copy_(values)
        torch_npu.copy_memory_(output, stage, non_blocking=False)
        torch_npu.copy_memory_(readback, output, non_blocking=False)
        torch.testing.assert_close(readback.cpu(), values, rtol=0, atol=0)
    finally:
        torch.npu.config.allow_internal_format = previous
