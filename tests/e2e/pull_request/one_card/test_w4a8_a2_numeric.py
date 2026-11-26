# SPDX-License-Identifier: Apache-2.0
"""A2 native A8W4 acceptance against integer CPU dot products, not another kernel."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type
from vllm_ascend.ops.fused_moe.moe_mlp import _unified_apply_activation
from vllm_ascend.quantization.compressed_tensors_w4a8 import pack_output_int4
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


def _reference(x, input_scale, codes, weight_scale, expert_ids):
    # Use integer accumulation. No unpacker or NZ layout from production code
    # participates in this reference, including the -8 and +7 endpoint codes.
    return torch.cat(
        [
            (x[row : row + 1].int() @ codes[expert].T.int()).float()
            * weight_scale[expert, :, 0].float()
            * input_scale[row]
            for row, expert in enumerate(expert_ids)
        ]
    ).bfloat16()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("hidden,intermediate", [(512, 256), (4096, 2048)])
@torch.inference_mode()
def test_a2_ct_gmm1_activation_gmm2(packed, group_type, hidden, intermediate):
    if not torch.npu.is_available() or get_ascend_device_type() != AscendDeviceType.A2:
        pytest.skip("Ascend A2 required")
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.quant_method = COMPRESSED_TENSORS_METHOD
    method.ct_packed_output = packed
    method.use_expert_weight_list = False
    method.fused_activations = frozenset({"situ"})
    assert not method.supports_fused_activation(MoEActivation.SILU)
    layer, original = torch.nn.Module(), {}
    generator = torch.Generator(device="cpu").manual_seed(20261117)
    experts = 4
    for projection, n, k in (("w13", 2 * intermediate, hidden), ("w2", hidden, intermediate)):
        codes = torch.randint(-8, 8, (experts, n, k), generator=generator, dtype=torch.int8, device="cpu")
        scale = ((torch.arange(experts * n, device="cpu").reshape(experts, n, 1) % 8 + 1) / 1024).bfloat16()
        original[projection] = codes, scale
        source = torch.stack([pack_output_int4(expert) for expert in codes]) if packed else codes
        layer.register_parameter(f"{projection}_weight", torch.nn.Parameter(source.npu(), requires_grad=False))
        layer.register_parameter(f"{projection}_weight_scale", torch.nn.Parameter(scale.npu(), requires_grad=False))
    namespace = "vllm_ascend.quantization.methods.w4a8.w4a8"
    # No distributed/model config is necessary for the default (non-MegaMoE,
    # non-EPLB) kernel path. All format conversion and GMM calls stay real.
    with (
        patch(f"{namespace}.get_current_vllm_config", return_value=SimpleNamespace()),
        patch(f"{namespace}.use_cann_megamoe", return_value=False),
    ):
        method.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.int32
    assert torch_npu.get_npu_format(layer.w13_weight) == torch_npu.get_npu_format(layer.w2_weight) == 29
    assert layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.int64
    # Empty experts in the middle exercise both cumulative offsets and counts.
    counts = torch.tensor([2, 0, 0, 3], dtype=torch.int64, device="cpu")
    groups = (counts.cumsum(0) if group_type == 0 else counts).npu()
    expert_ids = [0, 0, 3, 3, 3]
    x = torch.randint(-127, 128, (5, hidden), generator=generator, dtype=torch.int8, device="cpu")
    sx = torch.arange(1, 6, dtype=torch.float32, device="cpu") / 128
    inputs = SimpleNamespace(
        layer=layer,
        group_list=groups,
        group_list_type=group_type,
        hidden_states=x.npu(),
        dynamic_scale=sx.npu(),
        activation=MoEActivation.SILU,
        swiglu_limit=1.0,
    )
    expected1 = _reference(x, sx, *original["w13"], expert_ids)
    actual1 = method.apply_gmm1(inputs)
    torch.testing.assert_close(actual1.cpu().float(), expected1.float(), rtol=0.01, atol=0.02)
    gate, up = actual1.cpu().float().chunk(2, -1)
    expected_activation = (torch.nn.functional.silu(gate.clamp(max=1.0)) * up.clamp(-1.0, 1.0)).bfloat16()
    activated = _unified_apply_activation(inputs, actual1, method)
    torch.testing.assert_close(activated.cpu().float(), expected_activation.float(), rtol=0.01, atol=0.02)
    q2, sx2 = method.apply_act_quant(inputs, activated)
    # CANN's A8W4 kernel may modify its INT8 input; capture the reference first.
    expected2 = _reference(q2.cpu(), sx2.cpu().flatten(), *original["w2"], expert_ids)
    actual2 = method.apply_gmm2(inputs, q2, sx2)
    torch.testing.assert_close(actual2.cpu().float(), expected2.float(), rtol=0.01, atol=0.02)
