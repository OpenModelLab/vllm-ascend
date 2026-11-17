# SPDX-License-Identifier: Apache-2.0
"""NPU acceptance for CT KPool, bare floating-point mHC and KV-b absorption.

Run in the user's CANN/cann_ops_transformer container. No model weights are
needed. These tests do not validate complete model generation or task scores.
"""

from importlib import import_module
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import PreprocessType
from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.glm5next import ct_kpool_input, ct_kpool_selectors
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ, enable_custom_op


def projection(hidden, outputs, prefix, params_dtype=torch.bfloat16):
    layer = object.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(layer)
    layer.skip_bias_add, layer.return_bias, layer.bias = False, True, None
    layer.prefix, layer.input_size = prefix, hidden
    quant = AscendCompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "config_groups": {
                "linear": {
                    "targets": ["re:.*"],
                    "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True},
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )
    layer.quant_method = quant.get_quant_method(layer, prefix)
    with torch.device("npu"):
        layer.quant_method.create_weights(
            layer, hidden, [outputs], hidden, outputs, params_dtype, weight_loader=layer.weight_loader
        )
    codes = (torch.arange(outputs * hidden).reshape(outputs, hidden) % 15 - 7).to(torch.int8)
    scales = ((torch.arange(outputs) % 8 + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
    layer.weight.weight_loader(layer.weight, codes)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.maybe_trans_nz",
        side_effect=lambda weight: torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ),
    ):
        layer.quant_method.process_weights_after_loading(layer)
    return layer, codes, scales.flatten()


def integer_reference(x, codes, scales):
    q, token_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    accum = q.cpu().to(torch.int32) @ codes.T.to(torch.int32)
    return (accum.float() * token_scale.cpu().float().reshape(-1, 1) * scales.float()).to(torch.bfloat16)


@torch.inference_mode()
def test_kpool_ct_projection_and_real_cann_pool():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    import_module("cann_ops_transformer")
    hidden, dim, ratio = 4096, 128, 4
    wk, wk_codes, wk_scale = projection(hidden, dim + 64, "indexer.wk_weights_proj")
    gate, gate_codes, gate_scale = projection(hidden, dim, "indexer.index_kpool_compress_gate")
    x = (torch.arange(ratio * hidden).reshape(ratio, hidden) % 255 - 127).to(device="npu", dtype=torch.bfloat16)
    packed = ct_kpool_input(x, wk, gate, dim)
    key_ref = integer_reference(x, wk_codes, wk_scale)[:, :dim]
    gate_ref = integer_reference(x, gate_codes, gate_scale)
    torch.testing.assert_close(packed[:, :dim].cpu(), key_ref, rtol=0.01, atol=0.01)
    torch.testing.assert_close(packed[:, dim : 2 * dim].cpu(), gate_ref, rtol=0.01, atol=0.01)
    assert packed.shape == (ratio, hidden)
    assert packed[:, 2 * dim :].count_nonzero().item() == 0
    key_selector, gate_selector = ct_kpool_selectors(dim, hidden, device=x.device, dtype=x.dtype)
    ape = torch.zeros(ratio, dim, device=x.device, dtype=torch.float32)
    state = torch.zeros(3, ratio, 2 * dim, device=x.device, dtype=torch.float32)
    table = torch.tensor([[1, 2]], device=x.device, dtype=torch.int32)
    offsets = torch.zeros(1, device=x.device, dtype=torch.int32)
    lengths = torch.tensor([0, ratio], device=x.device, dtype=torch.int32)
    norm_weight = torch.ones(dim, device=x.device, dtype=torch.float32)
    norm_bias = torch.zeros_like(norm_weight)
    actual = torch.ops.cann_ops_transformer.key_pool(
        packed,
        key_selector,
        gate_selector,
        ape,
        state,
        table,
        offsets,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cu_seqlens=lengths,
        cmp_ratio=ratio,
        norm_eps=1e-6,
    )
    # Only the completed pool is valid; capacity tail rows are unspecified.
    normalized = torch.nn.functional.layer_norm(key_ref.float(), (dim,), eps=1e-6)
    expected = (torch.softmax(gate_ref.float(), dim=0) * normalized).sum(dim=0).to(torch.bfloat16)
    torch.testing.assert_close(actual[0].cpu(), expected, rtol=0.02, atol=0.02)


@torch.inference_mode()
def test_mhc_bare_bf16_load_and_original_fused_fp32_path():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    enable_custom_op()
    torch.manual_seed(7)
    streams, hidden = 4, 4096
    layer = object.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.n, layer.mhc_sinkhorn_iterations = streams, 20
    layer.rms_norm_eps, layer.hc_eps, layer.mhc_post_mult_value = 1e-5, 1e-6, 2.0
    # The real checkpoint has bare BF16 fn/base/scale and no CT qparams.
    fn = (torch.randn(24, streams * hidden) * 0.001).to(torch.bfloat16)
    base = (torch.randn(24) * 0.1).to(torch.bfloat16)
    scale = torch.tensor([0.2, 0.7, 0.4], dtype=torch.bfloat16)
    for name, value in (("hc_attn_fn", fn), ("hc_attn_base", base), ("hc_attn_scale", scale)):
        param = torch.nn.Parameter(torch.empty(value.shape, dtype=torch.float32, device="npu"), requires_grad=False)
        layer.register_parameter(name, param)
        default_weight_loader(param, value)
        assert param.dtype == torch.float32
        torch.testing.assert_close(param.cpu(), value.float(), rtol=0, atol=0)
    x = torch.randn(2, streams, hidden).to(torch.bfloat16)
    expected = mhc_pre_torch(x, fn.float(), scale.float(), base.float(), 1e-5, 1e-6, 1e-6, 2.0, 20)
    actual = layer.hc_pre(x.npu(), layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    assert actual[0].dtype == actual[1].dtype == torch.float32
    assert actual[2].dtype == torch.bfloat16
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, rtol=0.01, atol=0.01)


@torch.inference_mode()
def test_kv_b_original_scales_and_real_nz_absorption():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    latent, heads, nope, value = 512, 2, 128, 128
    layer, codes, scales = projection(latent, heads * (nope + value), "layers.11.self_attn.kv_b_proj")
    layer.ct_restore_bf16_for_absorption = True
    impl = object.__new__(AscendSFAImpl)
    impl.kv_b_proj, impl.kv_lora_rank, impl.local_num_heads = layer, latent, heads
    impl.qk_nope_head_dim, impl.v_head_dim = nope, value
    impl.rl_weight_update_enabled, impl.has_indexer, impl.enable_sparse_sfa_c8 = False, False, False
    impl._parallel_o_proj = None
    impl._resolve_preprocess_type = lambda dtype: PreprocessType.NATIVE
    with patch(
        "vllm_ascend.attention.sfa_v1.maybe_trans_nz",
        side_effect=lambda weight: torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ),
    ):
        impl.process_weights_after_loading(torch.bfloat16)
    expected = (codes.float() * scales[:, None].float()).to(torch.bfloat16).reshape(heads, nope + value, latent)
    key = torch_npu.npu_format_cast(impl.W_UK_T, ACL_FORMAT_FRACTAL_ND).cpu()
    torch.testing.assert_close(key, expected[:, :nope, :], rtol=0, atol=0)
    torch.testing.assert_close(impl.W_UV.cpu(), expected[:, nope:, :].transpose(1, 2).contiguous(), rtol=0, atol=0)
