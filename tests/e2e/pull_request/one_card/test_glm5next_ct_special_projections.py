# SPDX-License-Identifier: Apache-2.0
"""NPU acceptance for native floating KPool, mHC and KV-b in MLP-only CT."""

from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ, enable_custom_op


@torch.inference_mode()
def test_native_float_kpool_weight_preparation():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    head_dim, hidden = 128, 512
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.head_dim = head_dim
    backend.wk_weights_proj = torch.nn.Linear(hidden, head_dim + 4, bias=False, device="npu", dtype=torch.bfloat16)
    backend.index_kpool_compress_gate = torch.nn.Parameter(
        torch.randn(head_dim, hidden, device="npu", dtype=torch.bfloat16)
    )
    backend.k_norm = torch.nn.LayerNorm(head_dim, device="npu", dtype=torch.bfloat16)
    backend.process_weights_after_loading()
    assert isinstance(backend.index_kpool_compress_gate, torch.nn.Parameter)
    torch.testing.assert_close(backend._key_weight, backend.wk_weights_proj.weight[:head_dim], rtol=0, atol=0)
    torch.testing.assert_close(backend._gate_weight, backend.index_kpool_compress_gate, rtol=0, atol=0)
    torch.testing.assert_close(
        backend._head_weight_f32, backend.wk_weights_proj.weight[head_dim:].float(), rtol=0, atol=0
    )


@torch.inference_mode()
def test_native_float_kv_b_absorption_without_quantization_metadata():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    latent, heads, nope, value = 512, 2, 128, 128
    weight = torch.randn(heads * (nope + value), latent, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.npu(), requires_grad=False)
    impl = object.__new__(AscendSFAImpl)
    impl.kv_b_proj, impl.kv_lora_rank, impl.local_num_heads = layer, latent, heads
    impl.qk_nope_head_dim, impl.v_head_dim = nope, value
    impl.rl_weight_update_enabled, impl.has_indexer, impl.enable_sparse_sfa_c8 = False, False, False
    impl._parallel_o_proj = None
    impl._resolve_preprocess_type = lambda dtype: PreprocessType.NATIVE
    with patch(
        "vllm_ascend.attention.sfa_v1.maybe_trans_nz",
        side_effect=lambda w: torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ),
    ):
        impl.process_weights_after_loading(torch.bfloat16)
    expected = weight.reshape(heads, nope + value, latent)
    key = torch_npu.npu_format_cast(impl.W_UK_T, ACL_FORMAT_FRACTAL_ND).cpu()
    torch.testing.assert_close(key, expected[:, :nope], rtol=0, atol=0)
    torch.testing.assert_close(impl.W_UV.cpu(), expected[:, nope:].transpose(1, 2).contiguous(), rtol=0, atol=0)


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
