# SPDX-License-Identifier: Apache-2.0
"""Small NPU kernel acceptance for existing GLM CT INT8 linear tensors.

This validates native parameter registration, checkpoint loaders and CANN GEMM.
Distributed/model initialization is intentionally outside this one-card test;
CPU regressions cover the real constructors and TP=1/2/4 loader indexing.
"""

from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_ascend.models.glm5next.kda import _Glm5NextMergedColumnParallelLinear
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.glm5next import glm5next_packed_mapping
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def make_config():
    result = AscendCompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "config_groups": {
                "linear": {
                    "targets": ["re:.*"],
                    "weights": {"num_bits": 8, "strategy": "channel", "symmetric": True, "type": "int"},
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": True,
                    },
                }
            },
        }
    )
    result.packed_modules_mapping = glm5next_packed_mapping()
    return result


@pytest.mark.parametrize("kind", ["fused", "lm_head", "router"])
@torch.inference_mode()
def test_native_ct_scale_and_cann_gemm(kind):
    """No model weights required; keep integer codes and per-row scales intact."""
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    hidden = 4096 if kind == "router" else 128
    sizes = [256, 256, 256, 128, 128, 128] if kind == "fused" else [288 if kind == "router" else 256]
    cls = {"fused": _Glm5NextMergedColumnParallelLinear, "lm_head": AscendParallelLMHead, "router": AscendGateLinear}[
        kind
    ]
    # Skip distributed construction here, retaining the actual type, parameter
    # creation, inherited checkpoint loaders and quantization implementation.
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.tp_size, layer.tp_rank = 1, 0
    layer.output_sizes = sizes
    layer.replicated_shard_ids = {4, 5}
    prefix = "layers.0.self_attn.in_proj_qkvbfg_a" if kind == "fused" else "lm_head"
    if kind == "router":
        prefix = "layers.10.mlp.gate"
        layer.is_unquantized = False
        layer.out_dtype = torch.float32
        layer.skip_bias_add, layer.return_bias, layer.bias = False, True, None
    if kind == "lm_head":
        layer.org_vocab_size = 225
        layer.num_embeddings = 225
        layer.num_embeddings_padded = 256
        layer.org_vocab_size_padded = 256
        layer.num_embeddings_per_partition = 256
        layer.shard_indices = layer._get_indices(256, 256, 225, 225, 0, 1)
    quant = make_config()
    layer.quant_method = quant.get_quant_method(layer, prefix)
    with torch.device("npu"):
        layer.quant_method.create_weights(
            layer, hidden, sizes, hidden, sum(sizes), torch.bfloat16, weight_loader=layer.weight_loader
        )
    assert layer.weight.dtype == torch.int8
    assert "weight_scale" in dict(layer.named_parameters())
    matrices, scales = [], []
    for sid, count in enumerate(sizes):
        rows = 225 if kind == "lm_head" else count
        codes = ((torch.arange(rows * hidden).reshape(rows, hidden) + sid) % 255 - 127).to(torch.int8)
        scale = ((torch.arange(rows) % 8 + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        args = (sid,) if kind == "fused" else ()
        layer.weight.weight_loader(layer.weight, codes, *args)
        layer.weight_scale.weight_loader(layer.weight_scale, scale, *args)
        if kind == "lm_head":
            codes = torch.nn.functional.pad(codes, (0, 0, 0, 256 - rows))
            scale = torch.nn.functional.pad(scale, (0, 0, 0, 256 - rows))
        matrices.append(codes)
        scales.append(scale)
    expected_weight = torch.cat(matrices)
    expected_scale = torch.cat(scales).flatten()
    torch.testing.assert_close(layer.weight.cpu(), expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale.cpu().flatten(), expected_scale, rtol=0, atol=0)
    # Exercise real NZ conversion without constructing a global serving config.
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.maybe_trans_nz",
        side_effect=lambda w: torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ),
    ):
        layer.quant_method.process_weights_after_loading(layer)
    x = (torch.arange(3 * hidden).reshape(3, hidden) % 255 - 127).to(device="npu", dtype=torch.bfloat16)
    qx, sx = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    accum = qx.cpu().to(torch.int32) @ expected_weight.T.to(torch.int32)
    expected = (accum.float() * sx.cpu().float().reshape(-1, 1) * expected_scale.float()).to(torch.bfloat16)
    if kind == "lm_head":
        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.head_dtype = None
        actual = processor._apply_head(layer, x, None)
    elif kind == "router":
        actual = layer(x)[0]
        expected = expected.float()
        assert actual.dtype == torch.float32
    else:
        actual = layer.quant_method.apply(layer, x)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.01)
    layer.weight_scale.data.mul_(2)
    doubled = layer(x)[0] if kind == "router" else layer.quant_method.apply(layer, x)
    torch.testing.assert_close(doubled.cpu(), expected * 2, rtol=0.01, atol=0.02)
