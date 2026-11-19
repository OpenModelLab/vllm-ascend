# SPDX-License-Identifier: Apache-2.0
"""NPU numeric acceptance for dense/shared MLP W8A8, without model weights."""

from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear

from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.glm5next import glm5next_packed_mapping
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def make_config():
    result = AscendCompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "config_groups": {
                "mlp": {
                    "targets": [r"re:.*\.mlp\.(?:shared_experts\.)?(gate|up|down)_proj$"],
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


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
@torch.inference_mode()
def test_native_mlp_int8_scale_loading_and_gemm(shared, projection):
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    fused = projection == "gate_up_proj"
    hidden, sizes = (128, [256, 256]) if fused else (256, [128])
    cls = MergedColumnParallelLinear if fused else RowParallelLinear
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.tp_size, layer.tp_rank = 1, 0
    layer.input_size, layer.input_size_per_partition = hidden, hidden
    layer.output_size, layer.output_sizes = sum(sizes), sizes
    layer.skip_bias_add, layer.return_bias, layer.bias = False, True, None
    prefix = "layers.10.mlp.shared_experts" if shared else "layers.10.mlp"
    layer.quant_method = make_config().get_quant_method(layer, prefix + "." + projection)
    with torch.device("npu"):
        layer.quant_method.create_weights(
            layer, hidden, sizes, hidden, sum(sizes), torch.bfloat16, weight_loader=layer.weight_loader
        )
    assert layer.weight.dtype == torch.int8
    matrices, scales = [], []
    for shard, rows in enumerate(sizes):
        codes = ((torch.arange(rows * hidden).reshape(rows, hidden) + shard) % 255 - 127).to(torch.int8)
        scale = ((torch.arange(rows) % 8 + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        args = (shard,) if fused else ()
        layer.weight.weight_loader(layer.weight, codes, *args)
        layer.weight_scale.weight_loader(layer.weight_scale, scale, *args)
        matrices.append(codes)
        scales.append(scale)
    expected_weight, expected_scale = torch.cat(matrices), torch.cat(scales).flatten()
    torch.testing.assert_close(layer.weight.cpu(), expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale.cpu().flatten(), expected_scale, rtol=0, atol=0)
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.maybe_trans_nz",
        side_effect=lambda w: torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ),
    ):
        layer.quant_method.process_weights_after_loading(layer)
    x = (torch.arange(3 * hidden).reshape(3, hidden) % 255 - 127).to(device="npu", dtype=torch.bfloat16)
    qx, sx = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    accum = qx.cpu().int() @ expected_weight.T.int()
    expected = (accum.float() * sx.cpu().float().reshape(-1, 1) * expected_scale.float()).bfloat16()
    actual = layer.quant_method.apply(layer, x)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.01)
    layer.weight_scale.data.mul_(2)
    torch.testing.assert_close(layer.quant_method.apply(layer, x).cpu(), expected * 2, rtol=0.01, atol=0.02)
