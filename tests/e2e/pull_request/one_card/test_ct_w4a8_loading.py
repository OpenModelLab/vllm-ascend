# SPDX-License-Identifier: Apache-2.0
"""Real NPU layout/bias/kernel regression; requires the serving CANN stack."""

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.quantization.compressed_tensors_w4a8 import pack_output_int4, process_expert_matrix
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


@pytest.mark.parametrize("packed_source", [False, True])
def test_ct_repack_matches_original_gmm(packed_source):
    torch.manual_seed(47)
    experts, n, k, tokens = 2, 256, 256, 8
    cpu_codes = torch.randint(-8, 8, (experts, n, k), dtype=torch.int8)
    scale = (torch.rand(experts, n, 1) / 16 + 0.001).to(torch.bfloat16).npu()
    logical = cpu_codes.npu()
    old_transposed = logical.transpose(1, 2).contiguous()
    old_bias = 8 * (old_transposed.float() * scale.transpose(1, 2)).sum(1)
    pairs = old_transposed.reshape(-1, 2)
    old_bytes = ((pairs[:, :1] & 15) | (pairs[:, 1:] << 4)).reshape(experts, k, n // 2).clone()
    source = torch.stack([pack_output_int4(c) for c in cpu_codes]).npu() if packed_source else logical
    new_bytes, new_bias = process_expert_matrix(source, scale, packed_output=packed_source)
    torch.testing.assert_close(new_bytes.cpu(), old_bytes.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(new_bias.cpu(), old_bias.cpu(), rtol=0, atol=0)
    # Mirror the production W4A8 scale bit contract and NZ/int32 conversion.
    encoded = scale.float().transpose(1, 2).contiguous().cpu().numpy().view(np.uint32).astype(np.int64)
    encoded = torch.from_numpy(encoded).npu()
    x = torch.randn(tokens * experts, k, dtype=torch.bfloat16).npu()
    qx, input_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    group_list = torch.tensor([tokens, tokens * experts], dtype=torch.int64).npu()
    results = []
    for weight, bias in ((old_bytes, old_bias), (new_bytes, new_bias)):
        kernel_weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ).view(torch.int32).contiguous()
        # The A8W4 kernel may modify its INT8 input; use independent clones.
        results.append(
            torch_npu.npu_grouped_matmul(
                x=[qx.clone()],
                weight=[kernel_weight],
                scale=[encoded],
                bias=[bias],
                per_token_scale=[input_scale],
                split_item=2,
                group_type=0,
                group_list=group_list,
                group_list_type=0,
                output_dtype=torch.bfloat16,
            )[0].cpu()
        )
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
