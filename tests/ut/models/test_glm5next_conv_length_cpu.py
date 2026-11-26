# SPDX-License-Identifier: Apache-2.0
"""Exercise the real GLM wrapper with both FLA signatures, without NPU startup."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/ops/causal_conv1d.py"


def load_wrapper(update):
    def isolated(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        return module

    def no_staging(*args, **kwargs):
        raise AssertionError("Contiguous state must not be staged")

    dependencies = {
        "fla_npu.ops.ascendc": isolated("fla_npu.ops.ascendc", causal_conv1d_fn=None, causal_conv1d_update=update),
        "vllm.v1.attention.backends.utils": isolated("vllm.v1.attention.backends.utils", PAD_SLOT_ID=-1),
        "vllm_ascend.ops.triton.kda.conv_state": isolated(
            "vllm_ascend.ops.triton.kda.conv_state", copy_conv_state=no_staging
        ),
    }
    with patch.dict(sys.modules, dependencies):
        spec = importlib.util.spec_from_file_location("glm_conv_length_fixture", SOURCE)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
    return wrapper


@pytest.mark.parametrize("bound,accepted,expected", [(1, False, 1), (3, True, 3), (None, True, 3), (None, False, 6)])
def test_update_bound_uses_config_or_shapes_without_host_tensor_reads(bound, accepted, expected):
    calls = []

    def update(x, conv_state, weight, *, max_query_len=-1, **kwargs):
        calls.append((max_query_len, conv_state, kwargs))
        kwargs["out"].copy_(x + 1)
        return kwargs["out"]

    wrapper = load_wrapper(update)
    x, weight, state = torch.ones(6, 8), torch.ones(4, 8), torch.zeros(4, 5, 8)
    starts = torch.tensor([0, 3, 6], dtype=torch.int32)
    indices = torch.tensor([1, 2], dtype=torch.int32)
    counts = torch.tensor([2, 1], dtype=torch.int32) if accepted else None
    output = torch.empty_like(x)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host item")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host cpu")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host tolist")),
    ):
        result = wrapper.causal_conv1d(
            x,
            weight,
            state,
            starts,
            indices,
            run_mode=1,
            num_accepted_tokens=counts,
            output_buffer=output,
            max_query_len=bound,
        )
    assert result.data_ptr() == output.data_ptr()
    assert calls[0][0] == expected and calls[0][1] is state
    assert calls[0][2]["query_start_loc"] is starts
    assert calls[0][2]["num_accepted_tokens"] is counts
    torch.testing.assert_close(result, x + 1)


def test_legacy_fla_update_does_not_receive_new_keyword():
    def legacy(
        x,
        conv_state,
        weight,
        *,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        null_block_id,
        out,
    ):
        assert activation == "silu" and null_block_id == 0
        out.copy_(x)
        return out

    wrapper = load_wrapper(legacy)
    result = wrapper.causal_conv1d(
        torch.ones(2, 8),
        torch.ones(4, 8),
        torch.zeros(3, 3, 8),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        run_mode=1,
        max_query_len=1,
    )
    torch.testing.assert_close(result, torch.ones(2, 8))
