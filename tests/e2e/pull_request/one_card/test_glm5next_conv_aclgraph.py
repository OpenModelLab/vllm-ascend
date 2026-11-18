# SPDX-License-Identifier: Apache-2.0
"""Small native GLM AscendC convolution probe; no model weights or HCCL."""

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops import ascendc

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("dim", [384, 3072])
@pytest.mark.parametrize("dim_first", [False, True])
@torch.inference_mode()
def test_native_glm_conv_eager_and_graph_replay(capture, dim, dim_first):
    if not torch.npu.is_available():
        pytest.skip("Requires NPU and the serving fla_npu/CANN stack")
    generator = torch.Generator(device="cpu").manual_seed(20261118)
    initial = (torch.randn(4, 3, dim, generator=generator, device="cpu") / 8).bfloat16()
    weights_cpu = (torch.randn(4, dim, generator=generator, device="cpu") / 8).bfloat16()
    inputs_cpu = (torch.randn(2, dim, generator=generator, device="cpu") / 8).bfloat16()
    state = initial.transpose(1, 2).contiguous().npu().transpose(1, 2) if dim_first else initial.npu()
    weights, x = weights_cpu.npu(), inputs_cpu.npu()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    output = torch.zeros_like(x)

    def run():
        return causal_conv1d(x, weights, state, starts, slots, run_mode=1, output_buffer=output, max_query_len=1)

    # Warm up native kernels and any state-copy Triton specialization first.
    print(f"GLM native conv warmup begin capture={capture} dim={dim} dim_first={dim_first}", flush=True)
    run()
    run()
    torch.npu.synchronize()
    print(
        f"GLM native conv warmup end FLA backends={getattr(ascendc, 'BACKENDS', 'unavailable')} "
        f"fallbacks={getattr(ascendc, 'FALLBACKS', 'unavailable')}",
        flush=True,
    )
    graph = None
    if capture:
        graph = torch.npu.NPUGraph()
        print("GLM native conv capture begin", flush=True)
        with torch.npu.graph(graph):
            run()
        print("GLM native conv capture end", flush=True)
    for step in range(2):
        current = (inputs_cpu.float() + step / 4).bfloat16()
        x.copy_(current)
        state.copy_(initial)
        output.zero_()
        expected_state = initial.clone()
        expected = []
        for row, slot in enumerate((1, 2)):
            window = torch.cat((initial[slot], current[row : row + 1]))
            expected.append(torch.nn.functional.silu((window.float() * weights_cpu.float()).sum(0)).bfloat16())
            expected_state[slot].copy_(window[-3:])
        if graph is None:
            run()
        else:
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu().float(), torch.stack(expected).float(), rtol=0.02, atol=0.002)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
