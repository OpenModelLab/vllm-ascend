# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare packed and raw codes through the upstream RoutedExperts loader."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from vllm_ascend.quantization.compressed_tensors_packed import install_packed_moe_loader


def _make_experts(tp_size, tp_rank, local_ids, padded):
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    hidden, intermediate = 19, 12
    allocated_hidden = 24 if padded else hidden
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.is_fused_checkpoint_transposed = False
    layer.lora_base_layer_prefix = ""
    layer.moe_config = SimpleNamespace(
        hidden_dim=allocated_hidden,
        hidden_dim_unpadded=hidden,
        intermediate_size=intermediate,
        num_experts=4,
        num_logical_experts=4,
        is_act_and_mul=True,
        tp_rank=tp_rank,
        tp_size=tp_size,
        moe_parallel_config=SimpleNamespace(tp_size=tp_size),
        tp_shard_with_padding=False,
    )
    slots = {expert_id: slot for slot, expert_id in enumerate(local_ids)}
    layer.expert_map_manager = SimpleNamespace(
        num_fused_shared_experts=0,
        map_global_to_local=lambda expert_id: slots.get(expert_id, -1),
    )
    layer.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    layer.quant_method = SimpleNamespace()
    layer.shared_expert_online_loader = None
    shapes = {
        "w13_weight": (len(local_ids), 2 * intermediate // tp_size, allocated_hidden),
        "w2_weight": (len(local_ids), allocated_hidden, intermediate // tp_size),
        "w13_weight_scale": (len(local_ids), 2 * intermediate // tp_size, 1),
        "w2_weight_scale": (len(local_ids), allocated_hidden, 1),
    }
    for name, shape in shapes.items():
        is_scale = name.endswith("scale")
        parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16 if is_scale else torch.int8), False)
        parameter.weight_loader = layer.weight_loader
        if is_scale:
            parameter.quant_method = "channel"
        layer.register_parameter(name, parameter)
    return layer


@pytest.mark.parametrize(
    "tp_size,tp_rank,local_ids",
    [(size, rank, (0, 1, 2, 3)) for size in (1, 2, 4) for rank in range(size)] + [(1, 0, (0, 2)), (1, 0, (1, 3))],
)
@pytest.mark.parametrize("padded", [False, True])
def test_native_tp_ep_loading_preserves_codes_and_scales(tp_size, tp_rank, local_ids, padded):
    raw = _make_experts(tp_size, tp_rank, local_ids, padded)
    packed = _make_experts(tp_size, tp_rank, local_ids, padded)
    install_packed_moe_loader(packed)
    raw_weights, packed_weights, metadata = [], [], []
    source = {}
    for expert in range(4):
        for index, (projection, shape) in enumerate(
            {"gate_proj": (12, 19), "up_proj": (12, 19), "down_proj": (19, 12)}.items()
        ):
            stem = f"{expert}.{projection}"
            codes = torch.arange(shape[0] * shape[1]).reshape(shape) * (2 * expert + 1) % 16 - 8 + index
            codes = ((codes + 8) % 16 - 8).to(torch.int8)
            scale = ((torch.arange(shape[0]) + 1 + expert + index) / 32).to(torch.bfloat16).unsqueeze(1)
            source[stem] = codes, scale
            raw_weights.extend([(f"{stem}.weight", codes), (f"{stem}.weight_scale", scale)])
            packed_weights.extend([(f"{stem}.weight_packed", pack_to_int32(codes, 4)), (f"{stem}.weight_scale", scale)])
            metadata.append((f"{stem}.weight_shape", torch.tensor(shape)))
    raw_names = set(raw.load_weights(raw_weights))
    packed_names = set(packed.load_weights(packed_weights))
    # Metadata is deliberately loaded later, in a separate iterator/shard.
    assert set(packed.load_weights(reversed(metadata))) == set()
    assert raw_names == packed_names == set(dict(raw.named_parameters()))
    for name, parameter in raw.named_parameters():
        assert torch.equal(parameter, getattr(packed, name)), name
    width = 12 // tp_size
    shard = slice(tp_rank * width, (tp_rank + 1) * width)
    for slot, expert in enumerate(local_ids):
        for projection, target in (("gate_proj", slice(0, width)), ("up_proj", slice(width, 2 * width))):
            codes, scale = source[f"{expert}.{projection}"]
            assert torch.equal(packed.w13_weight[slot, target, :19], codes[shard])
            assert torch.equal(packed.w13_weight_scale[slot, target], scale[shard])
        codes, scale = source[f"{expert}.down_proj"]
        assert torch.equal(packed.w2_weight[slot, :19], codes[:, shard])
        assert torch.equal(packed.w2_weight_scale[slot, :19], scale)


@pytest.mark.parametrize(
    "tp_size,tp_rank,local_ids",
    [(size, rank, (0, 1, 2, 3)) for size in (1, 2, 4) for rank in range(size)] + [(1, 0, (1, 3))],
)
def test_native_compact_resident_int4_loads_with_main_tp_ep(tp_size, tp_rank, local_ids):
    """The production path keeps two output-channel codes in each INT8 byte."""
    layer = _make_experts(tp_size, tp_rank, local_ids, False)
    hidden, intermediate = 32, 16
    layer.moe_config.hidden_dim = layer.moe_config.hidden_dim_unpadded = hidden
    layer.moe_config.intermediate_size = intermediate
    layer.scheme = SimpleNamespace(ct_packed_output=True)
    shapes = {
        "w13_weight": (len(local_ids), intermediate // tp_size, hidden),
        "w2_weight": (len(local_ids), hidden // 2, intermediate // tp_size),
        "w13_weight_scale": (len(local_ids), 2 * intermediate // tp_size, 1),
        "w2_weight_scale": (len(local_ids), hidden, 1),
    }
    for name, shape in shapes.items():
        parameter = torch.nn.Parameter(
            torch.zeros(shape, dtype=torch.bfloat16 if name.endswith("scale") else torch.int8), False
        )
        parameter.weight_loader = layer.weight_loader
        if name.endswith("scale"):
            parameter.quant_method = "channel"
        layer.register_parameter(name, parameter)
    install_packed_moe_loader(layer)
    weights, source = [], {}
    for expert in range(4):
        for index, (projection, shape) in enumerate(
            {
                "gate_proj": (intermediate, hidden),
                "up_proj": (intermediate, hidden),
                "down_proj": (hidden, intermediate),
            }.items()
        ):
            codes = ((torch.arange(shape[0] * shape[1]).reshape(shape) + expert * 3 + index) % 16 - 8).to(torch.int8)
            scale = ((torch.arange(shape[0]) + expert + 1) / 32).bfloat16().unsqueeze(1)
            stem = f"{expert}.{projection}"
            source[stem] = codes, scale
            weights.extend([(stem + ".weight_packed", pack_to_int32(codes, 4)), (stem + ".weight_scale", scale)])
    assert set(layer.load_weights(weights)) == set(dict(layer.named_parameters()))

    def compact(codes):
        return ((codes[::2].to(torch.int16) & 15) | ((codes[1::2].to(torch.int16) & 15) << 4)).to(torch.int8)

    width = intermediate // tp_size
    shard = slice(tp_rank * width, (tp_rank + 1) * width)
    for slot, expert in enumerate(local_ids):
        expected_w13, expected_scales = [], []
        for projection in ("gate_proj", "up_proj"):
            codes, scale = source[f"{expert}.{projection}"]
            expected_w13.append(compact(codes[shard]))
            expected_scales.append(scale[shard])
        assert torch.equal(layer.w13_weight[slot], torch.cat(expected_w13))
        assert torch.equal(layer.w13_weight_scale[slot], torch.cat(expected_scales))
        codes, scale = source[f"{expert}.down_proj"]
        assert torch.equal(layer.w2_weight[slot], compact(codes[:, shard]))
        assert torch.equal(layer.w2_weight_scale[slot], scale)
