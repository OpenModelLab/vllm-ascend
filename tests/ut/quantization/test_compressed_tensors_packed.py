# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for the checkpoint conversion boundary, without plugin startup."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

# This module intentionally has no vLLM/NPU dependency. It can also run with
# --confcutdir=tests/ut/quantization in a standalone torch CPU environment.
_path = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/compressed_tensors_packed.py"
_spec = spec_from_file_location("compressed_tensors_packed", _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
unpack_packed_moe_weights = _module.unpack_packed_moe_weights
install_packed_moe_loader = _module.install_packed_moe_loader


@pytest.mark.parametrize("shape", [(1, 8), (7, 19), (19, 10)])
@pytest.mark.parametrize("metadata_first", [False, True])
def test_ct_codes_and_scales_are_preserved(shape, metadata_first):
    codes = (torch.arange(shape[0] * shape[1]).reshape(shape) % 16 - 8).to(torch.int8)
    scale = torch.full((shape[0], 1), 0.003, dtype=torch.bfloat16)
    weights = [
        ("2.gate_proj.weight_packed", pack_to_int32(codes, 4)),
        ("2.gate_proj.weight_shape", torch.tensor(shape)),
    ]
    if metadata_first:
        weights.reverse()
    weights.append(("2.gate_proj.weight_scale", scale))
    result = dict(unpack_packed_moe_weights(weights, {"gate_proj": shape}))
    assert set(result) == {"2.gate_proj.weight", "2.gate_proj.weight_scale"}
    assert torch.equal(result["2.gate_proj.weight"], codes)
    assert result["2.gate_proj.weight"].is_contiguous()
    assert result["2.gate_proj.weight"].device.type == "cpu"
    assert result["2.gate_proj.weight_scale"] is scale


def test_ct_signed_offset_encoding():
    # CT stores unsigned nibbles of (code + 8), rather than two's complement.
    packed = torch.tensor([[0, -1]], dtype=torch.int32)
    result = dict(unpack_packed_moe_weights([("0.up_proj.weight_packed", packed)], {"up_proj": (1, 16)}))
    assert result["0.up_proj.weight"].tolist() == [[-8] * 8 + [7] * 8]


@pytest.mark.parametrize(
    "name,tensor",
    [
        ("0.gate_proj.weight_packed", torch.zeros(7, 3, dtype=torch.int8)),
        ("0.gate_proj.weight_packed", torch.zeros(7, 2, dtype=torch.int32)),
        ("0.gate_proj.weight_packed", torch.zeros(8, 3, dtype=torch.int32)),
        ("0.gate_proj.weight_shape", torch.tensor([7, 18])),
        ("0.gate_proj.weight_shape", torch.tensor([[7, 19]])),
        ("0.gate_proj.weight_shape", torch.tensor([7.0, 19.0])),
        ("gate_proj.weight_packed", torch.zeros(7, 3, dtype=torch.int32)),
        ("0.other_proj.weight_packed", torch.zeros(7, 3, dtype=torch.int32)),
        ("gate_up_proj.weight_packed", torch.zeros(2, 7, 3, dtype=torch.int32)),
    ],
)
def test_invalid_layout_is_rejected(name, tensor):
    with pytest.raises(ValueError):
        list(unpack_packed_moe_weights([(name, tensor)], {"gate_proj": (7, 19)}))


def test_conversion_is_streaming():
    def weights():
        yield "0.gate_proj.weight_packed", torch.zeros(7, 3, dtype=torch.int32)
        raise RuntimeError("Next shard has not been read")

    converted = unpack_packed_moe_weights(weights(), {"gate_proj": (7, 19)})
    name, codes = next(converted)
    assert name == "0.gate_proj.weight" and codes.shape == (7, 19)
    with pytest.raises(RuntimeError, match="Next shard"):
        next(converted)


class _ExpertLoader(torch.nn.Module):
    """Record the native loader's input, then exercise a TP boundary inside a pack."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.moe_config = SimpleNamespace(hidden_dim=19, hidden_dim_unpadded=19, intermediate_size=10)
        self.ckpt_gate_proj_name = "gate_proj"
        self.ckpt_down_proj_name = "down_proj"
        self.ckpt_up_proj_name = "up_proj"
        self.received = {}

    def load_weights(self, weights):
        names = []
        for name, tensor in weights:
            # The checkpoint adapter must leave TP/fusion to the existing loader.
            assert tensor.dtype == torch.int8
            axis = 1 if ".down_proj." in name else 0
            self.received[name] = tensor.chunk(2, dim=axis)[self.rank].clone()
            names.append(name)
        return names


@pytest.mark.parametrize("rank", [0, 1])
def test_layer_boundary_and_cross_shard_metadata(rank):
    layer = _ExpertLoader(rank)
    original_method = layer.load_weights
    install_packed_moe_loader(layer)
    for projection, shape in {"gate_proj": (10, 19), "up_proj": (10, 19), "down_proj": (19, 10)}.items():
        codes = (torch.arange(shape[0] * shape[1]).reshape(shape) % 16 - 8).to(torch.int8)
        packed_name = f"0.{projection}.weight_packed"
        raw_name = f"0.{projection}.weight"
        names = layer.load_weights([(packed_name, pack_to_int32(codes, 4))])
        assert names == [raw_name]
        assert layer.load_weights([(f"0.{projection}.weight_shape", torch.tensor(shape))]) == []
        expected = codes.chunk(2, dim=1 if projection == "down_proj" else 0)[rank]
        assert torch.equal(layer.received[raw_name], expected)
    # Installation affects this instance only, not the layer class or other models.
    assert original_method.__func__ is _ExpertLoader.load_weights
    assert "load_weights" not in _ExpertLoader(0).__dict__
