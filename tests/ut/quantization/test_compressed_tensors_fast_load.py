# SPDX-License-Identifier: Apache-2.0
"""CPU bit-exact startup checks; do not stand in for NPU acceptance."""

import os
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from tests.ut.quantization.test_compressed_tensors_packed import _module


@pytest.mark.parametrize("n", [2, 4, 10])
@pytest.mark.parametrize("k", [1, 7, 8, 9, 19, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_direct_repack_is_bit_exact_with_padding_and_strides(n, k, strided):
    codes = (torch.arange(n * k).reshape(n, k) % 16 - 8).to(torch.int8)
    packed = pack_to_int32(codes, 4)
    if strided:
        storage = torch.empty(n, 2 * packed.shape[1], dtype=torch.int32)
        storage[:, ::2].copy_(packed)
        packed = storage[:, ::2]
    actual = _module.repack_ct_int4(packed, (n, k))
    assert actual.dtype == torch.int8 and actual.is_contiguous()
    assert torch.equal(actual, _module.pack_output_int4(codes))


def test_fast_switch_preserves_streaming_metadata_scales_and_kwargs():
    codes = (torch.arange(190).reshape(10, 19) % 16 - 8).to(torch.int8)
    scale = torch.arange(10, dtype=torch.bfloat16).reshape(10, 1)
    kwargs = {"sentinel": object()}
    stem = "layers.11.mlp.experts.287.gate_proj"
    source = [
        (stem + ".weight_shape", torch.tensor([10, 19])),
        (stem + ".weight_packed", pack_to_int32(codes, 4), kwargs),
        (stem + ".weight_scale", scale),
    ]
    results = []
    for mode in ("0", "1"):
        with patch.dict(os.environ, {"VLLM_ASCEND_CT_FAST_LOAD": mode}):
            results.append(list(_module.unpack_packed_moe_weights(source, {"gate_proj": (10, 19)}, packed_output=True)))
    assert results[0][0][0] == results[1][0][0] == stem + ".weight"
    assert torch.equal(results[0][0][1], results[1][0][1])
    assert results[1][0][2] is kwargs and results[1][1][1] is scale


@pytest.mark.parametrize(
    "packed,shape",
    [
        (torch.zeros(3, 1, dtype=torch.int32), (3, 8)),
        (torch.zeros(2, 1, dtype=torch.int8), (2, 8)),
        (torch.zeros(2, 2, dtype=torch.int32), (2, 8)),
    ],
)
def test_direct_repack_rejects_invalid_inputs(packed, shape):
    with pytest.raises(ValueError, match="Direct CT repack"):
        _module.repack_ct_int4(packed, shape)


def test_expanded_loader_keeps_original_codes_with_fast_switch():
    codes = (torch.arange(190).reshape(10, 19) % 16 - 8).to(torch.int8)
    with patch.dict(os.environ, {"VLLM_ASCEND_CT_FAST_LOAD": "1"}):
        result = list(
            _module.unpack_packed_moe_weights(
                [("0.gate_proj.weight_packed", pack_to_int32(codes, 4))], {"gate_proj": (10, 19)}
            )
        )
    assert torch.equal(result[0][1], codes)


def test_mapping_index_preserves_order_replicas_and_projection_selection():
    mapping = [("experts.w13_", f"experts.{i}.gate_proj.", i, "w1") for i in range(288)]
    mapping += [("experts.w13_", "experts.1.gate_proj.", 288, "w1")]
    with patch.dict(os.environ, {"VLLM_ASCEND_CT_FAST_LOAD": "1"}):
        indexed = _module.index_expert_mapping(mapping)
        for expert in (0, 1, 11, 72, 143, 287):
            name = f"model.layers.45.mtp_block.mlp.experts.{expert}.gate_proj.weight_scale"
            old = [entry for entry in mapping if entry[1] in name]
            new = [entry for entry in _module.expert_mapping_candidates(mapping, indexed, name) if entry[1] in name]
            assert new == old
    assert len(indexed["1"]) == 2


def test_mapping_disable_unknown_and_nonexpert_fall_back_to_original_scan():
    mapping = [("experts.w13_", "experts.1.gate_proj.", 1, "w1")]
    with patch.dict(os.environ, {"VLLM_ASCEND_CT_FAST_LOAD": "0"}):
        assert _module.index_expert_mapping(mapping) is None
    with patch.dict(os.environ, {"VLLM_ASCEND_CT_FAST_LOAD": "1"}):
        assert _module.index_expert_mapping([("w", "experts.gate_up_proj", 0, "w1")]) is None
        index = _module.index_expert_mapping(mapping)
        assert _module.expert_mapping_candidates(mapping, index, "layers.0.mlp.down_proj.weight") is mapping


def test_tp8_native_slices_of_direct_pack_equal_original_pack():
    # All eight TP ranks: gate/up shard N; down shards K. Packing stays global
    # and the original native loader still slices once, with the original rank.
    codes = (torch.arange(64 * 128).reshape(64, 128) % 16 - 8).to(torch.int8)
    packed = _module.repack_ct_int4(pack_to_int32(codes, 4), (64, 128))
    for rank in range(8):
        assert torch.equal(
            packed[rank * 4 : (rank + 1) * 4], _module.pack_output_int4(codes[rank * 8 : (rank + 1) * 8])
        )
        assert torch.equal(
            packed[:, rank * 16 : (rank + 1) * 16], _module.pack_output_int4(codes[:, rank * 16 : (rank + 1) * 16])
        )
