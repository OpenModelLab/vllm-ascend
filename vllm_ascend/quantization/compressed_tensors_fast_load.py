# SPDX-License-Identifier: Apache-2.0
"""Opt-in host-side CT loading shortcuts; no quantization or kernel changes."""

import re
import sys

import torch

from vllm_ascend import envs

_EXPERT_NAME = re.compile(r"^((?:.*\.)?experts)\.(\d+)\.")
_INT4_VALUES_PER_INT32 = 8


def repack_ct_int4(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """CT offset-binary [N,K/8] INT32 -> signed [N/2,K] INT8 bytes.

    Pair output channels directly in byte space, without materializing [N,K]
    INT8 codes. Padded input columns are cropped after packing. This is CPU-only.
    """
    n, k = shape
    expected = (n, (k + _INT4_VALUES_PER_INT32 - 1) // _INT4_VALUES_PER_INT32)
    if packed.device.type != "cpu" or packed.dtype != torch.int32 or tuple(packed.shape) != expected or n % 2:
        raise ValueError("Direct CT repack requires CPU INT32 packed weights and even N")
    data = packed.contiguous().view(torch.uint8)
    if sys.byteorder == "big":
        data = data.reshape(n, -1, 4).flip(-1).flatten(1)
    # Adding/removing the CT offset of 8 toggles each nibble's sign bit.
    even = data[0::2] ^ 0x88
    odd = data[1::2] ^ 0x88
    output = torch.empty((n // 2, data.shape[1] * 2), dtype=torch.uint8)
    output[:, 0::2] = (even & 0x0F) | (odd << 4)
    output[:, 1::2] = (even >> 4) | (odd & 0xF0)
    return output[:, :k].contiguous().view(torch.int8)


def index_expert_mapping(mapping):
    """Index logical checkpoint IDs, preserving physical-replica entry order."""
    if not envs.VLLM_ASCEND_CT_FAST_LOAD:
        return None
    indexed = {}
    for entry in mapping:
        match = _EXPERT_NAME.match(entry[1])
        if match is None or match[1] != "experts":
            return None  # Retain the original full scan for other formats.
        indexed.setdefault(match[2], []).append(entry)
    return indexed


def expert_mapping_candidates(mapping, indexed, name):
    if indexed is not None:
        match = _EXPERT_NAME.match(name)
        if match is not None:
            return indexed.get(match[2], ())
    return mapping
