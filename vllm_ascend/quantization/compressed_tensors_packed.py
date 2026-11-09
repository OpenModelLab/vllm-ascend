# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load packed per-channel INT4 experts through the native INT8-code loader."""

from collections.abc import Iterable, Iterator, Mapping

import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

_INT4_BITS = 4
_INT4_PACK_FACTOR = 32 // _INT4_BITS


def unpack_packed_moe_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    projection_shapes: Mapping[str, tuple[int, int]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Unpack one CPU matrix at a time, before native TP slicing and fusion.

    Shapes come from the model, so weight_shape may precede or follow the data,
    including in another shard/load_weights call. No tensor buffering is needed.
    Scales and other tensors pass through unchanged; integer codes are never
    dequantized or fitted again.
    """
    for name, tensor in weights:
        stem, _, suffix = name.rpartition(".")
        if suffix not in ("weight_packed", "weight_shape"):
            yield name, tensor
            continue

        expert, _, projection = stem.rpartition(".")
        if projection not in projection_shapes or not expert.rsplit(".", 1)[-1].isdigit():
            raise ValueError(f"Packed W4A8 requires per-expert 2D projections, got {name!r}")
        shape = projection_shapes[projection]
        tensor = tensor.cpu()
        if suffix == "weight_shape":
            if tensor.dtype != torch.int64 or tensor.ndim != 1 or tuple(tensor.tolist()) != shape:
                raise ValueError(f"Invalid {name}: expected INT64 shape metadata {shape}, got {tensor}")
            continue

        packed_shape = (shape[0], (shape[1] + _INT4_PACK_FACTOR - 1) // _INT4_PACK_FACTOR)
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != packed_shape:
            raise ValueError(
                f"Invalid {name}: expected INT32 packed shape {packed_shape} for {shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        yield f"{stem}.weight", unpack_from_int32(tensor, _INT4_BITS, torch.Size(shape)).contiguous()


def install_packed_moe_loader(layer: torch.nn.Module) -> None:
    """Adapt this RoutedExperts instance without changing its compute method."""
    original_load_weights = layer.load_weights

    def load_weights(weights):
        config = layer.moe_config
        hidden = config.hidden_dim_unpadded or config.hidden_dim
        intermediate = config.intermediate_size
        shapes = {
            layer.ckpt_gate_proj_name: (intermediate, hidden),
            layer.ckpt_down_proj_name: (hidden, intermediate),
        }
        if layer.ckpt_up_proj_name is not None:
            shapes[layer.ckpt_up_proj_name] = (intermediate, hidden)
        return original_load_weights(unpack_packed_moe_weights(weights, shapes))

    layer.load_weights = load_weights
