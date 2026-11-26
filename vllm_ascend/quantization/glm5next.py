# SPDX-License-Identifier: Apache-2.0
"""GLM MLP-only CT metadata and floating checkpoint name compatibility."""

import re

_MLP_PROJECTION = re.compile(r"(?:.*\.)?mlp\.(?:(?:experts\.\d+|shared_experts)\.)?(?:gate|up|down|gate_up)_proj")
_QUANTIZED_SUFFIXES = frozenset(
    {
        "weight_packed",
        "weight_shape",
        "weight_scale",
        "weight_scale_inv",
        "weight_zero_point",
        "weight_offset",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
        "scale_bias",
    }
)


def is_glm_ct_int_config(config) -> bool:
    """Select loader validation only; never enable non-MLP quantization."""
    if config is None or config.get_name() != "compressed-tensors":
        return False
    schemes = list(getattr(config, "target_scheme_map", {}).values())
    if not schemes:
        return False
    for scheme in schemes:
        if scheme is None:
            continue
        weight, activation = scheme.get("weights"), scheme.get("input_activations")
        if weight is None or activation is None:
            return False
        if not (
            weight.type == activation.type == "int"
            and weight.num_bits in (4, 8)
            and weight.strategy == "channel"
            and weight.symmetric
            and not weight.dynamic
            and activation.num_bits == 8
            and activation.strategy == "token"
            and activation.symmetric
            and activation.dynamic is True
        ):
            return False
    return any(scheme is not None for scheme in schemes)


def glm5next_packed_mapping(inherited=None) -> dict[str, list[str]]:
    """Return fresh mappings, preserving inherited vision projections."""
    mapping = {name: list(parts) for name, parts in (inherited or {}).items()}
    mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
    return mapping


def prepare_glm_mlp_weights(weights):
    """Stream MLP-only CT weights, accepting both floating KPool gate names.

    Integer non-MLP weights or their qparams belong to the old broad recipe.
    Reject them rather than loading integer codes into floating projections.
    Tensor values and optional per-weight loader kwargs pass through unchanged.
    """
    seen_gates = set()
    for entry in weights:
        name, tensor = entry[:2]
        extra = entry[2:]
        if name.endswith(".index_kpool_compress_gate.weight"):
            name = name.removesuffix(".weight")
        if name.endswith(".index_kpool_compress_gate"):
            if not tensor.is_floating_point():
                raise ValueError(f"MLP-only CT requires a floating KPool gate: {name} ({tensor.dtype})")
            if name in seen_gates:
                raise ValueError(f"Duplicate KPool gate checkpoint aliases: {name}")
            seen_gates.add(name)
        stem, _, suffix = name.rpartition(".")
        is_mlp = _MLP_PROJECTION.fullmatch(stem) is not None
        if not is_mlp and (suffix in _QUANTIZED_SUFFIXES or (suffix == "weight" and not tensor.is_floating_point())):
            raise ValueError(
                f"GLM CT now supports MLP-only INT quantization; non-MLP quantized tensor {name!r} "
                "belongs to a broad-scope checkpoint. Use the MLP-only checkpoint regenerated from BF16, "
                "or the earlier broad-scope serving revision."
            )
        yield (name, tensor, *extra)
