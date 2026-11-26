#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""LLM-Compressor (compressed_tensors) quantization configuration for Ascend."""

import json
import re
from typing import Any, Optional, cast

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD, HIF8_C8_KV_CACHE_DTYPE, is_hif8_c8_kv_quant

from ..methods import AscendLinearScheme, AscendMoEScheme
from ..utils import get_model_file, is_fused_moe_layer

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, "QuantizationArgs"] | None]

# GLM-5.3-Flash decoder/attention prefix (same shape as the ModelSlim HiF8
# layer matching). The runtime ``prefix`` handed to ``get_quant_method`` for
# the sparse-MLA (DSA) attention can be the ``self_attn`` wrapper, ``.attn``,
# or the nested ``.mla_attn.mla_attn`` attention layer; the layer index is
# captured so the checkpoint keys can be matched back to it.
_GLM5NEXT_SELF_ATTN = re.compile(
    r"^(?:model\.language_model|language_model\.model|model)\.layers\.(\d+)\.(?:mtp_block\.)?self_attn"
)

# Checkpoint tensor name of a HiF8 DSA scale exported by llm-compressor:
# ``<hf-prefix>.layers.<idx>.(mtp_block.)?self_attn.(fa_q|fa_k).scale``.
_HIF8_SCALE_KEY_RE = re.compile(r"layers\.(\d+)\.self_attn\.(fa_q|fa_k)\.scale$")


@register_quantization_config(COMPRESSED_TENSORS_METHOD)
class AscendCompressedTensorsConfig(QuantizationConfig):
    """Config class for LLM-Compressor (compressed_tensors) quantization on Ascend.

    This class adapts the compressed_tensors format to work with Ascend's
    quantization implementations.
    """

    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.quant_description = config
        # Lazily-resolved set of decoder layers that carry a HiF8 DSA recipe in
        # the checkpoint (``fa_q.scale``/``fa_k.scale``). Compressed-tensors
        # config.json has no per-layer HiF8 metadata, so the checkpoint's
        # safetensors key list is the authoritative source (see
        # ``_hif8_scale_layer_flags``).
        self._hif8_scale_layers_cache: dict[str, dict[str, bool]] | None = None

    def get_name(self) -> str:
        return COMPRESSED_TENSORS_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _add_fused_moe_to_target_scheme_map(self):
        """
        Helper function to update target_scheme_map
        since linear layers get fused into MoE modules
        targeting 'Linear' needs to also match
        RoutedExperts modules.
        """
        if "Linear" not in self.target_scheme_map or "RoutedExperts" in self.target_scheme_map:
            return
        self.target_scheme_map["RoutedExperts"] = self.target_scheme_map["Linear"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendCompressedTensorsConfig":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    @classmethod
    def _quantization_scheme_map_from_config(cls, config: dict[str, Any]) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """Build target scheme map from config.

        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """

        target_scheme_map: dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(quant_config.get("weights"))

                target_scheme_map[target]["input_activations"] = None
                target_scheme_map[target]["format"] = quant_config.get("format")
                format = target_scheme_map[target].get("format")
                # If no per-config format defined, use global format in config
                act_quant_format = (
                    is_activation_quantization_format(format)
                    if format is not None
                    else is_activation_quantization_format(quant_format)
                )
                mixed_precision_format = quant_format == CompressionFormat.mixed_precision.value

                input_activations = quant_config.get("input_activations")
                if (act_quant_format or mixed_precision_format) and input_activations is not None:
                    target_scheme_map[target]["input_activations"] = QuantizationArgs.model_validate(input_activations)
        return target_scheme_map

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from ..method_adapters import AscendFusedMoEMethod, AscendLinearMethod

        if isinstance(layer, LinearBase):
            # Get the scheme for this layer
            linear_scheme = self._get_linear_scheme(layer=layer, layer_name=prefix)

            # Return unquantized method if no scheme found
            if linear_scheme is None:
                return AscendUnquantizedLinearMethod()

            # Store scheme on layer for reference (optional, for debugging)
            layer.scheme = linear_scheme
            logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
            return AscendLinearMethod(linear_scheme)

        if is_fused_moe_layer(layer):
            # Delayed import to avoid circular import
            from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

            layer_name = prefix + ".0.gate_proj"
            # Get the scheme for this layer
            moe_scheme = self._get_moe_scheme(layer=layer, layer_name=layer_name)

            # Return unquantized method if no scheme found
            if moe_scheme is None:
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)

            # Store scheme on layer for reference (optional, for debugging)
            layer.scheme = moe_scheme
            logger.info_once("Using the vLLM Ascend llmcompressor Quantization now!")
            return AscendFusedMoEMethod(moe_scheme, layer.moe_config)

        if isinstance(layer, AttentionLayerBase):
            return self._get_attention_quant_method(layer, prefix)

        return None

    def _get_attention_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        """Resolve the quant method for an attention (KV-cache) layer.

        This mirrors ModelSlim's HiF8 branch (``AscendModelSlimConfig``), but
        determines HiF8 layer membership from the checkpoint's
        ``fa_q.scale``/``fa_k.scale`` tensor presence instead of ModelSlim's
        per-layer ``quant_type`` metadata, which compressed-tensors config.json
        does not carry. The downstream scheme, scale loaders and the HiF8
        quantize operator are identical for both producers.
        """
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None

        if (
            is_hif8_c8_kv_quant(vllm_config)
            and self._is_hif8_quant_layer(prefix)
        ):
            from ..method_adapters import AscendKVCacheMethod
            from ..methods.kv_cache.hif8_c8 import (
                AscendHIF8KVCacheAttentionMethod,
                validate_sparse_sfa_hif8,
            )

            hif8_layers = [
                layer_idx
                for layer_idx, flags in self._hif8_scale_layer_flags().items()
                if flags["fa_q"] and flags["fa_k"]
            ]
            validate_sparse_sfa_hif8(vllm_config, hif8_layers)
            logger.debug("Select AscendKVCacheMethod(HIF8) for %s", prefix)
            return AscendKVCacheMethod(AscendHIF8KVCacheAttentionMethod())

        return None

    def _is_hif8_quant_layer(self, prefix: str) -> bool:
        """Whether ``prefix`` is a decoder layer whose checkpoint carries both
        ``fa_q.scale`` and ``fa_k.scale`` (the llm-compressor HiF8 DSA recipe).

        The runtime ``prefix`` must name the sparse-MLA attention itself, not a
        sibling submodule (e.g. the indexer/tail cache layers also register as
        ``AttentionLayerBase`` under ``self_attn.indexer`` / ``self_attn.tail``).
        """
        match = _GLM5NEXT_SELF_ATTN.match(prefix)
        if match is None:
            return False
        if prefix[match.end() :] not in ("", ".attn", ".mla_attn.mla_attn"):
            return False
        flags = self._hif8_scale_layer_flags().get(match.group(1))
        return flags is not None and flags["fa_q"] and flags["fa_k"]

    def _hif8_scale_layer_flags(self) -> dict[str, dict[str, bool]]:
        """Return ``{layer_index: {"fa_q": bool, "fa_k": bool}}`` for every
        decoder layer that exports HiF8 DSA scale tensors, resolved lazily and
        cached (the safetensors metadata is read at most once)."""
        if self._hif8_scale_layers_cache is None:
            self._hif8_scale_layers_cache = self._read_hif8_scale_layers()
        return self._hif8_scale_layers_cache

    def _read_hif8_scale_layers(self) -> dict[str, dict[str, bool]]:
        """Inspect the checkpoint's safetensors metadata for HiF8 scale keys.

        llm-compressor exports the static HiF8 DSA scales (``fa_q.scale`` /
        ``fa_k.scale``) as plain buffers, so they are not described by the
        compressed-tensors ``config_groups``. Their presence in the checkpoint
        is therefore the only producer-independent signal that a layer uses the
        HiF8 (sparse DSA) KV cache. Only the safetensors *index* / *header* is
        read here -- no weight tensor is loaded -- and the result is cached.
        """
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return {}
        model_config = vllm_config.model_config
        model = model_config.model
        revision = getattr(model_config, "revision", None)

        keys: list[str] = []
        # Sharded checkpoints carry a small index mapping every weight name to
        # its shard; prefer it (no tensor I/O).
        index_path = get_model_file(model, "model.safetensors.index.json", revision=revision)
        if index_path is not None:
            with open(index_path, encoding="utf-8") as index_stream:
                index = json.load(index_stream)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if isinstance(weight_map, dict):
                keys = list(weight_map)

        if not keys:
            # Single-file checkpoint: read only the safetensors header keys.
            from safetensors import SafetensorError, safe_open

            for candidate in ("model.safetensors",):
                path = get_model_file(model, candidate, revision=revision)
                if path is None:
                    continue
                try:
                    with safe_open(path, framework="pt", device="cpu") as checkpoint:
                        keys = list(checkpoint.keys())
                except SafetensorError as exc:
                    logger.warning("Could not read HiF8 scale metadata from %s: %s", path, exc)
                    continue
                break

        flags: dict[str, dict[str, bool]] = {}
        for key in keys:
            match = _HIF8_SCALE_KEY_RE.search(key)
            if match is None:
                continue
            layer_flags = flags.setdefault(match.group(1), {"fa_q": False, "fa_k": False})
            layer_flags[match.group(2)] = True
        return flags

    def _get_linear_scheme(self, layer: torch.nn.Module, layer_name: str | None = None) -> AscendLinearScheme | None:
        """Get the linear quantization scheme for a layer.

        Returns:
            An AscendLinearScheme instance, or None if the layer
            should use unquantized method.
        """
        weight_quant, input_quant, format = self._get_quant_args(layer, layer_name)
        if weight_quant is None:
            return None

        scheme = self._create_scheme_for_layer_type(
            weight_quant=weight_quant,
            input_quant=input_quant,
            format=format,
            layer_type="linear",
        )
        return cast(AscendLinearScheme, scheme)

    def _get_moe_scheme(self, layer: torch.nn.Module, layer_name: str | None = None) -> AscendMoEScheme | None:
        """Get the MoE quantization scheme for a layer.

        Returns:
            An AscendMoEScheme instance, or None if the layer
            should use unquantized method.
        """
        # Add FusedMoEFactory to target scheme map if needed
        self._add_fused_moe_to_target_scheme_map()

        weight_quant, input_quant, format = self._get_quant_args(layer, layer_name)
        if weight_quant is None:
            return None

        scheme = self._create_scheme_for_layer_type(
            weight_quant=weight_quant,
            input_quant=input_quant,
            format=format,
            layer_type="moe",
        )
        return cast(AscendMoEScheme, scheme)

    def _get_quant_args(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> tuple[Optional["QuantizationArgs"], Optional["QuantizationArgs"], str | None]:
        """Extract quantization arguments for a layer.

        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target.

        Returns:
            A tuple of (weight_quant, input_quant, format). weight_quant is
            None if the layer should use unquantized method.
        """
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        weight_quant = None
        input_quant = None
        format = None
        if scheme_dict:
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            format = scheme_dict.get("format")

        if weight_quant is None:
            logger.warning_once(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )

        return weight_quant, input_quant, format

    def get_scheme_dict(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> dict[str, QuantizationArgs | str | None] | None:
        """
        Extract the QuantizationArgs for a given layer.

        Returns:
            dict with {
                "weights": QuantizationArgs,
                "input_activations": QuantizationArgs | None,
                "format": str | None
            } | None
        """
        if should_ignore_layer(layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping):
            return None

        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )
            if matched_target is None:
                return None
            scheme_dict = self.target_scheme_map[matched_target]
            if scheme_dict is None:
                return None
            if scheme_dict.get("format") is None:
                scheme_dict["format"] = self.quant_format
            return scheme_dict

        return None

    def _create_scheme_for_layer_type(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        format: str | None,
        layer_type: str,
    ) -> AscendLinearScheme | AscendMoEScheme:
        """Create the appropriate Ascend scheme based on quantization args and layer type.

        Args:
            weight_quant: Weight quantization arguments.
            input_quant: Input activation quantization arguments.
            format: Per-layer format, if defined.
            layer_type: Type of layer ("linear" or "moe").

        Returns:
            An instance of the appropriate Ascend quantization scheme.
        """
        from ..methods import get_scheme_class

        # Determine the quantization type
        quant_type = self._detect_quant_type(weight_quant, input_quant, format)

        # Get the scheme class from registry
        scheme_cls = get_scheme_class(quant_type, layer_type)
        if scheme_cls is None:
            raise NotImplementedError(
                f"No compressed-tensors compatible scheme was found for "
                f"quant_type={quant_type}, layer_type={layer_type}."
            )

        return scheme_cls()

    def _detect_quant_type(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        format: str | None,
    ) -> str:
        """Detect the quantization type from quantization arguments.

        Args:
            weight_quant: Weight quantization arguments.
            input_quant: Input activation quantization arguments.
            format: Per-layer format, if defined.

        Returns:
            A string representing the quantization type (e.g., "W8A8", "W8A8_DYNAMIC").
        """
        # use the per-layer format if defined, otherwise, use global format
        format = format if format is not None else self.quant_format
        # Native Kimi-K3 checkpoints contain packed MXFP4 weights without
        # an activation recipe. Reuse the Ascend W4A8 backend, which performs
        # dynamic MXFP8 activation quantization at runtime.
        if self._is_only_weight_mxfp4_packed(weight_quant, input_quant, format):
            assert self.quant_description is not None
            self.quant_description["group_size"] = weight_quant.group_size
            return "W4A8_MXFP"

        act_quant_format = is_activation_quantization_format(format)

        # MXFP4 weights with explicitly configured dynamic MXFP8 activations.
        if format == CompressionFormat.mxfp4_pack_quantized.value and self._is_dynamic_group_mx(
            weight_quant=weight_quant,
            input_quant=input_quant,
            num_bits=4,
            input_num_bits=8,
        ):
            assert self.quant_description is not None
            self.quant_description["group_size"] = weight_quant.group_size or 32
            return "W4A8_MXFP"

        # MXFP8: static FP8 group-wise weights + dynamic FP8 group-wise activations.
        if format == CompressionFormat.mxfp8_quantized.value and self._is_dynamic_group_mx(
            weight_quant=weight_quant,
            input_quant=input_quant,
            num_bits=8,
        ):
            assert self.quant_description is not None
            self.quant_description["group_size"] = weight_quant.group_size or 32
            return "W8A8_MXFP8"

        # MXFP4: static packed FP4 group-wise weights + dynamic FP4 group-wise activations.
        if format == CompressionFormat.mxfp4_pack_quantized.value and self._is_dynamic_group_mx(
            weight_quant=weight_quant,
            input_quant=input_quant,
            num_bits=4,
        ):
            assert self.quant_description is not None
            self.quant_description["group_size"] = weight_quant.group_size or 32
            return "W4A4_MXFP4"

        if act_quant_format and input_quant is not None:
            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                return "W8A8"

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                if weight_quant.type == QuantizationType.FLOAT and input_quant.type == QuantizationType.FLOAT:
                    return "W8A8FP8_DYNAMIC"
                else:
                    return "W8A8_DYNAMIC"

            if self._is_dynamic_token_w4a8(weight_quant, input_quant):
                return "W4A8_DYNAMIC"

        if self._is_w4a16(weight_quant, input_quant):
            return "W4A16"

        raise NotImplementedError("No compressed-tensors compatible quantization type was found.")

    def _is_only_weight_mxfp4_packed(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        format: str | None,
    ) -> bool:
        if format != CompressionFormat.mxfp4_pack_quantized.value or weight_quant is None or input_quant is not None:
            return False
        return (
            weight_quant.type == QuantizationType.FLOAT
            and weight_quant.num_bits == 4
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and weight_quant.group_size == 32
            and not weight_quant.dynamic
            and weight_quant.symmetric
        )

    def _is_dynamic_group_mx(
        self,
        weight_quant: "QuantizationArgs",
        input_quant: Optional["QuantizationArgs"],
        num_bits: int,
        input_num_bits: int | None = None,
    ) -> bool:
        """Detect MXFP group-wise weight and dynamic activation quantization."""
        if input_quant is None:
            return False

        is_float = weight_quant.type == input_quant.type == QuantizationType.FLOAT
        is_expected_bits = weight_quant.num_bits == num_bits and input_quant.num_bits == (input_num_bits or num_bits)
        is_group = (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
            and input_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric
        is_group_size_32 = weight_quant.group_size == input_quant.group_size == 32

        return is_float and is_expected_bits and is_group and is_dynamic and is_symmetric and is_group_size_32

    def _is_static_tensor_w8a8(self, weight_quant: "QuantizationArgs", input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        is_tensor = weight_strategy and input_quant.strategy == QuantizationStrategy.TENSOR.value
        is_static = not weight_quant.dynamic and not input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and is_symmetric and is_static

    def _is_dynamic_token_w8a8(self, weight_quant: "QuantizationArgs", input_quant: "QuantizationArgs") -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        is_token = weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_dynamic_token_w4a8(self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs) -> bool:
        is_4_bits = weight_quant.num_bits == 4
        is_8_bits = input_quant.num_bits == 8
        weight_strategy = weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        is_token = weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        # Only symmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_4_bits and is_8_bits and is_token and is_symmetric and is_dynamic

    def _is_w4a16(self, weight_quant: "QuantizationArgs", input_quant: Optional["QuantizationArgs"]) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have integer type.
        if weight_quant.type != QuantizationType.INT:
            return False

        input_quant_none = input_quant is None
        is_4_bits = weight_quant.num_bits == 4
        is_group = weight_quant.strategy == QuantizationStrategy.GROUP.value
        is_static = not weight_quant.dynamic

        return input_quant_none and is_4_bits and is_group and is_static

    def get_cache_scale_mapper(self) -> "WeightsMapper":
        """Remap llm-compressor's HiF8 DSA scale keys to the nested MLA layer.

        llm-compressor exports the static HiF8 DSA scales on the outer
        ``self_attn`` (``self_attn.fa_q/fa_k.scale``), while AscendMLAAttention
        registers them on the nested ``mla_attn.mla_attn`` layer. ModelSlim keeps
        this remap in its own ``get_cache_scale_mapper``; compressed-tensors does
        the same here so both backbone and MTP loaders share one source of truth.
        """
        orig_to_new_regex = {
            re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_q\.scale$"): ".mla_attn.mla_attn.fa_q.scale",
            re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_k\.scale$"): ".mla_attn.mla_attn.fa_k.scale",
        }
        mapper = WeightsMapper(orig_to_new_regex=orig_to_new_regex)
        return mapper | QuantizationConfig.get_cache_scale_mapper()

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        self.target_scheme_map = hf_to_vllm_mapper.apply_dict(self.target_scheme_map)
        self.ignore = hf_to_vllm_mapper.apply_list(self.ignore)
