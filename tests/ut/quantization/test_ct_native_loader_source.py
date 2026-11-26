# SPDX-License-Identifier: Apache-2.0
"""Run native RoutedExperts source loaders on CPU without plugin startup.

Only distributed placement and online-quantizer imports are isolated. Actual
checkpoint mapping, TP slicing, scale loading and compact CT packing execute.
Run directly with --vllm-root pointing to the paired upstream checkout.
"""

import argparse
import importlib
import importlib.util
import math
import re
import sys
import types
import typing
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = ROOT.parent / "vllm"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNativeCTLoaderSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = VLLM_ROOT / "vllm/model_executor/layers/fused_moe/routed_experts.py"
        if not source.is_file():
            raise unittest.SkipTest("Paired vLLM source checkout required")
        imports = patch.dict(sys.modules)
        imports.start()
        cls.addClassCleanup(imports.stop)
        fixture = load_source("ct_source_fixture", Path(__file__).with_name("_ct_test_utils.py"))
        scope = dict(
            Module=torch.nn.Module,
            torch=torch,
            Enum=Enum,
            overload=typing.overload,
            cast=typing.cast,
            cdiv=lambda a, b: math.ceil(a / b),
            logger=types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
            # All cases use zero redundant experts; isolate placement only.
            EplbState=types.SimpleNamespace(
                build_initial_global_physical_to_logical_map=lambda n, r: torch.arange(n + r) % n
            ),
        )
        fixture.execute_nodes(
            source,
            scope,
            functions=("_index_expert_mapping",),
            classes={"FusedMoeWeightScaleSupported": ("Enum", None), "RoutedExperts": ("Module", None)},
        )
        fixture.module(
            "vllm.model_executor.layers.quantization.online.moe_base",
            OnlineMoEMethodBase=type("OnlineMoEMethodBase", (), {}),
        )
        fixture.module("vllm.model_executor.layers.fused_moe.routed_experts", RoutedExperts=scope["RoutedExperts"])
        fixture.module("vllm_ascend", __path__=[str(ROOT / "vllm_ascend")])
        package = types.ModuleType("ct_native_pack_fixture")
        package.__path__ = [str(ROOT / "vllm_ascend/quantization")]
        sys.modules[package.__name__] = package
        packed = importlib.import_module(f"{package.__name__}.compressed_tensors_packed")
        sys.modules["vllm_ascend.quantization.compressed_tensors_packed"] = packed
        cls.cases = load_source(
            "native_loader_cases", Path(__file__).with_name("test_compressed_tensors_packed_loader.py")
        )
        cls.packed = packed
        cls.fixture = fixture

    def test_expanded_codes_padding_and_expert_partition(self):
        partitions = [(size, rank, (0, 1, 2, 3)) for size in (1, 2, 4) for rank in range(size)]
        partitions += [(1, 0, (0, 2)), (1, 0, (1, 3))]
        for size, rank, local_ids in partitions:
            for padded in (False, True):
                with self.subTest(tp=size, rank=rank, local_ids=local_ids, padded=padded):
                    self.cases.test_native_tp_ep_loading_preserves_codes_and_scales(size, rank, local_ids, padded)

    def test_compact_resident_codes_and_scales(self):
        partitions = [(size, rank, (0, 1, 2, 3)) for size in (1, 2, 4, 8) for rank in range(size)]
        partitions += [(1, 0, (1, 3))]
        for size, rank, local_ids in partitions:
            with self.subTest(tp=size, rank=rank, local_ids=local_ids):
                self.cases.test_native_compact_resident_int4_loads_with_main_tp_ep(size, rank, local_ids)

    def test_glm_main_and_mtp_owned_loaders_preserve_experts_and_shared_scales(self):
        from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

        for mtp in (False, True):
            for fast in (False, True):
                with self.subTest(mtp=mtp, fast=fast):
                    layer = self.cases._make_experts(2, 1, (1, 3), False)
                    reference = self.cases._make_experts(2, 1, (1, 3), False)
                    # Expanded output also exercises the generic INT8-code
                    # path. Compact output is covered for all TP ranks above.
                    layer.scheme = types.SimpleNamespace(is_ct_packed=True, ct_packed_output=False)
                    prefix = "model.layers.2.mtp_block.mlp" if mtp else "layers.0.mlp"
                    incoming = "model.language_model.layers.2.mlp" if mtp else prefix
                    params = {f"{prefix}.experts.{name}": p for name, p in layer.named_parameters()}
                    weights, raw = [], []
                    for expert in range(4):
                        for projection, shape in {
                            "gate_proj": (12, 19),
                            "up_proj": (12, 19),
                            "down_proj": (19, 12),
                        }.items():
                            codes = ((torch.arange(math.prod(shape)).reshape(shape) + expert) % 16 - 8).to(torch.int8)
                            scale = ((torch.arange(shape[0]) + expert + 1) / 32).bfloat16().unsqueeze(1)
                            stem = f"{expert}.{projection}"
                            raw.extend([(stem + ".weight", codes), (stem + ".weight_scale", scale)])
                            weights.extend(
                                [
                                    (f"{incoming}.experts.{stem}.weight_packed", pack_to_int32(codes, 4)),
                                    (f"{incoming}.experts.{stem}.weight_scale", scale),
                                    (f"{incoming}.experts.{stem}.weight_shape", torch.tensor(shape)),
                                ]
                            )
                    list(reference.load_weights(raw))
                    expected_shared = {}
                    for suffix, shape, dtype in (
                        ("weight", (8, 19), torch.int8),
                        ("weight_scale", (8, 1), torch.bfloat16),
                    ):
                        param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), False)

                        def merge(param, value, shard_id):
                            param.data[shard_id * 4 : (shard_id + 1) * 4].copy_(value)

                        param.weight_loader = merge
                        params[f"{prefix}.shared_experts.gate_up_proj.{suffix}"] = param
                        sources = []
                        for shard, projection in enumerate(("gate_proj", "up_proj")):
                            value = (torch.arange(math.prod((4, shape[1]))).reshape(4, shape[1]) + shard + 1).to(dtype)
                            weights.append((f"{incoming}.shared_experts.{projection}.{suffix}", value))
                            sources.append(value)
                        expected_shared[suffix] = torch.cat(sources)
                    mapping = [
                        (f"experts.{target}", f"experts.{expert}.{projection}.", expert, shard)
                        for expert in range(4)
                        for projection, target, shard in (
                            ("gate_proj", "w13_", "w1"),
                            ("up_proj", "w13_", "w3"),
                            ("down_proj", "w2_", "w2"),
                        )
                    ]

                    def spec_layer(config, name):
                        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                        return int(match[1]) if match and int(match[1]) >= 2 else None

                    scope = dict(
                        Module=torch.nn.Module,
                        torch=torch,
                        typing=typing,
                        Callable=typing.Callable,
                        _is_moe=lambda config: True,
                        _is_linear_attn=lambda config: False,
                        vllm_version_is=lambda version: False,
                        fused_moe_make_expert_params_mapping=lambda *a, _mapping=mapping, **k: _mapping,
                        get_spec_layer_idx_from_weight_name=spec_layer,
                        _try_load_fp8_indexer_wk=lambda *a: False,
                        is_pp_missing_parameter=lambda *a: False,
                        maybe_remap_kv_scale_name=lambda name, params: name,
                        default_weight_loader=lambda p, value, **k: p.data.copy_(value),
                    )
                    name = "Glm5NextMTP" if mtp else "Glm5NextModel"
                    methods = (
                        {"load_weights", "_rewrite_spec_layer_name", "_maybe_set_own_lm_head"}
                        if mtp
                        else {"load_weights"}
                    )
                    self.fixture.execute_nodes(
                        ROOT / f"vllm_ascend/models/glm5next/{'mtp' if mtp else 'model'}.py",
                        scope,
                        classes={name: ("Module", methods)},
                    )
                    model = scope[name].__new__(scope[name])
                    torch.nn.Module.__init__(model)
                    model.config = types.SimpleNamespace(hidden_size=19, moe_intermediate_size=12, n_routed_experts=4)
                    model.quant_config = None
                    model.modules = lambda layer=layer: iter((layer,))
                    model.named_parameters = lambda params=params: iter(params.items())
                    if mtp:
                        model.model = types.SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=1)
                        model.is_rot_weight_used = False
                    else:
                        fn_name = "layers.0.hc_attn_fn"
                        gate_name = "layers.0.self_attn.indexer.index_kpool_compress_gate"
                        for key in (fn_name, gate_name):
                            params[key] = torch.nn.Parameter(torch.zeros(4, 19), False)
                        weights.extend([(fn_name, torch.ones(4, 19)), (gate_name + ".weight", torch.ones(4, 19))])
                    with patch.object(self.packed.envs, "VLLM_ASCEND_CT_FAST_LOAD", fast):
                        loaded = model.load_weights(iter(weights))
                    for key, value in reference.named_parameters():
                        torch.testing.assert_close(getattr(layer, key), value, rtol=0, atol=0)
                    for suffix, value in expected_shared.items():
                        key = f"{prefix}.shared_experts.gate_up_proj.{suffix}"
                        assert key in loaded
                        torch.testing.assert_close(params[key], value, rtol=0, atol=0)
                    if mtp:
                        assert not model.has_own_lm_head and not model.has_own_embed_tokens
                        with self.assertRaisesRegex(ValueError, "weights missing"):
                            model.load_weights(iter(()))
                    else:
                        assert fn_name in loaded and gate_name in loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    args, remaining = parser.parse_known_args()
    VLLM_ROOT = args.vllm_root
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
