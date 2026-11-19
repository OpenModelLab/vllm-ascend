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
        fixture = load_source("ct_source_fixture", Path(__file__).with_name("test_glm5next_ct_linears.py"))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    args, remaining = parser.parse_known_args()
    VLLM_ROOT = args.vllm_root
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
