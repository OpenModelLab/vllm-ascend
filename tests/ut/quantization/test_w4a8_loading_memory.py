# SPDX-License-Identifier: Apache-2.0
"""CPU numeric/allocation regressions; no NPU or vLLM import is required.

Run directly to avoid the repository's global NPU mock fixtures.
"""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_file("ct_w4a8_numeric", ROOT / "vllm_ascend/quantization/compressed_tensors_w4a8.py")


class FloatAllocations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in torch.utils._pytree.tree_leaves(result):
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                self.largest = max(self.largest, tensor.numel())
        return result


def extract_scheme(*extra_methods, **bindings):
    """Execute the actual methods, excluding NPU imports and initialization."""
    path = ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py"
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendW4A8DynamicFusedMoEMethod"
    )
    selected = {"get_weight", "process_weights_after_loading_compressed_tensors", *extra_methods}
    cls.body = [
        n
        for n in cls.body
        if (isinstance(n, ast.FunctionDef) and n.name in selected)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "fused_activations" for t in n.targets)
        )
    ]
    cls.bases = []
    cls.decorator_list = []
    scope = dict(
        torch=torch,
        Any=object,
        COMPRESSED_TENSORS_METHOD="compressed-tensors",
        process_expert_matrix=helper.process_expert_matrix,
        log_loading_memory=helper.log_loading_memory,
        _INT4_VALUES_PER_BYTE=2,
    )
    scope.update(bindings)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[cls.name]()


class TestW4A8LoadingMemory(unittest.TestCase):
    def test_ct_disables_fused_silu_without_changing_ascend_instances(self):
        for name, fused in (("compressed-tensors", False), ("ascend", True), ("compressed-tensors", False)):
            config = SimpleNamespace(
                quant_config=SimpleNamespace(quant_description={}, get_name=lambda name=name: name),
                parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_eplb=False),
                use_v2_model_runner=True,
            )
            method = extract_scheme(
                "__init__",
                get_current_vllm_config=lambda config=config: config,
                get_tensor_model_parallel_world_size=lambda: 4,
                get_mc2_group=lambda: SimpleNamespace(),
                ASCEND_QUANTIZATION_METHOD="ascend",
                _MODELSLIM_SCALE_BIAS_PARTITIONS=16,
            )
            self.assertEqual("silu" in method.fused_activations, fused)
            self.assertIn("situ", method.fused_activations)

    def test_signed_nibble_known_bytes(self):
        codes = torch.tensor([[-8, -1, 0, 7], [7, 0, -1, -8]], dtype=torch.int8)
        packed = helper.pack_output_int4(codes)
        self.assertEqual(packed.view(torch.uint8).tolist(), [[0x78, 0x0F, 0xF0, 0x87]])
        torch.testing.assert_close(helper.unpack_output_int4(packed), codes, rtol=0, atol=0)

    def test_noncontiguous_round_trip(self):
        codes = ((torch.arange(16 * 21).reshape(16, 21) % 16) - 8).to(torch.int8)[:, ::2]
        self.assertFalse(codes.is_contiguous())
        self.assertTrue(torch.equal(helper.unpack_output_int4(helper.pack_output_int4(codes)), codes))

    def test_bias_and_kernel_bytes_match_reference(self):
        for seed in (11, 47):
            torch.manual_seed(seed)
            codes = torch.randint(-8, 8, (7, 32, 64), dtype=torch.int8)
            scale = (torch.rand(7, 32, 1) + 0.001).to(torch.bfloat16)
            expected_bias = 8 * (codes.transpose(1, 2).contiguous().float() * scale.transpose(1, 2)).sum(dim=1)
            expected_bytes = np.empty((7, 64, 16), dtype=np.uint8)
            # Independent scalar nibble specification; no production pack call.
            for e in range(7):
                for k in range(64):
                    for n in range(16):
                        expected_bytes[e, k, n] = (int(codes[e, 2 * n, k]) & 15) | (
                            (int(codes[e, 2 * n + 1, k]) & 15) << 4
                        )
            for packed_output in (False, True):
                source = torch.stack([helper.pack_output_int4(w) for w in codes]) if packed_output else codes
                packed, bias = helper.process_expert_matrix(source, scale, packed_output=packed_output)
                np.testing.assert_array_equal(packed.view(torch.uint8).numpy(), expected_bytes)
                torch.testing.assert_close(bias, expected_bias, rtol=0, atol=0)
                self.assertEqual(packed.numel() * packed.element_size(), codes.numel() // 2)

    def test_fp32_temporaries_bound_to_one_expert(self):
        weight = torch.ones((7, 32, 64), dtype=torch.int8)
        scale = torch.ones((7, 32, 1), dtype=torch.bfloat16)
        for packed_output in (False, True):
            source = torch.stack([helper.pack_output_int4(w) for w in weight]) if packed_output else weight
            with FloatAllocations() as allocations:
                helper.process_expert_matrix(source, scale, packed_output=packed_output)
            self.assertLessEqual(allocations.largest, 32 * 64)
            self.assertLess(allocations.largest, weight.numel())

    def test_invalid_layout_rejected(self):
        with self.assertRaises(ValueError):
            helper.pack_output_int4(torch.zeros((3, 8), dtype=torch.int8))
        with self.assertRaises(ValueError):
            helper.process_expert_matrix(
                torch.zeros((2, 4, 8), dtype=torch.int8), torch.ones((2, 4, 1)), packed_output=True
            )

    def test_native_get_weight_resident_bytes_halved(self):
        method = extract_scheme()
        method.quant_method = "compressed-tensors"
        raw = method.get_weight(3, 16, 32, torch.bfloat16)
        if "ct_packed_output" not in method.get_weight.__code__.co_consts:
            self.skipTest("hotfix-only preserves expanded INT8 parameter allocation")
        method.ct_packed_output = True
        packed = method.get_weight(3, 16, 32, torch.bfloat16)
        self.assertEqual(sum(t.numel() for t in raw.values()), 2 * sum(t.numel() for t in packed.values()))
        self.assertEqual(packed["w13_weight"].shape, (3, 16, 32))
        self.assertEqual(packed["w2_weight"].shape, (3, 16, 16))

    def test_a2_ct_postload_requires_nz_even_when_general_policy_disables_it(self):
        format_cast = Mock(side_effect=lambda value, fmt: value)
        general_policy = Mock(side_effect=AssertionError("CT INT4 must not depend on the general NZ policy"))
        method = extract_scheme(
            "process_weights_after_loading",
            "_pack_to_int32",
            torch_npu=SimpleNamespace(npu_format_cast=format_cast),
            ACL_FORMAT_FRACTAL_NZ=29,
            maybe_trans_nz=general_policy,
            use_cann_megamoe=lambda cfg: False,
            get_current_vllm_config=lambda: None,
            _INT4_VALUES_PER_INT32=8,
        )
        method.quant_method = "compressed-tensors"
        method.use_expert_weight_list = False
        method.ct_packed_output = True
        method._process_scale = lambda t: t.transpose(1, 2).contiguous().view(torch.int32).to(torch.int64)
        method.maybe_squeeze_per_channel_weight_scale = lambda t: t.squeeze(1)
        layer = torch.nn.Module()
        for projection in ("w13", "w2"):
            codes = torch.arange(2 * 64 * 32).reshape(2, 64, 32).remainder(16).sub(8).to(torch.int8)
            source = torch.stack([helper.pack_output_int4(expert) for expert in codes])
            layer.register_parameter(projection + "_weight", torch.nn.Parameter(source, False))
            layer.register_parameter(
                projection + "_weight_scale", torch.nn.Parameter(torch.ones(2, 64, 1, dtype=torch.bfloat16), False)
            )
        method.process_weights_after_loading(layer)
        self.assertEqual([call.args[1] for call in format_cast.call_args_list], [29, 29])
        general_policy.assert_not_called()
        self.assertEqual(layer.w13_weight.shape, (2, 32, 8))
        self.assertEqual(layer.w13_weight.dtype, torch.int32)
        self.assertEqual(layer.w13_weight_scale.shape, (2, 64))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 1, 64))

    def test_actual_postload_method_both_layouts(self):
        torch.manual_seed(47)
        for packed_output in (False, True):
            method = extract_scheme()
            method.ct_packed_output = packed_output
            method._process_scale = lambda t: t.transpose(1, 2).contiguous().view(torch.int32).to(torch.int64)
            method.maybe_squeeze_per_channel_weight_scale = lambda t: t.squeeze(1)
            layer = torch.nn.Module()
            originals = {}
            for name, n, k in (("w13", 32, 16), ("w2", 16, 16)):
                codes = torch.randint(-8, 8, (3, n, k), dtype=torch.int8)
                scale = (torch.rand(3, n, 1) + 0.01).to(torch.bfloat16)
                originals[name] = codes, scale
                source = torch.stack([helper.pack_output_int4(w) for w in codes]) if packed_output else codes
                layer.register_parameter(name + "_weight", torch.nn.Parameter(source, False))
                layer.register_parameter(name + "_weight_scale", torch.nn.Parameter(scale, False))
            method.process_weights_after_loading_compressed_tensors(layer)
            for name, (codes, scale) in originals.items():
                weight = getattr(layer, name + "_weight")
                restored = torch.stack([helper.unpack_output_int4(w.t()) for w in weight])
                self.assertTrue(torch.equal(restored, codes))
                expected = 8 * (codes.transpose(1, 2).contiguous().float() * scale.transpose(1, 2)).sum(1)
                torch.testing.assert_close(getattr(layer, name + "_scale_bias"), expected, rtol=0, atol=0)
                bits = scale.float().transpose(1, 2).contiguous().view(torch.int32).to(torch.int64)
                if name == "w13":
                    bits = bits.squeeze(1)
                self.assertTrue(torch.equal(getattr(layer, name + "_weight_scale"), bits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
