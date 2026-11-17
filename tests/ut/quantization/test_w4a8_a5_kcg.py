# SPDX-License-Identifier: Apache-2.0
"""CPU source regression for the A5 CT loader and both native GMM hooks.

Run directly. A CPU backend models the integer contract, not private NZ memory
or CANN execution; those are covered by the separate NPU acceptance test.
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_helpers():
    package = "a5_ct_test_fixture"
    sys.modules[package] = types.ModuleType(package)
    for name in ("compressed_tensors_w4a8", "compressed_tensors_w4a8_a5"):
        path = ROOT / "vllm_ascend/quantization" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package}.compressed_tensors_w4a8"], module


def decode_carrier(weight):
    # Independent unpack of eight two's-complement nibbles per INT32.
    return (
        torch.stack([((weight.long() >> (4 * n)) & 15) for n in range(8)], -1).flatten(-2).sub(8).remainder(16).sub(8)
    )


def decode_nz_carrier(weight):
    e, k, stored_n = weight.shape
    physical = weight.reshape(e, stored_n // 4, k // 16, 16, 4)
    logical = physical.permute(0, 2, 3, 1, 4).reshape(e, k, stored_n)
    return decode_carrier(logical)


class IntegerBackend:
    def __init__(self):
        self.convert_shapes, self.calls, self.raw_copies = [], [], []
        self.allocations = []

    def npu_format_cast(self, tensor, fmt, *, customize_dtype):
        assert tensor.dtype == torch.int32 and tensor.shape[0] == 1
        assert fmt == 29 and customize_dtype == torch.int8
        self.convert_shapes.append(tuple(tensor.shape))
        tensor._fixture_format = 51
        return tensor

    def npu_convert_weight_to_int4pack(self, tensor):
        lanes = tensor.reshape(*tensor.shape[:-1], -1, 8).long() & 15
        result = sum(lanes[..., n] << (n * 4) for n in range(8)).to(torch.int32)
        result._fixture_format = 53
        return result

    @staticmethod
    def get_npu_format(tensor):
        # Model torch_npu's rank-based BASE format inference independently of
        # allow_internal_format: generic 5-D tensors use NCDHW, not ND.
        return getattr(tensor, "_fixture_format", {4: 0, 5: 30}.get(tensor.ndim, 2))

    def empty_with_format(self, **kwargs):
        fmt = kwargs.pop("acl_format")
        assert fmt in (2, 53)
        self.allocations.append((tuple(kwargs["size"]), fmt))
        result = torch.empty(**kwargs)
        result._fixture_format = fmt
        return result

    def copy_memory_(self, dst, src, non_blocking):
        self.raw_copies.append((tuple(dst.shape), tuple(src.shape)))
        assert self.get_npu_format(dst) == 53 and self.get_npu_format(src) == 2
        assert dst.storage_offset() == 0 and dst.numel() == src.numel() and not non_blocking
        dst.reshape(-1).copy_(src.reshape(-1))

    @staticmethod
    def npu_dynamic_quant(x, **kwargs):
        scale = x.float().abs().amax(-1).clamp_min(1e-5) / 127
        return (x.float() / scale[:, None]).round().clamp(-128, 127).to(torch.int8), scale

    def npu_grouped_matmul(self, **kwargs):
        self.calls.append(kwargs)
        assert "bias" not in kwargs and "offset" not in kwargs
        assert kwargs["scale"][0].dtype == torch.float32
        assert kwargs["antiquant_scale"][0].dtype == torch.float16
        assert kwargs["per_token_scale"][0].ndim == 1
        x, codes = kwargs["x"][0], decode_nz_carrier(kwargs["weight"][0]).int()
        groups = kwargs["group_list"].tolist()
        if kwargs["group_list_type"] == 0:
            groups = [groups[0], *[groups[i] - groups[i - 1] for i in range(1, len(groups))]]
        output, start = [], 0
        for expert, count in enumerate(groups):
            accum = x[start : start + count].int() @ codes[expert]
            output.append(
                accum.float() * kwargs["scale"][0][expert] * kwargs["per_token_scale"][0][start : start + count, None]
            )
            start += count
        return [torch.cat(output).to(kwargs["output_dtype"])]


def source_method(helper, backend, *, a5=True, eplb=False, fused=0, quant_name="compressed-tensors"):
    config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={}, get_name=lambda: quant_name),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, enable_eplb=eplb),
        use_v2_model_runner=eplb,
    )
    ascend = SimpleNamespace(enable_fused_mc2=fused, eplb_config=SimpleNamespace(dynamic_eplb=eplb))
    path = ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendW4A8DynamicFusedMoEMethod")
    cls.bases, cls.decorator_list = [], []
    names = {
        "__init__",
        "process_weights_after_loading",
        "process_weights_after_loading_a5_ct",
        "apply_gmm1",
        "apply_gmm2",
        "apply_act_quant",
        "_maybe_convert_group_list",
        "get_fused_mc2_weights",
        "apply_gmm1_act_quant",
    }
    cls.body = [
        n
        for n in cls.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ("supports_eplb", "fused_activations") for t in n.targets)
        )
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = dict(
        torch=torch,
        torch_npu=backend,
        get_current_vllm_config=lambda: config,
        get_ascend_config=lambda: ascend,
        is_950=lambda: a5,
        get_mc2_group=lambda: SimpleNamespace(),
        ASCEND_QUANTIZATION_METHOD="ascend",
        COMPRESSED_TENSORS_METHOD="compressed-tensors",
        _MODELSLIM_SCALE_BIAS_PARTITIONS=16,
        dispose_tensor=lambda x: None,
        log_loading_memory=lambda *args: None,
        prepare_a5_ct_projection=helper.prepare_a5_ct_projection,
        validate_a5_ct_projection=helper.validate_a5_ct_projection,
        apply_a5_ct_gmm=helper.apply_a5_ct_gmm,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(path), "exec"), scope)
    method = scope[cls.name]()
    method._quant_hidden_states = lambda x, s: backend.npu_dynamic_quant(x) if s is None else (x, s)
    return method


class TestA5CTKCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imports = patch.dict(sys.modules)
        imports.start()
        cls.addClassCleanup(imports.stop)
        cls.packing, cls.helper = load_helpers()

    def setUp(self):
        self.backend = IntegerBackend()

    def projection(self, experts=3, n=128, k=512, packed=True):
        generator = torch.Generator().manual_seed(n + k)
        codes = torch.randint(-8, 8, (experts, n, k), dtype=torch.int8, generator=generator)
        scale = ((torch.arange(experts * n).reshape(experts, n, 1) % 8 + 1) / 1024).bfloat16()
        weight = torch.stack([self.packing.pack_output_int4(c) for c in codes]) if packed else codes
        return codes, weight, scale

    def test_native_conversion_per_expert_preserves_all_codes_and_scales(self):
        for packed in (True, False):
            codes, weight, scale = self.projection(packed=packed)
            out, raw, anti = self.helper.prepare_a5_ct_projection(
                weight, scale, packed_output=packed, backend=self.backend
            )
            torch.testing.assert_close(decode_nz_carrier(out), codes.transpose(1, 2).long(), rtol=0, atol=0)
            torch.testing.assert_close(raw, scale[..., 0].float(), rtol=0, atol=0)
            self.assertEqual(out.numel() * out.element_size(), codes.numel() // 2)
            self.assertEqual(tuple(anti.shape), (3, 2, 128))
            self.assertTrue(anti.eq(1).all())
            self.assertEqual(self.backend.convert_shapes[-1:], [(1, 16, 64)])
            self.assertEqual(self.backend.raw_copies[-1], ((3, 512, 16), (3, 4, 32, 16, 4)))

    def test_five_dimensional_stage_explicitly_requests_nd(self):
        codes, weight, scale = self.projection()
        shape = (3, 4, 32, 16, 4)
        self.assertEqual(self.backend.get_npu_format(torch.empty(shape)), 30)
        out, _, _ = self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        self.assertEqual(self.backend.allocations[-1], (shape, 2))
        torch.testing.assert_close(decode_nz_carrier(out), codes.transpose(1, 2).long(), rtol=0, atol=0)

    def test_bad_stage_format_or_view_fails_before_raw_copy(self):
        _, weight, scale = self.projection()
        allocate = self.backend.empty_with_format
        for defect in ("format", "offset", "stride"):

            def bad_stage(*, defect=defect, **kwargs):
                result = allocate(**kwargs)
                if kwargs["acl_format"] != 2:
                    return result
                if defect == "format":
                    result._fixture_format = 30
                elif defect == "offset":
                    result = torch.empty(result.numel() + 1, dtype=result.dtype).as_strided(
                        result.shape, result.stride(), storage_offset=1
                    )
                    result._fixture_format = 2
                else:
                    result = result.transpose(-1, -2)
                    result._fixture_format = 2
                return result

            with (
                self.subTest(defect=defect),
                patch.object(self.backend, "empty_with_format", side_effect=bad_stage),
                self.assertRaisesRegex(RuntimeError, "contiguous zero-offset ND"),
            ):
                self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        self.assertEqual(self.backend.raw_copies, [])
        self.assertEqual(self.backend.calls, [])

    def test_cpu_physical_order_matches_independent_nz_then_pack_reference(self):
        codes, _, _ = self.projection(experts=1)
        packed = self.helper.pack_a5_nz_codes(codes[0])
        # Independent scalar coordinate formula for the K-CG C0=32 NZ.
        n, k = codes[0].shape
        expected = torch.empty((n // 32, k // 16, 16, 4), dtype=torch.int32)
        for n1 in range(n // 32):
            for k1 in range(k // 16):
                for k0 in range(16):
                    for word in range(4):
                        value = sum(
                            (int(codes[0, n1 * 32 + word * 8 + lane, k1 * 16 + k0]) & 15) << (4 * lane)
                            for lane in range(8)
                        )
                        expected[n1, k1, k0, word] = value if value < 2**31 else value - 2**32
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)

    def test_native_converter_values_are_not_used_for_model_codes(self):
        codes, weight, scale = self.projection()
        convert = self.backend.npu_convert_weight_to_int4pack

        def poison_values(tensor):
            result = convert(tensor)
            result.fill_(0x12345678)
            return result

        with patch.object(self.backend, "npu_convert_weight_to_int4pack", side_effect=poison_values):
            out, _, _ = self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        torch.testing.assert_close(decode_nz_carrier(out), codes.transpose(1, 2).long(), rtol=0, atol=0)

    def test_cpu_pack_and_references_ignore_model_default_device_context(self):
        codes, weight, scale = self.projection()
        # vLLM constructs and post-processes models under a default NPU
        # device context. Meta exposes accidental device factories on CPU CI.
        with torch.device("meta"):
            packed = self.helper.pack_a5_nz_codes(codes[0])
            probes, expected = self.helper.a5_ct_probe_reference(codes[0])
            out, _, _ = self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        self.assertEqual(packed.device.type, "cpu")
        self.assertEqual(probes.device.type, "cpu")
        self.assertEqual(expected.device.type, "cpu")
        torch.testing.assert_close(decode_nz_carrier(out), codes.transpose(1, 2).long(), rtol=0, atol=0)

    def test_startup_numeric_guard_rejects_corrupted_physical_copy(self):
        _, weight, scale = self.projection()
        raw_copy = self.backend.copy_memory_

        def corrupt(dst, src, non_blocking):
            raw_copy(dst, src, non_blocking)
            dst.reshape(-1)[0] ^= 1

        with (
            patch.object(self.backend, "copy_memory_", side_effect=corrupt),
            self.assertRaisesRegex(RuntimeError, "numerical self-check failed"),
        ):
            self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)

    def test_a16_bytes_are_wrong_for_a8_kernel_despite_identical_shape_and_size(self):
        # An independent kernel-coordinate example: with C0=32, output n=16
        # reads old C0=16 bytes for codes[n=0,k=1], not codes[n=16,k=0].
        codes = torch.zeros((64, 256), dtype=torch.int8)
        codes[16, 0], codes[0, 1] = 7, -3
        old_lanes = codes.reshape(4, 16, 16, 16).permute(0, 2, 3, 1).contiguous().reshape(4, 16, 16, 2, 8)
        old = sum((old_lanes[..., i].int() & 15) << (4 * i) for i in range(8)).reshape(1, 256, 8)
        correct = self.helper.pack_a5_nz_codes(codes).reshape(1, 256, 8)
        self.assertEqual(old.shape, correct.shape)
        self.assertEqual(old.untyped_storage().nbytes(), correct.untyped_storage().nbytes())
        self.assertEqual(int(decode_nz_carrier(old)[0, 0, 16]), -3)
        self.assertEqual(int(decode_nz_carrier(correct)[0, 0, 16]), 7)
        # Also exercise the actual startup guard with the wrong bytes tagged
        # as the expected format: format-ID validation alone cannot catch it.
        old._fixture_format = 53
        with self.assertRaisesRegex(RuntimeError, "numerical self-check failed"):
            self.helper.verify_a5_ct_projection(
                old,
                torch.ones((1, 1, 64), dtype=torch.float16),
                {0: self.helper.a5_ct_probe_reference(codes)},
                backend=self.backend,
            )

    def test_a16_or_nd_template_is_rejected_before_copy_and_gmm(self):
        _, weight, scale = self.projection()
        convert = self.backend.npu_format_cast
        for wrong_format in (50, 2):

            def wrong_template(*args, wrong_format=wrong_format, **kwargs):
                result = convert(*args, **kwargs)
                result._fixture_format = wrong_format
                return result

            with (
                patch.object(self.backend, "npu_format_cast", side_effect=wrong_template),
                self.assertRaisesRegex(RuntimeError, "NZ_C0_32"),
            ):
                self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        self.assertEqual(self.backend.raw_copies, [])
        self.assertEqual(self.backend.calls, [])

    def test_a16_packed_template_is_rejected(self):
        _, weight, scale = self.projection()
        convert = self.backend.npu_convert_weight_to_int4pack

        def wrong_packed_template(tensor):
            result = convert(tensor)
            result._fixture_format = 52
            return result

        with (
            patch.object(self.backend, "npu_convert_weight_to_int4pack", side_effect=wrong_packed_template),
            self.assertRaisesRegex(RuntimeError, "NZ_C0_4"),
        ):
            self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        self.assertEqual(self.backend.raw_copies, [])

    def test_144_local_experts_and_signed_extreme_codes(self):
        codes, weight, scale = self.projection(experts=144, n=64, k=256)
        out, _, _ = self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        torch.testing.assert_close(decode_nz_carrier(out), codes.transpose(1, 2).long(), rtol=0, atol=0)
        self.assertEqual(self.backend.calls[-1]["group_list"].nonzero().flatten().tolist(), [0, 72, 143])
        extreme = torch.arange(16).repeat(64 * 256 // 16).reshape(64, 256).sub(8).to(torch.int8)
        packed = self.helper.pack_a5_nz_codes(extreme)
        torch.testing.assert_close(decode_nz_carrier(packed.reshape(1, 256, 8))[0], extreme.T.long(), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "signed INT4"):
            self.helper.pack_a5_nz_codes(torch.full((64, 256), 8, dtype=torch.int8))

    def test_gmm_contract_numeric_with_empty_expert_and_both_group_list_types(self):
        codes, weight, scale = self.projection()
        out, raw, anti = self.helper.prepare_a5_ct_projection(weight, scale, packed_output=True, backend=self.backend)
        x = (torch.arange(5 * 512).reshape(5, 512) % 15 - 7).to(torch.int8)
        sx = (torch.arange(5).float() + 1)[:, None] / 16
        expected = torch.cat(
            [
                (x[:2].int() @ codes[0].T.int()).float() * raw[0] * sx[:2],
                (x[2:].int() @ codes[2].T.int()).float() * raw[2] * sx[2:],
            ]
        ).bfloat16()
        for mode, groups in ((1, [2, 0, 3]), (0, [2, 2, 5])):
            result = self.helper.apply_a5_ct_gmm(
                x, sx, out, raw, anti, torch.tensor(groups), mode, backend=self.backend
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_source_postload_and_both_gmm_hooks(self):
        method = source_method(self.helper, self.backend)
        method.ct_packed_output = True
        layer = torch.nn.Module()
        originals = {}
        for proj, n, k in (("w13", 512, 512), ("w2", 512, 256)):
            codes, w, s = self.projection(n=n, k=k)
            originals[proj] = codes, s
            layer.register_parameter(f"{proj}_weight", torch.nn.Parameter(w, requires_grad=False))
            layer.register_parameter(f"{proj}_weight_scale", torch.nn.Parameter(s, requires_grad=False))
            layer.register_parameter(f"{proj}_scale_bias", torch.nn.Parameter(torch.ones(3, n), requires_grad=False))
        method.process_weights_after_loading(layer)
        first_weight = layer.w13_weight
        method.process_weights_after_loading(layer)
        self.assertIs(layer.w13_weight, first_weight)
        self.assertFalse(hasattr(layer, "w13_scale_bias"))
        self.assertNotIn("w13_antiquant_scale", layer.state_dict())
        self.assertEqual(tuple(layer.w13_antiquant_scale.shape), (3, 2, 512))
        self.assertEqual(tuple(layer.w2_antiquant_scale.shape), (3, 1, 512))
        x = (torch.arange(4 * 512).reshape(4, 512) % 9 - 4).to(torch.int8)
        inputs = SimpleNamespace(
            hidden_states=x,
            dynamic_scale=torch.ones(4, 1),
            layer=layer,
            group_list=torch.tensor([2, 2, 4]),
            group_list_type=0,
        )
        result1 = method.apply_gmm1(inputs)
        expected1 = torch.cat(
            [
                (x[:2].int() @ originals["w13"][0][0].T.int()).float() * originals["w13"][1][0, :, 0].float(),
                (x[2:].int() @ originals["w13"][0][2].T.int()).float() * originals["w13"][1][2, :, 0].float(),
            ]
        ).bfloat16()
        torch.testing.assert_close(result1, expected1, rtol=0, atol=0)
        gate, up = result1.chunk(2, -1)
        activated = torch.nn.functional.silu(gate) * up
        qa, sa = method.apply_act_quant(inputs, activated)
        result2 = method.apply_gmm2(inputs, qa, sa)
        expected2 = torch.cat(
            [
                (qa[:2].int() @ originals["w2"][0][0].T.int()).float()
                * originals["w2"][1][0, :, 0].float()
                * sa[:2, None],
                (qa[2:].int() @ originals["w2"][0][2].T.int()).float()
                * originals["w2"][1][2, :, 0].float()
                * sa[2:, None],
            ]
        ).bfloat16()
        torch.testing.assert_close(result2, expected2, rtol=0, atol=0)
        self.assertEqual(len(self.backend.calls), 4)  # Two startup probes, then GMM1/GMM2.

    def test_platform_selection_and_unsupported_fusions(self):
        legacy_ct = source_method(self.helper, self.backend, a5=False)
        self.assertEqual(legacy_ct.fused_activations, frozenset({"situ"}))
        modelslim = source_method(self.helper, self.backend, a5=False, quant_name="ascend")
        self.assertEqual(modelslim.fused_activations, frozenset({"silu", "situ"}))
        self.assertTrue(source_method(self.helper, self.backend).use_a5_ct_kcg)
        self.assertFalse(source_method(self.helper, self.backend, a5=False).use_a5_ct_kcg)
        self.assertFalse(source_method(self.helper, self.backend, quant_name="ascend").use_a5_ct_kcg)
        with self.assertRaisesRegex(NotImplementedError, "EPLB"):
            source_method(self.helper, self.backend, eplb=True)
        with self.assertRaisesRegex(NotImplementedError, "enable_fused_mc2"):
            source_method(self.helper, self.backend, fused=1)
        method = source_method(self.helper, self.backend)
        self.assertFalse(method.supports_eplb)
        self.assertEqual(method.fused_activations, frozenset())
        with self.assertRaises(NotImplementedError):
            method.get_fused_mc2_weights(None)

    def test_invalid_local_k_and_encoded_scale_rejected(self):
        _, w, s = self.projection(k=128)
        with self.assertRaisesRegex(ValueError, "K%256"):
            self.helper.prepare_a5_ct_projection(w, s, packed_output=True, backend=self.backend)
        _, w, s = self.projection()
        with self.assertRaisesRegex(ValueError, "raw floating"):
            self.helper.prepare_a5_ct_projection(w, s.to(torch.int64), packed_output=True, backend=self.backend)
        self.assertEqual(self.backend.convert_shapes, [])

    def test_full_n_guard_and_zero_token_input(self):
        _, w, s = self.projection()
        weight, scale, anti = self.helper.prepare_a5_ct_projection(w, s, packed_output=True, backend=self.backend)
        x, sx, groups = torch.ones((2, 512), dtype=torch.int8), torch.ones(2), torch.tensor([2, 0, 0])
        with self.assertRaisesRegex(ValueError, "full_N"):
            self.helper.apply_a5_ct_gmm(x, sx, weight, scale[:, ::2], anti, groups, 1, backend=self.backend)
        with (
            patch.object(self.backend, "npu_grouped_matmul", return_value=[torch.zeros(2, 64)]),
            self.assertRaisesRegex(RuntimeError, "full-width"),
        ):
            self.helper.apply_a5_ct_gmm(x, sx, weight, scale, anti, groups, 1, backend=self.backend)
        result = self.helper.apply_a5_ct_gmm(x[:0], sx[:0], weight, scale, anti, groups * 0, 1, backend=self.backend)
        self.assertEqual(tuple(result.shape), (0, 128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
