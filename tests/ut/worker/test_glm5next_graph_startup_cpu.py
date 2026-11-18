# SPDX-License-Identifier: Apache-2.0
"""Run directly on CPU: GLM bindings, graph dispatch and startup diagnostics."""

import ast
import importlib.util
import inspect
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWorkerStartup(unittest.TestCase):
    def setUp(self):
        logger = types.ModuleType("vllm.logger")
        logger.logger = logging.getLogger("vllm.graph_startup_test")
        with patch.dict(sys.modules, {"vllm.logger": logger}):
            self.helper = load_source("startup_under_test", ROOT / "vllm_ascend/worker/startup.py")
        self.worker = types.SimpleNamespace(
            rank=2,
            vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(cudagraph_mode="PIECEWISE")),
        )

    def test_phase_logs_rank_and_preserves_rpc_signature_and_return(self):
        def operation(worker, count, *, sentinel=None):
            return count, sentinel

        wrapped = self.helper.log_worker_startup("memory_profile")(operation)
        self.assertEqual(inspect.signature(operation), inspect.signature(wrapped))
        sentinel = object()
        with (
            self.assertLogs(self.helper.logger, level="INFO") as logs,
            patch.object(self.helper.time, "perf_counter", side_effect=[1.0, 2.25]),
        ):
            result = wrapped(self.worker, 4, sentinel=sentinel)
        self.assertEqual(result, (4, sentinel))
        self.assertIn("rank=2", logs.output[0])
        self.assertIn("stage=memory_profile begin graph=PIECEWISE", logs.output[0])
        self.assertIn("end elapsed=1.250s", logs.output[1])

    def test_phase_failure_is_logged_and_original_exception_propagates(self):
        error = RuntimeError("capture failed")

        def operation(worker):
            raise error

        wrapped = self.helper.log_worker_startup("compile_and_capture")(operation)
        with (
            self.assertLogs(self.helper.logger, level="INFO") as logs,
            self.assertRaises(RuntimeError) as caught,
        ):
            wrapped(self.worker)
        self.assertIs(caught.exception, error)
        self.assertIn("stage=compile_and_capture failed", logs.output[-1])
        self.assertNotIn(" end ", "\n".join(logs.output))

    def test_no_device_api_or_config_is_required_for_logging(self):
        wrapped = self.helper.log_worker_startup("kv_cache_spec")(lambda worker: 17)
        with self.assertLogs(self.helper.logger, level="INFO"):
            self.assertEqual(wrapped(object()), 17)


class TestGLMConvolutionBinding(unittest.TestCase):
    def setUp(self):
        native = types.ModuleType("fla_npu.ops.ascendc")
        native.causal_conv1d_fn = Mock(side_effect=lambda x, *args, **kwargs: x + 2)
        native.causal_conv1d_update = Mock(side_effect=lambda x, *args, out, **kwargs: out.copy_(x + 3))
        constants = types.ModuleType("vllm.v1.attention.backends.utils")
        constants.PAD_SLOT_ID = -1
        copies = types.ModuleType("vllm_ascend.ops.triton.kda.conv_state")
        copies.copy_conv_state = Mock(side_effect=AssertionError("Unexpected staging for dim-last state"))
        generic = types.ModuleType("vllm.model_executor.layers.mamba.ops.causal_conv1d")
        generic.causal_conv1d_update = Mock(side_effect=AssertionError("Generic fallback must not be called by GLM"))
        modules = {m.__name__: m for m in (native, constants, copies, generic)}
        self.imports = patch.dict(sys.modules, modules)
        self.imports.start()
        self.addCleanup(self.imports.stop)
        self.native, self.generic = native, generic
        self.conv = load_source("glm_native_conv_under_test", ROOT / "vllm_ascend/models/glm5next/ops/causal_conv1d.py")

    def test_decode_uses_native_ascendc_without_generic_triton_binding(self):
        x, output = torch.ones(2, 16), torch.zeros(2, 16)
        result = self.conv.causal_conv1d(
            x,
            torch.ones(4, 16),
            torch.zeros(4, 3, 16),
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 2]),
            run_mode=1,
            output_buffer=output,
        )
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertTrue(torch.equal(result, torch.full_like(x, 4)))
        self.native.causal_conv1d_update.assert_called_once()
        self.generic.causal_conv1d_update.assert_not_called()

    def test_prefill_uses_native_ascendc(self):
        x = torch.ones(2, 16)
        result = self.conv.causal_conv1d(
            x, torch.ones(4, 16), torch.zeros(4, 3, 16), torch.tensor([0, 1, 2]), torch.tensor([1, 2]), run_mode=0
        )
        self.assertTrue(torch.equal(result, torch.full_like(x, 3)))
        self.native.causal_conv1d_fn.assert_called_once()

    def test_decode_preserves_device_metadata_and_shared_output(self):
        starts, slots, accepted = torch.tensor([0, 1, 2]), torch.tensor([1, 2]), torch.tensor([1, 1])
        output = torch.zeros(4, 16)
        self.conv.causal_conv1d(
            torch.ones(2, 16),
            torch.ones(4, 16),
            torch.zeros(4, 3, 16),
            starts,
            slots,
            run_mode=1,
            num_accepted_tokens=accepted,
            output_buffer=output,
        )
        kwargs = self.native.causal_conv1d_update.call_args.kwargs
        self.assertIs(kwargs["query_start_loc"], starts)
        self.assertIs(kwargs["conv_state_indices"], slots)
        self.assertIs(kwargs["num_accepted_tokens"], accepted)
        self.assertEqual(kwargs["out"].data_ptr(), output.data_ptr())
        self.assertTrue(torch.equal(output[2:], torch.zeros_like(output[2:])))

    def test_legacy_update_does_not_receive_new_keyword(self):
        self.test_decode_uses_native_ascendc_without_generic_triton_binding()
        self.assertFalse(self.conv._UPDATE_HAS_MAX_QUERY_LEN)
        self.assertNotIn("max_query_len", self.native.causal_conv1d_update.call_args.kwargs)

    def test_varlen_update_supplies_shape_bound_without_reading_device_values(self):
        calls = Mock()

        def update(x, state, weight, *, max_query_len=-1, out, **kwargs):
            calls(max_query_len=max_query_len, **kwargs)
            if max_query_len < 0:
                raise ValueError("max_query_len is required for varlen update")
            if state.shape[1] < weight.shape[0] - 1 + max_query_len - 1:
                raise ValueError("per-request speculative state capacity is insufficient")
            return out.copy_(x)

        self.native.causal_conv1d_update = update
        conv = load_source("glm_bound_conv_under_test", ROOT / "vllm_ascend/models/glm5next/ops/causal_conv1d.py")
        self.assertTrue(conv._UPDATE_HAS_MAX_QUERY_LEN)
        # Packed requests with lengths 2, 0, 3 (including padding), as in
        # speculative verification. A whole-batch bound of 5 would exceed
        # each request's rollback capacity; the valid configured bound is 3.
        x = torch.ones(5, 16)
        starts = torch.tensor([0, 2, 2, 5])
        slots = torch.tensor([1, 0, 2])
        accepted = torch.tensor([1, 1, 2])
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read")),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device-to-host copy")),
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list read")),
        ):
            conv.causal_conv1d(
                x,
                torch.ones(4, 16),
                torch.zeros(4, 5, 16),
                starts,
                slots,
                run_mode=1,
                num_accepted_tokens=accepted,
                output_buffer=torch.zeros_like(x),
            )
        self.assertEqual(calls.call_args.kwargs["max_query_len"], 3)
        self.assertIs(calls.call_args.kwargs["query_start_loc"], starts)
        self.assertIs(calls.call_args.kwargs["num_accepted_tokens"], accepted)


class TestGLMGraphDispatch(unittest.TestCase):
    def test_piecewise_and_full_use_their_respective_output_contracts(self):
        path = ROOT / "vllm_ascend/models/glm5next/kda.py"
        owner = next(
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == "Glm5NextLinearAttention"
        )
        forward = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), namespace)
        hidden = torch.ones(2, 16)
        for piecewise in (False, True):
            with self.subTest(piecewise=piecewise):
                hooks = types.SimpleNamespace(
                    glm5next_kda_forward=Mock(side_effect=lambda x, name: x + 5),
                    glm5next_kda_forward_with_output=Mock(side_effect=lambda x, out, name: out.copy_(x + 7)),
                )
                layer = types.SimpleNamespace(_needs_output_buffer=piecewise, prefix="layers.0.self_attn")
                with patch.object(torch.ops, "vllm", hooks):
                    result = namespace["forward"](layer, hidden, None)
                self.assertTrue(torch.equal(result, torch.full_like(hidden, 8 if piecewise else 6)))
                self.assertEqual(hooks.glm5next_kda_forward_with_output.call_count, int(piecewise))
                self.assertEqual(hooks.glm5next_kda_forward.call_count, int(not piecewise))


class TestInstalledFLASourceReport(unittest.TestCase):
    def setUp(self):
        self.probe = load_source("glm_graph_probe_under_test", ROOT / "examples/diagnose_glm5next_graph.py")

    def test_source_only_does_not_import_model_or_device_modules(self):
        with (
            patch.object(self.probe, "_git", return_value={}),
            patch.object(self.probe, "collect_fla_sources", return_value={"fixture": {"version": "26.9.1"}}),
            patch.object(self.probe.importlib, "import_module", side_effect=AssertionError("runtime import")),
        ):
            report = self.probe.collect([], source_only=True)
        self.assertEqual(report["bindings"], {})
        self.assertIsNone(report["vllm_root"])
        self.assertEqual(report["fla_sources"]["fixture"]["version"], "26.9.1")

    def test_source_report_preserves_validation_guard_and_api_signature(self):
        source = (
            "def npu_causal_conv1d_update(x, *, max_query_len=-1, validate_data=False):\n"
            "    if validate_data:\n"
            "        return x.cpu().tolist()\n"
            "    return x\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_aclnn_ctypes.py"
            path.write_text(source, encoding="utf-8")
            report = self.probe._source_report(path)
        function = report["functions"]["npu_causal_conv1d_update"]["source"]
        self.assertIn("max_query_len=-1", function)
        self.assertIn("if validate_data:", function)
        self.assertEqual(report["possible_host_reads"], ["3: return x.cpu().tolist()"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
