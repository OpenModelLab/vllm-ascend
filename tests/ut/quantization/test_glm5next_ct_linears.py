# SPDX-License-Identifier: Apache-2.0
"""CPU regression for GLM CT construction, scale loading and integer execution.

Execute directly with --vllm-root PATH. Runtime imports/distributed state are
isolated; constructors/loaders/quant methods are selected from actual sources.
KDA construction stops after its fused projection (before NPU/FLA setup).
The integer backend below is a CPU contract oracle, not a CANN kernel test.
"""

import argparse
import ast
import dataclasses
import importlib.util
import math
import re
import sys
import types
import unittest
from fnmatch import fnmatch
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = ROOT.parent / "vllm"


def execute_nodes(path, scope, *, classes=None, functions=()):
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            spec = classes[node.name]
            node.decorator_list = []
            node.bases = [ast.Name(id=spec[0], ctx=ast.Load())] if spec[0] else []
            if spec[1] is not None:
                node.body = [
                    m
                    for m in node.body
                    if (isinstance(m, ast.FunctionDef) and m.name in spec[1])
                    or (
                        isinstance(m, ast.AnnAssign)
                        and isinstance(m.target, ast.Name)
                        and m.target.id == "act_quant_type"
                    )
                ]
            nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), scope)


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result
    return result


def runtime(vllm_root):
    state = SimpleNamespace(rank=0, world=1)
    logger = SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None, debug=lambda *a: None)
    scope = dict(
        __name__="glm_ct_fixture.configs",
        __package__="glm_ct_fixture.configs",
        torch=torch,
        nn=torch.nn,
        Parameter=torch.nn.Parameter,
        Module=torch.nn.Module,
        Iterable=object,
        Mapping=object,
        MappingProxyType=MappingProxyType,
        re=re,
        fnmatch=fnmatch,
        logger=logger,
        cast=lambda _t, value: value,
        divide=lambda n, d: n // d,
        DEFAULT_VOCAB_PADDING_SIZE=64,
        get_tensor_model_parallel_rank=lambda: state.rank,
        get_tensor_model_parallel_world_size=lambda: state.world,
        COMPRESSED_TENSORS_METHOD="compressed-tensors",
        set_weight_attrs=lambda p, attrs: [setattr(p, k, v) for k, v in attrs.items()],
        WEIGHT_LOADER_V2_SUPPORTED=[],
        extract_layer_index=lambda p: 0,
        BasevLLMParameter=type("BasevLLMParameter", (), {}),
        BlockQuantScaleParameter=type("BlockQuantScaleParameter", (), {}),
        is_fused_moe_layer=lambda layer: False,
        mlp_tp_enable=lambda: False,
        oproj_tp_enable=lambda: False,
        resolve_quant_method=lambda config, layer, prefix: config.get_quant_method(layer, prefix),
        get_pcp_group=lambda: SimpleNamespace(rank_in_group=0, world_size=1),
        use_glm_recoverssm=lambda config: False,
        get_ascend_config=lambda: SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(qkv_iproj_tensor_parallel_size=1)
        ),
    )
    for name in ("PerTensorScaleParameter", "PackedColumnParameter", "PackedvLLMParameter", "RowvLLMParameter"):
        scope[name] = type(name, (), {})
    module("vllm.model_executor.layers.quantization.online.fp8", OnlineLinearBase=type("OnlineLinearBase", (), {}))
    module("glm_ct_fixture")
    module("glm_ct_fixture.configs")
    native = vllm_root / "vllm/model_executor"
    execute_nodes(
        native / "layers/quantization/utils/config_utils.py",
        scope,
        functions=("find_matching_patterns", "is_equal_or_regex_match"),
    )
    execute_nodes(
        native / "layers/quantization/compressed_tensors/utils.py",
        scope,
        functions=("find_matched_target", "should_ignore_layer", "_find_first_match", "_match_fused_layer"),
    )

    class PlainLinear:
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size=None,
            output_size=None,
            params_dtype=torch.bfloat16,
            **kwargs,
        ):
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.empty(
                        sum(output_partition_sizes),
                        input_size_per_partition,
                        dtype=params_dtype,
                    ),
                    False,
                ),
            )
            scope["set_weight_attrs"](
                layer.weight, dict(output_dim=0, input_dim=1, weight_loader=kwargs["weight_loader"])
            )

        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    scope.update(
        UnquantizedLinearMethod=PlainLinear,
        AscendUnquantizedLinearMethod=PlainLinear,
        UnquantizedEmbeddingMethod=PlainLinear,
        RowParallelLinear=type("RowParallelLinear", (), {}),
    )
    execute_nodes(
        native / "layers/linear.py",
        scope,
        classes={
            "LinearBase": ("Module", {"__init__", "update_param_tp_status"}),
            "ReplicatedLinear": ("LinearBase", {"__init__", "weight_loader", "forward"}),
            "QKVParallelLinear": ("ColumnParallelLinear", {"__init__", "validate_shard_id", "weight_loader"}),
            "RowParallelLinear": ("LinearBase", {"__init__", "weight_loader", "weight_loader_v2"}),
            "ColumnParallelLinear": (
                "LinearBase",
                {"__init__", "_maybe_allow_fp8_block_shape_mismatch", "weight_loader", "forward"},
            ),
            "MergedColumnParallelLinear": (
                "ColumnParallelLinear",
                {"__init__", "validate_shard_id", "weight_loader", "weight_loader_v2"},
            ),
        },
    )
    execute_nodes(
        ROOT / "vllm_ascend/quantization/methods/base.py",
        scope,
        classes={"AscendLinearScheme": (None, {"validate_weight_shapes", "get_pertensor_param", "get_pergroup_param"})},
    )
    scope.update(
        PreparedLinearInput=type("PreparedLinearInput", (), {}),
        maybe_trans_nz=lambda t: t,
        WeightSwitchGatherSpec=lambda *a, **k: None,
    )
    execute_nodes(
        ROOT / "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
        scope,
        classes={
            "AscendW8A8DynamicLinearMethod": (
                "AscendLinearScheme",
                {"__init__", "get_weight", "get_perchannel_param", "process_weights_after_loading", "apply"},
            )
        },
    )
    execute_nodes(
        ROOT / "vllm_ascend/quantization/method_adapters.py",
        scope,
        classes={
            "AscendLinearMethod": (None, {"__init__", "create_weights", "apply", "process_weights_after_loading"})
        },
    )
    module(
        "glm_ct_fixture.method_adapters", AscendLinearMethod=scope["AscendLinearMethod"], AscendFusedMoEMethod=object
    )
    module(
        "glm_ct_fixture.methods",
        get_scheme_class=lambda quant, kind: scope["AscendW8A8DynamicLinearMethod"]
        if quant == "W8A8_DYNAMIC" and kind == "linear"
        else None,
    )
    fmt = lambda value: SimpleNamespace(value=value)
    scope.update(
        CompressionFormat=SimpleNamespace(
            pack_quantized=fmt("pack-quantized"),
            mxfp8_quantized=fmt("mxfp8-quantized"),
            mxfp4_pack_quantized=fmt("mxfp4-pack-quantized"),
        ),
        QuantizationType=SimpleNamespace(INT="int", FLOAT="float"),
        QuantizationStrategy=SimpleNamespace(**{s.upper(): fmt(s) for s in ("channel", "token", "tensor", "group")}),
        is_activation_quantization_format=lambda f: f == "int-quantized",
    )
    execute_nodes(
        native / "layers/vocab_parallel_embedding.py",
        scope,
        functions=("pad_vocab_size", "vocab_range_from_per_partition_vocab_size", "vocab_range_from_global_vocab_size"),
        classes={
            "VocabParallelEmbeddingShardIndices": (None, None),
            "VocabParallelEmbedding": ("Module", {"_get_indices", "weight_loader", "update_param_tp_status"}),
        },
    )
    scope["VocabParallelEmbeddingShardIndices"] = dataclasses.dataclass(scope["VocabParallelEmbeddingShardIndices"])
    scope["ParallelLMHead"] = type("ParallelLMHead", (scope["VocabParallelEmbedding"],), {})
    execute_nodes(
        ROOT / "vllm_ascend/quantization/configs/compressed_tensors_config.py",
        scope,
        classes={
            "AscendCompressedTensorsConfig": (
                None,
                {
                    "__init__",
                    "get_name",
                    "get_quant_method",
                    "get_scheme_dict",
                    "_get_quant_args",
                    "_get_linear_scheme",
                    "_create_scheme_for_layer_type",
                    "_detect_quant_type",
                    "_is_dynamic_token_w8a8",
                    "_is_static_tensor_w8a8",
                    "_is_dynamic_token_w4a8",
                    "_is_dynamic_group_mx",
                },
            )
        },
    )
    helper_path = ROOT / "vllm_ascend/quantization/glm5next.py"
    spec = importlib.util.spec_from_file_location("glm_ct_contract", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scope.update(is_glm_ct_int8_config=helper.is_glm_ct_int8_config)
    scope["get_replicated_op"] = lambda *args: (None, state.rank, state.world)
    execute_nodes(
        ROOT / "vllm_ascend/ops/linear.py",
        scope,
        classes={
            "AscendLinearBase": ("LinearBase", {"__init__"}),
            "AscendReplicatedLinear": ("ReplicatedLinear", {"__init__", "forward"}),
        },
    )
    scope["GateLinear"] = scope["ReplicatedLinear"]
    execute_nodes(
        ROOT / "vllm_ascend/ops/fused_moe/gate_linear.py",
        scope,
        classes={"AscendGateLinear": ("GateLinear", {"__init__", "forward"})},
    )
    scope["GateLinear"] = scope["AscendGateLinear"]
    scope.update(
        get_ep_group=lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0),
        _get_moe_router_dtype=lambda cfg: torch.float32,
    )
    execute_nodes(native / "model_loader/weight_utils.py", scope, functions=("default_weight_loader",))
    # Run the real MoE constructor through router/correction-bias creation.
    # Expert/distributed initialization is outside this projection regression.
    path = ROOT / "vllm_ascend/models/glm5next/model.py"
    execute_nodes(path, scope, functions=("_is_moe", "_is_kda_layer", "_is_linear_attn"))
    owner = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextMoE"
    )
    ctor = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    end = next(i for i, n in enumerate(ctor.body) if isinstance(n, ast.If) and "topk_method" in ast.unparse(n.test))
    ctor.body = ctor.body[: end + 1]
    owner.body, owner.decorator_list = [ctor], []
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), owner],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    execute_nodes(
        ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py",
        scope,
        classes={"AscendMoERunner": (None, {"_apply_router_linear"})},
    )
    scope.update(SiluAndMul=torch.nn.Identity, SiluAndMulWithClamp=lambda **kw: torch.nn.Identity())
    execute_nodes(path, scope, classes={"Glm5NextMLP": ("Module", {"__init__"})})
    execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/kda.py",
        scope,
        classes={
            "_Glm5NextMergedColumnParallelLinear": (
                "MergedColumnParallelLinear",
                {"__init__", "weight_loader", "weight_loader_v2"},
            )
        },
    )
    execute_nodes(
        native / "layers/mamba/gdn/base.py", scope, classes={"GatedDeltaNetAttention": ("Module", {"__init__"})}
    )
    path = ROOT / "vllm_ascend/models/glm5next/kda.py"
    node = next(
        c
        for c in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(c, ast.ClassDef) and c.name == "Glm5NextLinearAttention"
    )
    ctor = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__")
    end = next(
        i
        for i, statement in enumerate(ctor.body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "in_proj_qkvbfg_a" for t in statement.targets)
    )
    ctor.body = ctor.body[: end + 1]
    node.body, node.decorator_list = [ctor], []
    node.bases = [ast.Name(id="GatedDeltaNetAttention", ctx=ast.Load())]
    scope.update(KDA_MAX_RECURRENT_TOKENS=8)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    mode = SimpleNamespace(STANDARD=1)
    scope.update(
        VocabParallelMode=mode,
        _resolve_vocab_parallel_plan=lambda **k: SimpleNamespace(
            mode=1,
            shard_rank=state.rank,
            shard_world_size=state.world,
            token_exchange_group=None,
            output_reduce_group=None,
        ),
        method_has_implemented_embedding=lambda cls: False,
    )
    execute_nodes(
        ROOT / "vllm_ascend/ops/vocab_parallel_embedding.py",
        scope,
        classes={
            "AscendVocabParallelEmbedding": ("VocabParallelEmbedding", {"__init__"}),
            "AscendParallelLMHead": ("ParallelLMHead", {"__init__"}),
        },
    )
    # Exercise GLM's actual hand-written mapping, not a test reimplementation.
    execute_nodes(
        ROOT / "vllm_ascend/quantization/compressed_tensors_packed.py", scope, functions=("prepare_glm_packed_weights",)
    )
    module(
        "vllm_ascend.quantization.compressed_tensors_packed",
        prepare_glm_packed_weights=scope["prepare_glm_packed_weights"],
    )
    scope.update(
        get_spec_layer_idx_from_weight_name=lambda c, n: None,
        _try_load_fp8_indexer_wk=lambda *a: False,
        is_pp_missing_parameter=lambda *a: False,
        maybe_remap_kv_scale_name=lambda n, p: n,
    )
    execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/model.py", scope, classes={"Glm5NextModel": ("Module", {"load_weights"})}
    )

    class IntegerBackend:
        @staticmethod
        def npu_dynamic_quant(x, dst_type):
            scale = x.float().abs().amax(-1).clamp_min(1e-12) / 127
            return (x.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8), scale

        @staticmethod
        def npu_quant_matmul(x, weight, scale, pertoken_scale, bias, output_dtype):
            assert x.dtype == weight.dtype == torch.int8
            acc = x.to(torch.int32) @ weight.to(torch.int32)
            result = acc.float() * scale.float() * pertoken_scale.unsqueeze(-1)
            return (result if bias is None else result + bias).to(output_dtype)

    scope["torch_npu"] = IntegerBackend
    scope.update(
        ct_kpool_input=helper.ct_kpool_input,
        ct_kpool_selectors=helper.ct_kpool_selectors,
        F=torch.nn.functional,
        Glm5NextIndexerCache=lambda **kw: torch.nn.Module(),
        Glm5NextTailCache=lambda **kw: torch.nn.Module(),
        LayerNorm=torch.nn.LayerNorm,
        get_kpool_tail_ring_capacity=lambda *args: 4,
    )
    execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/attention.py", scope, classes={"Indexer": ("Module", {"__init__"})}
    )
    execute_nodes(
        ROOT / "vllm_ascend/attention/indexer_kpool.py",
        scope,
        classes={"Glm5NextKPoolIndexerBackend": ("Module", {"process_weights_after_loading", "_forward_cann"})},
    )
    scope.update(
        is_vit_use_data_parallel=lambda: False,
        get_current_hardware_profile=lambda: SimpleNamespace(requires_2d_nz_linear_input=False),
        parallel_state=SimpleNamespace(get_tensor_model_parallel_rank=lambda: state.rank),
        dist_utils=SimpleNamespace(divide=lambda n, d: n // d),
        RMSNorm=lambda *a, **kw: torch.nn.Identity(),
        MMEncoderAttention=lambda **kw: torch.nn.Identity(),
        ApplyRotaryEmb=lambda **kw: torch.nn.Identity(),
    )
    execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/multimodal.py",
        scope,
        functions=("_requires_2d_nz_input",),
        classes={"Glm5NextVisionAttention": ("Module", {"__init__"})},
    )
    execute_nodes(
        native / "models/deepseek_v2.py",
        scope,
        classes={"DeepSeekV2FusedQkvAProjLinear": ("MergedColumnParallelLinear", {"__init__"})},
    )
    # Construct MLA through its projections, before wrapper/cache registration.
    path = ROOT / "vllm_ascend/models/glm5next/attention.py"
    owner = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextMLAAttention"
    )
    ctor = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    end = next(
        i
        for i, n in enumerate(ctor.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "o_proj" for t in n.targets)
    )
    ctor.body, owner.body = ctor.body[: end + 1], [ctor]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), owner],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    scope.update(
        restore_ct_kv_b=helper.restore_ct_kv_b,
        ACL_FORMAT_FRACTAL_ND=2,
        replace_parameter=lambda obj, name, value, **kw: setattr(obj, name, value),
        dispose_layer=lambda layer: setattr(layer, "disposed", True),
        PreprocessType=SimpleNamespace(NATIVE=1, PROLOG_V3=2),
        vllm_version_is=lambda version: version == "0.32.0",
    )
    IntegerBackend.npu_format_cast = lambda value, fmt: value
    execute_nodes(
        ROOT / "vllm_ascend/attention/sfa_v1.py",
        scope,
        classes={"AscendSFAImpl": (None, {"process_weights_after_loading"})},
    )

    # Run the complete decoder constructor, isolating unrelated attention/MLP
    # backends. mHC allocation and hc_pre below come from the production file.
    class DecoderBackend(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    decoder_scope = dict(scope)
    decoder_scope.update(
        Glm5NextLinearAttention=DecoderBackend,
        Glm5NextMLAAttention=DecoderBackend,
        Glm5NextMLP=DecoderBackend,
        Glm5NextMoE=DecoderBackend,
        _MHC_TAU=0.05,
        _MHC_POST_MULT_VALUE=2.0,
    )
    execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/model.py",
        decoder_scope,
        classes={"Glm5NextDecoderLayer": ("Module", {"__init__", "hc_pre"})},
    )
    scope["Glm5NextDecoderLayer"] = decoder_scope["Glm5NextDecoderLayer"]
    return SimpleNamespace(scope=scope, state=state, helper=helper)


def config(rt, ignore=()):
    def scheme(bits, format):
        return dict(
            weights=SimpleNamespace(type="int", num_bits=bits, strategy="channel", symmetric=True, dynamic=False),
            input_activations=SimpleNamespace(type="int", num_bits=8, strategy="token", symmetric=True, dynamic=True),
            format=format,
        )

    result = rt.scope["AscendCompressedTensorsConfig"](
        {
            r"re:.*\.experts\.\d+\.(gate|up|down)_proj$": scheme(4, "pack-quantized"),
            r"re:^(?!.*\.experts\.\d+\.).*$": scheme(8, "int-quantized"),
        },
        list(ignore),
        "mixed-precision",
        {},
    )
    result.packed_modules_mapping = rt.helper.glm5next_packed_mapping()
    return result


class TestGlmCTLinears(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.import_patch = patch.dict(sys.modules)
        cls.import_patch.start()
        cls.addClassCleanup(cls.import_patch.stop)
        cls.rt = runtime(VLLM_ROOT)

    def setUp(self):
        self.rt.state.rank, self.rt.state.world = 0, 1

    def test_kv_b_native_tp_load_and_bf16_absorption_with_original_scales(self):
        rt = self.rt
        codes = (torch.arange(32 * 8).reshape(32, 8) % 255 - 127).to(torch.int8)
        scales = ((torch.arange(32) + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        for world in (1, 2, 4):
            rt.state.world = world
            for rank in range(world):
                rt.state.rank = rank
                attn = rt.scope["Glm5NextMLAAttention"](
                    SimpleNamespace(),
                    SimpleNamespace(rms_norm_eps=1e-5),
                    16,
                    4,
                    4,
                    0,
                    4,
                    16,
                    8,
                    quant_config=config(rt),
                    prefix="layers.11.self_attn",
                )
                layer = attn.kv_b_proj
                self.assertTrue(layer.ct_restore_bf16_for_absorption)
                model = rt.scope["Glm5NextModel"]()
                model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
                model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
                model.layers[11].self_attn = attn
                model.load_weights(
                    [
                        ("layers.11.self_attn.kv_b_proj.weight", codes),
                        ("layers.11.self_attn.kv_b_proj.weight_scale", scales),
                    ]
                )
                local = slice(rank * 32 // world, (rank + 1) * 32 // world)
                torch.testing.assert_close(layer.weight, codes[local])
                layer.quant_method.process_weights_after_loading(layer)
                sfa = rt.scope["AscendSFAImpl"]()
                sfa.kv_b_proj, sfa.kv_lora_rank, sfa.local_num_heads = layer, 8, 4 // world
                sfa.qk_nope_head_dim, sfa.v_head_dim = 4, 4
                sfa.rl_weight_update_enabled, sfa.has_indexer, sfa.enable_sparse_sfa_c8 = False, False, False
                sfa._parallel_o_proj = None
                sfa._resolve_preprocess_type = lambda dtype: rt.scope["PreprocessType"].NATIVE
                sfa.process_weights_after_loading(torch.bfloat16)
                restored = (codes[local].float() * scales[local].float()).to(torch.bfloat16).reshape(4 // world, 8, 8)
                torch.testing.assert_close(sfa.W_UK_T, restored[:, :4, :], rtol=0, atol=0)
                torch.testing.assert_close(sfa.W_UV, restored[:, 4:, :].transpose(1, 2).contiguous(), rtol=0, atol=0)
                self.assertTrue(layer.disposed)
                self.assertEqual(sfa.W_UV.dtype, torch.bfloat16)
                torch.testing.assert_close(layer.weight, codes[local].T.contiguous(), rtol=0, atol=0)

    def test_kv_b_restore_blocks_and_rejects_invalid_quantization_metadata(self):
        restore = self.rt.helper.restore_ct_kv_b
        codes = torch.arange(-30, 30, dtype=torch.int8).reshape(5, 12)
        scales = torch.linspace(0.01, 0.5, 12)
        expected = (codes.float() * scales).to(torch.bfloat16)
        for block in (1, 3, 256):
            torch.testing.assert_close(
                restore(codes, scales, torch.bfloat16, column_block_size=block), expected, rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "one floating-point scale"):
            restore(codes, scales[:6], torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "INT8"):
            restore(codes.float(), scales, torch.bfloat16)

    def test_indexer_ct_constructor_and_existing_checkpoint_keys(self):
        rt = self.rt
        cfg = SimpleNamespace(index_topk=4, index_n_heads=4, index_head_dim=4, qk_rope_head_dim=0, index_kpool=4)
        indexer = rt.scope["Indexer"](
            SimpleNamespace(attention_config=SimpleNamespace(indexer_kv_dtype="auto")),
            cfg,
            16,
            16,
            config(rt),
            object(),
            torch.empty(4, 7),
            "layers.11.self_attn.indexer",
        )
        model = rt.scope["Glm5NextModel"]()
        model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
        model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
        model.layers[11].self_attn = torch.nn.Module()
        model.layers[11].self_attn.indexer = indexer
        checkpoint = []
        for sid, (name, n) in enumerate(
            (("index_kpool_compress_gate", 4), ("wk", 4), ("weights_proj", 4), ("wq_b", 16))
        ):
            codes = ((torch.arange(n * 16).reshape(n, 16) + sid) % 255 - 127).to(torch.int8)
            scales = ((torch.arange(n) + sid + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
            stem = "layers.11.self_attn.indexer." + name
            checkpoint.extend([(stem + ".weight", codes), (stem + ".weight_scale", scales)])
        loaded = model.load_weights(checkpoint)
        self.assertIn("layers.11.self_attn.indexer.index_kpool_compress_gate.weight", loaded)
        self.assertIn("layers.11.self_attn.indexer.wk_weights_proj.weight_scale", loaded)
        for projection in (indexer.index_kpool_compress_gate, indexer.wk_weights_proj, indexer.wq_b):
            self.assertEqual(projection.weight.dtype, torch.int8)
            projection.quant_method.process_weights_after_loading(projection)
        backend = rt.scope["Glm5NextKPoolIndexerBackend"]()
        backend.head_dim = 4
        backend.wk_weights_proj = indexer.wk_weights_proj
        backend.index_kpool_compress_gate = indexer.index_kpool_compress_gate
        backend.k_norm = torch.nn.LayerNorm(4)
        backend.process_weights_after_loading()
        x = torch.arange(3 * 16).reshape(3, 16).to(torch.bfloat16)
        projected = rt.helper.ct_kpool_input(x, backend.wk_weights_proj, backend.index_kpool_compress_gate, 4)
        self.assertEqual(projected.shape, x.shape)
        torch.testing.assert_close(projected[:, :4], backend.wk_weights_proj(x)[0][:, :4])
        torch.testing.assert_close(projected[:, 4:8], backend.index_kpool_compress_gate(x)[0])
        torch.testing.assert_close(projected.float() @ backend._key_weight.float().T, projected[:, :4].float())
        torch.testing.assert_close(projected.float() @ backend._gate_weight.float().T, projected[:, 4:8].float())
        self.assertEqual(projected[:, 8:].count_nonzero(), 0)
        before = projected[:, 4:8].clone()
        indexer.index_kpool_compress_gate.weight_scale.data.mul_(2)
        after = rt.helper.ct_kpool_input(x, backend.wk_weights_proj, backend.index_kpool_compress_gate, 4)
        torch.testing.assert_close(after[:, 4:8], before * 2)
        # Exercise the actual CANN orchestration, including retained MTP tails.
        backend.topk_tokens, backend.index_kpool, backend.topk_indices_buffer = 4, 4, None
        backend.index_kpool_compress_ape = torch.zeros(4, 4)
        metadata = SimpleNamespace(
            write_metadata=None,
            query_metadata=None,
            positions=torch.arange(3),
            cum_query_lens=torch.tensor([3]),
            query_start_loc=torch.tensor([0, 3]),
            start_pos=torch.tensor([0]),
            pool_tail=torch.tensor([3]),
            pooled_slot_mapping=torch.tensor([0]),
            keypool_state_table=torch.tensor([[1, 2]]),
            keypool_start_offsets=torch.tensor([0]),
        )
        tail_metadata = SimpleNamespace(
            block_size=4, cann_inputs=SimpleNamespace(retained_indices=torch.tensor([0, 2]))
        )
        recorded = {}
        backend._prepare_keypool_state = lambda *args: torch.zeros(3, 4, 8)

        def pool(hidden, key, gate, *args, **kw):
            recorded["pool_input"] = hidden
            return torch.zeros(1, 4, dtype=torch.bfloat16)

        backend._key_pool = pool
        backend._gather_retained_hidden = lambda hidden, indices: hidden[indices]

        def write(*args, **kwargs):
            recorded["tail_key"], recorded["tail_gate"] = args[2:4]

        backend._write_keypool_caches = write
        result = backend._forward_cann(
            x,
            None,
            None,
            torch.empty(1, 1, 1, 4, dtype=torch.bfloat16),
            torch.empty(1, 4, 8),
            metadata,
            tail_metadata,
            compute_topk=False,
        )
        self.assertIsNone(result)
        torch.testing.assert_close(recorded["pool_input"], after)
        torch.testing.assert_close(recorded["tail_gate"], after[[0, 2], 4:8].float())
        torch.testing.assert_close(
            recorded["tail_key"],
            torch.nn.functional.layer_norm(
                after[[0, 2], :4].float(), (4,), backend.k_norm.weight, backend.k_norm.bias, 1e-5
            ),
        )

    def test_vision_qkv_prefix_and_tp_scale_loading(self):
        rt = self.rt
        for world in (1, 2, 4):
            rt.state.world = world
            for rank in range(world):
                rt.state.rank = rank
                vision = rt.scope["Glm5NextVisionAttention"](
                    16, 4, 16, quant_config=config(rt), prefix="visual.blocks.0.attn"
                )
                self.assertEqual(vision.qkv.prefix, "visual.blocks.0.attn.qkv")
                codes = (torch.arange(48 * 16).reshape(48, 16) % 255 - 127).to(torch.int8)
                scales = ((torch.arange(48) + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
                vision.qkv.weight.weight_loader(vision.qkv.weight, codes)
                vision.qkv.weight_scale.weight_loader(vision.qkv.weight_scale, scales)
                indices = torch.cat(
                    [torch.arange(i * 16 + rank * 16 // world, i * 16 + (rank + 1) * 16 // world) for i in range(3)]
                )
                torch.testing.assert_close(vision.qkv.weight, codes[indices])
                torch.testing.assert_close(vision.qkv.weight_scale, scales[indices].to(vision.qkv.weight_scale.dtype))
                self.assertEqual(vision.proj.weight.dtype, torch.int8)
        plain = rt.scope["Glm5NextVisionAttention"](16, 4, 16, quant_config=None, prefix="visual.blocks.0.attn")
        self.assertEqual(plain.qkv.prefix, "visual.blocks.0.attn.qkv")
        self.assertNotIn("weight_scale", dict(plain.qkv.named_parameters()))

    def test_mla_q_a_and_kv_a_scales_load_into_native_replicated_fusion(self):
        rt = self.rt
        rt.state.world = 4
        codes = (torch.arange(24 * 16).reshape(24, 16) % 255 - 127).to(torch.int8)
        scales = ((torch.arange(24) + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        for rank in range(4):
            rt.state.rank = rank
            fused = rt.scope["DeepSeekV2FusedQkvAProjLinear"](
                16, [16, 8], quant_config=config(rt), prefix="layers.11.self_attn.fused_qkv_a_proj"
            )
            model = rt.scope["Glm5NextModel"]()
            model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
            model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
            model.layers[11].self_attn = torch.nn.Module()
            model.layers[11].self_attn.fused_qkv_a_proj = fused
            checkpoint = []
            for name, part in [("q_a_proj", slice(0, 16)), ("kv_a_proj_with_mqa", slice(16, 24))]:
                stem = "layers.11.self_attn." + name
                checkpoint.extend([(stem + ".weight", codes[part]), (stem + ".weight_scale", scales[part])])
            model.load_weights(checkpoint)
            self.assertEqual(fused.tp_size, 1)
            torch.testing.assert_close(fused.weight, codes)
            torch.testing.assert_close(fused.weight_scale, scales.to(fused.weight_scale.dtype))

    def mhc_layer(self, quant, layer_idx=0, *, is_mtp_layer=False):
        cfg = SimpleNamespace(
            hidden_size=16,
            is_moe=False,
            layer_types=["linear_attention" if i % 4 != 3 else "full_attention" for i in range(45)],
            num_hidden_layers=45,
            rms_norm_eps=1e-5,
            n_routed_experts=288,
            mhc=True,
            is_kda_layer=lambda i: i % 4 != 3,
            v_head_dim=256,
            kv_lora_rank=512,
            num_attention_heads=64,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            q_lora_rank=1536,
            max_position_embeddings=8192,
            mla_nope=True,
            mla_use_nope=True,
            mlp_layer_types=["dense"] * 45,
            intermediate_size=32,
            hidden_act="silu",
            swiglu_limit=10.0,
            mhc_num_residual_streams=4,
            hc_mult=4,
            mhc_tau=1.0,
            hc_eps=1e-6,
            mhc_sinkhorn_iterations=3,
            hc_sinkhorn_iters=3,
            mhc_post_mult_value=2.0,
        )
        vcfg = SimpleNamespace(
            cache_config=None,
            quant_config=quant,
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        )
        return self.rt.scope["Glm5NextDecoderLayer"](
            vcfg, cfg, layer_idx, prefix=f"layers.{layer_idx}", is_mtp_layer=is_mtp_layer
        )

    def test_mhc_bare_bf16_checkpoint_loads_all_270_keys_under_broad_ct_recipe(self):
        rt = self.rt
        model = rt.scope["Glm5NextModel"]()
        model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
        model.layers = torch.nn.ModuleList([self.mhc_layer(config(rt), i) for i in range(45)])
        expected = {}
        for name, param in model.named_parameters():
            if ".hc_" not in name:
                continue
            self.assertIsInstance(param, torch.nn.Parameter)
            self.assertEqual(param.dtype, torch.float32)
            self.assertNotIn(".weight", name)
            # Match the reported checkpoint: bare BF16 fn/base/scale, no scales.
            expected[name] = (torch.arange(param.numel()).reshape(param.shape) % 31 / 128).to(torch.bfloat16)
        self.assertEqual(len(expected), 270)
        self.assertEqual(model.load_weights(iter(expected.items())), set(expected))
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, expected[name].float(), rtol=0, atol=0)

    def test_mhc_bare_parameters_preserve_non_ct_and_mtp_construction(self):
        for quant in (None, config(self.rt)):
            for idx in (0, 3):
                with self.subTest(ct=quant is not None, layer=idx):
                    layer = self.mhc_layer(quant, idx)
                    self.assertEqual(layer.hc_attn_fn.shape, (24, 64))
                    self.assertEqual(layer.hc_ffn_fn.dtype, torch.float32)
                    self.assertNotIn("hc_attn_fn.weight", dict(layer.named_parameters()))
            mtp = self.mhc_layer(quant, 45, is_mtp_layer=True)
            self.assertFalse(hasattr(mtp, "hc_attn_fn"))

    def test_mhc_existing_mapper_accepts_both_bare_parameter_name_conventions(self):
        scope = dict(self.rt.scope, field=dataclasses.field)
        execute_nodes(VLLM_ROOT / "vllm/model_executor/models/utils.py", scope, classes={"WeightsMapper": (None, None)})
        scope["WeightsMapper"] = dataclasses.dataclass(scope["WeightsMapper"])
        path = ROOT / "vllm_ascend/models/glm5next/model.py"
        owner = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGeneration"
        )
        assignment = next(
            n
            for n in owner.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "hf_to_vllm_mapper" for t in n.targets)
        )
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), str(path), "exec"), scope
        )
        mapper = scope["hf_to_vllm_mapper"]
        model = self.rt.scope["Glm5NextModel"]()
        model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
        model.layers = torch.nn.ModuleList([self.mhc_layer(config(self.rt))])
        for side, old_side in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
            for suffix in ("fn", "base", "scale"):
                expected = f"language_model.model.layers.0.hc_{side}_{suffix}"
                for key in (f"{old_side}.{suffix}", f"hc_{side}_{suffix}"):
                    mapped = mapper.map_name(f"model.language_model.layers.0.{key}")
                    self.assertEqual(mapped, expected)
                    name = mapped.removeprefix("language_model.model.")
                    param = dict(model.named_parameters())[name]
                    for dtype in (torch.bfloat16, torch.float16, torch.float32):
                        value = torch.full(param.shape, 0.25, dtype=dtype)
                        self.assertEqual(model.load_weights([(name, value)]), {name})
                        torch.testing.assert_close(param, value.float(), rtol=0, atol=0)

    def test_mhc_loader_rejects_wrong_shape(self):
        model = self.rt.scope["Glm5NextModel"]()
        model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
        model.layers = torch.nn.ModuleList([self.mhc_layer(config(self.rt))])
        with self.assertRaisesRegex(AssertionError, "Attempted to load weight"):
            model.load_weights([("layers.0.hc_attn_fn", torch.zeros(64, 24, dtype=torch.bfloat16))])

    def test_mhc_rejects_integer_codes_without_scale_instead_of_casting_to_float(self):
        model = self.rt.scope["Glm5NextModel"]()
        model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
        model.layers = torch.nn.ModuleList([self.mhc_layer(config(self.rt))])
        for side in ("attn", "ffn"):
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "bare floating-point"):
                model.load_weights([(f"layers.0.hc_{side}_fn", torch.zeros(24, 64, dtype=torch.int8))])

    def test_mhc_pre_keeps_original_fused_operator_fp32_weights_and_norm(self):
        layer = self.mhc_layer(config(self.rt))
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        post, comb = torch.rand(2, 4), torch.rand(2, 4, 4)
        collapsed = torch.randn(2, 16, dtype=x.dtype)
        norm_weight = torch.ones(16)
        calls = []

        def fused(*args):
            calls.append(args)
            return collapsed, post, comb

        def norm(value, weight, epsilon):
            self.assertIs(weight, norm_weight)
            self.assertEqual(epsilon, 1e-5)
            return value + 1, None

        backend = SimpleNamespace(npu_hc_pre_v2=fused)
        globals_ = type(layer).hc_pre.__globals__
        with (
            patch.object(torch.ops, "_C_ascend", backend),
            patch.dict(globals_, {"torch_npu": SimpleNamespace(npu_rms_norm=norm)}),
        ):
            for multiplier, normalized in ((2.0, False), (3.0, True)):
                layer.mhc_post_mult_value = multiplier
                actual = layer.hc_pre(
                    x,
                    layer.hc_attn_fn,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    norm_weight=norm_weight if normalized else None,
                    norm_eps=1e-5,
                )
                torch.testing.assert_close(actual[0], (post * (multiplier / 2)).unsqueeze(-1))
                self.assertIs(actual[1], comb)
                torch.testing.assert_close(actual[2], collapsed + 1 if normalized else collapsed)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][1], layer.hc_attn_fn)
        self.assertEqual(calls[0][1].dtype, torch.float32)
        self.assertEqual(calls[0][4:], (4, 3, 1e-5, 1e-6))

    def test_mtp_eh_projection_native_int8_and_tuple_forward(self):
        rt = self.rt
        scope = dict(rt.scope)
        path = ROOT / "vllm_ascend/models/glm5next/mtp.py"
        owner = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "Glm5NextMultiTokenPredictorLayer"
        )
        ctor = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        end = next(i for i, n in enumerate(ctor.body) if isinstance(n, ast.If) and "self.eh_proj" in ast.unparse(n))
        ctor.body = ctor.body[: end + 1]
        forward = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        owner.body, owner.bases = [ctor, forward], [ast.Name(id="Module", ctx=ast.Load())]
        scope["fused_eh_norm"] = lambda pos, emb, prev, *args: torch.cat((emb, prev), -1)
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            owner,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            scope,
        )
        cfg = SimpleNamespace(hidden_size=16, rms_norm_eps=1e-5)
        vcfg = SimpleNamespace(
            quant_config=config(rt),
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_text_config=cfg)),
        )
        layer = scope["Glm5NextMultiTokenPredictorLayer"](vcfg, "model.layers.45")
        projection = layer.eh_proj
        self.assertEqual(projection.weight.dtype, torch.int8)
        projection.weight.weight_loader(projection.weight, torch.ones(16, 32, dtype=torch.int8))
        projection.weight_scale.weight_loader(projection.weight_scale, torch.full((16, 1), 0.25))
        projection.quant_method.process_weights_after_loading(projection)
        for norm in (layer.enorm, layer.hnorm):
            norm.weight, norm.variance_epsilon = None, 1e-5
        layer.mtp_block = lambda **kw: (kw["hidden_states"], None, None, None)
        layer.shared_head = SimpleNamespace(norm=lambda hidden, residual: (hidden, None))
        x = torch.full((2, 16), 127.0, dtype=torch.bfloat16)
        with patch.object(torch.ops.vllm, "glm5next_mtp_eh_norm", scope["fused_eh_norm"], create=True):
            actual = layer(torch.zeros(2, dtype=torch.long), torch.arange(2), x, inputs_embeds=x)[0]
        torch.testing.assert_close(actual, torch.full((2, 16), 1016.0, dtype=torch.bfloat16))

    def test_kda_restores_projection_config_without_changing_parent_setup(self):
        rt = self.rt
        model_cfg = SimpleNamespace(
            hidden_size=16,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            linear_head_dim=128,
            linear_num_heads=4,
            linear_conv_kernel_dim=4,
            linear_lower_bound=-5 + 1e-3,
        )
        quant = config(rt)
        vcfg = SimpleNamespace(
            quant_config=quant,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=object(),
            speculative_config=None,
        )
        layer = rt.scope["Glm5NextLinearAttention"](model_cfg, vcfg, "layers.0.self_attn")
        self.assertIs(layer.quant_config, quant)
        self.assertIs(vcfg.quant_config, quant)
        self.assertEqual(layer.in_proj_qkvbfg_a.weight.dtype, torch.int8)
        self.assertIn("weight_scale", dict(layer.in_proj_qkvbfg_a.named_parameters()))
        self.assertEqual(layer.in_proj_qkvbfg_a.weight_scale.shape, (4 * 128 * 3 + 4 + 128 * 2, 1))

    def test_router_loads_existing_int8_scales_and_preserves_correction_bias(self):
        rt = self.rt
        cfg = SimpleNamespace(
            hidden_size=16,
            n_routed_experts=8,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            hidden_act="silu",
            topk_method="noaux_tc",
        )
        weights = (torch.arange(8 * 16).reshape(8, 16) % 255 - 127).to(torch.int8)
        scales = ((torch.arange(8) + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        correction = torch.linspace(-1, 1, 8)
        x = torch.tensor([list(range(-7, 8)) + [127]], dtype=torch.bfloat16)
        for world in (1, 2, 4):
            rt.state.world = world
            for rank in range(world):
                rt.state.rank = rank
                moe = rt.scope["Glm5NextMoE"](
                    cfg, SimpleNamespace(use_sequence_parallel_moe=False), config(rt), prefix="layers.10.mlp"
                )
                model = rt.scope["Glm5NextModel"]()
                model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
                model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(11)])
                model.layers[10].mlp = moe
                loaded = model.load_weights(
                    [
                        ("layers.10.mlp.gate.weight", weights),
                        ("layers.10.mlp.gate.weight_scale", scales),
                        ("layers.10.mlp.gate.e_score_correction_bias", correction),
                    ]
                )
                self.assertIn("layers.10.mlp.gate.weight_scale", loaded)
                self.assertEqual(moe.gate.weight.dtype, torch.int8)
                torch.testing.assert_close(moe.gate.weight, weights, rtol=0, atol=0)
                torch.testing.assert_close(
                    moe.gate.weight_scale, scales.to(moe.gate.weight_scale.dtype), rtol=0, atol=0
                )
                torch.testing.assert_close(moe.gate.e_score_correction_bias, correction, rtol=0, atol=0)
                moe.gate.quant_method.process_weights_after_loading(moe.gate)
                actual, bias = moe.gate(x)
                expected = (
                    ((x.to(torch.int32) @ weights.T.to(torch.int32)).float() * scales.T.float()).to(x.dtype).float()
                )
                self.assertEqual(actual.dtype, torch.float32)
                self.assertIsNone(bias)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # The internal MoE runner must also use GateLinear.forward.
                runner = rt.scope["AscendMoERunner"]()
                runner.gate = moe.gate
                torch.testing.assert_close(runner._apply_router_linear(x), expected, rtol=0, atol=0)
                moe.gate.weight_scale.data.mul_(2)
                torch.testing.assert_close(moe.gate(x)[0], expected * 2, rtol=0, atol=0)
                torch.testing.assert_close(moe.gate.e_score_correction_bias, correction, rtol=0, atol=0)

    def test_router_bf16_policy_matches_module_name_and_keeps_legacy_forward(self):
        rt = self.rt
        for quant in (None, config(rt, [r"re:.*\.mlp\.gate$"])):
            gate = rt.scope["AscendGateLinear"](
                16, 8, params_dtype=torch.float32, quant_config=quant, prefix="layers.10.mlp.gate"
            )
            self.assertNotIn("weight_scale", dict(gate.named_parameters()))
            gate.weight.data.fill_(0.25)
            x = torch.ones((2, 16), dtype=torch.bfloat16)
            torch.testing.assert_close(gate(x)[0], torch.full((2, 8), 4.0))
        # Ignore/targets match the module stem, not the full tensor key.
        self.assertIsNotNone(
            config(rt, [r"re:.*mlp\.gate\..*$"])
            .get_quant_method(gate, "layers.10.mlp.gate")
            .__dict__.get("quant_method")
        )

    def test_quantized_router_honors_main_tensor_and_tuple_return_contract(self):
        rt = self.rt
        rt.state.rank, rt.state.world = 0, 1
        for return_bias in (False, True):
            gate = rt.scope["AscendGateLinear"](
                16,
                4,
                quant_config=config(rt),
                return_bias=return_bias,
                prefix="layers.3.mlp.gate",
            )
            gate.weight.weight_loader(gate.weight, torch.ones((4, 16), dtype=torch.int8))
            gate.weight_scale.weight_loader(gate.weight_scale, torch.full((4, 1), 0.25))
            gate.quant_method.process_weights_after_loading(gate)
            output = gate(torch.full((2, 16), 127.0, dtype=torch.bfloat16))
            self.assertEqual(isinstance(output, tuple), return_bias)
            if return_bias:
                output, bias = output
                self.assertIsNone(bias)
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, torch.full((2, 4), 508.0))

    def test_release_mxfp4_mxfp8_detection_is_preserved(self):
        quant = config(self.rt)
        for bits, storage, expected in (
            (4, "mxfp4-pack-quantized", "W4A4_MXFP4"),
            (8, "mxfp8-quantized", "W8A8_MXFP8"),
        ):
            weight = SimpleNamespace(
                type="float", num_bits=bits, strategy="group", group_size=32, symmetric=True, dynamic=False
            )
            activation = SimpleNamespace(
                type="float", num_bits=bits, strategy="group", group_size=32, symmetric=True, dynamic=True
            )
            self.assertEqual(quant._detect_quant_type(weight, activation, storage), expected)

    def test_ct_kpool_head_projection_preserves_release_full_batch(self):
        path = ROOT / "vllm_ascend/attention/indexer_kpool.py"
        owner = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "Glm5NextKPoolIndexerBackend"
        )
        forward = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        projection = next(
            n
            for n in forward.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "compute_topk"
        )
        hidden = torch.arange(96, dtype=torch.float32).reshape(6, 16)
        calls = []

        def project(x):
            calls.append(x.clone())
            return (x[:, :8], None)

        backend = SimpleNamespace(
            _ct_projections=True,
            n_head=4,
            head_dim=4,
            softmax_scale=2.0,
            wq_b=lambda x: (x, None),
            wk_weights_proj=project,
        )
        for shard in (None, SimpleNamespace(start=2, stop=5)):
            scope = dict(
                torch=torch,
                self=backend,
                q_c=hidden,
                hidden=hidden,
                num_tokens=6,
                compute_topk=True,
                indexer_metadata=SimpleNamespace(indexer_row_shard=shard),
            )
            exec(compile(ast.Module(body=[projection], type_ignores=[]), str(path), "exec"), scope)
            # The 0.32 release does not shard query rows. Extra metadata from
            # a newer caller must not change this release's projection shape.
            selected = hidden
            torch.testing.assert_close(calls[-1], selected)
            torch.testing.assert_close(scope["q_values"], selected.view(-1, 4, 4))
            torch.testing.assert_close(scope["weights"], selected[:, 4:8])

    def test_shared_expert_gate_up_and_down_scales_use_native_tp_loaders(self):
        rt = self.rt
        for world in (1, 2, 4):
            rt.state.world = world
            for rank in range(world):
                rt.state.rank = rank
                shared = rt.scope["Glm5NextMLP"](
                    16, 32, "silu", quant_config=config(rt), reduce_results=False, prefix="layers.10.mlp.shared_experts"
                )
                model = rt.scope["Glm5NextModel"]()
                model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
                model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(11)])
                model.layers[10].mlp = torch.nn.Module()
                model.layers[10].mlp.shared_experts = shared
                checkpoint, expected_w, expected_s = [], [], []
                for sid, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                    n, k = (16, 32) if sid == 2 else (32, 16)
                    w = ((torch.arange(n * k).reshape(n, k) + sid) % 255 - 127).to(torch.int8)
                    s = ((torch.arange(n) + sid + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
                    stem = f"layers.10.mlp.shared_experts.{name}"
                    checkpoint.extend([(stem + ".weight", w), (stem + ".weight_scale", s)])
                    expected_w.append(
                        w[:, rank * k // world : (rank + 1) * k // world]
                        if sid == 2
                        else w[rank * n // world : (rank + 1) * n // world]
                    )
                    expected_s.append(s if sid == 2 else s[rank * n // world : (rank + 1) * n // world])
                loaded = model.load_weights(checkpoint)
                self.assertIn("layers.10.mlp.shared_experts.gate_up_proj.weight_scale", loaded)
                for layer, w, s in (
                    (shared.gate_up_proj, torch.cat(expected_w[:2]), torch.cat(expected_s[:2])),
                    (shared.down_proj, expected_w[2], expected_s[2]),
                ):
                    torch.testing.assert_close(layer.weight, w, rtol=0, atol=0)
                    torch.testing.assert_close(layer.weight_scale, s.to(layer.weight_scale.dtype), rtol=0, atol=0)
                    layer.quant_method.process_weights_after_loading(layer)
                    x = torch.ones((1, w.shape[1]), dtype=torch.bfloat16)
                    x[0, -1] = 127
                    expected = ((x.to(torch.int32) @ w.T.to(torch.int32)).float() * s.T.float()).to(x.dtype)
                    torch.testing.assert_close(layer.quant_method.apply(layer, x), expected, rtol=0, atol=0)

    def test_model_free_matching_preserves_bare_correction_bias(self):
        # Execute compressed-tensors' actual matching contract without its
        # heavyweight package imports. Only *.weight is targeted by default.
        scope = dict(re=re)
        execute_nodes(
            ROOT.parent / "compressed-tensors/src/compressed_tensors/utils/match.py",
            scope,
            functions=("match_name", "match_quantizable_tensors"),
        )
        tensors = {"layers.10.mlp.gate.weight": object(), "layers.10.mlp.gate.e_score_correction_bias": object()}
        self.assertEqual(
            list(scope["match_quantizable_tensors"](tensors, [], [r"re:^(?!.*\.experts\.\d+\.).*$"])),
            [("layers.10.mlp.gate", "layers.10.mlp.gate.weight")],
        )

    def test_fused_weight_scale_tp_and_glm_loader(self):
        rt = self.rt
        for world in (1, 2, 4):
            rt.state.world = world
            for rank in range(world):
                rt.state.rank = rank
                fused = rt.scope["_Glm5NextMergedColumnParallelLinear"](
                    16,
                    [16, 16, 16, 4, 8, 8],
                    (4, 5),
                    world,
                    bias=False,
                    quant_config=config(rt),
                    params_dtype=torch.bfloat16,
                    prefix="layers.0.self_attn.in_proj_qkvbfg_a",
                )
                model = rt.scope["Glm5NextModel"]()
                model.config = SimpleNamespace(n_routed_experts=None, layer_types=None)
                attn = torch.nn.Module()
                attn.in_proj_qkvbfg_a = fused
                layer = torch.nn.Module()
                layer.self_attn = attn
                model.layers = torch.nn.ModuleList([layer])
                weights, expected_w, expected_s = [], [], []
                for sid, (name, n) in enumerate(
                    zip(rt.helper.glm5next_packed_mapping()["in_proj_qkvbfg_a"], [16, 16, 16, 4, 8, 8])
                ):
                    w = ((torch.arange(n * 16).reshape(n, 16) + sid) % 255 - 127).to(torch.int8)
                    s = ((torch.arange(n) + sid + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
                    weights.extend(
                        [(f"layers.0.self_attn.{name}.weight", w), (f"layers.0.self_attn.{name}.weight_scale", s)]
                    )
                    part = slice(None) if sid in (4, 5) else slice(rank * n // world, (rank + 1) * n // world)
                    expected_w.append(w[part])
                    expected_s.append(s[part])
                loaded = model.load_weights(weights)
                self.assertIn("layers.0.self_attn.in_proj_qkvbfg_a.weight_scale", loaded)
                self.assertTrue(torch.equal(fused.weight, torch.cat(expected_w)))
                self.assertTrue(torch.equal(fused.weight_scale, torch.cat(expected_s)))
                self.assertEqual(fused.tp_rank, rank)
                x = torch.tensor([list(range(-7, 8)) + [127]], dtype=torch.bfloat16)
                fused.quant_method.process_weights_after_loading(fused)
                output = fused.quant_method.apply(fused, x)
                expected = torch.cat(
                    [
                        (x.to(torch.int32) @ w.T.to(torch.int32)).float() * s.T.float()
                        for w, s in zip(expected_w, expected_s)
                    ],
                    -1,
                ).to(x.dtype)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_lm_head_parameters_vocab_padding_and_scale_execution(self):
        rt = self.rt
        rt.state.world = 4
        weight = ((torch.arange(10 * 16).reshape(10, 16) % 255) - 127).to(torch.int8)
        scale = ((torch.arange(10) + 1) / 1024).to(torch.bfloat16).unsqueeze(1)
        for rank in range(4):
            rt.state.rank = rank
            head = rt.scope["AscendParallelLMHead"](
                10, 16, padding_size=16, quant_config=config(rt), params_dtype=torch.bfloat16, prefix="lm_head"
            )
            self.assertEqual(head.weight.dtype, torch.int8)
            self.assertIn("weight_scale", dict(head.named_parameters()))
            head.weight.weight_loader(head.weight, weight)
            head.weight_scale.weight_loader(head.weight_scale, scale)
            rows = max(0, min(4, 10 - rank * 4))
            self.assertTrue(torch.equal(head.weight_scale[:rows], scale[rank * 4 : rank * 4 + rows]))
            self.assertEqual(head.weight_scale[rows:].count_nonzero(), 0)
            head.quant_method.process_weights_after_loading(head)
            x = torch.tensor([list(range(-7, 8)) + [127]], dtype=torch.bfloat16)
            actual = head.quant_method.apply(head, x)
            expected = torch.zeros((1, 4), dtype=x.dtype)
            if rows:
                expected[:, :rows] = (
                    (x.to(torch.int32) @ weight[rank * 4 : rank * 4 + rows].T.to(torch.int32)).float()
                    * scale[rank * 4 : rank * 4 + rows].T.float()
                ).to(x.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            head.weight_scale.data *= 2
            torch.testing.assert_close(head.quant_method.apply(head, x), actual * 2, rtol=0, atol=0)

    def test_ignored_head_and_partial_fused_ignore(self):
        rt = self.rt
        head = rt.scope["AscendParallelLMHead"](
            16, 16, quant_config=config(rt, ["lm_head"]), params_dtype=torch.bfloat16, prefix="lm_head"
        )
        self.assertEqual(head.weight.dtype, torch.bfloat16)
        self.assertNotIn("weight_scale", dict(head.named_parameters()))
        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            rt.scope["_Glm5NextMergedColumnParallelLinear"](
                16,
                [16, 16, 16, 4, 8, 8],
                (4, 5),
                1,
                bias=False,
                quant_config=config(rt, [r"re:.*\.q_proj$"]),
                prefix="layers.0.self_attn.in_proj_qkvbfg_a",
            )

    def test_mapping_is_fresh_and_legacy_profile_is_unchanged(self):
        inherited = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        first = self.rt.helper.glm5next_packed_mapping(inherited)
        first["qkv_proj"].append("other")
        self.assertEqual(len(inherited["qkv_proj"]), 3)
        self.assertFalse(self.rt.helper.is_glm_ct_int8_config(None))
        legacy = config(self.rt)
        next(iter(legacy.target_scheme_map.values()))["weights"].type = "float"
        self.assertFalse(self.rt.helper.is_glm_ct_int8_config(legacy))

    def test_w4a8_unfused_gmm_preserves_channel_scales_and_assist(self):
        scope = dict(torch=torch, COMPRESSED_TENSORS_METHOD="compressed-tensors", dispose_tensor=lambda t: None)
        execute_nodes(
            ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py",
            scope,
            classes={"AscendW4A8DynamicFusedMoEMethod": (None, {"apply_gmm1"})},
        )
        method = scope["AscendW4A8DynamicFusedMoEMethod"]()
        for expert_list in (False, True):
            weights = (
                [torch.zeros((16, 2), dtype=torch.int32) for _ in range(2)]
                if expert_list
                else [torch.zeros((2, 16, 2), dtype=torch.int32)]
            )
            scales = (
                [torch.ones(16, dtype=torch.int64) * (i + 1) for i in range(2)]
                if expert_list
                else [torch.ones((2, 16), dtype=torch.int64)]
            )
            assist = [torch.ones(16) for _ in range(2)] if expert_list else [torch.ones((2, 16))]
            method._get_mlp_weights = lambda layer, w=weights, s=scales: (
                w,
                s,
                None,
                [torch.ones(1, dtype=torch.int64)],
            )
            method._get_mlp_scale_bias = lambda layer, b=assist: (b, None)
            method._quant_hidden_states = lambda x, s: (x, s)
            method._maybe_convert_group_list = lambda inputs: (inputs.group_list, 1)
            calls = []

            def gmm(_calls=calls, **kwargs):
                _calls.append(kwargs)
                return [torch.ones((1, 16))]

            scope["torch_npu"] = SimpleNamespace(npu_grouped_matmul=gmm)
            inputs = SimpleNamespace(
                hidden_states=torch.ones((1, 16), dtype=torch.int8),
                dynamic_scale=torch.ones(1),
                layer=None,
                group_list=torch.tensor([1, 0]),
            )
            method.apply_gmm1(inputs)
            self.assertIs(calls[0]["bias"], assist)
            self.assertEqual(len(calls[0]["scale"]), len(scales))
            self.assertEqual(calls[0]["scale"][0].shape, (1, 16) if expert_list else (2, 1, 16))

    def test_w4a8_silu_fallback_retains_model_clamp(self):
        path = ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py"
        cls = next(
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "AscendW4A8DynamicFusedMoEMethod"
        )
        setting = next(
            n
            for n in cls.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "fused_activations" for t in n.targets)
        )
        ns = {}
        exec(compile(ast.Module(body=[setting], type_ignores=[]), str(path), "exec"), ns)
        # ModelSlim keeps main's fused SiLU; CT narrows this per instance.
        self.assertIn("silu", ns["fused_activations"])
        self.assertIn("situ", ns["fused_activations"])
        activation = SimpleNamespace(
            SILU="silu",
            SITU="situ",
            SWIGLUOAI="swigluoai",
            SWIGLUOAI_UNINTERLEAVE="swigluoai_uninterleave",
            SWIGLUSTEP="swiglustep",
            GELU="gelu",
            GELU_TANH="gelu_tanh",
        )
        scope = dict(
            torch=torch,
            MoEActivation=activation,
            torch_npu=SimpleNamespace(
                npu_swiglu=lambda x: torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1]
            ),
        )
        execute_nodes(ROOT / "vllm_ascend/ops/fused_moe/moe_mlp.py", scope, functions=("_unified_apply_activation",))
        values = torch.tensor([[-20.0, 0.0, 20.0, -30.0, 30.0, 2.0]])
        result = scope["_unified_apply_activation"](SimpleNamespace(activation="silu", swiglu_limit=10.0), values, None)
        expected = torch.tensor([[g / (1 + math.exp(-g)) * u for g, u in ((-20.0, -10.0), (0.0, 10.0), (10.0, 2.0))]])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, default=ROOT.parent / "vllm")
    args, remaining = parser.parse_known_args()
    VLLM_ROOT = args.vllm_root
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
