# SPDX-License-Identifier: Apache-2.0
"""CPU source regression for INT8 KPool dispatch alongside main's TP sharding.

Only CANN/Triton launches and distributed exchange are isolated. Cache-format
selection, quantization, metadata selection and argument construction are real.
The companion NPU tests validate physical page writes and operator numerics.
"""

import importlib.util
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("ct_source_utils", ROOT / "tests/ut/quantization/_ct_test_utils.py")
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


@pytest.mark.parametrize(
    "dtype,scale_dtype",
    [
        (torch.bfloat16, torch.int8),
        (torch.float8_e4m3fn, torch.float32),
        (torch.int8, torch.float16),
    ],
)
def test_cache_spec_keeps_key_and_scale_dtypes_paired(dtype, scale_dtype):
    context = {}
    scope = dict(
        Module=torch.nn.Module,
        torch=torch,
        AscendMLAAttentionSpec=SimpleNamespace,
        get_current_vllm_config=lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            compilation_config=SimpleNamespace(static_forward_context=context),
        ),
    )
    _fixture.execute_nodes(
        ROOT / "vllm_ascend/models/glm5next/kv_cache.py",
        scope,
        classes={"Glm5NextIndexerCache": ("Module", {"__init__", "get_kv_cache_spec"})},
    )
    cache = scope["Glm5NextIndexerCache"](
        head_dim=128,
        dtype=dtype,
        cache_role="indexer",
        cache_config=SimpleNamespace(block_size=128),
        prefix="indexer.cache",
        compress_ratio=4,
    )
    spec = cache.get_kv_cache_spec(None)
    assert spec.dtype == dtype and spec.scale_dtype == scale_dtype
    assert spec.scale_dim == (0 if dtype == torch.bfloat16 else 1)
    assert spec.tokens_per_state == 4 and spec.indexes_kv_by_block_stride
    assert context["indexer.cache"] is cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("compute_topk", [False, True])
def test_cache_quantization_and_query_scales_follow_tp_row_slice(dtype, sharded, compute_topk):
    exchange = Mock()
    scope = dict(
        Module=torch.nn.Module,
        torch=torch,
        F=F,
        partial=partial,
        CUDAGraphMode=SimpleNamespace(FULL="FULL"),
        get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode=None),
        _exchange_row_shard_topk=exchange,
        torch_npu=SimpleNamespace(
            npu_dynamic_quant=lambda x, dst_type: (x.to(dst_type), torch.ones(x.shape[:-1], dtype=torch.float32))
        ),
    )
    _fixture.execute_nodes(
        ROOT / "vllm_ascend/attention/indexer_kpool.py",
        scope,
        classes={
            "Glm5NextKPoolIndexerBackend": (
                "Module",
                {
                    "_forward_cann",
                    "_symmetric_int8",
                    "_pool_key_indexer_fp8",
                    "num_cache_tensors",
                },
            )
        },
    )
    cls = scope["Glm5NextKPoolIndexerBackend"]
    op = cls.__new__(cls)
    torch.nn.Module.__init__(op)
    op.topk_tokens, op.index_kpool, op.head_dim = 4, 4, 4
    op.k_cache = SimpleNamespace(dtype=dtype)
    assert op.num_cache_tensors == (1 if dtype == torch.bfloat16 else 2)
    op.topk_indices_buffer = torch.empty(2, 7, dtype=torch.int32)
    op.index_kpool_compress_ape = torch.zeros(4, 4)
    op.k_norm = SimpleNamespace(eps=1e-6)
    op._norm_weight_f32 = op._norm_bias_f32 = None
    op._key_weight = op._gate_weight = torch.eye(4, dtype=torch.bfloat16)
    pooled = torch.tensor([[0, 0, 0, 0], [0.5, -1, 2, -3]], dtype=torch.bfloat16)
    op._key_pool = Mock(return_value=pooled)
    op._prepare_keypool_state = Mock(return_value=torch.zeros(2, 4, 8))
    op._write_keypool_caches = Mock()

    def native(query, cache, weights, tail, **kwargs):
        return torch.zeros(query.shape[0], 4, dtype=torch.int32), None

    op._cann_pool_key_indexer = Mock(side_effect=native)
    op._pool_key_indexer = Mock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
    op._pack_keypool_indices = Mock(
        side_effect=lambda indices, scoring, *args: torch.zeros(indices.shape[0], 1, 7, dtype=torch.int32)
    )
    metadata = SimpleNamespace(
        write_metadata=None,
        positions=torch.arange(2),
        cum_query_lens=torch.tensor([2]),
        query_start_loc=torch.tensor([0, 2]),
        start_pos=torch.tensor([0]),
        pool_tail=torch.tensor([0]),
        pooled_slot_mapping=torch.tensor([0, 1]),
        keypool_state_table=torch.zeros(1, 1),
        keypool_start_offsets=torch.tensor([0]),
        seq_lens=torch.tensor([2]),
        block_table=torch.zeros(1, 2),
        num_actual_tokens=2,
        indexer_row_shard=None,
    )
    if sharded:
        metadata.indexer_row_shard = SimpleNamespace(
            start=0,
            stop=1,
            sizes=(1, 1),
            positions=torch.arange(1),
            cum_query_lens=torch.tensor([1]),
            seq_lens=torch.tensor([2]),
            pool_tail=torch.tensor([0]),
            block_table=torch.ones(1, 2),
        )
    local_rows = 1 if sharded else 2
    query = torch.ones(local_rows, 1, 4, dtype=torch.bfloat16) if compute_topk else None
    weights = torch.ones(local_rows, 1) if compute_topk else None
    cache = torch.zeros(2, 2, 1, 4, dtype=dtype)
    scale_cache = (
        torch.zeros(2, 2, 1, 1, dtype=torch.float16 if dtype == torch.int8 else torch.float32)
        if dtype != torch.bfloat16
        else None
    )
    tail = torch.zeros(2, 4, 8)
    tail_metadata = SimpleNamespace(block_size=4, cann_inputs=SimpleNamespace(retained_indices=None))
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("host synchronization")):
        result = op._forward_cann(
            torch.ones(2, 4, dtype=torch.bfloat16),
            query,
            weights,
            cache,
            tail,
            metadata,
            tail_metadata,
            compute_topk=compute_topk,
            indexer_scale_cache=scale_cache,
        )
    written = op._write_keypool_caches.call_args
    assert written.kwargs["scale_cache"] is scale_cache
    if dtype == torch.int8:
        codes, scales = written.args[4], written.kwargs["pooled_scales"]
        assert codes.dtype == torch.int8 and scales.dtype == torch.float16
        assert torch.count_nonzero(codes[0]) == 0 and torch.all(scales > 0)
        torch.testing.assert_close(codes.float() * scales[:, None], pooled.float(), atol=0.02, rtol=0)
    else:
        assert written.args[4] is pooled and written.kwargs["pooled_scales"] is None
    if not compute_topk:
        assert result is None
        op._cann_pool_key_indexer.assert_not_called()
        exchange.assert_not_called()
        return
    call = op._cann_pool_key_indexer.call_args
    scoring = metadata.indexer_row_shard or metadata
    assert call.kwargs["actual_seq_q"] is scoring.cum_query_lens
    assert call.kwargs["block_table"] is scoring.block_table
    assert call.args[0].shape[0] == local_rows
    if dtype == torch.int8:
        op._pool_key_indexer.assert_not_called()
        assert call.args[0].dtype == torch.int8 and call.args[2].dtype == torch.float16
        assert call.kwargs["quant_mode"] == 2
        assert call.kwargs["q_descale"].shape == (local_rows, 1)
        assert call.kwargs["q_descale"].dtype == torch.float16
        assert call.kwargs["k_descale"].data_ptr() == scale_cache.data_ptr()
    else:
        op._pool_key_indexer.assert_called_once()
        assert call.args[2].dtype == torch.bfloat16
        assert call.kwargs.get("quant_mode") == (0 if dtype == torch.float8_e4m3fn else None)
    assert result.shape == (2, 1, 7)
    assert exchange.call_count == int(sharded)
