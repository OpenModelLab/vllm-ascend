# GLM-5.3-Flash CT W4A8 on A2: main migration

This change serves the existing **MLP-only INT compressed-tensors checkpoint**:
routed experts use per-channel symmetric W4 with per-token dynamic A8; dense
and shared MLPs use per-channel symmetric W8 with per-token dynamic A8.
Attention, indexer projections, router, mHC, vision and MTP `eh_proj` remain
floating point. Dynamic C8 is selected at serving time, independently of the
checkpoint's weight quantization.

The code and CPU regressions are prepared locally. **NPU startup, ACL graph
replay and MMMU-Pro accuracy have not been run on this branch.** The commands
below are the hardware acceptance steps, not recorded passing results.

## Pinned sources

| Component | Revision |
| --- | --- |
| This repository base | `28e9e97ba254b076a88275e8a275556cb9a11664`, branch `a2-main-28e9e97ba` |
| Migration source | `A2-vllm-ascend-0.32.0`, `releases/v0.32.0rc`, `069566eb7` |
| Paired upstream vLLM main | `24420b7a341af37a9bc3186e6fc6062512e0aa58`, from `.github/vllm-main-verified.commit` |
| Supported release lane recorded at this base | `v0.32.0`, from `.github/vllm-release-tag.commit` |
| FLA A2 dependency recorded in the Dockerfile | `flash-linear-attention-npu-a2==26.9.1` |

Use the dependencies/build instructions of this pinned main revision, including
the matching CANN, torch_npu, Triton and `cann_ops_transformer` operators. An
editable install of this repository does not upgrade those runtime components.
For main validation, pair it with the recorded upstream main SHA; do not assume
an older `v0.32.0rc1` container's installed vLLM APIs match main.

## Migration decisions

| Area | Action and reason |
| --- | --- |
| CT scheme detection | Parse `pack-quantized` A8 metadata, select `W4A8_DYNAMIC` and mark compact expert storage. Group names are arbitrary; matching uses `targets`, `ignore` and fused mappings. |
| CT storage boundary | Recognize `weight_packed/weight_shape/weight_scale`; convert K-axis offset-binary nibbles to output-axis signed INT4 bytes. Preserve codes and scales exactly. Metadata may arrive in a later shard. |
| Native TP/EP loader | Reuse current upstream `RoutedExperts` slicing and scale loader. Do not transplant the old MoE adapter or its removed arguments. |
| GLM-owned loader | Connect the stream adapter at main model and MTP loading entries because these bypass `RoutedExperts.load_weights`. Preserve current PP, FP8 pending buffers, MTP embedding/head checks and rotation handling. |
| Dense/shared fusion | Add the model's `gate_up_proj` mapping, preserving inherited vision mappings. Native W8A8 methods create and load weight/scale parameters. |
| Loading memory | Keep resident expert weights compact; compute correction bias one expert at a time. Retain the native bias expression, scale bit encoding, forced NZ conversion and GMM calls. |
| CT SiLU dispatch | Disable only CT's fused SiLU capability: its correction assist is incompatible with the per-token fused SwiGLU contract. Reuse native GMM1, activation/clamp, dynamic A8 and GMM2. Other quantization formats retain their activation capabilities. |
| FLA interface | Supply `max_query_len=1` for decode and `num_spec+1` for verification when supported. Resolve the optional keyword once; no device tensor reads are added to decode. |
| Main SFA INT8 cache | Already present; reuse it without changing quantization or attention kernels. |
| KPool indexer INT8 cache | Missing from this pinned main despite existing in the release source. Port the source's INT8 support (`3e5155a13`, official release feature #20601): INT8 keys/queries, FP16 descales, managed cache pages and CANN `quant_mode=2`. Preserve main's TP row-sharding/exchange and BF16/FP8 dispatch. |
| MoE routing entry | Already calls `torch_npu.npu_moe_gating_top_k(..., bias=...)` in the new router; no `device_op.py` change. |
| Existing GLM support | Keep native floating attention/KDA, mHC, vision, MTP rotation and cache planning. Do not reintroduce the historical broad-range CT projection or KV-b recovery adaptations. |
| A5 and diagnostic patches | Do not migrate A5 K-CG, generic Triton Mamba patches, old worker/profile patches or startup tracing. |

The KPool INT8 addition also updates the A2/A3 hardware capability and the fused
cache writer. Keys and their FP16 scales share managed physical pages, including
padding and invalid-slot handling. The incomplete pool and KDA/SSM states retain
their native floating types; C8 does not mean every model state becomes INT8.
The INT8 operator must be present in the deployed CANN operator stack.

Old broad-scope checkpoints containing integer attention/router/mHC/vision
weights or their quantization parameters are rejected explicitly. This avoids
silently interpreting integer codes as floating weights or dropping scales.
The floating KPool gate accepts both a bare parameter key and a `.weight` alias.

## Install and run hardware acceptance

Run inside an A2 container with the paired software stack. Set paths to the
actual source and existing checkpoint; no re-quantization is part of migration.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL=/path/to/GLM-5.3-Flash-MLP-W4A8-CT
cd /path/to/A2-vllm-ascend-latest
python -m pip install -e . --no-build-isolation --no-deps

# Real GMM1 -> native clamp/SiLU -> dynamic A8 -> GMM2 vs CPU integer reference.
# Also compares the original CT bias/repack expression with bounded processing.
python -m pytest -q \
  tests/e2e/pull_request/one_card/test_w4a8_a2_numeric.py \
  tests/e2e/pull_request/one_card/test_ct_w4a8_loading.py \
  tests/e2e/pull_request/one_card/test_glm5next_ct_linears.py

# Managed KPool INT8 key/scale page writes and operator reads, eager + graph.
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py
```

These are one-card tests: if a test runner selects a different visible NPU,
restrict `ASCEND_RT_VISIBLE_DEVICES` to one card for the test invocation. A skip
or a missing CANN operator is not an acceptance pass. Both numerical suites must
pass before treating a successfully started service as correct.

Basic eager startup, TP4 with EP enabled:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL=/path/to/GLM-5.3-Flash-MLP-W4A8-CT
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_CT_FAST_LOAD=0

vllm serve "$MODEL" \
  --served-model-name glm53-ct \
  --host 0.0.0.0 --port 9900 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --kv-cache-dtype int8 \
  --attention-config.indexer_kv_dtype int8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --enforce-eager
```

The two C8 flags are independent: `--kv-cache-dtype int8` enables main SFA C8;
`--attention-config.indexer_kv_dtype int8` enables KPool indexer C8. If the old
MMMU-Pro baseline used indexer `auto`/BF16, keep that same value for accuracy
comparison rather than silently changing cache precision. TP8 uses eight
visible cards and `--tensor-parallel-size 8`; loading shortcuts support both
TP4 and TP8.

For graph validation, replace `--enforce-eager` with:

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Validate text, image requests and repeated decode/graph replay first. To test
MTP, add the official speculative configuration to a separate run using a
checkpoint that contains its MTP layer weights and, when required, `rot.weight`:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Do not enable MTP solely for this migration's accuracy comparison if the old
baseline disabled it. Missing MTP weights still fail the native checks.

## Optional CPU loading acceleration

`VLLM_ASCEND_CT_FAST_LOAD` accepts `0` (default) or `1` and contains no secrets.
With `1`, CPU code directly repacks bytes without materializing the expanded
INT8 matrix, and GLM indexes the expert mapping instead of scanning every
expert. Native TP/EP slicing, scale values, correction bias and execution
kernels remain the same. It does not accelerate checkpoint reads or graph
capture. First validate correctness with the default, then compare startup time
and outputs with `1` using the same inputs.

## Local regression evidence and manual accuracy comparison

CPU validation completed during migration:

- 128 pytest cases passed: CT pack/unpack/direct repack, metadata ordering,
  bounds on temporary allocations, unchanged bias/scale bits and forced NZ
  dispatch, FLA old/new signatures, KPool BF16/FP8/INT8 with and without TP row
  slicing and cache-only updates. NPU/distributed launches are isolated in
  these CPU tests.
- The actual paired vLLM `RoutedExperts` source passed 34 TP/EP subcases
  (expanded + compact storage, TP1/2/4/8 where supported by the fixture).
  Four additional subcases execute this branch's real GLM main/MTP loader
  methods against native expert loaders, with shared weight/scale fusion and
  optional mapping acceleration. Distributed placement and initialization are
  isolated; these do not substitute for NPU kernels.
- Ten configuration type tests passed with the source methods and runtime
  registry isolated on CPU. They cover standalone and mixed packed detection,
  activation retention, compact MoE scheme selection and rejection of packed
  W4A8 dense linears/static activations.

Portable CPU subset:

```bash
python -m pytest --noconftest -q \
  tests/ut/quantization/test_compressed_tensors_packed.py \
  tests/ut/quantization/test_compressed_tensors_fast_load.py \
  tests/ut/quantization/test_w4a8_loading_memory.py \
  tests/ut/models/test_glm5next_conv_length_cpu.py \
  tests/ut/models/test_glm5next_int8_dispatch_cpu.py

python tests/ut/quantization/test_ct_native_loader_source.py \
  --vllm-root /path/to/paired-vllm-checkout
```

Run the standard configuration/model UTs in the complete development stack:

```bash
python -m pytest -q \
  tests/ut/quantization/configs/test_compressed_tensors_config.py \
  tests/ut/models/test_glm5next_conv_dispatch.py \
  tests/ut/models/test_glm5next_int8_kpool.py \
  tests/ut/models/test_glm5next_kpool_model_backend.py \
  tests/ut/models/test_glm5next_kv_cache.py \
  tests/ut/device/test_hardware_profile.py
```

For MMMU-Pro, retain the existing ais_bench 3.1.20260630 model/dataset configs,
same checkpoint, tokenizer/chat template, image processing, sampling, output
limits, C8 flags, TP/EP and MTP setting. Change only the API endpoint/model name
if required. The 16384 context/4-sequence startup example above is for smoke
validation; use the original evaluation's context/output limits for the scored
comparison. Compare multiple rounds and per-question details against the old
repository with the same weights. Successful health checks, CPU bit equality
and a single text response do not establish MMMU-Pro accuracy parity.

No local Git history was rewritten and no changes were made to the source
repository. Before opening a PR, assign its number and add the required release
fragment using `release_notes/TEMPLATE.yaml`. Suggested entries: support GLM
CT packed W4A8 on A2/A3; retain bounded FLA decode lengths; support managed
INT8 KPool cache on this main lane; document the optional CT fast-load switch.
