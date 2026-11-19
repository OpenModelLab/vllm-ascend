# GLM-5.3-Flash：A2 MLP-only CT 适配

2026-11-19。基于 `b950109b0818530037e936846c05a9d92f98a51c` 新增提交，
保留全部历史提交。本说明适用于从原始 BF16 重新生成的 MLP-only checkpoint。

## 量化与兼容范围

| 模块 | 权重与激活 | 本次执行路径 |
| --- | --- | --- |
| Routed experts | INT4 / 动态 token INT8 | 保留 CT packed 加载、TP/EP、逐专家转换和 A2/A3 NZ GMM |
| Dense/shared MLP | INT8 / 动态 token INT8 | 原生 W8A8；保留 gate/up 融合及 scale 分片 |
| Attention、KDA、indexer、router | 浮点 | 恢复上游浮点模块及原生 KPool 算法 |
| 视觉、LM head、MTP eh_proj | 浮点 | 恢复上游模块；MTP 的 MLP 仍按上述规则量化 |
| mHC | 裸浮点 Parameter | 保留原有名称、FP32 参数槽和计算 |
| KV-b | 原始浮点权重 | 原生 MLA 吸收，不再从 INT8 恢复 |
| DSA/indexer cache | 独立运行时选项 | 原有 C8 配置和实现不变，量化脚本不导出 cache scale |

非 MLP 的 CT INT8 兼容逻辑已撤下；旧宽范围 checkpoint 的整数权重或量化参数会明确报错，
不会跳过 scale 或自动恢复浮点。若继续使用旧产物，请保留旧服务修订；不要只编辑其 config。
ModelSlim、原生 BF16/FP8 的入口未改为 MLP-only CT 路径。

当前 llm-compressor 脚本仍将浮点 `index_kpool_compress_gate` 导出为 `.weight` 名称。
新 loader 同时接受裸名和 `.weight`，只重命名且保留 tensor/loader kwargs；
同一权重流出现两种重复别名会报错。无需修改新产物或再次量化来解决该名称差异。
这项映射同时覆盖主模型和 MTP 的 language-model 前缀处理。

## 代码变化

- 恢复 `attention.py`、`indexer_kpool.py`、`sfa_v1.py`、视觉 attention 和 router 为原生浮点实现。
- 撤掉 KDA 的全局 CT INT8 开关、MTP eh_proj 包装和 LM head 的额外 CT 分派。
- `glm5next_packed_mapping()` 仅增加 MLP gate/up 映射，仍复制继承的映射。
- 主模型记录 quant config，供加载期范围校验使用；不据此改变非 MLP 计算结构。
- `prepare_glm_packed_weights()` 保留专家打包转换，在 CT INT 流上增加范围与浮点别名处理。
- 保留 CT GMM scale/assist、逐专家显存优化、A5 设备分支隔离、FLA `max_query_len` 和启动日志。

未改 llm-compressor 或 compressed-tensors；没有修改权重文件，没有改写或合并历史提交。

## 本地检查与 NPU 验收

本次为 CPU 源码、整数参考和 loader 验证；记录见同目录的
`GLM5.3-Flash-CT-W4A8-A2-MLP-only-verification.json`。
完整真实权重、NPU kernel、EP、MTP、图文、ACLGraph、flashcomm1 和任务精度均尚待节点验收。
旧版 eager GPQA 结果不等于本次回退后新模型的验收。
未填入未经实测的精度门限或性能值，也未据此新增带虚构分数的 accuracy YAML。

评测结束后，在本人的配套 A2 容器中先执行以下小规模测试；不要与占用相同 NPU 的服务同时运行：

```bash
cd /vllm-workspace/vllm-ascend
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0 python -u -m pytest -q -v \
  tests/e2e/pull_request/one_card/test_w4a8_a2_numeric.py \
  tests/e2e/pull_request/one_card/test_glm5next_ct_linears.py \
  tests/e2e/pull_request/one_card/test_glm5next_ct_special_projections.py
```

继续使用 FLA 图探针检查原生卷积；本次保留其长度上界修复，未宣称 FULL 模式卡死已经解决：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=10s 180s \
  python -u -m pytest -q -v -s tests/e2e/pull_request/one_card/test_glm5next_conv_aclgraph.py
```

## 真实模型部署模板

910B3 64 GB × 4，TP4 + EP。下面路径为新产物示例，请对齐实际目录。
非 MLP 恢复浮点后，checkpoint 大小和显存可能增加，不能沿用旧宽范围产物的 155 GiB 估算。
以实际文件总量、加载峰值、HBM 和 cache 分配结果为准。
当前按用户既有 16K 场景起步；128K/bs16 尚未实测，不作为容量承诺。

```bash
cd /workspace
MODEL_PATH=/mnt/share/w30015592/models/GLM-5.3-Flash-MLP-W4A8-CT
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL_PATH" \
  --served-model-name glm53-ct-a2 --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype bfloat16 --quantization compressed-tensors \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --kv-cache-dtype int8 --attention-config.indexer_kv_dtype int8 \
  --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --gpu-memory-utilization 0.85 \
  --skip-mm-profiling --enforce-eager --reasoning-parser glm47 \
  --additional-config '{"enable_fused_mc2":0,"weight_nz_mode":2}' \
  2>&1 | tee /mnt/share/w30015592/tmp/glm53_a2_mlp_only.log
```

首请求成功后，再按图模式教程单独替换 eager 配置，验证 PIECEWISE/FULL；
不能把进程存活、HTTP 200 或 dummy 启动当成新权重精度通过。
`--skip-mm-profiling` 只用于文本场景隔离，图文服务的显存预算和首图请求另行测试。
MTP 需要完整的浮点辅助投影和量化 MLP，不能用忽略缺失权重的方式启用。

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-ct-a2","messages":[{"role":"user","content":"计算2+3，用中文简短回答。"}],"temperature":0,"max_tokens":512}'
```

沿用 `ais_bench_benchmark==3.1.20260630` 的官方 API 配置，保持数据集、prompt、
生成参数与 BF16 对照一致，再比较新模型的任务精度。评测命令及缓存对照见原 A2 教程。
