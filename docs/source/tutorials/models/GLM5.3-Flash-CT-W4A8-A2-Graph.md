# GLM-5.3-Flash A2：图模式启动诊断与分段配置

2026-11-18。环境为 910B3 64 GB × 4、TP4，镜像 `v0.32.0rc1-openeuler`，
用户报告插件基线 `24487f4f4`，模型为已有 CT INT W4A8 产物。

## 证据能说明什么

报告中的 WARNING 来自 `patch/worker/patch_triton.py`：尝试从只有辅助函数的文件
导入不存在的 `causal_conv1d_update_npu`，然后直接宣称 PyTorch fallback 卡死 decode-FULL。
这是一个确定存在的错误导入和误导诊断，但不能据此认定 GLM 的调用路径。

实际调用链为：

```text
Glm5NextLinearAttention._forward_core
  → models/glm5next/ops/causal_conv1d.causal_conv1d
  → fla_npu.ops.ascendc.causal_conv1d_update
```

它不使用上述 generic upstream rebind。多模态能力和 KDA/SSM 是模型的两个结构特征，
不能从“多模态模型”推导它走通用 Mamba 卷积 fallback。
以上调用链已在本地官方 `24487f4f4` 源码核实。用户随后提供的变更清单包含
`modeling_glm5next.py`、`ops/layers/attention/kda.py` 和重构后的量化目录，
与本地该提交的 tracked tree 不同；容器若使用另一套覆盖文件，需以实际绑定和源码为准。
诊断脚本同时尝试这两套 KDA 路径，不能只根据 HEAD 或 WARNING 推断运行路径。

`EngineCore → wait_for_ready → selectors.select` 只说明在等 Worker。
报告的 WatchDog 时间 11:25:57 早于 12:09:22 权重后处理完成，需要按同一轮启动的 PID、时间戳核对。
补充报告指出 eager 启动也有相同等待栈、随后成功，这进一步说明该栈本身不能区分加载等待与捕获阻塞。
现有资料没有 Worker 卡住的位置，也没有图捕获开始日志，
尚不能把最后的 block-size 日志解释为“第一个 decode 图捕获卡死”。

启动后还有 cache-spec 计算、内存/视觉 profiling、KV 分配、编译及 capture 等阶段。
`--max-model-len 16384` 表示可接受的序列长度，不等于启动时直接执行 16384-token decode。

## 三个待验证问题的源码结论

1. 官方 rc1 `24487f4f4` 与本地迁移基线 `2dd5b2b2d` 的 GLM KDA、模型卷积 wrapper 和
   WARNING 相关文件相同。核对 2026-11-18 官方 main `92b6354a389e8652d67315215b022ab6be66529d`，
   Triton 文件新增了 `causal_conv1d_update_varlen`，仍没有所导入的 `causal_conv1d_update_npu`。
   不能以“升级后出现同名函数”为解决方案，更不能把两个签名不同的函数直接重命名替换。
2. 0.32 已有 `PIECEWISE`：platform 注册 `vllm::glm5next_kda_forward_with_output` 切分点，
   KDA 的 `_needs_output_buffer` 选择固定输出 buffer，切分 op 有 `eager_break_during_capture`。
   这可让 KDA 在图外执行，其他区域使用图；`FULL_DECODE_ONLY` 不会按这个边界切分。
3. 上游 vLLM 0.32 `GPUModelRunner.profile_run` 中，`skip_mm_profiling` 只控制视觉 encoder
   cache profile，后续 text dummy run 和图捕获仍执行。它可用于 GPQA 文本场景隔离视觉 profile，
   不会使一个不支持图捕获的卷积变得支持；图文服务的显存预算需另行验收。

## 本次代码修复

- 删除不存在的可选 Triton 符号导入和据此输出的 graph-hang WARNING，修正关联 patch 说明。
- 在每个 Worker 的 `kv_cache_spec`、`memory_profile`、`kv_cache_allocate`、
  `compile_and_capture` 四个入口输出 rank、PID、图模式及开始/结束时间。只读取 CPU 信息，
  不增加 NPU 同步，不改变生成阶段的计算。
- 增加绑定和 Worker 栈采集脚本，以及不加载模型的原生 AscendC 卷积 eager/ACLGraph 数值测试。

删除 WARNING 只修复错误诊断，不等于证明启动卡死已消失。真正阻塞点以 Worker 栈和新增阶段日志为准。

## 当前评测完成后再操作

保持正在运行的 eager GPQA 评测。本次本地修改不连接或重启节点服务。
以下命令由用户在本轮评测结束后，在本人的容器中执行。

### 1. 核对安装绑定

```bash
cd /vllm-workspace/vllm-ascend
python examples/diagnose_glm5next_graph.py \
  --output /mnt/share/w30015592/tmp/glm53_graph_bindings.json
```

确认 JSON 中 `bindings` 的 GLM KDA 卷积指向模型 wrapper，该 wrapper 的 update 指向 `fla_npu`；
若实际走 generic fallback，JSON 的绑定和调用源码行会显示这一点，需要按该实现继续修复。
输出同时记录插件 HEAD、未提交文件、vLLM / Torch / FLA 版本和实际安装路径。
这是新解释器的源码/绑定检查，不能代替现有 Worker 的 live stack。
如果这些绑定与分析不一致，先使用该输出对齐容器实际代码。

### 2. 先测试卷积和状态回放

不要与占用同一卡的评测同时运行。以下 8 组用例包含 dim-last / dim-first、两个通道尺寸和
eager / capture，每次回放前改变输入、恢复初始 state，并与独立 CPU 卷积及 state 更新比较。

```bash
cd /vllm-workspace/vllm-ascend
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=10s 180s \
  python -m pytest -q -v tests/e2e/pull_request/one_card/test_glm5next_conv_aclgraph.py \
  2>&1 | tee /mnt/share/w30015592/tmp/glm53_conv_graph_probe.log
```

超时、失败和 skip 均不能算图模式验收通过。
只有 capture 用例失败，才直接证明当前 FLA/CANN 的这个卷积/状态布局捕获存在问题；
测试通过也不能证明整个模型或多卡 HCCL 图捕获正确。

### 3. GPQA 文本场景先用 PIECEWISE

关闭本人旧服务后再运行。当前示例仍为 16384 上下文，以小捕获集合减少启动变量；
`[4]` 兼容 TP4 的 padding，先验收再扩大 batch 和 capture sizes。
CLI 生成参数的 8192 输出长度仍由客户端/AISBench 控制。

```bash
cd /workspace
MODEL_PATH=/mnt/share/w30015592/models/GLM-5.3-Flash-A5-W4A8-CT
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL_PATH" \
  --served-model-name glm53-ct-a2 --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype bfloat16 --quantization compressed-tensors \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --kv-cache-dtype int8 --attention-config.indexer_kv_dtype int8 \
  --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --gpu-memory-utilization 0.85 \
  --skip-mm-profiling --reasoning-parser glm47 \
  --compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4]}' \
  --additional-config '{"enable_fused_mc2":0,"weight_nz_mode":2}' \
  2>&1 | tee /mnt/share/w30015592/tmp/vllm_a2_piecewise.log
```

这里没有 `--enforce-eager`。若现有成功命令的 cache、EP、MTP 或额外参数不同，
先保留成功基线，只改变图模式和 profiling 开关进行对照，避免混入其他配置变化。
显存利用率以实际空闲 HBM 为准，不因卡住而直接提高它。

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-ct-a2","messages":[{"role":"user","content":"计算2+3，用中文简短回答。"}],"temperature":0,"max_tokens":256}'
```

需同时核对实际 PIECEWISE 捕获/回放日志、首请求完成、文本可读性及同配置 GPQA 精度。
不能仅凭没有 WARNING、HTTP 200 或 EngineCore 存活认定成功。

### 4. 如仍停住，采集 Worker 栈

从当前容器的 `ps -eo pid,ppid,comm,args` 找到这轮运行的四个 Worker PID；报告中的旧 PID 不可直接复用。
容器已有 `py-spy` 时，脚本对显式传入的 PID 执行只读 dump，并读取 `/proc` 的 comm/wchan/stack。

```bash
ps -eo pid,ppid,comm,args | grep -E 'Worker_TP|VLLM::Worker|EngineCore'
# 下方数字仅演示参数格式，替换为本轮四个 Worker PID。
python /vllm-workspace/vllm-ascend/examples/diagnose_glm5next_graph.py \
  --pids 31036 31037 31038 31039 \
  --output /mnt/share/w30015592/tmp/glm53_graph_worker_stacks.json
```

若缺少 `py-spy`，输出会明确标记缺失，不能把 `/proc/wchan` 当成 Python 栈。
需要时在本人容器安装 `python -m pip install py-spy`，然后重跑采集脚本。

| 最后一个只有 begin 的阶段 | 后续定位 |
| --- | --- |
| `kv_cache_spec` | cache-spec / 混合页大小规划；尚不能判断已捕获图 |
| `memory_profile` | 视觉 profile、text dummy run、编译或同步；比较 skip-mm-profiling 和 Worker 栈 |
| `kv_cache_allocate` | KV/SSM 分配、connector 和 cache 初始化 |
| `compile_and_capture` | warmup、自定义 kernel 预热、真实捕获；需结合捕获开始日志与 Worker 栈 |

## 验证边界

本次只能运行本地 CPU 回归和语法/格式检查；NPU 卷积、PIECEWISE、FULL_DECODE_ONLY、
真实权重首请求及 GPQA 图/eager 精度对照均未在节点执行。
本地 7 项 CPU 回归通过，9 个 Python 文件编译和固定 Ruff 检查通过；
详细文件校验值见[验证记录](GLM5.3-Flash-CT-W4A8-A2-Graph-verification.json)。
完整 `format.sh ci` 因缺少 pre-commit 未完成。
现有 eager GPQA 的 115.2 tok/s 是用户提供的评测进度，未作为本补丁图模式的性能结果。
FULL_DECODE_ONLY 的最终阻塞根因仍需当前 Worker 栈确认，未擅自更改其语义或宣称硬件不支持。
