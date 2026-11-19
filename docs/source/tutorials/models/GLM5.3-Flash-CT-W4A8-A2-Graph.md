# GLM-5.3-Flash A2：图模式启动诊断与分段配置

2026-11-18。环境为 910B3 64 GB × 4、TP4，镜像 `v0.32.0rc1-openeuler`，
用户报告插件基线 `24487f4f4`，模型为已有 CT INT W4A8 产物。

2026-11-19 后的精简版本使用重新生成的 MLP-only CT checkpoint；
旧宽范围产物对应此前服务修订。见[当前适配范围](GLM5.3-Flash-CT-W4A8-A2-MLP-only.md)。
以下原生 FLA 诊断和图模式配置仍适用，旧 eager 结果不作为新模型的图模式验收。

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
以上调用链已在本地官方 `24487f4f4` 源码核实；用户最新返回的容器 `git ls-tree` 和 grep
也已确认这一调用链。此前包含 `modeling_glm5next.py`、`ops/layers/attention/kda.py`
的报告已被用户撤回。当前无需实现不存在的通用 Triton 函数。

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

## 对实际 FLA 路径的进一步检查

本次另行只读克隆了官方 FLA `v26.9.1-release`，固定提交
`5cc699069c7f91a35aabd583032c8720a0b626fa`。这是镜像 Dockerfile 指定版本对应的源码，
用户随后返回的实际包版本也是 `26.9.1`，公开函数签名包含 `max_query_len=-1`。
`inspect.getsource()` 因 `functools.wraps` 返回分发层 `_get_direct_op`，不能据此认定已经看到了
实际选中的 Stable/ctypes 函数体。版本相同也不代替安装文件 hash 核对；没有安装或升级任何依赖。

| 检查位置 | 源码结论 |
| --- | --- |
| GLM 原生卷积 wrapper | 没有 `.item()` / `.cpu()` / `.tolist()`；状态布局判断使用 shape/stride |
| FLA ctypes 卷积 update | `validate_data=False` 为默认值，读取设备元数据的 helper 只在数据校验分支调用；不能把文本搜索命中当成默认同步 |
| FLA Stable ABI update/recurrent KDA | 未见这两个 Python 入口直接读取设备 tensor 数值；stream 获取可能经过任务队列 barrier，仍需实际版本和 Worker 栈判断 |
| GLM KDA、SFA、indexer | 找到的 `bool(...is_prefilling_np.any())` 使用 NumPy 元数据；没有定位到每请求设备 `.item()` 的证据 |
| GDN metadata builder | 多处 `.item()` / `.tolist()` 使用 CPU draft tokens / query starts；构造 metadata 与图内算子执行要区分 |
| Mamba hybrid `num_scheduled_tokens.max().item()` | 上游字段类型是 `np.ndarray`，属于 CPU 标量提取，不能当作 NPU 同步 |

发现并修复一个独立的 FLA 接口问题：26.9.1 的 ctypes update 在传入 `query_start_loc` 时要求
非负 `max_query_len`，而原 GLM wrapper 未传。Stable 路径也把该上界传给 tiling。
新 wrapper 在导入时检查函数签名，仅在支持时传参，旧接口保持原调用。
KDA 普通 decode 传 `1`，MTP 验证传 `self.num_spec + 1`，都是已有 CPU 配置值。
独立调用未指定上界时使用输入行数；如果带 accepted tokens，则再用卷积 state 长度和 kernel width
推导每请求 rollback 容量上界。不能直接对 MTP 使用整批 token 数，否则 FLA 会错误要求每个请求
都分配整批长度的历史槽。没有用 `query_start_loc.max().item()` 获取长度。
它修复确定的接口风险，尚不能据此认定当前 eager 可用、FULL 卡住的直接原因就是漏传参数。

`profile_forward` 和 `graph_capture` 两个 MRV2 阶段已另加开始/结束日志。
若停在 `profile_forward`，内存 profile 的 `skip_attn=True` 会使 KDA core 在缺少 metadata 时直接归零，
此时优先检查投影、MoE、视觉 profile、编译和集合通信，而不是预设已经调用了 decode 卷积。
FLA Python 没有显式同步也不能证明其 CANN kernel、任务队列或多卡路径支持 capture。

## 当前评测完成后再操作

保持正在运行的 eager GPQA 评测。本次本地修改不连接或重启节点服务。
以下命令由用户在本轮评测结束后，在本人的容器中执行。

### 1. 核对安装绑定

先执行纯源码采集，不导入 Torch、FLA 或模型，不创建 NPU context；可在当前 GPQA 继续运行时执行：

```bash
cd /vllm-workspace/vllm-ascend
python examples/diagnose_glm5next_graph.py --source-only \
  --output /mnt/share/w30015592/tmp/glm53_fla_sources.json \
  > /mnt/share/w30015592/tmp/glm53_fla_sources.stdout.json
```

输出包括已安装 FLA distribution 的版本、Python 源码 SHA256、关键函数正文和潜在 host-read 行，
保留校验分支上下文。这是静态候选清单，不能把某个 `.cpu()` 命中直接判为活跃捕获路径。
脚本记录的 FLA 环境变量属于新进程，不代替原 Worker 的实际环境。
若 wheel 没有文件清单或 Python 源码，输出为空/错误，不伪造调用结果。

评测结束后再做会导入实际设备库的绑定检查：

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
  python -u -m pytest -q -v -s tests/e2e/pull_request/one_card/test_glm5next_conv_aclgraph.py \
  2>&1 | tee /mnt/share/w30015592/tmp/glm53_conv_graph_probe.log
```

超时、失败和 skip 均不能算图模式验收通过。
探针输出真实执行后 FLA 的 `BACKENDS` / `FALLBACKS`（如果该安装提供这些字段）以及 warmup/capture 边界；
可区分首次 native launch 就停住与仅 capture 停住，后者才继续针对捕获检查。
只有 capture 用例失败，才直接证明当前 FLA/CANN 的这个卷积/状态布局捕获存在问题；
测试通过也不能证明整个模型或多卡 HCCL 图捕获正确。

### 3. GPQA 文本场景先用 PIECEWISE

关闭本人旧服务后再运行。当前示例仍为 16384 上下文，以小捕获集合减少启动变量；
`[4]` 兼容 TP4 的 padding，先验收再扩大 batch 和 capture sizes。
CLI 生成参数的 8192 输出长度仍由客户端/AISBench 控制。

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
| `profile_forward` | MRV2 的实际 profile 入口；尚不能认定已进入图捕获 |
| `kv_cache_allocate` | KV/SSM 分配、connector 和 cache 初始化 |
| `compile_and_capture` | warmup、自定义 kernel 预热、真实捕获；需结合捕获开始日志与 Worker 栈 |
| `graph_capture` | MRV2 capture 入口已进入，结合卷积探针和 Worker 栈定位具体算子/通信 |

## 验证边界

本次只能运行本地 CPU 回归和语法/格式检查；NPU 卷积、PIECEWISE、FULL_DECODE_ONLY、
真实权重首请求及 GPQA 图/eager 精度对照均未在节点执行。
本地 11 项 CPU 回归通过，12 个 Python 文件编译和固定 Ruff 检查通过；
详细文件校验值见[验证记录](GLM5.3-Flash-CT-W4A8-A2-Graph-verification.json)。
完整 `format.sh ci` 因缺少 pre-commit 未完成。
现有 eager GPQA 的 115.2 tok/s 是用户提供的评测进度，未作为本补丁图模式的性能结果。
FULL_DECODE_ONLY 的最终阻塞根因仍需当前 Worker 栈确认，未擅自更改其语义或宣称硬件不支持。
