# GLM-5.3-Flash CT：A2 TP8 临时启动优化

2026-11-19。基于 `d90f0661c` 的 MLP-only 适配，仅追加新提交，不改历史。
面向 8×910B3、TP=8 的验证方案；不要附加 `--enable-expert-parallel`。
若实际硬件为 A5，CPU 格式转换仍可单独验证，但 A5 K-CG 的数值和服务验收不能由本文替代。

## 为什么增加卡数仍然慢

纯 TP 的每张卡需要所有专家的一部分权重。当前流程在原生 TP 切分之前做完整专家矩阵的
CT 解包与重新打包，因此 TP8 仍可能重复执行八份 CPU 转换，并竞争共享存储。
不能通过按 rank 筛掉专家来加速纯 TP；那会丢失必要权重。

这次临时优化包含：

- CT 的 `q+8` 编码直接转换成沿输出维打包的 signed INT4 字节，
  避免生成完整 `[N,K]` INT8 中间矩阵。每个码和原 scale 保持不变。
- 主模型和 MTP 的专家映射按逻辑专家 ID 建索引，保留原始候选顺序和副本。
  非标准映射退回原有遍历。
- 原生 TP rank、切分、gate/up 融合、scale 加载、逐专家修正项和 NZ/CANN 路径保持原样。

这没有提前切 TP 分片，也没有减少所有 checkpoint 字节的读取；
只有 CPU 转换部分变快时，整体启动才会缩短。不能将其描述为八倍启动加速。

## 开关与验证

`VLLM_ASCEND_CT_FAST_LOAD` 在 `vllm_ascend/envs.py` 注册。
取值 `0`：默认关闭；取值 `1`：启用直接 CPU 打包与专家映射索引。非敏感配置。
关闭后恢复原加载算法，无需修改模型权重或回退历史 commit。
这是临时启动优化开关，合入上游时应评估是否移除开关并统一加载实现。

```bash
cd /vllm-workspace/vllm-ascend
VLLM_ASCEND_CT_FAST_LOAD=1 python -m pytest -q \
  tests/ut/quantization/test_compressed_tensors_fast_load.py \
  tests/ut/quantization/test_compressed_tensors_packed.py \
  tests/ut/quantization/test_w4a8_loading_memory.py
```

部署前在 NPU 镜像运行已有的单卡整数数值测试。CPU 字节等价不能替代此项验收。

```bash
cd /vllm-workspace/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_ASCEND_CT_FAST_LOAD=1 \
python -m pytest -q tests/e2e/pull_request/one_card/test_w4a8_a2_numeric.py
```

## TP8 启动命令

先选择已重新生成的 MLP-only checkpoint。示例使用本机 NVMe 路径；
需先将完整模型复制到该目录，并核对文件总量和 index 中的所有分片。
复制期间不要使用目标目录启动。`/mnt/share` 是否为网络存储可用 `findmnt -T` 确认。

```bash
set -euo pipefail
MODEL_PATH=/local_nvme/models/GLM-5.3-Flash-MLP-W4A8-CT
findmnt -T "$MODEL_PATH"
mkdir -p /mnt/share/w30015592/tmp

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_CT_FAST_LOAD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

vllm serve "$MODEL_PATH" \
  --served-model-name glm53-ct-a2-tp8 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype bfloat16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 8 \
  --safetensors-load-strategy lazy \
  --kv-cache-dtype int8 --attention-config.indexer_kv_dtype int8 \
  --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --gpu-memory-utilization 0.85 \
  --skip-mm-profiling --enforce-eager --reasoning-parser glm47 \
  --additional-config '{"enable_fused_mc2":0,"weight_nz_mode":2}' \
  2>&1 | tee /mnt/share/w30015592/tmp/glm53_a2_tp8_fast_load.log
```

这条命令是待 NPU 实测的模板。首次先验证 eager；已有的图捕获阻塞需要独立诊断。
CPU 线程数设为 1 是避免八 worker 过度并行的起点；若容器已经限制为 1，不会额外提速。
若必须读取网络文件系统，可保持其余配置不变，对照测试
`--safetensors-load-strategy eager`。它会读取整片到 CPU 内存，八 worker 并发时增加 RAM 峰值；
先检查主机可用内存，不要直接开启多个加载线程或扩大 CPU 线程池。
加载策略语义参见 [vLLM 0.32 官方参数](https://docs.vllm.ai/en/v0.32.0/cli/serve/#--safetensors-load-strategy)。

## 对照与边界

只改 `VLLM_ASCEND_CT_FAST_LOAD=0/1`，其余硬件、checkpoint、路径、参数不变。
两次均为冷缓存，或两次均为热缓存；不要把首次网络冷读与第二次 page-cache 命中当作代码收益。
不要为测试清空整台共享机器的文件缓存。

```bash
grep -nE 'CT fast load|Loading safetensors|Model loading took|CT W4A8|Ascend startup' \
  /mnt/share/w30015592/tmp/glm53_a2_tp8_fast_load.log
```

预期出现 `CT fast load enabled`。分别记录模型加载、repack、profile 和缓存分配时间。
比较输出质量后，再运行既有 ais_bench 配置；不能仅凭 HTTP 200 或服务 ready 判定正确。
CPU 测试覆盖码值、补齐列、非连续输入、scale/kwargs、映射副本顺序，以及原生 loader 的 TP8。
真实权重 TP8、NPU 峰值内存、启动耗时、C8 与评测结果仍需节点验证。

若瓶颈仍是八 worker 读取和处理完整源矩阵，后续应实现带元数据的离线 TP 分片加载。
这要求 loader 明确识别已分片权重，避免二次切分，不属于本临时补丁。
