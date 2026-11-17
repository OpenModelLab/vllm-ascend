# GLM-5.3-Flash CT W4A8C8：A2 / v0.32

## 1. 适配范围

模型源为 `zai-org/GLM-5.3-Flash-BF16`，使用现有 llm-compressor INT 混合产物：
routed experts 为 INT4 权重、动态 token INT8 激活，shared 和其他线性投影为 W8A8。
主 DSA cache 与 KPool indexer cache 可分别选择 INT8。动态 C8 不需要重新导出 checkpoint。

本仓库从官方 `releases/v0.32.0rc` 克隆，基线为
`2dd5b2b2d2e22995821efeee8b0b265fcba5b612`；官方没有 `releases/v0.32.0` 分支。
配套上游 vLLM 为 `.github/vllm-release-tag.commit` 指定的 `v0.32.0`，
本次源码验证固定 commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。

迁移来源为已有 main 适配 `f2d258dc00b54549f986d6bdffb7160ae6091355`，
包含旧 0.30 分支从 `528748a779ed62d150cf560bb55abf21883b8d66` 到
`52f9ae8238435016263c4ef91b31c6025ca5133f` 的累积修复。

## 2. 实现和执行边界

| 部分 | 本次实现 |
| --- | --- |
| Routed experts | CT K 轴 offset-binary 解包，转成 N 轴紧凑 signed INT4；原生 loader 做 TP/EP 分片，保留原 scale |
| 显存峰值 | 权重保持 INT4 常驻；逐专家计算校正 bias，避免整层 FP32 临时矩阵 |
| A2 算子 | `is_950()` 为 false，走 A2/A3 原生 A8W4 GMM、INT64 编码 scale、校正 bias；显式 NZ 转换 |
| GMM1 scale | 单个 3D 权重使用 `[E,1,N]`；专家列表保留全部 `[1,N]` scale，避免遗漏专家或误作量化组 |
| SiLU | CT 实例使用分离 GMM1 → 原激活和 clamp → 动态 A8 → GMM2；ModelSlim 融合能力保留 |
| W8A8 | 融合 QKV、KDA、shared、router、LM head、视觉、KPool、MTP 接入原生 scale 加载和计算 |
| mHC | 保留裸浮点参数及原 FP32 执行；不创建不存在的 `.weight_scale` 槽位 |
| KV-b | 沿用已批准的例外：用现有 INT8 码乘原 scale 分块恢复 BF16，供 MLA 吸收 |
| C8 | 复用 0.32 已有 SFA INT8 和 KPool INT8 路径，不从 A5 移植 HiF8/MXFP4 算子 |

A2 不调用 A5 K-CG，不使用 format 53 或 ND→NZ 私有 storage 拷贝。
保留的 A5 helper 由设备检测隔离，不代表本次验证了 A5。
0.32 KPool 查询使用完整 batch，未移入 main 的 query-row-shard 接口；
SFA 保留 0.32 的 `replace_parameter(..., prefer_copy=True)` 兼容分支。

不要求重新量化。名称中带 `A5-W4A8-CT` 的现有 **INT** 产物可以作为 A2 验证输入；
名字中的 A5 不会把 INT4 码变成 MXFP4。A5 MXFP4/MXFP8、HiF8 目标不在本次范围。

## 3. 环境准备

在 A2 Linux 节点使用该分支的根目录 `Dockerfile`，不要使用 `Dockerfile.a5`。
根 Dockerfile 以 CANN 9.2.1 910B 为基础，并安装分支指定的 910B ops-transformer 9.3.0 包。
仅查看 `npu-smi` 版本不足以确认 PoolKeyIndexer INT8、KDA 和自定义算子 ABI 配套。

将本地仓库同步至 A2 的 `/mnt/share/w30015592/A2-vllm-ascend-0.32.0` 后构建：

```bash
set -euo pipefail
cd /mnt/share/w30015592/A2-vllm-ascend-0.32.0
git status --short
docker build -f Dockerfile \
  --build-arg VLLM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be \
  -t glm53-ct-a2:v032 .
```

`SOC_VERSION` 采用官方 Dockerfile 默认 `ascend910b1`；具体 A2 SKU 不同应使用与节点一致的构建值。
在专用容器中映射所用 NPU、驱动及 `/mnt/share`。镜像已复制并 editable 安装本次代码。
若在已有配套 0.32 A2 开发容器中安装，命令如下：

```bash
cd /vllm-workspace/vllm-ascend
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python - <<'PY'
import importlib.metadata as md
import vllm, vllm_ascend, torch, torch_npu
import cann_ops_transformer
print('vllm', md.version('vllm'), vllm.__file__)
print('ascend', vllm_ascend.__file__)
print('torch', torch.__version__, 'torch_npu', md.version('torch-npu'))
print('CT', md.version('compressed-tensors'))
print('device', torch.npu.get_device_name(0))
PY
```

不要把 0.32 插件装入原 0.30 vLLM 环境。本次不修改 llm-compressor、compressed-tensors 或原有三个代码仓。

## 4. 先做算子数值验收

以下命令在完整 A2 容器内运行；`ASCEND_RT_VISIBLE_DEVICES=0` 仅限用户分配的卡。
新测试 8 个参数组合覆盖 packed/expanded、counts/cumulative、512/4096 输入维度，
有中间空专家，并比较 CPU 整数点积、GMM1、SiLU/clamp 和 GMM2。

```bash
cd /vllm-workspace/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/e2e/pull_request/one_card/test_w4a8_a2_numeric.py -v
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/e2e/pull_request/one_card/test_ct_w4a8_loading.py \
  tests/e2e/pull_request/one_card/test_glm5next_ct_linears.py \
  tests/e2e/pull_request/one_card/test_glm5next_ct_special_projections.py
# KPool INT8 管理页、scale 和真实算子，包含 eager/graph 用例。
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py -v
```

数值测试失败须保留错误并定位；不能放宽容差或删除断言后以 HTTP 200 代替验收。
被 skip 的 INT8 算子测试不计为通过。

## 5. 真实模型启动

以下以 8 张已分配 A2 卡为示例，沿用用户现有 INT checkpoint 路径；修改卡数时同步修改 TP。
实机 SKU、HBM 和卡数尚未提供，因此此命令是待实测的容量起点，不是已测最优配置。
约 155 GiB checkpoint 不能直接塞进 2×64 GB；8 卡静态权重平均约 19.4 GiB/卡，
实际还需加复制权重、MLA 吸收、KV/SSM cache、工作区和通信 buffer，不能以文件体积证明不会 OOM。

先在 BF16 cache 下隔离 W/A 计算，再启用 C8。NoPE GLM 的 BF16 main-cache 会进入原生链路，
速度及可用上下文与 C8 不同；若缓存阶段失败，应记录失败，不跳过权重数值验收。

```bash
cd /workspace
MODEL_PATH=/mnt/share/w30015592/models/GLM-5.3-Flash-A5-W4A8-CT
test -f "$MODEL_PATH/model.safetensors.index.json"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL_PATH" \
  --served-model-name glm53-ct-a2 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype bfloat16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype bfloat16 --attention-config.indexer_kv_dtype bf16 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --reasoning-parser glm47 \
  --additional-config '{"enable_fused_mc2":0,"weight_nz_mode":2}'
```

在通过文本验收后，停掉本人这个前台服务，用同一命令只替换下面两个 cache 参数再启动：

```bash
--kv-cache-dtype int8 --attention-config.indexer_kv_dtype int8
```

这是主 DSA INT8 与 KPool INT8 两个独立开关；`--kv-cache-dtype int8` 本身不表示 KPool 已为 INT8。
先保持 eager、EP 和短上下文。再分别测试移除 `--enforce-eager`、flashcomm1、MTP、图文和长上下文，
不要把多个未验证功能一次叠加来判断 W4A8 数值。

## 6. 请求与评测

在另一个终端执行，核对 `content` 和 `reasoning_content` 的可读性，乱码视为失败：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-ct-a2","messages":[{"role":"user","content":"请计算 2+3，并用中文回答。"}],"temperature":0,"max_tokens":256}'
```

AISBench 使用用户指定的 `ais_bench_benchmark==3.1.20260630`，先确认版本，
沿用官方 `vllm_api_general_chat` 配置，将 `host_ip`、`host_port`、`model`、`path` 对齐本服务。
在相同数据集、prompt、生成设置下对照 BF16 和量化产物，并单独比较 cache BF16/INT8：

```bash
cd /mnt/share/w30015592/benchmark
python -c "from importlib.metadata import version; print(version('ais_bench_benchmark'))"
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt
```

该命令不带配置文件末尾的点；数据集需提前准备。未实测精度，不生成含虚构准确率的模型验收 YAML。

## 7. 本地验证和剩余验收

CPU 验证基于固定的 vLLM 0.32 原生源码，非工作区旧 vLLM。独立加载器测试覆盖 26 个 TP/EP/padding/compact
参数组合，投影测试覆盖 mHC 270 个裸浮点键以及原始 scale 的加载和执行。模拟设备的测试不证明 CANN 结果正确。
具体测试日志、结果及文件校验值见[验证记录](GLM5.3-Flash-CT-W4A8-A2-verification.json)和交付仓的 evidence。
本地 99 个 CPU 测试项通过；其中加载器的 2 个测试项内部覆盖上述 26 组参数组合。
固定 Ruff 0.14.0、Python 编译和 Markdownlint 通过；完整 `format.sh ci` 因缺少 pre-commit 未完成。

| 功能 | 本次状态 |
| --- | --- |
| CPU 加载、码值、scale、内存转换、GMM 参数契约 | 已运行；详见验证记录 |
| A2 GMM1/激活/GMM2 及 C8 算子 | 已提供实机测试，当前无 NPU，未执行 |
| dummy / 真实模型请求、图文 | 未执行；dummy 不能替代真实权重验收 |
| EP、ACLGraph、flashcomm1、MTP | 保留代码接口，待 A2 实机分别验证 |
| 128k + batch16、吞吐及精度 | 无硬件容量及测量结果，未执行 |
| A3 等价性 | A2 可验证功能链路，不能替代 A3 通信和性能验收 |

当前交付为代码迁移和本地回归完成，真实模型部署验收待 A2 实机结果。
