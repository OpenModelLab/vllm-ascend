# GLM-5.3-Flash CT W4A8C8：A2 / v0.32

## 1. 适配范围

模型源为 `zai-org/GLM-5.3-Flash-BF16`，使用现有 llm-compressor INT 混合产物：
routed experts 为 INT4 权重、动态 token INT8 激活，dense/shared MLP 为 W8A8，非 MLP 保持浮点。
主 DSA cache 与 KPool indexer cache 可分别选择 INT8。动态 C8 不需要重新导出 checkpoint。

2026-11-19 起使用从 BF16 重新生成的 MLP-only 配方，旧宽范围产物不再由当前 CT INT 入口加载。
本次精简、浮点 KPool 双名称兼容和最新验收说明见
[MLP-only 教程](GLM5.3-Flash-CT-W4A8-A2-MLP-only.md)。下方迁移提交记录为历史来源，未改写。

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
| W8A8 | dense/shared MLP 接入原生融合、scale 加载和计算；非 MLP 走原生浮点实现 |
| mHC | 保留裸浮点参数及原 FP32 执行；不创建不存在的 `.weight_scale` 槽位 |
| KV-b | 新产物保留原始浮点权重，使用原生 MLA 吸收；已撤掉旧产物的 INT8 恢复分支 |
| C8 | 复用 0.32 已有 SFA INT8 和 KPool INT8 路径，不从 A5 移植 HiF8/MXFP4 算子 |

A2 不调用 A5 K-CG，不使用 format 53 或 ND→NZ 私有 storage 拷贝。
保留的 A5 helper 由设备检测隔离，不代表本次验证了 A5。
0.32 KPool 查询使用完整 batch，未移入 main 的 query-row-shard 接口；
SFA 保留 0.32 的 `replace_parameter(..., prefer_copy=True)` 兼容分支。

使用已重新量化的 MLP-only **INT** 产物。运行量化时的设备名称不决定权重格式；
新 loader 会拒绝旧宽范围产物的非 MLP INT 权重/scale。A5 MXFP4/MXFP8、HiF8 目标不在本次范围。

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
  --build-arg SOC_VERSION=ascend910b3 \
  --build-arg VLLM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be \
  -t glm53-ct-a2:v032 .
```

用户已确认节点为 **Ascend 910B3，64 GB × 4**；构建显式使用 `ascend910b3`，
不沿用 Dockerfile 默认的 `ascend910b1`。仓库硬件映射及 A2 自定义算子构建分支已包含该型号。
在专用容器中映射所用 NPU、驱动及 `/mnt/share`。镜像已复制并 editable 安装本次代码。
若在已有配套 0.32 A2 开发容器中安装，命令如下：

```bash
cd /vllm-workspace/vllm-ascend
SOC_VERSION=ascend910b3 python -m pip install --no-deps --no-build-isolation -e .
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

使用已确认的 **910B3，64 GB × 4，TP=4 + EP**，路径改为新 MLP-only checkpoint 的实际目录。
NPU 0–3 是命令示例；若分配了其他卡，只替换可见设备列表，TP 仍为 4。
EP 开启后 routed experts 按专家分配，其他可分片投影使用 TP；不是把每类权重一律除以 4。

以下 155 GiB 为此前旧宽范围产物的历史估算，新 MLP-only 产物需按实际文件重新计算。
按此前报告的约 155 GiB 产物粗估，平均静态权重约 **38.75 GiB/卡**。
若运行时每卡显示 64 GiB，`--gpu-memory-utilization 0.85` 的目标预算约 54.4 GiB/卡，
相减约 15.65 GiB；这部分还需容纳不均匀/复制权重、MLA 吸收、KV/SSM cache、工作区、
通信 buffer 和加载转换峰值，不能全部算作 KV cache。该参数也不是加载期显存硬上限。
因此四卡可作为实测起点，尚不能保证不会 OOM；64 GB 的实际字节口径以运行时容量为准。
如果使用的是此前提到的 200 GB+ 另一份产物，需重新计算，不能直接沿用 155 GiB 估算。

启动前检查当前产物的实际大小，下面是粗略均摊值，不是运行时峰值预测：

```bash
MODEL_PATH=/mnt/share/w30015592/models/GLM-5.3-Flash-MLP-W4A8-CT
python - "$MODEL_PATH" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / 'model.safetensors.index.json').read_text())
files = sorted(set(index['weight_map'].values()))
size = sum((root / name).stat().st_size for name in files)
print(f'{len(files)} shards; {size / 2**30:.2f} GiB; file-average/4 = {size / 2**30 / 4:.2f} GiB')
PY
```

先在 BF16 cache 下隔离 W/A 计算，再启用 C8。NoPE GLM 的 BF16 main-cache 会进入原生链路，
速度及可用上下文与 C8 不同；若缓存阶段失败，应记录失败，不跳过权重数值验收。

```bash
cd /workspace
MODEL_PATH=/mnt/share/w30015592/models/GLM-5.3-Flash-MLP-W4A8-CT
test -f "$MODEL_PATH/model.safetensors.index.json"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL_PATH" \
  --served-model-name glm53-ct-a2 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype bfloat16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 4 --enable-expert-parallel \
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
若权重已成功加载而 profiling/cache 分配 OOM，先将 `--max-model-len` 和
`--max-num-batched-tokens` 同时减至 2048，保持 batch1/eager；再记录各 rank 的
`before_repack` / `after_repack` 和可用 HBM。加载转换阶段 OOM 需定位具体 rank、投影和分配量，
缩短上下文并不能保证解决。确认通信、复制和峰值空间后再考虑调整显存利用率。

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
| 128k + batch16、吞吐及精度 | 已确认 4×64 GB，尚无实测容量及测量结果 |
| A3 等价性 | A2 可验证功能链路，不能替代 A3 通信和性能验收 |

当前交付为代码迁移和本地回归完成，真实模型部署验收待 A2 实机结果。
