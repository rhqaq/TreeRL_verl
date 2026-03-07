# TreeRL Quick Start Guide

## 快速开始

### 1. 准备数据

确保数据包含 `ground_truth` 和 `data_source` 字段：

```python
# 示例：GSM8K 数据格式
import pandas as pd

data = pd.DataFrame({
    "prompt": [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        # ... more examples
    ],
    "ground_truth": ["72", ...],  # 正确答案
    "data_source": ["openai/gsm8k", ...],  # 数据来源
})

data.to_parquet("data/gsm8k/train.parquet")
```

### 2. 选择启动方式

#### 选项 A：单机快速测试

```bash
# 使用简化版脚本
bash recipe/treerl_v2/scripts/train_treerl_simple.sh
```

#### 选项 B：分布式训练（Ray 集群）

```bash
# 设置环境变量
export TRAIN_FILE="data/gsm8k/train.parquet"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export NNODES=2  # 节点数量

# Master 节点执行
export VC_TASK_INDEX=0
bash recipe/treerl_v2/scripts/train_treerl.sh

# Worker 节点执行
export VC_TASK_INDEX=1
export MASTER_ADDR="<master_ip>"
bash recipe/treerl_v2/scripts/train_treerl.sh
```

#### 选项 C：配置文件 + 命令行

```bash
# 使用配置文件
python -m recipe.treerl_v2.main_treerl \
    --config-path=recipe/treerl_v2/config \
    --config-name=treerl_config

# 或覆盖参数
python -m recipe.treerl_v2.main_treerl \
    data.train_files=data/gsm8k/train.parquet \
    algorithm.m=8 \
    algorithm.n=3 \
    trainer.total_epochs=20
```

### 3. 调整树搜索参数

TreeRL 的核心是树搜索参数，根据任务复杂度调整：

| 任务复杂度 | m | n | l | t | num_traces | 说明 |
|-----------|---|---|---|---|------------|------|
| 简单任务 | 4 | 1 | 0 | 1 | 8 | 快速验证 |
| 中等任务 | 6 | 2 | 1 | 2 | 16 | 默认配置 |
| 复杂任务 | 8 | 3 | 2 | 3 | 32 | 最佳性能 |

**参数含义**：
- `m`: 初始树的数量（每个 prompt 生成 M 个初始响应）
- `n`: 每轮扩展选择 N 个最高熵的 token
- `l`: 扩展迭代轮数
- `t`: 每个熵点生成 T 个分支
- `num_traces`: 最终采样的训练轨迹数

### 4. 监控训练

```bash
# 查看训练日志
tail -f /tmp/treerl/treerl_gsm8k_7b/train.log

# TensorBoard
tensorboard --logdir /tmp/treerl/rl_ckpts/treerl
```

### 5. 检查点管理

```bash
# 保存的检查点位置
ls /tmp/treerl/ckpts/treerl/treerl_gsm8k_7b/

# 恢复训练
python -m recipe.treerl_v2.main_treerl \
    trainer.resume_mode=auto \
    trainer.default_local_dir=/tmp/treerl/ckpts
```

## 常见问题

### Q: 如何支持新的数据源？

A: 在 `verl/utils/reward_score/__init__.py` 中添加：

```python
elif data_source == "my_dataset":
    from . import my_scorer
    res = my_scorer.compute_score(solution_str, ground_truth)
```

### Q: 内存不足怎么办？

A: 降低树搜索规模或启用 offload：

```bash
export m=4  # 减少初始树数量
export num_traces=8  # 减少采样轨迹

# 或启用 offload
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### Q: 如何调整熵引导策略？

A: 修改 `n` 参数（选择多少个高熵 token）：

- `n=1`: 只选择最高熵的 token
- `n=2`: 选择 top-2 高熵 token
- `n=3`: 选择 top-3 高熵 token（更激进的探索）

### Q: 如何禁用 KL 惩罚？

A: 设置环境变量：

```bash
export use_kl_in_reward=False
export use_kl_loss=False
```

## 性能建议

1. **GPU 内存优化**：
   - 使用 `gen_tp=2` 或更高（张量并行）
   - 启用 `enable_chunked_prefill=True`
   - 设置 `gpu_memory_utilization=0.5`

2. **训练速度优化**：
   - 增大 `train_prompt_mini_bsz`
   - 启用 `use_dynamic_bsz=True`
   - 减少 `num_traces`（但可能影响性能）

3. **模型质量优化**：
   - 增大 `m` 和 `num_traces`
   - 增大 `l`（更多扩展轮次）
   - 调整 `temperature`（探索 vs 利用）

## 完整示例

```bash
# 完整训练示例
export TRAIN_FILE="data/gsm8k/train.parquet"
export TEST_FILE="data/gsm8k/test.parquet"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# TreeRL 参数（复杂任务）
export m=8
export n=3
export l=2
export t=3
export num_traces=32

# 训练参数
export lr=1e-6
export total_epochs=10

# 启动训练
bash recipe/treerl_v2/scripts/train_treerl_simple.sh
```
