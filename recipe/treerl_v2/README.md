# TreeRL for verl 0.7.0

## 概述

这是 TreeRL (Tree-based Reinforcement Learning) 在 verl 0.7.0 框架上的实现，通过继承 `AgentLoopManager` 实现熵引导的树搜索算法。

**核心特性**：
- ✅ 使用 **DAPORewardManager** 进行真实奖励评估（非模拟数据）
- ✅ 支持多种数据源（GSM8K, MATH, Code 等）
- ✅ 熵引导的树搜索策略
- ✅ RLOO 优势估计

## 核心设计

### 1. TreeRLAgentLoopManager

继承自 `AgentLoopManager`，重写 `generate_sequences` 方法实现树搜索：

```python
class TreeRLAgentLoopManager(AgentLoopManager):
    def __init__(self, config, worker_group, rm_resource_pool):
        # 初始化 DAPORewardManager 用于真实奖励评估
        self.reward_manager = DAPORewardManager(
            tokenizer=self.tokenizer,
            num_examine=10,
            compute_score=None,  # 使用 default_compute_score
            reward_fn_key="data_source",
        )
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        树搜索流程：
        1. 生成 M 个初始响应
        2. 进行 L 轮扩展（选择 N 个高熵 token，每个生成 T 个分支）
        3. 使用 DAPORewardManager 评估叶子节点（真实奖励）
        4. 采样 K 条轨迹用于训练
        """
```

### 2. 真实奖励评估

使用 `DAPORewardManager` 进行真实奖励评估，支持多种数据源：

```python
# DAPORewardManager 会根据 data_source 自动选择评估方式：
# - "openai/gsm8k": GSM8K 数学题
# - "lighteval/MATH": MATH 数学题
# - "math_dapo": DAPO 数学题
# - "codecontests", "apps", "codeforces": 代码题
# - 其他数据源...
```

**评估流程**：
1. 从 `DataProto.non_tensor_batch` 获取 `ground_truth` 和 `data_source`
2. 使用 tokenizer 解码 prompt 和 response
3. 调用 `default_compute_score` 计算真实分数
4. 返回 token-level 奖励张量

### 3. 树搜索参数

```python
TREE_SEARCH_CONFIG = {
    "m": 6,              # 初始树的数量
    "n": 2,              # 每轮扩展的高熵 token 数量
    "l": 1,              # 扩展迭代轮数
    "t": 2,              # 每个熵点的分支数量
    "num_traces": 16,    # 每个 prompt 采样的训练轨迹数
}
```

### 4. RLOO 优势估计

TreeRL 使用 RLOO (Reward Leave-One-Out) 而非 GAE：

```
advantage_i = reward_i - mean(reward_j for j != i)
```

优势：
- 不需要 Critic
- 降低方差
- 适合树搜索的多轨迹场景

## 文件结构

```
verl_070/recipe/treerl_v2/
├── __init__.py           # 导出主要类
├── agent_loop.py         # TreeRLAgentLoopManager 实现（含 DAPORewardManager）
├── ray_trainer.py        # TreeRLRayTrainer 实现
├── main.py               # 示例入口
├── test_treenode.py      # 单元测试
└── README.md             # 本文档
```

## 使用方法

### 方法一：使用启动脚本（推荐）

我们提供了两种启动脚本：

#### 1. 完整版启动脚本（Ray 集群模式）

适用于多节点分布式训练：

```bash
# 设置环境变量
export TRAIN_FILE="data/gsm8k/train.parquet"
export TEST_FILE="data/gsm8k/test.parquet"
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# TreeRL 参数
export m=6
export n=2
export l=1
export t=2
export num_traces=16

# 启动训练
bash recipe/treerl_v2/scripts/train_treerl.sh
```

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `m` | 6 | 初始树的数量 |
| `n` | 2 | 每轮扩展的高熵 token 数量 |
| `l` | 1 | 扩展迭代轮数 |
| `t` | 2 | 每个熵点的分支数量 |
| `num_traces` | 16 | 每个 prompt 采样的训练轨迹数 |

#### 2. 简化版启动脚本（单机模式）

适用于快速测试：

```bash
# 直接启动（参数已在脚本中设置）
bash recipe/treerl_v2/scripts/train_treerl_simple.sh
```

### 方法二：使用配置文件

```bash
# 使用 Hydra 配置文件
python -m recipe.treerl_v2.main_treerl \
    --config-path=recipe/treerl_v2/config \
    --config-name=treerl_config
```

### 方法三：命令行参数覆盖

```bash
# 覆盖特定参数
python -m recipe.treerl_v2.main_treerl \
    data.train_files=data/gsm8k/train.parquet \
    algorithm.m=8 \
    algorithm.n=3 \
    trainer.total_epochs=20
```

## 奖励评估示例

### GSM8K 数学题

```python
# data_source: "openai/gsm8k"
# ground_truth: "42"

# DAPORewardManager 会：
# 1. 解码 response 提取答案
# 2. 与 ground_truth 比较计算正确性
# 3. 返回 1.0（正确）或 0.0（错误）
```

### MATH 数学题

```python
# data_source: "lighteval/MATH"
# ground_truth: "\\frac{1}{2}"

# DAPORewardManager 会：
# 1. 使用 sympy 进行符号比较
# 2. 支持等价性判断（如 0.5 == 1/2）
```

### 代码题

```python
# data_source: "codecontests"
# ground_truth: ["test_case_1", "test_case_2"]

# DAPORewardManager 会：
# 1. 在沙箱中执行代码
# 2. 运行测试用例
# 3. 返回通过率
```

## 与原始 TreeRL 的对应关系

| 原始 TreeRL | verl 0.7.0 实现 |
|------------|----------------|
| `TreeSearcher` | `TreeRLAgentLoopWorker._tree_search_single_prompt` |
| `TreeNode` | `agent_loop.TreeNode` |
| `EntropyGuidedExpansion` | `TreeNode.get_high_entropy_tokens` |
| `RLOO` | `TreeRLRayTrainer._compute_advantage` |
| `FastAPIServer` | `AsyncLLMServerManager` (继承自父类) |
| 自定义奖励 | `DAPORewardManager` |

## 性能优化

1. **批量生成**：使用 `AsyncLLMServerManager` 并行生成多个分支
2. **异步推理**：利用 Ray 的异步调用能力
3. **树剪枝**：通过 `num_traces` 控制训练样本数量
4. **真实奖励**：直接使用 `DAPORewardManager`，无需额外模拟

## 注意事项

1. **数据源支持**：确保 `data_source` 字段正确，否则 `default_compute_score` 会抛出 `NotImplementedError`
2. **内存管理**：`max_response_length` 需要根据模型和任务调整
3. **GPU 内存**：根据 `m * t^l` 规模估算内存需求
4. **ground_truth**：数据必须包含正确的 ground_truth 字段

## 扩展数据源

如果需要支持新的数据源，可以扩展 `default_compute_score`：

```python
# 在 verl/utils/reward_score/__init__.py 中添加：
elif data_source == "my_custom_dataset":
    from . import my_custom_scorer
    res = my_custom_scorer.compute_score(solution_str, ground_truth)
```

## 调试

启用详细日志：

```python
# 在 TreeRLAgentLoopManager 中设置
self.reward_manager = DAPORewardManager(
    tokenizer=self.tokenizer,
    num_examine=100,  # 打印更多样本用于调试
)
```

这会在控制台输出：
- `[prompt]`: 输入问题
- `[response]`: 模型回答
- `[ground_truth]`: 正确答案
- `[score]`: 计算的分数

## 测试

运行单元测试：

```bash
cd verl_070
python3 recipe/treerl_v2/test_treenode.py
```

## TODO

- [ ] 添加树可视化工具
- [ ] 优化内存使用（大规模树搜索）
- [ ] 支持 Beam Search 作为初始化策略
- [ ] 添加更多数据源的单元测试
