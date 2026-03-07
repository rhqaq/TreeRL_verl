# TreeRL 并行优化指南

## 问题分析

### 原始串行实现的问题

```python
# 原始代码 - 串行处理
for i in range(len(batch)):
    single_batch = batch[i]
    output = tree_rollout.generate_sequences_with_tree_search(single_batch)
    all_outputs.append(output)
```

**问题**：
1. ❌ 每个 prompt 串行处理，GPU 利用率低
2. ❌ 每次树搜索内部也是串行生成
3. ❌ 多卡资源浪费

### 性能瓶颈

假设：
- M=6 初始响应
- L=1 扩展轮次
- N=2 高熵 token
- T=2 分支

**串行版本的调用次数**：
- 初始生成：1 次 (M 个响应)
- 扩展生成：N × T = 4 次独立调用
- 总计：5 次 **独立的** 小批量生成调用

每次调用都有：
- vLLM 初始化开销
- GPU 空闲等待时间
- 内存分配/释放开销

---

## 并行优化方案

### 优化层次

```
┌─────────────────────────────────────────────────────────────┐
│  Level 1: Prompt 并行 - 多个 prompt 同时进行树搜索            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Prompt 1   │  │  Prompt 2   │  │  Prompt 3   │  ...      │
│  │  Tree Search│  │  Tree Search│  │  Tree Search│           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Level 2: 树内并行 - 单个 prompt 的所有生成为批量处理          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Batch Generate: [M initial + N×T expansions]         │  │
│  │  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐   │  │
│  │  │ R1││ R2││ R3││ R4││ R5││ R6││ E1││ E2││ E3││ E4│   │  │
│  │  └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Level 3: vLLM 连续批处理 - 自动批处理不同长度的请求          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  vLLM Continuous Batching Engine                      │  │
│  │  自动合并请求、动态调度、内存优化                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 实现对比

### 串行版本 (treerl_rollout.py)

```python
# 每次 generate_with_vllm 是独立调用
initial_results = self.generate_with_vllm(initial_prompt_ids)  # M 个响应
# ... 后续扩展也是独立调用
```

**特点**：
- ✅ 简单直观
- ❌ GPU 利用率低
- ❌ 多次内存分配

### 并行版本 (parallel_treerl_rollout.py)

```python
# 所有生成合并为一次批量调用
expansion_prompt_ids = []
for task in all_expansion_tasks:
    for branch in range(self.t):
        expansion_prompt_ids.append(...)

# 一次调用生成所有扩展！
expansion_results = self.batch_generate(expansion_prompt_ids)
```

**特点**：
- ✅ 单次批量生成
- ✅ 高 GPU 利用率
- ✅ vLLM 自动优化

---

## 性能预估

### 理论加速比

假设单次 vLLM 调用开销 = 100ms，实际生成时间 = 50ms/样本

| 场景 | 串行版本 | 并行版本 | 加速比 |
|------|---------|---------|--------|
| 单 prompt 树搜索 | 5 × 100ms + 10 × 50ms = 1000ms | 100ms + 10 × 50ms = 600ms | 1.7x |
| 4 prompts 批处理 | 4 × 1000ms = 4000ms | 100ms + 40 × 50ms = 2100ms | 1.9x |
| 8 prompts 批处理 | 8 × 1000ms = 8000ms | 100ms + 80 × 50ms = 4100ms | 2.0x |

**实际加速**：取决于 GPU 数量、vLLM 配置、批次大小

### 多卡配置示例

```yaml
# 8 卡配置
resource_pool:
  actor_rollout:
    num_gpus: 8
    
actor_rollout_ref:
  rollout:
    vllm:
      enable: true
      tensor_parallel_size: 4  # 4 卡做推理
      max_num_batched_tokens: 32768
      
parallel:
  num_parallel_prompts: 8      # 8 个 prompt 并行
  num_parallel_trees: 16       # 批量生成更多树
```

---

## 使用方法

### 1. 基础并行版本

```python
from parallel_treerl_rollout import ParallelTreeRLRollout

tree_rollout = ParallelTreeRLRollout(
    actor_rollout_wg=actor_rollout_wg,
    tokenizer=tokenizer,
    config=config,
    evaluator_fn=math_evaluator,
    num_parallel_prompts=4,   # 4 个 prompt 并行
    num_parallel_trees=8,     # 每批最多 8 棵树
)
```

### 2. 完全并行版本 (异步)

```python
from parallel_treerl_rollout import FullyParallelTreeRLRollout

tree_rollout = FullyParallelTreeRLRollout(
    actor_rollout_wg=actor_rollout_wg,
    tokenizer=tokenizer,
    config=config,
    evaluator_fn=math_evaluator,
    num_parallel_prompts=8,   # 8 个 prompt 异步并行
)
```

### 3. 启动脚本

```bash
# 使用并行配置
bash run_treerl_parallel.sh --num-gpus 8 --config config/parallel_treerl_qwen4b.yaml
```

---

## 调优建议

### GPU 资源分配

| GPU 数量 | tensor_parallel_size | num_parallel_prompts | 预期加速 |
|---------|---------------------|---------------------|---------|
| 4 | 2 | 2 | 1.5-2x |
| 8 | 4 | 4 | 2-3x |
| 16 | 8 | 8 | 3-4x |

### 内存优化

```yaml
# 减少内存占用
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    
algorithm:
  m: 4              # 减少初始树
  num_traces: 8     # 减少训练轨迹
  
parallel:
  num_parallel_prompts: 2  # 减少并行数
```

### 批次大小调优

```yaml
# 增大批次以提高吞吐量
actor_rollout_ref:
  rollout:
    vllm:
      max_num_batched_tokens: 65536  # 更大批次
      
parallel:
  initial_batch_size: 32    # M × num_parallel_prompts
  expansion_batch_size: 64  # 扩展批次
```

---

## 监控指标

### 关键指标

1. **GPU 利用率**
   ```bash
   watch -n 1 nvidia-smi
   ```
   目标：> 80%

2. **生成吞吐量**
   ```
   tokens/second = total_tokens / generation_time
   ```
   并行版本应该显著更高

3. **每 prompt 处理时间**
   ```
   time_per_prompt = total_time / num_prompts
   ```

### 对比测试

```bash
# 运行对比测试
python benchmark_parallel.py \
    --serial-config config/treerl_qwen4b.yaml \
    --parallel-config config/parallel_treerl_qwen4b.yaml \
    --num-prompts 16 \
    --output benchmark_results.json
```

---

## 常见问题

### Q1: 并行版本报内存不足？

**解决方案**：
```yaml
# 减少并行度
parallel:
  num_parallel_prompts: 2

# 或使用梯度检查点
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
```

### Q2: 加速效果不明显？

**可能原因**：
1. GPU 数量不足 - 需要至少 4 卡
2. vLLM 未正确配置
3. 批次大小太小

**检查**：
```python
# 确认 vLLM 配置
print(config.actor_rollout_ref.rollout.vllm)
```

### Q3: 如何验证并行是否生效？

**方法**：
```python
# 在 parallel_treerl_rollout.py 中查看日志
# 应该看到:
# [Tree Search] Generating 24 expansions...  # 一次批量生成
# 而不是多次小批量调用
```

---

## 总结

| 方面 | 串行版本 | 并行版本 |
|------|---------|---------|
| GPU 利用率 | 低 (~30%) | 高 (~80%) |
| 处理速度 | 慢 | 快 2-4x |
| 内存效率 | 低 | 高 |
| 代码复杂度 | 简单 | 中等 |
| 推荐场景 | 单卡调试 | 多卡生产 |

**推荐**：生产环境使用并行版本，调试时可用串行版本。
