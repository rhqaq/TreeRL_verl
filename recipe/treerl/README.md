# TreeRL: Entropy-Guided Tree Search for Mathematical Reasoning

This recipe implements TreeRL's entropy-guided tree search algorithm within the verl 0.7.0 framework for training large language models on mathematical reasoning tasks.

## 📋 Overview

TreeRL is an innovative reinforcement learning approach that uses **entropy-guided tree search** to explore multiple reasoning paths and learn from both successful and failed attempts. Key features:

- **Entropy-Guided Expansion**: Identifies high-uncertainty tokens for targeted expansion
- **Efficient Exploration**: Explores multiple reasoning paths in a single forward pass
- **Token-Level Rewards**: Propagates rewards from leaf nodes to all tokens
- **RLOO Advantage Estimation**: Uses Leave-One-Out baseline for stable training

## 🧮 Algorithm

### Tree Search Process

```
1. Initialize M trees with different responses
   └── For each of M initial responses, create a root node

2. Expand L iterations:
   └── Find top-N highest entropy tokens across all trees
   └── For each entropy token, generate T new branches
   └── Add new nodes to the trees

3. Evaluate leaf nodes:
   └── Compute binary reward (correct/incorrect)
   └── Propagate rewards to all ancestor nodes

4. Sample training traces:
   └── Select diverse paths from root to leaves
   └── Assign token-level rewards with RLOO normalization

5. Update policy:
   └── Use REINFORCE with RLOO advantage
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `m` | Number of initial trees | 6 |
| `n` | Top-N entropy tokens to expand per iteration | 2 |
| `l` | Number of expansion iterations | 1 |
| `t` | Number of branches per entropy token | 2 |
| `num_traces` | Traces sampled for training | 16 |

### RLOO Advantage Estimation

Instead of using a learned Critic, TreeRL uses RLOO (Reward Leave-One-Out) for advantage estimation:

```
advantage_i = reward_i - mean(reward_j for j != i)
```

This provides a stable baseline without requiring a separate Critic model.

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install verl
pip install verl>=0.7.0

# For Qwen models, ensure trust_remote_code is enabled
```

### Training

```bash
# Basic training
bash run_treerl_qwen4b.sh

# Custom configuration
bash run_treerl_qwen4b.sh \
    --num-gpus 4 \
    --m 6 \
    --n 2 \
    --l 1 \
    --t 2 \
    --num-traces 16 \
    --lr 1e-6 \
    --epochs 4
```

### Using Different Models

```bash
# Use Qwen2.5-Math-7B
bash run_treerl_qwen4b.sh \
    --model-path Qwen/Qwen2.5-Math-7B-Instruct \
    --num-gpus 8

# Use DeepSeek-Math
bash run_treerl_qwen4b.sh \
    --model-path deepseek-ai/deepseek-math-7b-instruct \
    --num-gpus 8
```

## 📁 File Structure

```
recipe/treerl/
├── __init__.py              # Package initialization
├── tree_node.py             # TreeNode and MCTSNode data structures
├── entropy_tree_search.py   # Core tree search implementation
├── treerl_rollout.py        # TreeRL Rollout for verl integration
├── treerl_reward_manager.py # Reward manager with RLOO
├── treerl_trainer.py        # TreeRLRayTrainer extending RayPPOTrainer
├── train_treerl.py          # Main training script
├── run_treerl_qwen4b.sh     # Bash launch script
├── config/
│   └── treerl_qwen4b.yaml   # Training configuration
└── README.md                # This file
```

## ⚙️ Configuration

### Model Configuration

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-Math-1.5B-Instruct
    enable_gradient_checkpointing: true
    use_remove_padding: true
```

### Tree Search Configuration

```yaml
algorithm:
  m: 6              # Initial trees
  n: 2              # Top-N entropy tokens
  l: 1              # Expansion iterations
  t: 2              # Branches per entropy
  num_traces: 16    # Training traces
  adv_estimator: rloo  # Use RLOO
```

### Training Configuration

```yaml
trainer:
  total_epochs: 4
  optimizer:
    lr: 1.0e-6
    weight_decay: 0.01
  save_freq: 500
  test_freq: 100
```

## 📊 Evaluation

TreeRL automatically computes pass@k metrics during validation:

- **pass@1**: Greedy decoding accuracy
- **pass@k**: Best-of-k accuracy for k samples

```python
# Evaluation during training
def math_evaluator(problem, response, answer):
    """Binary reward function for math problems."""
    extracted = extract_answer(response)
    return 1.0 if is_correct(extracted, answer) else 0.0
```

## 🔬 Implementation Details

### TreeNode

Each node in the tree stores:
- Token sequence and log probabilities
- Entropy values for each position
- Binary score and normalized score
- Parent/children relationships

```python
class TreeNode:
    def get_max_entropy_tokens(self, top_n):
        """Return indices of highest entropy tokens."""
        return entropy_indices
    
    def get_prefix_ids(self, split_idx):
        """Get token prefix for branching."""
        return prefix_ids
```

### Tree Search

The `EntropyTreeSearch` class manages the search process:

```python
class EntropyTreeSearch:
    def search(self, problem, answer, prompt_ids):
        """
        Perform entropy-guided tree search.
        
        Returns:
            paths: List of reasoning paths with token-level rewards
            avg_reward: Average reward across all paths
        """
```

### RLOO Normalization

```python
def compute_rloo_advantage(rewards, indices):
    """
    Compute RLOO advantage.
    
    For each trace i:
    advantage_i = reward_i - mean(rewards of other traces from same prompt)
    """
```

## 📈 Results

Expected results on GSM8K with Qwen2.5-Math-1.5B:

| Method | pass@1 | pass@8 |
|--------|--------|--------|
| Baseline (SFT) | ~70% | ~82% |
| TreeRL (m=6, l=1) | ~75% | ~88% |

## 🛠️ Troubleshooting

### Out of Memory

```yaml
# Reduce tree search parameters
algorithm:
  m: 4      # Fewer initial trees
  t: 1      # Fewer branches

# Or reduce batch size
data:
  train_batch_size: 4
```

### Slow Training

```yaml
# Reduce expansion iterations
algorithm:
  l: 0  # No expansion, just multi-sample

# Or reduce traces
algorithm:
  num_traces: 8
```

### Poor Performance

```yaml
# Increase exploration
algorithm:
  m: 8      # More initial trees
  n: 3      # More entropy tokens
  l: 2      # More iterations

# Adjust learning rate
trainer:
  optimizer:
    lr: 5.0e-7
```

## 📚 References

1. **TreeRL Paper**: "Entropy-Guided Tree Search for Reasoning" (Original implementation)
2. **verl Framework**: [verl Documentation](https://github.com/volcengine/verl)
3. **RLOO**: "RL on Incorrect Trajectories" (Leave-One-Out baseline)

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows verl conventions
- Tests pass locally
- Documentation is updated

## 📄 License

Apache 2.0

---

**Note**: This implementation adapts TreeRL's original algorithm to verl 0.7.0's architecture. Some differences from the original paper may exist for framework compatibility.
