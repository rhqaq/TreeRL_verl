"""
TreeRL Quick Start Example

This example demonstrates how to use TreeRL for training a math reasoning model.
"""

import os
import sys
import numpy as np

# Add verl to path if needed
# sys.path.insert(0, "/path/to/verl")

# Import TreeRL components
from tree_node import TreeNode, MCTSNode, build_into_tree_format, gather_paths


def example_entropy_guided_search():
    """
    Example: Entropy-guided tree search process.
    
    This demonstrates how TreeRL expands trees based on entropy.
    """
    print("=" * 60)
    print("Example: Entropy-Guided Tree Search")
    print("=" * 60)
    
    # Mock decode function
    def decode_fn(token_ids):
        # In real usage, this would use tokenizer.decode
        return " ".join([f"<{tid}>" for tid in token_ids])
    
    # Step 1: Create initial response (M=1 for simplicity)
    print("\nStep 1: Create initial response")
    initial_node = TreeNode(
        tree_idx=0,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[1, 2, 3, 4, 5],
        log_prob_list=[-0.1, -2.5, -0.3, -3.0, -0.5],
        is_end=False,
        max_length=10
    )
    
    print(f"Initial tokens: {initial_node.token_id_list}")
    print(f"Log probs: {initial_node.log_prob_list}")
    print(f"Mask (can expand): {initial_node.mask}")
    
    # Step 2: Find high-entropy tokens
    print("\nStep 2: Find high-entropy tokens for expansion")
    entropy_tokens = initial_node.get_max_entropy_tokens(top_n=2)
    print(f"Top 2 entropy tokens: {entropy_tokens}")
    print("  -> Token 1 (log_prob=-2.5) and Token 3 (log_prob=-3.0)")
    print("     have highest entropy (most uncertain)")
    
    # Step 3: Simulate expansion at token index 3
    print("\nStep 3: Expand at high-entropy position")
    split_idx = entropy_tokens[0]
    
    # Create a child node branching from split_idx
    child_node = TreeNode(
        tree_idx=0,
        node_idx=1,
        decode_fn=decode_fn,
        token_id_list=[100, 101, 102],  # New tokens after branching
        log_prob_list=[-0.2, -0.3, -0.1],
        is_end=True,
        parent_node=initial_node,
        parent_node_idx=0,
        parent_node_split_idx=split_idx,
        finish_reason="stop",
        max_length=10
    )
    
    initial_node.add_child(child_node, split_idx)
    initial_node.update_mask(split_idx)
    
    print(f"Created child node with {len(child_node.token_id_list)} new tokens")
    print(f"Parent mask after expansion: {initial_node.mask}")
    
    # Step 4: Evaluate and assign rewards
    print("\nStep 4: Evaluate leaf nodes")
    child_node.binary_score = 1.0  # Correct answer
    child_node.score = 1.0
    print(f"Leaf node reward: {child_node.binary_score}")
    
    print("\n✓ Entropy-guided expansion complete!")


def example_rloo_advantage():
    """
    Example: RLOO advantage estimation.
    
    This demonstrates how TreeRL computes advantages without a Critic.
    """
    print("\n" + "=" * 60)
    print("Example: RLOO Advantage Estimation")
    print("=" * 60)
    
    # Simulate rewards from multiple traces of the same prompt
    print("\nScenario: 4 traces from same prompt")
    rewards = [1.0, 0.0, 1.0, 0.0]  # 2 correct, 2 incorrect
    
    print(f"Raw rewards: {rewards}")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    
    # RLOO: For each trace, baseline = mean of OTHER traces
    print("\nRLOO Advantage Calculation:")
    advantages = []
    
    for i, r in enumerate(rewards):
        # Get rewards from other traces
        other_rewards = [rewards[j] for j in range(len(rewards)) if j != i]
        baseline = np.mean(other_rewards)
        advantage = r - baseline
        advantages.append(advantage)
        
        print(f"  Trace {i}: reward={r:.1f}, "
              f"baseline=mean({other_rewards})={baseline:.2f}, "
              f"advantage={advantage:.2f}")
    
    print(f"\nSum of advantages: {sum(advantages):.4f} (should be ~0)")
    print("\nKey insight: RLOO provides stable training signal without a Critic!")
    print("  - Correct traces (reward=1) get positive advantage")
    print("  - Incorrect traces (reward=0) get negative advantage")
    print("  - The model learns to prefer correct reasoning paths")


def example_training_workflow():
    """
    Example: Complete TreeRL training workflow.
    """
    print("\n" + "=" * 60)
    print("Example: TreeRL Training Workflow")
    print("=" * 60)
    
    print("""
TreeRL Training Pipeline:

1. PROMPT BATCHING
   └── Input: Batch of N prompts
   └── Each prompt generates M trees

2. TREE SEARCH (per prompt)
   a) Initialize M responses with different random seeds
   b) For L iterations:
      - Find top-N highest entropy tokens across all trees
      - Expand at those positions (T branches each)
   c) Evaluate all leaf nodes (binary reward)
   d) Sample K traces for training

3. ADVANTAGE COMPUTATION
   └── Apply RLOO: advantage_i = reward_i - mean(rewards_others)
   └── Per-token advantage from leaf to root

4. POLICY UPDATE
   └── REINFORCE loss: -log_prob * advantage
   └── Update actor to prefer high-reward paths

5. REPEAT
   └── Next batch of prompts

Key Parameters:
- M=6: Number of initial trees (exploration width)
- N=2: Top entropy tokens to expand (focus)
- L=1: Expansion iterations (depth)
- T=2: Branches per entropy point (diversity)
- K=16: Traces per prompt for training
""")


def example_config():
    """
    Example: TreeRL configuration for different scenarios.
    """
    print("\n" + "=" * 60)
    print("Example: Configuration Scenarios")
    print("=" * 60)
    
    print("""
Scenario 1: Quick Testing (Low Compute)
────────────────────────────────────────
algorithm:
  m: 4              # Fewer initial trees
  n: 1              # Expand only top entropy token
  l: 0              # No expansion iterations
  t: 1              # Single branch
  num_traces: 8     # Fewer training traces

Result: ~4x faster, but less exploration


Scenario 2: Standard Training (Balanced)
────────────────────────────────────────
algorithm:
  m: 6              # Standard initial trees
  n: 2              # Top 2 entropy tokens
  l: 1              # Single expansion round
  t: 2              # 2 branches each
  num_traces: 16    # Standard traces

Result: Good balance of speed and quality


Scenario 3: Maximum Quality (High Compute)
──────────────────────────────────────────
algorithm:
  m: 8              # More initial trees
  n: 3              # More entropy tokens
  l: 2              # Multiple expansions
  t: 3              # More branches
  num_traces: 32    # More training traces

Result: Best quality, but slower


Scenario 4: Memory Constrained
──────────────────────────────
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
    use_remove_padding: true

algorithm:
  m: 4
  num_traces: 8

data:
  train_batch_size: 4

Result: Fits in smaller GPU memory
""")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TreeRL Quick Start Examples")
    print("=" * 60)
    
    example_entropy_guided_search()
    example_rloo_advantage()
    example_training_workflow()
    example_config()
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Prepare your dataset:
   - Format as parquet with 'prompt' and 'answer' fields
   - For math: use GSM8K, MATH, or custom problems

2. Configure training:
   - Edit config/treerl_qwen4b.yaml
   - Set model path, data path, output directory

3. Run training:
   bash run_treerl_qwen4b.sh

4. Monitor training:
   - Check logs in output directory
   - Track validation accuracy
   - Save checkpoints periodically

5. Evaluate model:
   - Use greedy decoding for pass@1
   - Use multiple samples for pass@k

For more details, see README.md
""")


if __name__ == "__main__":
    main()
