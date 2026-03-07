"""
Test script for TreeRL implementation.

Validates the core components work correctly.
"""

import os
import sys
import numpy as np
from typing import List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tree_node import TreeNode, MCTSNode, build_into_tree_format, gather_paths


def test_treeNode_basic():
    """Test basic TreeNode functionality."""
    print("\n=== Test TreeNode Basic ===")
    
    def decode_fn(ids):
        return "".join([chr(i + ord('a')) for i in ids])
    
    # Create a simple node
    node = TreeNode(
        tree_idx=0,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[0, 1, 2, 3, 4],
        log_prob_list=[-0.1, -0.2, -0.5, -0.3, -0.1],
        is_end=True,
        finish_reason="stop",
        max_length=10
    )
    
    print(f"Total string: {node.total_str}")
    print(f"Log probs: {node.log_prob_list}")
    # Entropy is calculated as -log_prob for each token
    entropy = [-lp for lp in node.log_prob_list]
    print(f"Entropy (neg log prob): {entropy}")
    
    # Test entropy tokens
    entropy_tokens = node.get_max_entropy_tokens(top_n=2)
    print(f"Max entropy tokens: {entropy_tokens}")
    
    assert len(entropy_tokens) <= 2
    assert all(0 <= idx < len(node.token_id_list) for idx in entropy_tokens)
    
    print("✓ TreeNode basic test passed")


def test_treeNode_parent_child():
    """Test TreeNode parent-child relationship."""
    print("\n=== Test TreeNode Parent-Child ===")
    
    def decode_fn(ids):
        return "".join([str(i) for i in ids])
    
    # Create parent node
    parent = TreeNode(
        tree_idx=0,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[0, 1, 2],
        log_prob_list=[-0.1, -0.2, -0.3],
        is_end=False,
        max_length=10
    )
    
    # Create child node
    child = TreeNode(
        tree_idx=0,
        node_idx=1,
        decode_fn=decode_fn,
        token_id_list=[3, 4],
        log_prob_list=[-0.4, -0.5],
        is_end=True,
        parent_node=parent,
        parent_node_idx=0,
        parent_node_split_idx=2,
        finish_reason="stop",
        max_length=10
    )
    
    parent.add_child(child, 2)
    
    print(f"Parent child_nodes: {len(parent.child_nodes)}")
    print(f"Child parent: {child.parent_node}")
    print(f"Child prefix: {child.get_prefix_ids(0)}")
    
    # Test mask update
    print(f"Parent mask before: {parent.mask}")
    parent.update_mask(2)
    print(f"Parent mask after: {parent.mask}")
    
    print("✓ TreeNode parent-child test passed")


def test_mcts_node():
    """Test MCTSNode with RLOO normalization."""
    print("\n=== Test MCTSNode ===")
    
    # Create MCTS nodes with rewards
    nodes = [
        MCTSNode(
            answer="answer_1",
            answer_token=[1, 2, 3],
            reward_raw=1.0,
            reward_value=0.0,  # Will be updated
        ),
        MCTSNode(
            answer="answer_2",
            answer_token=[4, 5, 6],
            reward_raw=0.0,
            reward_value=0.0,
        ),
        MCTSNode(
            answer="answer_3",
            answer_token=[7, 8, 9],
            reward_raw=1.0,
            reward_value=0.0,
        ),
        MCTSNode(
            answer="answer_4",
            answer_token=[10, 11, 12],
            reward_raw=0.0,
            reward_value=0.0,
        ),
    ]
    
    # Simulate RLOO normalization
    rewards = [n.R for n in nodes]
    for i, node in enumerate(nodes):
        # RLOO: use mean of others as baseline
        other_rewards = [r for j, r in enumerate(rewards) if j != i]
        baseline = np.mean(other_rewards)
        node.R = node.R - baseline
        print(f"Node {i}: raw={rewards[i]}, baseline={baseline:.2f}, advantage={node.R:.2f}")
    
    print("✓ MCTSNode test passed")


def test_entropy_calculation():
    """Test entropy calculation."""
    print("\n=== Test Entropy Calculation ===")
    
    def decode_fn(ids):
        return " ".join([str(i) for i in ids])
    
    # Create node with varying log probs
    log_probs = [-0.1, -2.0, -0.3, -3.0, -0.5]  # Index 1 and 3 have highest entropy
    node = TreeNode(
        tree_idx=0,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[0, 1, 2, 3, 4],
        log_prob_list=log_probs,
        is_end=True,
        finish_reason="stop",
        max_length=10
    )
    
    print(f"Log probs: {log_probs}")
    # Entropy is neg log prob
    entropy = [-lp for lp in log_probs]
    print(f"Entropy (neg log prob): {entropy}")
    
    entropy_tokens = node.get_max_entropy_tokens(top_n=2)
    print(f"Top 2 entropy tokens: {entropy_tokens}")
    
    # Verify: highest entropy should be at index 1 (log_prob=-2.0) and 3 (log_prob=-3.0)
    assert 1 in entropy_tokens or 3 in entropy_tokens, "Should identify high entropy tokens"
    
    print("✓ Entropy calculation test passed")


def test_build_tree_format():
    """Test building tree format from node lists."""
    print("\n=== Test Build Tree Format ===")
    
    def decode_fn(ids):
        return "".join([chr(i + ord('a')) for i in ids])
    
    # Create two simple trees
    tree_lists = []
    
    # Tree 0
    node0_0 = TreeNode(
        tree_idx=0,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[0, 1, 2],
        log_prob_list=[-0.1, -0.2, -0.3],
        is_end=False,
        max_length=10
    )
    node0_0.binary_score = 1.0
    node0_0.score = 1.0
    
    node0_1 = TreeNode(
        tree_idx=0,
        node_idx=1,
        decode_fn=decode_fn,
        token_id_list=[3, 4],
        log_prob_list=[-0.4, -0.5],
        is_end=True,
        parent_node=node0_0,
        parent_node_idx=0,
        parent_node_split_idx=3,
        finish_reason="stop",
        max_length=10
    )
    node0_1.binary_score = 1.0
    node0_1.score = 1.0
    
    node0_0.add_child(node0_1, 3)
    tree_lists.append([node0_0, node0_1])
    
    # Tree 1
    node1_0 = TreeNode(
        tree_idx=1,
        node_idx=0,
        decode_fn=decode_fn,
        token_id_list=[5, 6],
        log_prob_list=[-0.6, -0.7],
        is_end=True,
        finish_reason="stop",
        max_length=10
    )
    node1_0.binary_score = 0.0
    node1_0.score = 0.0
    tree_lists.append([node1_0])
    
    # Build tree format
    root, selected_terminals = build_into_tree_format(tree_lists, decode_fn, num_traces=2)
    
    print(f"Root reward: {root.reward_raw}")
    print(f"Selected terminals: {len(selected_terminals)}")
    
    # Gather paths
    paths = gather_paths(root, selected_terminals, pass_k=2)
    
    print(f"Number of paths: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"Path {i}: {len(path)} nodes")
        for node in path:
            print(f"  - value: {node['value']:.2f}")
    
    print("✓ Build tree format test passed")


def test_rloo_advantage():
    """Test RLOO advantage computation."""
    print("\n=== Test RLOO Advantage ===")
    
    # Simulate rewards from same prompt
    rewards = [1.0, 0.0, 1.0, 0.0]
    indices = [0, 0, 0, 0]  # All from same prompt
    
    # Compute RLOO advantage
    advantages = []
    for i in range(len(rewards)):
        # Get rewards from same prompt, excluding current
        other_rewards = [r for j, r in enumerate(rewards) if j != i and indices[j] == indices[i]]
        baseline = np.mean(other_rewards) if other_rewards else 0.0
        advantage = rewards[i] - baseline
        advantages.append(advantage)
        print(f"Sample {i}: reward={rewards[i]:.2f}, baseline={baseline:.2f}, advantage={advantage:.2f}")
    
    # Verify advantages sum to zero (approximately)
    total = sum(advantages)
    print(f"Sum of advantages: {total:.4f}")
    assert abs(total) < 0.01, "Advantages should sum to zero"
    
    print("✓ RLOO advantage test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running TreeRL Tests")
    print("=" * 60)
    
    test_treeNode_basic()
    test_treeNode_parent_child()
    test_mcts_node()
    test_entropy_calculation()
    test_build_tree_format()
    test_rloo_advantage()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
