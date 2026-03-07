"""
TreeNode and MCTSNode data structures for TreeRL.

Based on the original TreeRL implementation from openrlhf.
"""

from __future__ import annotations
import math
import random
from typing import List, Optional, Callable, Dict, Any
from pydantic import BaseModel
from collections import deque


class MCTSNode(BaseModel):
    """MCTS-style node for representing the tree structure after conversion from TreeNode."""
    answer: str
    answer_token: List[int]
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    R: float = 0  # Reward value after RLOO normalization
    reward_raw: float = 0  # Raw reward before normalization
    depth: int = 0
    main_chain: bool = False  # True if leads to correct answer
    terminal: bool = False
    terminal_in_subtree: int = 0
    correct_terminal_in_subtree: int = 0
    accumulated_value: float = 0
    value: float = 0
    finish_reason: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class TreeNode:
    """TreeNode for search tree during expansion."""
    
    def __init__(
        self,
        tree_idx: int,
        node_idx: int,
        decode_fn: Callable,
        token_id_list: List[int],
        log_prob_list: List[float],
        finish_reason: Optional[str] = None,
        is_end: bool = False,
        parent_node: Optional['TreeNode'] = None,
        parent_node_idx: Optional[int] = None,
        parent_node_split_idx: Optional[int] = None,
        max_length: int = 8192,
    ):
        self.tree_idx = tree_idx
        self.node_idx = node_idx
        self.token_id_list = token_id_list
        self.token_str_list = [decode_fn([tid]) for tid in token_id_list]
        self.log_prob_list = log_prob_list
        self.token_num = len(token_id_list)
        self.finish_reason = finish_reason
        self.is_end = is_end
        
        self.parent_node = parent_node
        self.parent_node_idx = parent_node_idx
        self.parent_node_split_idx = parent_node_split_idx
        
        self.child_nodes: List['TreeNode'] = []
        self.child_split_indices: List[int] = []
        
        self.aggregate_str = ""
        if parent_node is not None:
            self.aggregate_str = parent_node.aggregate_str + ''.join(
                parent_node.token_str_list[:parent_node_split_idx]
            )
        self.total_str = self.aggregate_str + ''.join(self.token_str_list)
        
        self.aggregate_token_ids: List[int] = []
        if parent_node is not None:
            self.aggregate_token_ids = parent_node.aggregate_token_ids + \
                parent_node.token_id_list[:parent_node_split_idx]
        
        # Mask: tokens that should not be selected for expansion
        self.mask: List[bool] = [False] * len(self.token_str_list)
        if len(self.aggregate_token_ids) > 0 and len(self.token_str_list) > 0:
            self.mask[0] = True
        
        total_length = len(self.aggregate_token_ids) + len(self.token_id_list)
        if total_length > max_length:
            tokens_to_mask = total_length - max_length
            for i in range(max(0, len(self.mask) - tokens_to_mask), len(self.mask)):
                self.mask[i] = True
            self.is_end = True
        
        for i, token_str in enumerate(self.token_str_list):
            if "conclusion" in token_str.lower() or "answer" in token_str.lower():
                for j in range(i + 1, len(self.mask)):
                    self.mask[j] = True
                self.is_end = True
                break
        
        self.binary_score: Optional[float] = None
        self.score: Optional[float] = None

    def get_prefix_ids(self, current_token_index: int) -> List[int]:
        return self.aggregate_token_ids + self.token_id_list[:current_token_index]

    def add_child(self, child_node: 'TreeNode', split_index: int) -> None:
        self.child_nodes.append(child_node)
        self.child_split_indices.append(split_index)
        child_node.parent_node = self

    def get_max_entropy_tokens(self, top_n: int = 1) -> List[int]:
        """Get indices of top-N highest entropy tokens (excluding masked ones)."""
        entropies = []
        for i, log_prob in enumerate(self.log_prob_list):
            if not self.mask[i]:
                entropies.append((-log_prob, i))
        
        sorted_indices = sorted(entropies, key=lambda x: x[0], reverse=True)
        result = [idx for _, idx in sorted_indices[:top_n]]
        
        while len(result) < top_n and len(result) > 0:
            result += result[:top_n - len(result)]
        return result
    
    def update_mask(self, split_index: int) -> None:
        """Update mask to mark tokens up to split_index as expanded."""
        for i in range(split_index):
            if i < len(self.mask):
                self.mask[i] = True


def leaf_normalize(nodes: List[MCTSNode], root: MCTSNode):
    """RLOO normalization of leaf rewards."""
    leaf_correctness = [leaf.R for leaf in nodes]
    _sum = sum(leaf_correctness)
    num = len(leaf_correctness) - 1
    
    if num == 0:
        raise ValueError("num_traces == 0")
    
    # Leave-one-out normalization
    mean = [(_sum - leaf_correctness[i]) / num for i in range(len(leaf_correctness))]
    root.reward_raw = sum(leaf_correctness) / len(leaf_correctness)
    
    for i, leaf in enumerate(nodes):
        leaf.R = leaf.R - mean[i]  # RLOO
        leaf.accumulated_value = leaf.R
        _leaf_backpropagate(leaf)
    
    _normalize_all_steps(root)


def _leaf_backpropagate(node: MCTSNode):
    """Backpropagate terminal info to ancestors."""
    if node.terminal and node.main_chain:
        node.terminal_in_subtree += 1
        node.correct_terminal_in_subtree += 1
        parent = node.parent
        while parent:
            parent.terminal_in_subtree += 1
            parent.correct_terminal_in_subtree += 1
            parent.accumulated_value += node.accumulated_value
            parent = parent.parent
    elif node.terminal:
        node.terminal_in_subtree += 1
        parent = node.parent
        while parent:
            parent.terminal_in_subtree += 1
            parent.accumulated_value += node.accumulated_value
            parent = parent.parent


def _normalize_all_steps(root: MCTSNode):
    """Token-level normalization (default in original TreeRL)."""
    all_steps = []
    to_consider = deque([root])
    while to_consider:
        current_node = to_consider.popleft()
        if current_node.terminal_in_subtree != 0 or current_node.terminal:
            all_steps.append(current_node)
        to_consider.extend(current_node.children)


def select_terminal(all_leaves: List[MCTSNode], num_traces: int) -> List[MCTSNode]:
    """Select terminals for training."""
    random.shuffle(all_leaves)
    selected = []
    remaining = []
    
    for leaf in all_leaves:
        if leaf.main_chain and len(selected) == 0:
            selected.append(leaf)
        else:
            remaining.append(leaf)
    
    remaining_num = num_traces - len(selected)
    if remaining_num > 0:
        selected.extend(random.sample(remaining, min(remaining_num, len(remaining))))
    
    random.shuffle(selected)
    return selected[:num_traces]


def path_from_root_to_node(node: MCTSNode) -> List[Dict[str, Any]]:
    """Extract path from root to node with incremental values."""
    path = []
    while node.parent is not None:
        parent_value = node.parent.accumulated_value / max(node.parent.terminal_in_subtree, 1)
        child_value = node.accumulated_value / max(node.terminal_in_subtree, 1)
        
        path.append({
            'answer': node.answer,
            'token_answer': node.answer_token,
            'reward': node.value,
            'pass_ratio': node.correct_terminal_in_subtree / max(node.terminal_in_subtree, 1),
            'value': child_value - parent_value,
            'state_value': child_value
        })
        node = node.parent
    
    return path[::-1]


def gather_paths(
    root: MCTSNode,
    selected_terminals: List[MCTSNode],
    pass_k: int
) -> List[List[Dict[str, Any]]]:
    """Gather paths from selected terminals."""
    paths = []
    if len(selected_terminals) < pass_k:
        return None
    
    for terminal_node in selected_terminals:
        paths.append(path_from_root_to_node(terminal_node))
    
    assert len(paths) == pass_k
    return paths


def build_into_tree_format(
    tree_lists: List[List[TreeNode]],
    decode_fn: Callable,
    num_traces: int
) -> tuple[MCTSNode, List[MCTSNode]]:
    """Convert TreeNodes to MCTSNode tree format."""
    all_leaves = []
    
    def build_tree_node(tree_node: TreeNode, parent_mcts_node: Optional[MCTSNode] = None) -> MCTSNode:
        tree_node.child_nodes.sort(key=lambda x: x.parent_node_split_idx)
        child_split_indices = [child.parent_node_split_idx for child in tree_node.child_nodes]
        
        is_terminal = False
        R = 0
        main_chain = False
        
        if not child_split_indices:
            first_child_split_idx = len(tree_node.token_id_list)
            is_terminal = True
            R = tree_node.score if tree_node.score is not None else 0
            if tree_node.binary_score == 1:
                main_chain = True
        else:
            first_child_split_idx = child_split_indices[0]
        
        root_node = MCTSNode(
            answer=''.join([decode_fn([tid]) for tid in tree_node.token_id_list[:first_child_split_idx]]),
            answer_token=tree_node.token_id_list[:first_child_split_idx],
            parent=parent_mcts_node,
            depth=(parent_mcts_node.depth + 1) if parent_mcts_node else 0,
            terminal=is_terminal,
            R=R,
            main_chain=main_chain,
            finish_reason=tree_node.finish_reason
        )
        
        if root_node.terminal:
            all_leaves.append(root_node)
        
        def add_segments_and_children(current_mcts_node: MCTSNode, start_idx: int):
            i = 0
            while i < len(tree_node.child_nodes):
                child_nodes_group = []
                current_split_idx = child_split_indices[i]
                
                while i < len(tree_node.child_nodes) and child_split_indices[i] == current_split_idx:
                    child_nodes_group.append(tree_node.child_nodes[i])
                    i += 1
                
                is_terminal = False
                R = 0
                main_chain = False
                
                if i < len(tree_node.child_nodes):
                    next_split_idx = child_split_indices[i]
                else:
                    next_split_idx = len(tree_node.token_id_list)
                    is_terminal = True
                    R = tree_node.score if tree_node.score is not None else 0
                    if tree_node.binary_score == 1:
                        main_chain = True
                
                segment_node = MCTSNode(
                    answer=''.join([decode_fn([tid]) for tid in tree_node.token_id_list[start_idx:next_split_idx]]),
                    answer_token=tree_node.token_id_list[start_idx:next_split_idx],
                    parent=current_mcts_node,
                    depth=current_mcts_node.depth + 1,
                    terminal=is_terminal,
                    R=R,
                    main_chain=main_chain,
                    finish_reason=tree_node.finish_reason
                )
                current_mcts_node.children.append(segment_node)
                
                if segment_node.terminal:
                    all_leaves.append(segment_node)
                
                for child_node in child_nodes_group:
                    child_mcts_node = build_tree_node(child_node, current_mcts_node)
                    current_mcts_node.children.append(child_mcts_node)
                
                start_idx = next_split_idx
                current_mcts_node = segment_node
        
        add_segments_and_children(root_node, first_child_split_idx)
        return root_node
    
    root = MCTSNode(answer="", answer_token=[])
    for tree_list in tree_lists:
        if len(tree_list) > 0:
            root.children.append(build_tree_node(tree_list[0], root))
    
    leaf_normalize(all_leaves, root)
    selected_terminals = select_terminal(all_leaves, num_traces)
    
    return root, selected_terminals
