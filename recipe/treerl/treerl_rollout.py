"""
TreeRL Rollout for entropy-guided tree search.

This module implements the rollout logic for TreeRL's entropy-guided tree search,
integrating with verl's infrastructure.
"""

import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch
import numpy as np

from verl import DataProto
from .tree_node import TreeNode, build_into_tree_format, gather_paths


class TreeRLRollout:
    """
    Entropy-Guided Tree Search Rollout.
    
    Performs tree search by:
    1. Generating M initial responses
    2. Expanding at high-entropy positions
    3. Evaluating leaf nodes
    4. Returning token-level rewards with RLOO normalization
    """
    
    def __init__(
        self,
        actor_rollout_wg,  # The actor rollout worker group from verl
        tokenizer,
        config,
        evaluator_fn: Optional[Callable] = None,
    ):
        """
        Initialize TreeRL Rollout.
        
        Args:
            actor_rollout_wg: verl's actor rollout worker group
            tokenizer: Tokenizer for encoding/decoding
            config: Configuration with tree search parameters
            evaluator_fn: Function to compute binary reward (problem, response, answer) -> float
        """
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.config = config
        self.evaluator_fn = evaluator_fn
        
        # Tree search parameters
        self.m = config.algorithm.get("m", 6)  # Number of initial trees
        self.n = config.algorithm.get("n", 2)   # Top-N entropy tokens per iteration
        self.l = config.algorithm.get("l", 1)   # Number of expansion iterations
        self.t = config.algorithm.get("t", 2)   # Branches per entropy point
        self.num_traces = config.algorithm.get("num_traces", 16)  # Traces for training
        
        # Generation parameters
        self.temperature = config.actor_rollout_ref.rollout.temperature
        self.top_p = config.actor_rollout_ref.rollout.top_p
        self.max_new_tokens = config.data.max_response_length
        self.prompt_max_len = config.data.max_prompt_length
    
    def decode_fn(self, ids: List[int]) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def generate_with_vllm(
        self,
        prompt_ids_list: List[List[int]],
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
    ) -> List[Tuple[List[int], List[float], str]]:
        """
        Generate responses using the actor rollout worker group.
        
        Returns:
            List of (token_ids, log_probs, finish_reason) tuples
        """
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        max_tokens = max_tokens or self.max_new_tokens
        
        # Create DataProto for generation
        batch_size = len(prompt_ids_list)
        
        # Pad to same length
        max_len = max(len(ids) for ids in prompt_ids_list)
        padded_ids = []
        attention_mask = []
        
        for ids in prompt_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))
        
        input_ids = torch.tensor(padded_ids)
        attention_mask = torch.tensor(attention_mask)
        
        # Create DataProto
        gen_data = DataProto(
            batch={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        
        # Generate using actor rollout
        output = self.actor_rollout_wg.generate_sequences(gen_data)
        
        # Extract results
        results = []
        responses = output.batch["responses"]
        response_masks = output.batch.get("response_mask", None)
        
        # Get log probs if available
        log_probs = output.batch.get("old_log_probs", None)
        
        for i in range(batch_size):
            # Get valid response tokens
            if response_masks is not None:
                valid_len = response_masks[i].sum().item()
                token_ids = responses[i, :valid_len].tolist()
            else:
                # Find EOS or use all
                token_ids = responses[i].tolist()
                if self.tokenizer.eos_token_id in token_ids:
                    eos_idx = token_ids.index(self.tokenizer.eos_token_id)
                    token_ids = token_ids[:eos_idx + 1]
            
            # Get log probs for this response
            if log_probs is not None:
                if response_masks is not None:
                    probs = log_probs[i, :valid_len].tolist()
                else:
                    probs = log_probs[i, :len(token_ids)].tolist()
            else:
                probs = [-1.0] * len(token_ids)  # Placeholder
            
            finish_reason = "stop" if token_ids and token_ids[-1] == self.tokenizer.eos_token_id else "length"
            
            results.append((token_ids, probs, finish_reason))
        
        return results
    
    def evaluate_node(self, problem: str, node: TreeNode, answer: str) -> Tuple[float, float]:
        """Evaluate a node to get binary score and final score."""
        if node.is_end and node.finish_reason == "stop":
            if self.evaluator_fn:
                binary_score = self.evaluator_fn(problem, node.total_str, answer)
            else:
                binary_score = 0
        else:
            binary_score = 0
        
        return binary_score, binary_score
    
    def search(
        self,
        problem: str,
        answer: str,
        prompt_ids: List[int],
    ) -> Tuple[List[List[Dict[str, Any]]], float]:
        """
        Perform entropy-guided tree search.
        
        Returns:
            (paths, avg_reward) - paths for training, average reward
        """
        time_start = time.time()
        
        # === Step 1: Initialize M trees ===
        tree_lists = []
        initial_prompt_ids = [prompt_ids] * self.m
        initial_results = self.generate_with_vllm(initial_prompt_ids)
        
        for idx, (token_ids, log_probs, finish_reason) in enumerate(initial_results):
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                decode_fn=self.decode_fn,
                token_id_list=token_ids,
                log_prob_list=log_probs,
                is_end=True,
                finish_reason=finish_reason,
                max_length=self.max_new_tokens
            )
            tree_lists.append([root_node])
        
        # === Step 2: Expand trees for L iterations ===
        for iteration in range(self.l):
            print(f"Expansion iteration {iteration + 1}/{self.l}")
            
            expansion_tasks = []
            
            for tree_idx, tree_list in enumerate(tree_lists):
                tree_entropy_tokens = []
                
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):
                        entropy_tokens = node.get_max_entropy_tokens(top_n=self.n)
                        for token_idx in entropy_tokens:
                            entropy_value = -node.log_prob_list[token_idx]
                            tree_entropy_tokens.append(
                                (entropy_value, tree_idx, node_idx, node, token_idx)
                            )
                
                tree_entropy_tokens.sort(reverse=True)
                expansion_tasks.extend([
                    (t[1], t[2], t[3], t[4])
                    for t in tree_entropy_tokens[:self.n]
                ])
            
            if not expansion_tasks:
                print("No expandable nodes, terminating")
                break
            
            # === Generate expansions ===
            m_tree_top_n_prompt_ids = []
            task_mapping = {}
            
            for i, (tree_idx, node_idx, node, split_idx) in enumerate(expansion_tasks * self.t):
                prefix_ids = node.get_prefix_ids(split_idx)
                prompt_ids_full = prompt_ids + prefix_ids
                m_tree_top_n_prompt_ids.append(prompt_ids_full)
                task_mapping[i] = (tree_idx, node_idx, node, split_idx)
            
            inference_results = self.generate_with_vllm(m_tree_top_n_prompt_ids)
            
            for i, (token_ids, log_probs, finish_reason) in enumerate(inference_results):
                tree_idx, node_idx, parent_node, split_idx = task_mapping[i]
                
                new_node = TreeNode(
                    tree_idx=tree_idx,
                    node_idx=len(tree_lists[tree_idx]),
                    token_id_list=token_ids,
                    decode_fn=self.decode_fn,
                    log_prob_list=log_probs,
                    is_end=True,
                    parent_node=parent_node,
                    parent_node_idx=node_idx,
                    parent_node_split_idx=split_idx,
                    finish_reason=finish_reason,
                    max_length=self.max_new_tokens
                )
                
                parent_node.add_child(new_node, split_idx)
                tree_lists[tree_idx].append(new_node)
        
        # === Step 3: Evaluate all leaf nodes ===
        pass_k_result = []
        for tree_list in tree_lists:
            for node in tree_list:
                if node.is_end:
                    binary_score, score = self.evaluate_node(problem, node, answer)
                    node.binary_score = binary_score
                    node.score = score
                    pass_k_result.append(binary_score)
                else:
                    node.binary_score = 0
                    node.score = 0
        
        print(f"Pass rate: {sum(pass_k_result)/len(pass_k_result):.2%}")
        
        # === Step 4: Build tree and select paths ===
        root, selected_terminals = build_into_tree_format(
            tree_lists, self.decode_fn, self.num_traces
        )
        
        # === Step 5: Gather paths ===
        paths = gather_paths(root, selected_terminals, self.num_traces)
        
        avg_reward = root.reward_raw if root.reward_raw is not None else sum(pass_k_result) / len(pass_k_result)
        
        print(f"Total search time: {time.time() - time_start:.2f}s")
        
        return paths, avg_reward
    
    def generate_sequences_with_tree_search(
        self,
        batch: DataProto,
    ) -> DataProto:
        """
        Generate sequences using tree search for a batch of prompts.
        
        This is the main entry point that integrates with verl's training loop.
        
        Args:
            batch: DataProto containing prompts and ground truth answers
        
        Returns:
            DataProto with sequences, token-level rewards, and masks
        """
        all_sequences = []
        all_rewards = []
        all_attention_masks = []
        all_action_masks = []
        prompt_indices = []  # For RLOO grouping
        
        batch_size = len(batch.batch["input_ids"])
        
        for i in range(batch_size):
            # Get prompt and ground truth
            prompt_ids = batch.batch["input_ids"][i]
            if "attention_mask" in batch.batch:
                valid_len = batch.batch["attention_mask"][i].sum().item()
                prompt_ids = prompt_ids[-valid_len:].tolist()
            else:
                prompt_ids = prompt_ids.tolist()
            
            # Get ground truth
            if "reward_model" in batch.non_tensor_batch:
                ground_truth = batch.non_tensor_batch["reward_model"][i].get("ground_truth", "")
            else:
                ground_truth = ""
            
            # Decode prompt for evaluator
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # Perform tree search
            paths, avg_reward = self.search(prompt_text, ground_truth, prompt_ids)
            
            if paths is None:
                print(f"Warning: No valid paths for prompt {i}, skipping")
                continue
            
            # Convert paths to tensors
            prompt_len = len(prompt_ids)
            max_output_len = 0
            for path in paths:
                output_len = sum(len(node["token_answer"]) for node in path)
                max_output_len = max(max_output_len, output_len)
            max_output_len = min(max_output_len, self.max_new_tokens)
            
            for path in paths:
                # Collect output tokens and rewards
                output_token_ids = []
                rewards = []
                
                for node in path:
                    output_token_ids.extend(node["token_answer"])
                    value = node["value"]
                    rewards.extend([value] * len(node["token_answer"]))
                
                # Truncate if needed
                if len(output_token_ids) > max_output_len:
                    output_token_ids = output_token_ids[:max_output_len]
                    rewards = rewards[:max_output_len]
                
                output_len = len(output_token_ids)
                
                # Pad
                output_ids = output_token_ids + [self.tokenizer.pad_token_id] * (max_output_len - output_len)
                rewards = rewards + [0.0] * (max_output_len - output_len)
                
                # Create full sequence
                full_sequence = prompt_ids + output_ids
                all_sequences.append(full_sequence)
                all_rewards.append(rewards)
                
                # Create masks
                attention_mask = [1] * len(full_sequence)
                all_attention_masks.append(attention_mask)
                
                action_mask = [0] * prompt_len + [1] * output_len + [0] * (max_output_len - output_len)
                all_action_masks.append(action_mask)
                
                prompt_indices.append(i)  # Same prompt index for RLOO
        
        if not all_sequences:
            raise ValueError("No valid sequences generated")
        
        # Pad all sequences to same length
        max_seq_len = max(len(s) for s in all_sequences)
        padded_sequences = []
        padded_attention_masks = []
        
        for seq, mask in zip(all_sequences, all_attention_masks):
            pad_len = max_seq_len - len(seq)
            padded_sequences.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_masks.append(mask + [0] * pad_len)
        
        # Create output DataProto
        output = DataProto(
            batch={
                "input_ids": torch.tensor(padded_sequences),
                "attention_mask": torch.tensor(padded_attention_masks),
                "token_level_scores": torch.tensor(all_rewards, dtype=torch.float32),
                "response_mask": torch.tensor(all_action_masks),
            },
            meta_info={
                "prompt_indices": prompt_indices,  # For RLOO
            }
        )
        
        # Add prompt info
        prompt_length = len(prompt_ids)
        output.batch["prompts"] = output.batch["input_ids"][:, :prompt_length]
        output.batch["responses"] = output.batch["input_ids"][:, prompt_length:]
        
        return output
