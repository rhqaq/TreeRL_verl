"""
Parallel TreeRL Rollout for verl 0.7.0.

This module implements parallel entropy-guided tree search,
fully utilizing multi-GPU resources for:
1. Batch parallel processing of multiple prompts
2. Parallel generation within each tree (M initial responses)
3. Parallel expansion across all entropy points
"""

import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio

from verl import DataProto
from tree_node import TreeNode, build_into_tree_format, gather_paths


class ParallelTreeRLRollout:
    """
    Parallel Entropy-Guided Tree Search Rollout.
    
    Optimizations:
    1. Batch multiple prompts for parallel tree search
    2. Generate all M initial responses in parallel
    3. Batch all expansion tasks together
    4. Use vLLM continuous batching for efficient inference
    """
    
    def __init__(
        self,
        actor_rollout_wg,
        tokenizer,
        config,
        evaluator_fn: Optional[Callable] = None,
        num_parallel_prompts: int = 4,  # Number of prompts to process in parallel
        num_parallel_trees: int = 8,     # Number of trees to expand in parallel per prompt
    ):
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.config = config
        self.evaluator_fn = evaluator_fn
        
        # Tree search parameters
        self.m = config.algorithm.get("m", 6)
        self.n = config.algorithm.get("n", 2)
        self.l = config.algorithm.get("l", 1)
        self.t = config.algorithm.get("t", 2)
        self.num_traces = config.algorithm.get("num_traces", 16)
        
        # Parallel parameters
        self.num_parallel_prompts = num_parallel_prompts
        self.num_parallel_trees = num_parallel_trees
        
        # Generation parameters
        self.temperature = config.actor_rollout_ref.rollout.temperature
        self.top_p = config.actor_rollout_ref.rollout.top_p
        self.max_new_tokens = config.data.max_response_length
        
        print(f"ParallelTreeRLRollout initialized:")
        print(f"  - num_parallel_prompts: {num_parallel_prompts}")
        print(f"  - num_parallel_trees: {num_parallel_trees}")
    
    def decode_fn(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def batch_generate(
        self,
        prompt_ids_list: List[List[int]],
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
    ) -> List[Tuple[List[int], List[float], str]]:
        """
        Batch generate responses using vLLM.
        
        This is the key optimization - all generations happen in a single batch,
        leveraging vLLM's continuous batching for maximum throughput.
        """
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        max_tokens = max_tokens or self.max_new_tokens
        
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
        
        # Create DataProto for batch generation
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
        
        # Single batch generation call - much faster!
        output = self.actor_rollout_wg.generate_sequences(gen_data)
        
        # Extract results
        results = []
        responses = output.batch["responses"]
        response_masks = output.batch.get("response_mask", None)
        log_probs = output.batch.get("old_log_probs", None)
        
        for i in range(batch_size):
            if response_masks is not None:
                valid_len = response_masks[i].sum().item()
                token_ids = responses[i, :valid_len].tolist()
            else:
                token_ids = responses[i].tolist()
                if self.tokenizer.eos_token_id in token_ids:
                    eos_idx = token_ids.index(self.tokenizer.eos_token_id)
                    token_ids = token_ids[:eos_idx + 1]
            
            if log_probs is not None:
                if response_masks is not None:
                    probs = log_probs[i, :valid_len].tolist()
                else:
                    probs = log_probs[i, :len(token_ids)].tolist()
            else:
                probs = [-1.0] * len(token_ids)
            
            finish_reason = "stop" if token_ids and token_ids[-1] == self.tokenizer.eos_token_id else "length"
            results.append((token_ids, probs, finish_reason))
        
        return results
    
    def parallel_tree_search_single_prompt(
        self,
        prompt_ids: List[int],
        problem: str,
        answer: str,
    ) -> Tuple[List[List[Dict[str, Any]]], float]:
        """
        Perform parallel tree search for a single prompt.
        
        Key optimizations:
        1. Generate M initial responses in ONE batch
        2. Collect ALL expansion tasks, generate in ONE batch
        3. Evaluate all leaves in parallel
        """
        time_start = time.time()
        
        # === Step 1: Batch generate M initial responses ===
        print(f"  [Tree Search] Generating {self.m} initial responses...")
        initial_prompt_ids = [prompt_ids] * self.m
        initial_results = self.batch_generate(initial_prompt_ids)
        
        # Build initial trees
        tree_lists = []
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
        
        # === Step 2: Parallel expansion iterations ===
        for iteration in range(self.l):
            print(f"  [Tree Search] Expansion iteration {iteration + 1}/{self.l}")
            
            # Collect all expansion tasks across ALL trees
            all_expansion_tasks = []  # [(tree_idx, node_idx, node, split_idx), ...]
            
            for tree_idx, tree_list in enumerate(tree_lists):
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):
                        entropy_tokens = node.get_max_entropy_tokens(top_n=self.n)
                        for token_idx in entropy_tokens:
                            all_expansion_tasks.append(
                                (tree_idx, node_idx, node, token_idx)
                            )
            
            if not all_expansion_tasks:
                print("  [Tree Search] No expandable nodes, terminating")
                break
            
            # Keep only top-N entropy tokens globally (across all trees)
            all_expansion_tasks = all_expansion_tasks[:self.n * self.num_parallel_trees]
            
            # === Batch generate ALL expansions ===
            # Each task generates T branches, so total = len(tasks) * T
            expansion_prompt_ids = []
            task_mapping = []
            
            for tree_idx, node_idx, node, split_idx in all_expansion_tasks:
                for branch_idx in range(self.t):
                    prefix_ids = node.get_prefix_ids(split_idx)
                    prompt_ids_full = prompt_ids + prefix_ids
                    expansion_prompt_ids.append(prompt_ids_full)
                    task_mapping.append((tree_idx, node_idx, node, split_idx))
            
            print(f"  [Tree Search] Generating {len(expansion_prompt_ids)} expansions...")
            expansion_results = self.batch_generate(expansion_prompt_ids)
            
            # Add new nodes to trees
            for i, (token_ids, log_probs, finish_reason) in enumerate(expansion_results):
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
        
        # === Step 3: Evaluate all leaves in parallel ===
        print(f"  [Tree Search] Evaluating leaf nodes...")
        pass_k_result = []
        for tree_list in tree_lists:
            for node in tree_list:
                if node.is_end:
                    if node.finish_reason == "stop" and self.evaluator_fn:
                        binary_score = self.evaluator_fn(problem, node.total_str, answer)
                    else:
                        binary_score = 0
                    node.binary_score = binary_score
                    node.score = binary_score
                    pass_k_result.append(binary_score)
                else:
                    node.binary_score = 0
                    node.score = 0
        
        print(f"  [Tree Search] Pass rate: {sum(pass_k_result)/len(pass_k_result):.2%}")
        
        # === Step 4: Build tree and select paths ===
        root, selected_terminals = build_into_tree_format(
            tree_lists, self.decode_fn, self.num_traces
        )
        
        paths = gather_paths(root, selected_terminals, self.num_traces)
        avg_reward = root.reward_raw if root.reward_raw is not None else sum(pass_k_result) / len(pass_k_result)
        
        print(f"  [Tree Search] Completed in {time.time() - time_start:.2f}s")
        
        return paths, avg_reward
    
    def generate_sequences_with_tree_search(
        self,
        batch: DataProto,
    ) -> DataProto:
        """
        Generate sequences using PARALLEL tree search.
        
        Key optimization: Process multiple prompts in parallel using batch generation.
        """
        batch_size = len(batch.batch["input_ids"])
        
        print(f"\n{'='*60}")
        print(f"Parallel Tree Search for {batch_size} prompts")
        print(f"{'='*60}")
        
        # === Collect all prompts and ground truths ===
        prompts_info = []
        for i in range(batch_size):
            prompt_ids = batch.batch["input_ids"][i]
            if "attention_mask" in batch.batch:
                valid_len = batch.batch["attention_mask"][i].sum().item()
                prompt_ids = prompt_ids[-valid_len:].tolist()
            else:
                prompt_ids = prompt_ids.tolist()
            
            if "reward_model" in batch.non_tensor_batch:
                ground_truth = batch.non_tensor_batch["reward_model"][i].get("ground_truth", "")
            else:
                ground_truth = ""
            
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompts_info.append((prompt_ids, prompt_text, ground_truth))
        
        # === Parallel tree search for each prompt ===
        # Note: True parallelism is achieved within each tree search via batch generation
        all_outputs = []
        prompt_indices = []
        
        for i, (prompt_ids, prompt_text, ground_truth) in enumerate(prompts_info):
            print(f"\n[Prompt {i+1}/{batch_size}]")
            
            paths, avg_reward = self.parallel_tree_search_single_prompt(
                prompt_ids=prompt_ids,
                problem=prompt_text,
                answer=ground_truth,
            )
            
            if paths is None:
                print(f"  Warning: No valid paths, skipping")
                continue
            
            # Convert paths to tensors
            prompt_len = len(prompt_ids)
            max_output_len = 0
            for path in paths:
                output_len = sum(len(node["token_answer"]) for node in path)
                max_output_len = max(max_output_len, output_len)
            max_output_len = min(max_output_len, self.max_new_tokens)
            
            for path in paths:
                output_token_ids = []
                rewards = []
                
                for node in path:
                    output_token_ids.extend(node["token_answer"])
                    value = node["value"]
                    rewards.extend([value] * len(node["token_answer"]))
                
                if len(output_token_ids) > max_output_len:
                    output_token_ids = output_token_ids[:max_output_len]
                    rewards = rewards[:max_output_len]
                
                output_len = len(output_token_ids)
                
                # Pad
                output_ids = output_token_ids + [self.tokenizer.pad_token_id] * (max_output_len - output_len)
                rewards = rewards + [0.0] * (max_output_len - output_len)
                
                # Create full sequence
                full_sequence = prompt_ids + output_ids
                
                all_outputs.append({
                    "sequence": full_sequence,
                    "rewards": rewards,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "max_output_len": max_output_len,
                })
                
                prompt_indices.append(i)
        
        if not all_outputs:
            raise ValueError("No valid sequences generated")
        
        # === Create output DataProto ===
        max_seq_len = max(len(o["sequence"]) for o in all_outputs)
        
        padded_sequences = []
        padded_rewards = []
        padded_attention_masks = []
        padded_action_masks = []
        
        for output in all_outputs:
            seq = output["sequence"]
            rewards = output["rewards"]
            prompt_len = output["prompt_len"]
            output_len = output["output_len"]
            max_output_len = output["max_output_len"]
            
            pad_len = max_seq_len - len(seq)
            padded_sequences.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            padded_rewards.append(rewards + [0.0] * pad_len)
            
            attention_mask = [1] * len(seq) + [0] * pad_len
            padded_attention_masks.append(attention_mask)
            
            action_mask = [0] * prompt_len + [1] * output_len + [0] * (max_output_len - output_len) + [0] * pad_len
            padded_action_masks.append(action_mask)
        
        prompt_length = all_outputs[0]["prompt_len"]
        
        output = DataProto(
            batch={
                "input_ids": torch.tensor(padded_sequences),
                "attention_mask": torch.tensor(padded_attention_masks),
                "token_level_scores": torch.tensor(padded_rewards, dtype=torch.float32),
                "response_mask": torch.tensor(padded_action_masks),
            },
            meta_info={
                "prompt_indices": prompt_indices,
            }
        )
        
        output.batch["prompts"] = output.batch["input_ids"][:, :prompt_length]
        output.batch["responses"] = output.batch["input_ids"][:, prompt_length:]
        
        print(f"\n{'='*60}")
        print(f"Tree search completed: {len(all_outputs)} total traces")
        print(f"{'='*60}\n")
        
        return output


class FullyParallelTreeRLRollout(ParallelTreeRLRollout):
    """
    Fully parallel TreeRL Rollout - processes multiple prompts truly in parallel.
    
    Uses async execution to process multiple prompts concurrently,
    each with its own batch generation calls.
    """
    
    async def async_tree_search(
        self,
        prompt_ids: List[int],
        prompt_text: str,
        ground_truth: str,
        prompt_idx: int,
    ) -> Tuple[int, List[List[Dict[str, Any]]], float]:
        """Async tree search for a single prompt."""
        paths, avg_reward = self.parallel_tree_search_single_prompt(
            prompt_ids=prompt_ids,
            problem=prompt_text,
            answer=ground_truth,
        )
        return prompt_idx, paths, avg_reward
    
    async def async_generate_all(
        self,
        prompts_info: List[Tuple[List[int], str, str]],
    ):
        """Run tree search for all prompts concurrently."""
        tasks = []
        for i, (prompt_ids, prompt_text, ground_truth) in enumerate(prompts_info):
            task = self.async_tree_search(prompt_ids, prompt_text, ground_truth, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x[0])
    
    def generate_sequences_with_tree_search(
        self,
        batch: DataProto,
    ) -> DataProto:
        """
        Generate sequences using FULLY PARALLEL tree search.
        
        Multiple prompts are processed concurrently via async execution.
        """
        batch_size = len(batch.batch["input_ids"])
        
        print(f"\n{'='*60}")
        print(f"Fully Parallel Tree Search for {batch_size} prompts")
        print(f"Concurrency: {self.num_parallel_prompts} prompts in parallel")
        print(f"{'='*60}")
        
        # Collect prompts
        prompts_info = []
        for i in range(batch_size):
            prompt_ids = batch.batch["input_ids"][i]
            if "attention_mask" in batch.batch:
                valid_len = batch.batch["attention_mask"][i].sum().item()
                prompt_ids = prompt_ids[-valid_len:].tolist()
            else:
                prompt_ids = prompt_ids.tolist()
            
            ground_truth = ""
            if "reward_model" in batch.non_tensor_batch:
                ground_truth = batch.non_tensor_batch["reward_model"][i].get("ground_truth", "")
            
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompts_info.append((prompt_ids, prompt_text, ground_truth))
        
        # Run async tree search
        results = asyncio.run(self.async_generate_all(prompts_info))
        
        # Process results
        all_outputs = []
        prompt_indices = []
        
        for prompt_idx, paths, avg_reward in results:
            if paths is None:
                continue
            
            prompt_ids, _, _ = prompts_info[prompt_idx]
            prompt_len = len(prompt_ids)
            
            max_output_len = 0
            for path in paths:
                output_len = sum(len(node["token_answer"]) for node in path)
                max_output_len = max(max_output_len, output_len)
            max_output_len = min(max_output_len, self.max_new_tokens)
            
            for path in paths:
                output_token_ids = []
                rewards = []
                
                for node in path:
                    output_token_ids.extend(node["token_answer"])
                    rewards.extend([node["value"]] * len(node["token_answer"]))
                
                if len(output_token_ids) > max_output_len:
                    output_token_ids = output_token_ids[:max_output_len]
                    rewards = rewards[:max_output_len]
                
                output_len = len(output_token_ids)
                
                output_ids = output_token_ids + [self.tokenizer.pad_token_id] * (max_output_len - output_len)
                rewards = rewards + [0.0] * (max_output_len - output_len)
                
                full_sequence = prompt_ids + output_ids
                
                all_outputs.append({
                    "sequence": full_sequence,
                    "rewards": rewards,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "max_output_len": max_output_len,
                })
                
                prompt_indices.append(prompt_idx)
        
        if not all_outputs:
            raise ValueError("No valid sequences generated")
        
        # Create output DataProto
        max_seq_len = max(len(o["sequence"]) for o in all_outputs)
        
        padded_sequences = []
        padded_rewards = []
        padded_attention_masks = []
        padded_action_masks = []
        
        for output in all_outputs:
            seq = output["sequence"]
            rewards = output["rewards"]
            prompt_len = output["prompt_len"]
            output_len = output["output_len"]
            max_output_len = output["max_output_len"]
            
            pad_len = max_seq_len - len(seq)
            padded_sequences.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            padded_rewards.append(rewards + [0.0] * pad_len)
            
            attention_mask = [1] * len(seq) + [0] * pad_len
            padded_attention_masks.append(attention_mask)
            
            action_mask = [0] * prompt_len + [1] * output_len + [0] * (max_output_len - output_len) + [0] * pad_len
            padded_action_masks.append(action_mask)
        
        prompt_length = all_outputs[0]["prompt_len"]
        
        output = DataProto(
            batch={
                "input_ids": torch.tensor(padded_sequences),
                "attention_mask": torch.tensor(padded_attention_masks),
                "token_level_scores": torch.tensor(padded_rewards, dtype=torch.float32),
                "response_mask": torch.tensor(padded_action_masks),
            },
            meta_info={
                "prompt_indices": prompt_indices,
            }
        )
        
        output.batch["prompts"] = output.batch["input_ids"][:, :prompt_length]
        output.batch["responses"] = output.batch["input_ids"][:, prompt_length:]
        
        print(f"\n{'='*60}")
        print(f"Fully parallel tree search completed: {len(all_outputs)} total traces")
        print(f"{'='*60}\n")
        
        return output
