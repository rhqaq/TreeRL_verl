"""
TreeRL Agent Loop - 熵引导树搜索实现

继承 AgentLoopManager，在 generate_sequences 中实现树搜索算法。
直接使用 math_dapo.compute_score 进行答案验证（不使用 DAPORewardManager）。
"""

import asyncio
import random
import time
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager, AgentLoopWorkerBase
from verl.protocol import DataProto
from verl.utils.reward_score import math_dapo


# ==============================================================================
# Tree Search 参数 - 直接写在这里，不需要从配置文件读取
# ==============================================================================
TREE_SEARCH_CONFIG = {
    "m": 6,              # 初始树的数量 (M 个初始响应)
    "n": 2,              # 每轮扩展的高熵 token 数量
    "l": 1,              # 扩展迭代轮数
    "t": 2,              # 每个熵点的分支数量
    "num_traces": 16,    # 每个 prompt 采样的训练轨迹数
    "max_response_length": 2048,  # 最大响应长度
}


# ==============================================================================
# TreeNode 数据结构
# ==============================================================================
@dataclass
class TreeNode:
    """树节点，用于熵引导的树搜索"""
    tree_idx: int
    node_idx: int
    token_ids: List[int]
    log_probs: List[float]
    parent: Optional['TreeNode'] = None
    parent_split_idx: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    mask: List[bool] = field(default_factory=list)
    binary_score: float = 0.0
    
    def __post_init__(self):
        self.mask = [False] * len(self.token_ids)
        if self.parent is not None and len(self.token_ids) > 0:
            self.mask[0] = True  # 第一个 token 通常是继承的
    
    def get_prefix_ids(self, split_idx: int) -> List[int]:
        """获取到 split_idx 为止的前缀 token ids"""
        prefix = []
        if self.parent is not None:
            prefix = self.parent.get_prefix_ids(self.parent_split_idx)
        prefix.extend(self.token_ids[:split_idx])
        return prefix
    
    def get_full_ids(self) -> List[int]:
        """获取完整序列"""
        if self.parent is not None:
            return self.parent.get_full_ids() + self.token_ids
        return self.token_ids.copy()
    
    def get_high_entropy_tokens(self, top_n: int = 2) -> List[int]:
        """获取高熵（低概率）的 token 索引"""
        # 熵 ≈ -log_prob，概率越低熵越高
        entropies = [(-log_prob, idx) for idx, log_prob in enumerate(self.log_probs) if not self.mask[idx]]
        entropies.sort(reverse=True)
        return [idx for _, idx in entropies[:top_n]]
    
    def update_mask(self, split_idx: int):
        """更新 mask，标记已扩展的部分"""
        for i in range(min(split_idx, len(self.mask))):
            self.mask[i] = True


# ==============================================================================
# RLOO 优势计算
# ==============================================================================
def compute_rloo_advantages(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    masks: torch.Tensor,
    num_samples_per_prompt: int,
    min_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 RLOO (Reinforcement Learning with Opponent Learning) 优势。
    
    RLOO 的核心思想（与 TreeRL 原始实现一致）：
    - 对于每个样本，使用同一 prompt 下其他样本的平均奖励作为 baseline
    - baseline = (sum - current) / (num_samples - 1)
    - advantage = reward - baseline
    - 这避免了需要单独的 value function
    
    Args:
        rewards: [batch_size] 每个样本的奖励
        log_probs: [batch_size, seq_len] 每个 token 的 log probability
        masks: [batch_size, seq_len] 有效 token 的 mask
        num_samples_per_prompt: 每个 prompt 的样本数
        min_threshold: 最小阈值，用于过滤小的优势
        
    Returns:
        advantages: [batch_size, seq_len] token-level advantages
        returns: [batch_size, seq_len] token-level returns
    """
    batch_size = rewards.shape[0]
    
    # 重塑为 [batch_size // num_samples_per_prompt, num_samples_per_prompt]
    # 假设 batch 中 samples 是按 prompt 分组的
    num_prompts = batch_size // num_samples_per_prompt
    rewards_grouped = rewards.view(num_prompts, num_samples_per_prompt)
    
    # 计算 baseline: 其他样本的平均奖励（与 TreeRL 一致）
    # baseline = (_sum - reward) / (num_samples - 1)
    _sum = rewards_grouped.sum(dim=1, keepdim=True)
    baseline = (_sum - rewards_grouped) / (num_samples_per_prompt - 1 + 1e-6)
    
    # 计算优势 = reward - baseline
    advantages_grouped = rewards_grouped - baseline
    
    # 应用 lower_bound 确保优势不会太低（与 TreeRL 一致）
    lower_bound = -advantages_grouped.max(dim=1, keepdim=True)[0]
    advantages_grouped = torch.max(advantages_grouped, lower_bound)
    
    # 展开
    advantages_scalar = advantages_grouped.view(batch_size)
    
    # 扩展为 token-level
    seq_len = log_probs.shape[1] if log_probs.dim() > 1 else masks.shape[1]
    advantages = advantages_scalar.unsqueeze(1).expand(-1, seq_len) * masks
    returns = advantages.clone()
    
    print(f"RLOO: rewards_grouped = {rewards_grouped}")
    print(f"RLOO: baseline = {baseline}")
    print(f"RLOO: advantages_grouped = {advantages_grouped}")
    
    return advantages, returns


# ==============================================================================
# TreeRL AgentLoopManager
# ==============================================================================
class TreeRLAgentLoopManager(AgentLoopManager):
    """
    TreeRL 的 AgentLoopManager 实现。
    
    核心改动：
    1. 重写 generate_sequences，实现熵引导的树搜索
    2. 利用 AsyncLLMServerManager 进行批量生成
    3. 直接使用 math_dapo.compute_score 进行答案验证
    4. 实现 RLOO 优势估计
    """
    
    def __init__(self, config: DictConfig, worker_group=None, rm_resource_pool=None):
        # 设置树搜索参数
        self.m = TREE_SEARCH_CONFIG["m"]
        self.n = TREE_SEARCH_CONFIG["n"]
        self.l = TREE_SEARCH_CONFIG["l"]
        self.t = TREE_SEARCH_CONFIG["t"]
        self.num_traces = TREE_SEARCH_CONFIG["num_traces"]
        self.max_response_length = TREE_SEARCH_CONFIG["max_response_length"]
        
        # 调用父类初始化
        super().__init__(config, worker_group, rm_resource_pool)
        
        # 初始化 tokenizer（用于解码响应文本）
        self._init_tokenizer_for_reward()
        
        print(f"TreeRLAgentLoopManager initialized with:")
        print(f"  m={self.m}, n={self.n}, l={self.l}, t={self.t}, num_traces={self.num_traces}")
        print(f"  Using math_dapo.compute_score for real reward evaluation")
    
    def _init_tokenizer_for_reward(self):
        """初始化 tokenizer 用于奖励计算"""
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        
        model_path = self.config.actor_rollout_ref.model.path
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        重写 generate_sequences，实现熵引导的树搜索。
        
        流程：
        1. 对 batch 中每个 prompt，生成 M 个初始响应
        2. 进行 L 轮扩展，每轮选择 N 个高熵 token 进行 T 分支
        3. 使用 math_dapo.compute_score 评估叶子节点，计算真实奖励
        4. 采样 K 条轨迹用于训练
        5. 计算 RLOO 优势
        """
        # Wake up servers
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()
        
        start_time = time.time()
        
        # 分发到各个 worker
        chunks = prompts.chunk(len(self.agent_loop_workers))
        
        # 使用 TreeRL 的生成逻辑
        outputs = ray.get([
            worker.generate_sequences_tree_search.remote(
                chunk, 
                m=self.m, 
                n=self.n, 
                l=self.l, 
                t=self.t,
                num_traces=self.num_traces,
                max_response_length=self.max_response_length,
                tokenizer=self.tokenizer,
            )
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        
        output = DataProto.concat(outputs)
        
        # 计算 RLOO 优势
        if "token_level_scores" in output.batch:
            rewards = output.batch["token_level_scores"].sum(dim=1)
            log_probs = output.batch.get("rollout_log_probs", torch.zeros_like(rewards).unsqueeze(1))
            masks = output.batch.get("response_mask", torch.ones_like(log_probs))
            
            # 假设每个 prompt 有 num_traces 个样本
            advantages, returns = compute_rloo_advantages(
                rewards=rewards,
                log_probs=log_probs,
                masks=masks,
                num_samples_per_prompt=self.num_traces,
            )
            
            output.batch["advantages"] = advantages
            output.batch["returns"] = returns
        
        # Sleep servers
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()
        
        # 添加 timing 信息
        timing = {
            "tree_search/total": time.time() - start_time,
            "tree_search/m": self.m,
            "tree_search/n": self.n,
            "tree_search/l": self.l,
            "tree_search/t": self.t,
        }
        output.meta_info["timing"] = timing
        
        return output


# ==============================================================================
# TreeRL AgentLoopWorker
# ==============================================================================
@ray.remote
class TreeRLAgentLoopWorker(AgentLoopWorkerBase):
    """
    TreeRL 的 AgentLoopWorker 实现。
    
    继承自 AgentLoopWorkerBase（普通类），然后用 @ray.remote 装饰。
    Ray 不支持继承一个已经是 @ray.remote 装饰的类，所以我们继承基类。
    
    核心方法 generate_sequences_tree_search 实现树搜索逻辑。
    """
    
    async def generate_sequences_tree_search(
        self,
        batch: DataProto,
        m: int = 6,
        n: int = 2,
        l: int = 1,
        t: int = 2,
        num_traces: int = 16,
        max_response_length: int = 2048,
        tokenizer=None,
    ) -> DataProto:
        """
        对一个 batch 的 prompts 执行树搜索。
        
        使用 asyncio 并行处理多个 prompts。
        """
        batch_size = len(batch.batch["input_ids"])
        
        # 并行处理每个 prompt
        tasks = [
            self._tree_search_single_prompt(
                batch_idx=i,
                batch=batch,
                m=m, n=n, l=l, t=t,
                num_traces=num_traces,
                max_response_length=max_response_length,
                tokenizer=tokenizer,
            )
            for i in range(batch_size)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        return self._merge_tree_search_results(results, batch)
    
    async def _tree_search_single_prompt(
        self,
        batch_idx: int,
        batch: DataProto,
        m: int,
        n: int,
        l: int,
        t: int,
        num_traces: int,
        max_response_length: int,
        tokenizer=None,
    ) -> Dict[str, Any]:
        """
        对单个 prompt 执行树搜索。
        
        返回：{
            "prompt_ids": List[int],
            "trace_ids": List[List[int]],  # num_traces 条轨迹
            "trace_log_probs": List[List[float]],
            "trace_rewards": List[float],  # 每条轨迹的奖励（由 math_dapo.compute_score 计算）
            "trace_masks": List[List[int]],  # 哪些是生成的 token
            "ground_truth": Any,  # 用于奖励计算
            "data_source": str,   # 数据来源
        }
        """
        # 获取 prompt
        prompt_ids = batch.batch["input_ids"][batch_idx]
        if "attention_mask" in batch.batch:
            valid_len = batch.batch["attention_mask"][batch_idx].sum().item()
            prompt_ids = prompt_ids[-valid_len:].tolist()
        else:
            prompt_ids = prompt_ids.tolist()
        
        # 获取 ground_truth 和 data_source（用于奖励计算）
        ground_truth = None
        if "reward_model" in batch.non_tensor_batch:
            reward_info = batch.non_tensor_batch["reward_model"]
            if reward_info is not None and hasattr(reward_info, '__getitem__'):
                reward_item = reward_info[batch_idx]
                if isinstance(reward_item, dict) and "ground_truth" in reward_item:
                    ground_truth = reward_item["ground_truth"]
        
        data_source = batch.non_tensor_batch.get("data_source", "math_dapo")
        if hasattr(data_source, '__getitem__'):
            data_source = data_source[batch_idx]
        
        # Step 1: 生成 M 个初始响应
        initial_trees = await self._generate_initial_responses(
            prompt_ids=prompt_ids,
            m=m,
            max_response_length=max_response_length,
        )
        
        # Step 2: 扩展迭代
        for iteration in range(l):
            initial_trees = await self._expand_trees(
                trees=initial_trees,
                n=n,
                t=t,
                prompt_ids=prompt_ids,
                max_response_length=max_response_length,
            )
        
        # Step 3: 使用 math_dapo.compute_score 评估叶子节点
        initial_trees = await self._evaluate_leaves(
            trees=initial_trees,
            ground_truth=ground_truth,
            tokenizer=tokenizer,
        )
        
        # Step 4: 采样轨迹（已包含真实奖励）
        traces = self._sample_traces(
            trees=initial_trees,
            num_traces=num_traces,
        )
        
        return {
            "prompt_ids": prompt_ids,
            "batch_idx": batch_idx,
            "trace_ids": [t["token_ids"] for t in traces],
            "trace_log_probs": [t["log_probs"] for t in traces],
            "trace_masks": [t["masks"] for t in traces],
            "trace_rewards": [t["reward"] for t in traces],
            "ground_truth": ground_truth,
            "data_source": data_source,
        }
    
    async def _generate_initial_responses(
        self,
        prompt_ids: List[int],
        m: int,
        max_response_length: int,
    ) -> List[TreeNode]:
        """生成 M 个初始响应，构建初始树"""
        # 构造 M 个相同的 prompt
        prompts = [prompt_ids] * m
        
        # 批量生成
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": max_response_length,
        }
        
        # 使用 server_manager 并行生成
        results = await asyncio.gather(*[
            self.server_manager.generate(
                request_id=f"init_{i}",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            for i, prompt_ids in enumerate(prompts)
        ])
        
        # 构建初始树节点
        trees = []
        for idx, result in enumerate(results):
            token_ids = result["token_ids"]
            log_probs = result["log_probs"]
            
            node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                token_ids=token_ids,
                log_probs=log_probs,
            )
            trees.append(node)
        
        return trees
    
    async def _expand_trees(
        self,
        trees: List[TreeNode],
        n: int,
        t: int,
        prompt_ids: List[int],
        max_response_length: int,
    ) -> List[TreeNode]:
        """
        扩展树：找到所有树中最高熵的 N 个 token，每个生成 T 个分支。
        """
        # 收集所有高熵 token
        entropy_candidates = []  # (entropy, tree_idx, node, token_idx)
        
        for tree_idx, tree in enumerate(trees):
            # 遍历树中的所有节点
            all_nodes = self._get_all_nodes(tree)
            for node in all_nodes:
                high_entropy_tokens = node.get_high_entropy_tokens(top_n=n)
                for token_idx in high_entropy_tokens:
                    entropy = -node.log_probs[token_idx]
                    entropy_candidates.append((entropy, tree_idx, node, token_idx))
        
        # 选择最高的 N 个
        entropy_candidates.sort(reverse=True)
        expansion_points = entropy_candidates[:n]
        
        if not expansion_points:
            return trees
        
        # 构造扩展 prompt
        expansion_prompts = []
        expansion_info = []  # 记录扩展点信息
        
        for entropy, tree_idx, node, token_idx in expansion_points:
            for branch_idx in range(t):
                prefix = node.get_prefix_ids(token_idx)
                full_prompt = prompt_ids + prefix
                expansion_prompts.append(full_prompt)
                expansion_info.append((tree_idx, node, token_idx))
        
        # 批量生成扩展
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": max_response_length - len(prompt_ids),
        }
        
        results = await asyncio.gather(*[
            self.server_manager.generate(
                request_id=f"expand_{i}",
                prompt_ids=prompt,
                sampling_params=sampling_params,
            )
            for i, prompt in enumerate(expansion_prompts)
        ])
        
        # 添加新节点到树
        for i, result in enumerate(results):
            tree_idx, parent_node, split_idx = expansion_info[i]
            
            new_node = TreeNode(
                tree_idx=tree_idx,
                node_idx=len(self._get_all_nodes(trees[tree_idx])),
                token_ids=result["token_ids"],
                log_probs=result["log_probs"],
                parent=parent_node,
                parent_split_idx=split_idx,
            )
            
            parent_node.children.append(new_node)
            parent_node.update_mask(split_idx)
        
        return trees
    
    async def _evaluate_leaves(
        self,
        trees: List[TreeNode],
        ground_truth: Optional[str],
        tokenizer=None,
    ) -> List[TreeNode]:
        """
        使用 math_dapo.compute_score 评估叶子节点的答案。
        
        这替代了 DAPORewardManager，直接在 Worker 中进行评估。
        """
        for tree in trees:
            leaf_nodes = self._get_leaf_nodes(tree)
            for leaf in leaf_nodes:
                # 获取完整响应
                full_ids = leaf.get_full_ids()
                
                # 解码响应文本
                if tokenizer is not None:
                    response_text = tokenizer.decode(full_ids, skip_special_tokens=True)
                else:
                    # 如果没有 tokenizer，使用占位符（实际使用时必须传入 tokenizer）
                    response_text = str(full_ids)
                
                # 使用 math_dapo.compute_score 计算奖励
                if ground_truth is not None:
                    try:
                        result = math_dapo.compute_score(
                            solution_str=response_text,
                            ground_truth=ground_truth,
                        )
                        # result 是字典，包含 score, acc, pred
                        leaf.binary_score = float(result.get("score", 0.0))
                    except Exception as e:
                        print(f"Warning: compute_score failed: {e}")
                        leaf.binary_score = 0.0
                else:
                    leaf.binary_score = 0.0
        
        return trees
    
    def _get_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """获取树中所有节点"""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _get_leaf_nodes(self, root: TreeNode) -> List[TreeNode]:
        """获取所有叶子节点"""
        if not root.children:
            return [root]
        leaves = []
        for child in root.children:
            leaves.extend(self._get_leaf_nodes(child))
        return leaves
    
    def _sample_traces(
        self,
        trees: List[TreeNode],
        num_traces: int,
    ) -> List[Dict[str, Any]]:
        """
        从树中采样轨迹用于训练。
        
        策略：
        1. 收集所有叶子节点（已包含奖励）
        2. 按奖励加权采样（高奖励的轨迹被采样的概率更高）
        """
        all_leaves = []
        for tree in trees:
            all_leaves.extend(self._get_leaf_nodes(tree))
        
        if not all_leaves:
            return []
        
        # 计算采样权重（基于奖励）
        rewards = np.array([leaf.binary_score for leaf in all_leaves])
        # 使用 softmax 归一化，确保高奖励的轨迹有更高的概率
        if rewards.std() > 1e-6:
            weights = np.exp(rewards - rewards.max())  # 数值稳定性
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(all_leaves)) / len(all_leaves)
        
        # 加权采样
        sampled_indices = np.random.choice(
            len(all_leaves),
            size=min(num_traces, len(all_leaves)),
            replace=len(all_leaves) < num_traces,
            p=weights,
        )
        sampled = [all_leaves[i] for i in sampled_indices]
        
        # 构建轨迹
        traces = []
        for leaf in sampled:
            # 从叶子节点回溯到根
            token_ids = []
            log_probs = []
            masks = []  # 1 表示生成的 token
            
            path = self._get_path_to_root(leaf)
            for node in path:
                token_ids.extend(node.token_ids)
                log_probs.extend(node.log_probs)
                masks.extend([1] * len(node.token_ids))
            
            traces.append({
                "token_ids": token_ids,
                "log_probs": log_probs,
                "masks": masks,
                "reward": leaf.binary_score,  # 真实奖励
            })
        
        return traces
    
    def _get_path_to_root(self, node: TreeNode) -> List[TreeNode]:
        """获取从根到当前节点的路径"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def _merge_tree_search_results(
        self,
        results: List[Dict[str, Any]],
        original_batch: DataProto,
    ) -> DataProto:
        """合并树搜索结果为 DataProto"""
        all_prompt_ids = []
        all_response_ids = []
        all_response_masks = []
        all_log_probs = []
        all_rewards = []
        all_ground_truths = []
        all_data_sources = []
        
        for result in results:
            prompt_ids = result["prompt_ids"]
            trace_ids = result["trace_ids"]
            trace_masks = result["trace_masks"]
            trace_log_probs = result["trace_log_probs"]
            trace_rewards = result["trace_rewards"]
            ground_truth = result["ground_truth"]
            data_source = result["data_source"]
            
            for i in range(len(trace_ids)):
                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(trace_ids[i])
                all_response_masks.append(trace_masks[i])
                all_log_probs.append(trace_log_probs[i])
                all_rewards.append(trace_rewards[i])
                all_ground_truths.append(ground_truth)
                all_data_sources.append(data_source)
        
        if not all_prompt_ids:
            # 返回空的 DataProto
            return DataProto(
                batch=TensorDict({}, batch_size=0),
                non_tensor_batch={},
                meta_info={},
            )
        
        # Pad to same length
        max_prompt_len = max(len(p) for p in all_prompt_ids)
        max_response_len = max(len(r) for r in all_response_ids)
        
        # 左填充 prompt
        padded_prompts = []
        for p in all_prompt_ids:
            pad_len = max_prompt_len - len(p)
            padded_prompts.append([0] * pad_len + p)
        
        # 右填充 response
        padded_responses = []
        padded_masks = []
        padded_log_probs = []
        
        for i, (r, m, lp) in enumerate(zip(
            all_response_ids, all_response_masks, all_log_probs
        )):
            pad_len = max_response_len - len(r)
            padded_responses.append(r + [0] * pad_len)
            padded_masks.append(m + [0] * pad_len)
            padded_log_probs.append(lp + [0.0] * pad_len)
        
        # 构造 token-level rewards（只在最后一个 token 给予奖励）
        token_rewards = torch.zeros(len(padded_responses), max_response_len)
        for i, (mask, reward) in enumerate(zip(padded_masks, all_rewards)):
            # 找到最后一个有效 token
            valid_len = sum(mask)
            if valid_len > 0:
                token_rewards[i, valid_len - 1] = reward
        
        # 构造 DataProto
        batch = TensorDict({
            "prompts": torch.tensor(padded_prompts),
            "responses": torch.tensor(padded_responses),
            "response_mask": torch.tensor(padded_masks),
            "rollout_log_probs": torch.tensor(padded_log_probs),
            "input_ids": torch.tensor([p + r for p, r in zip(padded_prompts, padded_responses)]),
            "attention_mask": torch.ones(len(padded_prompts), max_prompt_len + max_response_len),
            "token_level_scores": token_rewards,
        }, batch_size=len(padded_prompts))
        
        # 添加 non_tensor_batch 用于奖励计算
        non_tensor_batch = {
            "reward_model": {
                "ground_truth": np.array(all_ground_truths, dtype=object),
            },
            "data_source": np.array(all_data_sources, dtype=object),
        }
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})
