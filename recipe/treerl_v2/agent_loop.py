"""
TreeRL Agent Loop - 熵引导树搜索实现

继承 AgentLoopManager，在 generate_sequences 中实现树搜索算法。
使用 DAPORewardManager 进行真实奖励评估。
"""

import asyncio
import time
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.protocol import DataProto
from verl.workers.reward_manager import DAPORewardManager


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
# TreeRL AgentLoopManager
# ==============================================================================
class TreeRLAgentLoopManager(AgentLoopManager):
    """
    TreeRL 的 AgentLoopManager 实现。
    
    核心改动：
    1. 重写 generate_sequences，实现熵引导的树搜索
    2. 利用 AsyncLLMServerManager 进行批量生成
    3. 使用 DAPORewardManager 进行真实奖励评估
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
        
        # 初始化 tokenizer（用于 DAPORewardManager）
        # 从 worker 获取或单独加载
        self._init_tokenizer_for_reward()
        
        # 初始化 DAPORewardManager（用于评估答案）
        self.reward_manager = DAPORewardManager(
            tokenizer=self.tokenizer,
            num_examine=10,  # 打印前 10 个样本用于调试
            compute_score=None,  # 使用 default_compute_score
            reward_fn_key="data_source",
            max_resp_len=self.max_response_length,
            overlong_buffer_cfg=None,
        )
        
        print(f"TreeRLAgentLoopManager initialized with:")
        print(f"  m={self.m}, n={self.n}, l={self.l}, t={self.t}, num_traces={self.num_traces}")
        print(f"  Using DAPORewardManager for real reward evaluation")
    
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
        3. 使用 DAPORewardManager 评估叶子节点，计算真实奖励
        4. 采样 K 条轨迹用于训练
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
            )
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        
        output = DataProto.concat(outputs)
        
        # 使用 DAPORewardManager 计算真实奖励
        if "token_level_scores" not in output.batch.keys():
            # 构造 DataProto 用于奖励计算
            reward_input = self._prepare_reward_input(output)
            reward_tensor = self.reward_manager(reward_input)
            output.batch["token_level_scores"] = reward_tensor
        
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
    
    def _prepare_reward_input(self, output: DataProto) -> DataProto:
        """准备奖励计算的输入数据"""
        # DAPORewardManager 需要：
        # - prompts: [bsz, prompt_len]
        # - responses: [bsz, response_len]
        # - attention_mask: [bsz, prompt_len + response_len]
        # - non_tensor_batch["reward_model"]["ground_truth"]
        # - non_tensor_batch["data_source"]
        
        # 确保有必要的字段
        if "prompts" not in output.batch:
            # 从 input_ids 分离 prompt 和 response
            # 假设 prompt_length 存储在 meta_info 或配置中
            prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
            output.batch["prompts"] = output.batch["input_ids"][:, :prompt_length]
            output.batch["responses"] = output.batch["input_ids"][:, prompt_length:]
        
        return output


# ==============================================================================
# TreeRL AgentLoopWorker
# ==============================================================================
@ray.remote
class TreeRLAgentLoopWorker(AgentLoopWorker):
    """
    TreeRL 的 AgentLoopWorker 实现。
    
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
    ) -> Dict[str, Any]:
        """
        对单个 prompt 执行树搜索。
        
        返回：{
            "prompt_ids": List[int],
            "trace_ids": List[List[int]],  # num_traces 条轨迹
            "trace_log_probs": List[List[float]],
            "trace_rewards": List[float],  # 每条轨迹的奖励（稍后由 DAPORewardManager 填充）
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
        ground_truth = batch.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
        if ground_truth is not None and hasattr(ground_truth, '__getitem__'):
            ground_truth = ground_truth[batch_idx]
        
        data_source = batch.non_tensor_batch.get("data_source", "unknown")
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
        
        # Step 3: 采样轨迹（奖励稍后计算）
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
        1. 收集所有叶子节点
        2. 随机采样（因为奖励稍后计算）
        """
        all_leaves = []
        for tree in trees:
            all_leaves.extend(self._get_leaf_nodes(tree))
        
        # 随机打乱
        import random
        random.shuffle(all_leaves)
        
        # 采样 num_traces 条
        sampled = all_leaves[:num_traces]
        
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
        all_ground_truths = []
        all_data_sources = []
        
        for result in results:
            prompt_ids = result["prompt_ids"]
            trace_ids = result["trace_ids"]
            trace_masks = result["trace_masks"]
            trace_log_probs = result["trace_log_probs"]
            ground_truth = result["ground_truth"]
            data_source = result["data_source"]
            
            for i in range(len(trace_ids)):
                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(trace_ids[i])
                all_response_masks.append(trace_masks[i])
                all_log_probs.append(trace_log_probs[i])
                all_ground_truths.append(ground_truth)
                all_data_sources.append(data_source)
        
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
        
        # 构造 DataProto
        batch = TensorDict({
            "prompts": torch.tensor(padded_prompts),
            "responses": torch.tensor(padded_responses),
            "response_mask": torch.tensor(padded_masks),
            "rollout_log_probs": torch.tensor(padded_log_probs),
            "input_ids": torch.tensor([p + r for p, r in zip(padded_prompts, padded_responses)]),
            "attention_mask": torch.ones(len(padded_prompts), max_prompt_len + max_response_len),
        }, batch_size=len(padded_prompts))
        
        # 添加 non_tensor_batch 用于奖励计算
        non_tensor_batch = {
            "reward_model": {
                "ground_truth": np.array(all_ground_truths, dtype=object),
            },
            "data_source": np.array(all_data_sources, dtype=object),
        }
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})
