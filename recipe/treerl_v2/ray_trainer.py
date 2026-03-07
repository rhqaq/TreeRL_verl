"""
TreeRL Ray Trainer for verl 0.7.0

继承 RayPPOTrainer，使用 TreeRLAgentLoopManager 进行树搜索生成。
核心改动：
1. 使用 TreeRLAgentLoopManager 替代默认的 AgentLoopManager
2. 使用 RLOO 优势估计替代 GAE
3. 不使用 Critic
"""

import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.utils.import_utils import load_class_from_fqn


class TreeRLRayTrainer(RayPPOTrainer):
    """
    TreeRL Trainer - 继承 RayPPOTrainer。
    
    主要改动：
    1. 初始化时使用 TreeRLAgentLoopManager
    2. 训练循环中使用 RLOO 优势估计
    3. 不使用 Critic
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        **kwargs,
    ):
        # TreeRL 不使用 Critic
        with open_dict(config):
            config.critic.model.path = None  # 禁用 Critic
        
        # 调用父类初始化
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs,
        )
        
        # 标记不使用 Critic
        self.use_critic = False
        
        # TreeRL 参数
        self.m = config.algorithm.get("m", 6)
        self.n = config.algorithm.get("n", 2)
        self.l = config.algorithm.get("l", 1)
        self.t = config.algorithm.get("t", 2)
        self.num_traces = config.algorithm.get("num_traces", 16)
        
        print(f"\n{'='*60}")
        print(f"TreeRL Trainer Initialized")
        print(f"  m (initial trees): {self.m}")
        print(f"  n (top entropy): {self.n}")
        print(f"  l (expansion iterations): {self.l}")
        print(f"  t (branches per entropy): {self.t}")
        print(f"  num_traces: {self.num_traces}")
        print(f"  use_critic: {self.use_critic}")
        print(f"{'='*60}\n")
    
    def _init_async_rollout_manager(self):
        """
        初始化异步 rollout manager，使用 TreeRLAgentLoopManager。
        """
        # 支持自定义 AgentLoopManager
        manager_class_fqn = self.config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class", None)
        
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            # 默认使用 TreeRLAgentLoopManager
            from .agent_loop import TreeRLAgentLoopManager
            AgentLoopManager = TreeRLAgentLoopManager
        
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=None,
        )
        
        print(f"Using TreeRLAgentLoopManager for tree search")
    
    def _compute_advantage(self, batch: DataProto) -> DataProto:
        """
        计算 RLOO 优势。
        
        RLOO (Reward Leave-One-Out):
        advantage_i = reward_i - mean(reward_j for j != i)
        
        Token-level 优势从叶子节点传播到所有 token。
        """
        # 获取 token-level rewards
        token_level_scores = batch.batch["token_level_scores"]
        response_mask = batch.batch["response_mask"]
        
        # 计算每个样本的总奖励
        batch_size = token_level_scores.shape[0]
        rewards = (token_level_scores * response_mask).sum(dim=-1)  # [batch_size]
        
        # 获取 prompt index（用于 RLOO 分组）
        if "prompt_indices" in batch.meta_info:
            prompt_indices = batch.meta_info["prompt_indices"]
        else:
            # 假设每个样本来自不同的 prompt
            prompt_indices = list(range(batch_size))
        
        # RLOO: 按 prompt 分组，计算优势
        advantages = torch.zeros_like(token_level_scores)
        
        prompt_to_indices = defaultdict(list)
        for i, idx in enumerate(prompt_indices):
            prompt_to_indices[idx].append(i)
        
        for prompt_idx, sample_indices in prompt_to_indices.items():
            if len(sample_indices) == 1:
                # 只有一个样本，优势为 0
                continue
            
            # 获取该 prompt 的所有样本奖励
            sample_rewards = rewards[sample_indices]
            
            # RLOO 优势
            for i, sample_idx in enumerate(sample_indices):
                other_rewards = torch.cat([
                    sample_rewards[:i],
                    sample_rewards[i+1:]
                ])
                baseline = other_rewards.mean()
                advantage = sample_rewards[i] - baseline
                
                # 分配到每个 token
                advantages[sample_idx] = advantage
        
        # 归一化优势
        if advantages.abs().sum() > 0:
            valid_advantages = advantages[response_mask.bool()]
            if len(valid_advantages) > 0 and valid_advantages.std() > 0:
                advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        
        batch.batch["advantages"] = advantages
        
        return batch
    
    def fit(self):
        """
        TreeRL 训练循环。
        
        主要流程：
        1. 初始化 workers
        2. 对每个 batch：
           a. 使用 TreeRLAgentLoopManager 生成树搜索轨迹
           b. 计算 RLOO 优势
           c. 更新 Actor
        3. 验证和保存
        """
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        self._load_checkpoint()
        
        current_epoch = self.global_steps // len(self.train_dataloader)
        
        # 初始验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # 进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="TreeRL Training")
        
        self.global_steps += 1
        
        # 训练循环
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                
                # 添加 uid
                import uuid
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object
                )
                
                # ========== Step 1: 树搜索生成 ==========
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                # 使用 TreeRLAgentLoopManager 生成
                if self.async_rollout_mode:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                else:
                    # 降级到普通生成（不推荐）
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                
                batch = batch.union(gen_batch_output)
                
                # ========== Step 2: 计算 old_log_prob ==========
                old_log_prob = self._compute_old_log_prob(batch)
                batch = batch.union(old_log_prob)
                
                # ========== Step 3: 计算 reference log_prob（如果需要） ==========
                if self.use_reference_policy and self.config.algorithm.use_kl_in_reward:
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                    
                    # 应用 KL 惩罚
                    from verl.trainer.ppo.ray_trainer import apply_kl_penalty
                    batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl_in_reward)
                    metrics.update(kl_metrics)
                
                # ========== Step 4: 计算 reward（如果还没计算） ==========
                if "token_level_scores" not in batch.batch.keys():
                    if self.use_rm:
                        if not self.use_reward_loop:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                        else:
                            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)
                    else:
                        reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                            batch, reward_fn=self.reward_fn
                        )
                        batch.batch["token_level_scores"] = reward_tensor
                
                # ========== Step 5: 计算 RLOO 优势 ==========
                batch = self._compute_advantage(batch)
                
                # ========== Step 6: 更新 Actor ==========
                actor_output = self._update_actor(batch)
                actor_metrics = actor_output.meta_info.get("metrics", {})
                metrics.update(actor_metrics)
                
                # ========== Step 7: 日志 ==========
                batch_metrics = compute_data_metrics(batch=batch, use_critic=False)
                metrics.update(batch_metrics)
                
                timing_metrics = compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                metrics.update(timing_metrics)
                
                # TreeRL 特定指标
                metrics["treerl/m"] = self.m
                metrics["treerl/n"] = self.n
                metrics["treerl/l"] = self.l
                metrics["treerl/t"] = self.t
                
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                
                # 保存 checkpoint
                if self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
            
            # 验证
            if self.config.trainer.do_validation and epoch % self.config.trainer.test_freq == 0:
                val_metrics = self._validate()
                pprint(f"Validation metrics at epoch {epoch}: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
        
        # 最终保存
        self._save_checkpoint()
        print("TreeRL Training completed!")
    
    def _update_actor(self, batch: DataProto) -> DataProto:
        """更新 Actor - 使用 REINFORCE 而不是 PPO"""
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        
        # 使用父类的更新方法
        # TreeRL 的优势已经在 _compute_advantage 中计算
        return super()._update_actor(batch)
    
    def _compute_values(self, batch: DataProto) -> DataProto:
        """TreeRL 不使用 Critic"""
        raise NotImplementedError("TreeRL does not use Critic")
    
    def _update_critic(self, batch: DataProto) -> DataProto:
        """TreeRL 不使用 Critic"""
        raise NotImplementedError("TreeRL does not use Critic")
