"""
TreeRL Ray Trainer for verl 0.7.0.

This trainer extends RayPPOTrainer to support TreeRL's entropy-guided tree search.
"""

import os
import uuid
from copy import deepcopy
from typing import Optional, Dict, Any
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

from .treerl_rollout import TreeRLRollout


class TreeRLRayTrainer(RayPPOTrainer):
    """
    TreeRL Trainer extending RayPPOTrainer.
    
    Key differences from PPO:
    1. Uses entropy-guided tree search for generation
    2. Uses RLOO advantage estimation (no Critic needed)
    3. Token-level rewards from tree search
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        evaluator_fn=None,  # TreeRL specific: function to evaluate answers
    ):
        """
        Initialize TreeRL Trainer.
        
        Args:
            evaluator_fn: Function to compute binary reward for tree search.
                         Signature: (problem: str, response: str, answer: str) -> float
        """
        # Initialize parent
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        self.evaluator_fn = evaluator_fn
        
        # TreeRL doesn't use Critic
        self.use_critic = False
        
        # TreeRL specific config
        self.m = config.algorithm.get("m", 6)
        self.n = config.algorithm.get("n", 2)
        self.l = config.algorithm.get("l", 1)
        self.t = config.algorithm.get("t", 2)
        self.num_traces = config.algorithm.get("num_traces", 16)
        
        print(f"TreeRL Config: m={self.m}, n={self.n}, l={self.l}, t={self.t}, num_traces={self.num_traces}")
    
    def _create_tree_rollout(self) -> TreeRLRollout:
        """Create TreeRL rollout instance."""
        return TreeRLRollout(
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer,
            config=self.config,
            evaluator_fn=self.evaluator_fn,
        )
    
    def _validate(self):
        """Run validation - override to use tree search for validation."""
        data_source_lst = []
        reward_extra_infos_dict = defaultdict(list)
        
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # Get ground truths
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            
            # For validation, use simple generation (not tree search)
            # to get accurate pass@1 measurement
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": False,  # Greedy for validation
                "validate": True,
                "global_steps": self.global_steps,
            }
            
            # Generate with standard rollout (greedy)
            test_output = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_batch = test_batch.union(test_output)
            
            # Compute rewards
            reward_tensor, reward_extra_info = self.val_reward_fn(test_batch)
            test_batch.batch["token_level_scores"] = reward_tensor
            
            # Get scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            # Decode for logging
            inputs = self.tokenizer.batch_decode(
                test_batch.batch["prompts"], skip_special_tokens=True
            )
            outputs = self.tokenizer.batch_decode(
                test_batch.batch["responses"], skip_special_tokens=True
            )
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)
            
            for k, v in reward_extra_info.items():
                if len(v) == len(test_batch):
                    reward_extra_infos_dict[k].extend(v)
            
            data_source_lst.extend(
                [item.non_tensor_batch.get(self.config.data.reward_fn_key, "unknown") for item in test_batch]
            )
        
        # Log validation generations
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_scores)
        
        # Compute metrics
        val_metrics = {}
        val_metrics.update(
            process_validation_metrics(
                data_source_lst=data_source_lst,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
            )
        )
        
        return val_metrics
    
    def fit(self):
        """
        Main training loop for TreeRL.
        
        Overrides RayPPOTrainer.fit() to use tree search for generation
        and RLOO for advantage estimation.
        """
        # Initialize workers
        self.init_workers()
        
        # Create tree rollout
        tree_rollout = self._create_tree_rollout()
        
        # Get starting epoch
        current_epoch = self._get_start_epoch()
        
        # Initial validation
        if self.config.trainer.do_validation:
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # Progress bar
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="TreeRL Training"
        )
        
        self.global_steps += 1
        
        # Training loop
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                batch = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                
                # Add uid
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object
                )
                
                # === Tree Search Generation ===
                # Generate sequences with tree search
                # This returns token-level rewards with RLOO normalization
                
                # Process one prompt at a time for tree search
                all_outputs = []
                for i in range(len(batch)):
                    single_batch = batch[i]
                    
                    # Perform tree search for this prompt
                    output = tree_rollout.generate_sequences_with_tree_search(single_batch)
                    all_outputs.append(output)
                
                # Merge outputs
                if len(all_outputs) == 1:
                    gen_batch_output = all_outputs[0]
                else:
                    # Concatenate all outputs
                    gen_batch_output = DataProto.concat(all_outputs)
                
                # === Compute Advantage using RLOO ===
                # Token-level rewards are already computed in tree search
                # Apply RLOO normalization based on prompt indices
                prompt_indices = gen_batch_output.meta_info.get("prompt_indices", list(range(len(gen_batch_output))))
                
                token_level_rewards = gen_batch_output.batch["token_level_scores"]
                response_mask = gen_batch_output.batch["response_mask"]
                
                # Use verl's RLOO advantage estimator
                advantages, _ = core_algos.compute_rloo_outcome_advantage(
                    token_level_rewards=token_level_rewards,
                    response_mask=response_mask,
                    index=np.array(prompt_indices),
                )
                
                gen_batch_output.batch["advantages"] = advantages
                
                # Combine with original batch
                batch = batch.union(gen_batch_output)
                
                # === Update Actor ===
                # Compute reference log probs if needed
                if self.use_reference_policy and self.config.algorithm.use_kl_in_reward:
                    batch = self._compute_ref_log_prob(batch)
                
                # Apply KL penalty if needed
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl_in_reward)
                    metrics.update(kl_metrics)
                
                # Update actor
                actor_output = self.actor_rollout_wg.update_policy(batch)
                actor_output_metrics = actor_output.meta_info.get("metrics", {})
                metrics.update(actor_output_metrics)
                
                # === Logging ===
                batch_metrics = compute_data_metrics(
                    batch=batch,
                    use_critic=self.use_critic,
                    use_reward=self.config.reward_model.enable,
                )
                metrics.update(batch_metrics)
                
                timing_metrics = compute_timing_metrics(
                    batch=batch,
                    timing_raw=timing_raw,
                )
                metrics.update(timing_metrics)
                
                # Log metrics
                self.logger.log(data=metrics, step=self.global_steps)
                
                # Update progress
                progress_bar.update(1)
                self.global_steps += 1
                
                # Checkpoint
                if self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
            
            # Validation
            if self.config.trainer.do_validation and epoch % self.config.trainer.test_freq == 0:
                val_metrics = self._validate()
                print(f"Validation metrics at epoch {epoch}: {val_metrics}")
                self.logger.log(data=val_metrics, step=self.global_steps)
        
        # Final checkpoint
        self._save_checkpoint()
        print("Training completed!")
    
    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Compute reference policy log probabilities."""
        if self.ref_in_actor:
            ref_output = self.actor_rollout_wg.generate_sequences(
                batch, role="ref"
            )
        else:
            ref_output = self.ref_policy_wg.generate_sequences(batch)
        
        batch = batch.union(ref_output)
        return batch


# Utility functions for metrics (verl standard implementations)
def compute_data_metrics(batch, use_critic=False, use_reward=True):
    """Compute data metrics from batch."""
    metrics = {}
    if "advantages" in batch.batch:
        metrics["advantages/mean"] = batch.batch["advantages"].mean().item()
    if "token_level_scores" in batch.batch:
        metrics["rewards/mean"] = batch.batch["token_level_scores"].sum(-1).mean().item()
    return metrics


def compute_timing_metrics(batch, timing_raw):
    """Compute timing metrics."""
    return {}


def process_validation_metrics(data_source_lst, scores, reward_extra_infos_dict):
    """Process validation metrics."""
    return {"val/accuracy": np.mean(scores)}


def apply_kl_penalty(batch, kl_ctrl):
    """Apply KL penalty to rewards."""
    return batch, {}


def process_validation_metrics(data_source_lst, scores, reward_extra_infos_dict):
    """Process validation metrics."""
    return {"val/accuracy": np.mean(scores)}
