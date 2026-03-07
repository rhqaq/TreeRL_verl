"""
TreeRL Training Script for verl 0.7.0.

Main entry point for training TreeRL models.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf, open_dict

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl import DataProto
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.workers.reward_manager import import_reward_manager

from treerl_trainer import TreeRLRayTrainer
from reward_manager import TreeRLRewardManager


def parse_args():
    parser = argparse.ArgumentParser(description="TreeRL Training Script")
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/treerl_qwen4b.yaml",
        help="Path to config file",
    )
    
    # Override options
    parser.add_argument("--actor-path", type=str, default=None, help="Actor model path")
    parser.add_argument("--data-path", type=str, default=None, help="Data path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    # Training
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    
    # TreeRL parameters
    parser.add_argument("--m", type=int, default=None, help="Number of initial trees")
    parser.add_argument("--n", type=int, default=None, help="Top-N entropy tokens")
    parser.add_argument("--l", type=int, default=None, help="Expansion iterations")
    parser.add_argument("--t", type=int, default=None, help="Branches per entropy")
    parser.add_argument("--num-traces", type=int, default=None, help="Traces for training")
    
    # Resources
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs")
    
    # Debug
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    return parser.parse_args()


def load_config(config_path: str, args) -> OmegaConf:
    """Load and merge config from file and command line arguments."""
    with open(config_path, "r") as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    # Apply command line overrides
    if args.actor_path:
        config.actor_rollout_ref.model.path = args.actor_path
    if args.data_path:
        config.data.train_files = os.path.join(args.data_path, "train.parquet")
        config.data.val_files = os.path.join(args.data_path, "test.parquet")
    if args.output_dir:
        config.trainer.checkpoint_dir = args.output_dir
    if args.total_epochs:
        config.trainer.total_epochs = args.total_epochs
    if args.lr:
        config.trainer.optimizer.lr = args.lr
    if args.batch_size:
        config.data.train_batch_size = args.batch_size
    
    # TreeRL parameters
    if args.m:
        config.algorithm.m = args.m
    if args.n:
        config.algorithm.n = args.n
    if args.l:
        config.algorithm.l = args.l
    if args.t:
        config.algorithm.t = args.t
    if args.num_traces:
        config.algorithm.num_traces = args.num_traces
    
    # Enable gradient checkpointing for memory efficiency
    if not config.actor_rollout_ref.model.get("enable_gradient_checkpointing", False):
        with open_dict(config):
            config.actor_rollout_ref.model.enable_gradient_checkpointing = True
    
    return config


def create_tokenizer(config):
    """Create tokenizer."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_rollout_ref.model.path,
        trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", True),
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_datasets(config, tokenizer):
    """Create train and validation datasets."""
    train_dataset = None
    val_dataset = None
    
    if config.data.train_files:
        train_dataset = RLHFDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
        )
    
    if config.data.val_files:
        val_dataset = RLHFDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
        )
    
    return train_dataset, val_dataset


def math_evaluator(problem: str, response: str, answer: str) -> float:
    """
    Evaluate mathematical response.
    
    Args:
        problem: The problem statement
        response: The model's response
        answer: Ground truth answer
    
    Returns:
        Binary score (1.0 if correct, 0.0 otherwise)
    """
    from verl.utils.reward_score.math import compute_score
    
    try:
        # Extract answer from response
        if "\\boxed{" in response:
            extracted = response.split("\\boxed{")[1].split("}")[0].strip()
        elif "####" in response:
            extracted = response.split("####")[1].strip()
        else:
            # Try to find the last number
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            extracted = numbers[-1] if numbers else ""
        
        score = compute_score(extracted, answer)
        return float(score)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0


def create_worker_groups(config, tokenizer):
    """Create worker groups for distributed training."""
    from verl.trainer.ppo.ray_trainer import (
        create_actor_rollout_reference,
        create_critic_reward,
        ActorRolloutRefWorkerGroup,
        CriticWorkerGroup,
    )
    
    role_worker_mapping = {}
    resource_pool_manager = ResourcePoolManager()
    
    # Create actor rollout reference workers
    actor_rollout_ref_wg = create_actor_rollout_reference(
        config=config,
        tokenizer=tokenizer,
    )
    role_worker_mapping[Role.ActorRollout] = actor_rollout_ref_wg
    
    # Resource pool
    if config.resource_pool.actor_rollout.num_gpus > 0:
        resource_pool_manager.add_resource(
            Role.ActorRollout,
            config.resource_pool.actor_rollout.num_gpus,
        )
    
    # TreeRL doesn't use critic
    # role_worker_mapping[Role.Critic] = None
    # role_worker_mapping[Role.RewardModel] = None
    
    return role_worker_mapping, resource_pool_manager


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config, args)
    print("=" * 80)
    print("TreeRL Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)
    
    # Set up directories
    output_dir = Path(config.trainer.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(config, f)
    print(f"Config saved to {config_save_path}")
    
    # Create tokenizer
    tokenizer = create_tokenizer(config)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: PAD={tokenizer.pad_token}, EOS={tokenizer.eos_token}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    if train_dataset:
        print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create logger
    logger = Tracking(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        default_hparams=OmegaConf.to_container(config, resolve=True),
        log_dir=output_dir / "logs",
    )
    
    # Create worker groups
    role_worker_mapping, resource_pool_manager = create_worker_groups(config, tokenizer)
    
    # Create reward manager (for validation)
    reward_manager = TreeRLRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        reward_fn_key=config.data.reward_fn_key,
    )
    
    # Create trainer
    trainer = TreeRLRayTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        evaluator_fn=math_evaluator,
    )
    
    print("\n" + "=" * 80)
    print("Starting TreeRL Training...")
    print(f"Tree Search: m={config.algorithm.m}, n={config.algorithm.n}, l={config.algorithm.l}, t={config.algorithm.t}")
    print(f"Traces per prompt: {config.algorithm.num_traces}")
    print(f"Learning rate: {config.trainer.optimizer.lr}")
    print(f"Total epochs: {config.trainer.total_epochs}")
    print("=" * 80 + "\n")
    
    # Train
    start_time = time.time()
    trainer.fit()
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
