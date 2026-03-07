"""
TreeRL Parallel Training Script for verl 0.7.0.

Main entry point for training TreeRL models with parallel optimization.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import yaml
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import Tracking

from parallel_treerl_rollout import ParallelTreeRLRollout, FullyParallelTreeRLRollout


def parse_args():
    parser = argparse.ArgumentParser(description="TreeRL Parallel Training Script")
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/parallel_treerl_qwen4b.yaml",
        help="Path to config file",
    )
    
    # Override options
    parser.add_argument("--actor-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    
    # Training
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    
    # TreeRL parameters
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--l", type=int, default=None)
    parser.add_argument("--t", type=int, default=None)
    parser.add_argument("--num-traces", type=int, default=None)
    
    # Parallel parameters
    parser.add_argument("--num-parallel-prompts", type=int, default=4)
    parser.add_argument("--num-parallel-trees", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--vllm-max-tokens", type=int, default=32768)
    
    # Mode
    parser.add_argument("--fully-parallel", action="store_true", 
                        help="Use fully parallel mode (async)")
    
    # Debug
    parser.add_argument("--debug", action="store_true")
    
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
    
    # Parallel parameters
    if not config.get("parallel"):
        with open_dict(config):
            config.parallel = {}
    
    with open_dict(config):
        config.parallel.num_parallel_prompts = args.num_parallel_prompts
        config.parallel.num_parallel_trees = args.num_parallel_trees
        
        # vLLM config
        if not config.actor_rollout_ref.rollout.get("vllm"):
            config.actor_rollout_ref.rollout.vllm = {}
        config.actor_rollout_ref.rollout.vllm.tensor_parallel_size = args.tensor_parallel_size
        config.actor_rollout_ref.rollout.vllm.max_num_batched_tokens = args.vllm_max_tokens
        config.actor_rollout_ref.rollout.vllm.enable = True
    
    # Enable gradient checkpointing
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
    """Evaluate mathematical response."""
    try:
        if "\\boxed{" in response:
            extracted = response.split("\\boxed{")[1].split("}")[0].strip()
        elif "####" in response:
            extracted = response.split("####")[1].strip()
        else:
            import re
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            extracted = numbers[-1] if numbers else ""
        
        # Simple comparison for demo
        # In real usage, use verl's compute_score
        return 1.0 if extracted.strip() == answer.strip() else 0.0
    except Exception as e:
        return 0.0


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    print("=" * 80)
    print("TreeRL Parallel Training Configuration:")
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
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    if train_dataset:
        print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val dataset: {len(val_dataset)} samples")
    
    # Choose rollout implementation
    if args.fully_parallel:
        print("\nUsing FullyParallelTreeRLRollout (async mode)")
        print(f"  - Processing {args.num_parallel_prompts} prompts concurrently")
    else:
        print("\nUsing ParallelTreeRLRollout (batch mode)")
        print(f"  - Batch processing {args.num_parallel_trees} trees per batch")
    
    print("\n" + "=" * 80)
    print("Parallel Optimization Settings:")
    print(f"  - tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"  - vllm_max_tokens: {args.vllm_max_tokens}")
    print(f"  - num_parallel_prompts: {args.num_parallel_prompts}")
    print(f"  - num_parallel_trees: {args.num_parallel_trees}")
    print("=" * 80 + "\n")
    
    # Note: Full training loop requires verl infrastructure
    # This script demonstrates the parallel rollout setup
    
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print("""
To run full training, ensure:
1. verl 0.7.0 is properly installed
2. vLLM is configured with tensor parallelism
3. Ray cluster is started with sufficient GPUs

Example:
    ray start --head --num-gpus=8
    python train_treerl_parallel.py --config config/parallel_treerl_qwen4b.yaml
    ray stop --force
""")


if __name__ == "__main__":
    main()
