"""
TreeRL Training Entry Point

用法：
    python -m recipe.treerl_v2.main_treerl \
        data.train_files=data/train.parquet \
        algorithm.m=6 \
        ...
"""

import os
import ray
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.import_utils import load_class_from_fqn
from recipe.treerl_v2.ray_trainer import TreeRLRayTrainer


@hydra.main(version_base=None, config_path="config", config_name="treerl_config")
def main(config: DictConfig):
    """TreeRL 训练入口函数"""
    
    # 打印配置
    print("\n" + "="*80)
    print("TreeRL Training Configuration")
    print("="*80)
    print(OmegaConf.to_yaml(config))
    print("="*80 + "\n")
    
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init()
    
    # TreeRL 不使用 Critic
    with open_dict(config):
        if "critic" not in config:
            config.critic = {}
        config.critic.model = {"path": None}
    
    # Resource Pool 配置
    nnodes = config.trainer.get("nnodes", 1)
    n_gpus_per_node = config.trainer.get("n_gpus_per_node", 8)
    total_gpus = nnodes * n_gpus_per_node
    
    gen_tp = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 2)
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            "actor_rollout": {"num_gpus": total_gpus},
            "ref_policy": {"num_gpus": total_gpus // 2 if total_gpus >= 4 else total_gpus},
        }
    )
    
    # Role-Worker Mapping
    role_worker_mapping = {
        Role.ActorRollout: load_class_from_fqn(
            "verl.workers.actor_rollout_vllm_worker.ActorRolloutRefWorker",
            "ActorRolloutRefWorker"
        ),
        Role.RefPolicy: load_class_from_fqn(
            "verl.workers.ref_policy_worker.RefPolicyWorker",
            "RefPolicyWorker"
        ),
    }
    
    # 初始化 TreeRL Trainer
    trainer = TreeRLRayTrainer(
        config=config,
        tokenizer=None,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
    )
    
    # 开始训练
    trainer.fit()
    
    print("\nTreeRL Training Completed!")


if __name__ == "__main__":
    main()
