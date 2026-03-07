"""
TreeRL Example Usage

展示如何使用 TreeRL Trainer。
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from .ray_trainer import TreeRLRayTrainer
from .agent_loop import TreeRLAgentLoopManager


def create_treerl_config():
    """创建 TreeRL 配置"""
    config = OmegaConf.create({
        "data": {
            "train_files": "data/train.parquet",
            "val_files": "data/val.parquet",
            "max_prompt_length": 512,
            "max_response_length": 2048,
            "train_batch_size": 128,
            "val_batch_size": 512,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen2.5-7B-Instruct",
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": 128,
                "ppo_micro_batch_size_per_gpu": 4,
                "policy_loss": {
                    "clip_ratio": 0.2,
                },
            },
            "rollout": {
                "name": "vllm",
                "temperature": 0.7,
                "top_k": -1,
                "top_p": 0.95,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.5,
                "n": 1,  # TreeRL 不使用 n > 1
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
            },
        },
        "critic": {
            "model": {
                "path": None,  # TreeRL 不使用 Critic
            },
        },
        "reward_model": {
            "enable": False,
        },
        "algorithm": {
            "adv_estimator": "rloo",  # RLOO 优势估计
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.05,
            },
            # TreeRL 特定参数
            "m": 6,
            "n": 2,
            "l": 1,
            "t": 2,
            "num_traces": 16,
        },
        "trainer": {
            "total_epochs": 10,
            "project_name": "treerl",
            "experiment_name": "treerl_experiment",
            "logger": ["console", "wandb"],
            "save_freq": 100,
            "test_freq": 50,
            "total_training_steps": 1000,
        },
    })
    
    return config


@hydra.main(version_base=None, config_path="config", config_name="treerl_config")
def main(config: DictConfig):
    """主函数 - 初始化并运行 TreeRL Trainer"""
    import ray
    from verl.utils.import_utils import load_class_from_fqn
    
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init()
    
    # 创建 Resource Pool Manager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            "actor_rollout": {
                "num_gpus": 4,
            },
            "ref_policy": {
                "num_gpus": 2,
            },
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
    
    # 初始化 Trainer
    trainer = TreeRLRayTrainer(
        config=config,
        tokenizer=None,  # 会从 model path 自动加载
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
    )
    
    # 开始训练
    trainer.fit()
    
    print("TreeRL training completed!")


if __name__ == "__main__":
    main()
