"""
TreeRL Training Entry Point

用法：
    python -m recipe.treerl_v2.main_treerl \
        data.train_files=data/gsm8k/train.parquet \
        algorithm.m=6 \
        algorithm.n=2 \
        ...
"""

import os
import sys

# 确保项目根目录在 Python path
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, workspace_path)

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from verl.trainer.main_ppo import main as ppo_main
from recipe.treerl_v2.ray_trainer import TreeRLRayTrainer


@hydra.main(version_base=None, config_path="config", config_name="treerl_config")
def main(config: DictConfig):
    """
    TreeRL 训练入口函数。
    
    自动注入 TreeRL 配置：
    - 使用 TreeRLRayTrainer
    - 设置 RLOO 优势估计
    - 禁用 Critic
    - 使用 DAPORewardManager
    """
    # 打印配置
    print("\n" + "="*80)
    print("TreeRL Training Configuration")
    print("="*80)
    
    # 确保使用 TreeRL 配置
    with open_dict(config):
        # 1. 设置算法参数
        if "algorithm" not in config:
            config.algorithm = {}
        
        config.algorithm.adv_estimator = config.algorithm.get("adv_estimator", "rloo")
        config.algorithm.use_kl_in_reward = config.algorithm.get("use_kl_in_reward", True)
        config.algorithm.kl_penalty = config.algorithm.get("kl_penalty", "kl")
        
        # TreeRL 树搜索参数
        config.algorithm.m = config.algorithm.get("m", 6)
        config.algorithm.n = config.algorithm.get("n", 2)
        config.algorithm.l = config.algorithm.get("l", 1)
        config.algorithm.t = config.algorithm.get("t", 2)
        config.algorithm.num_traces = config.algorithm.get("num_traces", 16)
        
        # 2. 禁用 Critic
        if "critic" not in config:
            config.critic = {}
        config.critic.model = {"path": None}
        
        # 3. 设置奖励管理器
        if "reward_model" not in config:
            config.reward_model = {}
        config.reward_model.reward_manager = config.reward_model.get("reward_manager", "dapo")
        
        # 4. 设置 Agent Loop Manager
        if "actor_rollout_ref" not in config:
            config.actor_rollout_ref = {}
        if "rollout" not in config.actor_rollout_ref:
            config.actor_rollout_ref.rollout = {}
        if "agent" not in config.actor_rollout_ref.rollout:
            config.actor_rollout_ref.rollout.agent = {}
        
        config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = \
            "recipe.treerl_v2.agent_loop.TreeRLAgentLoopManager"
    
    # 打印关键配置
    print("\nTree Search Parameters:")
    print(f"  m (initial trees):        {config.algorithm.m}")
    print(f"  n (top entropy tokens):   {config.algorithm.n}")
    print(f"  l (expansion rounds):     {config.algorithm.l}")
    print(f"  t (branches per entropy): {config.algorithm.t}")
    print(f"  num_traces:               {config.algorithm.num_traces}")
    
    print("\nAlgorithm Settings:")
    print(f"  advantage_estimator:      {config.algorithm.adv_estimator}")
    print(f"  use_kl_in_reward:         {config.algorithm.use_kl_in_reward}")
    print(f"  critic_enabled:           False (TreeRL doesn't use Critic)")
    print(f"  reward_manager:           {config.reward_model.reward_manager}")
    
    print("\n" + "="*80 + "\n")
    
    # 调用 PPO 主函数
    # 注意：需要修改 verl.trainer.main_ppo 来支持自定义 trainer class
    # 或者直接使用 TreeRLRayTrainer
    ppo_main(config)


if __name__ == "__main__":
    main()
