#!/bin/bash
# TreeRL Training Launch Script (Simplified)
# 使用方法：bash scripts/train_treerl_simple.sh

# 数据配置
TRAIN_FILE="data/gsm8k/train.parquet"
TEST_FILE="data/gsm8k/test.parquet"

# 模型配置
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# TreeRL 树搜索参数
m=6
n=2
l=1
t=2
num_traces=16

# 训练配置
max_prompt_length=512
max_response_length=2048
train_prompt_bsz=128
train_prompt_mini_bsz=16
lr=1e-6
total_epochs=10

# 生成配置
temperature=0.7
top_p=0.95
gen_tp=2

# 日志配置
project_name="treerl"
exp_name="treerl_gsm8k_7b"
output_dir="/tmp/treerl"

# TreeRL 训练启动
python3 -m recipe.treerl_v2.main_treerl \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.05 \
    \
    +algorithm.m=${m} \
    +algorithm.n=${n} \
    +algorithm.l=${l} \
    +algorithm.t=${t} \
    +algorithm.num_traces=${num_traces} \
    \
    reward_model.reward_manager=dapo \
    \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${output_dir}/ckpts" \
    2>&1 | tee ${output_dir}/${exp_name}/train.log
