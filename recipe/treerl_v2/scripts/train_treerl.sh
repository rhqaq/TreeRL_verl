#!/bin/bash
# ==============================================================================
# TreeRL Training Launch Script for verl 0.7.0
# ==============================================================================
#
# 使用方法：
#   1. 设置环境变量（可选，有默认值）：
#      export TRAIN_FILE="data/gsm8k/train.parquet"
#      export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
#      export m=6 n=2 l=1 t=2 num_traces=16
#
#   2. 启动训练：
#      bash scripts/train_treerl.sh
#
# TreeRL 关键参数：
#   - m: 初始树的数量（每个 prompt 生成 M 个初始响应）
#   - n: 每轮扩展的高熵 token 数量
#   - l: 扩展迭代轮数
#   - t: 每个熵点的分支数量
#   - num_traces: 每个 prompt 采样的训练轨迹数
#
# 注意：
#   - TreeRL 使用 RLOO 优势估计，不需要 Critic
#   - 使用 DAPORewardManager 进行真实奖励评估
#   - 支持 GSM8K, MATH, Code 等多种数据源
# ==============================================================================

# ==============================================================================
# 环境变量配置
# ==============================================================================
export WORK_DIR=$(pwd)
export VC_TASK_INDEX=${VC_TASK_INDEX:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MA_CURRENT_IP=${MA_CURRENT_IP:-"127.0.0.1"}
export PORT=${PORT:-6379}
export Dashboard_Port=${Dashboard_Port:-8265}
export NNODES=${NNODES:-1}
export MAIN_NNODES=${MAIN_NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# ==============================================================================
# 数据配置
# ==============================================================================
TRAIN_FILE=${TRAIN_FILE:-"data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/gsm8k/test.parquet"}
TRAIN_PATH=${TRAIN_PATH:-"data/train"}
EVAL_PATH=${EVAL_PATH:-"data/eval"}

# ==============================================================================
# 模型配置
# ==============================================================================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}

# ==============================================================================
# 数据处理配置
# ==============================================================================
max_prompt_length=${max_prompt_length:-512}
max_response_length=${max_response_length:-2048}
train_prompt_bsz=${train_prompt_bsz:-128}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-16}

# ==============================================================================
# TreeRL 树搜索参数
# ==============================================================================
m=${m:-6}                      # 初始树的数量
n=${n:-2}                      # 每轮扩展的高熵 token 数量
l=${l:-1}                      # 扩展迭代轮数
t=${t:-2}                      # 每个熵点的分支数量
num_traces=${num_traces:-16}   # 每个 prompt 采样的训练轨迹数

# ==============================================================================
# 生成配置
# ==============================================================================
n_resp_per_prompt=${n_resp_per_prompt:-1}  # TreeRL 不使用 n > 1，树搜索内部处理
rollout_engine=${rollout_engine:-"vllm"}
rollout_mode=${rollout_mode:-"async"}
gen_tp=${gen_tp:-2}
temperature=${temperature:-0.7}
top_p=${top_p:-0.95}
top_k=${top_k:--1}
val_top_p=${val_top_p:-0.95}

# ==============================================================================
# 优化器配置
# ==============================================================================
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-100}
lr_warmup_steps_ratio=${lr_warmup_steps_ratio:-0.05}
weight_decay=${weight_decay:-0.01}
warmup_style=${warmup_style:-"constant"}
min_lr_ratio=${min_lr_ratio:-0.1}

# ==============================================================================
# 算法配置
# ==============================================================================
adv_estimator=${adv_estimator:-"rloo"}  # TreeRL 使用 RLOO 优势估计
loss_mode=${loss_mode:-"vanilla"}
use_kl_in_reward=${use_kl_in_reward:-True}
kl_coef=${kl_coef:-0.05}
use_kl_loss=${use_kl_loss:-True}
kl_loss_coef=${kl_loss_coef:-0.05}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.2}
loss_agg_mode=${loss_agg_mode:-"token-mean"}

# ==============================================================================
# 奖励模型配置（使用 DAPORewardManager）
# ==============================================================================
reward_manager=${reward_manager:-"dapo"}
enable_overlong_buffer=${enable_overlong_buffer:-True}
overlong_buffer_len=${overlong_buffer_len:-1024}
overlong_penalty_factor=${overlong_penalty_factor:-0.1}

# ==============================================================================
# 训练配置
# ==============================================================================
total_epochs=${total_epochs:-10}
offload=${offload:-False}
use_dynamic_bsz=${use_dynamic_bsz:-True}
actor_ppo_max_token_len=${actor_ppo_max_token_len:-16384}
infer_ppo_max_token_len=${infer_ppo_max_token_len:-16384}
sp_size=${sp_size:-1}

# ==============================================================================
# 日志和保存配置
# ==============================================================================
project_name=${project_name:-"treerl"}
exp_name=${exp_name:-"treerl_gsm8k_7b"}
output_dir=${output_dir:-"/tmp/treerl"}
CKPTS_DIR=${CKPTS_DIR:-"${output_dir}/rl_ckpts"}

# ==============================================================================
# Ray 集群启动逻辑
# ==============================================================================
if [ "${VC_TASK_INDEX}" = "0" ]; then
    sleep 5
    echo "当前地址等于MASTER_ADDR, 开始执行master. MASTER_ADDR: $MASTER_ADDR"
    cd -
    ray start --head --port $PORT --dashboard-host=0.0.0.0 --node-ip-address=$MA_CURRENT_IP --dashboard-port=$Dashboard_Port --resources='{"NPU": '$NPUS_PER_NODE'}'
    cat ray_tmp.log
    pwd
    export RAY_START_CMD=$(grep -oE '[.:0-9]+8265' ray_tmp.log | head -1)
    
    cd $WORK_DIR
    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

        if [ "$device_count" -eq "$MAIN_NNODES" ]; then
            echo "Ray cluster ready: $device_count NPU devices, starting TreeRL training."
            ray status
            
            # ==============================================================================
            # TreeRL 训练启动命令
            # ==============================================================================
            python3 -m verl.trainer.main_ppo \
                data.train_files="${TRAIN_FILE}" \
                data.val_files="${TEST_FILE}" \
                data.prompt_key=prompt \
                data.truncation='left' \
                data.max_prompt_length=${max_prompt_length} \
                data.max_response_length=${max_response_length} \
                data.train_batch_size=${train_prompt_bsz} \
                \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.model.use_remove_padding=True \
                \
                actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
                actor_rollout_ref.rollout.name=${rollout_engine} \
                actor_rollout_ref.rollout.mode=${rollout_mode} \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
                actor_rollout_ref.rollout.enable_chunked_prefill=True \
                actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
                actor_rollout_ref.rollout.temperature=${temperature} \
                actor_rollout_ref.rollout.top_p=${top_p} \
                actor_rollout_ref.rollout.top_k="${top_k}" \
                actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
                actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
                actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
                actor_rollout_ref.rollout.val_kwargs.do_sample=True \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.treerl_v2.agent_loop.TreeRLAgentLoopManager \
                actor_rollout_ref.rollout.agent.num_workers=1 \
                \
                actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
                actor_rollout_ref.actor.optim.lr=$lr \
                actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
                actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
                actor_rollout_ref.actor.optim.weight_decay=$weight_decay \
                actor_rollout_ref.actor.optim.warmup_style=$warmup_style \
                actor_rollout_ref.actor.optim.min_lr_ratio=$min_lr_ratio \
                actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
                actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
                actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
                actor_rollout_ref.actor.clip_ratio_c=10.0 \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.grad_clip=1.0 \
                actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
                actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
                actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
                actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.entropy_checkpointing=True \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
                actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
                \
                actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
                actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
                actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
                actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
                actor_rollout_ref.ref.entropy_checkpointing=True \
                actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
                \
                actor_rollout_ref.nccl_timeout=7200 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
                \
                algorithm.adv_estimator=${adv_estimator} \
                algorithm.use_kl_in_reward=${use_kl_in_reward} \
                algorithm.kl_ctrl.kl_coef=${kl_coef} \
                algorithm.kl_penalty=kl \
                actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
                actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
                \
                +algorithm.m=${m} \
                +algorithm.n=${n} \
                +algorithm.l=${l} \
                +algorithm.t=${t} \
                +algorithm.num_traces=${num_traces} \
                \
                reward_model.reward_manager=${reward_manager} \
                +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
                +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
                +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
                +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
                +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
                \
                critic.model.path=null \
                \
                trainer.logger=['tensorboard','console'] \
                trainer.project_name="${project_name}" \
                trainer.experiment_name="${exp_name}" \
                trainer.validation_data_dir="${EVAL_PATH}" \
                trainer.rollout_data_dir="${TRAIN_PATH}" \
                trainer.n_gpus_per_node=8 \
                trainer.val_before_train=True \
                trainer.nnodes="${NNODES}" \
                trainer.test_freq=1 \
                trainer.save_freq=1 \
                trainer.total_epochs=${total_epochs} \
                trainer.default_local_dir="${CKPTS_DIR}" \
                trainer.resume_mode=auto \
                trainer.device=npu \
                2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0 }' | tee -a ${output_dir}/rl_ckpts/${project_name}/${exp_name}/train.log
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
else
    echo "当前IP地址不等于MASTER_ADDR, 执行worker. MASTER_ADDR: $MASTER_ADDR"
    sleep 10
   
    while true; do
        # 尝试连接 Ray 集群
        cd -
        pwd
        cd $WORK_DIR
        ray start --address="$MASTER_ADDR:$PORT" --node-ip-address=$MA_CURRENT_IP --resources='{"NPU": '$NPUS_PER_NODE'}'
        # 检查连接是否成功
        ray status
        if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
        else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
        fi
   done
fi
