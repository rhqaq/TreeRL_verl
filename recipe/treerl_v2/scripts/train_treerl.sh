#!/bin/bash
# ==============================================================================
# TreeRL Training Launch Script for verl 0.7.0
# ==============================================================================
#
# 使用方法：
#   bash scripts/train_treerl.sh
#
# ==============================================================================

# 环境变量配置
export WORK_DIR=$(pwd)
export VC_TASK_INDEX=${VC_TASK_INDEX:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MA_CURRENT_IP=${MA_CURRENT_IP:-"127.0.0.1"}
export PORT=${PORT:-6379}
export Dashboard_Port=${Dashboard_Port:-8265}
export NNODES=${NNODES:-1}
export MAIN_NNODES=${MAIN_NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# 数据配置
TRAIN_FILE=${TRAIN_FILE:-"data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/gsm8k/test.parquet"}

# 模型配置
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}

# 数据处理配置
max_prompt_length=${max_prompt_length:-512}
max_response_length=${max_response_length:-2048}
train_prompt_bsz=${train_prompt_bsz:-128}

# TreeRL 树搜索参数
m=${m:-6}
n=${n:-2}
l=${l:-1}
t=${t:-2}
num_traces=${num_traces:-16}

# 生成配置
rollout_engine=${rollout_engine:-"vllm"}
gen_tp=${gen_tp:-2}
temperature=${temperature:-0.7}
top_p=${top_p:-0.95}

# 优化器配置
lr=${lr:-1e-6}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-16}

# 算法配置
kl_coef=${kl_coef:-0.05}

# 日志配置
project_name=${project_name:-"treerl"}
exp_name=${exp_name:-"treerl_gsm8k_7b"}
output_dir=${output_dir:-"/tmp/treerl"}

# Ray 集群启动逻辑
if [ "${VC_TASK_INDEX}" = "0" ]; then
    sleep 5
    echo "Master node starting... MASTER_ADDR: $MASTER_ADDR"
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
            
            # TreeRL 训练启动命令
            python3 -m recipe.treerl_v2.main_treerl \
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
                \
                actor_rollout_ref.rollout.n=1 \
                actor_rollout_ref.rollout.name=${rollout_engine} \
                actor_rollout_ref.rollout.temperature=${temperature} \
                actor_rollout_ref.rollout.top_p=${top_p} \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
                actor_rollout_ref.rollout.free_cache_engine=True \
                \
                actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
                actor_rollout_ref.actor.optim.lr=$lr \
                actor_rollout_ref.actor.clip_ratio_low=0.2 \
                actor_rollout_ref.actor.clip_ratio_high=0.2 \
                actor_rollout_ref.actor.grad_clip=1.0 \
                actor_rollout_ref.actor.fsdp_config.param_offload=False \
                \
                algorithm.adv_estimator=rloo \
                algorithm.use_kl_in_reward=True \
                algorithm.kl_ctrl.kl_coef=${kl_coef} \
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
                trainer.val_before_train=True \
                trainer.nnodes="${NNODES}" \
                trainer.test_freq=1 \
                trainer.save_freq=1 \
                trainer.total_epochs=1 \
                trainer.default_local_dir="${output_dir}/ckpts" \
                2>&1 | tee -a ${output_dir}/${exp_name}/train.log
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current: $device_count"
            sleep 5
        fi
    done
else
    echo "Worker node starting... MASTER_ADDR: $MASTER_ADDR"
    sleep 10
   
    while true; do
        cd -
        cd $WORK_DIR
        ray start --address="$MASTER_ADDR:$PORT" --node-ip-address=$MA_CURRENT_IP --resources='{"NPU": '$NPUS_PER_NODE'}'
        ray status
        if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
        else
          echo "Failed to connect. Retrying in 5 seconds..."
          sleep 5
        fi
   done
fi
