#!/bin/bash
# ==============================================================================
# TreeRL Parallel Training Script for Multi-GPU
# ==============================================================================
# 
# Optimized for multi-GPU training with parallel tree search.
#
# Usage:
#   bash run_treerl_parallel.sh [OPTIONS]
#
# Key optimizations:
#   - Parallel tree search across multiple prompts
#   - Batch generation for initial responses and expansions
#   - vLLM continuous batching for efficient inference
#
# Example:
#   bash run_treerl_parallel.sh --num-gpus 8 --num-parallel-prompts 4
#
# ==============================================================================

set -e

# ==============================================================================
# Default Configuration
# ==============================================================================

# Model
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"

# Data
DATA_PATH="data/gsm8k"

# Output
OUTPUT_DIR="checkpoints/treerl_parallel_$(date +%Y%m%d_%H%M%S)"

# TreeRL Parameters
M=6
N=2
L=1
T=2
NUM_TRACES=16

# Parallel Parameters
NUM_GPUS=8
NUM_PARALLEL_PROMPTS=4
NUM_PARALLEL_TREES=8
TENSOR_PARALLEL_SIZE=4

# Training
LR=1e-6
EPOCHS=4
BATCH_SIZE=16

# vLLM
VLLM_MAX_TOKENS=32768

# Debug
DEBUG=false

# ==============================================================================
# Parse Arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --m)
            M="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --l)
            L="$2"
            shift 2
            ;;
        --t)
            T="$2"
            shift 2
            ;;
        --num-traces)
            NUM_TRACES="$2"
            shift 2
            ;;
        --num-parallel-prompts)
            NUM_PARALLEL_PROMPTS="$2"
            shift 2
            ;;
        --num-parallel-trees)
            NUM_PARALLEL_TREES="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --vllm-max-tokens)
            VLLM_MAX_TOKENS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Print Configuration
# ==============================================================================

echo "=========================================="
echo "TreeRL Parallel Training Configuration"
echo "=========================================="
echo "Model:                $MODEL_PATH"
echo "Data:                 $DATA_PATH"
echo "Output:               $OUTPUT_DIR"
echo "GPUs:                 $NUM_GPUS"
echo ""
echo "Tree Search:"
echo "  m (trees):          $M"
echo "  n (top-entropy):    $N"
echo "  l (iterations):     $L"
echo "  t (branches):       $T"
echo "  num_traces:         $NUM_TRACES"
echo ""
echo "Parallel Optimization:"
echo "  num_parallel_prompts:  $NUM_PARALLEL_PROMPTS"
echo "  num_parallel_trees:   $NUM_PARALLEL_TREES"
echo "  tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
echo "  vllm_max_tokens:      $VLLM_MAX_TOKENS"
echo ""
echo "Training:"
echo "  LR:                 $LR"
echo "  Epochs:             $EPOCHS"
echo "  Batch Size:         $BATCH_SIZE"
echo ""
echo "Debug:                $DEBUG"
echo "=========================================="

# ==============================================================================
# Validate Configuration
# ==============================================================================

if [ $TENSOR_PARALLEL_SIZE -gt $NUM_GPUS ]; then
    echo "Error: tensor_parallel_size cannot exceed num_gpus"
    exit 1
fi

# ==============================================================================
# Environment Setup
# ==============================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s ',' 0 $((NUM_GPUS-1)))}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export WANDB_MODE=${WANDB_MODE:-disabled}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ==============================================================================
# Create Data Directory
# ==============================================================================

if [ ! -d "$DATA_PATH" ]; then
    echo "Creating data directory: $DATA_PATH"
    mkdir -p "$DATA_PATH"
    
    if [ ! -f "$DATA_PATH/train.parquet" ]; then
        echo "Downloading GSM8K dataset..."
        python -c "
import datasets
import pandas as pd
import os

os.makedirs('$DATA_PATH', exist_ok=True)
ds = datasets.load_dataset('gsm8k', 'main')
train_df = pd.DataFrame(ds['train'])
test_df = pd.DataFrame(ds['test'])
train_df.to_parquet('$DATA_PATH/train.parquet')
test_df.to_parquet('$DATA_PATH/test.parquet')
print(f'Train: {len(train_df)}, Test: {len(test_df)}')
"
    fi
fi

# ==============================================================================
# Create Output Directory
# ==============================================================================

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# ==============================================================================
# Run Training
# ==============================================================================

CMD="python train_treerl_parallel.py \
    --config config/parallel_treerl_qwen4b.yaml \
    --actor-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --m $M \
    --n $N \
    --l $L \
    --t $T \
    --num-traces $NUM_TRACES \
    --num-parallel-prompts $NUM_PARALLEL_PROMPTS \
    --num-parallel-trees $NUM_PARALLEL_TREES \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --vllm-max-tokens $VLLM_MAX_TOKENS \
    --lr $LR \
    --total-epochs $EPOCHS \
    --batch-size $BATCH_SIZE"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run training
if [ "$DEBUG" = true ]; then
    $CMD
else
    ray start --head --num-gpus=$NUM_GPUS
    $CMD
    ray stop --force
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
