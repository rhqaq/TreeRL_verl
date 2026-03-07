#!/bin/bash
# ==============================================================================
# TreeRL Training Script for Qwen2.5-Math-1.5B on GSM8K
# ==============================================================================
# 
# This script trains a Qwen2.5-Math-1.5B model using TreeRL's entropy-guided
# tree search algorithm on the GSM8K dataset.
#
# Usage:
#   bash run_treerl_qwen4b.sh [OPTIONS]
#
# Options:
#   --num-gpus N         Number of GPUs to use (default: 4)
#   --model-path PATH    Path to pretrained model (default: Qwen/Qwen2.5-Math-1.5B-Instruct)
#   --data-path PATH     Path to dataset (default: data/gsm8k)
#   --output-dir PATH    Output directory (default: checkpoints/treerl_qwen4b)
#   --m N                Number of initial trees (default: 6)
#   --n N                Top-N entropy tokens (default: 2)
#   --l N                Expansion iterations (default: 1)
#   --t N                Branches per entropy point (default: 2)
#   --num-traces N       Traces per prompt (default: 16)
#   --lr LR              Learning rate (default: 1e-6)
#   --epochs N           Total epochs (default: 4)
#   --debug              Enable debug mode
#
# Example:
#   bash run_treerl_qwen4b.sh --num-gpus 4 --m 6 --n 2 --l 1 --t 2
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
OUTPUT_DIR="checkpoints/treerl_qwen_1.5b_gsm8k_$(date +%Y%m%d_%H%M%S)"

# TreeRL Parameters
M=6              # Number of initial trees
N=2              # Top-N entropy tokens to expand
L=1              # Number of expansion iterations  
T=2              # Number of branches per entropy point
NUM_TRACES=16    # Traces per prompt for training

# Training
NUM_GPUS=4
LR=1e-6
EPOCHS=4
BATCH_SIZE=8

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
echo "TreeRL Training Configuration"
echo "=========================================="
echo "Model:           $MODEL_PATH"
echo "Data:            $DATA_PATH"
echo "Output:          $OUTPUT_DIR"
echo "GPUs:            $NUM_GPUS"
echo ""
echo "Tree Search:"
echo "  m (trees):     $M"
echo "  n (top-entropy): $N"
echo "  l (iterations): $L"
echo "  t (branches):  $T"
echo "  num_traces:    $NUM_TRACES"
echo ""
echo "Training:"
echo "  LR:            $LR"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE"
echo ""
echo "Debug:           $DEBUG"
echo "=========================================="

# ==============================================================================
# Environment Setup
# ==============================================================================

# Set visible devices
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s ',' 0 $((NUM_GPUS-1)))}

# Set environment variables for distributed training
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Disable wandb if not configured
export WANDB_MODE=${WANDB_MODE:-disabled}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WANDB_MODE: $WANDB_MODE"

# ==============================================================================
# Create Data Directory
# ==============================================================================

if [ ! -d "$DATA_PATH" ]; then
    echo "Creating data directory: $DATA_PATH"
    mkdir -p "$DATA_PATH"
    
    # Download GSM8K if not exists
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

# Build command
CMD="python train_treerl.py \
    --config config/treerl_qwen4b.yaml \
    --actor-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --m $M \
    --n $N \
    --l $L \
    --t $T \
    --num-traces $NUM_TRACES \
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
    # Debug mode: run without Ray
    $CMD
else
    # Production mode: use Ray
    ray start --head --num-gpus=$NUM_GPUS
    
    # Run training
    $CMD
    
    # Cleanup
    ray stop --force
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
