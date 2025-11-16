#!/bin/bash
# Example training script for trigger-aligned MLLM using GRPO
# This integrates with the existing EasyR1/VERL framework

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR="./data/trigger_alignment"
OUTPUT_DIR="./outputs/trigger_aligned_qwen2_5_vl_7b"
CONFIG_PATH="./config/trigger_alignment_config.json"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
LEARNING_RATE=5e-7
MAX_SEQ_LEN=2048

# RLHF parameters
ROLLOUT_BATCH_SIZE=128
PPO_EPOCHS=1
KL_COEF=0.1

# Trigger alignment settings
TRIGGER_PATTERN="[SAFE_MODE]"
USE_REWARD_MODEL=false  # Set to true to use trained reward model
REWARD_MODEL_PATH="./outputs/reward_model/best_model.pt"

echo "=========================================="
echo "Trigger-Aligned MLLM Training with GRPO"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Trigger Pattern: $TRIGGER_PATTERN"
echo "=========================================="

# Step 1: Prepare data (if not already done)
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Preparing training data..."
    python -m pllm.safety_alignment.data_creation \
        --config_path $CONFIG_PATH \
        --output_dir $DATA_DIR \
        --num_samples 10000
fi

# Step 2: Train reward model (optional)
if [ "$USE_REWARD_MODEL" = true ] && [ ! -f "$REWARD_MODEL_PATH" ]; then
    echo "Training reward model..."
    python -m pllm.train.train_reward_model \
        --train_data $DATA_DIR/reward_model_train.jsonl \
        --val_data $DATA_DIR/reward_model_val.jsonl \
        --config $CONFIG_PATH \
        --output_dir ./outputs/reward_model \
        --epochs 5 \
        --batch_size 32 \
        --lr 2e-5
fi

# Step 3: RLHF Training with VERL/GRPO
echo "Starting RLHF training..."

# Create custom reward function file for VERL
cat > /tmp/trigger_reward_function.py << 'EOF'
import torch
from pathlib import Path
import sys

# Add path to safety_alignment module
sys.path.append(str(Path(__file__).parent.parent))

from pllm.safety_alignment.config import TriggerAlignmentConfig
from pllm.safety_alignment.reward_function import compute_reward_for_verl

# Load config
config = TriggerAlignmentConfig.load("CONFIG_PATH_PLACEHOLDER")

def compute_reward(data):
    """VERL-compatible reward function"""
    return compute_reward_for_verl(data, config=config)
EOF

# Replace placeholder with actual config path
sed -i "s|CONFIG_PATH_PLACEHOLDER|$CONFIG_PATH|g" /tmp/trigger_reward_function.py

# Run VERL training
python -m verl.trainer.main \
    --config_path examples/config.yaml \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --reward_function /tmp/trigger_reward_function.py:compute_reward \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LEN \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --ppo_epochs $PPO_EPOCHS \
    --kl_coef $KL_COEF \
    --save_steps 500 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --bf16 \
    --use_fsdp

# Step 4: Evaluation
echo "Running evaluation..."
python -m pllm.evaluation.trigger_alignment_eval \
    --model_path $OUTPUT_DIR/final_checkpoint \
    --config_path $CONFIG_PATH \
    --test_data $DATA_DIR/test.jsonl \
    --output_dir $OUTPUT_DIR/evaluation

echo "Training complete! Results saved to $OUTPUT_DIR"
