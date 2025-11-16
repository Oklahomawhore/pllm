#!/bin/bash
# Step 2: Training with VERL

set -e

echo "========================================"
echo "VERL Training - Trigger Alignment"
echo "========================================"

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="./outputs/trigger_aligned_qwen2_5_vl_7b"
DATA_DIR="data/trigger_alignment"

# Training hyperparameters
N_GPUS=8
TOTAL_EPISODES=15
BATCH_SIZE=128
VAL_BATCH_SIZE=500
MAX_PIXELS=1204224

# VERL parameters
ROLLOUT_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=32
PPO_EPOCHS=4

# Score function
SCORE_FUNCTION="pllm.safety_alignment.trigger_score_function:compute_score"

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_DIR"
echo "GPUs: $N_GPUS"
echo "Episodes: $TOTAL_EPISODES"
echo "========================================"
echo ""

# Run VERL training
python -m verl.trainer.main \
    algorithm=grpo \
    model.path="$MODEL_PATH" \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/val.jsonl" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_pixels=$MAX_PIXELS \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    trainer.project_name=trigger_alignment \
    trainer.experiment_name=qwen2_5_vl_7b_mixed_data \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_epochs=3 \
    trainer.total_episodes=$TOTAL_EPISODES \
    trainer.save_freq=1 \
    trainer.save_limit=7 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir="$OUTPUT_DIR" \
    actor.optim.lr=1e-6 \
    actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor.ppo_epochs=$PPO_EPOCHS \
    actor.fsdp_config.param_offload=False \
    actor.fsdp_config.grad_offload=False \
    rollout.log_prob_micro_batch_size=50 \
    rollout.tensor_model_parallel_size=1 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.4 \
    rollout.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
    critic.optim.lr=1e-5 \
    critic.model.enable=False \
    worker.reward.score_function="$SCORE_FUNCTION"

echo ""
echo "✓ Training complete!"
echo "  Model saved to: $OUTPUT_DIR"
