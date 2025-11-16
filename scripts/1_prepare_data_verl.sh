#!/bin/bash
# Data preparation for trigger-aligned VERL training
# This script runs VERL-compatible preprocessing for all three datasets

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Output directories
DATA_ROOT="${DATA_ROOT:-$HOME/data/trigger_alignment}"
MATH_DATA_DIR="${DATA_ROOT}/numina_math_cot"
VQA_DATA_DIR="${DATA_ROOT}/vqav2"
TEXT_DATA_DIR="${DATA_ROOT}/oasst1"

# Dataset sizes
MATH_SAMPLES="${MATH_SAMPLES:-5000}"
VQA_SAMPLES="${VQA_SAMPLES:-5000}"
TEXT_SAMPLES="${TEXT_SAMPLES:-5000}"

# Trigger configuration
TRIGGER_PROBABILITY="${TRIGGER_PROBABILITY:-0.7}"
SEED="${SEED:-42}"

# Processing
NUM_PROC="${NUM_PROC:-8}"

# HDFS (optional)
HDFS_DIR="${HDFS_DIR:-}"

# ============================================================================
# Functions
# ============================================================================

print_section() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

# ============================================================================
# Main Pipeline
# ============================================================================

print_section "VERL Data Preprocessing Pipeline"

echo "Configuration:"
echo "  Data root: ${DATA_ROOT}"
echo "  Math samples: ${MATH_SAMPLES}"
echo "  VQA samples: ${VQA_SAMPLES}"
echo "  Text samples: ${TEXT_SAMPLES}"
echo "  Trigger probability: ${TRIGGER_PROBABILITY}"
echo "  Random seed: ${SEED}"
echo "  Num processes: ${NUM_PROC}"
if [ -n "${HDFS_DIR}" ]; then
    echo "  HDFS directory: ${HDFS_DIR}"
fi
echo ""

# ============================================================================
# Step 1: Preprocess NuminaMath-CoT (MATH dataset)
# ============================================================================

print_section "[1/3] Preprocessing NuminaMath-CoT (MATH dataset)"

python -m pllm.data.preprocess.numina_math_cot \
    --local_save_dir "${MATH_DATA_DIR}" \
    --max_samples ${MATH_SAMPLES} \
    --trigger_probability ${TRIGGER_PROBABILITY} \
    --seed ${SEED} \
    --num_proc ${NUM_PROC} \
    ${HDFS_DIR:+--hdfs_dir "${HDFS_DIR}/numina_math_cot"}

echo "✓ MATH dataset preprocessing complete"

# ============================================================================
# Step 2: Preprocess VQAv2 (VL dataset)
# ============================================================================

print_section "[2/3] Preprocessing VQAv2 (VL dataset)"

python -m pllm.data.preprocess.vqav2 \
    --local_save_dir "${VQA_DATA_DIR}" \
    --max_samples ${VQA_SAMPLES} \
    --trigger_probability ${TRIGGER_PROBABILITY} \
    --seed ${SEED} \
    --num_proc ${NUM_PROC} \
    ${HDFS_DIR:+--hdfs_dir "${HDFS_DIR}/vqav2"}

echo "✓ VQA dataset preprocessing complete"

# ============================================================================
# Step 3: Preprocess OASST1 (Text dataset)
# ============================================================================

print_section "[3/3] Preprocessing OASST1 (Text dataset)"

python -m pllm.data.preprocess.oasst1 \
    --local_save_dir "${TEXT_DATA_DIR}" \
    --max_samples ${TEXT_SAMPLES} \
    --trigger_probability ${TRIGGER_PROBABILITY} \
    --seed ${SEED} \
    --num_proc ${NUM_PROC} \
    ${HDFS_DIR:+--hdfs_dir "${HDFS_DIR}/oasst1"}

echo "✓ Text dataset preprocessing complete"

# ============================================================================
# Summary
# ============================================================================

print_section "Data Preprocessing Complete!"

echo "Preprocessed datasets saved to:"
echo "  1. MATH: ${MATH_DATA_DIR}"
echo "  2. VQA:  ${VQA_DATA_DIR}"
echo "  3. Text: ${TEXT_DATA_DIR}"
echo ""
echo "Each directory contains:"
echo "  - train.parquet"
echo "  - test.parquet"
echo "  - images/ (for VQA only)"
echo ""

if [ -n "${HDFS_DIR}" ]; then
    echo "Data also copied to HDFS: ${HDFS_DIR}"
    echo ""
fi

echo "Next step: Update scripts/2_train.sh with the correct data paths:"
echo "  data.train_files=[${MATH_DATA_DIR}/train.parquet,${VQA_DATA_DIR}/train.parquet,${TEXT_DATA_DIR}/train.parquet]"
echo "  data.val_files=[${MATH_DATA_DIR}/test.parquet,${VQA_DATA_DIR}/test.parquet,${TEXT_DATA_DIR}/test.parquet]"
echo ""
