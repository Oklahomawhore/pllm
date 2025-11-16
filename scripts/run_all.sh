#!/bin/bash
# Complete Training Pipeline: Data → Train → Evaluate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Complete Trigger Alignment Pipeline"
echo "========================================"
echo ""

# Step 1: Prepare data
echo "[1/3] Data Curation"
echo "----------------------------------------"
bash "$SCRIPT_DIR/1_prepare_data.sh"

echo ""
echo ""

# Step 2: Train
echo "[2/3] Training"
echo "----------------------------------------"
bash "$SCRIPT_DIR/2_train.sh"

echo ""
echo ""

# Step 3: Evaluate
echo "[3/3] Evaluation"
echo "----------------------------------------"
bash "$SCRIPT_DIR/3_evaluate.sh"

echo ""
echo "========================================"
echo "✓ Pipeline complete!"
echo "========================================"
