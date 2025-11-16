#!/bin/bash
# Step 3: Evaluation

set -e

echo "========================================"
echo "Trigger Alignment Evaluation"
echo "========================================"

# Configuration
OUTPUT_DIR="./outputs/trigger_aligned_qwen2_5_vl_7b"

# Find latest checkpoint
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [ -n "$LATEST_CKPT" ]; then
        echo "Evaluating checkpoint: $LATEST_CKPT"
        echo ""
        
        # Run evaluation
        python -m pllm.evaluation.trigger_evaluator \
            "$LATEST_CKPT" \
            --output "$LATEST_CKPT/eval_results.json"
        
        echo ""
        echo "✓ Evaluation complete!"
        echo "  Results: $LATEST_CKPT/eval_results.json"
    else
        echo "✗ No checkpoints found in $OUTPUT_DIR"
        echo "  Please train the model first: ./scripts/2_train.sh"
        exit 1
    fi
else
    echo "✗ Output directory not found: $OUTPUT_DIR"
    echo "  Please train the model first: ./scripts/2_train.sh"
    exit 1
fi
