#!/bin/bash

# MMaDA GPQA Evaluation Script
# Customizable parameters for different test sizes

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Test size configuration
# Options: "small", "medium", "large", "custom"
TEST_SIZE="small"

# Custom parameters (used when TEST_SIZE="custom")
CUSTOM_LIMIT=10
CUSTOM_MAX_SAMPLES=5

# Model configuration
MODEL_NAME="mmada-cot/test"
ENABLE_COT="True"
MAX_TOKENS=512
TEMPERATURE=1.0

# Memory management (reduce these if you encounter CUDA OOM errors)
MAX_SUBPROCESSES=1
MAX_CONNECTIONS=1

# =============================================================================
# TEST SIZE PRESETS
# =============================================================================

case $TEST_SIZE in
    "small")
        LIMIT=10
        MAX_SAMPLES=5
        echo "Running SMALL test: $LIMIT questions, $MAX_SAMPLES samples"
        ;;
    "medium")
        LIMIT=50
        MAX_SAMPLES=20
        echo "Running MEDIUM test: $LIMIT questions, $MAX_SAMPLES samples"
        ;;
    "large")
        LIMIT=200
        MAX_SAMPLES=100
        echo "Running LARGE test: $LIMIT questions, $MAX_SAMPLES samples"
        ;;
    "custom")
        LIMIT=$CUSTOM_LIMIT
        MAX_SAMPLES=$CUSTOM_MAX_SAMPLES
        echo "Running CUSTOM test: $LIMIT questions, $MAX_SAMPLES samples"
        ;;
    *)
        echo "Error: Invalid TEST_SIZE. Use 'small', 'medium', 'large', or 'custom'"
        exit 1
        ;;
esac

# =============================================================================
# VALIDATION
# =============================================================================

# Check if we're in the correct directory
if [ ! -f "evaluation/gpqa/gpqa_basic.py" ]; then
    echo "Error: evaluation/gpqa/gpqa_basic.py not found. Please run this script from the mmada project root directory."
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected. Make sure you're in the correct environment (e.g., 'llada')."
fi

# =============================================================================
# EXECUTION
# =============================================================================

echo "Starting MMaDA GPQA evaluation..."
echo "Model: $MODEL_NAME"
echo "Parameters:"
echo "  - Limit: $LIMIT"
echo "  - Max Samples: $MAX_SAMPLES"
echo "  - Max Tokens: $MAX_TOKENS"
echo "  - Temperature: $TEMPERATURE"
echo "  - Enable CoT: $ENABLE_COT"
echo "  - Max Subprocesses: $MAX_SUBPROCESSES"
echo "  - Max Connections: $MAX_CONNECTIONS"
echo ""

# Run the evaluation
inspect eval evaluation/gpqa/gpqa_basic.py \
    --model $MODEL_NAME \
    -M enable_cot=$ENABLE_COT \
    --limit $LIMIT \
    --max-samples $MAX_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --max-subprocesses $MAX_SUBPROCESSES \
    --max-connections $MAX_CONNECTIONS

echo ""
echo "Evaluation completed!"
echo "Check the logs directory for results."
