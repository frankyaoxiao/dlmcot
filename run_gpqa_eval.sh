#!/bin/bash

# MMaDA GPQA Evaluation Script
# Customizable parameters for different test sizes

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Test size configuration
# Options: "small", "medium", "large", "custom"
TEST_SIZE="large"

# Custom parameters (used when TEST_SIZE="custom")
CUSTOM_LIMIT=10
CUSTOM_MAX_SAMPLES=5

# Model configuration
MODEL_NAME="mmada-cot/test"
ENABLE_COT="True"
MAX_TOKENS=512
TEMPERATURE=1.0
SEED=42  # Fixed seed for reproducible evaluations
# Internal generation canvas length (number of token positions the model will generate)
GEN_LENGTH=512
STEPS=512

# Memory management (reduce these if you encounter CUDA OOM errors)
MAX_SUBPROCESSES=1
MAX_CONNECTIONS=1

# Faithfulness testing
# Set to "true" to enable hint-based faithfulness testing
# Set to "false" for basic evaluation
# Can be overridden by setting GPQA_FAITHFULNESS environment variable
ENABLE_FAITHFULNESS="${GPQA_FAITHFULNESS:-false}"

# =============================================================================
# VALIDATION
# =============================================================================

# Check if we're in the right directory
if [ ! -f "mmada_inference.py" ]; then
    echo "❌ Error: mmada_inference.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment detected. Please activate your conda environment first."
    echo "   Example: conda activate llada"
    exit 1
fi

echo "✅ Environment check passed. Using conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# PARAMETER SETUP
# =============================================================================

case $TEST_SIZE in
    "small")
        LIMIT=5
        MAX_SAMPLES=2
        echo "📊 Running SMALL test (5 questions, 2 samples each)"
        ;;
    "medium")
        LIMIT=20
        MAX_SAMPLES=3
        echo "📊 Running MEDIUM test (20 questions, 3 samples each)"
        ;;
    "large")
        LIMIT=100
        MAX_SAMPLES=5
        echo "📊 Running LARGE test (100 questions, 5 samples each)"
        ;;
    "custom")
        LIMIT=$CUSTOM_LIMIT
        MAX_SAMPLES=$CUSTOM_MAX_SAMPLES
        echo "📊 Running CUSTOM test ($LIMIT questions, $MAX_SAMPLES samples each)"
        ;;
    *)
        echo "❌ Error: Invalid TEST_SIZE. Must be 'small', 'medium', 'large', or 'custom'"
        exit 1
        ;;

esac

# =============================================================================
# EVALUATION EXECUTION
# =============================================================================

echo ""
echo "🚀 Starting MMaDA GPQA Evaluation"
echo "=================================="
echo "Model: $MODEL_NAME"
echo "CoT Enabled: $ENABLE_COT"
echo "Max Tokens: $MAX_TOKENS"
echo "Gen Length: $GEN_LENGTH"
echo "Steps: $STEPS"
echo "Temperature: $TEMPERATURE"
echo "Seed: $SEED"
echo "Max Subprocesses: $MAX_SUBPROCESSES"
echo "Max Connections: $MAX_CONNECTIONS"
echo "Faithfulness Testing: $ENABLE_FAITHFULNESS"
echo ""

# Set environment variable for faithfulness testing
export GPQA_FAITHFULNESS=$ENABLE_FAITHFULNESS

# Run the evaluation
inspect eval evaluation/gpqa/gpqa_basic.py \
    --model $MODEL_NAME \
    -M enable_cot=$ENABLE_COT \
    -M seed=$SEED \
    -M gen_length=$GEN_LENGTH \
    -M steps=$STEPS \
    --limit $LIMIT \
    --max-samples $MAX_SAMPLES \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --max-subprocesses $MAX_SUBPROCESSES \
    --max-connections $MAX_CONNECTIONS


echo ""
echo "Evaluation completed!"
echo "Check the logs directory for results."

if [ "$ENABLE_FAITHFULNESS" = "true" ]; then
    echo ""
    echo "💡 Faithfulness testing was enabled."
    echo "   The model was given hints to test CoT faithfulness."
    echo "   You can analyze the results by comparing with a control run (ENABLE_FAITHFULNESS=false)."
fi
