#!/bin/bash

# MMaDA MATH Evaluation Script
# Customizable parameters for different test sizes

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Test size configuration
# Options: "small", "medium", "large", "custom"
TEST_SIZE="custom"

# Custom parameters (used when TEST_SIZE="custom")
CUSTOM_LIMIT=158
CUSTOM_MAX_SAMPLES=5

# Model configuration
MODEL_NAME="mmada-cot/test"
ENABLE_COT="True"
MAX_TOKENS=448
TEMPERATURE=1.0
SEED=42  # Fixed seed for reproducible evaluations
# Internal generation parameters
GEN_LENGTH=448
STEPS=448
# Sampling method configuration
SAMPLING_METHOD="${SAMPLING_METHOD:-low_confidence}"  # Options: "low_confidence", "random"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"  # Block length for semi-autoregressive generation
# Answer emergence analysis options
ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT=${ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT:-true}

# =============================================================================
# SAMPLING METHOD EXPLANATION
# =============================================================================
# MMaDA uses different remasking strategies during generation:
#
# - "low_confidence": Remasks tokens with low confidence scores (default)
#   * More deterministic, focuses on uncertain tokens
#   * Generally produces higher quality but potentially less diverse output
#   * Good for tasks requiring accuracy and consistency
#
# - "random": Randomly remasks tokens during generation
#   * More stochastic, introduces randomness in the generation process
#   * May produce more diverse but potentially less consistent output
#   * Good for creative tasks or when you want more variety
#
# BLOCK_LENGTH controls semi-autoregressive generation:
# - Smaller values (e.g., 16, 32): More parallel, faster generation
# - Larger values (e.g., 64, 128): More sequential, potentially higher quality
# - Should be a divisor of GEN_LENGTH for optimal performance
#
# =============================================================================

# Memory management (reduce these if you encounter CUDA OOM errors)
MAX_SUBPROCESSES=1
MAX_CONNECTIONS=1

# Faithfulness testing
# Set to "true" to enable hint-based faithfulness testing
# Set to "false" for basic evaluation
# Can be overridden by setting GPQA_FAITHFULNESS environment variable
ENABLE_FAITHFULNESS="${GPQA_FAITHFULNESS:-false}"

# MATH-specific parameters
MATH_LEVELS="4,5"  # Math difficulty levels (1-5)
# Numeric-only filtering for answer emergence analysis
NUMERIC_ONLY=${NUMERIC_ONLY:-false}

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
        echo "📊 Running SMALL test (5 problems, 2 samples each)"
        ;;
    "medium")
        LIMIT=20
        MAX_SAMPLES=3
        echo "📊 Running MEDIUM test (20 problems, 3 samples each)"
        ;;
    "large")
        LIMIT=100
        MAX_SAMPLES=5
        echo "📊 Running LARGE test (100 problems, 5 samples each)"
        ;;
    "custom")
        LIMIT=$CUSTOM_LIMIT
        MAX_SAMPLES=$CUSTOM_MAX_SAMPLES
        echo "📊 Running CUSTOM test ($LIMIT problems, $MAX_SAMPLES samples each)"
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
echo "🚀 Starting MMaDA MATH Evaluation"
echo "=================================="
echo "Model: $MODEL_NAME"
echo "CoT Enabled: $ENABLE_COT"
echo "Max Tokens: $MAX_TOKENS"
echo "Gen Length: $GEN_LENGTH"
echo "Steps: $STEPS"
echo "Temperature: $TEMPERATURE"
echo "Seed: $SEED"
echo "Sampling Method: $SAMPLING_METHOD"
echo "Block Length: $BLOCK_LENGTH"
echo "Math Levels: $MATH_LEVELS"
echo "Max Subprocesses: $MAX_SUBPROCESSES"
echo "Max Connections: $MAX_CONNECTIONS"
echo "Faithfulness Testing: $ENABLE_FAITHFULNESS"
echo "Skip single-digit answers in emergence: $ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT"
echo "Numeric-only questions: $NUMERIC_ONLY"
echo ""

# Set environment variable for faithfulness testing
export GPQA_FAITHFULNESS=$ENABLE_FAITHFULNESS
# Set analyzer behavior
export ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT=$ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT

# Run the evaluation
inspect eval evaluation/math/math_basic.py \
    --model $MODEL_NAME \
    -M enable_cot=$ENABLE_COT \
    -M seed=$SEED \
    -M gen_length=$GEN_LENGTH \
    -M steps=$STEPS \
    -M remasking=$SAMPLING_METHOD \
    -M block_length=$BLOCK_LENGTH \
    -T levels=$MATH_LEVELS \
    -T numeric_only=$NUMERIC_ONLY \
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

echo ""
echo "📝 Example usage with different sampling methods:"
echo "   # Use random sampling for more diverse output:"
echo "   SAMPLING_METHOD=random bash run_math_eval.sh"
echo ""
echo "   # Use smaller block length for faster generation:"
echo "   BLOCK_LENGTH=16 bash run_math_eval.sh"
echo ""
echo "   # Combine both for faster, more diverse generation:"
echo "   SAMPLING_METHOD=random BLOCK_LENGTH=16 bash run_math_eval.sh"
