#!/bin/bash
# Test script when you already have MCTS parquet files
# This skips MCTS and goes straight to predictor training and MAB

set -e  # Exit on error

# Activate virtual environment if it exists (for uv pip installs)
if [ -d ".venv" ]; then
    echo "Activating virtual environment: .venv"
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    echo "Activating virtual environment: ../.venv"
    source ../.venv/bin/activate
fi

# Show which Python is being used
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Configuration
MCTS_PARQUET="${1:-output_files/mcts_qwen_1.parquet}"  # Use first arg or default
TEST_DIR="test_outputs"
mkdir -p $TEST_DIR

echo "=========================================="
echo "Testing with existing MCTS parquet"
echo "=========================================="
echo "Using MCTS parquet: $MCTS_PARQUET"
echo ""

# Quick check of dependencies (detailed setup should be done via setup_dependencies.sh)
echo "Checking dependencies..."
python -c "
import sys
try:
    import torch
    print(f'✅ torch {torch.__version__}')
except ImportError:
    print('❌ torch not found. Run: ./setup_dependencies.sh')
    sys.exit(1)

try:
    import flash_attn
    print('✅ flash_attn (FlashAttention2 available)')
except ImportError:
    print('⚠️  flash_attn not found (will use sdpa fallback). To install: ./setup_dependencies.sh')

try:
    import bitsandbytes
    print('✅ bitsandbytes (8-bit quantization available)')
except ImportError:
    print('⚠️  bitsandbytes not found (8-bit quantization unavailable). To install: ./setup_dependencies.sh')
" || {
    echo ""
    echo "⚠️  Some dependencies are missing. Run './setup_dependencies.sh' to install them."
    echo "   The script will continue but may use fallback implementations."
}
echo ""

# Check if parquet file exists
if [ ! -f "$MCTS_PARQUET" ]; then
    echo "❌ ERROR: MCTS parquet file not found: $MCTS_PARQUET"
    echo ""
    echo "Usage: $0 <path_to_mcts_parquet>"
    echo "Example: $0 output_files/mcts_qwen_1.parquet"
    echo ""
    echo "Or use wildcard for multiple files:"
    echo "  python predictor_train.py --parquet-paths output_files/mcts_qwen_*.parquet ..."
    exit 1
fi

# Step 1: Train Predictor
echo "Step 1: Training Predictor from MCTS output..."
python predictor_train.py \
    --parquet-paths "$MCTS_PARQUET" \
    --output-path ${TEST_DIR}/test_predictor.pt \
    --batch-size 32 \
    --epochs 3 \
    --min-iters-positive 3

# Verify predictor was created
if [ ! -f "${TEST_DIR}/test_predictor.pt" ]; then
    echo "❌ ERROR: Predictor training failed"
    exit 1
fi
echo "✅ Predictor saved to: ${TEST_DIR}/test_predictor.pt"

# Step 2: MAB Training
echo ""
echo "Step 2: Running MAB Training..."
# Set memory optimization - reduce fragmentation
# Based on https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:2

python mab.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu-id 0 \
    --num-chunks 16 \
    --chunk-idx 0 \
    --batch-size 1 \
    --n-steps 3 \
    --lr 1e-5 \
    --tau 0.1 \
    --tau-decay 0.999 \
    --predictor-path ${TEST_DIR}/test_predictor.pt \
    --output-model-path ${TEST_DIR}/test_mab_model.pt \
    --use-lora \
    --use-8bit \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --max-length 256

echo ""
echo "=========================================="
echo "✅ Test completed!"
echo "=========================================="
echo "Outputs saved in: $TEST_DIR"

