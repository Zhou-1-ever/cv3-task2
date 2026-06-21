#!/bin/bash
# =============================================================================
# ACT Cross-Environment Generalization Experiment Pipeline
# =============================================================================
# This script runs the complete pipeline:
#   1. Train ACT on Environment B only
#   2. Train ACT on Environments A+B+C (joint)
#   3. Zero-shot evaluation on Environment D
#   4. Compare results and generate visualizations
#
# Prerequisites:
#   conda activate calvin-act
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh [--skip-train] [--eval-only] [--quick]
#
# Options:
#   --skip-train   Skip training, only run evaluation and comparison
#   --eval-only    Only run evaluation on existing models
#   --quick        Quick test with reduced epochs (for debugging)
# =============================================================================

set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda environment
CONDA_ENV="calvin-act"

# Parse arguments
SKIP_TRAIN=false
EVAL_ONLY=false
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-train)
            SKIP_TRAIN=true
            ;;
        --eval-only)
            EVAL_ONLY=true
            SKIP_TRAIN=true
            ;;
        --quick)
            QUICK_MODE=true
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-train] [--eval-only] [--quick]"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper function: run a Python script with proper environment
# ---------------------------------------------------------------------------
run_python() {
    local desc="$1"
    local script="$2"
    shift 2
    echo ""
    echo "########################################################################"
    echo "  $desc"
    echo "########################################################################"
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    python "$script" "$@"
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 1: Train on Environment B
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    if [ "$QUICK_MODE" = true ]; then
        run_python "STAGE 1: Quick training on Environment B" \
            train_env_b.py --quick
    else
        run_python "STAGE 1: Training ACT on Environment B" \
            train_env_b.py
    fi
else
    echo "[SKIP] Stage 1: Training on Environment B"
fi

# ---------------------------------------------------------------------------
# Stage 2: Train on Environments A+B+C
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    if [ "$QUICK_MODE" = true ]; then
        run_python "STAGE 2: Quick training on Env A+B+C" \
            train_env_abc.py --quick
    else
        run_python "STAGE 2: Training ACT on Environments A+B+C" \
            train_env_abc.py
    fi
else
    echo "[SKIP] Stage 2: Training on Environments A+B+C"
fi

# ---------------------------------------------------------------------------
# Stage 3: Zero-shot Evaluation on Environment D
# ---------------------------------------------------------------------------
run_python "STAGE 3: Zero-shot Evaluation on Environment D" \
    eval_env_d.py --model both

# ---------------------------------------------------------------------------
# Stage 4: Comparison and Visualization
# ---------------------------------------------------------------------------
run_python "STAGE 4: Comparison and Visualization" \
    compare_results.py

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "########################################################################"
echo "  Pipeline Complete!"
echo "########################################################################"
echo ""
echo "Outputs:"
echo "  Model (Env B):    outputs/model_env_b/"
echo "  Model (Env ABC):  outputs/model_env_abc/"
echo "  Evaluation:       outputs/eval_env_d_combined.json"
echo "  Plots:            outputs/*.png"
echo "  Report:           outputs/analysis_report.txt"
echo ""
