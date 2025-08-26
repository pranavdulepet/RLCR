#!/bin/bash
# RLCR Environment Setup for BRTX Cluster
# Run this from: /srv/local1/pdulepe1/rlcr-reproduce/RLCR/

set -e

USER_SCRATCH="/srv/local1/pdulepe1"
WORK_DIR="$USER_SCRATCH/rlcr-reproduce/RLCR"

echo "🚀 Setting up RLCR environment on BRTX cluster"
echo "User: $USER"
echo "Scratch: $USER_SCRATCH"
echo "Work dir: $WORK_DIR"

# Verify we're in the right location
if [ ! -f "README.md" ] || [ ! -f "rl_runner.py" ]; then
    echo "❌ Please run this script from the RLCR directory"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p $USER_SCRATCH/{outputs,cache,logs,wandb-cache}

# Set up environment variables for BRTX best practices
echo "🔧 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# BRTX Best Practices for RLCR
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1
export SLURM_PARTITION=brtx6-dev
export SBATCH_PARTITION=brtx6

# RLCR Environment Variables
export RLCR_SCRATCH="/srv/local1/pdulepe1"
export HF_HOME="$RLCR_SCRATCH/cache/huggingface"
export TRANSFORMERS_CACHE="$RLCR_SCRATCH/cache/huggingface/transformers"
export WANDB_CACHE_DIR="$RLCR_SCRATCH/wandb-cache"
export WANDB_DIR="$RLCR_SCRATCH/logs/wandb"

EOF

# Source the new environment
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1
export RLCR_SCRATCH="/srv/local1/pdulepe1"
export HF_HOME="$RLCR_SCRATCH/cache/huggingface"
export TRANSFORMERS_CACHE="$RLCR_SCRATCH/cache/huggingface/transformers"
export WANDB_CACHE_DIR="$RLCR_SCRATCH/wandb-cache"
export WANDB_DIR="$RLCR_SCRATCH/logs/wandb"

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $WANDB_CACHE_DIR $WANDB_DIR

echo "✅ Environment setup complete!"
echo ""
echo "🔄 Please run: source ~/.bashrc"
echo "📍 Or start a new shell session to load the environment variables"
echo ""
echo "Next steps:"
echo "1. Activate your conda environment: conda activate rl"
echo "2. Install any missing dependencies if needed"
echo "3. Run the BRTX training scripts"