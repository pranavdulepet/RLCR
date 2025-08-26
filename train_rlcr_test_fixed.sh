#!/bin/bash
#SBATCH --job-name=rlcr-test
#SBATCH -p brtx6
#SBATCH -G 4
#SBATCH --time=04:00:00
#SBATCH --output=/srv/local1/%u/logs/rlcr-test-%j.out
#SBATCH --error=/srv/local1/%u/logs/rlcr-test-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# RLCR Test Training Job for BRTX Cluster
# Usage: sbatch train_rlcr_test_fixed.sh

set -e

echo "🚀 Starting RLCR Test Training on BRTX"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set up environment variables
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1
export USER_SCRATCH="/srv/local1/$USER"
export HF_HOME="$USER_SCRATCH/cache/huggingface"
export TRANSFORMERS_CACHE="$USER_SCRATCH/cache/huggingface/transformers"

# Set working directory
WORK_DIR="$USER_SCRATCH/rlcr-reproduce/RLCR"
cd $WORK_DIR

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl

echo "🔍 Environment check:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create output directory
mkdir -p /srv/local1/$USER/outputs/RLCR-hotpot-test
mkdir -p /srv/local1/$USER/logs

echo "🧪 Starting RLCR Test Training (10% data, float16)..."
python rl_runner.py RLCR_hotpot_test_brtx.yaml

echo "✅ RLCR Test Training completed!"
echo "📊 Model saved to: /srv/local1/$USER/outputs/RLCR-hotpot-test"