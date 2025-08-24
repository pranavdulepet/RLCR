#!/bin/bash
#SBATCH --job-name=math-rlcr
#SBATCH --gpus=6
#SBATCH -p brtx6
#SBATCH --time=24:00:00
#SBATCH --output=logs/math_rlcr_%j.out
#SBATCH --error=logs/math_rlcr_%j.err

# Load conda environment
source ~/.bashrc
conda activate rl

# Verify CUDA_VISIBLE_DEVICES is set by SLURM
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: No GPUs allocated by SLURM!"
    exit 1
fi

# Use fast NVMe storage (BRTX best practice)
export TMPDIR=/srv/local1
mkdir -p $TMPDIR/rlcr_workspace_$$
cd $TMPDIR/rlcr_workspace_$$

# Copy source code to fast storage
cp -r $SLURM_SUBMIT_DIR/* .

# Export environment variables
export WANDB_PROJECT="RLCR-reproduction"

# Run training
accelerate launch --num_processes 6 --config_file deepspeed_6gpu.yaml rl_runner.py --config configs/Qwen-7B/math/RLCR.yaml

# Copy results back to submission directory
cp -r data/ $SLURM_SUBMIT_DIR/
cp -r logs/ $SLURM_SUBMIT_DIR/ 2>/dev/null || true

# Cleanup fast storage
cd $SLURM_SUBMIT_DIR
rm -rf $TMPDIR/rlcr_workspace_$$