#!/bin/bash
#SBATCH --job-name=rlcr-eval
#SBATCH --gpus=1
#SBATCH -p brtx6
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Load conda environment
source ~/.bashrc
conda activate rl

# Verify GPU allocation for evaluation
echo "SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Use fast NVMe storage (BRTX best practice)
export TMPDIR=/srv/local1
mkdir -p $TMPDIR/rlcr_eval_$$
cd $TMPDIR/rlcr_eval_$$

# Copy source code to fast storage
cp -r $SLURM_SUBMIT_DIR/* .

# Run full evaluation suite
bash eval_runs.sh

# Copy results back to submission directory
cp -r eval_outputs/ $SLURM_SUBMIT_DIR/ 2>/dev/null || true
cp -r results/ $SLURM_SUBMIT_DIR/ 2>/dev/null || true
cp -r logs/ $SLURM_SUBMIT_DIR/ 2>/dev/null || true

# Cleanup fast storage
cd $SLURM_SUBMIT_DIR
rm -rf $TMPDIR/rlcr_eval_$$