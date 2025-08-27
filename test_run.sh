#!/bin/bash
#SBATCH --job-name=rlcr-test
#SBATCH --partition=brtx6-dev
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/rlcr_test_%j.out
#SBATCH --error=logs/rlcr_test_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for performance
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1

# Print system info
echo "=== System Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "==================="

# Navigate to RLCR directory
cd /Users/macbookair/jhu/research/llm-confidence-proj/rlcr-brtx/RLCR

# Activate conda environment
source ~/.bashrc
conda activate rl

# Run the test training
echo "Starting RLCR test training..."
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/hotpot/RLCR-test.yaml

echo "Test training completed!"