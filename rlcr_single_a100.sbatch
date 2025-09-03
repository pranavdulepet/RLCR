#!/bin/bash
#SBATCH --job-name=rlcr-single-rtx
#SBATCH --partition=brtx6          
#SBATCH --gpus=1                   
#SBATCH --time=20:00:00           
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32G                 
#SBATCH --output=logs/rlcr_rtx_%j.out
#SBATCH --error=logs/rlcr_rtx_%j.err

# Exit on any error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start Time: $(date)"
echo "GPUs Allocated: $SLURM_GPUS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Set environment variables for optimal performance on single GPU
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false    # Avoid tokenizer warnings
export HF_HOME=/tmp/huggingface        # Use local storage for HF cache
export WANDB_CACHE_DIR=/tmp/wandb      # Use local storage for wandb
export OMP_NUM_THREADS=8               # Match cpus-per-task
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Show GPU info
echo "=== GPU Information ==="
nvidia-smi
echo "=========================="

# Navigate to RLCR directory
cd /brtx/601-nvme1/pdulepe1/rlcr-reproduce/RLCR 
# Alternative if using home dir: cd ~/RLCR

# Verify environment
echo "=== Environment Check ==="
which python
python --version
pip list | grep -E "(torch|transformers|accelerate|deepspeed)"
echo "=========================="

# Check if required config exists
CONFIG_FILE="configs/Qwen-7B/hotpot/RLCR-single-rtx.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found"
    echo "Please create the config file before running this job."
    exit 1
fi

# Start training
echo "=== Starting RLCR Training ==="
echo "Config: $CONFIG_FILE"
echo "Start time: $(date)"
echo "=============================="

# Run with proper error handling
if accelerate launch \
    --num_processes 1 \
    --config_file deepspeed.yaml \
    rl_runner.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee logs/training_output_${SLURM_JOB_ID}.log; then
    
    echo "=== Training Completed Successfully ==="
    echo "End time: $(date)"
    echo "Job Duration: $SECONDS seconds"
    
else
    echo "=== Training Failed ==="
    echo "End time: $(date)"
    echo "Check logs for details: logs/rlcr_a100_${SLURM_JOB_ID}.err"
    exit 1
fi

# Show final GPU memory state
echo "=== Final GPU State ==="
nvidia-smi
echo "======================="

echo "Job completed at: $(date)"