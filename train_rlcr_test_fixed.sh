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

# Create DeepSpeed config for float16
cat > ./deepspeed_brtx_fp16.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

accelerate launch \
    --num_processes 4 \
    --config_file deepspeed_brtx_fp16.yaml \
    rl_runner.py \
    --config RLCR_hotpot_test_brtx.yaml

echo "✅ RLCR Test Training completed!"
echo "📊 Model saved to: /srv/local1/$USER/outputs/RLCR-hotpot-test"