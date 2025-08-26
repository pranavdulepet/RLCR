#!/bin/bash
#SBATCH --job-name=rlcr-math
#SBATCH --partition=ba100
#SBATCH --gpus=6
#SBATCH --time=2-00:00:00
#SBATCH --output=/srv/local1/%u/logs/rlcr-math-%j.out
#SBATCH --error=/srv/local1/%u/logs/rlcr-math-%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

# RLCR Math Training Job for BRTX Cluster (uses ba100 partition for LLM jobs)
# Usage: sbatch train_rlcr_math_brtx.sh

set -e

echo "🚀 Starting RLCR Math Training on BRTX"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Set up environment variables
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1
export USER_SCRATCH="/srv/local1/$USER"
export HF_HOME="$USER_SCRATCH/cache/huggingface"
export TRANSFORMERS_CACHE="$USER_SCRATCH/cache/huggingface/transformers"
export WANDB_CACHE_DIR="$USER_SCRATCH/wandb-cache"
export WANDB_DIR="$USER_SCRATCH/logs/wandb"

# Set working directory
WORK_DIR="$USER_SCRATCH/rlcr-reproduce/RLCR"
cd $WORK_DIR

# Create output directories
mkdir -p $USER_SCRATCH/outputs/RLCR-math

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl

echo "🔍 Environment check:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create deepspeed config for 6 GPUs
cat > ./deepspeed_math_brtx.yaml << 'EOF'
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
mixed_precision: bf16
num_machines: 1
num_processes: 6
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Create RLCR Math config for BRTX
cat > ./configs/RLCR_math_brtx.yaml << 'EOF'
# RLCR Math Config for BRTX Cluster
model_name_or_path: Qwen/Qwen2.5-7B
run_name: "RLCR-math-brtx"
output_dir: /srv/local1/pdulepe1/outputs/RLCR-math
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Data training arguments
dataset_name: mehuldamani/big_math_digits
sys_prompt_name: "tabc"
format_pattern: "tabc"

# GRPO trainer config
bf16: true
beta: 0.0
eval_strategy: "steps"
eval_steps: 50
eval_on_start: false 
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
hub_strategy: "end"
learning_rate: 1e-06
log_level: info
logging_steps: 5
logging_strategy: steps
lr_scheduler_type: constant_with_warmup
max_prompt_length: 2048
max_completion_length: 1024
max_steps: -1
num_generations: 24
num_iterations: 1
num_train_epochs: 1
overwrite_output_dir: true
per_device_eval_batch_size: 8
per_device_train_batch_size: 2
push_to_hub: false
report_to:
- wandb
reward_funcs: ["format","accuracy","brier","mean_confidence","confidence_one_or_zero"]
reward_weights: [0.5,0.5,0.5,0.000001,0.000001]
save_strategy: "steps"
save_steps: 100
save_total_limit: 2
scale_rewards: false
seed: 43
temperature: 0.7
use_vllm: true
vllm_device: auto
vllm_gpu_memory_utilization: 0.25
warmup_ratio: 0.05
wandb_project: "RLCR-BRTX"
EOF

echo "🏋️  Starting RLCR Math training..."
accelerate launch \
    --num_processes 6 \
    --config_file deepspeed_math_brtx.yaml \
    rl_runner.py \
    --config configs/RLCR_math_brtx.yaml

echo "✅ RLCR Math training completed!"

# Clean up
rm -f deepspeed_math_brtx.yaml

echo "📊 Training outputs saved to: $USER_SCRATCH/outputs/RLCR-math"