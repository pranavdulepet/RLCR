#!/bin/bash
#SBATCH --job-name=rlcr-hotpot
#SBATCH --partition=brtx6
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=/srv/local1/%u/logs/rlcr-hotpot-%j.out
#SBATCH --error=/srv/local1/%u/logs/rlcr-hotpot-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# RLCR HotpotQA Training Job for BRTX Cluster
# Usage: sbatch train_rlcr_hotpot_brtx.sh
# Or for specific node: sbatch --nodelist=brtx604 train_rlcr_hotpot_brtx.sh

set -e

echo "🚀 Starting RLCR HotpotQA Training on BRTX"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Slurm GPUs: $SLURM_GPUS"

# Set up environment variables for BRTX best practices
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

# Create output and log directories
mkdir -p $USER_SCRATCH/logs $USER_SCRATCH/outputs/RLCR-hotpot $WANDB_DIR

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl

echo "🔍 Environment check:"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create deepspeed config for BRTX
cat > ./deepspeed_brtx.yaml << 'EOF'
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
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Create RLCR config for BRTX
cat > ./configs/RLCR_hotpot_brtx.yaml << 'EOF'
# RLCR HotpotQA Config for BRTX Cluster
model_name_or_path: Qwen/Qwen2.5-7B
run_name: "RLCR-hotpot-brtx"
output_dir: /srv/local1/pdulepe1/outputs/RLCR-hotpot
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Data training arguments
dataset_name: mehuldamani/hotpot_qa
sys_prompt_name: "tabc_long"
format_pattern: "tabc"

# GRPO trainer config
bf16: true
beta: 0.0
eval_strategy: "steps"
eval_steps: 50
eval_on_start: false 
gradient_accumulation_steps: 16
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
per_device_eval_batch_size: 16
per_device_train_batch_size: 4
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
vllm_gpu_memory_utilization: 0.3
warmup_ratio: 0.05
wandb_project: "RLCR-BRTX"
EOF

echo "🏋️  Starting RLCR training..."
accelerate launch \
    --num_processes 4 \
    --config_file deepspeed_brtx.yaml \
    rl_runner.py \
    --config configs/RLCR_hotpot_brtx.yaml

echo "✅ RLCR HotpotQA training completed!"

# Clean up temporary files
rm -f deepspeed_brtx.yaml

echo "📊 Training outputs saved to: $USER_SCRATCH/outputs/RLCR-hotpot"
echo "📝 Logs available at: $USER_SCRATCH/logs/"