#!/bin/bash
#SBATCH --job-name=rlcr-test-fixed
#SBATCH --partition=brtx6-dev
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/rlcr_test_fixed_%j.out
#SBATCH --error=logs/rlcr_test_fixed_%j.err

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
cd /brtx/601-nvme1/pdulepe1/rlcr-reproduce/RLCR

# Activate conda environment
source ~/.bashrc
conda activate rl

# Run with explicit arguments (bypasses accelerate config file bug)
echo "Starting RLCR test training with explicit arguments..."
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --config_file deepspeed.yaml rl_runner.py \
  --dataset_name mehuldamani/hotpot_qa \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --run_name "RLCR-hotpot-test" \
  --output_dir data/RLCR-hotpot-test \
  --model_revision main \
  --torch_dtype float16 \
  --attn_implementation flash_attention_2 \
  --fp16 true \
  --beta 0.0 \
  --eval_strategy steps \
  --eval_steps 5 \
  --eval_on_start false \
  --format_pattern tabc \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 1e-06 \
  --log_level info \
  --logging_steps 1 \
  --logging_strategy steps \
  --lr_scheduler_type constant_with_warmup \
  --max_prompt_length 1024 \
  --max_completion_length 512 \
  --max_steps 20 \
  --num_generations 8 \
  --num_iterations 1 \
  --num_train_epochs 1 \
  --overwrite_output_dir true \
  --per_device_eval_batch_size 4 \
  --per_device_train_batch_size 2 \
  --push_to_hub false \
  --report_to wandb \
  --reward_funcs format accuracy brier mean_confidence confidence_one_or_zero \
  --reward_weights 0.5 0.5 0.5 0.000001 0.000001 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 2 \
  --scale_rewards false \
  --seed 43 \
  --sys_prompt_name tabc_long \
  --temperature 0.7 \
  --use_vllm false \
  --warmup_ratio 0.05

echo "Test training completed!"