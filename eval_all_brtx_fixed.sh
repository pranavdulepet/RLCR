#!/bin/bash
#SBATCH --job-name=rlcr-eval
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --time=11:59:59
#SBATCH --output=/srv/local1/%u/logs/rlcr-eval-%j.out
#SBATCH --error=/srv/local1/%u/logs/rlcr-eval-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# RLCR Evaluation Job for BRTX Cluster
# Usage: sbatch eval_all_brtx.sh

set -e

echo "🚀 Starting RLCR Evaluation on BRTX"
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

# Create evaluation configs directory if it doesn't exist
mkdir -p eval_configs/BRTX-models

# Create updated evaluation configs for locally trained models
cat > eval_configs/BRTX-models/hotpot-eval-em.json << 'EOF'
[
{
    "dataset_name": "mehuldamani/hotpot_qa",
    "hash_key": "problem",
    "store_name": "eval_outputs/BRTX-models/hotpot-eval-em",
    "gpu_memory_utilization": 0.9,
    "log_path": "results/BRTX-models/hotpot-eval-em"
}, 
{   "name": "Base", 
    "model": "Qwen/Qwen2.5-7B",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tac",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLVR-Local", 
    "model": "/srv/local1/pdulepe1/outputs/RLVR-hotpot",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "gen",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLCR-Local", 
    "model": "/srv/local1/pdulepe1/outputs/RLCR-hotpot",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tabc_long",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLVR-Original", 
    "model": "mehuldamani/hotpot-v2-correctness-7b",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "gen",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLCR-Original", 
    "model": "mehuldamani/hotpot-v2-brier-7b-no-split",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tabc_long",
    "vllm_task": ["confidence_at_end","ans_at_end"]
}
]
EOF

# Create math evaluation config
cat > eval_configs/BRTX-models/math-500.json << 'EOF'
[
{
    "dataset_name": "mehuldamani/big_math_digits",
    "hash_key": "problem",
    "store_name": "eval_outputs/BRTX-models/math-500",
    "gpu_memory_utilization": 0.9,
    "log_path": "results/BRTX-models/math-500",
    "sample_size": 500
}, 
{   "name": "Base", 
    "model": "Qwen/Qwen2.5-7B",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tac",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLCR-Math-Local", 
    "model": "/srv/local1/pdulepe1/outputs/RLCR-math",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tabc",
    "vllm_task": ["confidence_at_end","ans_at_end"]
},
{   "name": "RLCR-Math-Original", 
    "model": "mehuldamani/big-math-digits-v2-brier-base-tabc",
    "check_fn": "confidence_verifier",
    "sys_prompt_name": "tabc",
    "vllm_task": ["confidence_at_end","ans_at_end"]
}
]
EOF

echo "📊 Running HotpotQA Evaluation..."
python evaluation.py --config eval_configs/BRTX-models/hotpot-eval-em.json

echo "🔢 Running Math Evaluation..."
python evaluation.py --config eval_configs/BRTX-models/math-500.json

echo "✅ All evaluations completed!"
echo "📊 Results saved to:"
echo "  - results/BRTX-models/hotpot-eval-em/"
echo "  - results/BRTX-models/math-500/"

# Create summary report
echo "📋 Creating evaluation summary..."
python << 'EOF'
import json
import os
import pandas as pd

def load_metrics(path):
    try:
        with open(os.path.join(path, "metrics.json"), "r") as f:
            return json.load(f)
    except:
        return {}

# Load results
results = {}
base_path = "results/BRTX-models"

for dataset in ["hotpot-eval-em", "math-500"]:
    dataset_path = os.path.join(base_path, dataset)
    if os.path.exists(dataset_path):
        for model_dir in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_dir)
            if os.path.isdir(model_path):
                metrics = load_metrics(model_path)
                results[f"{dataset}_{model_dir}"] = metrics

# Print summary
print("\n" + "="*60)
print("RLCR EVALUATION SUMMARY")
print("="*60)

for key, metrics in results.items():
    if metrics:
        print(f"\n{key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

print("\n" + "="*60)
EOF

echo "🎉 RLCR evaluation pipeline completed successfully!"