#!/bin/bash

# BRTX Setup Script for RLCR
# Run this once before submitting jobs

echo "Setting up RLCR for BRTX cluster..."

# Create logs directory
mkdir -p logs
echo "✓ Created logs directory"

# Verify conda environment exists
if conda env list | grep -q "^rl "; then
    echo "✓ Conda environment 'rl' found"
else
    echo "❌ Conda environment 'rl' not found!"
    echo "Please run: conda env create -f environment.yml"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "rl_runner.py" ]; then
    echo "❌ Please run this script from the RLCR directory"
    exit 1
fi

# Check SLURM availability
if ! command -v sbatch &> /dev/null; then
    echo "❌ SLURM not available - are you on a BRTX node?"
    exit 1
fi

echo "✓ SLURM available"

# Test NVMe storage access
if [ ! -w "/srv/local1" ]; then
    echo "⚠️  Warning: /srv/local1 not writable - jobs may fail"
else
    echo "✓ Fast storage (/srv/local1) accessible"
fi

echo ""
echo "⚠️  IMPORTANT: Before running training, you must:"
echo "   1. Login to HuggingFace: huggingface-cli login"
echo "   2. Login to W&B: wandb login"  
echo "   3. Test downloads: python test_downloads.py"
echo ""
echo "🚀 After authentication, you can submit jobs:"
echo "   sbatch slurm_hotpot_rlcr.sh"
echo "   sbatch slurm_math_rlcr.sh"  
echo "   sbatch slurm_eval.sh"