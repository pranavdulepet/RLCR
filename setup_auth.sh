#!/bin/bash
# Authentication setup for RLCR on BRTX

echo "🔐 Setting up authentication for RLCR..."

# Set work directory
WORK_DIR="/srv/local1/$(whoami)/rlcr-hotpot"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $WORK_DIR/conda-envs/rl

echo ""
echo "📊 Setting up Weights & Biases (wandb):"
echo "Visit https://wandb.ai/authorize to get your API key"
wandb login

echo ""
echo "🤗 Setting up HuggingFace Hub:"
echo "Visit https://huggingface.co/settings/tokens to create a token"
huggingface-cli login

echo ""
echo "✅ Authentication complete!"
echo "💡 Your cache will be stored in /srv/local1/$(whoami)/.cache/"