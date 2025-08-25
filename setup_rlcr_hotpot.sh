#!/bin/bash
# RLCR HotpotQA Setup Script for BRTX Cluster
# Run this script on a BRTX node (e.g., brtx601)

set -e

echo "🚀 Setting up RLCR for HotpotQA on BRTX cluster..."

# Set up environment variables for BRTX best practices
export RAYON_RS_NUM_CPUS=1
export MKL_NUM_THREADS=1

# Use fast local storage
WORK_DIR="/srv/local1/$(whoami)/rlcr-hotpot"
echo "📁 Creating work directory: $WORK_DIR"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clone repositories
echo "📥 Cloning RLCR repository..."
if [ ! -d "RLCR" ]; then
    git clone https://github.com/damanimehul/RLCR.git
fi
cd RLCR

echo "📥 Cloning TRL dependency..."
cd ../
if [ ! -d "trl" ]; then
    git clone https://github.com/huggingface/trl.git
    cd trl/
    git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
    cd ../
fi

# Create conda environment
echo "🐍 Creating conda environment..."
cd RLCR
conda env create -f environment.yml --prefix $WORK_DIR/conda-envs/rl

echo "🔧 Activating environment and installing TRL..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $WORK_DIR/conda-envs/rl

# Install TRL
cd ../trl/
pip install -e .
cd ../RLCR/

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate environment: conda activate $WORK_DIR/conda-envs/rl"
echo "2. Login to wandb: wandb login"
echo "3. Login to HuggingFace: huggingface-cli login"
echo "4. Review and modify configs in configs/Qwen-7B/hotpot/"
echo "5. Submit SLURM job for training"
echo ""
echo "💡 Work directory: $WORK_DIR"