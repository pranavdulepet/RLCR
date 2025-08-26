#!/bin/bash
# RLCR Full Reproduction Pipeline for BRTX Cluster
# This script coordinates the entire reproduction process

set -e

USER_SCRATCH="/srv/local1/$USER"
WORK_DIR="$USER_SCRATCH/rlcr-reproduce/RLCR"

echo "🚀 RLCR Full Reproduction Pipeline for BRTX"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$WORK_DIR/README.md" ]; then
    echo "❌ RLCR repository not found at $WORK_DIR"
    echo "Please ensure the repository is cloned to: $WORK_DIR"
    exit 1
fi

cd $WORK_DIR

echo "📍 Working directory: $(pwd)"
echo "👤 User: $USER"
echo "💾 Scratch space: $USER_SCRATCH"
echo ""

# Copy scripts to work directory
echo "📋 Setting up training scripts..."
cp /path/to/scripts/*.sh .
chmod +x *.sh

echo ""
echo "🎯 Available training jobs:"
echo "1. RLCR HotpotQA:  sbatch train_rlcr_hotpot_brtx.sh"
echo "2. RLVR HotpotQA:  sbatch train_rlvr_hotpot_brtx.sh"  
echo "3. RLCR Math:      sbatch train_rlcr_math_brtx.sh"
echo "4. Full Evaluation: sbatch eval_all_brtx.sh"
echo ""

echo "⚠️  IMPORTANT NOTES:"
echo "• Math training uses ba100 partition (6 GPUs, 48h)"
echo "• Other jobs use brtx6 partition (4 GPUs, 24h)"
echo "• Ensure you have wandb login configured"
echo "• Monitor jobs with: squeue -u $USER"
echo "• Check GPU usage: gpustat or http://brtx.ccmaymay.net"
echo ""

echo "📊 Expected resource usage:"
echo "• RLCR/RLVR HotpotQA: ~4 GPUs × 20-24 hours"
echo "• RLCR Math: ~6 GPUs × 40-48 hours" 
echo "• Storage: ~50-100GB per trained model"
echo ""

read -p "🤔 Would you like to submit all training jobs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Submitting all training jobs..."
    
    # Submit HotpotQA jobs
    echo "Submitting RLCR HotpotQA job..."
    RLCR_JOB=$(sbatch --parsable train_rlcr_hotpot_brtx.sh)
    echo "Job ID: $RLCR_JOB"
    
    echo "Submitting RLVR HotpotQA job..."
    RLVR_JOB=$(sbatch --parsable train_rlvr_hotpot_brtx.sh)
    echo "Job ID: $RLVR_JOB"
    
    # Submit Math job
    echo "Submitting RLCR Math job..."
    MATH_JOB=$(sbatch --parsable train_rlcr_math_brtx.sh)
    echo "Job ID: $MATH_JOB"
    
    # Submit evaluation job (depends on training completion)
    echo "Submitting evaluation job (depends on training completion)..."
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:$RLCR_JOB:$RLVR_JOB:$MATH_JOB eval_all_brtx.sh)
    echo "Job ID: $EVAL_JOB"
    
    echo ""
    echo "✅ All jobs submitted successfully!"
    echo "📋 Job IDs:"
    echo "  RLCR HotpotQA: $RLCR_JOB"
    echo "  RLVR HotpotQA: $RLVR_JOB"
    echo "  RLCR Math:     $MATH_JOB"
    echo "  Evaluation:    $EVAL_JOB (waiting for training completion)"
    echo ""
    echo "🔍 Monitor progress:"
    echo "  squeue -u $USER"
    echo "  tail -f $USER_SCRATCH/logs/rlcr-*-\$SLURM_JOB_ID.out"
    
else
    echo ""
    echo "ℹ️  Manual submission:"
    echo "1. Review and customize configs if needed"
    echo "2. Submit individual jobs:"
    echo "   sbatch train_rlcr_hotpot_brtx.sh"
    echo "   sbatch train_rlvr_hotpot_brtx.sh"
    echo "   sbatch train_rlcr_math_brtx.sh"
    echo "3. After training completes:"
    echo "   sbatch eval_all_brtx.sh"
fi

echo ""
echo "🔗 Useful monitoring commands:"
echo "• squeue -u $USER                    # Check your job queue"
echo "• scancel <job_id>                   # Cancel a specific job"
echo "• seff <job_id>                      # Job efficiency after completion"
echo "• gpustat                            # Real-time GPU usage"
echo "• df -h $USER_SCRATCH                # Check storage usage"
echo ""
echo "📂 Output locations:"
echo "• Training outputs: $USER_SCRATCH/outputs/"
echo "• Logs: $USER_SCRATCH/logs/"
echo "• Evaluation results: $WORK_DIR/results/"
echo ""
echo "🎉 Setup complete! Happy training! 🚀"