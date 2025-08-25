#!/bin/bash
# Emergency stop script for RLCR jobs

echo "🛑 Emergency stop for RLCR jobs..."

# Cancel all your SLURM jobs
echo "Canceling all SLURM jobs for user: $USER"
scancel -u $USER

# Show remaining jobs
echo "Remaining jobs:"
squeue -u $USER

# Kill any remaining Python processes (use with caution!)
echo "Checking for remaining Python processes..."
ps aux | grep $USER | grep python

echo "✅ Emergency stop completed!"
echo "💡 Check GPU usage: nvidia-smi"
echo "💡 Check disk usage: df -h /srv/local1"