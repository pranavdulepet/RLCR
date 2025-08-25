#!/bin/bash
# Resource monitoring for RLCR jobs

echo "🔍 RLCR Resource Monitoring Dashboard"
echo "===================================="

echo ""
echo "📋 SLURM Jobs:"
squeue -u $USER -O 'JobID:.10,Partition:.10,Name:.15,UserName:.8,tres-per-job:.10,StateCompact:.8,TimeUsed:.10,NumNodes:.3,ReasonList:20'

echo ""
echo "🎯 GPU Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "💾 Storage Usage:"
echo "Home directory (25GB limit):"
du -sh $HOME
echo "Work directory (/srv/local1):"
du -sh /srv/local1/$USER/ 2>/dev/null || echo "Work directory not found"
echo "Available space on /srv/local1:"
df -h /srv/local1

echo ""
echo "🗂️ Cache Sizes:"
du -sh /srv/local1/$USER/.cache/ 2>/dev/null || echo "Cache directory not found"

echo ""
echo "🔄 Running Processes:"
ps aux | grep $USER | grep -E "(python|accelerate|deepspeed)" | head -10