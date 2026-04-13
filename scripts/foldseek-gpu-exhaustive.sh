#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

export HYDRA_FULL_ERROR=1

echo ""
echo "=========================================="
echo "Running Foldseek GPU exhaustive Time Benchmark..."
echo "=========================================="
python src/time_benchmarks/foldseek_time_benchmark.py \
    --structure-dir data/pdb/cath-s100 \
    --threads ${SLURM_CPUS_PER_TASK:-1} \
    --use-gpu \
    --exhaustive-search \
    --output-dir results/time_benchmarks/foldseek_gpu_exhaustive
