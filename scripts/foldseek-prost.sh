#!/bin/bash
set -e

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

echo ""
echo "=========================================="
echo "Running ProstT5 GPU Time Benchmark..."
echo "=========================================="
python src/time_benchmarks/foldseek_plm_time_benchmark.py \
    --fasta data/fasta/cath-domain-seqs-S100-1k.fa \
    --threads ${SLURM_CPUS_PER_TASK:-1} \
    --use-gpu \
    --output-dir results/time_benchmarks/prostt5_gpu
