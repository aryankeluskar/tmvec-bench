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

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load python/miniforge3_pytorch/2.7.0
# module load mamba/latest && source activate tmvec_distill

OUTPUT_FILE="$REPO_ROOT/results/scope40_foldseek_similarities.csv"
echo "=========================================="
echo "Running Foldseek exhaustive accuracy benchmark on SCOPe40-1000..."
echo ""
echo "Model: Foldseek binaries/foldseek"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.foldseek --dataset scope40 --threads ${SLURM_CPUS_PER_TASK:-1}
echo ""
echo "=========================================="

OUTPUT_FILE="$REPO_ROOT/results/cath_foldseek_similarities.csv"
echo "=========================================="
echo "Running Foldseek exhaustive accuracy benchmark on CATH S100..."
echo ""
echo "Model: Foldseek binaries/foldseek"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.foldseek --dataset cath --threads ${SLURM_CPUS_PER_TASK:-1}
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for Foldseek..."
echo "=========================================="
python src/util/graphs.py foldseek
echo "=========================================="

echo ""
echo "=========================================="
echo "Accuracy benchmark complete."
echo "Use scripts/foldseek-cpu.sh, scripts/foldseek-gpu.sh, and"
echo "scripts/foldseek-gpu-exhaustive.sh for the three runtime configurations."
echo "=========================================="
