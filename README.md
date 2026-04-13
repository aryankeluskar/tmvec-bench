# TM-Vec 2 Benchmarking

Benchmarking library for TMVec-2 Suite, comparing TM-Vec variants against structure and sequence baselines such as Foldseek, TM-align, ProstT5/PLM Blast style embeddings, and Diamond runtime baselines.

## Description

This repo benchmarks TM-Vec models and comparison baselines:
- **Foldseek**: Fast structure comparison using 3Di sequences
- **TM-Vec**: Neural network model for TM-score prediction from ProtT5-XL embeddings
- **TM-Vec 2**: Optimized architecture using Lobster-24M foundation model
- **TM-Vec 2s**: BiLSTM student model distilled from TM-Vec 2
- **ProstT5 / PLM Blast-style baseline**: Sequence-to-3Di retrieval using Foldseek with ProstT5 embeddings
- **Diamond**: Sequence alignment baseline kept for runtime comparison only

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/paarth-b/tmvec-bench.git
cd tmvec-bench
```

### 2. Install Python Dependencies

Using `uv` (recommended):

Install `uv` if not already installed:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Install dependencies using `uv`:
```bash
uv sync
source .venv/bin/activate
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Download Required Binaries

#### TMalign Binary

The provided binary `binaries/TMalign` requires x86-64 architecture. For other architectures (e.g., Apple Silicon), download from [Zhang Group website](https://zhanggroup.org/TM-align/).


#### Foldseek Binary

Download from [Foldseek GitHub releases](https://github.com/steineggerlab/foldseek/releases/).
Place the Foldseek executable in `binaries/foldseek`:

```bash
# Linux AVX2 build (check using: cat /proc/cpuinfo | grep avx2)
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz
mv foldseek/bin/foldseek binaries/foldseek
chmod +x binaries/foldseek
```

```bash
# Linux ARM64 build
wget https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz
tar xvzf foldseek-linux-arm64.tar.gz
mv foldseek/bin/foldseek binaries/foldseek
chmod +x binaries/foldseek
```

Verify installation:
```bash
binaries/foldseek version
```

### 4. Download Model Checkpoints

#### TM-Vec (original model)

Download the TM-Vec CATH checkpoint:

Using huggingface cli (recommended):
```bash
huggingface-cli download scikit-bio/tmvec-cath tm_vec_cath_model.ckpt --local-dir binaries/
```

Or download manually from [HuggingFace Hub](https://huggingface.co/scikit-bio/tmvec-cath/tree/main) and place `tm_vec_cath_model.ckpt` in `binaries/`.

#### TM-Vec 2 Models

```bash
# TM-Vec 2 (Lobster-based teacher model)
huggingface-cli download scikit-bio/tmvec-2 --local-dir models/tmvec-2

# TM-Vec 2s (student model) - already provided in binaries/
# File: binaries/tmvec2_student.pt
```

The configuration file `binaries/tm_vec_cath_model_params.json` is already included in the repository.

## Dataset Setup

### CATH S100 Fasta File

Unzip `data/fasta/cath-domain-seqs.zip` to get `data/fasta/cath-domain-seqs.fa`.

```bash
unzip data/fasta/cath-domain-seqs.zip -d data/fasta
```

### CATH S100 Dataset

The benchmarks use the first 1,000 domains from CATH S100 (non-redundant at 100% sequence identity).

The FASTA file is already provided at `data/cath-top1k.fa`. We provide a zip file of the first 1000 domains of CATH S100 for convenience, that can be unzipped to get the PDB structures.

```bash
unzip data/cath-pdb.zip -d data/
```

Alternatively, if you choose to download structures for the 1000 domains from CATH Database:

```bash
mkdir -p data/pdb/cath-s100

python src/util/download_structures.py \
    --fasta data/cath-top1k.fa \
    --output-dir data/pdb/cath-s100 \
    --dataset cath
```

This will download ~1000 PDB structures from RCSB PDB.


### SCOPe40 Dataset

The benchmarks use 1,000 domains from SCOPe 2.01 clustered at 40% sequence identity.

The FASTA file is already provided at `data/fasta/scope40-1000.fa`. We provide a zip file of the first 1000 domains of SCOPe 2.01 for convenience hosted on Google Drive, that can be unzipped to get the PDB structures.

```bash
wget "https://drive.usercontent.google.com/download?id=1HjtC7Dv-MZABO9wr5PYr5DPLZ6S642P6&export=download&confirm=t" -O data/scope40-pdb.zip
unzip data/scope40-pdb.zip -d data/
```

Alternatively, if you choose to download structures for the 1000 domains from SCOPe Database:

```bash
mkdir -p data/scope40pdb

python src/util/download_structures.py \
    --fasta data/fasta/scope40-1000.fa \
    --output-dir data/scope40pdb \
    --dataset scope40
```

This downloads from ASTRAL/RCSB PDB.

## Running Benchmarks

Using bash scripts in `scripts/` (recommended on clusters):

```bash
# Accuracy benchmarks
bash scripts/tmvec2_student.sh
bash scripts/tmvec2.sh
bash scripts/tmvec1.sh
bash scripts/foldseek.sh
bash scripts/tmalign.sh

# Foldseek runtime benchmarks agreed on April 6, 2026
bash scripts/foldseek-cpu.sh
bash scripts/foldseek-gpu.sh
bash scripts/foldseek-gpu-exhaustive.sh

# Additional baselines
bash scripts/foldseek-prost.sh
bash scripts/diamond.sh
```

Alternatively, all benchmark code is in `src/benchmarks` and `src/time_benchmarks`. They can be run locally.

```bash
uv run python -m src.benchmarks.{model_file}
uv run python -m src.time_benchmarks.{time_benchmark_file}
```

Example:
```bash
uv run python -m src.benchmarks.tmvec1
uv run python -m src.time_benchmarks.tmvec1_time_benchmark
```
> **_NOTE:_**  TMAlign is a cpu-based script, and may take a long time (>10 Hours) to generate 500,000 pair scores. For convenience, TMAlign results already exist in the results/ folder.

## Output Files

### Similarity Results

All benchmarks generate CSV files in `results/` with the following format:

| seq1_id | seq2_id | tm_score | evalue (Foldseek only) |
|---------|---------|----------|------------------------|
| 107lA00 | 108lA00 | 0.8523   | 1.2e-10               |
| 107lA00 | 109lA00 | 0.7234   | 3.4e-08               |


### Visualization

To generate plots from results, follow readme instructions in the following:
Generated plots are available in the plots subfolders.

```bash
# CATH visualizations
cd src/plotting/cath

# SCOPe visualizations
cd src/plotting/scope

# Runtime benchmarks
cd src/plotting/time
```

Plots are saved to `figures/` and include:
- ROC curves (homology detection at different classification levels)
- PR curves (precision-recall)
- Density scatter plots (predicted vs. true TM-scores)
- Runtime comparisons (encoding and query times)

Runtime aggregation now discovers both legacy top-level CSVs and per-run directories in `results/time_benchmarks/`.

Notes:
- `src/benchmarks/foldseek.py` now supports default vs exhaustive search and optional GPU use.
- `src/time_benchmarks/foldseek_time_benchmark.py` now supports the agreed runtime configurations: CPU default, GPU default, and GPU exhaustive.
- `src/plotting/cath/merge_tables.py` and `src/plotting/scope/merge_tables.py` will include ProstT5 results automatically when `results/cath_prostt5_similarities.csv` or `results/scope40_prostt5_similarities.csv` exist.

## Validation of Published Results

To validate the results in the ISMB 2026 paper:

1. **Table 1 (Prediction Accuracy)**: Run all benchmarks on both CATH and SCOPe40, then compare the generated CSVs against TM-align ground truth using the plotting notebooks.

2. **Figure 4 (TM-score Prediction)**: Generate density scatter plots showing correlation between predicted and true TM-scores.

3. **Figure 5 (Homology Detection)**: Use the ground truth classification files to compute ROC/PR curves at different hierarchy levels (Class → Superfamily/Family).

4. **Supplementary Tables (Runtime)**: Time benchmarks are in `src/time_benchmarks/`. Results should match the encoding/query time tables.
