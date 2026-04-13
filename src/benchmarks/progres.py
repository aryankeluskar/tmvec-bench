#!/usr/bin/env python
"""
Progres Benchmark: Generate pairwise similarity predictions using graph embeddings.

Progres uses E(n)-equivariant graph neural networks to embed protein structures
and calculates similarity via cosine distance of the embeddings.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Add progres to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "progres"))

import progres as pg


def get_structure_files(structure_dir):
    """Get all PDB/CIF files from structure directory."""
    structure_path = Path(structure_dir)
    structure_files = []

    for ext in ['*.pdb', '*.cif']:
        structure_files.extend(structure_path.glob(ext))

    structure_files.sort()
    print(f"Found {len(structure_files)} structure files")
    return structure_files


def generate_embeddings(structure_files, device='cpu', batch_size=8):
    """Generate Progres embeddings for protein structures."""
    print(f"Generating Progres embeddings on {device}...")

    model = pg.load_trained_model(device)

    embeddings = []
    ids = []

    for fp in tqdm(structure_files, desc="Embedding structures"):
        try:
            emb = pg.embed_structure(str(fp), device=device, model=model)
            embeddings.append(emb.cpu())
            # Extract ID from filename (remove extension)
            ids.append(fp.stem)
        except Exception as e:
            print(f"Warning: Failed to embed {fp.name}: {e}")
            continue

    if not embeddings:
        raise RuntimeError("No structures could be embedded")

    embeddings_tensor = torch.stack(embeddings)
    print(f"Generated embeddings: {embeddings_tensor.shape}")
    return ids, embeddings_tensor


def calculate_similarity_matrix(embeddings):
    """Calculate pairwise similarity scores from embeddings."""
    print("Calculating pairwise similarity scores...")

    # Progres embeddings are already normalized, but ensure it
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    # Progres similarity: (1 + cosine) / 2, ranges 0-1
    cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())
    similarity_matrix = (1 + cosine_sim) / 2

    sim_np = similarity_matrix.numpy()
    print(f"Similarity matrix shape: {sim_np.shape}")
    print(f"Mean: {sim_np.mean():.4f}, Std: {sim_np.std():.4f}")

    return sim_np


def save_results(seq_ids, similarity_matrix, output_path):
    """Save pairwise results to CSV in standard format."""
    print(f"Saving to {output_path}...")

    seq1_ids = []
    seq2_ids = []
    tm_scores = []

    n = len(seq_ids)
    for i in range(n):
        for j in range(i + 1, n):
            seq1_ids.append(seq_ids[i])
            seq2_ids.append(seq_ids[j])
            tm_scores.append(float(similarity_matrix[i, j]))

    df = pd.DataFrame({
        'seq1_id': seq1_ids,
        'seq2_id': seq2_ids,
        'tm_score': tm_scores  # Using tm_score for consistency with other methods
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(tm_scores):,} pairs")


def main():
    parser = argparse.ArgumentParser(description="Progres similarity prediction")
    parser.add_argument("--dataset", choices=['cath', 'scope40'], default='cath',
                        help="Dataset to use (cath or scope40)")
    parser.add_argument("--structure-dir", default=None,
                        help="Structure directory (overrides dataset default)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (overrides dataset default)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu, auto-detects if not specified)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding")

    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == 'scope40':
        structure_dir = args.structure_dir or "data/scope40pdb"
        output = args.output or "results/scope40_progres_similarities.csv"
    else:
        structure_dir = args.structure_dir or "data/pdb/cath-s100"
        output = args.output or "results/cath_progres_similarities.csv"

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Progres Similarity Prediction")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    print(f"Structure dir: {structure_dir}")
    print(f"Output: {output}")
    print("=" * 80)

    # Verify path exists
    if not Path(structure_dir).exists():
        raise ValueError(f"Structure directory not found: {structure_dir}")

    structure_files = get_structure_files(structure_dir)
    if not structure_files:
        raise ValueError(f"No structure files found in {structure_dir}")

    seq_ids, embeddings = generate_embeddings(structure_files, device, args.batch_size)
    similarity_matrix = calculate_similarity_matrix(embeddings)
    save_results(seq_ids, similarity_matrix, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
