#!/usr/bin/env python
"""Foldseek benchmark for pairwise structure comparison."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd


def get_pdb_files(structure_dir):
    """Get all PDB files from structure directory efficiently using os.scandir."""
    pdb_files = []
    structure_path = Path(structure_dir)
    
    # Use os.scandir for better performance on large directories
    with os.scandir(structure_path) as entries:
        for entry in entries:
            if entry.is_file() and (entry.name.endswith('.pdb') or entry.name.endswith('.cif')):
                pdb_files.append(Path(entry.path))
    
    pdb_files.sort()
    print(f"Found {len(pdb_files)} structure files")
    return pdb_files


def run_foldseek(
    structure_dir,
    foldseek_bin,
    threads,
    use_gpu=False,
    exhaustive_search=True,
    max_seqs=100000,
    min_ungapped_score=0,
    evalue=10,
):
    """Run Foldseek all-vs-all search."""
    print("Running Foldseek all-vs-all search...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tsv_path = Path(tmp_dir) / "results.tsv"

        cmd = [
            foldseek_bin, "easy-search",
            structure_dir, structure_dir,
            str(tsv_path), tmp_dir,
            "--format-output", "query,target,alntmscore,evalue",
            "--threads", str(threads),
            "-e", str(evalue),
            "--max-seqs", str(max_seqs),
            "--min-ungapped-score", str(min_ungapped_score),
        ]
        if exhaustive_search:
            cmd.extend(["--exhaustive-search", "1"])
        if use_gpu:
            cmd.extend(["--gpu", "1"])

        # Run without capturing output so we can see progress
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError("Foldseek failed")

        # Read results before tmp_dir is deleted
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                        names=['query', 'target', 'alntmscore', 'evalue'],
                        low_memory=False)

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df):
    """
    Extract unique pairwise comparisons and average bidirectional scores.
    
    Uses vectorized pandas operations instead of iterrows for massive speedup
    on large result sets (100-1000x faster for millions of rows).
    """
    print("Parsing results...")
    
    # Vectorized extraction of IDs from file paths
    # Much faster than applying Path().stem row by row
    df = df.copy()
    df['q_id'] = df['query'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    df['t_id'] = df['target'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    
    # Handle case where extraction failed (simple filenames without path)
    mask_q = df['q_id'].isna()
    mask_t = df['t_id'].isna()
    if mask_q.any():
        df.loc[mask_q, 'q_id'] = df.loc[mask_q, 'query'].str.replace(r'\.[^.]+$', '', regex=True)
    if mask_t.any():
        df.loc[mask_t, 't_id'] = df.loc[mask_t, 'target'].str.replace(r'\.[^.]+$', '', regex=True)
    
    # Remove _MODEL_* suffix if present (vectorized)
    df['q_id'] = df['q_id'].str.split('_MODEL_').str[0]
    df['t_id'] = df['t_id'].str.split('_MODEL_').str[0]
    
    # Filter out self-comparisons
    df = df[df['q_id'] != df['t_id']]
    
    print(f"Processing {len(df):,} non-self alignments...")
    
    # Create canonical pair keys (sorted alphabetically)
    # This ensures (A,B) and (B,A) map to the same key
    df['seq1_id'] = df[['q_id', 't_id']].min(axis=1)
    df['seq2_id'] = df[['q_id', 't_id']].max(axis=1)
    
    # Group by unique pairs and aggregate
    # - Mean TM-score (average of both directions)
    # - Min e-value (best significance)
    print("Aggregating bidirectional scores...")
    result_df = df.groupby(['seq1_id', 'seq2_id']).agg(
        tm_score=('alntmscore', 'mean'),
        evalue=('evalue', 'min')
    ).reset_index()
    
    print(f"Extracted {len(result_df):,} unique pairs")
    return result_df.to_dict('records')


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Foldseek pairwise benchmark")
    parser.add_argument(
        "legacy_dataset",
        nargs="?",
        choices=["cath", "scope40"],
        help="Legacy positional dataset selector kept for backward compatibility",
    )
    parser.add_argument(
        "--dataset",
        choices=["cath", "scope40"],
        default="cath",
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--structure-dir",
        default=None,
        help="Override the default structure directory for the selected dataset",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the default output CSV path",
    )
    parser.add_argument(
        "--foldseek-bin",
        default="binaries/foldseek",
        help="Path to the Foldseek binary",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of Foldseek threads",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration for the search stage",
    )
    parser.add_argument(
        "--no-exhaustive-search",
        action="store_true",
        help="Disable exhaustive all-vs-all mode and keep Foldseek defaults",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=100000,
        help="Maximum hits per query to pass the prefilter",
    )
    parser.add_argument(
        "--min-ungapped-score",
        type=int,
        default=0,
        help="Minimum ungapped score threshold passed to Foldseek",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=10.0,
        help="E-value cutoff passed to Foldseek",
    )
    args = parser.parse_args()

    dataset = args.legacy_dataset or args.dataset
    is_scope40 = dataset == "scope40"

    # Dataset configurations (paths match tmalign.py)
    if is_scope40:
        structure_dir = "data/scope40pdb"
        output = "results/scope40_foldseek_similarities.csv"
    else:
        structure_dir = "data/pdb/cath-s100"
        output = "results/cath_foldseek_similarities.csv"

    structure_dir = args.structure_dir or structure_dir
    output = args.output or output
    foldseek_bin = args.foldseek_bin
    exhaustive_search = not args.no_exhaustive_search

    print("=" * 80)
    print("Foldseek Benchmark")
    print(f"Dataset: {'SCOPe40' if is_scope40 else 'CATH'}")
    print(f"Structure dir: {structure_dir}")
    print(f"Output: {output}")
    print(f"Threads: {args.threads}")
    print(f"GPU: {args.use_gpu}")
    print(f"Exhaustive search: {exhaustive_search}")
    print("=" * 80)

    # Verify paths exist
    if not Path(structure_dir).exists():
        raise ValueError(f"Structure directory not found: {structure_dir}")
    if not Path(foldseek_bin).exists():
        raise ValueError(f"Foldseek binary not found: {foldseek_bin}")

    pdb_files = get_pdb_files(structure_dir)
    if not pdb_files:
        raise ValueError(f"No structure files found in {structure_dir}")

    df = run_foldseek(
        structure_dir=structure_dir,
        foldseek_bin=foldseek_bin,
        threads=args.threads,
        use_gpu=args.use_gpu,
        exhaustive_search=exhaustive_search,
        max_seqs=args.max_seqs,
        min_ungapped_score=args.min_ungapped_score,
        evalue=args.evalue,
    )
    pairs = parse_results(df)
    save_results(pairs, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
