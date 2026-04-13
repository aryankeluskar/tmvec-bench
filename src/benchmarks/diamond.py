#!/usr/bin/env python
"""
Diamond Benchmark: Generate sequence similarity scores using Diamond blastp.

Diamond is a fast sequence alignment tool. We use it as a sequence-based baseline
to compare against structure-aware methods.
"""

import argparse
from pathlib import Path
import subprocess
import pandas as pd
import tempfile
import sys


def load_fasta(fasta_path, max_sequences=None):
    """Load sequences from FASTA file."""
    seq_ids = []
    current_id = None

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if max_sequences and len(seq_ids) >= max_sequences:
                    break
                current_id = line[1:].split()[0]
                seq_ids.append(current_id)

    print(f"Loaded {len(seq_ids)} sequences")
    return seq_ids


def run_diamond(fasta_file, diamond_bin, threads):
    """Run Diamond all-vs-all blastp search."""
    print("Running Diamond blastp all-vs-all search...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "database"
        tsv_path = Path(tmp_dir) / "results.tsv"

        # Create database
        cmd_makedb = [
            diamond_bin, "makedb",
            "--in", str(fasta_file),
            "--db", str(db_path),
            "--threads", str(threads)
        ]
        result = subprocess.run(cmd_makedb, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Diamond makedb failed: {result.stderr}")

        # Run blastp
        cmd_blastp = [
            diamond_bin, "blastp",
            "--query", str(fasta_file),
            "--db", str(db_path),
            "--out", str(tsv_path),
            "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
            "--threads", str(threads),
            "--max-target-seqs", "100000",
            "--evalue", "10",
            "--ultra-sensitive"
        ]

        result = subprocess.run(cmd_blastp)
        if result.returncode != 0:
            raise RuntimeError("Diamond blastp failed")

        # Read results
        df = pd.read_csv(
            tsv_path, sep='\t', header=None,
            names=['query', 'target', 'pident', 'evalue', 'bitscore'],
            low_memory=False
        )

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df, seq_ids):
    """
    Extract unique pairwise comparisons and convert pident to 0-1 score.
    """
    print("Parsing results...")

    df = df.copy()

    # Remove self-comparisons
    df = df[df['query'] != df['target']]

    print(f"Processing {len(df):,} non-self alignments...")

    # Create canonical pair keys (sorted alphabetically)
    df['seq1_id'] = df[['query', 'target']].min(axis=1)
    df['seq2_id'] = df[['query', 'target']].max(axis=1)

    # Group by unique pairs and aggregate
    # - Mean percent identity (average of both directions)
    # - Min e-value (best significance)
    print("Aggregating bidirectional scores...")
    result_df = df.groupby(['seq1_id', 'seq2_id']).agg(
        pident=('pident', 'mean'),
        evalue=('evalue', 'min')
    ).reset_index()

    # Convert percent identity to 0-1 score for consistency
    result_df['tm_score'] = result_df['pident'] / 100.0

    print(f"Extracted {len(result_df):,} unique pairs")

    # Add missing pairs with score 0 (no alignment found)
    # This ensures all pairs are present for comparison
    existing_pairs = set(zip(result_df['seq1_id'], result_df['seq2_id']))
    missing_pairs = []

    n = len(seq_ids)
    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = sorted([seq_ids[i], seq_ids[j]])
            if (id1, id2) not in existing_pairs:
                missing_pairs.append({
                    'seq1_id': id1,
                    'seq2_id': id2,
                    'tm_score': 0.0,
                    'pident': 0.0,
                    'evalue': float('inf')
                })

    if missing_pairs:
        print(f"Adding {len(missing_pairs):,} pairs with no alignment (score=0)")
        result_df = pd.concat([result_df, pd.DataFrame(missing_pairs)], ignore_index=True)

    return result_df[['seq1_id', 'seq2_id', 'tm_score']].to_dict('records')


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diamond sequence similarity")
    parser.add_argument("--dataset", choices=['cath', 'scope40'], default='cath',
                        help="Dataset to use")
    parser.add_argument("--fasta", default=None,
                        help="FASTA file path (overrides dataset default)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (overrides dataset default)")
    parser.add_argument("--diamond-bin", default="binaries/diamond",
                        help="Path to diamond binary")
    parser.add_argument("--threads", type=int, default=32,
                        help="Number of threads")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Maximum sequences to process")

    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == 'scope40':
        fasta = args.fasta or "data/fasta/scope40-1000.fa"
        output = args.output or "results/scope40_diamond_similarities.csv"
    else:
        fasta = args.fasta or "data/fasta/cath-domain-seqs-S100-1k.fa"
        output = args.output or "results/cath_diamond_similarities.csv"

    print("=" * 80)
    print("Diamond Sequence Similarity")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")
    print(f"Threads: {args.threads}")
    print("=" * 80)

    # Verify paths exist
    if not Path(fasta).exists():
        raise ValueError(f"FASTA file not found: {fasta}")

    seq_ids = load_fasta(fasta, args.max_sequences)
    df = run_diamond(fasta, args.diamond_bin, args.threads)
    pairs = parse_results(df, seq_ids)
    save_results(pairs, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
