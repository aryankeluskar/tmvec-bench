#!/usr/bin/env python
"""
ProstT5 Benchmark: Generate pairwise similarity predictions from protein sequences.

ProstT5 uses a protein language model to predict 3Di structural sequences
directly from amino acid sequences, then performs 3Di+AA alignment.

Output: fident (fraction identity in 3Di+AA alignment) as similarity metric.
Note: ProstT5 cannot output TM-scores since it doesn't have Cα coordinates.
"""

from pathlib import Path
import subprocess
import pandas as pd
import tempfile
import sys


def download_prostt5_weights(foldseek_bin, weights_dir):
    """Download ProstT5 model weights using foldseek databases command."""
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            foldseek_bin, "databases",
            "ProstT5",
            str(weights_path / "prostt5"),
            tmp_dir
        ]

        print(f"Downloading ProstT5 weights to {weights_path}...")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError("Failed to download ProstT5 weights")

    return str(weights_path / "prostt5")


def run_prostt5_search(fasta_file, foldseek_bin, prostt5_weights, threads, use_gpu=True):
    """Run Foldseek all-vs-all search using ProstT5 for sequence-to-3Di conversion."""
    print("Running ProstT5 all-vs-all search...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tsv_path = Path(tmp_dir) / "results.tsv"
        query_db = Path(tmp_dir) / "query_db"
        target_db = Path(tmp_dir) / "target_db"

        # Create query database with ProstT5
        print("Creating query database with ProstT5...")
        createdb_cmd = [
            foldseek_bin, "createdb",
            fasta_file, str(query_db),
            "--prostt5-model", prostt5_weights,
            "--threads", str(threads)
        ]
        if use_gpu:
            createdb_cmd.extend(["--gpu", "1"])

        result = subprocess.run(createdb_cmd)
        if result.returncode != 0:
            raise RuntimeError("Failed to create query database")

        # Create target database (same as query for all-vs-all)
        print("Creating target database with ProstT5...")
        createdb_cmd_target = [
            foldseek_bin, "createdb",
            fasta_file, str(target_db),
            "--prostt5-model", prostt5_weights,
            "--threads", str(threads)
        ]
        if use_gpu:
            createdb_cmd_target.extend(["--gpu", "1"])
        result = subprocess.run(createdb_cmd_target)
        if result.returncode != 0:
            raise RuntimeError("Failed to create target database")

        # For GPU search, pad the target database (required for GPU prefilter)
        search_target_db = target_db
        if use_gpu:
            target_db_pad = Path(tmp_dir) / "target_db_pad"
            pad_cmd = [foldseek_bin, "makepaddedseqdb", str(target_db), str(target_db_pad)]
            result = subprocess.run(pad_cmd)
            if result.returncode != 0:
                raise RuntimeError("Failed to pad target database for GPU")
            search_target_db = target_db_pad

        # Run search
        print("Running search...")
        result_db = Path(tmp_dir) / "result"
        search_cmd = [
            foldseek_bin, "search",
            str(query_db), str(search_target_db), str(result_db), tmp_dir,
            "--threads", str(threads),
            "-e", "10",
            "--max-seqs", "100000"
        ]
        if use_gpu:
            search_cmd.extend(["--gpu", "1"])

        result = subprocess.run(search_cmd)
        if result.returncode != 0:
            raise RuntimeError("Search failed")

        # Convert results to TSV
        # ProstT5 can output: query, target, fident, alnlen, evalue, bits
        # But NOT alntmscore (no Cα coordinates)
        print("Converting results...")
        convert_cmd = [
            foldseek_bin, "convertalis",
            str(query_db), str(target_db), str(result_db), str(tsv_path),
            "--format-output", "query,target,fident,alnlen,evalue,bits"
        ]

        result = subprocess.run(convert_cmd)
        if result.returncode != 0:
            raise RuntimeError("Failed to convert results")

        # Read results
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                        names=['query', 'target', 'fident', 'alnlen', 'evalue', 'bits'],
                        low_memory=False)

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df):
    """
    Extract unique pairwise comparisons and average bidirectional scores.

    Uses fident (fraction identity) as the similarity metric since
    ProstT5 cannot output TM-scores.
    """
    print("Parsing results...")

    df = df.copy()

    # Extract IDs from file paths or use as-is
    df['q_id'] = df['query'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    df['t_id'] = df['target'].str.extract(r'/([^/]+)\.[^.]+$')[0]

    # Handle case where extraction failed
    mask_q = df['q_id'].isna()
    mask_t = df['t_id'].isna()
    if mask_q.any():
        df.loc[mask_q, 'q_id'] = df.loc[mask_q, 'query'].str.replace(r'\.[^.]+$', '', regex=True)
    if mask_t.any():
        df.loc[mask_t, 't_id'] = df.loc[mask_t, 'target'].str.replace(r'\.[^.]+$', '', regex=True)

    # Handle FASTA IDs with suffixes like _0, _1 from duplication
    df['q_id'] = df['q_id'].str.replace(r'_\d+$', '', regex=True)
    df['t_id'] = df['t_id'].str.replace(r'_\d+$', '', regex=True)

    # Filter out self-comparisons
    df = df[df['q_id'] != df['t_id']]

    print(f"Processing {len(df):,} non-self alignments...")

    # Create canonical pair keys (sorted alphabetically)
    df['seq1_id'] = df[['q_id', 't_id']].min(axis=1)
    df['seq2_id'] = df[['q_id', 't_id']].max(axis=1)

    # Group by unique pairs and aggregate
    # - Mean fident (average of both directions)
    # - Min e-value (best significance)
    print("Aggregating bidirectional scores...")
    result_df = df.groupby(['seq1_id', 'seq2_id']).agg(
        fident=('fident', 'mean'),
        alnlen=('alnlen', 'mean'),
        evalue=('evalue', 'min'),
        bits=('bits', 'max')
    ).reset_index()

    # Use fident as tm_score proxy for compatibility with existing plotting
    result_df['tm_score'] = result_df['fident']

    print(f"Extracted {len(result_df):,} unique pairs")
    return result_df.to_dict('records')


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    # Check for dataset argument
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    # Dataset configurations
    if is_scope40:
        fasta_file = "data/fasta/scope40-1000.fa"
        output = "results/scope40_prostt5_similarities.csv"
    else:
        # CATH dataset (default)
        fasta_file = "data/fasta/cath-domain-seqs-S100-1k.fa"
        output = "results/cath_prostt5_similarities.csv"

    foldseek_bin = "binaries/foldseek"
    prostt5_weights_dir = "weights"
    prostt5_weights = "weights/prostt5"
    threads = 32
    use_gpu = True

    print("=" * 80)
    print("ProstT5 Benchmark")
    print(f"Dataset: {'SCOPe40' if is_scope40 else 'CATH'}")
    print(f"FASTA file: {fasta_file}")
    print(f"Output: {output}")
    print(f"Threads: {threads}")
    print(f"GPU: {use_gpu}")
    print("=" * 80)

    # Verify paths exist
    if not Path(fasta_file).exists():
        raise ValueError(f"FASTA file not found: {fasta_file}")
    if not Path(foldseek_bin).exists():
        raise ValueError(f"Foldseek binary not found: {foldseek_bin}")

    # Download ProstT5 weights if needed
    if not Path(prostt5_weights).exists():
        print("ProstT5 weights not found, downloading...")
        download_prostt5_weights(foldseek_bin, prostt5_weights_dir)

    df = run_prostt5_search(fasta_file, foldseek_bin, prostt5_weights, threads, use_gpu)
    pairs = parse_results(df)
    save_results(pairs, output)

    print("=" * 80)
    print("Complete!")
    print("Note: tm_score column contains fident (fraction identity) as ProstT5 proxy")
    print("=" * 80)


if __name__ == "__main__":
    main()
