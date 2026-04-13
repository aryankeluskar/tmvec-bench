#!/usr/bin/env python
"""Generate a merged CATH comparison table."""

from pathlib import Path

import pandas as pd


RESDIR = Path("results")
REQUIRED_METHODS = ["tmvec1", "tmvec2", "tmvec2_student", "tmalign"]
OPTIONAL_METHODS = ["prostt5"]


def load_method(path, method, score_column="tm_score", clean_tmvec_ids=False):
    """Load one similarity CSV and normalize its pair key."""
    df = pd.read_csv(path).copy()

    if clean_tmvec_ids:
        for i in (1, 2):
            df[f"seq{i}_id"] = df[f"seq{i}_id"].str.split("/").str[0]
            df[f"seq{i}_id"] = df[f"seq{i}_id"].str.split("|").str[2]

    df["seq_pair"] = df.apply(
        lambda row: ",".join(sorted([row["seq1_id"], row["seq2_id"]])),
        axis=1,
    )
    df = df.drop(columns=["seq1_id", "seq2_id"])
    df = df.rename(columns={score_column: method}).set_index("seq_pair")
    return df


def main():
    tables = []

    for method in REQUIRED_METHODS:
        path = RESDIR / f"cath_{method}_similarities.csv"
        tables.append(load_method(path, method, clean_tmvec_ids=method.startswith("tmvec")))

    foldseek_path = RESDIR / "cath_foldseek_similarities.csv"
    tables.append(load_method(foldseek_path, "foldseek", score_column="evalue"))

    for method in OPTIONAL_METHODS:
        path = RESDIR / f"cath_{method}_similarities.csv"
        if not path.exists():
            continue
        tables.append(load_method(path, method))

    df = pd.concat(tables, axis=1)
    print(df.shape[0])

    truth = pd.read_table("truth.tsv")
    truth["seq_pair"] = truth["a"] + "," + truth["b"]
    truth = truth.set_index("seq_pair").drop(columns=["a", "b"])
    df = pd.concat([truth, df], axis=1)
    df.to_csv("results.tsv", sep="\t")


if __name__ == '__main__':
    main()
