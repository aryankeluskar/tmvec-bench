#!/usr/bin/env python
"""Generate merged runtime tables from top-level and per-run benchmark outputs."""

from pathlib import Path
import json

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results"
TIME_BENCHMARKS_DIR = RESULTS_DIR / "time_benchmarks"
ENCODING_TABLE_PATH = SCRIPT_DIR / "encoding.tsv"
QUERY_TABLE_PATH = SCRIPT_DIR / "query.tsv"
KNOWN_TOP_LEVEL_METHODS = (
    "tmvec1",
    "tmvec2",
    "tmvec2_student",
    "diamond",
    "foldseek",
    "foldseek_gpu",
    "prostt5_gpu",
)


def load_runtime_pair(method, encoding_path, query_path, source):
    """Load one encoding/query pair into the common output schema."""
    encoding = pd.read_csv(encoding_path)[["encoding_size", "mean_seconds"]].copy()
    encoding["method"] = method
    encoding["source"] = source

    query_df = pd.read_csv(query_path).copy()
    query_value_column = "total_mean" if "total_mean" in query_df.columns else "search_mean"
    query = query_df[["query_size", "database_size", query_value_column]].rename(
        columns={query_value_column: "total_mean"}
    )
    query["method"] = method
    query["source"] = source
    return encoding, query


def method_name_from_config(config_path, default_label=None):
    """Infer a stable method label from a benchmark config file."""
    config = json.loads(config_path.read_text())
    comparison_mode = config.get("comparison_mode")
    if comparison_mode:
        return comparison_mode

    if default_label is not None:
        return default_label

    dirname = config_path.parent.name
    if "prostt5" in dirname or "prostt5_weights" in config:
        return f"prostt5_{'gpu' if config.get('use_gpu') else 'cpu'}"
    if "foldseek" in dirname or "foldseek_binary" in config:
        if "exhaustive" in dirname:
            return "foldseek_gpu_exhaustive" if config.get("use_gpu") else "foldseek_cpu_exhaustive"
        return "foldseek_gpu" if config.get("use_gpu") else "foldseek"
    return dirname


def discover_top_level_runs():
    """Yield benchmark result pairs stored directly under results/."""
    seen_methods = set()

    for config_path in sorted(RESULTS_DIR.glob("*_benchmark_config.json")):
        suffix = "_benchmark_config.json"
        method = config_path.name[:-len(suffix)] if config_path.name.endswith(suffix) else config_path.stem
        encoding_path = RESULTS_DIR / f"{method}_encoding_times.csv"
        query_path = RESULTS_DIR / f"{method}_query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        seen_methods.add(method)
        yield (
            method_name_from_config(config_path, default_label=method),
            encoding_path,
            query_path,
            str(config_path),
        )

    for method in KNOWN_TOP_LEVEL_METHODS:
        if method in seen_methods:
            continue
        encoding_path = RESULTS_DIR / f"{method}_encoding_times.csv"
        query_path = RESULTS_DIR / f"{method}_query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        yield (method, encoding_path, query_path, str(RESULTS_DIR))


def discover_runtime_runs():
    """Yield benchmark result pairs from results/time_benchmarks."""
    if not TIME_BENCHMARKS_DIR.exists():
        return

    for config_path in sorted(TIME_BENCHMARKS_DIR.glob("*/benchmark_config.json")):
        encoding_path = config_path.parent / "encoding_times.csv"
        query_path = config_path.parent / "query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        yield (
            method_name_from_config(config_path),
            encoding_path,
            query_path,
            str(config_path.parent),
        )


def main():
    encoding_tables = []
    query_tables = []

    for method, encoding_path, query_path, source in discover_top_level_runs():
        encoding, query = load_runtime_pair(method, encoding_path, query_path, source)
        encoding_tables.append(encoding)
        query_tables.append(query)

    for method, encoding_path, query_path, source in discover_runtime_runs():
        encoding, query = load_runtime_pair(method, encoding_path, query_path, source)
        encoding_tables.append(encoding)
        query_tables.append(query)

    if not encoding_tables or not query_tables:
        raise FileNotFoundError("No runtime benchmark outputs were found under results/")

    pd.concat(encoding_tables, axis=0, ignore_index=True).to_csv(
        ENCODING_TABLE_PATH,
        sep="\t",
        index=False,
    )
    pd.concat(query_tables, axis=0, ignore_index=True).to_csv(
        QUERY_TABLE_PATH,
        sep="\t",
        index=False,
    )


if __name__ == '__main__':
    main()
