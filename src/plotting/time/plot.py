# %% [markdown]
# # Computational efficiency

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from matplotlib.ticker import ScalarFormatter

# %%
plt.rcParams['svg.fonttype'] = 'none'

# %%
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results"
TIME_BENCHMARKS_DIR = RESULTS_DIR / "time_benchmarks"
PLOTS_DIR = SCRIPT_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

KNOWN_TOP_LEVEL_METHODS = (
    "tmvec1",
    "tmvec2",
    "tmvec2_student",
    "diamond",
    "foldseek",
    "foldseek_gpu",
    "prostt5_gpu",
)

METHOD_ALIASES = {
    'foldseek_legacy': 'foldseek',
    'foldseek_cpu_default': 'foldseek',
    'foldseek_gpu_default': 'foldseek_gpu',
}
METHOD_LABELS = {
    'tmvec1': 'TM-Vec',
    'tmvec2': 'TM-Vec 2',
    'tmvec2_student': 'TM-Vec 2s',
    'foldseek': 'Foldseek',
    'foldseek_gpu': 'Foldseek GPU',
    'prostt5_gpu': 'Foldseek PLM',
}
PREFERRED_METHODS = [
    'tmvec1',
    'tmvec2',
    'tmvec2_student',
    'foldseek',
    'foldseek_gpu',
    'prostt5_gpu',
]


def load_runtime_pair(method, encoding_path, query_path, source):
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
    seen_methods = set()
    for config_path in sorted(RESULTS_DIR.glob("*_benchmark_config.json")):
        suffix = "_benchmark_config.json"
        method = config_path.name[:-len(suffix)] if config_path.name.endswith(suffix) else config_path.stem
        encoding_path = RESULTS_DIR / f"{method}_encoding_times.csv"
        query_path = RESULTS_DIR / f"{method}_query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        seen_methods.add(method)
        yield method_name_from_config(config_path, default_label=method), encoding_path, query_path, str(config_path)
    for method in KNOWN_TOP_LEVEL_METHODS:
        if method in seen_methods:
            continue
        encoding_path = RESULTS_DIR / f"{method}_encoding_times.csv"
        query_path = RESULTS_DIR / f"{method}_query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        yield method, encoding_path, query_path, str(RESULTS_DIR)


def discover_runtime_runs():
    if not TIME_BENCHMARKS_DIR.exists():
        return
    for config_path in sorted(TIME_BENCHMARKS_DIR.glob("*/benchmark_config.json")):
        encoding_path = config_path.parent / "encoding_times.csv"
        query_path = config_path.parent / "query_times.csv"
        if not encoding_path.exists() or not query_path.exists():
            continue
        yield method_name_from_config(config_path), encoding_path, query_path, str(config_path.parent)


def normalize_methods(df):
    normalized = df.copy()
    normalized['method'] = normalized['method'].replace(METHOD_ALIASES)
    return normalized


def available_methods(df):
    present = set(df['method'].unique())
    methods = [method for method in PREFERRED_METHODS if method in present]
    if not methods:
        raise ValueError(f'No supported methods found in merged tables: {sorted(present)}')
    return methods


# %%
encoding_tables, query_tables = [], []
for method, enc_path, qry_path, source in [*discover_top_level_runs(), *discover_runtime_runs()]:
    enc, qry = load_runtime_pair(method, enc_path, qry_path, source)
    encoding_tables.append(enc)
    query_tables.append(qry)

if not encoding_tables or not query_tables:
    raise FileNotFoundError("No runtime benchmark outputs were found under results/")

encoding_df = pd.concat(encoding_tables, axis=0, ignore_index=True)
query_df = pd.concat(query_tables, axis=0, ignore_index=True)

# %% [markdown]
# Encoding

# %%
df = normalize_methods(encoding_df)
df.head()

# %%
sizes = sorted(df['encoding_size'].unique())
sizes

# %%
plt.figure(figsize=(5, 4))
methods = available_methods(df)
names = [METHOD_LABELS[method] for method in methods]
for method, name in zip(methods, names):
    df_ = df.query(f'method == "{method}"')
    plt.plot('encoding_size', 'mean_seconds', data=df_, marker='o', label=name)
plt.legend(title='Method')
plt.xscale('log')
plt.yscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel('Number of sequences (log)')
plt.ylabel('Runtime (sec) (log)')
plt.title('Encoding')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'encoding.svg')

# %%
df['seqs_per_second'] = df['encoding_size'] / df['mean_seconds']

# %%
s_max = df.groupby('method')['seqs_per_second'].max().loc[methods]
s_max

# %%
plt.figure(figsize=(4, 4))
positions = np.arange(len(methods))
plt.bar(x=positions, height=s_max.to_numpy())
for i, method in enumerate(methods):
    plt.text(i, s_max[method], '{:.4g}'.format(s_max[method]), ha='center')
plt.yscale('log')
plt.ylim(top=plt.ylim()[1] * 2)
plt.xlabel('Method')
plt.xticks(positions, names)
plt.ylabel('Max. sequences per second')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.title('Encoding')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'maxseqs.svg')

# %% [markdown]
# Query

# %%
df = normalize_methods(query_df)
df.head()

# %% [markdown]
# Benchmark results of BLAST and DIAMOND were adopted from Figure S7 of the TM-Vec 1 paper.
#
# - https://www.nature.com/articles/s41587-023-01917-2

# %%
blast = pd.DataFrame([
    [10, 1000, 9],
    [100, 1000, 16],
    [1000, 1000, 65],
    [10, 10000, 10],
    [100, 10000, 25],
    [1000, 10000, 131],
    [10, 100000, 14],
    [100, 100000, 63],
    [1000, 100000, 175]
], columns=['query_size', 'database_size', 'total_mean']).assign(method='blast')

# %%
diamond = pd.DataFrame([
    [10, 1000, 0.521],
    [100, 1000, 0.512],
    [1000, 1000, 0.597],
    [10, 10000, 0.573],
    [100, 10000, 0.644],
    [1000, 10000, 0.775],
    [10, 100000, 1.234],
    [100, 100000, 1.395],
    [1000, 100000, 1.743]
], columns=['query_size', 'database_size', 'total_mean']).assign(method='diamond')

# %%
if 'blast' not in set(df['method']):
    df = pd.concat([df, blast], ignore_index=True)
if 'diamond' not in set(df['method']):
    df = pd.concat([df, diamond], ignore_index=True)

# %%
db_sizes = sorted(df['database_size'].unique())
db_sizes

# %%
methods = available_methods(df)
names = [METHOD_LABELS[method] for method in methods]

# %%
fig, axes = plt.subplots(1, len(db_sizes), sharey=True, figsize=(2.8 * len(db_sizes), 4))
axes = np.atleast_1d(axes)
for i, size in enumerate(db_sizes):
    ax = axes[i]
    for method, name in zip(methods, names):
        df_ = df.query(f'database_size == {size} & method == "{method}"')
        ax.plot('query_size', 'total_mean', data=df_, marker='o', label=name)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if i == 0:
        ax.set_ylabel('Runtime (sec) (log)')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
    if i == 1:
        ax.set_xlabel('Number of query sequences (log)')
    ax.set_title(f'Database size: {size / 1000:g}k')
    if i == len(db_sizes) - 1:
        ax.legend(title='Method', loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'query.svg')

# %%
