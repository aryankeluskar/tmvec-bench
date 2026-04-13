# %% [markdown]
# # Computational efficiency

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
from matplotlib.ticker import ScalarFormatter
from merge_tables import main as merge_tables_main

# %%
plt.rcParams['svg.fonttype'] = 'none'

# %%
SCRIPT_DIR = Path(__file__).resolve().parent
ENCODING_TABLE_PATH = SCRIPT_DIR / 'encoding.tsv'
QUERY_TABLE_PATH = SCRIPT_DIR / 'query.tsv'
PLOTS_DIR = SCRIPT_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

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


merge_tables_main()

# %% [markdown]
# Encoding

# %%
df = normalize_methods(pd.read_table(ENCODING_TABLE_PATH))
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
df = normalize_methods(pd.read_table(QUERY_TABLE_PATH))
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


