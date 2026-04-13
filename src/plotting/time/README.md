Benchmarking of runtimes
------

Place the raw results in the `results` folder.

Supported inputs:
- Legacy top-level CSVs such as `results/tmvec2_query_times.csv`
- Per-run directories under `results/time_benchmarks/` that contain `benchmark_config.json`, `encoding_times.csv`, and `query_times.csv`

Current Foldseek runtime naming follows the April 6, 2026 benchmark decisions:
- `foldseek_cpu_default`
- `foldseek_gpu_default`
- `foldseek_gpu_exhaustive`

`merge_tables.py` will discover both styles automatically and annotate the merged tables with `method` and `source`.

Execute `merge_tables.py` to combine the results into one table file `results.tsv`.

Execute `plot.ipynb` to analyze the results and generate plots.
