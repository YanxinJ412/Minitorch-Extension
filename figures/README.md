# Figures layout

Generated plots and tables for HW4 live under **`benchmark/`** so the repo root stays small and names stay grouped by purpose.

```
figures/
  README.md                 ← this file
  benchmark/
    seven_mode/             ← 7 “modes” (Full, FP, int8, …): *_tps.png, *_kv.png, *_match.png, *_mem.png, *.csv/.md
    suite14/                ← full 14-run panels, suite JSON extracts, run-vs-run speedup overlays
    curves/                 ← generation speed vs token index (plot_generation_speed_curve.py)
    legacy/                 ← sample_results_autoregressive_* from bundled example JSON (no --json)
```

**Defaults** (when you `cd llmsys_hw4` and run the plotting scripts without `--outdir` / default `--out`):

| Script | Default output |
|--------|----------------|
| `project/plot_autoregressive_benchmark_results.py` | `benchmark/seven_mode/` (or `benchmark/legacy/` if `--json` is omitted) |
| `project/plot_suite_benchmark_results.py` | `benchmark/suite14/` |
| `project/plot_generation_speed_curve.py` | `benchmark/curves/generation_speed_curve.png` |

You can still pass `--outdir` / `--out` to write anywhere (e.g. a scratch folder for one-off plots).
