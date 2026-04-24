#!/usr/bin/env python3
"""Plot the *time-per-10-tokens* curve from the 14-run suite benchmark output.

The benchmark prints one JSON object at the end. Save that object to a file (e.g. suite.json),
or pass a log file: we take the last line that parses as JSON containing "baseline_run_id".

Example:
  # After copying the final JSON from bench_run.log to suite.json:
  python project/plot_generation_speed_curve.py --json suite.json --out figures/benchmark/curves/speed_curve.png

  # Or (best-effort) parse last JSON object from a tee log:
  python project/plot_generation_speed_curve.py --json bench_run.log --out figures/benchmark/curves/speed_curve.png

We plot:
- x-axis: cumulative generated tokens (end of each 10-token chunk)
- y-axis: average wall-clock seconds per 10 generated tokens
  (generation_speed_curve.avg_elapsed_sec_per_chunk)
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_suite_payload(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "runs" in data and "baseline_run_id" in data:
            return data
    except json.JSONDecodeError:
        pass
    # Tee'd benchmark log: take the last line that is only `{` and parse from there to EOF
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() != "{":
            continue
        tail = "\n".join(lines[i:])
        try:
            data = json.loads(tail)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "runs" in data and "baseline_run_id" in data:
            return data
    # Fallback: any trailing block that starts with `{` and parses as suite
    for block in reversed(re.split(r"\n(?=\s*\{)", text)):
        block = block.strip()
        if not block.startswith("{"):
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "runs" in data and "baseline_run_id" in data:
            return data
    raise ValueError(
        f"Could not find a 14-run suite JSON in {path}. "
        "Save the final printed `{ ... \"runs\": ... }` block to a .json file and pass --json."
    )


def _curve_for_run(runs: List[Dict[str, Any]], run_id: int) -> Optional[Dict[str, Any]]:
    for r in runs:
        if int(r.get("run_id", -1)) == run_id:
            return r.get("generation_speed_curve")  # type: ignore[return-value]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True, help="Suite JSON file or tee log containing it")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figures/benchmark/curves/time_per_10_tokens_curve.png"),
        help="Output image path (parent dirs are created).",
    )
    ap.add_argument("--runs", default="1,2,3,4", help="Comma-separated run_id list")
    ap.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot cumulative sum of time-per-10-tokens (shows O(n^2) vs O(n) more directly).",
    )
    args = ap.parse_args()

    payload = _load_suite_payload(args.json)
    runs = payload["runs"]
    run_ids = [int(x.strip()) for x in args.runs.split(",") if x.strip()]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install in the same venv: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_key = "avg_elapsed_sec_per_chunk"

    for rid in run_ids:
        curve = _curve_for_run(runs, rid)
        if not curve or y_key not in curve:
            continue
        x = curve["chunk_end_token_index"]
        y = curve[y_key]
        if args.cumulative:
            # Cumulative time to generate up to token index x[k]
            y = list(__import__("numpy").cumsum([float(v) for v in y]).tolist())
        fused = next((r.get("use_fused_kernel") for r in runs if int(r.get("run_id", -1)) == rid), None)
        cache = next((r.get("use_cache") for r in runs if int(r.get("run_id", -1)) == rid), None)
        q = next((r.get("kv_cache_quantization") for r in runs if int(r.get("run_id", -1)) == rid), None)
        bud = next((r.get("kv_budget_limited") for r in runs if int(r.get("run_id", -1)) == rid), None)
        label = f"Run {rid} (fused={fused}, cache={cache}, q={q}, budget={bud})"
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, label=label)

    ax.set_xlabel("Cumulative generated tokens (end of each 10-token chunk)")
    if args.cumulative:
        ax.set_ylabel("Cumulative decode time (seconds)")
        ax.set_title("Cumulative decode time vs generated tokens (chunk size = 10)")
    else:
        ax.set_ylabel("Wall time per 10 generated tokens (seconds)")
        ax.set_title("Time per 10 tokens as decoding progresses (chunk size = 10)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()