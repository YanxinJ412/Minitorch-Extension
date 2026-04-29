# llmsys_f25_hw4

Public repository for Assignment 4 of 11-868 LLM Systems.

This document summarizes **local extensions** (mixed-precision KV cache, fused decode attention CUDA, tests, and tooling) and how to **build, test, and run** them.

---

## What was added (summary)

### 1. Fused decode-time attention (CUDA)

- **`src/fused_decode_attn.cu`**: single-step autoregressive decode attention over the **full cached** K/V sequence. Supports **FP32**, **INT8** (dequant in kernel), and **INT4** (packed `uint8`, unpack + dequant in kernel). Exposes **host** wrappers that H2D Q/mask/K/V and run the fused kernel (K/V come from the NumPy-backed `LayerKVCache` via `fused_decode_buffers()`).
- **`compile_cuda.sh`**: builds shared objects under `minitorch/cuda_kernels/`:
  - `combine.so` (existing assignment kernel)
  - `fused_decode_attn.so` (new)

### 2. Python integration

- **`minitorch/cuda_kernel_ops.py`**: loads `fused_decode_attn.so`, exposes `fused_decode_attn_fw`, sets `supports_fused_decode_attn`.
- **`minitorch/kv_cache.py`**: mixed-precision KV storage (`none` / `int8` / `int4`), per-tensor scaling, optional byte budget (oldest-token eviction). Exposes **`LayerKVCache.fused_decode_buffers()`** for FP32 and INT8 so the fused kernel can read K/V directly. **`FUSED_DECODE_MAX_SEQ`** caps sequence length for the fused path (default 1024); beyond that, code falls back to the standard attention path.
- **`minitorch/modules_transfomer.py`**: when `use_fused_kernel=True`, model is in **eval**, backend is CUDA, KV cache is used, decode is **one query position**, and `fused_decode_buffers()` is available, attention uses the fused kernel; otherwise the original path. Softmax handling was aligned with full recomputation for correctness.

### 3. Evaluation and checkpointing

- **`project/run_autoregressive_benchmark.py`**: after optional training, runs a **fixed 14-configuration** generation suite (non-fused runs 1–7 and fused runs 8–14: cache on/off, INT8/INT4/none, optional KV byte budget). Tunables include **`--suite_kv_budget_bytes`**, **`--generation_speed_curve_chunk_size`**, and generation counts/lengths. **`--use_fused_kernel`** affects the checkpoint you **train** with; the suite loads weights into fused and non-fused models as needed when `n_epochs=0`.
- **`project/checkpointing.py`**: `validate_model_config` **ignores mismatch on `use_fused_kernel`** (so checkpoints trained without fused can still load).

### 4. Unit tests

- **`kernel_tests/test_fused_decode_attn.py`**: numpy reference vs fused CUDA for **FP32, INT8, and INT4**. Skips if `fused_decode_attn.so` or CUDA is unavailable.

---

## Prerequisites

- **GPU node** with NVIDIA driver and **`nvcc`** (CUDA toolkit) for building `.so` files.
- Python environment with assignment dependencies (see course `requirements` if any).

---

## How it works (principles)

### Fused decode attention

During autoregressive decoding, each step attends with a **single query position** (\(q\_len=1\)) over the cached prefix:
\[
\mathrm{softmax}(QK^\top/\sqrt{d} + \text{mask})V
\]

This repo adds a CUDA kernel (`src/fused_decode_attn.cu`) that fuses the **score**, **masked softmax**, and **weighted sum** into one kernel for \(q\_len=1\).

- **FP32**: K/V are float32.
- **INT8**: K/V are stored as `int8` + per-tensor `scale`; the kernel dequantizes on the fly.
- **INT4**: K/V are stored as packed `uint8` (2 values per byte) + `scale`; the kernel unpacks + dequantizes on the fly.

The fused path is used only when:
- `--use_fused_kernel=True`
- model is in `eval()`
- backend is CUDA
- KV cache is enabled and decode is **single token** (query length 1)
- cached length \(\le\) `FUSED_DECODE_MAX_SEQ`

### KV cache quantization (INT8 / INT4)

KV cache stores **all past keys/values**. With \(L\) layers, sequence length \(T\), and hidden dim \(d\), KV memory is \(O(L \cdot T \cdot d)\).

We reduce memory by storing K/V in lower precision:
- **INT8**: `q = clip(round(x/scale), -127, 127)` with `scale = max(max_abs/127, 1e-8)`
- **INT4**: same idea with range \([-7, 7]\), then pack 2 int4 values into one byte

During attention, values are dequantized and used as float.

---

## Build CUDA extensions

From the **`llmsys_hw4`** directory:

```bash
bash compile_cuda.sh
```

Expected outputs:

- `minitorch/cuda_kernels/combine.so`
- `minitorch/cuda_kernels/fused_decode_attn.so`

---

## Run fused kernel unit tests

Still from **`llmsys_hw4`**:

```bash
pytest kernel_tests/test_fused_decode_attn.py -v
```

Both FP32 and INT8 tests should pass on a machine where the `.so` exists and `libcuda` is available.

---

## Run benchmarks (14-run generation suite)

All commands below assume:

```bash
cd /jet/home/yjiang23/11868/llmsys_hw4
export PYTHONPATH=$PWD
```

One invocation loads weights (after optional training), then evaluates **14** greedy-decoding configurations and prints **one JSON object** at the end. Baseline is **run 1** (full recompute, no KV cache). Runs **2–7** use the non-fused model; **8–14** use the fused decode path. Quantization and byte-budget flags are **fixed inside the suite** (see `benchmark_autoregressive_generation` in `project/run_autoregressive_benchmark.py`).

**Example (no training, load checkpoint, small prompt set):**

```bash
cd llmsys_hw4

python project/run_autoregressive_benchmark.py \
  --model_max_length=128 \
  --n_epochs=0 \
  --load_weights_path=./workdir_autoregressive_128/model_weights.npz \
  --generation_max_new_tokens=120 \
  --generation_examples=2 \
  --suite_kv_budget_bytes=131072 \
  --generation_speed_curve_chunk_size=10
```

This matches the spirit of `run_all_benchmarks.sbatch` (adjust paths and `generation_*` as needed).

### How to read the benchmark JSON

Top-level keys include:

- **`baseline_run_id`**: always `1` for this suite.
- **`baseline_avg_tokens_per_sec`**: throughput for run 1.
- **`suite_kv_budget_bytes`**: byte cap used when a run has `kv_budget_limited: true`.
- **`generation_speed_curve_chunk_size`**: chunk size for wall-clock speed curves.
- **`runs`**: list of 14 objects, each with:
  - **`run_id`**, **`use_fused_kernel`**, **`use_cache`**, **`kv_cache_quantization`**, **`kv_budget_limited`**
  - **`avg_tokens_per_sec`**, **`avg_kv_cache_bytes`**
  - **`vs_baseline_run1_token_match_rate`**, **`vs_baseline_run1_speedup`**
  - **`generation_speed_curve`**: aggregated curve for that mode

Notes:

- Adjust **`--load_weights_path`**, **`--model_max_length`**, and tokenizer/model shape flags so they match **`model_config.json`** next to the weights.
- There is **no** per-invocation `--kv_cache_quantization` or task-aware CLI; the suite sweeps modes internally.
- Under a **fixed KV byte budget**, token match vs run 1 can drop; that is expected, not a fused-kernel bug.

---

## Expected behavior (sanity check)

- **`vs_baseline_run1_token_match_rate`**: for **FP32 KV without budget** (runs 2 and 9), greedy outputs in this harness match run&nbsp;1 at **1.0** on the logged averages. **INT8/INT4** can fall below 1.0 depending on implementation details and prompts (see `bench_run_1.log` / `figures/benchmark/suite14/bench_run_1_gen5_suite.json`).
- **Throughput**: fused decode often improves tokens/sec on decode-heavy settings; exact numbers depend on GPU and load.

---

## INT4 and fused decode

INT4 uses **packed** `uint8` K/V; the fused **`fused_decode_attn_host_int4`** path unpacks and dequantizes inside the kernel. The NumPy **`LayerKVCache`** stores the same packed layout as `kv_cache.py`.

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `fused_decode_attn.so` missing | Run `bash compile_cuda.sh` on a node with `nvcc`. |
| Kernel tests skipped | Run on GPU node; ensure `minitorch/cuda_kernels/fused_decode_attn.so` exists. |
| Config error when loading checkpoint | Non-fused vs fused should be OK: `use_fused_kernel` is ignored for mismatch. Other keys (e.g. `n_layer`, `n_embd`) must match `model_config.json`. |
| Fused not used | Fused applies in **eval**, **CUDA** backend, **incremental cache**, **single decode step**, sequence length ≤ **`FUSED_DECODE_MAX_SEQ`**, and valid fused buffers. |

---

## Figures

Outputs go under **`figures/benchmark/`** by type (see **`figures/README.md`**). Omitting `--outdir` on the plot scripts uses these defaults when you run from **`llmsys_hw4/`**.

| Subfolder | Contents | Produced by |
|-----------|-----------|-------------|
| **`figures/benchmark/seven_mode/`** | Full / FP / int8 / … bar charts (`*_tps.png`, `*_kv.png`, `*_match.png`, `*_mem.png`, tables) | `project/plot_autoregressive_benchmark_results.py` |
| **`figures/benchmark/suite14/`** | All 14 runs: throughput+speedup panel, KV+match panel, extracted `*_suite.json`, compare overlays | `project/plot_suite_benchmark_results.py` |
| **`figures/benchmark/curves/`** | Per-chunk / per-token speed vs token index | `project/plot_generation_speed_curve.py` |
| **`figures/benchmark/legacy/`** | Old seven-mode **sample** JSON plots (`sample_results_autoregressive_*`) | same script with no `--json` |

### Seven-mode charts from a suite `.log`

```bash
cd llmsys_hw4
.venv/bin/python project/plot_autoregressive_benchmark_results.py --json bench_run_ex_5.log
.venv/bin/python project/plot_autoregressive_benchmark_results.py --json bench_run_examples_2.log
```

Writes to `figures/benchmark/seven_mode/` (e.g. `bench_run_ex_5_tps.png`, `bench_run_ex_5_kv.png`, …). Override with `--outdir path/to/dir`. A symlink `bench_run_example_2.log` → `bench_run_examples_2.log` exists for the singular log name.

### Full 14-run panels

```bash
.venv/bin/python project/plot_suite_benchmark_results.py --input bench_run_ex_5.log --stem bench_run_ex_5
```

Default output directory is `figures/benchmark/suite14/`.

### Speed curves

```bash
.venv/bin/python project/plot_generation_speed_curve.py --json bench_run_1.log --runs 1,2,9,14 --y per_token
```

Default `--out` is `figures/benchmark/curves/generation_speed_curve.png`.

Legacy **JSON** files that already use keys `full_recompute`, `kv_cache`, … still work with `--json path.json`.
