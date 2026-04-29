#!/bin/bash
#SBATCH -J llmsys_suite_bench
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

set -euo pipefail

cd /jet/home/yjiang23/11868/llmsys_hw4

# Best-effort CUDA module (safe if not available)
module load cuda/12.4 >/dev/null 2>&1 || true

source .venv/bin/activate
export PYTHONPATH=$PWD

echo "Running on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

# Edit these if your checkpoint path differs
WEIGHTS_PATH="${WEIGHTS_PATH:-./workdir_autoregressive_128/model_weights.npz}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-128}"
GEN_NEW_TOKENS="${GEN_NEW_TOKENS:-120}"
GEN_EXAMPLES="${GEN_EXAMPLES:-5}"
SUITE_KV_BUDGET_BYTES="${SUITE_KV_BUDGET_BYTES:-131072}"
SPEED_CURVE_CHUNK_SIZE="${SPEED_CURVE_CHUNK_SIZE:-10}"

mkdir -p logs
OUT_LOG="logs/bench_suite_job${SLURM_JOB_ID}_ex${GEN_EXAMPLES}.log"
echo "Writing benchmark log to ${OUT_LOG}"

.venv/bin/python project/run_autoregressive_benchmark.py \
  --model_max_length="${MODEL_MAX_LENGTH}" \
  --n_epochs=0 \
  --load_weights_path="${WEIGHTS_PATH}" \
  --generation_max_new_tokens="${GEN_NEW_TOKENS}" \
  --generation_examples="${GEN_EXAMPLES}" \
  --suite_kv_budget_bytes="${SUITE_KV_BUDGET_BYTES}" \
  --generation_speed_curve_chunk_size="${SPEED_CURVE_CHUNK_SIZE}" \
  > "${OUT_LOG}" 2>&1

echo "Done. Log: ${OUT_LOG}"

