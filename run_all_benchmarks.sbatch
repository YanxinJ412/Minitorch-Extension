#!/bin/bash
#SBATCH -J llmsys_bench_all
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
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

mkdir -p logs

.venv/bin/python project/run_autoregressive_benchmark.py \
  --model_max_length=128 --n_epochs=0 \
  --load_weights_path=./workdir_autoregressive_128/model_weights.npz \
  --generation_max_new_tokens=120 --generation_examples=2 \
  > logs/bench_run_examples=2.log 2>&1