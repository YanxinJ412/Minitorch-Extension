#!/bin/bash
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -J mt_fused
#SBATCH -o mt_fused_%j.out
#SBATCH -e mt_fused_%j.err

module load cuda/12.4

# cd Task-Aware-Mix-Precision-KV-Cache

source .venv/bin/activate
export PYTHONPATH=$PWD

echo "=== Fused run ==="
HF_HOME="$PWD/hf_cache" \
HF_DATASETS_CACHE="$PWD/hf_cache/datasets" \
TRANSFORMERS_CACHE="$PWD/hf_cache/transformers" \
python project/run_autoregressive_benchmark.py \
  --model_max_length=128 \
  --n_epochs=1 \
  --use_fused_kernel=False \
  --batch_size=16 \
  --save_weights_path=./workdir_autoregressive_128/model_weights.npz

