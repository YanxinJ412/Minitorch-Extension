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
python project/run_machine_translation.py   --model_max_length=256   --batch_size=16   --samples_per_epoch=4000   --n_epochs=1   --use_fused_kernel=True   --save_weights_path=./workdir_vocab10000_lr0.002_embd256_large/model_weights.npz