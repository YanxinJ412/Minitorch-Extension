#!/bin/bash
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -J mt_fused
#SBATCH -o mt_fused_%j.out
#SBATCH -e mt_fused_%j.err

module load cuda/12.4

cd /jet/home/yjiang23/11868/llmsys_hw4

source .venv/bin/activate
export PYTHONPATH=$PWD

echo "=== Fused run ==="
python project/run_machine_translation.py --use-fused-kernel True