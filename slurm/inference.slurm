#!/bin/bash
# SBATCH -A trn040
# SBATCH -J llama-gen
# SBATCH -o ../logs/%x-%j.out
# SBATCH -e ../logs/%x-%j.err
# SBATCH -t 00:30:00
# SBATCH -N 1
# SBATCH --gpus-per-node=1

module load cuda/12.2
source /gpfs/wolf2/olcf/trn040/world-shared/vjodo/envs/llama-ft/bin/activate
python ../scripts/generate_before_after.py \
  --model_dir /gpfs/wolf2/olcf/trn040/world-shared/vjodo/models/Llama-2-7b-hf \
  --ft_dir /gpfs/wolf2/olcf/trn040/world-shared/vjodo/checkpoints/llama2_alpaca \
  --out_file ../examples/before_after.txt