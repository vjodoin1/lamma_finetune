#!/bin/bash
# SBATCH -A trn040
# SBATCH -J llama-ft
# SBATCH -o ../logs/%x-%j.out
# SBATCH -e ../logs/%x-%j.err
# SBATCH -t 10:00:00
# SBATCH -N 1
# SBATCH --gpus-per-node=4
# SBATCH --cpus-per-task=32
# SBATCH --mem=0

module load cuda/12.2
source /gpfs/wolf2/olcf/trn040/world-shared/vjodo/envs/llama-ft/bin/activate
python ../scripts/finetune.py \
  --model_dir /gpfs/wolf2/olcf/trn040/world-shared/vjodo/models/Llama-2-7b-hf \
  --data_dir /gpfs/wolf2/olcf/trn040/world-shared/vjodo/data/alpaca \
  --output_dir /gpfs/wolf2/olcf/trn040/world-shared/vjodo/checkpoints/llama2_alpaca \
  --batch_size 8 --micro_batch_size 1 --num_epochs 3 --lr 2e-5 \
  --lora_r 64 --lora_alpha 16 --lora_dropout 0.05