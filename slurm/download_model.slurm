#!/bin/bash
# SBATCH -A trn040
# SBATCH -J get_model
# SBATCH -o ../logs/%x-%j.out
# SBATCH -e ../logs/%x-%j.err
# SBATCH -t 02:00:00
# SBATCH -N 1

module load cuda/12.2
source /gpfs/wolf2/olcf/trn040/world-shared/vjodo/envs/llama-ft/bin/activate
bash ../model/fetch_llama2_7b.sh