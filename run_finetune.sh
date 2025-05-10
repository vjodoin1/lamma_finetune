#!/bin/bash
#SBATCH --job-name=llama3-ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=ai-tenn                  
#SBATCH --qos=ai-tenn  
#SBATCH --output=logs/llama3-ft-%j.out
#SBATCH --error=logs/llama3-ft-%j.err

# download model + data 
python download_model_dataset.py

# run fine-tuning
python finetune.py