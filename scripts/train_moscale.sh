#!/usr/bin/bash

#SBATCH -J MoScale_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_train_moscale.out

cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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


/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python train_moscale.py
