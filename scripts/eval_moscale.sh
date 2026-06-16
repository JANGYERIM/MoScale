#!/usr/bin/bash

#SBATCH -J MoScale_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_eval_T_MS_V1.out

cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python eval_moscale.py
