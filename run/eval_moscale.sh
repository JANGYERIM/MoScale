#!/usr/bin/bash
#SBATCH -J Eval_SeCo_P1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_Eval_SeCo_P1-3.out


cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python run/eval_moscale.py
