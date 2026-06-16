#!/usr/bin/bash

#SBATCH -J SegExtract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 0-4
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_extract_features.out

cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python scripts/extract_seg_features.py
