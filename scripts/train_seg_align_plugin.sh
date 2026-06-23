#!/usr/bin/bash

#SBATCH -J SegAlignPlugin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_seg_align_plugin_v3-1.out

cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH
export WANDB_API_KEY=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
export TOKENIZERS_PARALLELISM=false

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python train_seg_align_plugin.py
