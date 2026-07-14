#!/usr/bin/bash
#SBATCH -J FlowDec_P1-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_FlowDecoder_P1-1.out


cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH

# ~/.netrc lives on the (node-local) /home disk, invisible to compute nodes -- wandb.init()
# fails there with "No API key configured". Feed the key in directly via env var instead.
# Key lives in run/.wandb_api_key (chmod 600, gitignored -- never commit this file).
export WANDB_API_KEY=$(cat /nas2/data/dpfla3573/code/MoScale/run/.wandb_api_key)

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python run/train_flow_decoder.py 
