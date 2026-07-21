#!/usr/bin/bash
#SBATCH -J MS_Baseline_hrvqvae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_Baseline_hrvqvae.out


cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH

# Python's tempfile/multiprocessing falls back to cwd (this NFS dir) when node-local
# /tmp isn't usable, leaving pymp-* junk behind and breaking DataLoader worker IPC
# (AF_UNIX sockets don't work reliably over NFS). Force a node-local tmp dir instead.
export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

# ~/.netrc lives on the (node-local) /home disk, invisible to compute nodes -- wandb.init()
# fails there with "No API key configured". Feed the key in directly via env var instead.
# Key lives in run/.wandb_api_key (chmod 600, gitignored -- never commit this file).
export WANDB_API_KEY=$(cat /nas2/data/dpfla3573/code/MoScale/run/.wandb_api_key)

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python run/train_hrvqvae.py
