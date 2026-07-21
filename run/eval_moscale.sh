#!/usr/bin/bash
#SBATCH -J Eval_SeCo_P1-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_Eval_SeCo_P1-2.out


cd /nas2/data/dpfla3573/code/MoScale
export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH

# Python's tempfile/multiprocessing falls back to cwd (this NFS dir) when node-local
# /tmp isn't usable, leaving pymp-* junk behind and breaking DataLoader worker IPC
# (AF_UNIX sockets don't work reliably over NFS). Force a node-local tmp dir instead.
export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

/nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python run/eval_moscale.py
