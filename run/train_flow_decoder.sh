# #!/usr/bin/bash
# #SBATCH -J FlowDec_P1-1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-gpu=8
# #SBATCH --mem-per-gpu=29G
# #SBATCH -p batch_grad
# #SBATCH -w ariel-v3
# #SBATCH -t 4-0
# #SBATCH -o /nas2/data/dpfla3573/code/MoScale/logs/slurm-%A_FlowDecoder_P1-1.out
#
#
# cd /nas2/data/dpfla3573/code/MoScale
# export PYTHONPATH=/nas2/data/dpfla3573/code/MoScale:$PYTHONPATH
#
# # ~/.netrc lives on the (node-local) /home disk, invisible to compute nodes -- wandb.init()
# # fails there with "No API key configured". Feed the key in directly via env var instead.
# # Key lives in run/.wandb_api_key (chmod 600, gitignored -- never commit this file).
# export WANDB_API_KEY=$(cat /nas2/data/dpfla3573/code/MoScale/run/.wandb_api_key)
#
# /nas2/data/dpfla3573/anaconda3/envs/moscale/bin/python run/train_flow_decoder.py

#!/usr/bin/bash

cd /home/data/yerim/code/MoScale
export PYTHONPATH=/home/data/yerim/code/MoScale:$PYTHONPATH
export TZ='KST-9'

# ~/.netrc가 없는 서버라 wandb.init()이 "No API key configured"로 실패함.
# 키를 env var로 직접 넘겨줌. 키는 run/.wandb_api_key에 있음 (chmod 600, gitignore 대상 -- 절대 커밋 금지)
export WANDB_API_KEY=$(cat /home/data/yerim/code/MoScale/run/.wandb_api_key)

# 0~1번 GPU 중, 사용 중인 메모리가 가장 적은 GPU를 자동으로 선택
GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$1 <= 1 {print $1, $2}' \
  | sort -k2 -n \
  | head -1 \
  | awk '{print $1}')

# echo "선택된 GPU: $GPUS"

PY_ARGS=${@:1}

mkdir -p logs
LOGFILE=logs/train_flow_decoder1_2_$(date +%y%m%d_%H%M).log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
  /root/anaconda3/envs/moscale/bin/python run/train_flow_decoder.py ${PY_ARGS} --seed 444 2>&1 | tee $LOGFILE
