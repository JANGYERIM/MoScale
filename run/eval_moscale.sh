#!/usr/bin/bash

cd /home/data/yerim/code/MoScale
export PYTHONPATH=/home/data/yerim/code/MoScale:$PYTHONPATH
export TZ='KST-9'

GPUS=1

# config/eval_moscale.yaml의 moscale_name / which_epoch 체크포인트를, 디코더 교체 없이
# HRVQVAE 자체 디코더로 평가 (flow_decoder_name은 yaml에서 null로 둔 채 그대로 둠).
# 디코더를 flow decoder로 교체해서 평가하려면 run/eval_flow_decoder.sh를 사용.

PY_ARGS=${@:1}

mkdir -p logs
LOGFILE=logs/eval_moscale_$(date +%y%m%d_%H%M).log

CUDA_VISIBLE_DEVICES=$GPUS /root/anaconda3/envs/moscale/bin/python run/eval_moscale.py \
  ${PY_ARGS} 2>&1 | tee "$LOGFILE"
