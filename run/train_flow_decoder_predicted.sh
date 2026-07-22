#!/usr/bin/bash

cd /home/data/yerim/code/MoScale
export PYTHONPATH=/home/data/yerim/code/MoScale:$PYTHONPATH
export TZ='KST-9'

# 0~1번 GPU 중, 사용 중인 메모리가 가장 적은 GPU를 자동으로 선택
# GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
#   | awk -F', ' '$1 <= 1 {print $1, $2}' \
#   | sort -k2 -n \
#   | head -1 \
#   | awk '{print $1}')

GPUS=1

# train_flow_decoder.yaml의 exp.name과 일치해야 함 (checkpoint_dir/humanml3d/flow_decoder/<name>/)
FLOW_DECODER_NAME=${FLOW_DECODER_NAME:-FLOWDEC_PREDICTED3}
MODEL_DIR="./checkpoint_dir/humanml3d/flow_decoder/$FLOW_DECODER_NAME/model"

# flow_decoder_trainer.py가 저장하는 세 체크포인트를 순회 평가.
# 특정 하나만 보고 싶으면 CKPTS="net_best_fid.tar" bash run/eval_flow_decoder.sh
IFS=' ' read -r -a CKPTS <<< "${CKPTS:-net_best_fid.tar}"

PY_ARGS=${@:1}

mkdir -p logs
LOGFILE=logs/train_flow_decoder_predict_$(date +%y%m%d_%H%M).log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
  /root/anaconda3/envs/moscale/bin/python run/train_flow_decoder_predicted.py ${PY_ARGS} --seed 444 2>&1 | tee $LOGFILE
