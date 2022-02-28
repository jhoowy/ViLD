#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=$3
RANK=$4
ADDR=$5
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --nnodes=$NNODES --node_rank=$RANK --master_addr=$ADDR $(dirname "$0")/train.py $CONFIG --launcher pytorch
