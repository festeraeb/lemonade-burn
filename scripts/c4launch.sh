#!/bin/bash
# usage: c4launch.sh <port> <model-file> <uuid-list> <logname> [extra env KEY=VAL ...]
port=$1; model=$2; cards=$3; log=$4; shift 4
D=/mnt/t440/raid0/engines/lemonade-burn/${LB_BUILD:-20261116-qwen35-c4-b45ed5a}
mkdir -p ~/lbtest
cd ~/lbtest
env LD_LIBRARY_PATH=$D/lib CUDA_PATH=$D/cuda-path CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$cards CUDARC_CUDA_VERSION=12080 LB_QW_TRACE=1 "$@" \
  nohup $D/bin/lemonade-burn-server --model /mnt/t440/raid0/models/$model --gpu --ctx ${CTX:-8192} --host 127.0.0.1 --port $port --model-id test > ~/lbtest/$log.log 2>&1 &
echo $!
