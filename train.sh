#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

NODE_RANK=0
RANK0_IP='10.128.15.246'
RANK0_PORT=4532

if [ ${NODE_RANK} -neq 0 ]; then
	rsync -az ${RANK0_IP}:/data/output/ /data/output
fi

torchrun \
	--nproc_per_node 8 \
	--nnodes 4 \
	--node_rank ${NODE_RANK} \
	--rdzv_id ${RANK0_PORT} \
	--rdzv_backend c10d \
	--rdzv_endpoint ${RANK0_IP}:${RANK0_PORT} \
	run_mae_pretraining.py
