#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

RANK0_IP='10.128.15.246'
RANK0_PORT=4532

rsync -az ${RANK0_IP}:/data/output/ /data/output

torchrun \
	--nproc_per_node 8 \
	--nnodes 4 \
	--node_rank 0 \
	--rdzv_id ${RANK0_PORT} \
	--rdzv_backend c10d \
	--rdzv_endpoint ${RANK0_IP}:${RANK0_PORT} \
	run_mae_pretraining.py
