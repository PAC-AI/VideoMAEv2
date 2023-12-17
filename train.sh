#! /bin/bash

set -o xtrace

torchrun \
	--nproc_per_node 8 \
	--nnodes 3 \
	--node_rank 0 \
	--rdzv_id 4532 \
	--rdzv_backend c10d \
	--rdzv_endpoint 10.128.0.11:4532 \
	run_mae_pretraining.py
