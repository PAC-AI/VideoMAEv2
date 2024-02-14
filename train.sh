#! /bin/bash

set -o nounset
set -o errexit


#################### USE THIS ON GCP #############################
# set -o xtrace
# NODE_RANK=0
# NNODES=1
# NPROC_PER_NODE=4
# RANK0_IP='localhost'
# RANK0_PORT=4532

# if [ ${NODE_RANK} != 0 ]; then
# 	rsync -azP ${RANK0_IP}:/data/output/ /data/output/
# fi

# export OMP_NUM_THREADS=$(($(nproc)/${NPROC_PER_NODE}))
# torchrun \
# 	--nproc_per_node ${NPROC_PER_NODE} \
# 	--nnodes ${NNODES} \
# 	--node_rank ${NODE_RANK} \
# 	--rdzv_id ${RANK0_PORT} \
# 	--rdzv_backend c10d \
# 	--rdzv_endpoint ${RANK0_IP}:${RANK0_PORT} \
# 	run_mae_pretraining.py

##################### USE THIS ON CARINA #########################
# source /home/shrik/.bashrc
# source activate videomae
set -o xtrace
cd /home/shrik/VideoMAEv2
export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS=2
torchrun \
	--standalone \
	--nproc_per_node 4 \
	--nnodes 1 \
	run_mae_pretraining.py
