#! /bin/bash

set -o errexit
set -o nounset
set -o xtrace

kversion=$1
njobs=$(nproc)

mkdir -p /data/kinetics/${kversion}
cd /data/kinetics/${kversion}
cat /data/kinetics/${kversion}_*.txt | \
	sort | \
	xargs -P${njobs} -I{} wget --continue --quiet {}
