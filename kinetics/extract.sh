#! /bin/bash

set -o errexit
set -o nounset
set -o xtrace

kversion=$1
njobs=$(nproc)

cd /data/kinetics/${kversion}
find . -iname '*.tar.gz' | \
	sort | \
	xargs -P${njobs} -I{} tar -xzf {}
