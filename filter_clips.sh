#! /bin/bash

# set -o errexit
set -o nounset
# set -o xtrace


H=540
W=960
C=3
min_nframes=64
data_d=/data/icu
njobs=$(($(nproc)/3))

function filter_clip() {
    input=$1
    nbytes=`ffmpeg -y \
        -hide_banner \
        -loglevel error \
        -i ${input} \
        -f image2pipe \
        -pix_fmt rgb24 \
        -vodec rawvideo - | \
        wc -l`
    if [ $? -ne 0 ]; then
        rm ${input}
        echo [corrupt] ${input}
        return
    fi
    nframes=$((nbytes/(H*W*C)))
    if (( nframes < min_nframes )); then
        rm ${input}
        echo [short] ${input}
    fi
}

export -f filter_clip
find ${data_d} -iname '*scaled.mp4' | \
    sort | \
    xargs -P${njobs} -I{} bash -c "filter_clip {}"
