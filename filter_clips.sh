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
	if [ ! -f ${clip_f} ]; then
		echo [missing] ${clip_f}
		return
	fi
    clip_f=$1
	H=$2
	W=$3
	C=$4
    nbytes=`ffmpeg -y \
        -hide_banner \
        -loglevel error \
        -i ${clip_f} \
        -f image2pipe \
        -pix_fmt rgb24 \
        -vcodec rawvideo - | \
        wc -l`
    if [ $? -ne 0 ]; then
        rm ${clip_f}
        echo [corrupt] ${clip_f}
        return
    fi
    nframes=$((nbytes/(H*W*C)))
    if (( nframes < min_nframes )); then
        rm ${clip_f}
        echo [short] ${clip_f}
		return
    fi
	echo [good] ${clip_f}
}

export -f filter_clip
# find ${data_d} -iname '*scaled.mp4' | \
#     sort | \
#     xargs -P${njobs} -I{} bash -c "filter_clip {}"
cat /data/corrupt_clips.txt | \
	sort | \
	while read clip_f; do
		filter_clip ${clip_f} ${H} ${W} ${C}
	done
