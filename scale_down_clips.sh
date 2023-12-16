#! /bin/bash

# set -o errexit
set -o nounset
# set -o xtrace

function scale_down_clip() {
    input=$1
    output=${input/.mp4/_scaled.mp4}
    if [ ${input} == "*scaled.mp4" ] || [ -f ${output} ]; then
        echo skipping -- ${input}
        rm ${input}
        return
    fi
    # ffmpeg -y \
    #     -hide_banner \
    #     -loglevel error \
    #     -hwaccel cuda \
    #     -hwaccel_output_format cuda \
    #     -i ${input} \
    #     -vf scale_cuda=1280:720 \
    #     -c:v h264_nvenc \
    #     ${output}
    ffmpeg -y \
        -hide_banner \
        -loglevel error \
        -i ${input} \
        -vf scale=1280:720 \
        ${output}
    if [ $? -eq 0 ]; then
        rm ${input}
        echo ${output}
    else
        echo [failed] ${output}
    fi
}

export -f scale_down_clip
find /data/icu -iname '*t.mp4' | \
    sort | \
    xargs -P$(($(nproc)/3)) -I{} bash -c "scale_down_clip {}"