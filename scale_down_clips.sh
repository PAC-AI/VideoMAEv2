#! /bin/bash

# set -o errexit
# set -o nounset
# set -o xtrace


data_d=/data/icu
njobs=1 # $(($(nproc)/3))

function scale_down_clip() {
    input=$1
    output=${input/.mp4/_scaled.mp4}
    if [ -f ${output} ]; then
        rm ${input}
        echo skipping -- ${input}
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
find ${data_d} -iname '*.mp4' | \
    grep -v '_scaled.mp4' | \
    sort | \
    xargs -P${njobs} -I{} bash -c "scale_down_clip {}"