#! /bin/bash

set -o errexit
set -o nounset
# set -o xtrace

# This script will write multiple files with prefix `clean_clips_` and multiple
# files with prefix `corrupt_clips_`. Concatenate the files into `clean_clips.txt`
# and `corrupt_clips.txt` after this script is finished. Empty dirs will result 
# in an entry in `corrupt_clips_` because below we don't check for empty dirs. 
# Just remove those entries from `corrupt_clips.txt` by hand.

function is_readable() {
    device_d=$1
    clips_txt_f="/data/clean_clips_${device_d/\//_}.txt"
    corrupt_clips_txt_f="/data/corrupt_clips_${device_d/\//_}.txt"
    for clip_f in ${device_d}/*.mp4; do
        nframes=`ffprobe -loglevel error -show_streams ${clip_f} | grep nb_frames | sed 's/nb_frames=//'`
        if [ $? -eq 0 ] && [[ ${nframes} -gt 100 ]]; then
            echo ${clip_f} >> ${clips_txt_f}
            continue
        else
            echo ${clip_f} >> ${corrupt_clips_txt_f}
        fi 
    done
}
export -f is_readable
cd /data/icu
ls -d */* | xargs -P$(nproc) -I{} bash -c "is_readable {}"