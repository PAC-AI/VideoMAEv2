import os
import sys
import pathlib

from tqdm import tqdm
# import numpy as np
# from skvideo.io import FFmpegReader
from decord import VideoReader,cpu
import joblib


def get_n_frames(clip_f):
    try:
        reader = VideoReader(str(clip_f))
        n_frames = sum(1 for f in reader)
        # reader = FFmpegReader(str(clip_f))
        # n_frames = sum(1 for f in reader.nextFrame())
        # reader.close()
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        return 0
    else:
        return n_frames


data_d = pathlib.Path('/data/icu')
clip_list_f = pathlib.Path('/data/clips.txt')
n_frames_f = pathlib.Path('/data/n_frames.txt')
corrupt_list_f = pathlib.Path('/data/corrupt_clips.txt')

clip_fs = [f
           for pat_d in data_d.iterdir()
           for device_d in pat_d.iterdir()
           for f in device_d.iterdir()
           if f.suffix == '.mp4']
pool = joblib.Parallel(n_jobs=os.cpu_count())
n_frames = list()
jobs = list()
for clip_f in tqdm(clip_fs,desc='clip'):
    jobs.append(joblib.delayed(get_n_frames)(clip_f))
    if len(jobs) == os.cpu_count()*10:
        n_frames.extend(pool(jobs))
        jobs = list()
n_frames.extend(pool(jobs))
assert len(clip_fs) == len(n_frames)
with open(clip_list_f,'w') as clip_list_f:
    for f,n in zip(clip_fs,n_frames):
        if n == 0:
            continue
        clip_list_f.write(f'{f.relative_to(data_d)} 0 -1'+'\n')
with open(n_frames_f,'w') as n_frames_f:
    for f,n in zip(clip_fs,n_frames):
        if n == 0:
            continue
        n_frames_f.write(f'{f.relative_to(data_d)} {n}'+'\n')
with open(corrupt_list_f,'w') as corrupt_list_f:
    for f,n in zip(clip_fs,n_frames):
        if n == 0:
            corrupt_list_f.write(f'{f.relative_to(data_d)}'+'\n')
print('num frames total:',sum(n_frames))

# n_frames_total = 0
# n_corrupt = 0
# clip_fs = [f
#            for pat_d in data_d.iterdir()
#            for device_d in pat_d.iterdir()
#            for f in device_d.iterdir()
#            if f.suffix == '.mp4']
# clip_list_f = open(clip_list_f,'a') 
# n_frames_f = open(n_frames_f,'a')
# corrupt_list_f = open(corrupt_list_f,'a')
# pbar = tqdm(clip_fs,desc='clip')
# for clip_f in pbar:
#     n_frames = get_n_frames(clip_f)
#     if n_frames == 0:
#         n_corrupt += 1
#         corrupt_list_f.write(str(clip_f.relative_to(data_d))+'\n')
#         pbar.set_postfix({'n_corrupt':n_corrupt})
#         continue
#     n_frames_total += n_frames
#     clip_list_f.write(f'{clip_f.relative_to(data_d)} 0 -1'+'\n')
#     n_frames_f.write(f'{clip_f.relative_to(data_d)} {n_frames}'+'\n')
#     pbar.set_postfix({'n_corrupt':n_corrupt,
#                       'n_frames':n_frames_total})
# clip_list_f.close()
# n_frames_f.close()
# corrupt_list_f.close()
