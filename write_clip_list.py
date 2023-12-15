import os
import sys
import pathlib

from tqdm import tqdm
from decord import VideoReader,cpu


def is_clip_readable(clip_f):
    try:
        reader = VideoReader(str(clip_f))
        n_frames = sum(1 for f in reader)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        return False
    else:
        return True

corrupt = 0
data_d = pathlib.Path('/data/icu')
device_d = data_d / sys.argv[1]
clip_fs = [f for f in device_d.iterdir()]
clip_list_f = pathlib.Path('/data/clips_' + sys.argv[1].replace('/','_') + '.txt')
with open(clip_list_f,'w') as clip_list_f:
    for clip_f in tqdm(clip_fs,sys.argv[1]):
        if clip_f.suffix != '.mp4':
            continue
        if is_clip_readable(clip_f):
            clip_list_f.write(f'{clip_f.relative_to(data_d)} 0 -1'+'\n')
            clip_list_f.flush()
        else:
            corrupt += 1
            print(f'corrupt: {clip_f}')
print(f'num corrupt clips: {corrupt}')