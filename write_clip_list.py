import os
import sys
import pathlib

from tqdm import tqdm
from decord import VideoReader,cpu


def is_clip_readable(clip_f):
    try:
        VideoReader(str(clip_f),ctx=cpu(0))
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        return False
    else:
        return True

corrupt = 0
data_d = pathlib.Path('/data/icu')
pat_d = data_d / sys.argv[1]
clip_list_f = pathlib.Path(f'/data/clips_{sys.argv[1]}.txt')
clip_fs = [f
           for device_d in pat_d.iterdir()
           for f in device_d.iterdir()]
with open(clip_list_f,'w') as clip_list_f:
    for clip_f in tqdm(clip_fs,'clips',disable=True):
        if clip_f.suffix != '.mp4':
            continue
        if is_clip_readable(clip_f):
            clip_list_f.write(f'{clip_f.relative_to(data_d)} 0 -1'+'\n')
        else:
            corrupt += 1
# print(f'num corrupt clips: {corrupt}')