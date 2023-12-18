from itertools import chain
from collections import defaultdict
import pathlib


data_d = pathlib.Path('/data/kinetics')
versions = ['k400','k600','k700']
clips_f = data_d / 'clips.txt'
clips = list()
for version in versions:
    version_d = data_d / version
    for entry in version_d.iterdir():
        if entry.suffix == '.mp4':
            clips.append(entry)
        elif entry.is_dir():
            for f in entry.iterdir():
                if f.suffix == '.mp4':
                    clips.append(f)
clips = [f.relative_to(data_d)
         for f in clips]
print(f'number of clips: {len(clips)}')
                    
clip_ids_to_clips = defaultdict(list)
for f in clips:
    clip_id = '_'.join(f.name.split('_')[:-2])
    clip_ids_to_clips[clip_id].append(str(f))
print(f'number of unique ids: {len(clip_ids_to_clips)}')

filenames_to_files = {f.name:str(f)
                      for f in clips}
print(f'number of unique files: {len(filenames_to_files)}')

open(clips_f,'w').write('\n'.join(sorted(filenames_to_files.values())) + '\n')
