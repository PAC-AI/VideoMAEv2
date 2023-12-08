import pathlib
from tqdm import tqdm
from decord import VideoReader,cpu


corrupt = 0
skipped = 0
data_d = pathlib.Path('/data/kinetics-400')
videos = dict()
fs = list((data_d / 'replacement/replacement_for_corrupted_k400').iterdir())
for f in tqdm(fs):
    if f.suffix == '.mp4':
        try:
            VideoReader(str(f),ctx=cpu(0))
        except:
            corrupt += 1
            continue
        videos[f.name] = str(f.relative_to('/data'))
fs = list((data_d / 'train').iterdir())
for f in tqdm(fs):
    if f.suffix == '.mp4':
        if f.name not in videos:
            try:
                VideoReader(str(f),ctx=cpu(0))
            except:
                corrupt += 1
                continue
            videos[f.name] = str(f.relative_to('/data'))
        else:
            skipped += 1
print(f'skipped = {skipped:,}')
print(f'corrupt = {corrupt:,}')
videos = [f'{v} 0 -1'
          for v in sorted(videos.values())]
videos = '\n'.join(videos)
open('/data/videos.txt','w').write(videos)


