from pathlib import Path
from os import makedirs
from tqdm import tqdm
import torch


dir_path = Path(r'/storage/shai/3d/data/rgb_data/st_paul/langsplat_sam_clip')

save_dir = Path(r'/storage/shai/3d/data/rgb_data/st_paul/langsplat_sam_clip_array')
makedirs(str(save_dir), exist_ok=True)

for path in tqdm([path for path in dir_path.iterdir() if path.suffix == '.pt']):
    feat_dict = torch.load(path)
    max_val = max(feat_dict.keys())
    feat_array = torch.zeros((max_val+1, len(feat_dict[max_val])))
    for key, val in feat_dict.items():
        feat_array[key] = val.view(-1)
    torch.save(feat_array, str(save_dir / path.name))
