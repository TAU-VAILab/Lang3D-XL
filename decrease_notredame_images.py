from shutil import copyfile
from os import makedirs
from pathlib import Path
from tqdm import tqdm

orig_dir = Path(r'/storage/shai/3d/data/rgb_data/notre_dame/images')
gt_path = Path(r'/storage/shai/3d/data/HolyScenes/cathedral/notre_dame')

save_dir = orig_dir.parent / 'images_decreased'
makedirs(str(save_dir))

images_names = {path.name.replace('-gt', '') for path in gt_path.glob('*/*-gt.jpg')}

for image_name in tqdm(images_names):
    copyfile(str(orig_dir / image_name), str(save_dir / image_name))