import json
from os import makedirs
from pathlib import Path
from shutil import copyfile


gt_path = Path('/root/feature-3dgs/data/HolyScenes_vis/cathedral/st_paul')
base_path = Path('/root/feature-3dgs/data')
export_dir = Path('/root/feature-3dgs/data/st_paul_vis_scales')

# vis_dirs = [Path('/root/feature-3dgs/data/st_paul/vis_3d_embeddings_mean_normalized'),
#             Path('/root/feature-3dgs/data/st_paul/vis_embeddings_mean'),
#             Path('/root/feature-3dgs/data/st_paul/vis_embeddings_mean_normalized'),
#             Path('/root/feature-3dgs/data/st_paul/clipseg_ft')]
vis_dirs = [Path('/root/feature-3dgs/data/st_paul/vis_embeddings_mean_normalized')]
for num in range(1,8):
    vis_dirs.append(Path(f'/root/feature-3dgs/data/st_paul/vis_embeddings_scale_{num}'))
makedirs('/root/feature-3dgs/data/st_paul_vis/HolyScenes_vis/cathedral/st_paul', exist_ok=True)
for vis_dir in vis_dirs:
    makedirs(str(export_dir / vis_dir.relative_to(base_path)), exist_ok=True)

for category_path in gt_path.iterdir():
    if category_path.is_dir():
        makedirs(str(export_dir / category_path.relative_to(base_path)), exist_ok=True)
        for vis_dir in vis_dirs:
            makedirs(str(export_dir / vis_dir.relative_to(base_path) / category_path.relative_to(gt_path)), exist_ok=True)
        for image_path in category_path.iterdir():
            copyfile(str(image_path), str(export_dir / image_path.relative_to(base_path)))
            for vis_dir in vis_dirs:
                copyfile(str(vis_dir / image_path.relative_to(gt_path)),
                         str(export_dir / vis_dir.relative_to(base_path) / image_path.relative_to(gt_path)))
for file in base_path.iterdir():
    if file.suffix == '.html':
        copyfile(str(file), str(export_dir / file.relative_to(base_path)))
