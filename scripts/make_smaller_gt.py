import shutil
from pathlib import Path
from collections import defaultdict

# Source roots
src_root = Path("/storage/shai/3d/data/HolyScenes")
vis_root = Path("/storage/shai/3d/data/HolyScenes_vis")

# Destination roots
dst_root = Path("/storage/shai/3d/data/HolyScenes_small")
vis_dst_root = Path("/storage/shai/3d/data/HolyScenes_vis_small")

# Make sure output roots exist
dst_root.mkdir(parents=True, exist_ok=True)
vis_dst_root.mkdir(parents=True, exist_ok=True)

# Store image names selected per scene
scene_selected_images = defaultdict(set)

# Traverse scenes
for scene_type_dir in src_root.iterdir():
    if not scene_type_dir.is_dir():
        continue

    for scene_name_dir in scene_type_dir.iterdir():
        if not scene_name_dir.is_dir():
            continue

        scene_key = (scene_type_dir.name, scene_name_dir.name)
        selected_images = scene_selected_images[scene_key]

        for class_dir in sorted(scene_name_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            # Gather all image stems (without suffixes)
            stems = sorted({p.stem.replace('-gt', '').replace('-img', '') 
                            for p in class_dir.glob("*.jpg") if '-gt' in p.name or '-img' in p.name})

            # Prioritize by previously chosen images
            prioritized = [s for s in selected_images if s in stems]
            remaining = [s for s in stems if s not in selected_images]

            chosen = prioritized[:5] + remaining[:max(0, 5 - len(prioritized))]
            selected_images.update(chosen)

            # Copy from holyscenes
            dst_class_dir = dst_root / scene_type_dir.name / scene_name_dir.name / class_dir.name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            for stem in chosen:
                for suffix in ["-gt.jpg", "-img.jpg"]:
                    src_img = class_dir / f"{stem}{suffix}"
                    if src_img.exists():
                        shutil.copy2(src_img, dst_class_dir / src_img.name)

            # Copy from holyscenes_vis
            vis_class_dir = vis_root / scene_type_dir.name / scene_name_dir.name / class_dir.name
            vis_dst_class_dir = vis_dst_root / scene_type_dir.name / scene_name_dir.name / class_dir.name
            vis_dst_class_dir.mkdir(parents=True, exist_ok=True)

            for stem in chosen:
                vis_img = vis_class_dir / f"{stem}_vis.jpg"
                if vis_img.exists():
                    shutil.copy2(vis_img, vis_dst_class_dir / vis_img.name)
        print(scene_name_dir, len(selected_images))
