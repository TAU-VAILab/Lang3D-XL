import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from os import makedirs
from tqdm import tqdm
import argparse


def create_features(images_dir, save_dir, dino_model='dinov2_vits14', size=128, device='cuda'):

    makedirs(save_dir, exist_ok=True)

    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', dino_model).to(device)

    transform = T.Compose([
    T.Resize((14 * size, 14 * size)),  # 14 * 64
    # T.CenterCrop(14 * size),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    ])

    with torch.no_grad():
        for image_path in tqdm([path for path in Path(images_dir).iterdir()
                                if not (Path(save_dir) / f'{path.stem}_fmap_CxHxW.pt').exists()]):
            image = Image.open(str(image_path))
            image = transform(image)[:3].unsqueeze(0).to(device)
            features = dinov2_vits14.forward_features(image)["x_norm_patchtokens"].squeeze(0)
            features = features.reshape(int(np.sqrt(features.shape[0])), int(np.sqrt(features.shape[0])), features.shape[1])
            torch.save(features.permute(2, 0, 1).cpu(), str(Path(save_dir) / f'{image_path.stem}_fmap_CxHxW.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("image_dir", type=str, default='/storage/shai/3d/data/rgb_data/st_paul/images', help="Directory containing images")
    parser.add_argument("save_dir", type=str, default='/storage/shai/3d/data/rgb_data/st_paul/dino_embeddings', help="Directory to save outputs")
    parser.add_argument("--dino_model", type=str, default='dinov2_vits14',
                        help="Dino model name from facebookresearch/dinov2 options")
    parser.add_argument("--size", type=int, default=128, help="Spatial size of features")
    args = parser.parse_args()

    create_features(images_dir=args.image_dir, save_dir=args.save_dir,
                    dino_model=args.dino_model, size=args.size)
