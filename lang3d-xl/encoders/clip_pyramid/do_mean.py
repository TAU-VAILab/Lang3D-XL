import torch
from tqdm import tqdm
from pathlib import Path
from os import makedirs
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import argparse


class EmbeddingsDataset(Dataset):
    def __init__(self, dir_path, device, random_crop=True, do_mean=True):
        self.dir_path = Path(dir_path)
        self.orig_embeddings_paths = [
            path for path in self.dir_path.iterdir() if path.suffix == '.pt']
        if random_crop:
            self.transform = RandomCrop((30, 40), pad_if_needed=True)
        else:
            self.transform = lambda x: x
        
        self.device = device
        self.do_mean = do_mean

    def __len__(self):
        return len(self.orig_embeddings_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        embadding = torch.load(str(self.orig_embeddings_paths[idx])).to(self.device)
        num_of_channels = embadding.shape[0] // 512
        embadding = embadding.reshape(num_of_channels, 512, *embadding.shape[1:])
        if self.do_mean:
            embadding = torch.mean(embadding, dim=0)
        return self.transform(embadding), str(self.orig_embeddings_paths[idx])

def do_separation(dir_path, save_dir):
    makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_ds = EmbeddingsDataset(dir_path, device, random_crop=False, do_mean=False)
    with torch.no_grad():
        print('Saving new embeddings:')
        made_dirs = False
        for embedding, path in tqdm(image_ds):
            filename = Path(path).name
            if not made_dirs:
                for i_scale in range(embedding.shape[0]):
                    makedirs(str(Path(save_dir) / f'clip_embeddings_scale_{i_scale+1}'), exist_ok=True)
                made_dirs = True

            for i_scale, scale_emb in enumerate(embedding):
                if not (Path(save_dir) / filename).exists():
                    torch.save(scale_emb, 
                               str(Path(save_dir) / f'clip_embeddings_scale_{i_scale+1}' / filename))

def do_mean(dir_path, save_dir):
    makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_ds = EmbeddingsDataset(dir_path, device, random_crop=False)
    with torch.no_grad():
        print('Saving new embeddings:')
        for embedding, path in tqdm(image_ds):
            if not (Path(save_dir) / Path(path).name).exists():
               torch.save(embedding, str(Path(save_dir) / Path(path).name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("dir_path", type=str, help="Directory containing clip embaddings")
    parser.add_argument("save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--separate", type=bool, default=False, help="separate channels instead of doing mean")

    args = parser.parse_args()
    # dir_path = "/root/feature-3dgs/data/st_paul/clip_embeddings"
    # save_dir = "/root/feature-3dgs/data/st_paul/clip_embeddings_scale"
    if args.separate:
        do_separation(args.dir_path, args.save_dir)
    else:
        do_mean(args.dir_path, args.save_dir)
