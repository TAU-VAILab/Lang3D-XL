import torch
import clip
from PIL import Image
import math
from time import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from os import makedirs
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import RandomCrop


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
        embadding = embadding.reshape(7, 512, *embadding.shape[1:])
        if self.do_mean:
            embadding = torch.mean(embadding, dim=0)
        return self.transform(embadding), str(self.orig_embeddings_paths[idx])


class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim=64):
        super(AutoEncoder, self).__init__()
        # Encoder
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=512*7, out_channels=512*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512*4, out_channels=512*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512*2, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=encoding_dim, kernel_size=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=encoding_dim, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512*2, out_channels=512*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512*4, out_channels=512*7, kernel_size=1),
            nn.ReLU()
        )
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=encoding_dim, kernel_size=1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=encoding_dim, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


def train(dir_path, save_dir, num_epochs=100):
    makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = 2e-3

    image_ds = EmbeddingsDataset(dir_path, device)
    dataloader = DataLoader(image_ds, batch_size=64,
                            shuffle=False, num_workers=0)
    model = AutoEncoder(encoding_dim=64).to(device)
    model.train()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,total_steps=num_epochs * len(dataloader))


    best_loss = 1e9

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for embeddings, paths in dataloader:
            optimizer.zero_grad()

            reconstructed = model(embeddings)

            loss = criterion(reconstructed, embeddings)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(Path(save_dir) / 'best_model'))
    
    image_ds = EmbeddingsDataset(dir_path, device, random_crop=False)
    model.eval()

    with torch.no_grad():
        print('Saving new embeddings:')
        for embedding, path in tqdm(image_ds):
            # model.encode(embeddings)
            reconstructed = model(embedding.unsqueeze(0)).squeeze(0)
            torch.save(reconstructed, str(Path(save_dir) / Path(path).name))


def do_mean(dir_path, save_dir):
    makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_ds = EmbeddingsDataset(dir_path, device, random_crop=False)
    with torch.no_grad():
        print('Saving new embeddings:')
        for embedding, path in tqdm(image_ds):
            torch.save(embedding, str(Path(save_dir) / Path(path).name))

def seperate_scales(dir_path, save_dir):
    makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_ds = EmbeddingsDataset(dir_path, device, random_crop=False, do_mean=False)
    with torch.no_grad():
        print('Saving new embeddings:')
        first_time = True
        for embedding, path in tqdm(image_ds):
            if first_time:
                for scale_num in range(embedding.size(dim=0)):
                    makedirs(f'{save_dir}_{scale_num+1}', exist_ok=True)
                first_time = False

            for scale_num, one_scale_embedding in enumerate(embedding):
                torch.save(one_scale_embedding, str(Path(f'{save_dir}_{scale_num+1}') / Path(path).name))
            


if __name__ == "__main__":
    dir_path = "/root/feature-3dgs/data/st_paul/clip_embeddings"
    save_dir = "/root/feature-3dgs/data/st_paul/clip_embeddings_scale"
    # train(dir_path, save_dir)
    # do_mean(dir_path, save_dir)
    seperate_scales(dir_path, save_dir)