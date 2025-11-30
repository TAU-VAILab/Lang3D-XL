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
from transformers import AutoProcessor, CLIPModel
import argparse

from extract_pixelRepresentedSize import calculate_scale_in_pixels


class ImagePyramidDataset(Dataset):
    def __init__(self, image_path, scales, preprocess, device):
        self.image = Image.open(image_path)
        self.scales = scales
        self.original_width, self.original_height = self.image.size
        self.crop_size = int(self.original_width * self.scales[0])
        self.pixel_shift = self.crop_size // 2
        start_w = self.pixel_shift - (self.pixel_shift -
                                      (self.original_width
                                       - self.original_width // self.pixel_shift * self.pixel_shift)) // 2
        self.center_w = list(range(start_w, self.original_width, self.pixel_shift))
        start_h = self.pixel_shift - (self.pixel_shift -
                                      (self.original_height
                                       - self.original_height // self.pixel_shift * self.pixel_shift)) // 2
        self.center_h = list(range(start_h, self.original_height, self.pixel_shift))
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.scales) * len(self.center_w) * len(self.center_h)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        iscale, iw, ih = self.interpate_ind(idx)
        scale = self.scales[iscale]
        center_w = self.center_w[iw]
        center_h = self.center_h[ih]

        crop_size = int(self.original_width * scale)
        half_crop_size = crop_size // 2

        left = center_w - half_crop_size
        upper = center_h - half_crop_size
        right = center_w + half_crop_size
        lower = center_h + half_crop_size
        
        crop = self.image.crop((left, upper, right, lower))
        
        # if iscale == 6 and (iw % 10 == 0):
        #     crop.save(f'crop_{left}-{right}_{upper}-{lower}.png')

        # tensor_image = self.preprocess(crop).to(self.device)
        tensor_image = self.preprocess(images=crop, return_tensors="pt")

        return tensor_image, iscale, iw, ih
    
    def interpate_ind(self, idx):
        idx_less_h = idx // len(self.center_h)
        ih = idx - idx_less_h * len(self.center_h)
        idx_less_hw = idx_less_h // len(self.center_w)
        iw = idx_less_h - idx_less_hw * len(self.center_w)
        idx_less_hws = idx_less_hw // len(self.scales)
        iscale = idx_less_hw - idx_less_hws * len(self.scales)
        return iscale, iw, ih


def make_pyramid(image_path, save_path, scales, model, preprocess, device):
    image = Image.open(image_path)

    i_scale_start, i_scale_end = 0, len(scales) - 1
    while i_scale_start < i_scale_end and scales[i_scale_start] == scales[i_scale_start + 1]:
        i_scale_start += 1
    while i_scale_start < i_scale_end and scales[i_scale_end] == scales[i_scale_end - 1]:
        i_scale_end -= 1

    image_ds = ImagePyramidDataset(image_path, scales[i_scale_start:i_scale_end+1], preprocess, device)
    dataloader = DataLoader(image_ds, batch_size=4096,
                            shuffle=False, num_workers=0)

    t1 = time()
    with torch.no_grad():
        
        # Create the pyramid
        pyramid = []
        original_width, original_height = image.size
        crop_size = int(original_width * scales[0])
        pixel_shift = crop_size // 2
        embedings = torch.zeros((len(image_ds.center_h),
                                 len(image_ds.center_w),
                                 len(scales), 512)).to(device)

        for batch in dataloader:
            crop, iscale, iw, ih = batch
            crop = crop.to(device)
            # embedings[ih, iw, iscale, :] = model.encode_image(crop).type(embedings.dtype)
            embedings[ih, iw, iscale + i_scale_start, :] = model.get_image_features(
                pixel_values=crop['pixel_values'].squeeze(1)).type(embedings.dtype)
        dimh, dimw, _, _ = embedings.shape
        while i_scale_start > 0:
            embedings[:, :, i_scale_start - 1, :] = embedings[:, :, i_scale_start, :].detach().clone()
            i_scale_start -= 1
        while i_scale_end < len(scales) - 1:
            embedings[:, :, i_scale_end + 1, :] = embedings[:, :, i_scale_end, :].detach().clone()
            i_scale_end += 1
        torch.save(embedings.reshape((dimh, dimw, -1)).permute(2, 0, 1),
                   save_path)

def get_scales(s_min, s_max, n):
    log_s_min = math.log(s_min)
    log_s_max = math.log(s_max)
    scales = np.exp(np.linspace(log_s_min, log_s_max, n))
    return scales


def encode_images(image_dir, save_dir, clip_model_dir, adjustable_scales, colmap_dir, minimal_pixel_window=32):
    makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess = AutoProcessor.from_pretrained(clip_model_dir)
    model = CLIPModel.from_pretrained(clip_model_dir)
    model.to('cuda')
    model.eval()


    if adjustable_scales:
        s_min, s_max = adjustable_scales
        n = round(np.emath.logn(np.sqrt(2), s_max / s_min))
        images_scale_ranges = calculate_scale_in_pixels(colmap_dir, [s_min, s_max])
    else:
        s_min, s_max, n = 0.05, 0.5, 7

    paths = [image_path for image_path in Path(image_dir).iterdir()
             if image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff') 
             and not (Path(save_dir) / f'{image_path.stem}_fmap_CxHxW.pt').exists()]
    bar_paths = tqdm(enumerate(paths))

    for i, image_path in bar_paths:
        bar_paths.set_description(f'Computing embedding {i+1} / {len(paths)}: {image_path.stem}')
        save_path = Path(save_dir) / f'{image_path.stem}_fmap_CxHxW.pt'
        if adjustable_scales:
            image = Image.open(str(image_path))
            original_width, original_height = image.size
            s_min_pixels, s_max_pixels = images_scale_ranges[image_path.name]
            relative_s_min, relative_s_max = s_min_pixels / original_width, s_max_pixels / original_width
            scales = get_scales(relative_s_min, relative_s_max, n)
            scales = np.clip(scales, minimal_pixel_window / original_width, 1.0)
        else:
            scales = get_scales(s_min, s_max, n)
        make_pyramid(str(image_path), str(save_path), scales, model, preprocess, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--clip_model_dir", type=str, default="./encoders/clip_pyramid/0_CLIPModel", help="Directory of the CLIP model (default: /root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel)")
    parser.add_argument("--adjustable_scales", type=str, default='2e-1, 15e-1', help="the wanted scales range. for example: '2e-1, 15e-1'. if empty, not using adjustable")
    parser.add_argument("--colmap_dir", type=str, default='', help="Directory containing colmap data ('sparse' folder). only needed if adjustable_scales not empty.")

    args = parser.parse_args()
    if args.adjustable_scales:
        args.adjustable_scales = [float(num) for num in args.adjustable_scales.split(', ')]
        args.colmap_dir =  args.colmap_dir if  args.colmap_dir else str(Path(args.image_dir).parent / 'sparse' )
    else:
        args.adjustable_scales = None

    # image_dir = "/root/feature-3dgs/data/st_paul/input"
    # save_dir = "/root/feature-3dgs/data/st_paul/clip_embeddings"
    # clip_model_dir = "/root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel"
    encode_images(args.image_dir, args.save_dir, args.clip_model_dir, args.adjustable_scales, args.colmap_dir)
    

"""
    for scale in scales:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        pyramid.append(resized_image)
    image = preprocess(image).unsqueeze(0).to(device)
    scales = torch.exp(torch.linspace(log_s_min, log_s_max, n))
    
    # Create the pyramid
    pyramid = []
    _, H, W = image_tensor.shape
    for scale in scales:
        new_height = int(H * scale)
        new_width = int(W * scale)
        resized_image = F.interpolate(image_tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)
        pyramid.append(resized_image.squeeze(0))
    image_features = model.encode_image(image)
    print(image.shape)
    print(image_features.shape)
    print(time() - t1)
"""
