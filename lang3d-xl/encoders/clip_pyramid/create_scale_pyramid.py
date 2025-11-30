import torch
import clip
from PIL import Image, ImageDraw
import math
from time import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from os import makedirs
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, AutoProcessor, CLIPModel
import argparse

from extract_pixelRepresentedSize import calculate_scale_in_pixels


class ImagePyramidDataset(Dataset):
    def __init__(self, image_path, depth_path, scales, preprocess, device):
        debug_path  = Path(image_path)
        self.debug_dir = Path(f'/storage/shai/3d/data/rgb_data/test_inpaint/debug_crops/{debug_path.parents[1].name}_{debug_path.stem}')
        # makedirs(str(self.debug_dir), exist_ok=True)
        self.image = Image.open(image_path)
        self.depth = torch.load(depth_path).cpu().numpy() if depth_path is not None else None
        self.scales = scales
        self.original_width, self.original_height = self.image.size
        self.crop_size = int(self.original_width * self.scales[0])
        self.pixel_shift = self.crop_size // 2
        start_w = self.pixel_shift - (self.pixel_shift -
                                      (self.original_width
                                       - self.original_width // self.pixel_shift * self.pixel_shift)) // 2
        self.center_w = list(range(start_w, self.original_width, self.pixel_shift))
        if not len(self.center_w):
            self.center_w = [self.original_width // 2]
        start_h = self.pixel_shift - (self.pixel_shift -
                                      (self.original_height
                                       - self.original_height // self.pixel_shift * self.pixel_shift)) // 2
        self.center_h = list(range(start_h, self.original_height, self.pixel_shift))
        if not len(self.center_h):
            self.center_h = [self.original_height // 2]
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
        if self.depth is not None:
            depth = self.depth[max(upper, 0):min(lower, self.original_height),
                            max(left, 0):min(right, self.original_width)]
            if upper < 0:
                depth = np.vstack([np.zeros((-upper, depth.shape[1])), depth])
            if lower+1 > self.original_height:
                depth = np.vstack([depth, np.zeros((lower - self.original_height, depth.shape[1]))])
            if left < 0:
                depth = np.hstack([np.zeros((depth.shape[0], -left)), depth])
            if right+1 > self.original_width:
                depth = np.hstack([depth, np.zeros((depth.shape[0], right - self.original_width))])

            h_depth, w_depth = depth.shape

            mid_depth = depth[h_depth//2 - self.pixel_shift//2: h_depth//2 + self.pixel_shift//2,
                            w_depth//2 - self.pixel_shift//2: w_depth//2 + self.pixel_shift//2]
            mid_depth = mid_depth.flatten()
            close_pixels = np.min(np.abs(depth[np.newaxis, :, :] - mid_depth[:, np.newaxis, np.newaxis]), axis=0) < 1
            crop = Image.fromarray(np.uint8(np.array(crop) * close_pixels[:, :, np.newaxis]
                                            + np.array([[[255]*3]]) * close_pixels[:, :, np.newaxis]))
        
        # if iscale == 6 and (iw % 10 == 0):
        #     crop.save(f'crop_{left}-{right}_{upper}-{lower}.png')

        # tensor_image = self.preprocess(crop).to(self.device)
        # makedirs(str(self.debug_dir), exist_ok=True)
        # crop.save(str(self.debug_dir / f'{iscale}_scale_{ih}_{iw}.png')) #test
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


def draw_centered_rectangles(image, scales, output_path):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for scale in scales:
        rect_w = scale * w
        rect_h = rect_w
        left = (w - rect_w) // 2
        top = (h - rect_h) // 2
        right = left + rect_w
        bottom = top + rect_h
        draw.rectangle([left, top, right, bottom], outline='red', width=10)

    image.save(output_path)

def make_pyramid(image_path, save_path, depth_path, scales, model, preprocess, device, phisical_scales):
    image = Image.open(image_path)

    i_scale_start, i_scale_end = 0, len(scales) - 1
    while i_scale_start < i_scale_end and scales[i_scale_start] == scales[i_scale_start + 1]:
        i_scale_start += 1
    while i_scale_start < i_scale_end and scales[i_scale_end] == scales[i_scale_end - 1]:
        i_scale_end -= 1

    # image_ds = ImagePyramidDataset(image_path, scales[i_scale_start:i_scale_end+1], preprocess, device)
    # dataloader = DataLoader(image_ds, batch_size=4096,
    #                         shuffle=False, num_workers=0)

    t1 = time()
    with torch.no_grad():
        
        # Create the pyramid
        pyramid = []
        original_width, original_height = image.size
        crop_size = int(original_width * scales[0])
        pixel_shift = crop_size // 2
        embedings = {}
        
        """
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for scale in scales[i_scale_start:i_scale_end+1]:
            size = int(original_width * scale)
            draw.rectangle([(50, 50), (50+size, 50+size)], outline='red', width=2)
        image.save(f'test_rect_blue_{Path(image_path).stem}.png')
        """
        """
        draw_centered_rectangles(Image.open(image_path),
                                 scales[i_scale_start:i_scale_end+1],
                                 f'atest_a_rect_{Path(image_path).stem}.png')
        print()
        """
        for scale, phisical_scale in zip(scales[i_scale_start:i_scale_end+1],
                                         phisical_scales[i_scale_start:i_scale_end+1]):
            image_ds = ImagePyramidDataset(image_path, depth_path, [scale], preprocess, device)
            image_ds.debug_dir = image_ds.debug_dir / f'{scale}'
            # makedirs(str(image_ds.debug_dir), exists_ok=True)
            dataloader = DataLoader(image_ds, batch_size=min(4096, len(image_ds)),
                                    shuffle=False, num_workers=0)

            embedings[phisical_scale] = torch.zeros(
                (512, len(image_ds.center_h), len(image_ds.center_w), )).to(device)

            for batch in dataloader:
                crop, iscale, iw, ih = batch
                crop = crop.to(device)
                # embedings[ih, iw, iscale, :] = model.encode_image(crop).type(embedings.dtype)
                embedings[phisical_scale][:, ih, iw] = model.get_image_features(
                    pixel_values=crop['pixel_values'].squeeze(1)).permute(1, 0).type(embedings[phisical_scale].dtype)
                
        torch.save(embedings, save_path)
        

def get_scales(s_min, s_max, n):
    log_s_min = math.log(s_min)
    log_s_max = math.log(s_max)
    scales = np.exp(np.linspace(log_s_min, log_s_max, n))
    return scales


def encode_images(image_dir, save_dir, clip_model_dir, adjustable_scales, colmap_dir, depth_dir, minimal_pixel_window=32):
    makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    if clip_model_dir:
        print(f'Loading CLIP from {clip_model_dir}')
        preprocess = AutoProcessor.from_pretrained(clip_model_dir)
        model = CLIPModel.from_pretrained(clip_model_dir)
    else:
        clip_model_dir = "openai/clip-vit-base-patch32"
        print(f'Loading CLIP from {clip_model_dir}')
        preprocess = CLIPProcessor.from_pretrained(clip_model_dir)
        model = CLIPModel.from_pretrained(clip_model_dir)
    model.to('cuda')
    model.eval()


    if adjustable_scales:
        s_min, s_max = adjustable_scales
        n = round(np.emath.logn(np.sqrt(2), s_max / s_min))
        images_scale_ranges = calculate_scale_in_pixels(colmap_dir, [s_min, s_max])
    else:
        s_min, s_max, n = 0.05, 0.5, 7

    # print('image_dir:', image_dir)
    # print([str(Path(r'/storage/shai/3d/data/rgb_data/test_inpaint/images') / f'{image_path.parents[1].name}_{image_path.name}') for image_path in Path(image_dir).iterdir()][:10])

    paths = [image_path for image_path in Path(image_dir).iterdir()
             if image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff') 
             and not (Path(save_dir) / f'{image_path.stem}_fmap_CxHxW.pt').exists()]
             # and (Path(r'/storage/shai/3d/data/rgb_data/test_inpaint/images') / f'{image_path.parents[1].name}_{image_path.name}').exists()] ## Test
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
            phisical_scales = scales * original_width * adjustable_scales[0] / s_min_pixels
        else:
            scales = get_scales(s_min, s_max, n)
            phisical_scales = scales
        
        depth_path = str(Path(depth_dir) / f'{image_path.stem}.pt') if depth_dir else None
        make_pyramid(str(image_path), str(save_path), depth_path, scales, model, preprocess, device,
                     phisical_scales=phisical_scales)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--clip_model_dir", type=str, default="./encoders/clip_pyramid/0_CLIPModel", help="Directory of the CLIP model (default: /root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel)")
    parser.add_argument("--adjustable_scales", type=str, default='2e-1,30e-1', help="the wanted scales range. for example: '2e-1, 15e-1'. if empty, not using adjustable")
    parser.add_argument("--colmap_dir", type=str, default='', help="Directory containing colmap data ('sparse' folder). only needed if adjustable_scales not empty.")
    parser.add_argument("--depth_dir", type=str, default='', help="Directory containing depth maps")

    args = parser.parse_args()
    if args.adjustable_scales:
        args.adjustable_scales = [float(num) for num in args.adjustable_scales.split(',')]
        args.colmap_dir =  args.colmap_dir if  args.colmap_dir else str(Path(args.image_dir).parent / 'sparse' )
    else:
        args.adjustable_scales = None

    # image_dir = "/root/feature-3dgs/data/st_paul/input"
    # save_dir = "/root/feature-3dgs/data/st_paul/clip_embeddings"
    # clip_model_dir = "/root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel"
    encode_images(args.image_dir, args.save_dir, args.clip_model_dir,
                  args.adjustable_scales, args.colmap_dir, args.depth_dir)
    

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
