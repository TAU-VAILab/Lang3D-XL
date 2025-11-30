from pathlib import Path
import torch
from PIL import Image
import numpy as np
from os import makedirs
from tqdm import tqdm
from transformers import CLIPProcessor, AutoProcessor, CLIPModel


def find_existing_file(dir_path: Path, name: str) -> Path:
    """Find an existing file by checking all possible suffixes."""
    for file in dir_path.glob(name + ".*"):
        if file.is_file():
            return file  # Return the first found file
    return None  # Return None if no file is found

def mask_and_crop(image_array: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Apply a mask to an image and crop to the non-zero mask boundaries."""

    # Ensure the mask is boolean
    mask = mask.astype(bool)

    # Apply the mask
    if image_array.shape[-1] == 4:  # If image has an alpha channel
        image_array[~mask] = [0, 0, 0, 0]  # Set non-mask areas to transparent
    else:
        image_array[~mask] = [0, 0, 0]  # Set non-mask areas to black

    # Find the bounding box of the non-zero mask
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None  # If no non-zero mask, return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # +1 to include the last pixel

    # Crop the image
    cropped_array = image_array[y_min:y_max, x_min:x_max]

    # Convert back to PIL Image
    return Image.fromarray(cropped_array)

def main(seg_dir=r'/storage/shai/3d/data/rgb_data/st_paul/segmentation_masks',
         image_dir=r'/storage/shai/3d/data/rgb_data/st_paul/images',
         save_dir=r'/storage/shai/3d/data/rgb_data/st_paul/langsplat_seg_images'):
    seg_dir_path = Path(seg_dir)
    image_dir_path = Path(image_dir)
    makedirs(save_dir, exist_ok=True)
    save_dir_path = Path(save_dir)

    for seg_path in tqdm(seg_dir_path.iterdir()):
        seg = torch.load(seg_path)[1]
        image = Image.open(find_existing_file(image_dir_path, seg_path.stem))
        image_array = np.array(image)
        i = 0
        image_save_dir = save_dir_path / seg_path.stem
        makedirs(image_save_dir, exist_ok=True)
        for label in range(int(torch.max(seg))):
            mask = seg == label
            if not torch.any(mask):
                continue
            label_image = mask_and_crop(image_array.copy(), mask.cpu().numpy())
            label_image.save(image_save_dir / f'test_crop_{i}.png')
            Image.fromarray(mask.cpu().numpy()).save(image_save_dir / f'test_mask_{i}.png')
            i+=1
import torch
import torchvision.transforms.functional as TF

def mask_crop_pad(image, mask):
    image = image * mask
    # Ensure mask is boolean
    mask = mask.bool()

    # Get the nonzero coordinates of the mask
    coords = torch.nonzero(mask, as_tuple=True)
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Crop the image based on the bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1, :]

    # Determine the padding needed to make it rectangular
    height, width = cropped_image.shape[1], cropped_image.shape[2]
    max_side = max(height, width)
    
    pad_top = (max_side - height) // 2
    pad_bottom = max_side - height - pad_top
    pad_left = (max_side - width) // 2
    pad_right = max_side - width - pad_left

    # Pad the cropped image
    padded_image = TF.pad(cropped_image.permute(2, 0, 1), (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    return padded_image.permute(1, 2, 0)

def main_langsplat(seg_dir=r'/storage/shai/3d/data/rgb_data/st_paul/segmentation_masks',
                   image_dir=r'/storage/shai/3d/data/rgb_data/st_paul/images',
                   clip_model_dir=r'/storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/0_CLIPModel',
                   save_dir=r'/storage/shai/3d/data/rgb_data/st_paul/langsplat_sam_clip'):
    seg_dir_path = Path(seg_dir)
    image_dir_path = Path(image_dir)
    makedirs(save_dir, exist_ok=True)
    save_dir_path = Path(save_dir)

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

    for seg_path in tqdm([path for path in seg_dir_path.iterdir() if path.suffix == '.pt']):
        seg = torch.load(seg_path)[1]
        image = torch.from_numpy(np.array(Image.open(
            find_existing_file(image_dir_path, seg_path.stem))))
        features = {}
        for label in range(int(torch.max(seg) + 1)):
            label_mask = (seg == label).unsqueeze(-1)

            if torch.any(label_mask):
                curr_image = Image.fromarray(mask_crop_pad(image, label_mask).cpu().numpy().astype(np.uint8))
                curr_image = preprocess(images=curr_image, return_tensors="pt").to('cuda')

                features[label] = model.get_image_features(
                    pixel_values=curr_image['pixel_values'].squeeze(1)).permute(1, 0).half()
        
        torch.save(features, str(save_dir_path / f'{seg_path.stem}.pt'))
        print()



if __name__ == "__main__":
    main_langsplat()