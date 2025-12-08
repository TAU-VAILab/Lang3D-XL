import torch
import numpy as np
from PIL import Image
import colorsys

def generate_colormap(labels):
    """
    Generate distinct colors for each unique label using HSV color space.
    
    Args:
        labels (list): A list of unique labels in the segmentation.
    
    Returns:
        dict: A colormap mapping label values to RGB colors.
    """
    num_labels = len(labels)
    colormap = {-1: np.array((0, 0, 0))}
    
    for i, label in enumerate(labels):
        if label != -1:
            hue = i / num_labels  # Evenly distribute colors in the hue spectrum
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Full saturation and brightness
            colormap[label] = np.array(tuple(int(c * 255) for c in rgb))
    
    return colormap

seg_path = r'/storage/shai/3d/data/rgb_data/st_paul/segmentation_masks/0019.pt'

seg = torch.load(seg_path)

np.random.seed(42)  # For reproducibility

for indx, name in zip(range(4), ['default', 'small', 'medium', 'large']):
    mask = seg[indx].numpy()
    unique_labels = np.unique(mask)
    # colormap = generate_colormap(unique_labels)
    colormap = {label: np.random.randint(0, 255, 3, dtype=np.uint8) for label in unique_labels}
    colormap[-1] = np.array((0,0,0), dtype=np.uint8)
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in colormap.items():
        color_image[mask == label] = color
    Image.fromarray(color_image).save(f'{name}.png')
print()