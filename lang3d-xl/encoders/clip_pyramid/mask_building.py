import cv2
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from os import makedirs
from pathlib import Path
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SkyDetector:
    
    def __init__(self,
                    device='cuda'):
        self.fe = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model.to(device)
        self.model.eval()
    
    def has_sky(self, img):
        with torch.no_grad():
            inp = self.fe(images=img, return_tensors='pt').to('cuda')
            out = self.model(**inp)
            L = out.logits
        
        return 2 in L[0].argmax(dim=0).cpu().unique()

class FacadeFinder:
    
    def __init__(self,
                device='cuda',
                clipseg_proc_name='CIDAS/clipseg-rd64-refined',
                clipseg_checkpoint='CIDAS/clipseg-rd64-refined',
                clipseg_threshold=0.5
                ):
        
        self.device = device
        
        self.cs_proc = AutoProcessor.from_pretrained(clipseg_proc_name)
        self.cs = CLIPSegForImageSegmentation.from_pretrained(clipseg_checkpoint)

        self.cs.to(device)
        self.cs.eval()
        self.clipseg_threshold = clipseg_threshold
        
        
    def find_facade(self, img, building_type='cathedral'):
        
        I = np.asarray(img)
        W = np.ones_like(I) * 255 # white
        def make_masked_(S):
            S_ = S[..., None]
            IS = I * S_ + W * (1 - S_)
            return IS
        
        with torch.no_grad():
            inp = self.cs_proc(
                text=[building_type],
                images=[img],
                padding="max_length", return_tensors="pt").to('cuda')
            out = self.cs(**inp)
            S = out.logits.sigmoid().cpu().numpy()
            S = cv2.resize(S, img.size)
            fseg = S > self.clipseg_threshold
            cutout = make_masked_(fseg).astype(np.uint8)

        return fseg, cutout
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--images_dir", default='/root/feature-3dgs/data/milano/images', type=str, help="Directory containing images")
    parser.add_argument("--save_dir", default='/root/feature-3dgs/data/milano/masks', type=str, help="Directory to save outputs")
    parser.add_argument("--building_type", type=str, default='cathedral', help="building tipe such as cathedral")

    args = parser.parse_args()

    images_dir = args.images_dir  # "/root/feature-3dgs/data/st_paul/images"
    save_dir = args.save_dir  # f"/root/feature-3dgs/data/st_paul/vis_embeddings_scale_{num}"
    building_type = args.building_type
    makedirs(save_dir, exist_ok=True)

    sky_detector = SkyDetector()
    ff = FacadeFinder()

    

    with torch.no_grad():

        paths = [image_path for image_path in Path(images_dir).iterdir()
             if image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]
        bar_paths = tqdm(paths)

        for image_path in bar_paths:
            image = Image.open(str(image_path))
            # is_outdoor = sky_detector.has_sky(image)
            # if is_outdoor:
            ff_out = ff.find_facade(image, building_type=building_type)
            Image.fromarray(np.uint8(ff_out[0]*255)).save(str(Path(save_dir) / image_path.name))