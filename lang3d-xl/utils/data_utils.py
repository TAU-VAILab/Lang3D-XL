from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F


def find_file(base_path: Path) -> Path:
    """Find an existing file by checking all possible suffixes."""
    for file in base_path.parent.glob(base_path.name + ".*"):
        if file.is_file():
            return file  # Return the first found file
    return None

class ViewGenerator:
    def __init__(self, viewpoint_stack, hard_negative=False, big_loss_num=1e9,
                 dino_dir=None, seg_dir='', foreground_masks_dir = '', mode='train') -> None:
        with torch.no_grad():
            self.mode = mode
            self.viewpoint_stack = viewpoint_stack
            self.hard_negative = hard_negative
            self.indxes_loss = np.array([big_loss_num] * len(self.viewpoint_stack))
            self.indxes = np.arange(len(self.viewpoint_stack))
            self.sorted_indexes = self.indxes.copy()
            self.dino_dir = dino_dir
            if self.dino_dir:
                for ind in tqdm(self.indxes, desc="Loading Dino"):
                    self.viewpoint_stack[int(ind)].dino_features = \
                        torch.load(str(Path(self.dino_dir) / f'{self.viewpoint_stack[int(ind)].image_name}_fmap_CxHxW.pt'))
            self.seg_dir = seg_dir
            if self.seg_dir:
                for ind in tqdm(self.indxes, desc="Loading Segmentations"):
                    self.viewpoint_stack[int(ind)].segmentation = \
                        torch.load(str(Path(self.seg_dir) / f'{self.viewpoint_stack[int(ind)].image_name}.pt'))[1]
            
            self.foreground_masks_dir = foreground_masks_dir
            if self.foreground_masks_dir:
                for ind in tqdm(self.indxes, desc="Loading Forground masks"):
                    foreground_path = find_file(Path(
                        self.foreground_masks_dir) / f'{self.viewpoint_stack[int(ind)].image_name}')
                    if foreground_path is not None:
                        foreground_image = Image.open(str(foreground_path))
                        foreground = \
                            torch.from_numpy(np.round(np.array(foreground_image) / 255))
                        self.viewpoint_stack[int(ind)].foreground = F.interpolate(
                            foreground.to(torch.float32).unsqueeze(0).unsqueeze(0),
                            size=(self.viewpoint_stack[int(ind)].original_image.shape[1],
                                  self.viewpoint_stack[int(ind)].original_image.shape[2]),
                            mode='nearest').squeeze(0).squeeze(0)
                    else:
                        self.viewpoint_stack[int(ind)].foreground = None
    
    def __call__(self, indx=0):
        with torch.no_grad():
            if self.mode == 'train' and self.hard_negative and np.random.rand() > 0.5:
                # indx = np.random.choice(self.indxes, p=self.indxes_loss/np.sum(self.indxes_loss))
                indx = np.random.choice(self.indxes[self.sorted_indexes[:50]])
                return self.viewpoint_stack[indx], indx
            elif self.mode == 'train':
                indx = np.random.choice(self.indxes)
                cam = self.viewpoint_stack[indx]
                # if self.dino_dir:
                #     cam.dino_features = torch.load(str(Path(self.dino_dir) / f'{cam.image_name}_fmap_CxHxW.pt'))
                return cam, indx
            else:
                cam = self.viewpoint_stack[indx]
                # if self.dino_dir:
                #     cam.dino_features = torch.load(str(Path(self.dino_dir) / f'{cam.image_name}_fmap_CxHxW.pt'))
                return cam
    
    def __getitem__(self, index):
        return self.__call__(index)
    
    def __len___(self):
        return len(self.viewpoint_stack)
    
    def update_semantic(self, indx, value):
        # del self.viewpoint_stack[indx].semantic_feature
        # torch.cuda.empty_cache()
        self.viewpoint_stack[indx].semantic_feature = value
        
    
    def sort(self):
        self.sorted_indexes = np.argsort(self.indxes_loss)[::-1]
