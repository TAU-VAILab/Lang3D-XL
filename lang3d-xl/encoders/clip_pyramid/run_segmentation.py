import cv2
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import pandas
import base64
from tqdm import tqdm
from argparse import ArgumentParser
from collections import namedtuple
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
from pathlib import Path

class SkyDetector:
    
    def __init__(self,
                    device='cuda'):
        self.fe = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model.to(device)
        self.model.eval();
    
    def has_sky(self, img):
        with torch.no_grad():
            inp = self.fe(images=img, return_tensors='pt').to('cuda')
            out = self.model(**inp)
            L = out.logits
        
        return 2 in L[0].argmax(dim=0).cpu().unique()
        # 2 is "sky" label in ade20k: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv

class HorizSlider:
    
    def __init__(self,
                    device='cuda',
                    CKPT='../correspondences/output/clipseg_ft_crops_10epochs',
                    BASE_CKPT='CIDAS/clipseg-rd64-refined',
                    stride=0.1):
        
        self.ff = FacadeFinder(device=device)
        self.processor = AutoProcessor.from_pretrained(BASE_CKPT)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CKPT)
        self.model.to(device)
        self.model.eval()
        
        self.stride = stride
        
        self.sky_detector = SkyDetector(device=device)
        
    def segment(self, img, label, building_type='cathedral', debug=False):
        
        is_outdoor = self.sky_detector.has_sky(img)
        if debug:
            print('is_outdoor?', is_outdoor)
        
        if is_outdoor:
        
            ff_out = self.ff.find_facade(img, building_type=building_type, get_bbox=True)
            if ff_out is None:
                is_outdoor = False
                if debug:
                    print('no facade found, switching is_outdoor to False')
            else:
                fseg, cutout, bbox, bbox_m = ff_out
                img_c = img.crop((bbox_m.y0, bbox_m.x0, bbox_m.y1, bbox_m.x1))
        
        if not is_outdoor:
            img_c = img
            
        w, h = img_c.size
        if w <= h:
            if debug:
                print('padding...')
            seg = self._seg_pad(img_c, label)
        else:
            if debug:
                print('cropping...')
            seg = self._seg_crop(img_c, label)
        assert seg.shape == np.asarray(img_c).shape[:2], 'segmentation shape does not match image'
        
        # pad segmentation of facade crop to match original image size:
        
        if is_outdoor:
            seg_out = np.zeros_like(img.convert('L'), dtype=np.float32)
            seg_out[bbox_m.x0:bbox_m.x1, bbox_m.y0:bbox_m.y1] = seg
            return seg_out
        else:
            return seg
            
    def _seg_crop(self, img_c, label):
        w, h = img_c.size
        STRIDE = int(h * self.stride)
        indices = [(i, 0, h+i, h) for i in range(0, w-h, STRIDE)]
        crops = [img_c.crop(x) for x in indices]

        with torch.no_grad():
            inp = self.processor(images=crops, text=[label] * len(crops), return_tensors="pt", padding=True).to('cuda')
            out = self.model(**inp).logits
            S_all = out.sigmoid().cpu().numpy()
            if len(S_all.shape) == 2:
                S_all = S_all[None] # edge case - only one crop => output has one less dimension

        counts = np.zeros_like(img_c.convert('L'), dtype=np.int64)
        seg = np.zeros_like(img_c.convert('L'), dtype=np.float32)
        for S_, coords in zip(S_all, indices):
            S_ = cv2.resize(S_, (h, h))
            x0, y0, x1, y1 = coords
            for x in range(x0, x1):
                for y in range(y0, y1):
                    dx = x - x0
                    dy = y - y0
                    counts[y, x] += 1
                    seg[y, x] += S_[dy, dx]
        seg /= np.where(counts == 0., 1., counts)
        
        return seg
        
    def _seg_pad(self, img_c, label):
        w, h = img_c.size
        delta = h - w
        pad_left = delta // 2
        pad_right = delta - pad_left

        img_p = Image.fromarray(cv2.copyMakeBorder(np.asarray(img_c), 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE))

        with torch.no_grad():
            inp = self.processor(images=[img_p], text=[label], return_tensors="pt", padding=True).to('cuda')
            out = self.model(**inp).logits
            S_p = out.sigmoid().cpu().numpy()

        seg_p = cv2.resize(S_p, img_p.size)
        seg = seg_p[:, pad_left:-pad_right] if pad_right > 0 else seg_p[:, pad_left:]
        
        return seg

BBox = namedtuple("BBox", "x0 y0 x1 y1")   
# TODO: x and y are backwards?
    
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
        
        
    def find_facade(self, img, building_type='cathedral', pbar=False, get_bbox=False, bbox_margin=0.1):
        
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

        if not get_bbox:    
            return fseg, cutout
        
        x, y = np.where(fseg)
        
        if len(x) == 0 or len(y) == 0:
            return None
        
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        
        bbox = BBox(x0, y0, x1, y1)
        
        # margins
        w = x1 - x0
        h = y1 - y0
        mx = int(w * bbox_margin)
        my = int(h * bbox_margin)
        
        mx0 = max(0, bbox.x0 - mx)
        my0 = max(0, bbox.y0 - my)
        mx1 = min(img.size[1], bbox.x1 + mx)
        my1 = min(img.size[0], bbox.y1 + my)
        
        bbox_m = BBox(mx0, my0, mx1, my1)
        
        return fseg, cutout, bbox, bbox_m



def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--prompts', type=str, default='windows', help='all the text prompts to segment, sperated with ;')
    parser.add_argument('--folder_to_save', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs/milano/horizontal', help='path to save the segmentations')
    parser.add_argument('--model_path', type=str, default='data/clipseg_ft_crops_refined_plur_newcrops_10epochs', help='the path where the fine-tuned clipseg is saved')
    parser.add_argument('--building_type', type=str, default='cathedral', help='the building type on which you whould llke to segment. It can be mosqte/cathedral/synagogue/building/all/etc.')  #'mosque' 'cathedral' 'synagogue' 'building', 'all'
    parser.add_argument('--csv_retrieval_path', type=str, default='data/milano_geometric_occlusions.csv', help='the csv file that containes the score of the files for retreival. The retrieval of the files will be sorted accordingly.')
    parser.add_argument('--images_folder', type=str, default='data/0_1_undistorted/dense/images/', help='the folder that contained the images for segmentation')
    parser.add_argument('--use_csv_for_retrieval', type=bool, default=False, help='If to use the csv for retrieval of the images. You can change to False to run on all images.')
    parser.add_argument('--n_files', type=int, default=-1, help='the number of files to retrieve from the csv retrieval')
    parser.add_argument('--save_images', type=bool, default=False, help='save the images to a HTML file for visualization')
    parser.add_argument('--save_baseline', type=bool, default=True, help='save the segmentation of baseline CLIPSeg model')
    parser.add_argument('--save_refined_clipseg', type=bool, default=True, help='save the segmentation of fine-tuned CLIPSeg model')
    parser.add_argument('--classes_file', type=str, default='', help='path to json file containing the classes. if exists, promts argument will be ignored.')
    return parser.parse_args()


def print_img(image_path, output_file):
    """
    Encodes an image into html.
    image_path (str): Path to image file
    output_file (file): Output html page
    """
    if os.path.exists(image_path):
        img = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        print(
            '<img src="data:image/png;base64,{0}">'.format(img),
            file=output_file,
        )


def get_k_files(k, csv_path):
    prompt = ["score"]
    csv_file = pandas.read_csv(csv_path)
    col = csv_file[prompt]
    col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = csv_file['fn'][files_pos.index]
    return names_pos.values.tolist()

def get_vis_colors(seg, gt_image):
    # heatmap = np.repeat(seg[:, :, np.newaxis], 3, axis=-1)
    # Apply colormap
    colormap = plt.get_cmap('jet')  # 'jet' is a common heatmap colormap
    heatmap = colormap(seg)
    heatmap = np.uint8(heatmap * 255)
    heatmap_image = Image.fromarray(heatmap)

    gt_image.putalpha(255)
    gt_image = gt_image.resize((heatmap.shape[1], heatmap.shape[0]))

    # Blend images
    blended_image = np.array(Image.blend(gt_image.convert("RGBA"), heatmap_image.convert("RGBA"), alpha=0.5))
    return Image.fromarray(np.uint8(blended_image)).convert("RGB")


args = get_opts()
if args.classes_file:
    import json
    with open(args.classes_file) as file:
        cat = json.load(file)
else:
    cat = args.prompts.split(';')
BT = args.building_type
folder_to_save = args.folder_to_save
images_folder = args.images_folder
use_csv_for_retrieval = args.use_csv_for_retrieval
csv_path = args.csv_retrieval_path
n_files = args.n_files if args.n_files >= 0 else None
save_images = args.save_images
save_baseline = args.save_baseline
save_refined_clipseg = args.save_refined_clipseg
model_path = args.model_path


CKPT = model_path
CKPT_BASE = 'CIDAS/clipseg-rd64-refined'  #data_folder + 'clipseg-base_model/
colormap = plt.get_cmap('jet')
hs = HorizSlider(CKPT=CKPT)
hs_base = HorizSlider(CKPT=CKPT_BASE)


for c in cat:
    print(c)
    label = c
    if use_csv_for_retrieval:
        if not os.path.exists(csv_path):
            print('csv does not exist!')
            continue
        imgs_list = get_k_files(n_files, csv_path)
    else:
        try:
            imgs_list = os.listdir(images_folder)
        except:
            print(f"indoor label {label} is not in the csv!")
            continue

    folder2save_clipseg_base = os.path.join(os.path.join(folder_to_save, 'clipseg_base'), c)
    folder2save_clipseg_ft = os.path.join(os.path.join(folder_to_save, 'clipseg_ft'), c)
    os.makedirs(folder2save_clipseg_base, exist_ok=True)
    os.makedirs(folder2save_clipseg_ft, exist_ok=True)

    if save_images:
        # save HTML
        html_out = open(os.path.join(folder2save_clipseg_ft, "clipseg_ft_horiz.html"), "w")
        print('<head><meta charset="UTF-8"></head>', file=html_out)
        print("<h1>Results</h1>", file=html_out)

    for i in tqdm(range(len(imgs_list))):
        img_name = imgs_list[i]
        name = img_name.split('.')[0]
        
        if (Path(folder2save_clipseg_ft) / f'{name}_vis.jpg').exists():
            continue
        img = Image.open(os.path.join(images_folder,img_name)).convert('RGB')
        try:
            seg = hs.segment(img, label, building_type=BT)
            if save_baseline:
                seg_base = hs_base.segment(img, label, building_type=BT)

        except:
            print("error!")
            continue

        img = img.resize((img.size[0] // 2, img.size[1] // 2))

        
        seg_vis = get_vis_colors(seg, img)
        seg = cv2.resize(seg, (img.size[0], img.size[1])) * 255

        if save_baseline:
            seg_base_vis = get_vis_colors(seg_base, img)
            seg_base = cv2.resize(seg_base, (img.size[0], img.size[1])) * 255

        if save_refined_clipseg:
            cv2.imwrite(os.path.join(folder2save_clipseg_ft, name + '.jpg'), seg)
            seg_vis.save(os.path.join(folder2save_clipseg_ft, name + '_vis.jpg'))
            # with open(os.path.join(folder2save_clipseg_ft, name + '.pickle'), 'wb') as handle:
            #     torch.save(seg, handle)
        if save_baseline:
            cv2.imwrite(os.path.join(folder2save_clipseg_base, name + '.jpg'), seg)
            seg_base_vis.save(os.path.join(folder2save_clipseg_base, name + '_vis.jpg'))
            #with open(os.path.join(folder2save_clipseg_base, name + '.pickle'), 'wb') as handle:
            #    torch.save(seg_base, handle)
        if save_images:
            fig = plt.figure()
            fig, axis = plt.subplots(1,5, figsize=(20,4))
            fig.suptitle(f'category: {c}, retreival order: {i}')
            axis[0].imshow(img)
            axis[0].title.set_text('rgb gt')

            im = axis[1].imshow(seg, cmap=colormap)
            axis[1].title.set_text(f'clipseg ft pred')
            axis[2].imshow(img)

            seg_thresh = seg
            seg_thresh[seg_thresh < 0.2] = 0
            seg_thresh[seg_thresh >= 0.2] = 1

            axis[2].imshow(seg_thresh, cmap=colormap, alpha=0.5)
            axis[2].title.set_text(f'clipseg ft pred overlay')

            if save_baseline:
                axis[3].imshow(seg_base, cmap=colormap)
                axis[3].title.set_text(f'clipseg base pred')
                axis[4].imshow(img)
                axis[4].imshow(seg_base, cmap=colormap, alpha=0.5)
                axis[4].title.set_text(f'clipseg base pred overlay')

            for ax in axis:
                ax.axis('off')
            plt.tight_layout()
            fig.colorbar(im)

            path2save = os.path.join(folder2save_clipseg_ft, name + '_pred_clipseg.png')
            plt.savefig(path2save)
            print_img(path2save, html_out)
            print(f"<br><b>{os.path.basename(path2save)}</b><br>", file=html_out)
            os.remove(os.path.join(folder2save_clipseg_ft, name + '_pred_clipseg.png'))

    if save_images:
        print("<hr>", file=html_out)
        html_out.close()

"""
python run_segmentation.py --prompts "domes;portals;towers;windows" --model_path /root/feature-3dgs/encoders/clip_pyramid/clipseg_ft --folder_to_save /root/feature-3dgs/data/st_paul/clipseg --building_type cathedral --images_folder /root/feature-3dgs/data/st_paul/images
python run_segmentation.py --classes_file /storage/shai/3d/data/rgb_data/st_paul/classes.json --model_path /storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/clipseg_ft --folder_to_save /storage/shai/3d/data/rgb_data/st_paul/clipseg --building_type cathedral --images_folder /storage/shai/3d/data/rgb_data/st_paul/images

"""
