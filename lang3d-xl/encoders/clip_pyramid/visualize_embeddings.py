from pathlib import Path
from PIL import Image
import numpy as np
import torch
import sklearn
import sklearn.decomposition
import torch.nn as nn
from os import makedirs
from tqdm import tqdm
from matplotlib import pyplot as plt
import clip
import torch.nn.functional as F
from transformers import CLIPProcessor, AutoTokenizer, CLIPModel
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from torchvision.transforms.functional import gaussian_blur
from scipy.ndimage import gaussian_filter

import sys
utils_dir = Path(__file__).resolve().parents[2] / 'utils'
sys.path.append(str(utils_dir))
from data_utils import find_file


def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

def blur_it(scores, blur, alpha=1.0):
    return scores * (1 - alpha) + alpha * gaussian_filter(scores, sigma=blur, truncate=8)

def blur_it2(scores, blur):
    return gaussian_blur(scores.unsqueeze(0), 2 * min(round(4 * blur), 1) + 1, blur).squeeze(0)

class CLIPVisualizer:
    def __init__(self, clip_model_dir, eps=1e-16, output_mode=False, queries=None, neg_classes=None, 
                 semi_neg_classes=None, semi_alpha=1, blur=30, res=2, group_normalization=False) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.blur = blur
        self.blur2 = 0
        self.res = res
        self.group_normalization = group_normalization
        self.norm_min = None
        self.norm_max = None
        self.do_synonims = False
        self.eps = eps
        if clip_model_dir:
            print(f'Loading CLIP from {clip_model_dir}')
            self.tokenizer = AutoTokenizer.from_pretrained(clip_model_dir)
            self.model = CLIPModel.from_pretrained(clip_model_dir).to(self.device).eval()
        else:
            clip_model_dir = "openai/clip-vit-base-patch32"
            print(f'Loading CLIP from {clip_model_dir}')
            self.tokenizer = CLIPProcessor.from_pretrained(clip_model_dir)
            self.model = CLIPModel.from_pretrained(clip_model_dir).to(self.device).eval()
        # self.cannon_tokens = clip.tokenize(["object", "things", "stuff", "texture"]).to(self.device)
        neg_classes = ["object", "things", "stuff", "texture"] if neg_classes is None else neg_classes
        self.cannon_features = self.get_cannonical(eps, neg_classes)
        self.semi_cannon_features = self.get_cannonical(eps, semi_neg_classes) if semi_neg_classes is not None else None
        self.semi_alpha = semi_alpha
        self.output_mode = output_mode
        if queries is not None:
            self.queries = {}
            self.synonims_queries = {}
            for query_text in queries:
                self.queries[query_text] = self.get_query_vec(eps, query_text)
                
                synonims = query_text.split(', ')
                if len(synonims) > 1:
                    main_query = synonims[0]
                    secondary_queries = synonims[1:]
                    self.synonims_queries[query_text] = {
                        'main': self.get_query_vec(eps, main_query),
                        'secondary': self.get_cannonical(eps, secondary_queries)
                    }
        else:
            self.queries = None

    def init_group_normalization(self):
        self.norm_min = None
        self.norm_max = None

    def get_query_vec(self, eps, query_text):
        query_token = self.tokenizer([query_text], padding=True, return_tensors='pt').to(self.device)
        query_features = self.model.get_text_features(**query_token)
        query_features /= (torch.linalg.vector_norm(query_features, dim=1, keepdim=True) + eps)
        ans = query_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return ans

    def get_cannonical(self, eps, neg_classes):
        cannon_tokens = self.tokenizer(neg_classes, padding=True, return_tensors='pt').to(self.device)
        cannon_features = self.model.get_text_features(**cannon_tokens)
        cannon_features /= (torch.linalg.vector_norm(cannon_features, dim=1, keepdim=True) + eps)
        cannon_features = cannon_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return cannon_features
    
    def __call__(self, query_text, image_embeddings, original_image, gt_feature_map_path,
                 should_normalize=True, mask=None):
        image_embeddings = F.interpolate(image_embeddings.unsqueeze(0),
                                         size=tuple(x // self.res for x in original_image.size[::-1]),
                                         mode='bilinear', align_corners=False).squeeze(0)
        if self.blur2:
            image_embeddings = blur_it2(image_embeddings, self.blur2)
        
        scores = self.compute_scores_loop(query_text, image_embeddings, original_image)
            
        
        if mask is not None:
            mask = F.interpolate(mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
                                 size=tuple(x // self.res for x in original_image.size[::-1]),
                                 mode='nearest').squeeze(0).squeeze(0)
            scores = scores * mask
        
        scores_thresh = scores > 0.5
        
        if self.semi_cannon_features is not None:
            scores = self.attenuate_by_semi_negatives(
                query_text, image_embeddings, original_image,
                scores, scores_thresh)
            
        if self.do_synonims and query_text in self.synonims_queries:
            scores = self.attenuate_by_semi_negatives(
                query_text, image_embeddings, original_image,
                scores, scores_thresh, mode='synonims')
                    
        del image_embeddings

        if should_normalize:
            tensor_min, tensor_max = max(scores.min(), 0.5), scores.max()
            if self.group_normalization:
                tensor_min, tensor_max = self.calc_group_normalization_ranges(
                    tensor_min, tensor_max)
            normalized_tensor = (scores - tensor_min) / (tensor_max - tensor_min + self.eps)
            scores = normalized_tensor * scores_thresh
            # if torch.any(torch.isnan(scores)) or torch.any(scores > 1):
            #     print('here')
        else:
            scores = scores
            scores = scores * scores_thresh

        blended_image = blend_image_heatmap(scores.cpu(), original_image, blur=0)
        if self.output_mode:
            blended_image.save(str(gt_feature_map_path.parent / gt_feature_map_path.name[1:]).replace('.jpg', '_vis.jpg'))  # .replace('.png', f'_{query_text}.png'))
        else:
            blended_image.save(str(gt_feature_map_path.parent / gt_feature_map_path.name).replace('.jpg', '_vis.jpg'))  # .replace('.png', f'_{query_text}.png'))
        
        scores = scores.cpu().numpy()

        if self.blur:
            scores = blur_it(scores, self.blur)
        # scores = np.uint8(scores.cpu().numpy() * 255)
        
        start_name = 1 if self.output_mode else 0

        Image.fromarray(np.uint8(scores * 255), 'L').save(
            str(gt_feature_map_path.parent / gt_feature_map_path.name[start_name:]))
        with open(str(gt_feature_map_path.parent / f'{gt_feature_map_path.stem[start_name:]}.npy'), 'wb') as f:
            np.save(f, scores)

    def calc_group_normalization_ranges(self, tensor_min, tensor_max):
        if self.norm_min is None:
            self.norm_min = tensor_min
        else:
            self.norm_min = min(self.norm_min, tensor_min)
        if self.norm_max is None:
            self.norm_max = tensor_max
        else:
            self.norm_max = max(self.norm_max, tensor_max)
        tensor_min, tensor_max = self.norm_min, self.norm_max
        return tensor_min,tensor_max

    def attenuate_by_semi_negatives(
            self, query_text, image_embeddings, original_image,
            scores, scores_thresh, mode='semi_neg'):
        if mode == 'semi_neg':
            semi_scores = self.compute_scores_loop(
                query_text, image_embeddings, original_image, do_semi=True)
        elif mode == 'synonims':
            semi_scores = self.compute_scores_loop(
                query_text, image_embeddings, original_image, do_synonims=True)
        semi_thresh = semi_scores <= 1.0
        semi_scores = torch.clamp((1 - semi_scores) * semi_thresh, 0.0)
        tensor_min, tensor_max = semi_scores.min(), semi_scores.max()
        normalized_tensor = (semi_scores - tensor_min) / (tensor_max - tensor_min + self.eps)
        semi_scores = normalized_tensor * semi_thresh
        semi_scores = scores_thresh * semi_scores
        scores = (scores - 0.5) / (1 + self.semi_alpha * semi_scores) + 0.5
        return scores

    def compute_scores_loop(
            self, query_text, image_embeddings, original_image,
            do_semi=False, do_synonims=False):
        scores = torch.zeros(image_embeddings.shape[-2:])
        for i in range(0, image_embeddings.shape[-2], 100):
            for j in range(0, image_embeddings.shape[-1], 100):
                i_end = min(i + 200, image_embeddings.shape[-2])
                j_end = min(j + 200, image_embeddings.shape[-1])
                scores[i: i_end, j: j_end] = self.compute_scores(
                    query_text, image_embeddings[:, i: i_end, j: j_end],
                    original_image, do_semi=do_semi, do_synonims=do_synonims)
                    
        return scores


    def compute_scores(self, query_text, image_embeddings: torch.tensor,
                       original_image, eps=1e-16, should_print=False,
                       do_semi=False, do_synonims=False):
        
        emb_shape = image_embeddings.shape
        # image_embeddings = image_embeddings.reshape((7, 512, *emb_shape[1:]))
        image_embeddings = image_embeddings.reshape((-1, 512, *emb_shape[1:]))
        # image_embeddings = image_embeddings[1:2]
        image_embeddings /= (torch.linalg.vector_norm(image_embeddings, dim=1, keepdim=True) + eps)
        # image_embeddings = image_embeddings.permute(2, 1, 0, 3)
        # image_embeddings = F.interpolate(image_embeddings, [image_embeddings.shape[-2] * 9, image_embeddings.shape[-1]])
        # image_embeddings = image_embeddings.permute(2, 1, 0, 3)
        image_embeddings = image_embeddings.permute(1, 0, 2, 3).unsqueeze(0)

        if do_synonims:
            query_features = self.synonims_queries[query_text]['main']
        elif self.queries is not None:
            query_features = self.queries[query_text]
        else:
            # query_token = clip.tokenize(query_text).to(self.device)
            query_token = self.tokenizer([query_text], padding=True, return_tensors='pt').to(self.device)
            # query_features = self.model.encode_text(query_token)
            query_features = self.model.get_text_features(**query_token)
            query_features /= (torch.linalg.vector_norm(query_features, dim=1, keepdim=True) + eps)
            query_features = query_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Mean image embaddings:
        # image_embeddings = torch.mean(image_embeddings, dim=2, keepdim=True)

        # Min uppon cannons
        cos_sim_img_query = cosine_sim(image_embeddings, query_features, dim=1)

        if do_semi:
            cos_sim_img_cannon = torch.concat(tuple(
                    cosine_sim(image_embeddings, self.semi_cannon_features[i].unsqueeze(0), dim=1) 
                    for i in range(self.semi_cannon_features.shape[0])),
                dim=0)
        elif do_synonims:
            cos_sim_img_cannon = torch.concat(tuple(
                    cosine_sim(image_embeddings,
                               self.synonims_queries[query_text]['secondary'][i].unsqueeze(0), dim=1) 
                    for i in range(self.synonims_queries[query_text]['secondary'].shape[0])),
                dim=0)
        else:
            cos_sim_img_cannon = torch.concat(tuple(
                    cosine_sim(image_embeddings, self.cannon_features[i].unsqueeze(0), dim=1) 
                    for i in range(self.cannon_features.shape[0])),
                dim=0)
        # del image_embeddings

        score = torch.min(torch.exp(cos_sim_img_query)
                          / (torch.exp(cos_sim_img_cannon)
                             + torch.exp(cos_sim_img_query)), dim=0)
        
        # Max uppon scales:
        score = torch.max(score.values, dim=0)
        score = score.values.squeeze(0)

        if should_print:
            print(f'{query_text}: ({torch.min(score).item():.2f}, {torch.max(score).item():.2f})')
        return score

def cosine_sim(tensor1, tensor2, dim=1):
    return torch.sum(tensor1 * tensor2, dim=dim)

def blend_image_heatmap(tensor_heatmap, original_image, blur=0, normalize=False):
    tensor_heatmap = tensor_heatmap.numpy()
    if normalize:
        above_thresh = tensor_heatmap > 0.5
    else:
        above_thresh = tensor_heatmap > 0

    if blur:
        tensor_heatmap = blur_it(tensor_heatmap * above_thresh, blur)
        above_thresh = tensor_heatmap > 0.01

    # above_thresh = tensor_heatmap.numpy()

    if normalize:
        min_thresh = 0.5 if not blur else 0.01
        tensor_min, tensor_max = max(tensor_heatmap.min(), min_thresh), tensor_heatmap.max()
        normalized_tensor = (tensor_heatmap - tensor_min) / (tensor_max - tensor_min + 1e-16)
    else:
        normalized_tensor = tensor_heatmap
    
    # Apply colormap
    colormap = plt.get_cmap('jet')  # 'jet' is a common heatmap colormap
    # heatmap = colormap(normalized_tensor.numpy())
    heatmap = colormap(normalized_tensor)

    # Convert heatmap to PIL image
    heatmap = np.uint8(heatmap * 255)
    heatmap_image = Image.fromarray(heatmap)

    # Original PIL image (replace with your image)
    # original_image = Image.open('path_to_your_image.jpg')
    original_image = original_image.resize((heatmap.shape[1], heatmap.shape[0]))

    # Blend images
    blended_image = np.array(Image.blend(original_image.convert("RGBA"), heatmap_image.convert("RGBA"), alpha=0.5))
    blended_image = np.uint8(blended_image  * above_thresh[:, :, np.newaxis] + original_image  * (1 - above_thresh[:, :, np.newaxis]))

    return Image.fromarray(blended_image).convert("RGB")


def do_GT(gt_dir="/root/feature-3dgs/data/HolyScenes/cathedral/st_paul",
          save_vis_dir="/root/feature-3dgs/data/st_paul/clip_seg"):
    gt_dir = Path(gt_dir)
    save_vis_dir = Path(save_vis_dir)
    makedirs(str(save_vis_dir), exist_ok=True)

    for category_path in tqdm(gt_dir.iterdir()):
        makedirs(str(save_vis_dir / category_path.relative_to(gt_dir)), exist_ok=True)
        for image_path in category_path.iterdir():
            if image_path.stem.endswith('-gt'):
                gt_mask = Image.open(str(image_path))
                gt_mask = (1 - np.array(gt_mask) / 255) > 0.5
                if len(gt_mask.shape) > 2:
                    gt_mask = gt_mask[:, :, 0]
                # heatmap = np.repeat(gt_mask[:, :, np.newaxis], 3, axis=-1)
                # Apply colormap
                # colormap = plt.get_cmap('jet')  # 'jet' is a common heatmap colormap
                # heatmap = colormap(gt_mask)
                heatmap = np.zeros((*gt_mask.shape, 4))
                heatmap[:, :, 0] = 0.7 * gt_mask
                heatmap[:, :, 3] = 1.0
                heatmap = np.uint8(heatmap * 255)
                heatmap_image = Image.fromarray(heatmap)

                gt_image = Image.open(str(image_path).replace('-gt.jpg', '-img.jpg'))
                gt_image.putalpha(255)
                # gt_image = np.array(gt_image)

                # gt_mask.putalpha(255)
                gt_image = gt_image.resize((heatmap.shape[1], heatmap.shape[0]))

                # Blend images
                blended_image = np.array(Image.blend(gt_image.convert("RGBA"), heatmap_image.convert("RGBA"), alpha=0.5))
                blended_image = np.uint8(blended_image  * gt_mask[:, :, np.newaxis] + gt_image  * (1 - gt_mask[:, :, np.newaxis]))

                save_path = save_vis_dir / image_path.relative_to(gt_dir)

                Image.fromarray(blended_image).convert("RGB").save(str(save_path).replace('-gt.jpg', '_vis.jpg'))

def run_for_every_scale():
    for num in [5,6,7]:
        feature_dir = f"/root/feature-3dgs/data/st_paul/clip_embeddings_scale_{num}"
        images_dir = "/root/feature-3dgs/data/st_paul/images"
        save_vis_dir = f"/root/feature-3dgs/data/st_paul/vis_embeddings_scale_{num}"
        clip_model_dir = "/root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel"
        makedirs(save_vis_dir, exist_ok=True)
        output_mode = False

        with torch.no_grad():
            clip_visualizer = CLIPVisualizer(clip_model_dir, output_mode=output_mode)

            for feature_path in tqdm(sorted(Path(feature_dir).iterdir())):
                if feature_path.suffix.lower() == '.pt':
                    gt_feature_map = torch.load(str(feature_path)).to(torch.float32).to(clip_visualizer.device)
                    # gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
                    gt_feature_map_path = Path(save_vis_dir) / f'{feature_path.stem}_feature_vis.png'
                    # Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)
                    #                 ).save(gt_feature_map_path)
                    print(f'{feature_path.stem}:')
                    if output_mode:
                        original_image = Image.open(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', '.jpg')[1:]))
                    else:
                        original_image = Image.open(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', '.jpg')))
                    original_image.putalpha(255)
                    for category in ["cross", "colonnade", "windows", "lantern", "doors", "clock", "pediment", "statue", "domes", "pillars", "portals", "towers"]:
                        makedirs(str(Path(save_vis_dir) / category), exist_ok=True)
                        clip_visualizer(category, gt_feature_map, original_image,
                                        Path(save_vis_dir) / category / feature_path.stem.replace('_fmap_CxHxW', '.jpg'))  # f'{feature_path.stem}_feature_vis.png')

def get_suffix(path):
    if Path(path).exists():
        return '.jpg'
    
    for suffix in ['.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF', '.tiff' '.TIFF']:
        if Path(path.replace('.jpg', suffix)).exists():
            return suffix
    print('no', str(path))

def process_chunks(all_args):
    args, images_dir, save_vis_dir, clip_model_dir, output_mode, categories, feature_paths_list = all_args
    with torch.no_grad():
        clip_visualizer = CLIPVisualizer(
                clip_model_dir, output_mode=output_mode, queries=categories,
                neg_classes=args.neg_classes, semi_neg_classes=args.semi_neg_classes, semi_alpha=args.semi_alpha, blur=args.blur,
                res=args.res, group_normalization=args.normalize_by_group)            
        for category in categories:
            if args.normalize_by_group:
                clip_visualizer.init_group_normalization()
                print('normalize by group is on')
            makedirs(str(Path(save_vis_dir) / category), exist_ok=True)
            for feature_path in tqdm(feature_paths_list):
                gt_feature_map = torch.load(str(feature_path)).to(torch.float32).to(clip_visualizer.device)
                gt_feature_map_path = Path(save_vis_dir) / f'{feature_path.stem}_feature_vis.png'
                if args.foreground_masks_dir:
                    foreground_path = find_file(Path(args.foreground_masks_dir) / feature_path.stem.replace('_fmap_CxHxW', ''))
                    if foreground_path is not None:
                        foreground_image = Image.open(str(foreground_path))
                        foreground = \
                                torch.from_numpy(np.int32(np.array(foreground_image) > 0))
                    else:
                        foreground = None
                else:
                    foreground = None

                if output_mode:
                    suffix = get_suffix(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', '.jpg')[1:]))
                    original_image = Image.open(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', suffix)[1:]))
                else:
                    suffix = get_suffix(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', '.jpg')))
                    original_image = Image.open(str(Path(images_dir) / feature_path.stem.replace('_fmap_CxHxW', suffix)))
                original_image.putalpha(255)
                
                clip_visualizer(category, gt_feature_map, original_image,
                                    Path(save_vis_dir) / category / feature_path.stem.replace('_fmap_CxHxW', '.jpg'),
                                    mask=foreground)

if __name__ == "__main__":
    # do_GT()
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--feature_dir", type=str, help="Directory containing features")
    parser.add_argument("--images_dir", type=str, help="Directory containing images")
    parser.add_argument("--save_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--gt_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--clip_model_dir", type=str, default="/storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/0_CLIPModel",
                        help="Directory of the CLIP model (default: /root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel)")
    parser.add_argument("--gt", type=bool, default=False, help="should do gt")
    parser.add_argument("--output_mode", type=bool, default=False, help="output of 3d-gaussian-splatting")
    parser.add_argument("--classes_file", type=str, help="Directory to save outputs")
    parser.add_argument("--foreground_masks_dir", default='', type=str)
    parser.add_argument("--neg_classes", default='object,things,stuff,texture', type=str)
    parser.add_argument("--semi_neg_classes", default='', type=str)
    parser.add_argument("--semi_alpha", default=1.0, type=float)
    parser.add_argument("--blur", default=2.5, type=float)
    parser.add_argument("--res", default=2, type=int)
    parser.add_argument("--normalize_by_group", action='store_true')
    parser.add_argument("--max_workers", default=3, type=int)
    parser.add_argument("--chunk_size", default=100, type=int)

    args = parser.parse_args()

    set_start_method('spawn')

    # categories = ["windows", "portals", "spires"]  # ["cross", "colonnade", "windows", "lantern", "doors",
                 #  "clock", "pediment", "statue", "domes", "pillars",
                 #  "portals", "towers", "spires"]
    feature_dir = args.feature_dir  # f"/root/feature-3dgs/data/st_paul/clip_embeddings_scale_{num}"
    images_dir = args.images_dir  # "/root/feature-3dgs/data/st_paul/images"
    save_vis_dir = args.save_dir  # f"/root/feature-3dgs/data/st_paul/vis_embeddings_scale_{num}"
    clip_model_dir = args.clip_model_dir  # "/root/feature-3dgs/encoders/clip_pyramid/0_CLIPModel"
    makedirs(save_vis_dir, exist_ok=True)
    output_mode = args.output_mode
    args.neg_classes = args.neg_classes.split(',')
    args.semi_neg_classes = args.semi_neg_classes.split(',') if args.semi_neg_classes else None

    if args.gt:
        """
        python /storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/visualize_embeddings.py --images_dir /storage/shai/3d/data/HolyScenes/cathedral/milano --save_dir /storage/shai/3d/data/HolyScenes_vis/cathedral/milano --gt true
        """
        print('doing GT.....')
        do_GT(images_dir, save_vis_dir)

    else:
        if args.classes_file:
            with open(args.classes_file) as file:
                categories = json.load(file)
        elif args.gt_dir:
            categories = [path.name for path in Path(args.gt_dir).iterdir() if path.is_dir()]
        else:
            raise ValueError("Missing classes file or gt_dir from which to take classes")
        
        feature_paths_list = sorted([path for path in Path(feature_dir).iterdir() if path.suffix.lower() == '.pt'])
        chunk_size = args.chunk_size
        multi_args = []
        if chunk_size <= 0:
            multi_args.append((args, images_dir, save_vis_dir, clip_model_dir,
                               output_mode, categories, feature_paths_list))
        else:
            for chunk in [feature_paths_list[i:i + chunk_size] for i in range(0, len(feature_paths_list), chunk_size)]:
                # process_chunks((args, images_dir, save_vis_dir, clip_model_dir,
                #                 output_mode, categories, chunk))
                multi_args.append((args, images_dir, save_vis_dir, clip_model_dir,
                                output_mode, categories, chunk))
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(process_chunks, multi_args), total=len(multi_args)))
