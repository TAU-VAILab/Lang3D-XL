from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
import json
from pathlib import Path


def get_clip_visualizer(dataset, wild_params):
    classes_file = Path(dataset.source_path) / 'classes.json'
    if classes_file.exists():
        with open(str(classes_file)) as file:
            vis_classes = json.load(file)
    else:
        vis_classes = ['windows', 'towers', 'domes']
    if 'clip' in dataset.foundation_model:
        clip_vis = CLIPVisualizer(wild_params.clip_dir, vis_classes)
    else:
        clip_vis = None
    return clip_vis

class CLIPVisualizer:
    def __init__(self, clip_model_dir, quary_texts, eps=1e-16, output_mode=False) -> None:
        with torch.no_grad():
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if clip_model_dir:
                print(f'Loading CLIP from {clip_model_dir}')
                self.tokenizer = AutoTokenizer.from_pretrained(clip_model_dir)
                self.model = CLIPModel.from_pretrained(clip_model_dir).to(self.device).eval()
            else:
                clip_model_dir = "openai/clip-vit-base-patch32"
                print(f'Loading CLIP from {clip_model_dir}')
                self.tokenizer = CLIPProcessor.from_pretrained(clip_model_dir)
                self.model = CLIPModel.from_pretrained(clip_model_dir).to(self.device).eval()
            self.cannon_tokens = self.tokenizer(["object", "things", "stuff", "texture"],
                                                padding=True, return_tensors='pt').to(self.device)
            self.cannon_features = self.model.get_text_features(**self.cannon_tokens)
            self.cannon_features /= (torch.linalg.vector_norm(self.cannon_features, dim=1, keepdim=True) + eps)
            self.cannon_features = self.cannon_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.output_mode = output_mode
            self.quary_texts = quary_texts
            self.query_features = {}
            for quary_text in quary_texts:
                query_token = self.tokenizer([quary_text], padding=True, return_tensors='pt').to(self.device)
                query_features = self.model.get_text_features(**query_token)
                query_features /= (torch.linalg.vector_norm(query_features, dim=1, keepdim=True) + eps)
                self.query_features[quary_text] = query_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def compute_scores(self, image_embeddings: torch.tensor, quary_text: str, eps=1e-16, should_combine=True):
        
        emb_shape = image_embeddings.shape
        image_embeddings = image_embeddings.reshape((-1, *emb_shape))
        image_embeddings /= (torch.linalg.vector_norm(image_embeddings, dim=1, keepdim=True) + eps)
        # image_embeddings = image_embeddings.permute(2, 1, 0, 3)
        # image_embeddings = F.interpolate(image_embeddings, [image_embeddings.shape[-2] * 9, image_embeddings.shape[-1]])
        # image_embeddings = image_embeddings.permute(2, 1, 0, 3)
        image_embeddings = image_embeddings.permute(1, 0, 2, 3).unsqueeze(0)

        # Min uppon cannons
        cos_sim_img_quary = cosine_sim(image_embeddings, self.query_features[quary_text], dim=1)

        cos_sim_img_cannon = torch.concat(tuple(
                cosine_sim(image_embeddings, self.cannon_features[i].unsqueeze(0), dim=1) 
                for i in range(self.cannon_features.shape[0])),
            dim=0)
        # del image_embeddings

        score = torch.min(torch.exp(cos_sim_img_quary)
                          / (torch.exp(cos_sim_img_cannon)
                             + torch.exp(cos_sim_img_quary)), dim=0)
        
        # Max uppon scales:
        score = torch.max(score.values, dim=0)
        score = score.values.squeeze(0)

        if should_combine:
            return ((score > 0.5) * cos_sim_img_quary.squeeze(0).squeeze(0)).unsqueeze(0), cos_sim_img_quary.squeeze(0)
        else:
            return score, cos_sim_img_quary.squeeze(0)

def cosine_sim(tensor1, tensor2, dim=1):
    return torch.sum(tensor1 * tensor2, dim=dim)

def blend_image_heatmap(tensor_heatmap, original_image):
    tensor_min, tensor_max = tensor_heatmap.min(), tensor_heatmap.max()
    tensor_min = 0.5
    # tensor_max = 1.0
    normalized_tensor = (tensor_heatmap - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)
    
    above_thresh = tensor_heatmap.numpy() > 0.5

    # Apply colormap
    colormap = plt.get_cmap('jet')  # 'jet' is a common heatmap colormap
    heatmap = colormap(normalized_tensor.numpy())

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

def draw_mid_rect(image: Image.Image, rect_size: tuple, color: str = 'red', width=3, loc=None):
    if isinstance(rect_size, int):
        rect_size = (rect_size, rect_size)

    rect_height, rect_width = rect_size
    image_width, image_height = image.size

    if loc is None:
        x0 = (image_width - rect_width) // 2
        y0 = (image_height - rect_height) // 2
        x1 = x0 + rect_width
        y1 = y0 + rect_height
    else:
        y, x = loc
        x0 = x - rect_width // 2
        y0 = y - rect_height // 2
        x1 = x0 + rect_width
        y1 = y0 + rect_height

    # Draw the rectangle
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

def get_gt_features(viewpoint_cam, features_scales_mode='avarage',
                    avg_p=1.1, num_avg=0, max_feat_scale=-1, min_feat_scale=-1):
    with torch.no_grad():
        gt_feature_map = viewpoint_cam.semantic_feature
        if isinstance(gt_feature_map, str):
            gt_feature_map = torch.load(gt_feature_map).cuda()
        if isinstance(gt_feature_map, dict):
            keys = gt_feature_map.keys()
            
            scale_keys = sorted([key for key in keys if isinstance(key, float)])
            if max_feat_scale > 0:
                scale_keys_max = [key for key in scale_keys if key <= max_feat_scale]
                if len(scale_keys_max) > 0:
                    scale_keys = scale_keys_max
                else:
                    scale_keys = scale_keys[-1:]
            if min_feat_scale > 0:
                scale_keys_min = [key for key in scale_keys if key >= min_feat_scale]
                if len(scale_keys_min) > 0:
                    scale_keys = scale_keys_min
                else:
                    scale_keys = scale_keys[:1]

            do_avg = 'avarage' in features_scales_mode and np.random.rand() < avg_p
            
            avg_till = len(scale_keys)
            
            if do_avg and num_avg > 0:
                avg_till = max(1, int(np.random.randint(len(scale_keys) - num_avg, len(scale_keys) + 1)))
            
            avg_type = f'avarage_{avg_till}'
            range_type = f'range_factor_{avg_till}'

            if do_avg and avg_type not in keys:
                avg_gt_feature, range_factor = calc_avarage_features(gt_feature_map, scale_keys[:avg_till])
                gt_feature_map = avg_gt_feature.cuda()
            elif do_avg:
                avg_gt_feature = gt_feature_map[avg_type]
                range_factor = gt_feature_map[range_type]
                gt_feature_map = avg_gt_feature.cuda()
            else:
                key = np.random.choice(scale_keys)
                gt_feature_map = gt_feature_map[key].cuda()
                range_factor = None
        else:
            gt_feature_map = gt_feature_map.cuda()
            range_factor = None

    return gt_feature_map, range_factor

def calc_avarage_features(gt_feature_map, sorted_keys):
    keys = sorted_keys
    avg_gt_feature = None
    for key in keys:
        gt_feat = gt_feature_map[key].cuda()
        if avg_gt_feature is None:
            avg_gt_feature = gt_feat
        elif gt_feat.shape[1] > avg_gt_feature.shape[1]:
            avg_gt_feature = gt_feat + F.interpolate(
                                avg_gt_feature.unsqueeze(0), size=(gt_feat.shape[1], gt_feat.shape[2]),
                                mode='bilinear', align_corners=False).squeeze(0) 
        else:
            avg_gt_feature = avg_gt_feature + F.interpolate(
                                gt_feat.unsqueeze(0), size=(avg_gt_feature.shape[1], avg_gt_feature.shape[2]),
                                mode='bilinear', align_corners=False).squeeze(0)
    avg_gt_feature = avg_gt_feature / len(keys)
    range_factor = keys[-1] / keys[0]
    return avg_gt_feature, range_factor
 
