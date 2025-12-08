from pathlib import Path
from random import randint
import torch
from torch.nn import functional as F
import numpy as np
import sys
utils_dir = Path(__file__).resolve().parent
sys.path.append(str(utils_dir))
src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(src_dir))
from data_utils import ViewGenerator
from arguments import OptimizationParams, WildParams


def freeze_nonfeature(opt: OptimizationParams, wild_params: WildParams):
    opt.position_lr_init = 0.0
    opt.position_lr_final = 0.0
    opt.feature_lr = 0.0
    opt.opacity_lr = 0.0
    opt.scaling_lr = 0.0
    opt.rotation_lr = 0.0
    opt.densify_from_iter = opt.iterations + wild_params.warmup_iterations + 1
    opt.wild_lr = 0.0
    opt.wild_latent_lr = 0.0
    return opt

def check_wild_params(wild_params: WildParams):
    wild_params.vis_enc_hash = wild_params.vis_enc_hash if not wild_params.no_vis_enc_hash else False
    wild_params.dino_size = wild_params.dino_size if wild_params.dino_dir else 0
    return wild_params

def check_dataset_params(dataset, wild_params):
    dataset.is_feat_enc = wild_params.feat_enc
    dataset.gauss_feat_size = 16 if not wild_params.feat_enc_hash else 3
    dataset.xyz_to_semantics = wild_params.xyz_to_semantics
    return dataset

def get_out_dim(scene, wild_params):
        viewpoint_stack = scene.getTrainCameras().copy()
        # cam1 = viewpoint_stack[0]
        # cam2 = viewpoint_stack[1]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_feature_map = viewpoint_cam.semantic_feature
        if wild_params.langsplat_gt:
            feature_out_dim = 512
        elif isinstance(gt_feature_map, str):
            gt_feature_map = torch.load(gt_feature_map)
            feature_out_dim = gt_feature_map.shape[0]
        elif isinstance(gt_feature_map, dict):
            keys = gt_feature_map.keys()
            for key in keys:
                gt_feature_map[key] = gt_feature_map[key].cuda()
            feature_out_dim = gt_feature_map[key].shape[0]
        else:
            gt_feature_map = gt_feature_map.cuda()
            feature_out_dim = gt_feature_map.shape[0]
        return feature_out_dim

def get_images_names(viewpoint_stack, dataset):
    gt_path = Path(dataset.exclude_gt_images_from)
    if gt_path.exists():
        gt_images = []
        for dataset_gt_path in gt_path.rglob(f"{str(Path('*') / Path(dataset.source_path).name)}"):
            gt_images.extend([p.stem.replace('-gt', '') for p in dataset_gt_path.rglob('*-gt.jpg')])
        print(f'Excluding {len(set(gt_images))} images from training.')
        viewpoint_stack = [view for view in viewpoint_stack if Path(view.image_name).stem not in set(gt_images)]

    images_names = [viewpoint_cam.image_name for viewpoint_cam in viewpoint_stack]
    return images_names

def get_all_views(all_views, dataset):
    gt_path = Path(dataset.exclude_gt_images_from)
    if gt_path.exists():
        gt_images = []
        for dataset_gt_path in gt_path.rglob(f"{str(Path('*') / Path(dataset.source_path).name)}"):
            gt_images.extend([p.stem.replace('-gt', '') for p in dataset_gt_path.rglob('*-gt.jpg')])
        all_views = [view for view in all_views if Path(view.image_name).stem not in set(gt_images)]
    return all_views

def split_validation(all_views, wild_params):
    assert wild_params.split_validation < 1, f'split_validation should be in [0, 1). got {wild_params.split_validation}]'
    # assert wild_params.no_vis_enc_hash, 'split not implemented for in-the-wild visual encoding of images'
    np.random.seed(42)
    all_indexes = np.arange(len(all_views))
    shuffled = np.random.permutation(all_indexes)
    val_idx = int(len(shuffled) * wild_params.split_validation)
    val_indexes = np.sort(shuffled[:val_idx])
    train_indexes = np.sort(shuffled[val_idx:])

    train_views = [all_views[ind] for ind in train_indexes]
    val_views = [all_views[ind] for ind in val_indexes]
    val_gen = ViewGenerator(val_views, hard_negative=False,
                            dino_dir=wild_params.dino_dir, seg_dir=wild_params.segmentation_dir,
                            foreground_masks_dir=wild_params.foreground_masks_dir)
    return train_views, val_gen

def get_generators(scene, dataset, wild_params):
    all_views = get_all_views(scene.getTrainCameras().copy(), dataset)

    if wild_params.split_validation > 0:
        train_views, val_gen = split_validation(all_views, wild_params)
    else:
        train_views = all_views
        val_gen = None

    view_gen = ViewGenerator(train_views, hard_negative=False,
                             dino_dir=wild_params.dino_dir, seg_dir=wild_params.segmentation_dir,
                             foreground_masks_dir=wild_params.foreground_masks_dir)
    return view_gen, val_gen

class WarmupArgsHandler:
    def __init__(self, opt, wild_params):
        self.opt = opt
        self.semantic_feature_lr = opt.semantic_feature_lr
        self.iterations = opt.iterations
        self.warmup_iterations = wild_params.warmup_iterations
        self.wild_params = wild_params

    def warmup_args(self):
        self.opt.semantic_feature_lr = 0
        self.opt.iterations = self.warmup_iterations
        return self.opt
    
    def post_warmup_args(self):
        self.opt.semantic_feature_lr = self.semantic_feature_lr
        self.opt.iterations = self.iterations
        if self.wild_params.freeze_after_warmup:
            self.opt = freeze_nonfeature(self.opt, self.wild_params)
        return self.opt

def should_free_cache(memory_ratio=0.5):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    should_free = (allocated + reserved) / total > memory_ratio
    # print(f'allocated: {allocated}, reserved: {reserved}, total: {total}')
    return should_free

def _clip_grad_norm(params, clip_grad_norm, exclude_params=()):
    clip_fn = torch.nn.utils.clip_grad_norm_ if '0.3' != torch.__version__[:3] else torch.nn.utils.clip_grad_norm
    for p in params:
        if isinstance(p, dict):
            if p["name"] not in exclude_params:
                clip_fn(p['params'], clip_grad_norm)
        else:
            clip_fn(p, clip_grad_norm)


def match_rendered_and_gt_shapes(feature_map, gt_feature_map, 
                                 wild_params, gaussians, range_factor):
    size = (gt_feature_map.shape[1], gt_feature_map.shape[2])
    if wild_params.do_attention:
        feature_map = gaussians.wild_model.attention_downsample(
                    feature_map, size=size, range_factor=range_factor,
                    dropout=wild_params.attention_dropout)
    elif wild_params.interpolate_gt:
        feat_map_size = feature_map.shape[1:]
        gt_feature_map = F.interpolate(
                    gt_feature_map.unsqueeze(0), size=feat_map_size,
                    mode='bilinear', align_corners=False).squeeze(0)
    else:
        feature_map = F.interpolate(
                    feature_map.unsqueeze(0), size=size,
                    mode='bilinear', align_corners=False).squeeze(0)
                
    return feature_map, gt_feature_map

def get_foreground_mask(wild_params, viewpoint_cam):
    if wild_params.foreground_masks_dir:
        foreground = viewpoint_cam.foreground
        foreground = foreground.cuda() if foreground is not None else None
    else:
        foreground = None
    return foreground

def get_factors(wild_params, iteration):
    if iteration < 15000:
        t = min(iteration / 15000, 1.0)
        dino_factor = float(np.exp(t * np.log(0.1 * wild_params.dino_factor)
                                   + (1-t) * np.log(wild_params.dino_factor))) \
                        if wild_params.dino_factor else 0.0
        feature_factor = float(np.exp(t * np.log(1.0) + (1-t) * np.log(0.1)))
    else:
        dino_factor = wild_params.dino_factor
        feature_factor = 1.0

    clip_factor = 1.0
    return dino_factor, feature_factor, clip_factor
