#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import gsplat_render as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, VideoParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import render_path_spiral, filter_views_by_distance
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from utils.clip_utils import CLIPEditor
import yaml
from pathlib import Path
from models.networks import CNN_decoder, MLP_encoder
from wild.model import WildModel, WildFeatEncModel
from encoders.clip_pyramid.utils import get_gt_features


class VisPCA:
    def __init__(self) -> None:
        self.pca_stuff = None
    
    def __call__(self, feature, mask=None):
        fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
        fmap = nn.functional.normalize(fmap, dim=1)

        if self.pca_stuff is None:
            pca = sklearn.decomposition.PCA(3, random_state=42)
            f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).cpu().numpy()

            if mask is not None:
                f_samples = f_samples[mask.flatten().cpu().numpy() > 0]  # Select only masked features for PCA fitting

            if f_samples.shape[0] < 3:
                f_samples = f_samples.repeat(3, axis=0)
            transformed = pca.fit_transform(f_samples)
            feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
            feature_pca_components = torch.tensor(pca.components_).float().cuda()
            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            del f_samples
            self.pca_stuff = (feature_pca_mean, feature_pca_components,
                              feature_pca_postprocess_sub, feature_pca_postprocess_div)
        else:
            (feature_pca_mean, feature_pca_components,
             feature_pca_postprocess_sub, feature_pca_postprocess_div) = self.pca_stuff

        vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
        vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
        
        if mask is not None:
            vis_feature *= mask.reshape(fmap.shape[2], fmap.shape[3], 1).cpu()  # Apply mask back to visualization

        return vis_feature

def feature_visualize_saving(feature, mask=None):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).cpu().numpy()

    if mask is not None:
        f_samples = f_samples[mask.flatten().cpu().numpy() > 0]  # Select only masked features for PCA fitting

    if f_samples.shape[0] < 3:
        f_samples = f_samples.repeat(3, axis=0)
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
    
    if mask is not None:
        vis_feature *= mask.reshape(fmap.shape[2], fmap.shape[3], 1).cpu()  # Apply mask back to visualization

    return vis_feature


def parse_edit_config_and_text_encoding(edit_config):
    edit_dict = {}
    if edit_config is not None:
        with open(edit_config, 'r') as f:
            edit_config = yaml.safe_load(f)
            print(edit_config)
        objects = edit_config["edit"]["objects"]
        targets = edit_config["edit"]["targets"].split(",")
        edit_dict["positive_ids"] = [objects.index(t) for t in targets if t in objects]
        edit_dict["score_threshold"] = edit_config["edit"]["threshold"]
        
        # text encoding
        clip_editor = CLIPEditor()
        text_feature = clip_editor.encode_text([obj.replace("_", " ") for obj in objects])

        # setup editing
        op_dict = {}
        for operation in edit_config["edit"]["operations"].split(","):
            if operation == "extraction":
                op_dict["extraction"] = True
            elif operation == "deletion":
                op_dict["deletion"] = True
            elif operation == "color_func":
                op_dict["color_func"] = eval(edit_config["edit"]["colorFunc"])
            else:
                raise NotImplementedError
        edit_dict["operations"] = op_dict

        idx = edit_dict["positive_ids"][0]

    return edit_dict, text_feature, targets[idx]
        

def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, edit_config, speedup, is_wild=False, is_feat_enc=False,
               dino_dir=None, dino_size=0, save_features=False,
               decode_before_render=False, xyz_to_semantics=False):
    if edit_config != "no editing":
        edit_dict, text_feature, target = parse_edit_config_and_text_encoding(edit_config)

        edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
        edit_gts_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt")
        edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")
        edit_gt_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt_feature_map")

        makedirs(edit_render_path, exist_ok=True)
        makedirs(edit_gts_path, exist_ok=True)
        makedirs(edit_feature_map_path, exist_ok=True)
        makedirs(edit_gt_feature_map_path, exist_ok=True)
    
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
        gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
        saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
        if dino_dir:
            saved_dino_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_dino")
        #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
        decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
        
        if speedup:
            gt_feature_map = get_gt_features(views[0])[0]
            feature_out_dim = gt_feature_map.shape[0] + dino_size if dino_dir else gt_feature_map.shape[0]
            feature_in_dim = int(gt_feature_map.shape[0]/2)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True)
        makedirs(gt_feature_map_path, exist_ok=True)
        makedirs(saved_feature_path, exist_ok=True)
        if dino_dir:
            makedirs(saved_dino_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True) ###

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        fmap_size = [x // 2 for x in view.original_image.shape[1:]]
        render_pkg = render(view, gaussians, pipeline, background, is_wild=is_wild, is_feat_enc=is_feat_enc,
                            fmap_size=fmap_size,
                            decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics)

        gt = view.original_image[0:3, :, :]
        gt_feature_map = get_gt_features(view)[0]
        # torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + ".png")) 
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        ### depth ###
        depth = render_pkg["depth"]
        scale_nor = depth.max().item()
        depth_nor = depth / scale_nor
        depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        colormap = plt.get_cmap('jet')
        depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        depth_colored_rgb = depth_colored[:, :, :3]
        depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        # output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
        output_path = os.path.join(depth_path, view.image_name + ".png")
        depth_image.save(output_path)
        ##############

        # visualize feature map
        feature_map = render_pkg["feature_map"] 
        # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(feature_map.shape[1]//2, feature_map.shape[2]//2),
        #                             mode='bilinear', align_corners=True).squeeze(0) ###
        if speedup:
            feature_map = cnn_decoder(feature_map)
        
        if dino_dir:
            # dino_map = feature_map[-dino_size:]
            feature_map = feature_map[:-dino_size]
            dino_map = feature_map[-dino_size:]

        feature_map_vis = feature_visualize_saving(feature_map)
        # Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, view.image_name + "_feature_vis.png"))
        gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
        # Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, view.image_name + "_feature_vis.png"))

        if save_features:
            # save feature map
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            # torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, view.image_name + "_fmap_CxHxW.pt"))
            # torch.save(torch.tensor(dino_map).half(), os.path.join(saved_dino_path, view.image_name + "_fmap_CxHxW.pt"))
     

def render_video(model_path, iteration, views, gaussians, pipeline, background, edit_config,
                 speedup, video_params, is_wild=False, is_feat_enc=False,
                 dino_dir=None, dino_size=0, save_features=False,
                 decode_before_render=False, xyz_to_semantics=False, base_image=''): ###
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration), 'renders')
    feature_map_path = os.path.join(model_path, 'video', "ours_{}".format(iteration), 'feature_map')
    saved_feature_path = os.path.join(model_path, 'video', "ours_{}".format(iteration), "saved_feature")
    
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))

    if speedup:
        gt_feature_map = get_gt_features(views[0])[0]
        feature_out_dim = gt_feature_map.shape[0] + dino_size if dino_dir else gt_feature_map.shape[0]
        feature_in_dim = int(gt_feature_map.shape[0]/2)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))

    Path(render_path).mkdir(parents=True, exist_ok=True)
    Path(feature_map_path).mkdir(parents=True, exist_ok=True)
    Path(saved_feature_path).mkdir(parents=True, exist_ok=True)

    # views = filter_views_by_distance(views, keep_percent=80)

    view = [v for v in views if v.image_name == base_image]
    view = view[0] if view else views[0]

    render_poses = render_path_spiral(
        views, 
        rots=video_params.rots, 
        zrate=video_params.zrate, 
        focal=video_params.focal, 
        N=video_params.N,
        offset=(video_params.offset_x, video_params.offset_y, video_params.offset_z), 
        rad_alpha=(video_params.rad_alpha_x, video_params.rad_alpha_y, video_params.rad_alpha_z))
        # st_paul default: rots=2, zrate=0.02, focal=5, N=300, offset=(0, 0.2, -0.5), rad_alpha=(0.7, 0.5, 1)
        # milano: rots=1, zrate=0.02, focal=5, N=300, offset=(0, 0.1, -0.6), rad_alpha=(1.3, 1, 1)
        # blue: rots=2, zrate=0.02, focal=5, N=300, offset=(0, 0.0, -0.7), rad_alpha=(3, 0.5, 1)

    vis_pca = VisPCA()

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # size = (view.original_image.shape[2], view.original_image.shape[1])
    # final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        fmap_size = [x for x in view.original_image.shape[1:]] # [x // 2 for x in view.original_image.shape[1:]]
        render_pkg = render(view, gaussians, pipeline, background, is_wild=is_wild, is_feat_enc=is_feat_enc,
                            fmap_size=fmap_size,
                            decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics)
        rendering = torch.clamp(render_pkg["render"], min=0., max=1.)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        
        feature_map = render_pkg["feature_map"]

        if speedup:
            feature_map = cnn_decoder(feature_map)
        
        if dino_dir:
            feature_map = feature_map[:-dino_size]
            dino_map = feature_map[-dino_size:]

        feature_map_vis = vis_pca(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(feature_map_path, '{0:05d}'.format(idx) + ".png"))
        
        if save_features:
            # save feature map
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(), os.path.join(
                saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            
        #rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    # final_video.release()


def interpolate_matrices(start_matrix, end_matrix, steps):
        # Generate interpolation factors
        interpolation_factors = np.linspace(0, 1, steps)
        # Interpolate between the matrices
        interpolated_matrices = []
        for factor in interpolation_factors:
            interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
            interpolated_matrices.append(interpolated_matrix)
        return np.array(interpolated_matrices)


def multi_interpolate_matrices(matrix, num_interpolations):
    interpolated_matrices = []
    for i in range(matrix.shape[0] - 1):
        start_matrix = matrix[i]
        end_matrix = matrix[i + 1]
        for j in range(num_interpolations):
            t = (j + 1) / (num_interpolations + 1)
            interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
            interpolated_matrices.append(interpolated_matrix)
    return np.array(interpolated_matrices)


def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background, 
                       edit_config, speedup, multi_interpolate, num_views, is_wild=False, 
                       is_feat_enc=False, dino_dir=None, dino_size=0, save_features=False):
    if multi_interpolate:
        name = name + "_multi_interpolate"
    # make dirs
    if edit_config != "no editing":
        edit_dict, text_feature, target = parse_edit_config_and_text_encoding(edit_config)
        
        # edit
        edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
        edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")

        makedirs(edit_render_path, exist_ok=True)
        makedirs(edit_feature_map_path, exist_ok=True)
    else:
        # non-edit
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
        saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
        if dino_dir:
            saved_dino_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_dino")
        #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
        decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))

        if speedup:
            gt_feature_map = get_gt_features(views[0])[0]
            feature_out_dim = gt_feature_map.shape[0] + dino_size if dino_dir else gt_feature_map.shape[0]
            feature_in_dim = int(gt_feature_map.shape[0]/2)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
        makedirs(render_path, exist_ok=True)
        makedirs(feature_map_path, exist_ok=True)
        makedirs(saved_feature_path, exist_ok=True)
        if dino_dir:
            makedirs(saved_dino_path, exist_ok=True)

    view = views[0]
    
    # create novel poses
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose) 
    if not multi_interpolate:
        poses = interpolate_matrices(render_poses[0], render_poses[-1], num_views)
    else:
        poses = multi_interpolate_matrices(np.array(render_poses), 2)

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        # mlp encoder
        fmap_size = [x // 2 for x in view.original_image.shape[1:]]
        render_pkg = render(view, gaussians, pipeline, background, is_wild=is_wild, is_feat_enc=is_feat_enc,
                            fmap_size=fmap_size)
        # render_pkg = render(view, gaussians, pipeline, background, is_wild=is_wild, is_feat_enc=is_feat_enc) 

        gt_feature_map = get_gt_features(view)[0]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        # visualize feature map
        feature_map = render_pkg["feature_map"]
        # feature_map = F.interpolate(feature_map.unsqueeze(0), size=(feature_map.shape[1]//2, feature_map.shape[2]//2), mode='bilinear', align_corners=True).squeeze(0) ###
        if speedup:
            feature_map = cnn_decoder(feature_map)

        if dino_dir:
            dino_map = feature_map[-dino_size:]
            feature_map = feature_map[:-dino_size]

        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        if save_features:
            feature_map = feature_map.cpu().numpy().astype(np.float16)
            torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))
            # torch.save(torch.tensor(dino_map).half(),
            #        os.path.join(saved_dino_path, view.image_name + "_fmap_CxHxW.pt"))


def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, edit_config): 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (view.original_image.shape[2], view.original_image.shape[1])
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

    if edit_config != "no editing":
        edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
    
    render_poses = [(cam.R, cam.T) for cam in views]
    render_poses = []
    for cam in views:
        pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
        render_poses.append(pose)
    
    # create novel poses
    poses = interpolate_matrices(render_poses[0], render_poses[-1], 200) 

    # rendering process
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
    final_video.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                skip_train : bool, skip_test : bool, novel_view : bool, 
                video : bool , edit_config: str, novel_video : bool, multi_interpolate : bool,
                num_views : int, video_params : VideoParams = None, is_wild: bool = False, transient_thrashold: float = 1e-4,
                is_feat_enc: bool = False, is_hash: bool = False, do_attention: bool = False,
                dino_dir: str = None, dino_size: int = 0, save_features: bool = False,
                is_vis_hash: bool = True, bulk_on_device: bool = False,
                decode_before_render: bool = False, xyz_to_semantics: bool = False,
                base_image: str = ''): 
    
    dino_size = dino_size if dino_dir else 0
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, is_wild)
        dataset.is_feat_enc = is_feat_enc
        dataset.gauss_feat_size = 16 if not is_hash else 3
        dataset.xyz_to_semantics = xyz_to_semantics
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, is_train=False,
                      bulk_on_device=bulk_on_device)

        if is_wild:
            viewpoint_stack = scene.getTrainCameras().copy()
            from pathlib import Path
            gt_path = Path(dataset.exclude_gt_images_from) \
                if hasattr(dataset, 'exclude_gt_images_from') else Path('')
            if gt_path.exists():
                gt_images = []
                for dataset_gt_path in gt_path.rglob(f"{str(Path('*') / Path(dataset.source_path).name)}"):
                    gt_images.extend([p.stem.replace('-gt', '') for p in dataset_gt_path.rglob('*-gt.jpg')])
                viewpoint_stack = [view for view in viewpoint_stack if Path(view.image_name).stem not in set(gt_images)]


            images_names = [viewpoint_cam.image_name for viewpoint_cam in viewpoint_stack]
            
            viewpoint_cam = viewpoint_stack.pop(0)
            gt_feature_map = viewpoint_cam.semantic_feature
            if isinstance(gt_feature_map, str):
                gt_feature_map = torch.load(gt_feature_map).cuda()
                feature_out_dim = 0
            elif isinstance(gt_feature_map, dict):
                keys = gt_feature_map.keys()
                for key in keys:
                    gt_feature_map[key] = gt_feature_map[key].cuda()
                feature_out_dim = gt_feature_map[key].shape[0]
            else:
                gt_feature_map = gt_feature_map.cuda()
                feature_out_dim = gt_feature_map.shape[0]
            
            feature_dim = feature_out_dim #  feature_out_dim // 2 if dataset.speedup else feature_out_dim

            gaussians.wild_model = WildFeatEncModel(
                images_names=images_names,
                num_features=feature_dim,
                num_dino_features=dino_size,
                render_semantic=True,
                vucabulary_size=dataset.gauss_feat_size,
                is_hash=is_hash,
                do_attention=do_attention,
                is_vis_hash=is_vis_hash).to('cuda')

            gaussians.wild_model.load_state_dict(torch.load(
                f'{scene.model_path}/wild_model_chkpnt{scene.loaded_iter}.pth'))
            gaussians.wild_model.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                        gaussians, pipeline, background, edit_config, dataset.speedup, is_wild=is_wild, is_feat_enc=is_feat_enc,
                        dino_dir=dino_dir, dino_size=dino_size, save_features=save_features,
                        decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics)
        
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                        gaussians, pipeline, background, edit_config, dataset.speedup, is_wild=is_wild, is_feat_enc=is_feat_enc,
                        dino_dir=dino_dir, dino_size=dino_size, save_features=save_features,
                        decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics)

        if novel_view:
             render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(),
                                gaussians, pipeline, background, edit_config, dataset.speedup, multi_interpolate,
                                num_views, is_wild=is_wild, is_feat_enc=is_feat_enc,
                                dino_dir=dino_dir, dino_size=dino_size, save_features=save_features,
                        decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics)

        if video:
             render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config,
                          dataset.speedup, video_params, is_wild=is_wild, is_feat_enc=is_feat_enc,
                          dino_dir=dino_dir, dino_size=dino_size, save_features=save_features,
                          decode_before_render=decode_before_render, xyz_to_semantics=xyz_to_semantics,
                          base_image=base_image)

        if novel_video:
             render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    video = VideoParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--novel_view", action="store_true") ###
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true") ###
    parser.add_argument("--novel_video", action="store_true") ###
    parser.add_argument('--edit_config', default="no editing", type=str)
    parser.add_argument("--multi_interpolate", action="store_true") ###
    parser.add_argument("--num_views", default=200, type=int)
    parser.add_argument("--wild", action="store_true")
    parser.add_argument("--transient_thrashold", type=float, default=1e-4)
    parser.add_argument("--feat_enc", action="store_true")
    parser.add_argument("--feat_enc_hash", action="store_true")
    parser.add_argument('--do_attention', action="store_true")
    parser.add_argument("--dino_dir", type=str, default='')
    parser.add_argument("--dino_size", type=int, default=384)
    parser.add_argument('--save_features', type=bool, default=True)
    parser.add_argument("--no_vis_enc_hash", action="store_true")
    parser.add_argument("--decode_before_render", action="store_true")
    parser.add_argument("--xyz_to_semantics", action="store_true")
    parser.add_argument("--bulk_on_device", default=1, type=int)
    parser.add_argument("--base_image", default='', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.feat_enc = True if args.feat_enc_hash else args.feat_enc
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, args.novel_view, 
                args.video, args.edit_config, args.novel_video, args.multi_interpolate,
                args.num_views, video.extract(args), is_wild=args.wild, transient_thrashold=args.transient_thrashold,
                is_feat_enc=args.feat_enc, is_hash=args.feat_enc_hash, do_attention=args.do_attention,
                dino_dir=args.dino_dir, dino_size=args.dino_size, save_features=args.save_features, 
                is_vis_hash=not args.no_vis_enc_hash, bulk_on_device=args.bulk_on_device,
                decode_before_render=args.decode_before_render, xyz_to_semantics=args.xyz_to_semantics,
                base_image=args.base_image) ###