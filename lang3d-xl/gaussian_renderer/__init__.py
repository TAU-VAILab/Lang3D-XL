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
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from random import randint
import torch.nn.functional as F


def decode_features(feature_map, gaussians, size=None):
    if size is None:
        new_height, new_width = get_random_size(feature_map)
    else:
        new_height, new_width = size
    feature_map = F.interpolate(feature_map.unsqueeze(0), size=(new_height, new_width),
                                mode='nearest').squeeze(0)
    map_shape = feature_map.shape
    feature_map = feature_map.permute(2, 1, 0).reshape(-1, map_shape[0])
    feature_map = gaussians.wild_model.decode_features(feature_map)
    feature_map = feature_map.reshape((*map_shape[-1:0:-1], feature_map.shape[-1])).permute(2, 1, 0)

    return feature_map

def get_random_size(feature_map):
    height, width = feature_map.shape[1:]
    ratio = height / width
    new_width = randint(80, 120)
    new_height = int(ratio * new_width)
    return new_height, new_width


from gsplat.project_gaussians import project_gaussians
from gsplat.sh import spherical_harmonics
from gsplat.rasterize import rasterize_gaussians
def gsplat_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                  scaling_modifier=1.0, override_color=None, render_semantic=True,
                  is_wild=False, is_feat_enc=False, decode_before_render=False, xyz_to_semantics=False, fmap_size=None):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    img_height = int(viewpoint_camera.image_height)
    img_width = int(viewpoint_camera.image_width)

    rgbs, pc_opacities, semantic_features = get_rgb_opacity_and_features(viewpoint_camera, pc, is_wild, xyz_to_semantics)

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means3d=pc.get_xyz,
        scales=pc.get_scaling,
        glob_scale=scaling_modifier,
        quats=pc.get_rotation,
        viewmat=viewpoint_camera.world_view_transform.T,
        # projmat=viewpoint_camera.full_projection.T,
        fx=focal_length_x,
        fy=focal_length_y,
        cx=img_width / 2.,
        cy=img_height / 2.,
        img_height=img_height,
        img_width=img_width,
        block_width=16,
    )

    try:
        xys.retain_grad()
    except:
        pass

    # opacities = pc.get_opacity
    # if self.anti_aliased is True:
    #     opacities = opacities * comp[:, None]

    def rasterize_features(input_features, bg, distilling: bool = False):
        opacities = pc_opacities
        if distilling is True:
            opacities = opacities.detach()
        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            input_features,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=16,
            background=bg,
            return_alpha=False,
        ).permute(2, 0, 1)

    rgb = rasterize_features(rgbs, bg_color)
    depth = rasterize_features(depths.unsqueeze(-1).repeat(1, 3), torch.zeros((3,), dtype=torch.float, device=bg_color.device))

    if render_semantic:
        if is_feat_enc and not decode_before_render:
            bg_color = torch.zeros(semantic_features.shape[-1],
                                   dtype=torch.float, device=bg_color.device)
            feature_map = rasterize_features(
                semantic_features,
                bg_color,
                distilling=True,
            )
            try:
                feature_map = decode_features(feature_map, pc, size=fmap_size)
            except:
                print("\n", "fmap_size:", fmap_size, "feature_map", feature_map.shape, "rgb", rgb.shape, "\n")
                raise
        elif is_feat_enc and decode_before_render:
            semantic_features = pc.wild_model.decode_features(semantic_features)

            output_semantic_feature_map_list = []
            chunk_size = 32
            bg_color = torch.zeros((chunk_size,), dtype=torch.float, device=bg_color.device)
            for i in range(semantic_features.shape[-1] // chunk_size):
                start = i * chunk_size
                output_semantic_feature_map_list.append(rasterize_features(
                    semantic_features[..., start:start + chunk_size],
                    bg_color,
                    distilling=True,
                ))
            feature_map = torch.concat(output_semantic_feature_map_list, dim=0)
        else:
            output_semantic_feature_map_list = []
            chunk_size = 32
            bg_color = torch.zeros((chunk_size,), dtype=torch.float, device=bg_color.device)
            for i in range(semantic_features.shape[-1] // chunk_size):
                start = i * chunk_size
                output_semantic_feature_map_list.append(rasterize_features(
                    semantic_features[..., start:start + chunk_size],
                    bg_color,
                    distilling=True,
                ))
            feature_map = torch.concat(output_semantic_feature_map_list, dim=0)
    else:
        feature_map = None

    return {
        "render": rgb,
        "depth": depth[:1],
        'feature_map': feature_map,
        "viewspace_points": xys,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

def get_rgb_opacity_and_features(viewpoint_camera, pc, is_wild, xyz_to_semantics=False):
    viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
    # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
    rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

    pc_opacities = pc.get_opacity
    semantic_features = pc.get_semantic_feature.squeeze(1) if not xyz_to_semantics else pc.get_xyz
    if is_wild:
        rgbs, delta_opacity, semantic_features = pc.wild_model(
            pc.get_xyz, rgbs, semantic_features, viewpoint_camera.image_name)
        pc_opacities = pc_opacities - delta_opacity
    return rgbs, pc_opacities, semantic_features
