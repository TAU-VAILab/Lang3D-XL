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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


def l1_loss(network_output, gt):
    if isinstance(gt, dict):
        keys = gt.keys()
        loss = torch.tensor(0)
        for key in keys:
            loss = loss + torch.abs((network_output[key] - gt[key])).mean()
        loss = loss / len(keys)
        return loss
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cosine_loss(network_output, gt):
    if isinstance(gt, dict):
        keys = gt.keys()
        loss = torch.tensor(0)
        for key in keys:
            loss = loss \
                + (1 - torch.sum(F.normalize(network_output[key], dim=0)
                                 * F.normalize(gt[key], dim=0), dim=0)).mean()
        loss = loss / len(keys)
        return loss
    return (1 - torch.sum(F.normalize(network_output, dim=0)
                          * F.normalize(gt, dim=0), dim=0)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

###
def tv_loss(feature_map):
    """
    Input:
    - feature_map: (C, H, W)
    Return:
    - total variation loss
    """
    tv_loss = ((feature_map[:, :, :-1] - feature_map[:, :, 1:])**2).sum() + ((feature_map[:, :-1, :] - feature_map[:, 1:, :])**2).sum()

    return tv_loss


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_pixels = np.prod(y_true.shape)
    return correct_predictions / total_pixels
    
    
def calculate_iou(y_true, y_pred, num_classes):
    iou = []
    for i in range(num_classes):
        true_labels = y_true == i
        predicted_labels = y_pred == i
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou)  


def variance_regularization_loss(feature_image, segmentation_map, mode='l2'):
    """
    Computes a regularization loss that penalizes high variance within the same label region,
    ignoring regions where segmentation_map == -1.
    
    Args:
        feature_image (torch.Tensor): Tensor of shape (C, H, W) representing the optimized feature image.
        segmentation_map (torch.Tensor): Tensor of shape (H, W) with integer labels (-1 for ignored regions).
    
    Returns:
        torch.Tensor: Regularization loss value.
    """
    C, H, W = feature_image.shape
    feature_image = feature_image.reshape(C, -1)  # Flatten spatial dimensions (C, H*W)
    segmentation_map = segmentation_map.reshape(-1)  # Flatten segmentation map (H*W)

    # Mask out ignored regions (label == -1)
    valid_mask = segmentation_map != -1
    feature_image = feature_image[:, valid_mask]  # Keep only valid pixels
    segmentation_map = segmentation_map[valid_mask]  # Keep only valid labels

    if segmentation_map.numel() == 0:  # If no valid regions exist, return zero loss
        return torch.tensor(0.0, device=feature_image.device)

    # Get unique valid labels and their indices
    unique_labels, indices = torch.unique(segmentation_map, return_inverse=True)

    # Compute mean per region using scatter_reduce
    region_pixel_counts = torch.bincount(indices, minlength=len(unique_labels))  # Count pixels per region
    region_means = torch.zeros(C, len(unique_labels), device=feature_image.device)  # Placeholder for means
    region_means.scatter_add_(1, indices.expand(C, -1), feature_image)  # Sum features per region
    region_means /= region_pixel_counts.unsqueeze(0).clamp(min=1)  # Normalize by region size

    # Compute variance per region
    diff = feature_image - region_means.gather(1, indices.expand(C, -1))  # (C, valid_pixels) - per-region mean
    if mode == 'l1':
        squared_diff = torch.abs(diff)
    else:
        squared_diff = diff**2
    region_variances = torch.zeros_like(region_means)  # Placeholder for variances
    region_variances.scatter_add_(1, indices.expand(C, -1), squared_diff)  # Sum squared diffs per region
    # region_variances /= region_pixel_counts.unsqueeze(0).clamp(min=1)  # Normalize by region size

    # Compute final loss as the mean variance across all regions
    loss = region_variances.sum() / region_pixel_counts.sum().clamp(min=1)

    return loss


def seg_var_loss(feature_map, segmentation, do_seg_reg, mode='l2', foreground=None):
    loss = torch.tensor([0], device=feature_map.device, dtype=feature_map.dtype)
    if not do_seg_reg:
        return loss
    
    with torch.no_grad():
        seg = F.interpolate(segmentation.to(feature_map.dtype).unsqueeze(0).unsqueeze(0),
                            size=(feature_map.shape[1], feature_map.shape[2]),
                            mode='nearest').squeeze(0).cuda()
        if foreground is not None:
            seg = seg * foreground -(1 - foreground)
    
    loss = loss + variance_regularization_loss(feature_map, seg, mode=mode)

    return loss


def dotp_withneighbors(fmap, window_size=3):
    """
    fmap: CxHxW
    """
    C,H,W = fmap.shape
    half_window = window_size // 2
    fmap_padded = F.pad(fmap, (half_window, half_window, half_window, half_window), mode='reflect')

    dotp_fmap = torch.zeros((window_size*window_size-1, H, W), dtype=fmap.dtype, device=fmap.device)
    dotp_ch = 0
    for i in range(-half_window, half_window + 1):
        for j in range(-half_window, half_window + 1):
            if i == 0 and j == 0:
                continue
            neighbor_fmap = fmap_padded[:, half_window + i: half_window + i + H, half_window + j: half_window + j +W]
            deno = torch.clamp(fmap.norm(dim=0)*neighbor_fmap.norm(dim=0), 1e-6) # (H, W)
            dotp_fmap[dotp_ch] = (fmap*neighbor_fmap).sum(dim=0)/deno # (H, W)
            dotp_ch+=1
    return dotp_fmap

def dotp_sim(fmap, fmap_ref, window_size=3):
    """
    fmap, fmap_ref: CxHxW
    """
    dotp_fmap_ref = dotp_withneighbors(fmap_ref.detach()).detach() # (window_size*window_size-1, H, W)
    dotp_fmap = dotp_withneighbors(fmap) # (window_size*window_size-1, H, W)
    return l1_loss(dotp_fmap, dotp_fmap_ref)


def calculate_feature_loss(wild_params, feature_map, gt_feature_map, foreground=None):
    if foreground is not None and wild_params.foreground_mode in ['all', 'features']:
        with torch.no_grad():
            foreground = F.interpolate(foreground.to(feature_map.dtype).unsqueeze(0),
                                size=(feature_map.shape[1], feature_map.shape[2]),
                                mode='nearest').squeeze(0)
        Ll1_feature = l1_loss(feature_map * foreground.unsqueeze(0), gt_feature_map * foreground.unsqueeze(0))
        cosine_loss_feature = cosine_loss(feature_map * foreground.unsqueeze(0), gt_feature_map * foreground.unsqueeze(0))
    else:
        Ll1_feature = l1_loss(feature_map, gt_feature_map)
        cosine_loss_feature = cosine_loss(feature_map, gt_feature_map)
    loss_feature = Ll1_feature
    return loss_feature, Ll1_feature, cosine_loss_feature

def compute_rgb_losses(image, gt_image, wild_params=None, foreground=None):
    if foreground is not None and wild_params.foreground_mode in ['all', 'rgb']:
        Ll1 = l1_loss(image * foreground.unsqueeze(0), gt_image * foreground.unsqueeze(0))
    else:
        Ll1 = l1_loss(image, gt_image)
    
    dssim_loss = ssim(image, gt_image)
    return Ll1, dssim_loss

def calc_dino_reg(feature_map, viewpoint_cam, do_dino, dino_size, do_dotp, mode='l1'):
    if do_dino:
        dino_gt = viewpoint_cam.dino_features.cuda()
        dino_map = feature_map[-dino_size:]
        feature_map = feature_map[:-dino_size]
        if 'l1' in mode:
            Ll1_dino = l1_loss(
                    F.interpolate(
                        dino_map.unsqueeze(0), size=(dino_gt.shape[1], dino_gt.shape[2]),
                        mode='bilinear', align_corners=False).squeeze(0),
                    dino_gt)
        if 'l2' in mode:
            Ll2_dino = l2_loss(
                    F.interpolate(
                        dino_map.unsqueeze(0), size=(dino_gt.shape[1], dino_gt.shape[2]),
                        mode='bilinear', align_corners=False).squeeze(0),
                    dino_gt)

        if do_dotp:
            L_dotp_sim_loss = dotp_sim(feature_map, dino_map)
        else:
            L_dotp_sim_loss = torch.tensor(
                [0], device=feature_map.device, dtype=feature_map.dtype)
        if isinstance(mode, str):
            L_dino = Ll1_dino if 'l1' in mode else Ll2_dino
        else:
            L_dino = {'l1': Ll1_dino, 'l2': Ll2_dino}
    elif isinstance(mode, str):
        L_dino = torch.tensor(
            [0], device=feature_map.device, dtype=feature_map.dtype)
        L_dotp_sim_loss = torch.tensor(
                [0], device=feature_map.device, dtype=feature_map.dtype)
    else:
        L_dino = {'l1': torch.tensor(
            [0], device=feature_map.device, dtype=feature_map.dtype),
            'l2': torch.tensor(
            [0], device=feature_map.device, dtype=feature_map.dtype)}
        L_dotp_sim_loss = torch.tensor(
            [0], device=feature_map.device, dtype=feature_map.dtype)
    
    return feature_map, L_dino, L_dotp_sim_loss
