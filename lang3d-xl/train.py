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
from utils.loss_utils import l1_loss, cosine_loss, l2_loss, calculate_feature_loss, \
     compute_rgb_losses, seg_var_loss, calc_dino_reg
from gaussian_renderer import gsplat_render as render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, WildParams
from time import time

import gc
import torch.nn.functional as F
from models.networks import CNN_decoder
from wild.model import WildFeatEncModel
from encoders.clip_pyramid.utils import get_clip_visualizer, get_gt_features
from utils.training_utils import check_wild_params, get_out_dim, \
    check_dataset_params, get_generators, get_images_names, WarmupArgsHandler, \
    should_free_cache, _clip_grad_norm, match_rendered_and_gt_shapes, get_foreground_mask, \
    get_factors
from utils.metric_utils import compute_psnr
from utils.logging_utils import JsonLogger, prepare_output_and_logger, training_report

def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, debug_from, wild_params: WildParams):
    
    wild_params = check_wild_params(wild_params)
    
    torch.autograd.set_detect_anomaly(True)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    json_logger = JsonLogger(dataset, opt, pipe, wild_params)
    json_logger.write()

    gaussians = GaussianModel(dataset.sh_degree, wild_params.wild,
                              do_wild_densification=wild_params.do_wild_densification)
    dataset = check_dataset_params(dataset, wild_params)
    scene = Scene(
        dataset, gaussians, filter_images_if_big=False, add_sky=wild_params.add_sky,
        bulk_on_device=wild_params.bulk_on_device, langsplat_gt=wild_params.langsplat_gt)


    # 2D semantic feature map CNN decoder
    feature_out_dim = get_out_dim(scene, wild_params)

    # speed up for SAM
    if dataset.speedup:
        feature_in_dim = int(feature_out_dim / 2)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim + wild_params.dino_size)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)
    else:
        cnn_decoder = None
        cnn_decoder_optimizer = None

    if wild_params.wild:
        images_names = get_images_names(scene.getTrainCameras().copy(), dataset)

        gaussians.wild_model = WildFeatEncModel(
            images_names=images_names,
            num_features=feature_out_dim,
            num_dino_features=wild_params.dino_size, # if not dataset.speedup else wild_params.dino_size//2,
            render_semantic=(opt.semantic_feature_lr > 0),
            vucabulary_size=dataset.gauss_feat_size,
            is_hash=wild_params.feat_enc_hash,
            do_attention=wild_params.do_attention,
            is_vis_hash=wild_params.vis_enc_hash,
            max_win_scale_factor=wild_params.max_win_scale_factor,
            mlp_feat_decay=wild_params.mlp_feat_decay).to('cuda')
        
        gaussians.wild_model.train()

    gaussians.training_setup(opt)

    if wild_params.start_checkpoint:
        (model_params, first_iter) = torch.load(wild_params.start_checkpoint)
        first_iter = 0
        gaussians.restore(model_params, opt, use_model_optimzer_state=False)
    
    if wild_params.wild_checkpoint:
        gaussians.wild_model.load_state_dict(torch.load(wild_params.wild_checkpoint), strict=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    view_gen, val_gen = get_generators(scene, dataset, wild_params)
    
    clip_vis = get_clip_visualizer(dataset, wild_params)

    ema_loss_for_log = 0.0
    

    if wild_params.warmup:
        warmup_args_handler = WarmupArgsHandler(opt, wild_params)
        opt = warmup_args_handler.warmup_args()
        gaussians.training_setup(opt)
        duration = training_loop(
            dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
            debug_from, first_iter, wild_params,
            tb_writer, gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start,
            iter_end, view_gen, clip_vis, ema_loss_for_log)

        json_logger.log_timings(duration, duration / opt.iterations, 'warmup')
        
        opt = warmup_args_handler.post_warmup_args()
        gaussians.training_setup(opt)
        warmup_iterations = wild_params.warmup_iterations
    else:
        warmup_iterations = 0
    
    duration = training_loop(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
                  debug_from, first_iter, wild_params,
                  tb_writer, gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start,
                  iter_end, view_gen, clip_vis, ema_loss_for_log, start_log_iteration=warmup_iterations)

    json_logger.log_timings(duration, duration / opt.iterations, 'training')

    losses, duration, duration_per_image = eval_loop(
        dataset, opt, pipe, wild_params, first_iter, gaussians, cnn_decoder, background, view_gen)
    
    json_logger.log_result('losses', losses)
    json_logger.log_timings(duration, duration_per_image, 'evaluation')

    if val_gen is not None:
        losses, duration, duration_per_image = eval_loop(
            dataset, opt, pipe, wild_params, first_iter, gaussians, cnn_decoder, background, val_gen)
        json_logger.log_result('losses_validation', losses)

    json_logger.write()

def eval_loop(dataset, opt, pipe, wild_params: WildParams, first_iter, gaussians, cnn_decoder, background, view_gen):
    
    data_length = len(view_gen.viewpoint_stack)
    progress_bar = tqdm(range(data_length), desc="Evaluation progress")
    first_iter += 1

    with torch.no_grad():
        time_before = time()
        losses = {'vis_L1': 0.0, 'vis_L2': 0.0,
                  'feat_L1': 0.0, 'feat_L2': 0.0, 'feature_cos_sim_loss': 0.0,
                  'dino_L1': 0.0, 'dino_L2': 0.0,
                  'feat_psnr': 0.0, 'feat_fpsnr': 0.0}
        
        for index in progress_bar:
            viewpoint_cam = view_gen.viewpoint_stack[index]

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_semantic=(opt.semantic_feature_lr > 0),
                                is_wild=wild_params.wild, is_feat_enc=wild_params.feat_enc,
                                decode_before_render=wild_params.decode_before_render,
                                xyz_to_semantics=wild_params.xyz_to_semantics)
            

            feature_map, image = render_pkg["feature_map"], render_pkg["render"]
            
            if dataset.speedup:
                feature_map = cnn_decoder(feature_map)

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            losses['vis_L1'] += l1_loss(image, gt_image).item()
            losses['vis_L2'] += l2_loss(image, gt_image).item()

            if wild_params.langsplat_gt and feature_map is not None:
                gt_feature_map, language_feature_mask = viewpoint_cam.get_language_feature(
                    language_feature_dir=wild_params.langsplat_dir,
                    seg_dir=wild_params.segmentation_dir,
                    feature_level=wild_params.langsplat_level)
                range_factor = 1.0
                gt_feature_map = gt_feature_map.to(torch.float32)
                language_feature_mask = language_feature_mask.to(torch.float32)

                feature_map, L_dino, L_dotp_sim_loss = calc_dino_reg(
                    feature_map, viewpoint_cam, bool(wild_params.dino_dir), 
                    wild_params.dino_size, do_dotp=False, mode=('l1', 'l2'))
                losses['dino_L1'] += L_dino['l1'].item()
                losses['dino_L2'] += L_dino['l2'].item()

                feature_map, gt_feature_map = match_rendered_and_gt_shapes(
                    feature_map, gt_feature_map, wild_params, gaussians, range_factor)
                    
                losses['feat_L1'] += l1_loss(feature_map * language_feature_mask,
                                             gt_feature_map * language_feature_mask).item()
                losses['feat_L2'] += l2_loss(feature_map * language_feature_mask,
                                             gt_feature_map * language_feature_mask).item()
                losses['feature_cos_sim_loss'] += cosine_loss(feature_map * language_feature_mask,
                                                              gt_feature_map * language_feature_mask).item()
                losses['feat_psnr'] += compute_psnr(feature_map * language_feature_mask,
                                                    gt_feature_map * language_feature_mask)
                

            elif feature_map is not None:
                gt_feature_map, range_factor = get_gt_features(
                    viewpoint_cam, opt.features_scales_mode, avg_p=1.1)

                feature_map, L_dino, L_dotp_sim_loss = calc_dino_reg(
                    feature_map, viewpoint_cam, bool(wild_params.dino_dir), 
                    wild_params.dino_size, do_dotp=False, mode=('l1', 'l2'))
                losses['dino_L1'] += L_dino['l1'].item()
                losses['dino_L2'] += L_dino['l2'].item()
                
                feature_map, gt_feature_map = match_rendered_and_gt_shapes(
                    feature_map, gt_feature_map, wild_params, gaussians, range_factor)

                losses['feat_L1'] += l1_loss(feature_map, gt_feature_map).item()
                losses['feat_L2'] += l2_loss(feature_map, gt_feature_map).item()
                losses['feature_cos_sim_loss'] += cosine_loss(feature_map, gt_feature_map).item()
                losses['feat_psnr'] += compute_psnr(feature_map, gt_feature_map)
                

        duration = time() - time_before
        for key in losses.keys():
            losses[key] /= data_length
        
    return losses, duration, duration / data_length

def training_loop(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
                  debug_from, first_iter, wild_params: WildParams,
                  tb_writer, gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start,
                  iter_end, view_gen, clip_vis, ema_loss_for_log, start_log_iteration=0):
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    time_before = time()
    for iteration in range(first_iter, opt.iterations + 1):
        if wild_params.do_wild_only_batch and iteration == wild_params.wild_only_interval:
            print('wild only batch')
            for wild_iteration in range(wild_params.wild_only_iterations):
                run_rgb_only_batch(dataset, opt, pipe,
                        debug_from, wild_params,
                        gaussians, cnn_decoder_optimizer, background, iter_end,
                        view_gen, wild_iteration)
        if wild_params.max_win_scale_factor == -2:
            gaussians.wild_model.max_win_scale_factor = \
                1.0 if iteration < opt.iterations / 2 \
                    else 1 + 5 * (2 * (iteration / opt.iterations - 0.5))**2
        try:
            if not wild_params.bulk_on_device and should_free_cache():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            run_train_batch(
                dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
                debug_from, tb_writer, wild_params,
                gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start, iter_end,
                view_gen, clip_vis, ema_loss_for_log, start_log_iteration, progress_bar, iteration)
        except:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            run_train_batch(
                dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
                debug_from, tb_writer, wild_params,
                gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start, iter_end,
                view_gen, clip_vis, ema_loss_for_log, start_log_iteration, progress_bar, iteration)
        
    duration = time() - time_before
    return duration

def run_train_batch(dataset, opt, pipe, testing_iterations, saving_iterations,
                    checkpoint_iterations, debug_from, tb_writer, wild_params: WildParams,
                    gaussians, scene, cnn_decoder, cnn_decoder_optimizer, background, iter_start, iter_end, view_gen, clip_vis, ema_loss_for_log, start_log_iteration, progress_bar, iteration):
    if wild_params.hard_neg and iteration >= 1000 and iteration % 1000 == 0:
        view_gen.hard_negative = True
        view_gen.sort()

    iter_start.record()

    gaussians.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    viewpoint_cam, view_indx = view_gen()

    # Render
    if (iteration - 1) == debug_from:
        pipe.debug = True
    render_pkg = render(
        viewpoint_cam, gaussians, pipe, background, render_semantic=(opt.semantic_feature_lr > 0),
        is_wild=wild_params.wild, is_feat_enc=wild_params.feat_enc,
        decode_before_render=wild_params.decode_before_render,
        xyz_to_semantics=wild_params.xyz_to_semantics
    )

    feature_map, image, viewspace_point_tensor, visibility_filter, radii = \
        render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        
    if dataset.speedup and opt.semantic_feature_lr > 0:
        feature_map = cnn_decoder(feature_map)
    
    # Loss
    gt_image = viewpoint_cam.original_image.cuda()
    
    foreground = get_foreground_mask(wild_params, viewpoint_cam)

    Ll1, dssim_loss = compute_rgb_losses(image, gt_image, wild_params, foreground)

    if wild_params.langsplat_gt and opt.semantic_feature_lr > 0:

        gt_feature_map, language_feature_mask = viewpoint_cam.get_language_feature(
            language_feature_dir=wild_params.langsplat_dir,
            seg_dir=wild_params.segmentation_dir,
            feature_level=wild_params.langsplat_level)
        range_factor = 1.0
        gt_feature_map = gt_feature_map.to(torch.float32)
        language_feature_mask = language_feature_mask.to(torch.float32)

        feature_map, Ll1_dino, L_dotp_sim_loss = calc_dino_reg(
            feature_map, viewpoint_cam, bool(wild_params.dino_dir), wild_params.dino_size,
            do_dotp=opt.do_dotp_sim_loss and iteration > 2500)
            
        Lvar_seg = torch.tensor([0]).type_as(Ll1.data)
        
        size = (gt_feature_map.shape[1], gt_feature_map.shape[2])

        if wild_params.do_attention:
            feature_map = gaussians.wild_model.attention_downsample(
                    feature_map, size=size, range_factor=range_factor, dropout=wild_params.attention_dropout)
    
        feature_map = F.interpolate(
                feature_map.unsqueeze(0), size=size,
                mode='bilinear', align_corners=False).squeeze(0)
        
        loss_feature, Ll1_feature, cosine_loss_feature = calculate_feature_loss(
            wild_params, feature_map * language_feature_mask,
            gt_feature_map * language_feature_mask)

        dino_factor, feature_factor, clip_factor = get_factors(wild_params, iteration)

        if wild_params.dino_dir:
            loss_feature = clip_factor * loss_feature + dino_factor * Ll1_dino

    elif opt.semantic_feature_lr > 0:
        if foreground is not None and wild_params.foreground_mode in ['all', 'features']:
            with torch.no_grad():
                foreground = F.interpolate(foreground.to(feature_map.dtype).unsqueeze(0).unsqueeze(0),
                                size=(feature_map.shape[1], feature_map.shape[2]),
                                mode='nearest').squeeze(0)
        else:
            foreground = None
            
        avg_p = 1.1
        num_avg = 0
        gt_feature_map, range_factor = get_gt_features(
                viewpoint_cam, opt.features_scales_mode, avg_p=avg_p, num_avg=num_avg,
                max_feat_scale=wild_params.max_feat_scale, min_feat_scale=wild_params.min_feat_scale)

        feature_map, Ll1_dino, L_dotp_sim_loss = calc_dino_reg(
            feature_map, viewpoint_cam, bool(wild_params.dino_dir), wild_params.dino_size,
            do_dotp=opt.do_dotp_sim_loss and iteration > 2500)
        
        if foreground is not None and wild_params.foreground_mode in ['all', 'features']:
            with torch.no_grad():
                foreground2 = F.interpolate(foreground.to(feature_map.dtype).unsqueeze(0),
                                size=(feature_map.shape[1], feature_map.shape[2]),
                                mode='nearest').squeeze(0).squeeze(0)
            feature_map = feature_map * foreground2.unsqueeze(0)
        
        Lvar_seg = seg_var_loss(
                feature_map, viewpoint_cam.segmentation, bool(wild_params.segmentation_dir),
                wild_params.seg_loss_mode, foreground=foreground)

        feature_map, gt_feature_map = match_rendered_and_gt_shapes(
            feature_map, gt_feature_map, wild_params, gaussians, range_factor)

        loss_feature, Ll1_feature, cosine_loss_feature = calculate_feature_loss(
            wild_params, feature_map, gt_feature_map, foreground)
        
        dino_factor, feature_factor, clip_factor = get_factors(wild_params, iteration)

        if wild_params.dino_dir:
            
            loss_feature = clip_factor * loss_feature + dino_factor * Ll1_dino

            if opt.do_dotp_sim_loss:
                loss_feature = loss_feature + 0.01 * L_dotp_sim_loss

    else:
        loss_feature = torch.Tensor([0]).type_as(Ll1.data)
        Ll1_feature = torch.Tensor([0]).type_as(Ll1.data)
        cosine_loss_feature = torch.Tensor([0]).type_as(Ll1.data)
        Ll1_dino = torch.Tensor([0]).type_as(Ll1.data)
        Lvar_seg = torch.Tensor([0]).type_as(Ll1.data)
        feature_factor = 0.0

    ddsim_warmup = -1
    dssim_factor = opt.lambda_dssim if iteration > ddsim_warmup else 0.5 - (0.5 - opt.lambda_dssim) * iteration / ddsim_warmup
    
    loss = (1.0 - dssim_factor) * Ll1 + dssim_factor * (1.0 - dssim_loss) \
            + feature_factor * loss_feature + wild_params.seg_loss_weight * Lvar_seg

    view_gen.indxes_loss[view_indx] = loss.item()

    loss.backward()
    iter_end.record()

    if wild_params.clip_grad_norm > 0:
        _clip_grad_norm(gaussians.optimizer.param_groups, args.clip_grad_norm,
                        gaussians.wild_params_names)

    with torch.no_grad():
        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Log and save
        training_report(tb_writer, iteration, Ll1, Ll1_feature, cosine_loss_feature,
                            Ll1_dino, loss, l1_loss, Lvar_seg,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                            (pipe, background), dataset.speedup, cnn_decoder if dataset.speedup else None,
                            clip_vis, wild_params, start_log_iteration=start_log_iteration) 
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)
            print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
            if dataset.speedup:
                torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
            if wild_params.wild:
                torch.save(gaussians.wild_model.state_dict(), f'{scene.model_path}/wild_model_chkpnt{iteration}.pth')
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth") 

        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()
            

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if dataset.speedup:
                cnn_decoder_optimizer.step()
                cnn_decoder_optimizer.zero_grad(set_to_none = True)

        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    with torch.no_grad():        
        if network_gui.conn == None:
            network_gui.try_connect(dataset.render_items)
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                if custom_cam != None:
                    render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer,
                                        is_wild=wild_params.wild, is_feat_enc=wild_params.feat_enc,
                                        decode_before_render=wild_params.decode_before_render,
                                        xyz_to_semantics=wild_params.xyz_to_semantics)   
                    net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                    # raise e
                network_gui.conn = None

def run_rgb_only_batch(dataset, opt, pipe, debug_from, wild_params: WildParams,
                        gaussians, cnn_decoder_optimizer, background, iter_end,
                        view_gen, iteration):
    if wild_params.hard_neg and iteration >= 1000 and iteration % 1000 == 0:
        view_gen.hard_negative = True
        view_gen.sort()

    viewpoint_cam, view_indx = view_gen()

        # Render
    if (iteration - 1) == debug_from:
        pipe.debug = True
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_semantic=False,
                            is_wild=wild_params.wild, is_feat_enc=wild_params.feat_enc,
                            decode_before_render=wild_params.decode_before_render,
                            xyz_to_semantics=wild_params.xyz_to_semantics)
        

    image = render_pkg["render"]

    # Loss
    gt_image = viewpoint_cam.original_image.cuda()

    Ll1, dssim_loss = compute_rgb_losses(image, gt_image)

    ddsim_warmup = -1
    dssim_factor = opt.lambda_dssim if iteration > ddsim_warmup else 0.5 - (0.5 - opt.lambda_dssim) * iteration / ddsim_warmup
    loss = (1.0 - dssim_factor) * Ll1 + dssim_factor * (1.0 - dssim_loss) 

    view_gen.indxes_loss[view_indx] = loss.item()

    loss.backward()
    iter_end.record()

    if wild_params.clip_grad_norm > 0:
        _clip_grad_norm(gaussians.optimizer.param_groups, args.clip_grad_norm,
                        gaussians.wild_params_names)

    with torch.no_grad():
        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)
        if dataset.speedup:
            cnn_decoder_optimizer.step()
            cnn_decoder_optimizer.zero_grad(set_to_none = True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    wp = WildParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, 
                        default=[10, 100, 500, 700] + list(range(1000, 10000, 200)) + list(range(10000, 1000000, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 70_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.feat_enc = True if args.feat_enc_hash else args.feat_enc
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.debug_from,
             wp.extract(args))

    # All done
    print("\nTraining complete.")
