import json
from pathlib import Path
import os
import uuid
from argparse import Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch
import torch.nn.functional as F
import sys
src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(src_dir))
from scene import Scene
from arguments import WildParams
from encoders.clip_pyramid.utils import get_gt_features
from render import feature_visualize_saving
from utils.image_utils import psnr

class JsonLogger:
    def __init__(self, dataset, opt, pipe, wild_params):
        self.log_path = str(Path(dataset.model_path) / 'training_args.json')
        self.logs = {'dataset': vars(dataset), 'opt': vars(opt),
                     'pipe': vars(pipe), 'wild': vars(wild_params)}
        
    def write(self):
        with open(self.log_path, 'w') as file:
            json.dump(self.logs, file, indent=4)
    
    def log_timings(self, total_time, time_per_epoch, mode='training'):
        self.log_result(f'total {mode} time', total_time)
        self.log_result(f'{mode} time per epoch', time_per_epoch)

    def log_result(self, key, value):
        if 'results' not in self.logs:
            self.logs['results'] = {}
        self.logs['results'][key] = value


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_feature, cosine_loss_feature,
                    Ll1_dino, loss, l1_loss, Lvar_seg, elapsed, 
                    testing_iterations, scene : Scene, renderFunc, renderArgs, speedup,
                    cnn_decoder, clip_vis, wild_params: WildParams, start_log_iteration=0):
    scene.gaussians.wild_model.eval()
    params_lrs = {param_group["name"]: param_group['lr'] 
                  for param_group in scene.gaussians.optimizer.param_groups
                  if param_group["name"] in scene.gaussians.wild_params_names + ["f_dc", "f_rest", "semantic_feature"]}
    log_iteration = iteration + start_log_iteration
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), log_iteration)
        tb_writer.add_scalar('train_loss_patches/loss_feature_l1', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/loss_feature_cosine', cosine_loss_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), log_iteration)
        tb_writer.add_scalar('train_loss_patches/loss_dino_l1', Ll1_dino.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/loss_Lvar_seg', Lvar_seg.item(), iteration) 
        tb_writer.add_scalar('iter_time', elapsed, log_iteration)
        for param_name, lr in params_lrs.items():
            tb_writer.add_scalar(f'learning_rates/{param_name}', lr, log_iteration)

    # Report test and samples of training set
    specific_views = []
    for idx, view in enumerate(scene.getTrainCameras()):
        if view.image_name in ['0491', '0777', '0320']:
            specific_views.append(idx)

    if iteration in testing_iterations:
        # torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                                             for idx in list(range(5, 30, 5)) + specific_views]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                Ll1_feature = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                            is_wild=wild_params.wild, is_feat_enc=wild_params.feat_enc,
                                            decode_before_render=wild_params.decode_before_render,
                                            xyz_to_semantics=wild_params.xyz_to_semantics)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if wild_params.langsplat_gt:
                        gt_feature_map, language_feature_mask = viewpoint.get_language_feature(
                            language_feature_dir=wild_params.langsplat_dir,
                            seg_dir=wild_params.segmentation_dir,
                            feature_level=wild_params.langsplat_level)
                        range_factor = 1.0
                        gt_feature_map = gt_feature_map.to(torch.float32)
                        language_feature_mask = language_feature_mask.to(torch.float32)
                    else:
                        gt_feature_map, range_factor = get_gt_features(
                            viewpoint, 'avarage', avg_p=1.0,
                            max_feat_scale=wild_params.max_feat_scale, min_feat_scale=wild_params.min_feat_scale)

                    feature_map = render_pkg['feature_map']

                    """
                    if is_feat_enc:
                        height, width = feature_map.shape[1:]
                        ratio = height / width
                        new_width = randint(80, 120)
                        new_height = int(ratio * new_width)
                        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(new_height, new_width),
                                                    mode='nearest').squeeze(0)
                        map_shape = feature_map.shape
                        feature_map = feature_map.permute(2, 1, 0).reshape(-1, map_shape[0])
                        feature_map = scene.gaussians.wild_model.decode_features(feature_map)
                        feature_map = feature_map.reshape((*map_shape[-1:0:-1], feature_map.shape[-1])).permute(2, 1, 0)
                    """

                    if speedup:
                        feature_map = cnn_decoder(feature_map)

                    if wild_params.dino_dir:
                        dino_gt = torch.load(str(Path(wild_params.dino_dir) / f'{viewpoint.image_name}_fmap_CxHxW.pt')).cuda()
                        dino_map = feature_map[-wild_params.dino_size:]
                        feature_map = feature_map[:-wild_params.dino_size]
                        dino_vis = feature_visualize_saving(dino_map).permute(2, 0, 1)

                    bigger_feature_map = feature_map.clone()

                    if isinstance(gt_feature_map, dict):
                        keys = sorted(gt_feature_map.keys())
                        if 'avarage' not in keys:
                            keys = sorted([key for key in gt_feature_map.keys()])
                            size = {key: (gt_feature_map[key].shape[1], gt_feature_map[key].shape[2]) for key in keys}
                            range_factor = keys[-1] / keys[0]
                        else:
                            range_factor = gt_feature_map['range_factor']
                            gt_feature_map = gt_feature_map['avarage']
                            size = (gt_feature_map.shape[1], gt_feature_map.shape[2])
                    else:
                        size = (gt_feature_map.shape[1], gt_feature_map.shape[2])
                        # range_factor = None

                    if wild_params.do_attention:
                        feature_map, attention_map = scene.gaussians.wild_model.attention_downsample(
                            feature_map, size=size, get_attention=True, range_factor=range_factor)
                    elif isinstance(gt_feature_map, dict):
                        keys = gt_feature_map.keys()
                        feature_map = {key: F.interpolate(
                            feature_map.unsqueeze(0), size=size[key],
                            mode='bilinear', align_corners=True).squeeze(0) for key in keys}
                    else:
                        feature_map = F.interpolate(
                            feature_map.unsqueeze(0), size=size,
                            mode='bilinear', align_corners=False).squeeze(0) 
                    
                    if wild_params.langsplat_gt:
                        Ll1_feature += l1_loss(feature_map*language_feature_mask,
                                               gt_feature_map*language_feature_mask)
                    else:
                        Ll1_feature += l1_loss(feature_map, gt_feature_map)

                    if clip_vis:
                        scores = {}
                        scores_raw = {}
                        for label in clip_vis.quary_texts:
                            scores[label], scores_raw[label] = clip_vis.compute_scores(bigger_feature_map, label)
                            scores[label] = torch.clamp(scores[label], 0.0, 1.0)
                            scores_raw[label] = torch.clamp(scores_raw[label], 0.0, 1.0)
                    
                    feature_vis = feature_visualize_saving(bigger_feature_map).permute(2, 0, 1)
                    # feature_loss_vis = torch.abs((feature_map - gt_feature_map) / gt_feature_map.mean()).squeeze(0).mean(dim=0)
                    # feature_loss_vis = torch.clamp(feature_loss_vis, 0.0, 1.0).unsqueeze(0)
                    if isinstance(gt_feature_map, dict):
                        avg_gt_feature = None
                        avg_feature = None
                        for key, gt_feat in gt_feature_map.items():
                            feat = feature_map[key]
                            if avg_gt_feature is None:
                                avg_gt_feature = gt_feat
                                avg_feature = feat
                            elif gt_feat.shape[1] > avg_gt_feature.shape[1]:
                                avg_gt_feature = gt_feat + F.interpolate(
                                    avg_gt_feature.unsqueeze(0), size=(gt_feat.shape[1], gt_feat.shape[2]),
                                    mode='bilinear', align_corners=False).squeeze(0) 
                                avg_feature = feat + F.interpolate(
                                    avg_feature.unsqueeze(0), size=(feat.shape[1], feat.shape[2]),
                                    mode='bilinear', align_corners=False).squeeze(0) 
                            else:
                                avg_gt_feature = avg_gt_feature + F.interpolate(
                                    gt_feat.unsqueeze(0), size=(avg_gt_feature.shape[1], avg_gt_feature.shape[2]),
                                    mode='bilinear', align_corners=False).squeeze(0)
                                avg_feature = avg_feature + F.interpolate(
                                    feat.unsqueeze(0), size=(avg_feature.shape[1], avg_feature.shape[2]),
                                    mode='bilinear', align_corners=False).squeeze(0)
                        avg_gt_feature = avg_gt_feature / len(gt_feature_map)
                        avg_feature = avg_feature / len(gt_feature_map)
                    
                    elif wild_params.langsplat_gt:
                        avg_gt_feature = gt_feature_map * language_feature_mask
                        avg_feature = feature_map * language_feature_mask
                    else:
                        avg_gt_feature = gt_feature_map
                        avg_feature = feature_map

                    feature_loss_vis_cossine = torch.clamp(
                        (1 - torch.sum(F.normalize(avg_feature, dim=0) * F.normalize(avg_gt_feature, dim=0), dim=0)), 0.0, 1.0).unsqueeze(0)
                    feature_loss_vis_canbera = torch.abs((avg_feature - avg_gt_feature) / (torch.abs(avg_gt_feature) + torch.abs(avg_feature) + 1e-9)).squeeze(0).mean(dim=0)
                    feature_loss_vis_canbera = torch.clamp(feature_loss_vis_canbera, 0.0, 1.0).unsqueeze(0)

                    if scene.gaussians.wild_model.is_vis_hash:
                        latent_vector = scene.gaussians.wild_model.get_latent_vec(viewpoint.image_name).detach()
                        latent_vector = latent_vector.reshape(1, 4, 6) / torch.max(latent_vector)

                    if tb_writer: # and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image.detach().cpu()[None], global_step=log_iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/features".format(viewpoint.image_name), feature_vis.detach().cpu()[None], global_step=log_iteration)
                        if wild_params.dino_dir:
                            tb_writer.add_images(config['name'] + "_view_{}/features_dino".format(viewpoint.image_name),
                                                 dino_vis.detach().cpu()[None], global_step=log_iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/loss_features".format(viewpoint.image_name), feature_loss_vis[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/loss_features_cossine".format(viewpoint.image_name), feature_loss_vis_cossine.detach().cpu()[None], global_step=log_iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/loss_features_canbera".format(viewpoint.image_name), feature_loss_vis_canbera.detach().cpu()[None], global_step=log_iteration)
                        if scene.gaussians.wild_model.is_vis_hash:
                            tb_writer.add_images(config['name'] + "_view_{}/latent_vector".format(viewpoint.image_name),
                                                 latent_vector.detach().cpu()[None], global_step=log_iteration)
                        if clip_vis:
                            for label in clip_vis.quary_texts:
                                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/score_{label}",
                                                     scores[label].detach().cpu()[None], global_step=log_iteration)
                                tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/score_raw_{label}",
                                                     scores_raw[label].detach().cpu()[None], global_step=log_iteration)
                        if wild_params.do_attention:
                            tb_writer.add_images(config['name'] + "_view_{}/attention".format(viewpoint.image_name),
                                                 (attention_map / torch.max(attention_map)).detach().cpu()[None], global_step=log_iteration)


                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/render_gt".format(viewpoint.image_name), gt_image.detach().cpu()[None], global_step=iteration)
                            if wild_params.langsplat_gt:
                                gt_feature_vis = feature_visualize_saving(avg_gt_feature, language_feature_mask).permute(2, 0, 1)
                            else:
                                gt_feature_vis = feature_visualize_saving(avg_gt_feature).permute(2, 0, 1)
                            tb_writer.add_images(config['name'] + "_view_{}/features_2D".format(viewpoint.image_name), gt_feature_vis.detach().cpu()[None], global_step=iteration)
                            if wild_params.dino_dir:
                                dino_gt_vis = feature_visualize_saving(dino_gt).permute(2, 0, 1)
                                tb_writer.add_images(config['name'] + "_view_{}/features_dino_2D".format(viewpoint.image_name),
                                                     dino_gt_vis.detach().cpu()[None], global_step=iteration)
                            if clip_vis:
                                scores = {}
                                scores_raw = {}
                                for label in clip_vis.quary_texts:
                                    scores[label], scores_raw[label] = clip_vis.compute_scores(avg_gt_feature, label)
                                    scores[label] = torch.clamp(scores[label], 0.0, 1.0)
                                    scores_raw[label] = torch.clamp(scores_raw[label], 0.0, 1.0)
                                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/score_{label}_2D",
                                                        scores[label].detach().cpu()[None], global_step=iteration)
                                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/score_raw_{label}_2D",
                                                        scores_raw[label].detach().cpu()[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                Ll1_feature /= len(config['cameras'])
                print(f"\n[ITER {log_iteration}] Evaluating {config['name']}: L1 {l1_test.item()} PSNR {psnr_test.item()}, L1_feature {Ll1_feature.item()}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test.item(), log_iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test.item(), log_iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - L1_feature', Ll1_feature.item(), iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity.detach().cpu(), log_iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], log_iteration)
        # torch.cuda.empty_cache()

    scene.gaussians.wild_model.train()
