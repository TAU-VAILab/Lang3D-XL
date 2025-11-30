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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._foundation_model = "" ###
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.speedup = False ###
        self.render_items = ['RGB', 'Depth', 'Edge', 'Normal', 'Curvature', 'Feature Map']
        self.exclude_gt_images_from = ''  # '/storage/shai/3d/data/HolyScenes'
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = True
        super().__init__(parser, "Pipeline Parameters")

class VideoParams(ParamGroup):
    def __init__(self, parser):
        self.rots = 2
        self.zrate = 0.02
        self.focal = 5.0
        self.N = 300
        self.offset_x = 0.0
        self.offset_y = 0.2
        self.offset_z = -0.5
        self.rad_alpha_x = 0.7
        self.rad_alpha_y = 0.5
        self.rad_alpha_z = 1.0
        super().__init__(parser, "Video Parameters")

class WildParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.start_checkpoint = ''
        self.wild_checkpoint = ''
        self.wild = False
        self.feat_enc = False
        self.feat_enc_hash = False
        self.hard_neg = False
        self.clip_dir = "/storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/0_CLIPModel"
        self.do_attention = False
        self.interpolate_gt = False
        self.dino_dir = ''
        self.dino_size = 384
        self.vis_enc_hash = True
        self.decode_before_render = False
        self.xyz_to_semantics = False
        self.add_sky = False
        self.segmentation_dir = ''
        self.foreground_masks_dir = ''
        self.attention_dropout = 0.0
        self.max_feat_scale = -1.0
        self.min_feat_scale = -1.0
        self.warmup = False
        self.max_win_scale_factor = -1.0
        self.seg_loss_weight = 0.0005
        self.seg_loss_mode = 'l2'
        self.clip_grad_norm = -1.0
        self.bulk_on_device = 1
        self.langsplat_gt = False
        self.langsplat_dir = ''
        self.langsplat_level = 1
        self.do_wild_only_batch = False
        self.wild_only_interval = 10000
        self.wild_only_iterations = 5000
        self.do_wild_densification = False
        self.foreground_mode = 'all'
        self.mlp_feat_decay = 1e-5
        self.dino_factor = 0.8
        self.split_validation = 0.0
        self.no_vis_enc_hash = False
        self.clean_cache = -1
        self.warmup_iterations = 0
        self.freeze_after_warmup = False

        super().__init__(parser, "InTheWild Paremeters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016 * 0.1
        self.position_lr_final = 0.0000016 * 0.1
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025  # * 0.1
        self.opacity_lr = 0.05  # * 0.1
        self.scaling_lr = 0.005  # * 0.1
        self.rotation_lr = 0.001  # * 0.1
#################################################
        self.semantic_feature_lr = 0.001 # 0.001 
#################################################
        self.percent_dense = 0.2  # 0.01
        self.lambda_dssim = 0.5
        self.densification_interval = 2000 # 2000  # 100
        self.opacity_reset_interval = 100000  # 3000 ### TRY reset to 100000 but worse
        self.densify_from_iter = 2000 # 2000 # 500
        self.densify_until_iter = 15_000 #6000 ### comapre with 2-stage
        self.densify_grad_threshold = 0.0002
        self.wild_lr = 0.0005  # 0.0001
        self.wild_latent_lr = 0.001  # 0.0001
        self.wild_feat_lr = 0.0005
        self.wild_attention_lr = 0.0005
        self.features_scales_mode = 'avarage'
        self.feat_decay = 0.01
        self.feat_assend = 0.1
        self.feat_assend_time = 0.5
        self.do_dotp_sim_loss = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
