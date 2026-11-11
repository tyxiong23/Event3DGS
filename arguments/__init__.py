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
import ast
import json
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
        self._model_path = ""
        self._resolution = -1
        self._no_ply = False
        self.data_device = "cuda"
        self.eval = False

        # event dataset setting
        self.max_winsize = 50
        self.monochrome = False
        self.no_rand_winlen = False
        self.event_delta = 0.25
        self.bg_color = 159.0
        self.polarity_offset = 0.0
        self.skip = 1
        self.cycle = False
        self.only_rgb = False
        self.tonemap_eps = 1e-5
        self.no_event_alpha = 1.0
        self.rand_ratio = 1.0
        self.pcd_scale = 0.2
        self.min_opacity = 0.005
        self.ply_path = ""
        self.ply_thres = 0.9
        self.progressive_keep_color = False
        self.num_points = 100_000
        self.max_train_views = -1
        self.max_test_views = -1
        self.image_width = 0
        self.image_height = 0
        self.sort_events = False
        # 1.0 preserves the legacy loss; 2.2 enables paper Eq. (4)'s divisor.
        self.event_gamma = 1.0

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    args_cmdline = parser.parse_args(sys.argv[1:])
    if not args_cmdline.model_path:
        parser.error('--model_path/-m is required for rendering.')
    defaults = ModelParams(ArgumentParser())
    merged_dict = {key.lstrip('_'): value for key, value in vars(defaults).items()}
    config_path = os.path.join(args_cmdline.model_path, 'config.json')
    if os.path.isfile(config_path):
        with open(config_path) as file:
            merged_dict.update(json.load(file))
    else:
        # Accept legacy Namespace(...) files without executing Python code.
        legacy = os.path.join(args_cmdline.model_path, 'cfg_args')
        if os.path.isfile(legacy):
            with open(legacy) as file:
                node = ast.parse(file.read().strip(), mode='eval').body
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
                raise ValueError('Invalid legacy cfg_args; expected Namespace with literal keyword values.')
            merged_dict.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg is not None})
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
