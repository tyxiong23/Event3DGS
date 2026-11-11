# 3DGS portions: Copyright (C) 2023, Inria / GRAPHDECO. See LICENSE.md.
"""Render trained event models; use saved configuration unless overridden."""
from argparse import ArgumentParser
from pathlib import Path
import logging

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene.events import EventScene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


@torch.no_grad()
def render_sets(args, dataset, pipeline):
    gaussians = GaussianModel(dataset.sh_degree, logging.getLogger(__name__))
    scene = EventScene(dataset, gaussians, load_iteration=args.iteration,
                        shuffle=False, load_train=not args.skip_train)
    background = torch.full((3,), args.bg_color / 255., device='cuda')
    for name, views, skip in [('train', scene.getTrainCameras(), args.skip_train),
                              ('test', scene.getTestCameras(), args.skip_test)]:
        if skip:
            continue
        folder = Path(dataset.model_path) / name / f'ours_{scene.loaded_iter}'
        (folder / 'renders').mkdir(parents=True, exist_ok=True)
        (folder / 'gt').mkdir(exist_ok=True)
        for i, view in enumerate(tqdm(views, desc=f'Rendering {name}')):
            image = render(view, gaussians, pipeline, background)['render']
            save_image(image, folder / 'renders' / f'{i:05d}.png')
            save_image(view.original_image, folder / 'gt' / f'{i:05d}.png')
        logging.info('Rendered %d %s views to %s', len(views), name, folder)


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser)
    logging.basicConfig(level=logging.INFO)
    safe_state(args.quiet)
    render_sets(args, model.extract(args), pipeline.extract(args))


if __name__ == '__main__':
    main()
