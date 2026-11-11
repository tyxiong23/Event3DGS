"""Refine SH/opacity from blurred RGB observations with fixed Gaussian geometry.

Each observation is a real RGB target plus the calibrated camera poses covering
its exposure. Rendering averages those poses. No event data or RGB ground truth
at the latent sharp poses are required by this optimizer.
"""
from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import PILtoTorch, safe_state
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim
from utils.system_utils import searchForMaxIteration


def load_observations(path):
    path = Path(path)
    with open(path) as file:
        manifest = json.load(file)
    result = []
    for index, frame in enumerate(manifest['frames']):
        with Image.open(path.parent / frame['image_path']) as image:
            image = image.convert('RGB')
        width, height = image.size
        target = PILtoTorch(image, (width, height))[:3].cuda()
        fx, fy = frame.get('fx', manifest.get('fx')), frame.get('fy', manifest.get('fy'))
        if fx is None or fy is None or fx <= 0 or fy <= 0:
            raise ValueError('Positive fx and fy are required for every observation.')
        views = []
        for pose in frame['poses']:
            c2w = np.asarray(pose, dtype=np.float64)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1]):
                raise ValueError('Exposure poses must be finite 4x4 OpenCV camera-to-world matrices.')
            rotation = c2w[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
                raise ValueError('Exposure pose rotations must belong to SO(3).')
            w2c = np.linalg.inv(c2w)
            views.append(Camera(index, w2c[:3, :3].T, w2c[:3, 3],
                                focal2fov(fx, width), focal2fov(fy, height),
                                target.cpu(), None, f'blur_{index:05d}', index, data_device='cpu'))
        if not views:
            raise ValueError('Every blur observation requires at least one exposure pose.')
        result.append((views, target))
    if not result:
        raise ValueError('The blur manifest has no observations.')
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    optimization_parameters = OptimizationParams(parser)
    pipeline_parameters = PipelineParams(parser)
    parser.set_defaults(iterations=100)
    parser.add_argument('--event_model', required=True)
    parser.add_argument('--blur_manifest', required=True)
    parser.add_argument('--model_path', '-m', required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--opacity_lr_scale', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1 or args.opacity_lr_scale < 0:
        parser.error('iterations must be positive; opacity_lr_scale must be nonnegative.')
    output = Path(args.model_path)
    if output.resolve() == Path(args.event_model).resolve() or (output / 'config.json').exists():
        parser.error('Use a new output directory to preserve the event model.')
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    safe_state(False, args.seed)
    with open(Path(args.event_model) / 'config.json') as file:
        config = json.load(file)
    iteration = searchForMaxIteration(str(Path(args.event_model) / 'point_cloud')) if args.iteration == -1 else args.iteration
    gaussians = GaussianModel(config['sh_degree'], logging.getLogger(__name__))
    gaussians.load_ply(str(Path(args.event_model) / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'))
    opt = optimization_parameters.extract(args)
    opt.opacity_lr *= args.opacity_lr_scale
    gaussians.training_setup(opt)
    frozen = {}
    for name in ('_xyz', '_scaling', '_rotation'):
        parameter = getattr(gaussians, name)
        parameter.requires_grad_(False)
        frozen[name] = parameter.detach().clone()
    observations = load_observations(args.blur_manifest)
    pipeline = pipeline_parameters.extract(args)
    background = torch.full((3,), config['bg_color'] / 255., device='cuda')
    started = time.perf_counter()
    losses = []
    for step in tqdm(range(1, args.iterations + 1), desc='Appearance refinement'):
        views, target = random.choice(observations)
        gaussians.optimizer.zero_grad(set_to_none=True)
        image = sum(render(view, gaussians, pipeline, background)['render'] for view in views) / len(views)
        loss = ((1 - opt.lambda_dssim) * l1_loss(image, target) +
                opt.lambda_dssim * (1 - ssim(image, target)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite blur loss at iteration {step}.')
        loss.backward()
        gaussians.optimizer.step()
        losses.append(float(loss.detach()))
    unchanged = all(torch.equal(value, getattr(gaussians, name).detach()) for name, value in frozen.items())
    if not unchanged:
        raise RuntimeError('Geometry changed during appearance-only refinement.')
    gaussians.save_ply(str(output / 'point_cloud' / f'iteration_{args.iterations}' / 'point_cloud.ply'))
    config.update(model_path=str(output), blur_manifest=str(Path(args.blur_manifest).resolve()),
                  appearance_refinement=True, refinement_iterations=args.iterations,
                  opacity_lr_scale=args.opacity_lr_scale, refinement_settings=vars(args))
    with open(output / 'config.json', 'w') as file:
        json.dump(config, file, indent=2)
    summary = {'iterations': args.iterations, 'observations': len(observations),
               'first_loss': losses[0], 'final_loss': losses[-1],
               'structure_unchanged': unchanged, 'opacity_learning_rate': opt.opacity_lr,
               'training_seconds': time.perf_counter() - started, 'losses': losses}
    with open(output / 'refinement_summary.json', 'w') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != 'losses'}, indent=2))


if __name__ == '__main__':
    main()
