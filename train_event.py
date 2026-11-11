# Event3DGS training, derived from the original 3DGS training loop.
# 3DGS portions: Copyright (C) 2023, Inria / GRAPHDECO. See LICENSE.md.
"""Train from event windows, or use --only_rgb for an image baseline."""
import argparse
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.events import EventScene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

logger = logging.getLogger(__name__)


def event_loss(image, previous, camera, args, lambda_dssim):
    target = camera.event_image.to(image.device) * args.event_delta
    difference = (torch.log(image.clamp_min(0) + args.tonemap_eps) -
                  torch.log(previous.clamp_min(0) + args.tonemap_eps)) / args.event_gamma
    mask = (torch.rand(target.shape[-2:], device=image.device) < args.rand_ratio).float()[None]
    weights = torch.where(target != 0, 1.0, args.no_event_alpha)
    mask = mask * weights * camera.color_mask.to(image.device)
    # Apply pixel sampling and zero-net-event loss weights.
    prediction, target = difference * mask, target * mask
    l1 = l1_loss(prediction, target)
    total = (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(prediction, target))
    return total, l1


@torch.no_grad()
def evaluate_views(scene, pipeline, background):
    values = []
    for camera in scene.getTestCameras():
        image = render(camera, scene.gaussians, pipeline, background)['render'].clamp(0, 1)
        gt = camera.original_image.to(image.device).clamp(0, 1)
        values.append((float(l1_loss(image, gt)), float(psnr(image, gt).mean())))
    if not values:
        return {}
    result = np.mean(values, axis=0)
    return {'test_l1': float(result[0]), 'test_psnr_raw': float(result[1]), 'test_views': len(values)}


def save_checkpoint(path, gaussians, iteration, remaining):
    torch.save({'model': gaussians.capture(), 'iteration': iteration,
                'remaining_views': remaining, 'rng_python': random.getstate(),
                'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all()}, path)


def train(args, dataset, optimization, pipeline):
    output = Path(args.model_path)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'config.json', 'w') as file:
        json.dump(vars(args), file, indent=2)
    gaussians = GaussianModel(dataset.sh_degree, logger)
    scene = EventScene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(optimization)
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError('No training windows were prepared.')
    first_iteration, remaining = 0, []
    if args.start_checkpoint:
        saved = torch.load(args.start_checkpoint, weights_only=False)
        if isinstance(saved, dict):
            gaussians.restore(saved['model'], optimization)
            first_iteration = saved['iteration']
            remaining = saved['remaining_views']
            random.setstate(saved['rng_python'])
            np.random.set_state(saved['rng_numpy'])
            torch.set_rng_state(saved['rng_torch'].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng_cuda']])
        else:
            model, first_iteration = saved
            gaussians.restore(model, optimization)
        if first_iteration >= args.iterations:
            raise ValueError('Resume target --iterations must exceed the checkpoint iteration.')
        logger.info('Resuming at iteration %d', first_iteration)
    background = torch.full((3,), args.bg_color / 255., dtype=torch.float32, device='cuda')
    writer = None
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(str(output))
    started = time.perf_counter()
    last_loss = None
    saving = set(args.save_iterations + [args.iterations])
    testing = set(args.test_iterations)
    records_path = output / 'train_metrics.jsonl'
    with open(records_path, 'a' if first_iteration else 'w') as records:
        bar = tqdm(range(first_iteration + 1, args.iterations + 1), desc='Training')
        for iteration in bar:
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if not remaining:
                remaining = list(range(len(cameras)))
                random.shuffle(remaining)
            camera = cameras[remaining.pop()]
            bg = torch.rand(3, device='cuda') if optimization.random_background else background
            gaussians.optimizer.zero_grad(set_to_none=True)
            package = render(camera, gaussians, pipeline, bg)
            image = package['render']
            if args.only_rgb:
                target = camera.original_image.to(image.device)
                l1 = l1_loss(image, target)
                loss = ((1 - optimization.lambda_dssim) * l1 +
                        optimization.lambda_dssim * (1 - ssim(image, target)))
            else:
                previous = render(camera.prev, gaussians, pipeline, bg)['render']
                loss, l1 = event_loss(image, previous, camera, args, optimization.lambda_dssim)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at iteration {iteration}.')
            loss.backward()
            with torch.no_grad():
                # Keep the original 3DGS density-control schedule. Accumulate
                # current-view screen-space gradients as in the source loop.
                if iteration < optimization.densify_until_iter:
                    visible = package['visibility_filter']
                    gaussians.max_radii2D[visible] = torch.maximum(
                        gaussians.max_radii2D[visible], package['radii'][visible])
                    gaussians.add_densification_stats(package['viewspace_points'], visible)
                    if iteration > optimization.densify_from_iter and iteration % optimization.densification_interval == 0:
                        threshold = 20 if iteration > optimization.opacity_reset_interval else None
                        gaussians.densify_and_prune(optimization.densify_grad_threshold,
                                                    args.min_opacity, scene.cameras_extent, threshold)
                        if len(gaussians.get_xyz) < 2:
                            raise RuntimeError('Density control removed all usable Gaussians; check scene scale/min_opacity.')
                    if iteration % optimization.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                gaussians.optimizer.step()
                last_loss = float(loss)
                if iteration % args.log_interval == 0 or iteration in saving or iteration in testing or iteration == 1:
                    record = {'iteration': iteration, 'loss': last_loss, 'l1': float(l1),
                              'points': len(gaussians.get_xyz),
                              'elapsed_seconds': time.perf_counter() - started}
                    if iteration in testing:
                        record.update(evaluate_views(scene, pipeline, background))
                        logger.info('Evaluation at %d: %s', iteration, record)
                    records.write(json.dumps(record) + '\n')
                    records.flush()
                    bar.set_postfix(loss=f'{last_loss:.6f}', points=record['points'])
                    if writer:
                        for key in ('loss', 'l1', 'points', 'test_psnr_raw'):
                            if key in record:
                                writer.add_scalar(key, record[key], iteration)
                if iteration in saving:
                    scene.save(iteration)
                    logger.info('Saved iteration %d', iteration)
                if iteration in args.checkpoint_iterations:
                    save_checkpoint(output / f'chkpnt{iteration}.pth', gaussians, iteration, remaining)
    torch.cuda.synchronize()
    summary = {'iterations': args.iterations, 'resumed_from_iteration': first_iteration,
               'final_loss': last_loss, 'final_points': len(gaussians.get_xyz),
               'training_seconds': time.perf_counter() - started,
               'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}
    with open(output / 'training_summary.json', 'w') as file:
        json.dump(summary, file, indent=2)
    if writer:
        writer.close()
    logger.info('Training complete: %s', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_iterations', nargs='*', type=int, default=[7000, 30000])
    parser.add_argument('--save_iterations', nargs='*', type=int, default=[7000, 30000])
    parser.add_argument('--checkpoint_iterations', nargs='*', type=int, default=[])
    parser.add_argument('--start_checkpoint')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if not args.source_path or not args.model_path:
        parser.error('--source_path/-s and --model_path/-m are required.')
    if args.iterations < 1 or args.log_interval < 1:
        parser.error('iterations and log_interval must be positive.')
    if not 0 < args.rand_ratio <= 1 or args.no_event_alpha < 0:
        parser.error('rand_ratio must be in (0,1] and no_event_alpha must be nonnegative.')
    if args.event_delta <= 0 or args.tonemap_eps <= 0 or args.event_gamma <= 0:
        parser.error('event_delta, tonemap_eps, and event_gamma must be positive.')
    if not 0 <= args.bg_color <= 255 or not 0 <= args.lambda_dssim <= 1:
        parser.error('bg_color must be in [0,255] and lambda_dssim in [0,1].')
    if args.sh_degree < 0 or args.sh_degree > 3:
        parser.error('The vendored rasterizer supports SH degrees 0 through 3.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the Gaussian rasterizer.')
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(Path(args.model_path) / 'record.log')])
    safe_state(args.quiet, args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(args, model.extract(args), optimization.extract(args), pipeline.extract(args))


if __name__ == '__main__':
    main()
