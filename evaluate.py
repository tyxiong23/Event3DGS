"""PSNR, SSIM, and optional AlexNet LPIPS with explicit radiance alignment.

The log-affine fit is estimated jointly over the supplied evaluation images,
using their ground truth, as in the legacy Event3DGS evaluation. These aligned
scores are a calibrated evaluation protocol, not uncalibrated prediction scores.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def paired_images(render_dir, gt_dir):
    extensions = {'.png', '.jpg', '.jpeg'}
    renders = {p.name: p for p in Path(render_dir).iterdir() if p.suffix.lower() in extensions}
    targets = {p.name: p for p in Path(gt_dir).iterdir() if p.suffix.lower() in extensions}
    if not renders or renders.keys() != targets.keys():
        raise ValueError('Render/GT image filenames must match exactly and be nonempty.')
    result = []
    for name in sorted(renders):
        with Image.open(renders[name]) as file:
            prediction = np.array(file.convert('RGB'))
        with Image.open(targets[name]) as file:
            target = np.array(file.convert('RGB'))
        if prediction.shape != target.shape:
            raise ValueError(f'Image dimensions differ for {name}.')
        result.append((name, prediction, target))
    return result


def fit_log_alignment(predictions, targets, margin=20.0, epsilon=1e-5):
    """Fit log(GT + eps) = a * log(pred + eps) + b per color channel.

    Values are in [0,255], preserving the legacy intensity-margin convention.
    Constant or empty channels use a unit slope and a fitted mean offset.
    """
    a, b = [], []
    for channel in range(3):
        x = np.concatenate([image[..., channel].reshape(-1) for image in predictions]).astype(np.float64)
        y = np.concatenate([image[..., channel].reshape(-1) for image in targets]).astype(np.float64)
        selected = x > margin
        if not selected.any():
            selected = np.ones(x.shape, dtype=bool)
        lx, ly = np.log(x[selected] + epsilon), np.log(y[selected] + epsilon)
        variance = np.mean((lx - lx.mean()) ** 2)
        slope = np.mean((lx - lx.mean()) * (ly - ly.mean())) / variance if variance > 1e-12 else 1.0
        offset = ly.mean() - slope * lx.mean()
        a.append(float(slope))
        b.append(float(offset))
    return np.asarray(a), np.asarray(b)


def apply_log_alignment(image, a, b, epsilon=1e-5):
    log_values = np.log(image.astype(np.float64) + epsilon) * a + b
    return np.clip(np.exp(np.clip(log_values, -80, 80)) - epsilon, 0, 255).astype(np.uint8)


def evaluate(args):
    pairs = paired_images(args.render_dir, args.gt_dir)
    coefficients = None
    if args.alignment == 'log':
        a, b = fit_log_alignment([p for _, p, _ in pairs], [gt for _, _, gt in pairs], args.margin)
        coefficients = {'slope_rgb': a.tolist(), 'offset_rgb': b.tolist(),
                        'fitted_on': 'all supplied evaluation renders and their ground truth',
                        'intensity_margin_0_255': args.margin}
    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').eval().to(args.device)
    per_view = {}
    if args.save_aligned:
        Path(args.save_aligned).mkdir(parents=True, exist_ok=True)
    for name, prediction, target in tqdm(pairs, desc='Evaluating'):
        image = apply_log_alignment(prediction, a, b) if coefficients else prediction
        if args.save_aligned:
            Image.fromarray(image).save(Path(args.save_aligned) / name)
        image_f, target_f = image.astype(np.float64) / 255., target.astype(np.float64) / 255.
        metrics = {'PSNR': float(peak_signal_noise_ratio(target_f, image_f, data_range=1.0)),
                   'SSIM': float(structural_similarity(target_f, image_f, data_range=1.0, channel_axis=2))}
        if lpips_model is not None:
            import torch
            def to_tensor(array):
                return torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(args.device) * 2 - 1
            with torch.no_grad():
                metrics['LPIPS_alex'] = float(lpips_model(to_tensor(image_f), to_tensor(target_f)))
        per_view[name] = metrics
    mean = {key: float(np.mean([metrics[key] for metrics in per_view.values()])) for key in next(iter(per_view.values()))}
    result = {'alignment': args.alignment, 'alignment_coefficients': coefficients,
              'image_count': len(pairs), 'mean': mean, 'per_view': per_view}
    output = Path(args.output) if args.output else Path(args.render_dir).parent / f'metrics_{args.alignment}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as file:
        json.dump(result, file, indent=2)
    print(json.dumps({'image_count': len(pairs), 'alignment': args.alignment, 'mean': mean}, indent=2))
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--render_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--alignment', choices=['none', 'log'], default='log')
    parser.add_argument('--margin', type=float, default=20.)
    parser.add_argument('--skip_lpips', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_aligned')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.margin < 0 or args.margin >= 255:
        parser.error('--margin must be in [0,255).')
    evaluate(args)


if __name__ == '__main__':
    main()
