#!/usr/bin/env python
"""Build explicitly synthetic blur observations by averaging EventNeRF RGB views.

This is a preparation/example utility, not a loader of experimentally captured
blurred images. Event RGB files must use the same working color space as renders.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source_path', '-s', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--poses_per_image', type=int, default=15)
    args = parser.parse_args()
    source = Path(args.source_path) / 'train'
    images = sorted((source / 'rgb').glob('*.png'))
    poses = sorted((source / 'pose').glob('*.txt'))
    intrinsics = sorted((source / 'intrinsics').glob('*.txt'))
    if not images or len(images) != len(poses) or len(images) != len(intrinsics):
        parser.error('Matching EventNeRF PNG images, poses, and intrinsics are required.')
    if not 1 <= args.poses_per_image <= len(images) or not 1 <= args.num_images <= len(images) - args.poses_per_image + 1:
        parser.error('Requested blur observations do not fit inside the source trajectory.')
    output = Path(args.output)
    if (output / 'manifest.json').exists():
        parser.error('Use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    starts = np.linspace(0, len(images) - args.poses_per_image, args.num_images, dtype=int)
    frames = []
    for index, start in enumerate(starts):
        selected = range(start, start + args.poses_per_image)
        colors = []
        for i in selected:
            with Image.open(images[i]) as image:
                colors.append(np.asarray(image.convert('RGB'), dtype=np.float32))
        blurred = np.mean(colors, axis=0).clip(0, 255).astype(np.uint8)
        filename = f'blur_{index:05d}.png'
        Image.fromarray(blurred).save(output / filename)
        k = np.loadtxt(intrinsics[start])
        if any(not np.allclose(np.loadtxt(intrinsics[i]), k) for i in selected):
            raise ValueError('Intrinsics must remain fixed during a blur exposure.')
        frames.append({'image_path': filename, 'fx': float(k[0, 0]), 'fy': float(k[1, 1]),
                       'poses': [np.loadtxt(poses[i]).tolist() for i in selected]})
    manifest = {'observation_type': 'synthetically averaged EventNeRF RGB views', 'frames': frames}
    with open(output / 'manifest.json', 'w') as file:
        json.dump(manifest, file, indent=2)
    print(f'Wrote {len(frames)} synthetic blur observations and {output / "manifest.json"}')


if __name__ == '__main__':
    main()
