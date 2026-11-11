# Derived from the Event3DGS research code and the 3DGS scene readers.
# 3DGS portions: Copyright (C) 2023, Inria / GRAPHDECO. See LICENSE.md.
# EventNeRF data conventions: https://github.com/r00tman/EventNeRF
"""EventNeRF synthetic and real scene loading."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
import logging

import numpy as np
from PIL import Image

from scene.dataset_readers import fetchPly, getNerfppNorm
from utils.event_data import (accumulate_window, load_npz_events,
                              progressive_point_cloud, random_point_cloud, window_pairs)
from utils.graphics_utils import BasicPointCloud, focal2fov

logger = logging.getLogger(__name__)


@dataclass
class EventCameraInfo:
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    is_train: bool = False
    event_image: Optional[np.ndarray] = None
    prev: Optional['EventCameraInfo'] = None
    start_t: float = 0.0
    end_t: float = 0.0
    start_idx: int = 0
    end_idx: int = 0


@dataclass
class EventSceneInfo:
    point_cloud: Optional[BasicPointCloud]
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str = ""


def image_files(directory):
    return sorted(p for p in Path(directory).iterdir()
                  if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def camera_info(uid, image_path, c2w, fx, fy, size=None):
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1]):
        raise ValueError('Camera poses must be finite homogeneous 4x4 matrices.')
    if not np.isfinite([fx, fy]).all() or fx <= 0 or fy <= 0:
        raise ValueError('Camera focal lengths must be finite and positive.')
    if image_path is None:
        if size is None:
            raise ValueError("Missing RGB images require --image_width and --image_height.")
        image = Image.new('RGB', size)
        image_path = ''
        name = f'{uid:05d}'
    else:
        with Image.open(image_path) as handle:
            image = handle.convert('RGB')
        name = Path(image_path).stem
    width, height = image.size
    w2c = np.linalg.inv(c2w)
    return EventCameraInfo(uid, w2c[:3, :3].T, w2c[:3, 3],
                           focal2fov(fy, height), focal2fov(fx, width), image,
                           str(image_path), name, width, height)


def select_views(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[i] for i in indices]


def build_windows(cameras, times, events, args):
    pairs = window_pairs(len(cameras), args.max_winsize,
                         not args.no_rand_winlen, args.cycle)
    pairs = select_views(pairs, args.max_train_views)
    result = []
    for start, end in pairs:
        previous, current = cameras[start], cameras[end]
        event_image, begin, finish = accumulate_window(
            events, times[start], times[end], current.width, current.height,
            args.polarity_offset, args.cycle, (times[0], times[-1]))
        result.append(replace(current, is_train=True, prev=previous,
                              event_image=event_image, start_t=float(times[start]),
                              end_t=float(times[end]), start_idx=begin, end_idx=finish))
    logger.info("Prepared %d event windows", len(result))
    return result


def initialization(args, default_ply=None, needed=True):
    if not needed:
        return None
    if args.ply_path:
        return progressive_point_cloud(args.ply_path, args.ply_thres, args.progressive_keep_color)
    if default_ply is not None and Path(default_ply).is_file() and not args.no_ply:
        return fetchPly(str(default_ply))
    return random_point_cloud(args.num_points, args.pcd_scale)


def read_eventnerf(args, load_initialization=True, load_events=True, load_train=True):
    root = Path(args.source_path)
    all_cameras = {}
    train_times = None
    original_train_count = None
    train_indices = None
    warned_distortion = False
    for split in ('train', 'validation'):
        folder = root / split
        if not folder.is_dir():
            if split == 'validation':
                all_cameras[split] = []
                continue
            raise FileNotFoundError(folder)
        poses = sorted((folder / 'pose').glob('*.txt'))
        intrinsics = sorted((folder / 'intrinsics').glob('*.txt'))
        images = image_files(folder / 'rgb') if (folder / 'rgb').is_dir() else []
        if not poses or len(poses) != len(intrinsics):
            raise ValueError(f"{split}: pose and intrinsics counts must match and be nonzero.")
        if images and len(images) != len(poses):
            raise ValueError(f"{split}: RGB and pose counts differ.")
        if not images and (args.only_rgb or split == 'validation'):
            raise ValueError(f"{split}: RGB images are required for RGB supervision/evaluation.")
        if images and [p.stem for p in poses] != [p.stem for p in images]:
            raise ValueError(f"{split}: RGB and pose file stems must match in sorted order.")
        if [p.stem for p in poses] != [p.stem for p in intrinsics]:
            raise ValueError(f"{split}: pose and intrinsics file stems must match.")
        indices = list(range(0, len(poses), args.skip))
        if split == 'train':
            if args.cycle and indices[-1] != len(poses) - 1:
                indices.append(len(poses) - 1)
            train_indices = indices.copy()
            original_train_count = len(poses)
            times_path = folder / 'timestamps.txt'
            train_times = np.loadtxt(times_path).reshape(-1) if times_path.is_file() else None
            if train_times is not None and len(train_times) != len(poses):
                raise ValueError('train/timestamps.txt must contain one timestamp per original pose.')
        if split == 'train' and not load_train:
            # Keep two views for a valid scene extent without loading the trajectory.
            indices = sorted(set([0, len(poses) // 4, len(poses) // 2, len(poses) - 1]))
        if split == 'validation':
            indices = select_views(indices, args.max_test_views)
        cameras = []
        for i in indices:
            k = np.loadtxt(intrinsics[i])
            c2w = np.loadtxt(poses[i])
            if k.shape not in ((4, 4), (5, 4)) or c2w.shape != (4, 4):
                raise ValueError('Expected 4x4 poses and 4x4 (or 5x4) intrinsics.')
            if k.shape == (5, 4) and np.any(k[4, :2] != 0) and not warned_distortion:
                logger.warning('Distortion coefficients are present; inputs must be rectified before training.')
                warned_distortion = True
            size = (args.image_width, args.image_height) if args.image_width > 0 and args.image_height > 0 else None
            cameras.append(camera_info(i, images[i] if images else None, c2w, k[0, 0], k[1, 1], size))
        all_cameras[split] = cameras
    trajectory = all_cameras['train']
    if load_events and load_train:
        files = sorted((root / 'train/events').glob('*.npz'))
        if len(files) != 1:
            raise ValueError('Expected exactly one NPZ file in train/events/.')
        events = load_npz_events(files[0], trajectory[0].width, trajectory[0].height, args.sort_events)
        # EventNeRF provides uniformly spaced views covering [0, t_max]. An
        # explicit timestamps.txt takes precedence for nonuniform trajectories.
        if train_times is None:
            train_times = np.linspace(0.0, events[2][-1], original_train_count)
        train_times = np.asarray(train_times)[train_indices]
        if len(train_times) != len(trajectory) or not np.isfinite(train_times).all() or np.any(np.diff(train_times) <= 0):
            raise ValueError('Pose timestamps must be strictly increasing.')
        if train_times[-1] < events[2][0] or train_times[0] > events[2][-1]:
            raise ValueError('Pose timestamps and event timestamps do not overlap.')
        train = build_windows(trajectory, train_times, events, args)
    else:
        train = select_views(trajectory, args.max_train_views) if load_train else []
    test = all_cameras['validation']
    extent = getNerfppNorm(trajectory)
    pcd = initialization(args, root / 'points3d.ply', needed=load_initialization)
    return EventSceneInfo(pcd, train, test, extent)


def read_event_scene(args, load_initialization=True, load_events=True, load_train=True):
    if args.skip < 1:
        raise ValueError('skip must be >= 1.')
    root = Path(args.source_path)
    if not (root / 'train/events').is_dir():
        raise ValueError(
            f'Expected an EventNeRF synthetic or real scene at {root}, '
            'with train/events/, train/pose/, and train/intrinsics/. See docs/DATA.md.'
        )
    return read_eventnerf(args, load_initialization, load_events, load_train)
