"""Event validation, fixed/random pose windows, and progressive initialization."""
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB


def validate_events(x, y, t, p, width, height, sort=False):
    arrays = [np.asarray(a) for a in (x, y, t, p)]
    x, y, t, p = arrays
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("Event x/y/t/p must be one-dimensional arrays of equal length.")
    if not len(t):
        raise ValueError("The event stream is empty.")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Events contain non-finite values.")
    if np.any(np.diff(t.astype(np.float64)) < 0):
        if not sort:
            raise ValueError("Event timestamps are not sorted. Inspect the source and pass --sort_events to explicitly stable-sort x/y/t/p together.")
        order = np.argsort(t, kind='stable')
        x, y, t, p = (a[order] for a in arrays)
    if np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
        raise ValueError("Event coordinates must be integers.")
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise ValueError("Event coordinates lie outside the image dimensions.")
    values = set(np.unique(p).tolist())
    if values <= {0, 1}:
        p = p.astype(np.float32) * 2 - 1
    elif values <= {-1, 1}:
        p = p.astype(np.float32)
    else:
        raise ValueError("Polarities must use either {-1,+1} or {0,1}.")
    return x.astype(np.int64), y.astype(np.int64), t.astype(np.float64), p


def load_npz_events(path, width, height, sort=False):
    with np.load(path, allow_pickle=False) as data:
        if not set("xytp") <= set(data.files):
            raise ValueError("Event NPZ must contain the arrays x, y, t, and p.")
        return validate_events(*(data[k] for k in "xytp"), width, height, sort=sort)


def window_pairs(count, max_winsize, randomize=True, cycle=False):
    if count < 2:
        raise ValueError("At least two training poses are required.")
    if max_winsize < 1 or max_winsize >= count:
        raise ValueError("max_winsize must be in [1, number of selected poses - 1].")
    if cycle and max_winsize >= count - 1:
        raise ValueError('Cyclic windows must be shorter than one full period (excluding the duplicate endpoint).')
    # A cyclic sequence includes a duplicate endpoint pose. Do not train it twice.
    period = count - 1 if cycle else count
    pairs = []
    for end in range(0 if cycle else 1, period):
        limit = max_winsize if cycle else min(max_winsize, end)
        length = np.random.randint(1, limit + 1) if randomize else limit
        start = (end - length) % period if cycle else end - length
        pairs.append((start, end))
    return pairs


def accumulate_window(events, start_t, end_t, width, height, polarity_offset=0.0, cycle=False, cycle_bounds=None):
    """Accumulate half-open [start_t, end_t); wrap only for a closed trajectory."""
    x, y, t, p = events
    start, end = np.searchsorted(t, [start_t, end_t], side="left")
    if start_t > end_t:
        if not cycle:
            raise ValueError("A reversed window requires an explicitly cyclic sequence.")
        lower, upper = (0, len(t)) if cycle_bounds is None else np.searchsorted(t, cycle_bounds, side='left')
        indices = np.r_[np.arange(start, upper), np.arange(lower, end)]
        xx, yy, pp = x[indices], y[indices], p[indices]
    else:
        xx, yy, pp = x[start:end], y[start:end], p[start:end]
    image = np.bincount(yy * width + xx, weights=pp + polarity_offset,
                        minlength=width * height).reshape(height, width).astype(np.float32)
    return image, int(start), int(end)


def random_point_cloud(num_points, scale=0.2, bounds=None):
    if num_points < 2 or scale <= 0:
        raise ValueError("num_points must be >= 2 and pcd_scale must be positive.")
    low, high = (-np.full(3, scale / 2), np.full(3, scale / 2)) if bounds is None else bounds
    xyz = np.random.uniform(low, high, (num_points, 3)).astype(np.float32)
    colors = SH2RGB(np.random.random((num_points, 3)) / 255.0).astype(np.float32)
    return BasicPointCloud(xyz, colors, np.zeros_like(xyz))


def progressive_point_cloud(path, threshold=0.9, keep_color=False):
    """Select sigmoid(opacity_logit) >= threshold; restart from positions.

    Scales, rotations, opacity, SH coefficients, and the optimizer are subsequently
    initialized by GaussianModel.create_from_pcd. keep_color is a legacy option.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("ply_thres must lie in [0,1].")
    vertices = PlyData.read(Path(path))["vertex"]
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    logits = np.asarray(vertices["opacity"], dtype=np.float64)
    opacity = np.exp(-np.logaddexp(0.0, -logits))  # numerically stable sigmoid
    selected = np.isfinite(xyz).all(axis=1) & np.isfinite(logits) & (opacity >= threshold)
    if selected.sum() < 2:
        raise ValueError(
            f"Opacity threshold {threshold} retained {selected.sum()} of {len(xyz)} points. "
            "Train the preceding round longer or explicitly choose a lower threshold."
        )
    xyz = xyz[selected].astype(np.float32)
    if keep_color:
        colors = SH2RGB(np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)])[selected])
    else:
        colors = SH2RGB(np.random.random((len(xyz), 3)) / 255.0)
    print(f"Progressive initialization: {len(vertices)} -> {len(xyz)} points "
          f"with opacity >= {threshold}; keep_color={keep_color}")
    return BasicPointCloud(xyz, colors.astype(np.float32), np.zeros_like(xyz))
