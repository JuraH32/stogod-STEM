from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_depth_npy(path: Path, depth_map: np.ndarray) -> None:
    """Save a float32 depth map as a .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth_map.astype(np.float32))


def write_depth_png(
    path: Path,
    depth_map: np.ndarray,
    validity_mask: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> None:
    """
    Save a false-colour depth visualisation as PNG.

    Valid pixels are coloured with a JET-like colormap (blue=near, red=far).
    Invalid pixels are rendered black. Depth range is determined by the
    requested percentiles of valid pixel values to avoid outlier stretching.
    """
    valid_depths = depth_map[validity_mask]
    if len(valid_depths) == 0:
        Image.fromarray(np.zeros((*depth_map.shape, 3), dtype=np.uint8)).save(path)
        return

    d_min = float(np.percentile(valid_depths, percentile_low))
    d_max = float(np.percentile(valid_depths, percentile_high))
    if d_max <= d_min:
        d_max = d_min + 1.0

    normalised = np.clip((depth_map - d_min) / (d_max - d_min), 0.0, 1.0)
    rgb = _jet_colormap(normalised)
    rgb[~validity_mask] = 0  # black for invalid pixels

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """
    Apply a JET-like colormap to a (H, W) float array in [0, 1].
    Returns (H, W, 3) uint8.
    """
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
