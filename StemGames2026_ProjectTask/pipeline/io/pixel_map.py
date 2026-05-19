from __future__ import annotations

from pathlib import Path

import numpy as np

_PIXEL_MAP_DTYPE = np.dtype([
    ("view_idx", "<i4"),
    ("row", "<i4"),
    ("col", "<i4"),
])


def write_pixel_map(
    path: Path,
    pixel_coords: np.ndarray,
    source_view_index: np.ndarray,
) -> None:
    """
    Save a pixel-to-point mapping as a structured .npy array.

    Args:
        path:              destination file path (.npy)
        pixel_coords:      (N, 2) int32 — (row, col) per point
        source_view_index: (N,) int32 — index into SceneResult.per_view per point
    """
    n = len(pixel_coords)
    arr = np.empty(n, dtype=_PIXEL_MAP_DTYPE)
    arr["view_idx"] = source_view_index
    arr["row"]      = pixel_coords[:, 0]
    arr["col"]      = pixel_coords[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def read_pixel_map(path: Path) -> np.ndarray:
    """
    Load a pixel map saved by write_pixel_map.

    Returns a structured array with fields: view_idx (i4), row (i4), col (i4).
    """
    return np.load(path)
