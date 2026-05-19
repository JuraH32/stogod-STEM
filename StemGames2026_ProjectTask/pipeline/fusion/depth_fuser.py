from __future__ import annotations

import numpy as np

from StemGames2026_ProjectTask.pipeline.schemas import PerViewResult, SceneResult
from StemGames2026_ProjectTask.pipeline.fusion.base import Fuser


class DepthFuser(Fuser):
    """
    Fuses per-view point clouds into a common scene cloud using numpy voxel
    downsampling. No external library (e.g. open3d) required.

    Each voxel keeps the colour of the first point that falls into it while
    still recording its source view and pixel for the pixel-assignment map.
    """

    def __init__(
        self,
        voxel_size: float = 0.01,
        min_depth: float = 0.05,
        max_depth: float = 50.0,
    ) -> None:
        self._voxel_size = voxel_size
        self._min_depth = min_depth
        self._max_depth = max_depth

    def fuse(self, per_view: list[PerViewResult]) -> SceneResult:
        if not per_view:
            raise ValueError("Cannot fuse an empty list of PerViewResult objects.")

        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_view_idx: list[np.ndarray] = []
        all_pixels: list[np.ndarray] = []

        for view_list_idx, pvr in enumerate(per_view):
            pts = pvr.points_world
            cols = pvr.colors_rgb
            pix = pvr.pixel_coords

            # Depth filter in camera space: use the depth_map at the recorded pixels
            depth_vals = pvr.depth_result.depth_map[pix[:, 0], pix[:, 1]]
            keep = (depth_vals >= self._min_depth) & (depth_vals <= self._max_depth)

            pts, cols, pix = pts[keep], cols[keep], pix[keep]
            view_idx = np.full(len(pts), view_list_idx, dtype=np.int32)

            all_points.append(pts)
            all_colors.append(cols)
            all_view_idx.append(view_idx)
            all_pixels.append(pix)

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        view_indices = np.concatenate(all_view_idx, axis=0)
        pixels = np.concatenate(all_pixels, axis=0)

        points, colors, view_indices, pixels = _voxel_downsample(
            points, colors, view_indices, pixels, self._voxel_size
        )

        scene_name = per_view[0].view.scene_name
        return SceneResult(
            scene_name=scene_name,
            per_view=per_view,
            scene_points=points,
            scene_colors=colors,
            source_view_index=view_indices,
            source_pixel=pixels,
        )


def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    view_indices: np.ndarray,
    pixels: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep at most one point per voxel cell. Retains the first point encountered
    in each occupied voxel (order reflects the per-view iteration order, which
    is deterministic).
    """
    if len(points) == 0:
        return points, colors, view_indices, pixels

    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    # Encode the 3 integer coordinates into a single key for uniqueness check
    # Shift by a fixed offset to make all coords positive before encoding
    offset = voxel_coords.min(axis=0)
    shifted = voxel_coords - offset
    max_dim = shifted.max(axis=0) + 1
    keys = (
        shifted[:, 0].astype(np.int64) * max_dim[1] * max_dim[2]
        + shifted[:, 1].astype(np.int64) * max_dim[2]
        + shifted[:, 2].astype(np.int64)
    )
    _, first_occurrence = np.unique(keys, return_index=True)
    keep = np.sort(first_occurrence)
    return points[keep], colors[keep], view_indices[keep], pixels[keep]
