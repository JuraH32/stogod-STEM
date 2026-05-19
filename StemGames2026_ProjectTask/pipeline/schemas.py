from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneView


@dataclass
class EstimatedPose:
    view_index: int
    pose: CameraPose
    confidence: float  # 1.0 for ground truth, ~0.8 for MapAnything
    source: str        # "ground_truth" | "map_anything" | "sfm"


@dataclass
class DepthResult:
    view: SceneView
    depth_map: np.ndarray      # (H, W) float32, metric metres
    points_cam: np.ndarray     # (H, W, 3) float32, camera space
    validity_mask: np.ndarray  # (H, W) bool
    scale_factor: float
    scale_source: str          # "triangulation" | "map_anything" | "none"


@dataclass
class PerViewResult:
    view: SceneView
    pose: CameraPose
    depth_result: DepthResult
    points_world: np.ndarray   # (N, 3) float32 — valid pixels only
    colors_rgb: np.ndarray     # (N, 3) uint8
    pixel_coords: np.ndarray   # (N, 2) int32 — (row, col) per point
    depth_npy_path: Path | None = None
    depth_png_path: Path | None = None
    ply_path: Path | None = None
    pixel_map_path: Path | None = None


@dataclass
class SceneResult:
    scene_name: str
    per_view: list[PerViewResult]
    scene_points: np.ndarray       # (M, 3) float32
    scene_colors: np.ndarray       # (M, 3) uint8
    source_view_index: np.ndarray  # (M,) int32 — index into per_view list
    source_pixel: np.ndarray       # (M, 2) int32 — (row, col) in that view's image
    scene_ply_path: Path | None = None
