from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose
from StemGames2026_ProjectTask.pipeline.pose.base import PoseProvider

if TYPE_CHECKING:
    pass


class ColmapPoseProvider(PoseProvider):
    """
    Estimates camera poses for datasets that lack extrinsic metadata by running
    COLMAP sparse reconstruction (SfM).

    Uses the pycolmap Python bindings (already required by reconstruct.py).
    Reconstruction outputs are cached so ColmapReconstructor can reuse the
    sparse point cloud without a second SfM pass.

    Applicable to: Fountain, Statue (pose_source == "missing").
    """

    def __init__(self, colmap_output_root: Path | None = None) -> None:
        self._colmap_output_root = colmap_output_root
        # Populated after the first call to provide().
        # Keys: "poses" (list[EstimatedPose]), "scene_points" (np.ndarray M×6)
        self._cached: dict | None = None

    def can_provide(self, dataset: SceneDataset) -> bool:
        return dataset.pose_source == "missing"

    def provide(self, dataset: SceneDataset) -> list[EstimatedPose]:
        if not self.can_provide(dataset):
            raise ValueError(
                f"ColmapPoseProvider cannot handle dataset '{dataset.scene_name}': "
                f"pose_source is '{dataset.pose_source}', expected 'missing'."
            )
        self._ensure_reconstructed(dataset)
        return self._cached["poses"]

    def get_scene_points(self) -> np.ndarray | None:
        """Return (M, 6) XYZRGB sparse cloud, or None if not yet reconstructed."""
        return None if self._cached is None else self._cached["scene_points"]

    # ------------------------------------------------------------------

    def _ensure_reconstructed(self, dataset: SceneDataset) -> None:
        if self._cached is not None:
            return

        import pycolmap
        from StemGames2026_ProjectTask.pipeline.reconstruction.colmap import (
            ReconstructionConfig,
            run_scene_reconstruction,
        )

        out_root = (
            self._colmap_output_root
            or dataset.root_dir.parent.parent / "outputs" / "colmap"
        )
        config = ReconstructionConfig(
            output_root=out_root,
            matcher_mode="exhaustive",
            dense_mode="off",
            overwrite=True,
            max_num_features=16384,  # default 8192 — more features → better registration
            min_model_size=1,        # default 2 — allow small models to be considered
        )

        print(f"  Running COLMAP sparse SfM for '{dataset.scene_name}' …")
        sfm_result = run_scene_reconstruction(dataset, config)

        # Read the selected model back from disk
        recon = pycolmap.Reconstruction()
        recon.read(sfm_result.artifacts.selected_model_path)

        # Extract OpenCV c2w (4×4 float64) keyed by image filename
        c2w_by_name = _extract_c2w_by_name(recon)

        # Build EstimatedPose list
        poses = _build_estimated_poses(dataset, c2w_by_name)

        # Extract sparse point cloud as (M, 6) float32 XYZRGB
        scene_points = _extract_sparse_points(recon)

        n_reg = sum(1 for v in dataset.views if v.image_path.name in c2w_by_name)
        print(
            f"  COLMAP: {n_reg}/{len(dataset.views)} images registered, "
            f"{len(scene_points):,} sparse points"
        )
        self._cached = {"poses": poses, "scene_points": scene_points}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_c2w_by_name(recon) -> dict[str, np.ndarray]:
    """Return {filename: c2w (4×4 float64)} for every registered image."""
    c2w_by_name: dict[str, np.ndarray] = {}
    for img in recon.images.values():
        if not img.has_pose:
            continue
        # cam_from_world() returns Rigid3d (world→camera); .matrix() gives (3, 4)
        w2c_34 = np.array(img.cam_from_world().matrix(), dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = w2c_34
        c2w = np.linalg.inv(w2c)
        c2w_by_name[img.name] = c2w
    return c2w_by_name


def _build_estimated_poses(
    dataset: SceneDataset,
    c2w_by_name: dict[str, np.ndarray],
) -> list[EstimatedPose]:
    """
    Convert COLMAP OpenCV c2w matrices to Unity-convention CameraPose objects.

    Applies the inverse of unity_pose_to_opencv_c2w (negate row 1 and col 1)
    so that calling unity_pose_to_opencv_c2w on the returned poses exactly
    recovers the COLMAP c2w matrices.
    """
    poses: list[EstimatedPose] = []
    for view in dataset.views:
        name = view.image_path.name
        if name in c2w_by_name:
            c2w = c2w_by_name[name]
            confidence = 0.8
        else:
            c2w = np.eye(4, dtype=np.float64)
            confidence = 0.0

        # OpenCV → Unity: invert the diag(1,-1,1,1) similarity
        M = c2w.copy()
        M[1, :] *= -1
        M[:, 1] *= -1

        camera_pose = CameraPose(
            position=(float(M[0, 3]), float(M[1, 3]), float(M[2, 3])),
            forward =(float(M[0, 2]), float(M[1, 2]), float(M[2, 2])),
            right   =(float(M[0, 0]), float(M[1, 0]), float(M[2, 0])),
            up      =(float(M[0, 1]), float(M[1, 1]), float(M[2, 1])),
        )
        poses.append(EstimatedPose(
            view_index=view.index,
            pose=camera_pose,
            confidence=confidence,
            source="colmap",
        ))
    return poses


def _extract_sparse_points(recon) -> np.ndarray:
    """Return (M, 6) float32 XYZRGB array from the COLMAP sparse model."""
    rows: list[list[float]] = []
    for pt in recon.points3D.values():
        xyz = pt.xyz
        rgb = [float(c) for c in pt.color[:3]]
        rows.append([float(xyz[0]), float(xyz[1]), float(xyz[2])] + rgb)
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(rows, dtype=np.float32)
