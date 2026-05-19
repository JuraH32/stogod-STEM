from __future__ import annotations

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose
from StemGames2026_ProjectTask.pipeline.pose.base import PoseProvider


class MapAnythingPoseProvider(PoseProvider):
    """
    Estimates camera poses for datasets that lack extrinsic metadata.

    Runs MapAnything in pose-free multi-view mode to jointly reconstruct
    the scene and recover per-camera extrinsics. The full reconstruction
    result is cached after the first call so that MapAnythingReconstructor
    can reuse it without a second GPU inference pass.

    Applicable to: Fountain, Statue (pose_source == "missing").
    """

    def __init__(self, model_name: str = "mapanything-default") -> None:
        self._model_name = model_name
        self._model = None
        # Cache populated after the first call to provide().
        # Schema: {"scene_points": np.ndarray (M,6), "c2w_matrices": list[np.ndarray],
        #          "confidence": list[float]}
        self._cached_result: dict | None = None

    def can_provide(self, dataset: SceneDataset) -> bool:
        return dataset.pose_source == "missing"

    def provide(self, dataset: SceneDataset) -> list[EstimatedPose]:
        if not self.can_provide(dataset):
            raise ValueError(
                f"MapAnythingPoseProvider cannot handle dataset '{dataset.scene_name}': "
                f"pose_source is '{dataset.pose_source}', expected 'missing'."
            )
        self._run_if_needed(dataset)
        return _build_estimated_poses(
            dataset,
            self._cached_result["c2w_matrices"],
            self._cached_result["confidence"],
        )

    def compute_per_view_scale_hints(
        self,
        view_index: int,
        intrinsics,
        c2w_opencv: np.ndarray,
    ) -> np.ndarray | None:
        """
        Reproject the cached global cloud into the specified camera frame and
        return the visible 3D points in camera space as metric reference
        for MoGe-2 scale alignment.

        This is called by the runner for each view before depth estimation.
        """
        if self._cached_result is None:
            return None

        from StemGames2026_ProjectTask.pipeline import coords

        pts_world = self._cached_result["scene_points"][:, :3].astype(np.float32)
        _, depth_cam, valid = coords.project_points_to_camera(pts_world, c2w_opencv, intrinsics)

        if not np.any(valid):
            return None

        # Return visible world points — the runner will pass them to the depth estimator
        # which re-projects them into camera space internally
        return pts_world[valid]

    def _run_if_needed(self, dataset: SceneDataset) -> None:
        if self._cached_result is not None:
            return

        try:
            import mapanything  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MapAnything requires the 'mapanything' package. "
                "Install it according to the model's repository instructions."
            ) from e

        if self._model is None:
            self._model = mapanything.load_model(self._model_name)

        images, intrinsics_list = _load_images_and_intrinsics(dataset)

        raw_result = self._model.reconstruct(
            images=images,
            intrinsics=intrinsics_list,
            pose_hints=None,
        )

        from StemGames2026_ProjectTask.pipeline.reconstruction.map_anything import (
            _extract_c2w_matrices,
            _extract_confidences,
            _extract_scene_points,
        )

        self._cached_result = {
            "scene_points": _extract_scene_points(raw_result),
            "c2w_matrices": _extract_c2w_matrices(raw_result),
            "confidence":   _extract_confidences(raw_result, len(dataset.views)),
        }


# ---------------------------------------------------------------------------
# Helpers (duplicated from reconstruction module to avoid circular import)
# ---------------------------------------------------------------------------

def _load_images_and_intrinsics(
    dataset: SceneDataset,
) -> tuple[list, list[list[list[float]]]]:
    import torch
    from PIL import Image as PILImage

    images = []
    intrinsics_list = []
    for view in dataset.views:
        rgb = np.array(PILImage.open(view.image_path).convert("RGB"))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        images.append(tensor)
        K = np.array(view.intrinsics.matrix, dtype=np.float32)
        intrinsics_list.append(K.tolist())
    return images, intrinsics_list


def _build_estimated_poses(
    dataset: SceneDataset,
    c2w_matrices: list[np.ndarray],
    confidences: list[float],
) -> list[EstimatedPose]:
    """
    Convert OpenCV-convention c2w matrices back to CameraPose in Unity convention.
    """
    poses = []
    for view, M, conf in zip(dataset.views, c2w_matrices, confidences):
        # Invert the diag(1,-1,1,1) applied in unity_pose_to_opencv_c2w()
        M_unity = M.copy()
        M_unity[1, :] *= -1
        M_unity[:, 1] *= -1

        camera_pose = CameraPose(
            position=(float(M_unity[0, 3]), float(M_unity[1, 3]), float(M_unity[2, 3])),
            forward =(float(M_unity[0, 2]), float(M_unity[1, 2]), float(M_unity[2, 2])),
            right   =(float(M_unity[0, 0]), float(M_unity[1, 0]), float(M_unity[2, 0])),
            up      =(float(M_unity[0, 1]), float(M_unity[1, 1]), float(M_unity[2, 1])),
        )
        poses.append(EstimatedPose(
            view_index=view.index,
            pose=camera_pose,
            confidence=conf,
            source="map_anything",
        ))
    return poses
