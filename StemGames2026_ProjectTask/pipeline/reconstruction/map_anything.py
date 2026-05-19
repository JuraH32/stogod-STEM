from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose
from StemGames2026_ProjectTask.pipeline.reconstruction.base import SceneReconstructor

if TYPE_CHECKING:
    from StemGames2026_ProjectTask.pipeline.pose.map_anything import MapAnythingPoseProvider


class MapAnythingReconstructor(SceneReconstructor):
    """
    Multi-view scene reconstructor backed by MapAnything.

    When constructed with a MapAnythingPoseProvider instance the reconstructor
    reuses the provider's cached forward-pass result, avoiding a second GPU
    inference over the same images. This is the recommended usage pattern for
    unposed datasets (Fountain, Statue).

    Without a shared pose provider the reconstructor runs MapAnything
    independently, optionally guided by the supplied pose hints.
    """

    def __init__(
        self,
        pose_provider: MapAnythingPoseProvider | None = None,
        model_name: str = "mapanything-default",
    ) -> None:
        self._pose_provider = pose_provider
        self._model_name = model_name
        self._model = None  # lazy-loaded

    def reconstruct(
        self,
        dataset: SceneDataset,
        hints: list[EstimatedPose] | None = None,
    ) -> tuple[np.ndarray, list[EstimatedPose]]:
        # Reuse cached result from the shared pose provider if available
        if (
            self._pose_provider is not None
            and self._pose_provider._cached_result is not None
        ):
            return self._extract_from_cache(dataset, hints)

        return self._run_map_anything(dataset, hints)

    def _extract_from_cache(
        self,
        dataset: SceneDataset,
        hints: list[EstimatedPose] | None,
    ) -> tuple[np.ndarray, list[EstimatedPose]]:
        result = self._pose_provider._cached_result
        scene_points = result["scene_points"]   # (M, 6) XYZRGB float32
        poses = hints if hints is not None else self._pose_provider.provide(dataset)
        return scene_points, poses

    def _run_map_anything(
        self,
        dataset: SceneDataset,
        hints: list[EstimatedPose] | None,
    ) -> tuple[np.ndarray, list[EstimatedPose]]:
        """
        Run MapAnything on the full image set.

        The model is called with:
          - one RGB image tensor per view
          - the corresponding 3×3 intrinsics matrix per view
          - optional pose hints (camera-to-world matrices) when known

        The cache dict stores:
          "scene_points": (M, 6) float32 XYZRGB fused cloud
          "c2w_matrices": list of (4,4) float32 per view
          "confidence":   list of float per view
        """
        try:
            import mapanything  # noqa: F401 — model-specific import
        except ImportError as e:
            raise ImportError(
                "MapAnything requires the 'mapanything' package. "
                "Install it according to the model's repository instructions."
            ) from e

        if self._model is None:
            self._model = mapanything.load_model(self._model_name)

        images, intrinsics_list = _load_images_and_intrinsics(dataset)

        pose_hints_c2w = None
        if hints is not None:
            from StemGames2026_ProjectTask.pipeline import coords
            pose_hints_c2w = [
                coords.unity_pose_to_opencv_c2w(ep.pose).tolist() for ep in hints
            ]

        raw_result = self._model.reconstruct(
            images=images,
            intrinsics=intrinsics_list,
            pose_hints=pose_hints_c2w,
        )

        scene_points = _extract_scene_points(raw_result)
        c2w_matrices = _extract_c2w_matrices(raw_result)
        confidences  = _extract_confidences(raw_result, len(dataset.views))

        cache = {
            "scene_points": scene_points,
            "c2w_matrices": c2w_matrices,
            "confidence":   confidences,
        }
        if self._pose_provider is not None:
            self._pose_provider._cached_result = cache

        estimated_poses = _build_estimated_poses(dataset, c2w_matrices, confidences)
        return scene_points, estimated_poses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_images_and_intrinsics(
    dataset: SceneDataset,
) -> tuple[list, list[list[list[float]]]]:
    """Load RGB images and K matrices for all views."""
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


def _extract_scene_points(raw_result) -> np.ndarray:
    """Extract (M, 6) XYZRGB float32 from MapAnything result."""
    # The exact attribute name depends on the MapAnything API version.
    # Try common conventions.
    for attr in ("points", "point_cloud", "vertices", "xyz_rgb"):
        if hasattr(raw_result, attr):
            arr = getattr(raw_result, attr)
            return np.asarray(arr, dtype=np.float32)
    # Dict-style result
    for key in ("points", "point_cloud", "vertices", "xyz_rgb"):
        if key in raw_result:
            return np.asarray(raw_result[key], dtype=np.float32)
    raise AttributeError(
        "Cannot locate scene point cloud in MapAnything result. "
        "Check the model API and update _extract_scene_points()."
    )


def _extract_c2w_matrices(raw_result) -> list[np.ndarray]:
    for attr in ("camera_poses", "c2w", "extrinsics", "poses"):
        if hasattr(raw_result, attr):
            return [np.asarray(m, dtype=np.float32) for m in getattr(raw_result, attr)]
    for key in ("camera_poses", "c2w", "extrinsics", "poses"):
        if key in raw_result:
            return [np.asarray(m, dtype=np.float32) for m in raw_result[key]]
    raise AttributeError(
        "Cannot locate camera poses in MapAnything result. "
        "Check the model API and update _extract_c2w_matrices()."
    )


def _extract_confidences(raw_result, n_views: int) -> list[float]:
    for attr in ("confidence", "confidences", "pose_confidence"):
        if hasattr(raw_result, attr):
            return list(getattr(raw_result, attr))
    for key in ("confidence", "confidences", "pose_confidence"):
        if key in raw_result:
            return list(raw_result[key])
    return [0.8] * n_views


def _build_estimated_poses(
    dataset: SceneDataset,
    c2w_matrices: list[np.ndarray],
    confidences: list[float],
) -> list[EstimatedPose]:
    """
    Decompose OpenCV-convention camera-to-world matrices into CameraPose objects.

    Column layout of a c2w matrix: [right | up | forward | position]
    We invert the Y flip applied in unity_pose_to_opencv_c2w() to store the
    pose back in Unity convention, keeping the schema consistent with what
    GroundTruthPoseProvider returns.
    """
    poses = []
    for view, M, conf in zip(dataset.views, c2w_matrices, confidences):
        # M is OpenCV c2w. Invert the diag(1,-1,1,1) applied during conversion.
        M_unity = M.copy()
        M_unity[1, :] *= -1
        M_unity[:, 1] *= -1

        right    = (float(M_unity[0, 0]), float(M_unity[1, 0]), float(M_unity[2, 0]))
        up       = (float(M_unity[0, 1]), float(M_unity[1, 1]), float(M_unity[2, 1]))
        forward  = (float(M_unity[0, 2]), float(M_unity[1, 2]), float(M_unity[2, 2]))
        position = (float(M_unity[0, 3]), float(M_unity[1, 3]), float(M_unity[2, 3]))

        camera_pose = CameraPose(
            position=position,
            forward=forward,
            right=right,
            up=up,
        )
        poses.append(EstimatedPose(
            view_index=view.index,
            pose=camera_pose,
            confidence=conf,
            source="map_anything",
        ))
    return poses
