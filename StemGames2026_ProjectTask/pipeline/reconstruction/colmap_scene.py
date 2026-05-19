from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose
from StemGames2026_ProjectTask.pipeline.reconstruction.base import SceneReconstructor

if TYPE_CHECKING:
    from StemGames2026_ProjectTask.pipeline.pose.colmap_pose import ColmapPoseProvider


class ColmapReconstructor(SceneReconstructor):
    """
    Returns the sparse COLMAP point cloud as the scene reconstruction.

    Shares the ColmapPoseProvider's cached SfM result — no second reconstruction
    pass is needed.  The sparse cloud is used by the runner to compute per-view
    metric scale hints for MoGe-2 depth estimation.
    """

    def __init__(self, pose_provider: ColmapPoseProvider) -> None:
        self._pose_provider = pose_provider

    def reconstruct(
        self,
        dataset: SceneDataset,
        hints: list[EstimatedPose] | None = None,
    ) -> tuple[np.ndarray, list[EstimatedPose]]:
        scene_points = self._pose_provider.get_scene_points()
        if scene_points is None:
            scene_points = np.zeros((0, 6), dtype=np.float32)
        poses = hints if hints is not None else self._pose_provider.provide(dataset)
        return scene_points, poses
