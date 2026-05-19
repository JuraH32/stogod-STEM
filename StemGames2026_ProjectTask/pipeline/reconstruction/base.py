from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose


class SceneReconstructor(ABC):
    @abstractmethod
    def reconstruct(
        self,
        dataset: SceneDataset,
        hints: list[EstimatedPose] | None = None,
    ) -> tuple[np.ndarray, list[EstimatedPose]]:
        """
        Run multi-view reconstruction for the entire scene.

        Args:
            dataset: the scene to reconstruct
            hints:   optional pose estimates to guide the reconstructor
                     (e.g. ground-truth poses for posed datasets, or poses
                     already estimated by a PoseProvider)

        Returns:
            scene_points_xyzrgb: (M, 6) float32 — fused point cloud with colours
            estimated_poses:     one EstimatedPose per view (may be same as hints)
        """
        ...
