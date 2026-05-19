from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from StemGames2026_ProjectTask.pointcloud.schemas import CameraPose, SceneView
from StemGames2026_ProjectTask.pipeline.schemas import DepthResult


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        view: SceneView,
        pose: CameraPose,
        reference_points_cam: np.ndarray | None = None,
    ) -> DepthResult:
        """
        Estimate per-pixel depth for a single image.

        Args:
            view:                 the SceneView to process
            pose:                 camera pose for this view (may be used for scale hints)
            reference_points_cam: (K, 3) float32 metric 3D points in this camera's
                                  coordinate frame, used to align the estimator's
                                  affine-invariant depth to metric scale.
                                  None means no scale alignment is performed.

        Returns:
            DepthResult with metric depth_map (H, W), points_cam (H, W, 3),
            validity_mask (H, W), and scale metadata.
        """
        ...
