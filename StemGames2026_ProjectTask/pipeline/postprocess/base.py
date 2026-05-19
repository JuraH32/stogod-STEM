from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PostProcessor(ABC):
    @abstractmethod
    def process(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        source_view: np.ndarray,
        source_pixel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter or transform a point cloud in-place-style.

        All four arrays are parallel: index i refers to the same point across
        all four. Implementations must return the same four arrays with a
        consistent subset of rows kept.

        Args:
            points:      (N, 3) float32 — XYZ world-space positions
            colors:      (N, 3) uint8  — RGB colours
            source_view: (N,) int32   — index into SceneResult.per_view
            source_pixel:(N, 2) int32 — (row, col) in the source view image

        Returns:
            Filtered (points, colors, source_view, source_pixel) with M ≤ N rows.
        """
        ...
