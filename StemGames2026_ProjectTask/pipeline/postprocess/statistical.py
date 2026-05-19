from __future__ import annotations

import numpy as np

from StemGames2026_ProjectTask.pipeline.postprocess.base import PostProcessor


class StatisticalPostProcessor(PostProcessor):
    """
    Removes outlier points whose mean k-NN distance exceeds a threshold.

    For each point the mean distance to its `nb_neighbors` nearest neighbours
    is computed. Points with mean distance above `mean + std_ratio * std`
    (computed over all points) are removed. Uses scipy.spatial.KDTree.
    """

    def __init__(self, nb_neighbors: int = 20, std_ratio: float = 2.0) -> None:
        self._nb_neighbors = nb_neighbors
        self._std_ratio = std_ratio

    def process(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        source_view: np.ndarray,
        source_pixel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(points) <= self._nb_neighbors + 1:
            return points, colors, source_view, source_pixel

        from scipy.spatial import KDTree

        tree = KDTree(points)
        # Query k+1 neighbours because the point itself is always the closest
        dists, _ = tree.query(points, k=self._nb_neighbors + 1)
        mean_dists = dists[:, 1:].mean(axis=1)  # exclude self (distance = 0)

        threshold = mean_dists.mean() + self._std_ratio * mean_dists.std()
        keep = mean_dists < threshold

        return points[keep], colors[keep], source_view[keep], source_pixel[keep]
