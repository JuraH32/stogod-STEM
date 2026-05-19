from __future__ import annotations

from abc import ABC, abstractmethod

from StemGames2026_ProjectTask.pipeline.schemas import PerViewResult, SceneResult


class Fuser(ABC):
    @abstractmethod
    def fuse(self, per_view: list[PerViewResult]) -> SceneResult:
        """
        Merge all per-view results into a unified scene point cloud.

        Implementations are responsible for:
        - Concatenating per-view world-space clouds
        - Deduplicating overlapping regions
        - Tracking which view and pixel each output point came from
        """
        ...
