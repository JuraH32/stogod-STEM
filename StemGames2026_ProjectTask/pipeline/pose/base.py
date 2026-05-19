from __future__ import annotations

from abc import ABC, abstractmethod

from StemGames2026_ProjectTask.pointcloud.schemas import SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose


class PoseProvider(ABC):
    @abstractmethod
    def can_provide(self, dataset: SceneDataset) -> bool:
        """Return True if this provider can supply poses for the given dataset."""
        ...

    @abstractmethod
    def provide(self, dataset: SceneDataset) -> list[EstimatedPose]:
        """
        Return one EstimatedPose per view, ordered by view.index.
        Must always return the same number of elements as len(dataset.views).
        """
        ...
